/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpucanvascontext

use core::cell::Cell;

use ak::Utf16String;
use libjs_rust::api::{ForeignCellSlot, Gc, Realm, Trace, Vm};
use webgpu_wire::{CanvasId, Command, ObjectId, TextureFormat};

use crate::bindings::conversions::CppInterface;
use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, Exception, WebIdlResult};
use crate::web_gl::canvas::Canvas;
use crate::web_gpu::client;
use crate::web_gpu::gpu::GPUTextureFormat;
use crate::web_gpu::gpu_device::GPUDevice;
use crate::web_gpu::gpu_texture::GPUTexture;

unsafe extern "C" {
    // Defined in Libraries/LibWeb/WebGPU/GPUCanvasContext.cpp. Returns 0 without a compositor.
    fn web_webgpu_create_compositor_canvas(canvas: *mut core::ffi::c_void) -> u64;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPUCanvasAlphaMode {
    Opaque,
    Premultiplied,
}

pub(crate) struct GPUCanvasConfiguration {
    pub(crate) device: Ref<GPUDevice>,
    pub(crate) format: GPUTextureFormat,
    pub(crate) usage: u32,
    // FIXME: The canvas is presented as opaque whatever the alpha mode is.
    pub(crate) alpha_mode: GPUCanvasAlphaMode,
}

/// https://gpuweb.github.io/gpuweb/#dom-gpucanvascontext-configuration-slot
#[derive(Clone, Copy, Trace)]
#[gc(crate = libjs_rust)]
struct Configuration {
    /// The canvas's textures are made on the device, which has to live as long as the configuration names it.
    device: Ref<GPUDevice>,
    #[gc(untraced)]
    format: GPUTextureFormat,
    #[gc(untraced)]
    usage: u32,
    #[gc(untraced)]
    alpha_mode: GPUCanvasAlphaMode,
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUCanvasContext {
    realm: Gc<Realm>,
    canvas_cell: ForeignCellSlot,
    #[gc(untraced)]
    canvas: Canvas,
    /// The compositor's name for the canvas, under which its frames are presented, once it has one (0 before). It is
    /// asked for when the canvas is first configured, since a page can make the context before its compositor is
    /// connected, but cannot have a device to configure it with until then.
    #[gc(untraced)]
    canvas_id: Cell<CanvasId>,
    configuration: Cell<Option<Configuration>>,
    /// https://gpuweb.github.io/gpuweb/#dom-gpucanvascontext-currenttexture-slot
    current_texture: Cell<Option<Ref<GPUTexture>>>,
}

impl Wrappable for GPUCanvasContext {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPUCanvasContext {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, canvas: Canvas, canvas_id: CanvasId) -> Ref<Self> {
        let canvas_cell = ForeignCellSlot::empty();
        // SAFETY: The canvas is a live cell of the heap that the runtime allocates from.
        unsafe { canvas_cell.set(Some(canvas.cell)) };
        allocate(vm, |_| Self {
            realm,
            canvas_cell,
            canvas,
            canvas_id: Cell::new(canvas_id),
            configuration: Cell::new(None),
            current_texture: Cell::new(None),
        })
    }

    /// The compositor's name for the canvas, or 0 if it has none yet.
    pub(crate) fn canvas_id(&self) -> CanvasId {
        self.canvas_id.get()
    }

    /// The compositor's name for the canvas, which is asked for the first time, or None without a compositor.
    fn compositor_canvas_id(&self) -> Option<CanvasId> {
        if self.canvas_id.get() == 0 {
            // SAFETY: The canvas is alive for as long as its context is.
            let canvas_id = unsafe { web_webgpu_create_compositor_canvas(self.canvas.cell.as_ptr()) };
            self.canvas_id.set(canvas_id);
        }
        Some(self.canvas_id.get()).filter(|canvas_id| *canvas_id != 0)
    }

    /// Hands the current texture to the compositor to show, as the steps that update the rendering do:
    /// https://gpuweb.github.io/gpuweb/#abstract-opdef-expire-the-current-texture
    /// Returns whether there was a texture to present.
    pub(crate) fn take_current_texture_for_presentation(&self) -> bool {
        self.current_texture.take().is_some()
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucanvascontext-canvas
    pub(crate) fn canvas(&self, cx: &mut CallContext) -> WebIdlResult<CppInterface> {
        Ok(self.canvas.wrapper(cx.realm()))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucanvascontext-configure
    pub(crate) fn configure(&self, _: &mut CallContext, configuration: GPUCanvasConfiguration) -> WebIdlResult<()> {
        // 3. If the supported context formats do not contain configuration.format, throw a TypeError.
        // https://gpuweb.github.io/gpuweb/#supported-context-formats
        let format = match configuration.format {
            GPUTextureFormat::Bgra8unorm => TextureFormat::Bgra8Unorm,
            GPUTextureFormat::Rgba8unorm => TextureFormat::Rgba8Unorm,
            GPUTextureFormat::Rgba16float => TextureFormat::Rgba16Float,
            _ => return Err(Exception::type_error("The format is not one a canvas supports")),
        };
        // FIXME: Validate the configuration against the device, as the spec does.
        let Some(canvas_id) = self.compositor_canvas_id() else {
            return Err(Exception::Dom(
                "OperationError",
                Utf16String::from_utf8("There is no compositor to present the canvas"),
            ));
        };
        self.current_texture.set(None);
        self.configuration.set(Some(Configuration {
            device: configuration.device,
            format: configuration.format,
            usage: configuration.usage,
            alpha_mode: configuration.alpha_mode,
        }));
        client::send(
            self.realm,
            &Command::ConfigureCanvas {
                canvas: canvas_id,
                device: configuration.device.id,
                format,
                usage: configuration.usage,
            },
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucanvascontext-getconfiguration
    pub(crate) fn get_configuration(&self, _: &mut CallContext) -> WebIdlResult<Option<GPUCanvasConfiguration>> {
        Ok(self.configuration.get().map(|configuration| GPUCanvasConfiguration {
            device: configuration.device,
            format: configuration.format,
            usage: configuration.usage,
            alpha_mode: configuration.alpha_mode,
        }))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucanvascontext-unconfigure
    pub(crate) fn unconfigure(&self, _: &mut CallContext) -> WebIdlResult<()> {
        self.configuration.set(None);
        self.current_texture.set(None);
        if self.canvas_id() != 0 {
            client::send(
                self.realm,
                &Command::UnconfigureCanvas {
                    canvas: self.canvas_id(),
                },
            );
        }
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucanvascontext-getcurrenttexture
    pub(crate) fn get_current_texture(&self, cx: &mut CallContext) -> WebIdlResult<Ref<GPUTexture>> {
        // 1. If this.[[configuration]] is null, throw an InvalidStateError and return.
        let Some(configuration) = self.configuration.get() else {
            return Err(Exception::Dom(
                "InvalidStateError",
                Utf16String::from_utf8("The canvas context is not configured"),
            ));
        };

        // 3. If this.[[currentTexture]] is null, replace the drawing buffer of this, and set
        //    this.[[currentTexture]] to the result.
        if let Some(texture) = self.current_texture.get() {
            return Ok(texture);
        }
        let (width, height) = self.canvas.bitmap_size();
        let (width, height) = (
            u32::try_from(width).unwrap_or(0).max(1),
            u32::try_from(height).unwrap_or(0).max(1),
        );
        let id: ObjectId = client::allocate_object_id();
        client::send(
            self.realm,
            &Command::GetCurrentTexture {
                canvas: self.canvas_id(),
                id,
                width,
                height,
            },
        );
        let texture = GPUTexture::for_canvas(
            cx.vm(),
            self.realm,
            id,
            width,
            height,
            configuration.format,
            configuration.usage,
        );
        self.current_texture.set(Some(texture));

        // NB: What script draws into the texture is presented with the next rendering update, which this asks for.
        self.canvas.did_change_content();
        Ok(texture)
    }
}
