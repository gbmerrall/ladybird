/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpuqueue

use core::cell::Cell;
use core::ffi::c_void;

use ak::Utf16String;
use libjs_rust::api::{Gc, MarkedVec, Realm, Trace, Value, Vm};
use webgpu_wire::{Blob, Command, ObjectId, Reply, Request};

use crate::bindings::conversions::{BufferSource, CppInterface, copy_of_bytes_of_buffer};
use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::promise::{WebIdlPromise, resolve_promise};
use crate::bindings::{CallContext, Exception, WebIdlResult};
use crate::canvas_2d::canvas::CanvasImageSourceKind;
use crate::html::image_bitmap::ImageBitmap;
use crate::html::image_data::ImageData;
use crate::html::predefined_color_space::PredefinedColorSpace;
use crate::web_gl::canvas::{
    TexImageSourceFrame, TexImageSourceKind, read_tex_image_frame, read_tex_image_pixels, read_tex_image_source,
};
use crate::web_gl::constants::{GL_RGBA, GL_UNSIGNED_BYTE};
use crate::web_gl::web_gl_rendering_context_overloads::TexImageSource;
use crate::web_gpu::gpu_buffer::GPUBuffer;
use crate::web_gpu::gpu_command_buffer::GPUCommandBuffer;
use crate::web_gpu::gpu_command_encoder::{GPUTexelCopyBufferLayout, GPUTexelCopyTextureInfo};
use crate::web_gpu::gpu_device::GPUDevice;
use crate::web_gpu::gpu_texture::GPUExtent3D;
use crate::web_gpu::{Label, client};
use crate::webidl::buffers::AllowSharedBufferSource;

unsafe extern "C" {
    // Defined in Libraries/LibWeb/WebGPU/WebGPUClient.cpp.
    fn web_webgpu_frame_size(frame: *const c_void, out_width: *mut u32, out_height: *mut u32) -> bool;
    fn web_webgpu_read_frame_texels(
        frame: *const c_void,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        flip_y: bool,
        premultiplied: bool,
        out: *mut u8,
    ) -> bool;
}

/// The union GPUCopyExternalImageSource of WebGPU/GPUQueue.idl, whose members are those of WebGL's TexImageSource.
pub(crate) enum GPUCopyExternalImageSource {
    ImageBitmap(Ref<ImageBitmap>),
    ImageData(Ref<ImageData>),
    HTMLImageElement(CppInterface),
    HTMLVideoElement(CppInterface),
    HTMLCanvasElement(CppInterface),
    OffscreenCanvas(CppInterface),
}

impl From<&GPUCopyExternalImageSource> for TexImageSource {
    fn from(source: &GPUCopyExternalImageSource) -> Self {
        match *source {
            GPUCopyExternalImageSource::ImageBitmap(image_bitmap) => Self::ImageBitmap(image_bitmap),
            GPUCopyExternalImageSource::ImageData(image_data) => Self::ImageData(image_data),
            GPUCopyExternalImageSource::HTMLImageElement(object) => Self::HTMLImageElement(object),
            GPUCopyExternalImageSource::HTMLVideoElement(object) => Self::HTMLVideoElement(object),
            GPUCopyExternalImageSource::HTMLCanvasElement(object) => Self::HTMLCanvasElement(object),
            GPUCopyExternalImageSource::OffscreenCanvas(object) => Self::OffscreenCanvas(object),
        }
    }
}

pub(crate) struct GPUCopyExternalImageSourceInfo {
    pub(crate) source: GPUCopyExternalImageSource,
    pub(crate) origin: GPUOrigin2D,
    pub(crate) flip_y: bool,
}

pub(crate) struct GPUCopyExternalImageDestInfo {
    pub(crate) parent: GPUTexelCopyTextureInfo,
    pub(crate) color_space: PredefinedColorSpace,
    pub(crate) premultiplied_alpha: bool,
}

#[derive(Default)]
pub(crate) struct GPUOrigin2DDict {
    pub(crate) x: u32,
    pub(crate) y: u32,
}

/// The union GPUOrigin2D of WebGPU/GPUQueue.idl.
pub(crate) enum GPUOrigin2D {
    Sequence(Vec<u32>),
    GPUOrigin2DDict(GPUOrigin2DDict),
}

impl GPUOrigin2D {
    // https://gpuweb.github.io/gpuweb/#abstract-opdef-validate-gpuorigin2d-shape
    fn to_coordinates(&self) -> WebIdlResult<(u32, u32)> {
        match self {
            Self::GPUOrigin2DDict(origin) => Ok((origin.x, origin.y)),
            Self::Sequence(values) => match values.as_slice() {
                [] => Ok((0, 0)),
                [x] => Ok((*x, 0)),
                [x, y] => Ok((*x, *y)),
                _ => Err(Exception::type_error(
                    "A GPUOrigin2D sequence must have at most two elements",
                )),
            },
        }
    }
}

fn invalid_state_error(message: &str) -> Exception {
    Exception::Dom("InvalidStateError", Utf16String::from_utf8(message))
}

/// https://html.spec.whatwg.org/multipage/canvas.html#the-image-argument-is-not-origin-clean
fn is_not_origin_clean(source: &TexImageSource) -> bool {
    match *source {
        TexImageSource::ImageBitmap(image_bitmap) => !image_bitmap.origin_clean(),
        TexImageSource::ImageData(_) => false,
        TexImageSource::HTMLImageElement(object) => CanvasImageSourceKind::HTMLImageElement.is_not_origin_clean(object),
        TexImageSource::HTMLCanvasElement(object) => {
            CanvasImageSourceKind::HTMLCanvasElement.is_not_origin_clean(object)
        }
        TexImageSource::HTMLVideoElement(object) => CanvasImageSourceKind::HTMLVideoElement.is_not_origin_clean(object),
        TexImageSource::OffscreenCanvas(object) => CanvasImageSourceKind::OffscreenCanvas.is_not_origin_clean(object),
    }
}

/// The current image of a source, read as WebGL reads a texture image source as RGBA and UNSIGNED_BYTE.
fn read_external_image(source: &TexImageSource) -> WebIdlResult<crate::web_gl::host::Frame> {
    let frame = match *source {
        TexImageSource::ImageBitmap(image_bitmap) => match image_bitmap.frame() {
            Some(image_frame) => read_tex_image_frame(&image_frame, GL_RGBA, GL_UNSIGNED_BYTE),
            None => return Err(invalid_state_error("The ImageBitmap is detached")),
        },
        TexImageSource::ImageData(image_data) => match image_data.pixels() {
            Some(pixels) => {
                let (width, height) = image_data.size();
                // SAFETY: Nothing runs script while the pixels are copied.
                unsafe { read_tex_image_pixels(pixels, width, height, GL_RGBA, GL_UNSIGNED_BYTE) }
            }
            None => return Err(invalid_state_error("The ImageData is detached")),
        },
        TexImageSource::HTMLImageElement(object) => {
            read_tex_image_source(object, TexImageSourceKind::HTMLImageElement, GL_RGBA, GL_UNSIGNED_BYTE)
        }
        TexImageSource::HTMLCanvasElement(object) => {
            read_tex_image_source(object, TexImageSourceKind::HTMLCanvasElement, GL_RGBA, GL_UNSIGNED_BYTE)
        }
        TexImageSource::HTMLVideoElement(object) => {
            read_tex_image_source(object, TexImageSourceKind::HTMLVideoElement, GL_RGBA, GL_UNSIGNED_BYTE)
        }
        TexImageSource::OffscreenCanvas(object) => {
            read_tex_image_source(object, TexImageSourceKind::OffscreenCanvas, GL_RGBA, GL_UNSIGNED_BYTE)
        }
    };
    match frame {
        TexImageSourceFrame::Frame(frame) => Ok(frame),
        TexImageSourceFrame::NoFrame | TexImageSourceFrame::InvalidValue => {
            Err(invalid_state_error("The source has no image to copy"))
        }
    }
}

fn operation_error(message: &str) -> Exception {
    Exception::Dom("OperationError", Utf16String::from_utf8(message))
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUQueue {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    /// A device has one queue, which the compositor finds through the device.
    #[gc(untraced)]
    device: ObjectId,
    /// Keeps the device alive for as long as its queue is, since submitting names it.
    device_object: Cell<Option<Ref<GPUDevice>>>,
}

impl Wrappable for GPUQueue {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPUQueue {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, device: ObjectId) -> Ref<Self> {
        allocate(vm, |_| Self {
            label: Label::default(),
            realm,
            device,
            device_object: Cell::new(None),
        })
    }

    pub(crate) fn set_device(&self, device: Ref<GPUDevice>) {
        self.device_object.set(Some(device));
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuqueue-submit
    pub(crate) fn submit(
        &self,
        _: &mut CallContext,
        command_buffers: MarkedVec<'_, Ref<GPUCommandBuffer>>,
    ) -> WebIdlResult<()> {
        let command_buffers = command_buffers.values().iter().map(|buffer| buffer.id).collect();
        client::send(
            self.realm,
            &Command::Submit {
                device: self.device,
                command_buffers,
            },
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuqueue-onsubmittedworkdone
    pub(crate) fn on_submitted_work_done(&self, cx: &mut CallContext, promise: WebIdlPromise) -> WebIdlResult<()> {
        fn steps(vm: &Vm, request: &client::PendingRequest, _: Option<Reply>) {
            // NB: Without a compositor the device is lost, whose work counts as done.
            resolve_promise(vm, request.promise, Value::UNDEFINED);
        }
        client::request(
            cx.vm(),
            self.realm,
            promise,
            self.device,
            Request::OnSubmittedWorkDone { device: self.device },
            steps,
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuqueue-writebuffer
    pub(crate) fn write_buffer(
        &self,
        _: &mut CallContext,
        buffer: Ref<GPUBuffer>,
        buffer_offset: u64,
        data: AllowSharedBufferSource,
        data_offset: u64,
        size: Option<u64>,
    ) -> WebIdlResult<()> {
        // 1. Let contents be the contents of the copy of the bytes held by data. The offset and size are in elements
        //    of a TypedArray, and in bytes otherwise.
        let extent = BufferSource::from(data).extent();
        if extent.is_out_of_bounds {
            return Err(operation_error("The data is out of bounds"));
        }
        let element_size = extent.element_size.max(1) as u64;
        let content_elements = extent.byte_length as u64 / element_size;
        // 4. If dataOffset is greater than contentsSize, throw an OperationError.
        if data_offset > content_elements {
            return Err(operation_error("The data offset is past the end of the data"));
        }
        let size = size.unwrap_or(content_elements - data_offset);
        // 5-6. If dataOffset + size is greater than contentsSize, or dataSize is not a multiple of 4, throw an
        //      OperationError.
        let byte_size = size.checked_mul(element_size);
        if data_offset.checked_add(size).is_none_or(|end| end > content_elements)
            || byte_size.is_none_or(|size| !size.is_multiple_of(4))
        {
            return Err(operation_error(
                "The data size is past the end of the data, or not a multiple of 4",
            ));
        }
        let start = extent.byte_offset + usize::try_from(data_offset * element_size).unwrap_or(0);
        let length = usize::try_from(byte_size.unwrap_or(0)).unwrap_or(0);
        let Some(bytes) = copy_of_bytes_of_buffer(&extent.buffer, start..start + length) else {
            return Err(operation_error("The data is detached"));
        };
        client::send(
            self.realm,
            &Command::WriteBuffer {
                device: self.device,
                buffer: buffer.id,
                offset: buffer_offset,
                data: Blob(bytes),
            },
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuqueue-writetexture
    pub(crate) fn write_texture(
        &self,
        _: &mut CallContext,
        destination: GPUTexelCopyTextureInfo,
        data: AllowSharedBufferSource,
        data_layout: GPUTexelCopyBufferLayout,
        size: GPUExtent3D,
    ) -> WebIdlResult<()> {
        // 1. Let dataBytes be a copy of the bytes held by the buffer source data. The device validates the layout
        //    against them.
        let extent = BufferSource::from(data).extent();
        if extent.is_out_of_bounds {
            return Err(operation_error("The data is out of bounds"));
        }
        let range = extent.byte_offset..extent.byte_offset + extent.byte_length;
        let Some(bytes) = copy_of_bytes_of_buffer(&extent.buffer, range) else {
            return Err(operation_error("The data is detached"));
        };
        client::send(
            self.realm,
            &Command::WriteTexture {
                device: self.device,
                texture: destination.to_wire()?,
                data: Blob(bytes),
                layout: data_layout.to_wire(),
                size: size.to_dimensions()?,
            },
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuqueue-copyexternalimagetotexture
    pub(crate) fn copy_external_image_to_texture(
        &self,
        _: &mut CallContext,
        source: GPUCopyExternalImageSourceInfo,
        destination: GPUCopyExternalImageDestInfo,
        copy_size: GPUExtent3D,
    ) -> WebIdlResult<()> {
        let (width, height, depth) = copy_size.to_dimensions()?;
        let (x, y) = source.origin.to_coordinates()?;
        let texture = destination.parent.to_wire()?;

        let image = TexImageSource::from(&source.source);
        // 3. If source is not origin-clean, throw a SecurityError.
        if is_not_origin_clean(&image) {
            return Err(Exception::Dom(
                "SecurityError",
                Utf16String::from_ascii(b"Cannot copy an image that is not origin-clean"),
            ));
        }
        let frame = read_external_image(&image)?;
        let (mut source_width, mut source_height) = (0, 0);
        // SAFETY: The frame is alive, and the function stores the size when it returns true.
        if !unsafe { web_webgpu_frame_size(frame.as_ptr(), &raw mut source_width, &raw mut source_height) } {
            // FIXME: Copy frames of videos, whose pixels are in the compositor.
            return Err(invalid_state_error("Copying a frame of a video is not supported yet"));
        }

        // 5-6. If the source origin plus the copy size is past the source's width or height, or the depth is more
        //      than 1, throw an OperationError.
        if x.checked_add(width).is_none_or(|right| right > source_width)
            || y.checked_add(height).is_none_or(|bottom| bottom > source_height)
            || depth > 1
        {
            return Err(operation_error("The copy is outside the source image"));
        }
        if width == 0 || height == 0 || depth == 0 {
            return Ok(());
        }

        // FIXME: Convert to the colorSpace of the destination; this copies sRGB as it is.
        let _ = destination.color_space;
        let mut texels = vec![0u8; width as usize * height as usize * 4];
        // SAFETY: The frame is alive, the rectangle is inside it, and the texels have room for it.
        let read = unsafe {
            web_webgpu_read_frame_texels(
                frame.as_ptr(),
                x,
                y,
                width,
                height,
                source.flip_y,
                destination.premultiplied_alpha,
                texels.as_mut_ptr(),
            )
        };
        if !read {
            return Err(invalid_state_error("The source has no image to copy"));
        }
        client::send(
            self.realm,
            &Command::CopyExternalImageToTexture {
                device: self.device,
                texture,
                width,
                height,
                rgba8: Blob(texels),
            },
        );
        Ok(())
    }
}

label_attribute!(GPUQueue);
