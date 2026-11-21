/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gputexture

use libjs_rust::api::{Gc, Realm, Trace, Vm};
use webgpu_wire::{Command, ObjectId, TextureDescriptor, TextureViewDescriptor};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, Exception, WebIdlResult};
use crate::web_gpu::Label;
use crate::web_gpu::gpu::{GPUObjectDescriptorBase, GPUTextureFormat};
use crate::web_gpu::gpu_texture_view::GPUTextureView;
use crate::web_gpu::{client, labelled};

string_enum!(GPUTextureDimension, "gputexturedimension" {
    _1d => "1d",
    _2d => "2d",
    _3d => "3d",
});

string_enum!(GPUTextureViewDimension, "gputextureviewdimension" {
    _1d => "1d",
    _2d => "2d",
    _2dArray => "2d-array",
    Cube => "cube",
    CubeArray => "cube-array",
    _3d => "3d",
});

string_enum!(GPUTextureAspect, "gputextureaspect" {
    All => "all",
    StencilOnly => "stencil-only",
    DepthOnly => "depth-only",
});

pub(crate) struct GPUExtent3DDict {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth_or_array_layers: u32,
}

/// The union GPUExtent3D of WebGPU/GPUTexture.idl.
pub(crate) enum GPUExtent3D {
    Sequence(Vec<u32>),
    GPUExtent3DDict(GPUExtent3DDict),
}

impl GPUExtent3D {
    // https://gpuweb.github.io/gpuweb/#abstract-opdef-validate-gpuextent3d-shape
    pub(crate) fn to_dimensions(&self) -> WebIdlResult<(u32, u32, u32)> {
        match self {
            Self::GPUExtent3DDict(extent) => Ok((extent.width, extent.height, extent.depth_or_array_layers)),
            Self::Sequence(values) => match values.as_slice() {
                [width] => Ok((*width, 1, 1)),
                [width, height] => Ok((*width, *height, 1)),
                [width, height, depth] => Ok((*width, *height, *depth)),
                _ => Err(Exception::type_error(
                    "A GPUExtent3D sequence must have one to three elements",
                )),
            },
        }
    }
}

pub(crate) struct GPUTextureDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) size: GPUExtent3D,
    pub(crate) mip_level_count: u32,
    pub(crate) sample_count: u32,
    pub(crate) dimension: GPUTextureDimension,
    pub(crate) format: GPUTextureFormat,
    pub(crate) usage: u32,
    pub(crate) view_formats: Vec<GPUTextureFormat>,
}

pub(crate) struct GPUTextureViewDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) format: Option<GPUTextureFormat>,
    pub(crate) dimension: Option<GPUTextureViewDimension>,
    pub(crate) aspect: GPUTextureAspect,
    pub(crate) base_mip_level: u32,
    pub(crate) mip_level_count: Option<u32>,
    pub(crate) base_array_layer: u32,
    pub(crate) array_layer_count: Option<u32>,
}

/// The size and kind of a texture, as its attributes report them.
#[derive(Clone, Copy)]
pub(crate) struct TextureShape {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth_or_array_layers: u32,
    pub(crate) mip_level_count: u32,
    pub(crate) sample_count: u32,
    pub(crate) dimension: GPUTextureDimension,
    pub(crate) format: GPUTextureFormat,
    pub(crate) usage: u32,
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUTexture {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
    #[gc(untraced)]
    shape: TextureShape,
}

impl Wrappable for GPUTexture {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUTexture {
    /// A texture the compositor made for a canvas under the id.
    pub(crate) fn for_canvas(
        vm: &Vm,
        realm: Gc<Realm>,
        id: ObjectId,
        width: u32,
        height: u32,
        format: GPUTextureFormat,
        usage: u32,
    ) -> Ref<Self> {
        let shape = TextureShape {
            width,
            height,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: GPUTextureDimension::_2d,
            format,
            usage,
        };
        allocate(vm, |_| Self {
            realm,
            id,
            shape,
            label: Label::default(),
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createtexture
    pub(crate) fn create(
        vm: &Vm,
        realm: Gc<Realm>,
        device: ObjectId,
        descriptor: &GPUTextureDescriptor,
    ) -> WebIdlResult<Ref<Self>> {
        let (width, height, depth_or_array_layers) = descriptor.size.to_dimensions()?;
        let shape = TextureShape {
            width,
            height,
            depth_or_array_layers,
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: descriptor.dimension,
            format: descriptor.format,
            usage: descriptor.usage,
        };
        let id = client::allocate_object_id();
        client::send(
            realm,
            &Command::CreateTexture {
                device,
                id,
                descriptor: TextureDescriptor {
                    width,
                    height,
                    depth_or_array_layers,
                    mip_level_count: descriptor.mip_level_count,
                    sample_count: descriptor.sample_count,
                    dimension: descriptor.dimension.name().to_owned(),
                    format: descriptor.format.name().to_owned(),
                    usage: descriptor.usage,
                    view_formats: descriptor
                        .view_formats
                        .iter()
                        .map(|format| format.name().to_owned())
                        .collect(),
                },
            },
        );
        Ok(allocate(vm, |_| Self {
            realm,
            id,
            shape,
            label: Label::default(),
        }))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gputexture-createview
    pub(crate) fn create_view(
        &self,
        cx: &mut CallContext,
        descriptor: GPUTextureViewDescriptor,
    ) -> WebIdlResult<Ref<GPUTextureView>> {
        let id = client::allocate_object_id();
        client::send(
            self.realm,
            &Command::CreateTextureView {
                texture: self.id,
                id,
                descriptor: TextureViewDescriptor {
                    format: descriptor.format.map(|format| format.name().to_owned()),
                    dimension: descriptor.dimension.map(|dimension| dimension.name().to_owned()),
                    aspect: descriptor.aspect.name().to_owned(),
                    base_mip_level: descriptor.base_mip_level,
                    mip_level_count: descriptor.mip_level_count,
                    base_array_layer: descriptor.base_array_layer,
                    array_layer_count: descriptor.array_layer_count,
                },
            },
        );
        Ok(labelled(
            GPUTextureView::create(cx.vm(), self.realm, id),
            &descriptor.parent.label,
        ))
    }

    /// The view of the whole texture that a texture stands for where a view is expected.
    pub(crate) fn default_view(&self, vm: &Vm) -> Ref<GPUTextureView> {
        let id = client::allocate_object_id();
        client::send(
            self.realm,
            &Command::CreateTextureView {
                texture: self.id,
                id,
                descriptor: TextureViewDescriptor {
                    format: None,
                    dimension: None,
                    aspect: "all".to_owned(),
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                },
            },
        );
        GPUTextureView::create(vm, self.realm, id)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gputexture-destroy
    pub(crate) fn destroy(&self, _: &mut CallContext) -> WebIdlResult<()> {
        client::send(self.realm, &Command::Destroy { id: self.id });
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gputexture-width
    pub(crate) fn width(&self, _: &mut CallContext) -> WebIdlResult<u32> {
        Ok(self.shape.width)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gputexture-height
    pub(crate) fn height(&self, _: &mut CallContext) -> WebIdlResult<u32> {
        Ok(self.shape.height)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gputexture-depthorarraylayers
    pub(crate) fn depth_or_array_layers(&self, _: &mut CallContext) -> WebIdlResult<u32> {
        Ok(self.shape.depth_or_array_layers)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gputexture-miplevelcount
    pub(crate) fn mip_level_count(&self, _: &mut CallContext) -> WebIdlResult<u32> {
        Ok(self.shape.mip_level_count)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gputexture-samplecount
    pub(crate) fn sample_count(&self, _: &mut CallContext) -> WebIdlResult<u32> {
        Ok(self.shape.sample_count)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gputexture-dimension
    pub(crate) fn dimension(&self, _: &mut CallContext) -> WebIdlResult<GPUTextureDimension> {
        Ok(self.shape.dimension)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gputexture-format
    pub(crate) fn format(&self, _: &mut CallContext) -> WebIdlResult<GPUTextureFormat> {
        Ok(self.shape.format)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gputexture-usage
    pub(crate) fn usage(&self, _: &mut CallContext) -> WebIdlResult<u32> {
        Ok(self.shape.usage)
    }
}

label_attribute!(GPUTexture);
