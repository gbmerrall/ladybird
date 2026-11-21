/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpucommandencoder

use core::cell::Cell;

use ak::Utf16String;
use libjs_rust::api::{Gc, MarkedVec, Realm, Trace, Vm};
use webgpu_wire::{
    ColorAttachment, Command, DepthStencilAttachment, LoadOp, ObjectId, StoreOp, TexelCopyBufferLayout,
    TexelCopyTexture,
};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, Exception, WebIdlResult};
use crate::web_gpu::Label;
use crate::web_gpu::gpu::GPUObjectDescriptorBase;
use crate::web_gpu::gpu_buffer::GPUBuffer;
use crate::web_gpu::gpu_command_buffer::GPUCommandBuffer;
use crate::web_gpu::gpu_compute_pass_encoder::{GPUComputePassDescriptor, GPUComputePassEncoder};
use crate::web_gpu::gpu_query_set::{GPUQuerySet, timestamp_writes};
use crate::web_gpu::gpu_render_pass_encoder::{GPURenderPassEncoder, PassQueries};
use crate::web_gpu::gpu_texture::{GPUExtent3D, GPUTexture, GPUTextureAspect};
use crate::web_gpu::gpu_texture_view::GPUTextureView;
use crate::web_gpu::{client, labelled};

pub(crate) struct GPUCommandBufferDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
}

pub(crate) struct GPURenderPassDescriptor<'vm> {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) color_attachments: MarkedVec<'vm, Option<GPURenderPassColorAttachment>>,
    pub(crate) depth_stencil_attachment: Option<GPURenderPassDepthStencilAttachment>,
    pub(crate) occlusion_query_set: Option<Ref<GPUQuerySet>>,
    pub(crate) timestamp_writes: Option<GPURenderPassTimestampWrites>,
}

pub(crate) struct GPURenderPassTimestampWrites {
    pub(crate) query_set: Ref<GPUQuerySet>,
    pub(crate) beginning_of_pass_write_index: Option<u32>,
    pub(crate) end_of_pass_write_index: Option<u32>,
}

pub(crate) struct GPURenderPassDepthStencilAttachment {
    pub(crate) view: GPUTextureOrGPUTextureView,
    pub(crate) depth_clear_value: Option<f32>,
    pub(crate) depth_load_op: Option<GPULoadOp>,
    pub(crate) depth_store_op: Option<GPUStoreOp>,
    pub(crate) depth_read_only: bool,
    pub(crate) stencil_clear_value: u32,
    pub(crate) stencil_load_op: Option<GPULoadOp>,
    pub(crate) stencil_store_op: Option<GPUStoreOp>,
    pub(crate) stencil_read_only: bool,
}

/// The union (GPUTexture or GPUTextureView) of WebGPU/GPUCommandEncoder.idl.
#[derive(Clone, Copy, Trace)]
#[gc(crate = libjs_rust)]
#[allow(clippy::enum_variant_names, reason = "the variants have the members' names")]
pub(crate) enum GPUTextureOrGPUTextureView {
    GPUTexture(Ref<GPUTexture>),
    GPUTextureView(Ref<GPUTextureView>),
}

impl GPUTextureOrGPUTextureView {
    /// The view, or the default view of the texture.
    fn view(self, vm: &Vm) -> Ref<GPUTextureView> {
        match self {
            Self::GPUTexture(texture) => texture.default_view(vm),
            Self::GPUTextureView(view) => view,
        }
    }
}

pub(crate) struct GPUTexelCopyBufferLayout {
    pub(crate) offset: u64,
    pub(crate) bytes_per_row: Option<u32>,
    pub(crate) rows_per_image: Option<u32>,
}

impl GPUTexelCopyBufferLayout {
    pub(crate) fn to_wire(&self) -> TexelCopyBufferLayout {
        TexelCopyBufferLayout {
            offset: self.offset,
            bytes_per_row: self.bytes_per_row,
            rows_per_image: self.rows_per_image,
        }
    }
}

pub(crate) struct GPUTexelCopyBufferInfo {
    pub(crate) parent: GPUTexelCopyBufferLayout,
    pub(crate) buffer: Ref<GPUBuffer>,
}

pub(crate) struct GPUTexelCopyTextureInfo {
    pub(crate) texture: Ref<GPUTexture>,
    pub(crate) mip_level: u32,
    pub(crate) origin: GPUOrigin3D,
    pub(crate) aspect: GPUTextureAspect,
}

impl GPUTexelCopyTextureInfo {
    pub(crate) fn to_wire(&self) -> WebIdlResult<TexelCopyTexture> {
        Ok(TexelCopyTexture {
            texture: self.texture.id,
            mip_level: self.mip_level,
            origin: self.origin.to_coordinates()?,
            aspect: self.aspect.name().to_owned(),
        })
    }
}

#[derive(Default)]
pub(crate) struct GPUOrigin3DDict {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) z: u32,
}

/// The union GPUOrigin3D of WebGPU/GPUCommandEncoder.idl.
pub(crate) enum GPUOrigin3D {
    Sequence(Vec<u32>),
    GPUOrigin3DDict(GPUOrigin3DDict),
}

impl GPUOrigin3D {
    // https://gpuweb.github.io/gpuweb/#abstract-opdef-validate-gpuorigin3d-shape
    fn to_coordinates(&self) -> WebIdlResult<(u32, u32, u32)> {
        match self {
            Self::GPUOrigin3DDict(origin) => Ok((origin.x, origin.y, origin.z)),
            Self::Sequence(values) => match values.as_slice() {
                [] => Ok((0, 0, 0)),
                [x] => Ok((*x, 0, 0)),
                [x, y] => Ok((*x, *y, 0)),
                [x, y, z] => Ok((*x, *y, *z)),
                _ => Err(Exception::type_error(
                    "A GPUOrigin3D sequence must have at most three elements",
                )),
            },
        }
    }
}

fn load_op_name(op: GPULoadOp) -> &'static str {
    match op {
        GPULoadOp::Load => "load",
        GPULoadOp::Clear => "clear",
    }
}

fn store_op_name(op: GPUStoreOp) -> &'static str {
    match op {
        GPUStoreOp::Store => "store",
        GPUStoreOp::Discard => "discard",
    }
}

#[derive(Clone, Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPURenderPassColorAttachment {
    pub(crate) view: GPUTextureOrGPUTextureView,
    pub(crate) depth_slice: Option<u32>,
    pub(crate) resolve_target: Option<GPUTextureOrGPUTextureView>,
    #[gc(untraced)]
    pub(crate) clear_value: Option<GPUColor>,
    #[gc(untraced)]
    pub(crate) load_op: GPULoadOp,
    #[gc(untraced)]
    pub(crate) store_op: GPUStoreOp,
}

#[derive(Clone)]
pub(crate) struct GPUColorDict {
    pub(crate) r: f64,
    pub(crate) g: f64,
    pub(crate) b: f64,
    pub(crate) a: f64,
}

/// The union (sequence<double> or GPUColorDict) of WebGPU/GPUCommandEncoder.idl.
#[derive(Clone)]
pub(crate) enum GPUColor {
    Sequence(Vec<f64>),
    GPUColorDict(GPUColorDict),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPULoadOp {
    Load,
    Clear,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPUStoreOp {
    Store,
    Discard,
}

// https://gpuweb.github.io/gpuweb/#abstract-opdef-validate-gpucolor-shape
pub(crate) fn color_components(color: Option<&GPUColor>) -> WebIdlResult<[f64; 4]> {
    match color {
        None => Ok([0.0; 4]),
        Some(GPUColor::GPUColorDict(color)) => Ok([color.r, color.g, color.b, color.a]),
        Some(GPUColor::Sequence(components)) => <[f64; 4]>::try_from(components.as_slice())
            .map_err(|_| Exception::TypeError(Utf16String::from_utf8("A GPUColor sequence must have four components"))),
    }
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUCommandEncoder {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    id: ObjectId,
    #[gc(untraced)]
    finished: Cell<bool>,
}

impl Wrappable for GPUCommandEncoder {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUCommandEncoder {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, id: ObjectId) -> Ref<Self> {
        allocate(vm, |_| Self {
            label: Label::default(),
            realm,
            id,
            finished: Cell::new(false),
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucommandencoder-beginrenderpass
    pub(crate) fn begin_render_pass(
        &self,
        cx: &mut CallContext,
        descriptor: GPURenderPassDescriptor<'_>,
    ) -> WebIdlResult<Ref<GPURenderPassEncoder>> {
        let mut color_attachments = Vec::with_capacity(descriptor.color_attachments.len());
        let mut views = Vec::with_capacity(descriptor.color_attachments.len() + 1);
        for attachment in descriptor.color_attachments.values().iter() {
            let Some(attachment) = attachment else {
                color_attachments.push(None);
                continue;
            };
            let load = match attachment.load_op {
                GPULoadOp::Clear => LoadOp::Clear(color_components(attachment.clear_value.as_ref())?),
                GPULoadOp::Load => LoadOp::Load,
            };
            let store = match attachment.store_op {
                GPUStoreOp::Store => StoreOp::Store,
                GPUStoreOp::Discard => StoreOp::Discard,
            };
            let view = attachment.view.view(cx.vm());
            let resolve_target = attachment.resolve_target.map(|target| target.view(cx.vm()));
            views.push(view);
            views.extend(resolve_target);
            color_attachments.push(Some(ColorAttachment {
                view: view.id,
                depth_slice: attachment.depth_slice,
                resolve_target: resolve_target.map(|target| target.id),
                load,
                store,
            }));
        }
        let depth_stencil = descriptor.depth_stencil_attachment.as_ref().map(|attachment| {
            let view = attachment.view.view(cx.vm());
            views.push(view);
            DepthStencilAttachment {
                view: view.id,
                depth_load_op: attachment.depth_load_op.map(|op| load_op_name(op).to_owned()),
                depth_store_op: attachment.depth_store_op.map(|op| store_op_name(op).to_owned()),
                // FIXME: The spec makes a missing depthClearValue with a "clear" depthLoadOp a validation error.
                depth_clear_value: attachment.depth_clear_value.unwrap_or(0.0),
                depth_read_only: attachment.depth_read_only,
                stencil_load_op: attachment.stencil_load_op.map(|op| load_op_name(op).to_owned()),
                stencil_store_op: attachment.stencil_store_op.map(|op| store_op_name(op).to_owned()),
                stencil_clear_value: attachment.stencil_clear_value,
                stencil_read_only: attachment.stencil_read_only,
            }
        });
        let pass = GPURenderPassEncoder::create(
            cx.vm(),
            self.realm,
            self.id,
            color_attachments,
            depth_stencil,
            views,
            PassQueries {
                occlusion_query_set: descriptor.occlusion_query_set,
                timestamp_writes: descriptor.timestamp_writes.as_ref().map(|writes| {
                    let wire = timestamp_writes(
                        writes.query_set,
                        writes.beginning_of_pass_write_index,
                        writes.end_of_pass_write_index,
                    );
                    (writes.query_set, wire)
                }),
            },
        );
        Ok(labelled(pass, &descriptor.parent.label))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucommandencoder-begincomputepass
    pub(crate) fn begin_compute_pass(
        &self,
        cx: &mut CallContext,
        descriptor: GPUComputePassDescriptor,
    ) -> WebIdlResult<Ref<GPUComputePassEncoder>> {
        let pass = GPUComputePassEncoder::create(cx.vm(), self.realm, self.id, &descriptor);
        Ok(labelled(pass, &descriptor.parent.label))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucommandencoder-resolvequeryset
    pub(crate) fn resolve_query_set(
        &self,
        _: &mut CallContext,
        query_set: Ref<GPUQuerySet>,
        first_query: u32,
        query_count: u32,
        destination: Ref<GPUBuffer>,
        destination_offset: u64,
    ) -> WebIdlResult<()> {
        let command = Command::ResolveQuerySet {
            encoder: self.id,
            query_set: query_set.id,
            first_query,
            query_count,
            destination: destination.id,
            destination_offset,
        };
        client::send(self.realm, &command);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucommandencoder-copybuffertobuffer
    pub(crate) fn copy_buffer_to_buffer_from_start(
        &self,
        cx: &mut CallContext,
        source: Ref<GPUBuffer>,
        destination: Ref<GPUBuffer>,
        size: Option<u64>,
    ) -> WebIdlResult<()> {
        self.copy_buffer_to_buffer(cx, source, 0, destination, 0, size)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucommandencoder-copybuffertobuffer-source-sourceoffset-destination-destinationoffset-size
    pub(crate) fn copy_buffer_to_buffer(
        &self,
        _: &mut CallContext,
        source: Ref<GPUBuffer>,
        source_offset: u64,
        destination: Ref<GPUBuffer>,
        destination_offset: u64,
        size: Option<u64>,
    ) -> WebIdlResult<()> {
        // NB: Without a size, the copy runs to the end of the source.
        let size = size.unwrap_or_else(|| source.size_in_bytes().saturating_sub(source_offset));
        client::send(
            self.realm,
            &Command::CopyBufferToBuffer {
                encoder: self.id,
                source: source.id,
                source_offset,
                destination: destination.id,
                destination_offset,
                size,
            },
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucommandencoder-copybuffertotexture
    pub(crate) fn copy_buffer_to_texture(
        &self,
        _: &mut CallContext,
        source: GPUTexelCopyBufferInfo,
        destination: GPUTexelCopyTextureInfo,
        copy_size: GPUExtent3D,
    ) -> WebIdlResult<()> {
        let command = Command::CopyBufferToTexture {
            encoder: self.id,
            buffer: source.buffer.id,
            layout: source.parent.to_wire(),
            texture: destination.to_wire()?,
            size: copy_size.to_dimensions()?,
        };
        client::send(self.realm, &command);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucommandencoder-copytexturetobuffer
    pub(crate) fn copy_texture_to_buffer(
        &self,
        _: &mut CallContext,
        source: GPUTexelCopyTextureInfo,
        destination: GPUTexelCopyBufferInfo,
        copy_size: GPUExtent3D,
    ) -> WebIdlResult<()> {
        let command = Command::CopyTextureToBuffer {
            encoder: self.id,
            texture: source.to_wire()?,
            buffer: destination.buffer.id,
            layout: destination.parent.to_wire(),
            size: copy_size.to_dimensions()?,
        };
        client::send(self.realm, &command);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucommandencoder-copytexturetotexture
    pub(crate) fn copy_texture_to_texture(
        &self,
        _: &mut CallContext,
        source: GPUTexelCopyTextureInfo,
        destination: GPUTexelCopyTextureInfo,
        copy_size: GPUExtent3D,
    ) -> WebIdlResult<()> {
        let command = Command::CopyTextureToTexture {
            encoder: self.id,
            source: source.to_wire()?,
            destination: destination.to_wire()?,
            size: copy_size.to_dimensions()?,
        };
        client::send(self.realm, &command);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucommandencoder-clearbuffer
    pub(crate) fn clear_buffer(
        &self,
        _: &mut CallContext,
        buffer: Ref<GPUBuffer>,
        offset: u64,
        size: Option<u64>,
    ) -> WebIdlResult<()> {
        let command = Command::ClearBuffer {
            encoder: self.id,
            buffer: buffer.id,
            offset,
            size,
        };
        client::send(self.realm, &command);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucommandencoder-finish
    pub(crate) fn finish(
        &self,
        cx: &mut CallContext,
        descriptor: GPUCommandBufferDescriptor,
    ) -> WebIdlResult<Ref<GPUCommandBuffer>> {
        // FIXME: The spec makes a second finish() an error the device reports, and returns an invalid buffer.
        let id = client::allocate_object_id();
        if !self.finished.replace(true) {
            client::send(self.realm, &Command::FinishCommandEncoder { encoder: self.id, id });
        }
        Ok(labelled(
            GPUCommandBuffer::create(cx.vm(), self.realm, id),
            &descriptor.parent.label,
        ))
    }
}

label_attribute!(GPUCommandEncoder);
