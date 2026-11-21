/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpurenderbundleencoder

use core::cell::{Cell, RefCell};

use libjs_rust::api::{Gc, GcRefCell, Realm, Trace, Vm};
use webgpu_wire::{Command, ObjectId, RenderBundleDescriptor, RenderPassOp};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};
use crate::web_gpu::Label;
use crate::web_gpu::gpu::{GPUObjectDescriptorBase, GPUTextureFormat};
use crate::web_gpu::gpu_bind_group::GPUBindGroup;
use crate::web_gpu::gpu_buffer::GPUBuffer;
use crate::web_gpu::gpu_device::GPUIndexFormat;
use crate::web_gpu::gpu_render_bundle::GPURenderBundle;
use crate::web_gpu::gpu_render_pipeline::GPURenderPipeline;
use crate::web_gpu::{client, labelled};

pub(crate) struct GPURenderBundleDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
}

pub(crate) struct GPURenderPassLayout {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) color_formats: Vec<Option<GPUTextureFormat>>,
    pub(crate) depth_stencil_format: Option<GPUTextureFormat>,
    pub(crate) sample_count: u32,
}

pub(crate) struct GPURenderBundleEncoderDescriptor {
    pub(crate) parent: GPURenderPassLayout,
    pub(crate) depth_read_only: bool,
    pub(crate) stencil_read_only: bool,
}

impl GPURenderBundleEncoderDescriptor {
    fn to_wire(&self) -> RenderBundleDescriptor {
        RenderBundleDescriptor {
            color_formats: self
                .parent
                .color_formats
                .iter()
                .map(|format| format.map(|format| format.name().to_owned()))
                .collect(),
            depth_stencil_format: self.parent.depth_stencil_format.map(|format| format.name().to_owned()),
            sample_count: self.parent.sample_count,
            depth_read_only: self.depth_read_only,
            stencil_read_only: self.stencil_read_only,
        }
    }
}

/// What a bundle's recorded commands name only by id, kept alive until the bundle is sent.
#[derive(Clone, Copy, Trace)]
#[gc(crate = libjs_rust)]
enum Used {
    Pipeline(Ref<GPURenderPipeline>),
    BindGroup(Ref<GPUBindGroup>),
    Buffer(Ref<GPUBuffer>),
}

/// A render bundle encoder records its commands here, as a render pass does, and sends them as one when script
/// finishes it.
#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPURenderBundleEncoder {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    device: ObjectId,
    #[gc(untraced)]
    descriptor: RenderBundleDescriptor,
    #[gc(untraced)]
    ops: RefCell<Vec<RenderPassOp>>,
    #[gc(untraced)]
    ended: Cell<bool>,
    used: GcRefCell<Vec<Used>>,
}

impl Wrappable for GPURenderBundleEncoder {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPURenderBundleEncoder {
    pub(crate) fn create(
        vm: &Vm,
        realm: Gc<Realm>,
        device: ObjectId,
        descriptor: &GPURenderBundleEncoderDescriptor,
    ) -> Ref<Self> {
        allocate(vm, |_| Self {
            label: Label::default(),
            realm,
            device,
            descriptor: descriptor.to_wire(),
            ops: RefCell::new(Vec::new()),
            ended: Cell::new(false),
            used: GcRefCell::default(),
        })
    }

    fn record(&self, vm: &Vm, op: RenderPassOp, used: Option<Used>) {
        // FIXME: Commands after finish() make the encoder invalid, which the device would report.
        if self.ended.get() {
            return;
        }
        if let Some(used) = used {
            self.used.borrow_mut(vm).push(used);
        }
        self.ops.borrow_mut().push(op);
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurenderbundleencoder-finish
    pub(crate) fn finish(
        &self,
        cx: &mut CallContext,
        descriptor: GPURenderBundleDescriptor,
    ) -> WebIdlResult<Ref<GPURenderBundle>> {
        let id = client::allocate_object_id();
        // FIXME: The spec makes a second finish() an error the device reports, and returns an invalid bundle.
        if !self.ended.replace(true) {
            client::send(
                self.realm,
                &Command::CreateRenderBundle {
                    device: self.device,
                    id,
                    descriptor: self.descriptor.clone(),
                    ops: self.ops.take(),
                },
            );
        }
        Ok(labelled(
            GPURenderBundle::create(cx.vm(), self.realm, id),
            &descriptor.parent.label,
        ))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-setpipeline
    pub(crate) fn set_pipeline(&self, cx: &mut CallContext, pipeline: Ref<GPURenderPipeline>) -> WebIdlResult<()> {
        self.record(
            cx.vm(),
            RenderPassOp::SetPipeline { pipeline: pipeline.id },
            Some(Used::Pipeline(pipeline)),
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpubindingcommandsmixin-setbindgroup
    pub(crate) fn set_bind_group(
        &self,
        cx: &mut CallContext,
        index: u32,
        bind_group: Option<Ref<GPUBindGroup>>,
        dynamic_offsets: Vec<u32>,
    ) -> WebIdlResult<()> {
        let op = RenderPassOp::SetBindGroup {
            index,
            group: bind_group.map(|group| group.id),
            dynamic_offsets,
        };
        self.record(cx.vm(), op, bind_group.map(Used::BindGroup));
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-setindexbuffer
    pub(crate) fn set_index_buffer(
        &self,
        cx: &mut CallContext,
        buffer: Ref<GPUBuffer>,
        index_format: GPUIndexFormat,
        offset: u64,
        size: Option<u64>,
    ) -> WebIdlResult<()> {
        let op = RenderPassOp::SetIndexBuffer {
            buffer: buffer.id,
            format: index_format.name().to_owned(),
            offset,
            size,
        };
        self.record(cx.vm(), op, Some(Used::Buffer(buffer)));
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-setvertexbuffer
    pub(crate) fn set_vertex_buffer(
        &self,
        cx: &mut CallContext,
        slot: u32,
        buffer: Option<Ref<GPUBuffer>>,
        offset: u64,
        size: Option<u64>,
    ) -> WebIdlResult<()> {
        let op = RenderPassOp::SetVertexBuffer {
            slot,
            buffer: buffer.map(|buffer| buffer.id),
            offset,
            size,
        };
        self.record(cx.vm(), op, buffer.map(Used::Buffer));
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-draw
    pub(crate) fn draw(
        &self,
        cx: &mut CallContext,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> WebIdlResult<()> {
        let op = RenderPassOp::Draw {
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        };
        self.record(cx.vm(), op, None);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-drawindexed
    pub(crate) fn draw_indexed(
        &self,
        cx: &mut CallContext,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> WebIdlResult<()> {
        let op = RenderPassOp::DrawIndexed {
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        };
        self.record(cx.vm(), op, None);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-drawindirect
    pub(crate) fn draw_indirect(
        &self,
        cx: &mut CallContext,
        indirect_buffer: Ref<GPUBuffer>,
        indirect_offset: u64,
    ) -> WebIdlResult<()> {
        let op = RenderPassOp::DrawIndirect {
            buffer: indirect_buffer.id,
            offset: indirect_offset,
        };
        self.record(cx.vm(), op, Some(Used::Buffer(indirect_buffer)));
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurendercommandsmixin-drawindexedindirect
    pub(crate) fn draw_indexed_indirect(
        &self,
        cx: &mut CallContext,
        indirect_buffer: Ref<GPUBuffer>,
        indirect_offset: u64,
    ) -> WebIdlResult<()> {
        let op = RenderPassOp::DrawIndexedIndirect {
            buffer: indirect_buffer.id,
            offset: indirect_offset,
        };
        self.record(cx.vm(), op, Some(Used::Buffer(indirect_buffer)));
        Ok(())
    }
}

label_attribute!(GPURenderBundleEncoder);
