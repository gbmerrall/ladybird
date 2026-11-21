/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpurenderpassencoder

use core::cell::{Cell, RefCell};

use libjs_rust::api::{Gc, GcRefCell, MarkedVec, Realm, Trace, Vm};
use webgpu_wire::{ColorAttachment, Command, DepthStencilAttachment, ObjectId, RenderPassOp, TimestampWrites};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};
use crate::web_gpu::gpu_bind_group::GPUBindGroup;
use crate::web_gpu::gpu_buffer::GPUBuffer;
use crate::web_gpu::gpu_command_encoder::{GPUColor, color_components};
use crate::web_gpu::gpu_device::GPUIndexFormat;
use crate::web_gpu::gpu_query_set::GPUQuerySet;
use crate::web_gpu::gpu_render_bundle::GPURenderBundle;
use crate::web_gpu::gpu_render_pipeline::GPURenderPipeline;
use crate::web_gpu::gpu_texture_view::GPUTextureView;
use crate::web_gpu::{Label, client};

/// What a render pass's recorded commands name only by id, kept alive until the pass is sent.
#[derive(Clone, Copy, Trace)]
#[gc(crate = libjs_rust)]
enum Used {
    Pipeline(Ref<GPURenderPipeline>),
    BindGroup(Ref<GPUBindGroup>),
    Buffer(Ref<GPUBuffer>),
    QuerySet(Ref<GPUQuerySet>),
    Bundle(Ref<GPURenderBundle>),
}

/// The queries of a render pass: its occlusion query set and its timestamp writes.
#[derive(Default)]
pub(crate) struct PassQueries {
    pub(crate) occlusion_query_set: Option<Ref<GPUQuerySet>>,
    pub(crate) timestamp_writes: Option<(Ref<GPUQuerySet>, TimestampWrites)>,
}

/// A render pass records its commands here and sends them as one when it ends, so that the compositor does not hold
/// a pass open across messages.
#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPURenderPassEncoder {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    encoder: ObjectId,
    #[gc(untraced)]
    color_attachments: Vec<Option<ColorAttachment>>,
    #[gc(untraced)]
    depth_stencil: Option<DepthStencilAttachment>,
    #[gc(untraced)]
    occlusion_query_set: Option<ObjectId>,
    #[gc(untraced)]
    timestamp_writes: Option<TimestampWrites>,
    #[gc(untraced)]
    ops: RefCell<Vec<RenderPassOp>>,
    #[gc(untraced)]
    ended: Cell<bool>,
    views: Vec<Ref<GPUTextureView>>,
    used: GcRefCell<Vec<Used>>,
}

impl Wrappable for GPURenderPassEncoder {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPURenderPassEncoder {
    pub(crate) fn create(
        vm: &Vm,
        realm: Gc<Realm>,
        encoder: ObjectId,
        color_attachments: Vec<Option<ColorAttachment>>,
        depth_stencil: Option<DepthStencilAttachment>,
        views: Vec<Ref<GPUTextureView>>,
        queries: PassQueries,
    ) -> Ref<Self> {
        let used = queries
            .occlusion_query_set
            .into_iter()
            .chain(queries.timestamp_writes.as_ref().map(|(query_set, _)| *query_set))
            .map(Used::QuerySet)
            .collect();
        allocate(vm, |_| Self {
            label: Label::default(),
            realm,
            encoder,
            color_attachments,
            depth_stencil,
            occlusion_query_set: queries.occlusion_query_set.map(|query_set| query_set.id),
            timestamp_writes: queries.timestamp_writes.map(|(_, writes)| writes),
            ops: RefCell::new(Vec::new()),
            ended: Cell::new(false),
            views,
            used: GcRefCell::new(used),
        })
    }

    fn record(&self, vm: &Vm, op: RenderPassOp, used: Option<Used>) {
        // FIXME: Commands after end() make the encoder invalid, which the device would report.
        if self.ended.get() {
            return;
        }
        if let Some(used) = used {
            self.used.borrow_mut(vm).push(used);
        }
        self.ops.borrow_mut().push(op);
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

    // https://gpuweb.github.io/gpuweb/#dom-gpurenderpassencoder-setviewport
    #[expect(clippy::too_many_arguments, reason = "the arguments of setViewport()")]
    pub(crate) fn set_viewport(
        &self,
        cx: &mut CallContext,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> WebIdlResult<()> {
        let op = RenderPassOp::SetViewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        };
        self.record(cx.vm(), op, None);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurenderpassencoder-setscissorrect
    pub(crate) fn set_scissor_rect(
        &self,
        cx: &mut CallContext,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> WebIdlResult<()> {
        self.record(cx.vm(), RenderPassOp::SetScissorRect { x, y, width, height }, None);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurenderpassencoder-setblendconstant
    pub(crate) fn set_blend_constant(&self, cx: &mut CallContext, color: GPUColor) -> WebIdlResult<()> {
        let color = color_components(Some(&color))?;
        self.record(cx.vm(), RenderPassOp::SetBlendConstant { color }, None);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurenderpassencoder-setstencilreference
    pub(crate) fn set_stencil_reference(&self, cx: &mut CallContext, reference: u32) -> WebIdlResult<()> {
        self.record(cx.vm(), RenderPassOp::SetStencilReference { reference }, None);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurenderpassencoder-beginocclusionquery
    pub(crate) fn begin_occlusion_query(&self, cx: &mut CallContext, query_index: u32) -> WebIdlResult<()> {
        self.record(cx.vm(), RenderPassOp::BeginOcclusionQuery { index: query_index }, None);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurenderpassencoder-executebundles
    pub(crate) fn execute_bundles(
        &self,
        cx: &mut CallContext,
        bundles: MarkedVec<'_, Ref<GPURenderBundle>>,
    ) -> WebIdlResult<()> {
        if self.ended.get() {
            return Ok(());
        }
        let bundles = bundles.values();
        for bundle in bundles.iter() {
            self.used.borrow_mut(cx.vm()).push(Used::Bundle(*bundle));
        }
        let op = RenderPassOp::ExecuteBundles {
            bundles: bundles.iter().map(|bundle| bundle.id).collect(),
        };
        self.record(cx.vm(), op, None);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurenderpassencoder-endocclusionquery
    pub(crate) fn end_occlusion_query(&self, cx: &mut CallContext) -> WebIdlResult<()> {
        self.record(cx.vm(), RenderPassOp::EndOcclusionQuery, None);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpurenderpassencoder-end
    pub(crate) fn end(&self, _: &mut CallContext) -> WebIdlResult<()> {
        if self.ended.replace(true) {
            // FIXME: The spec makes ending a pass twice a validation error of the encoder.
            return Ok(());
        }
        client::send(
            self.realm,
            &Command::EncodeRenderPass {
                encoder: self.encoder,
                color_attachments: self.color_attachments.clone(),
                depth_stencil: self.depth_stencil.clone(),
                occlusion_query_set: self.occlusion_query_set,
                timestamp_writes: self.timestamp_writes.clone(),
                ops: self.ops.take(),
            },
        );
        Ok(())
    }
}

label_attribute!(GPURenderPassEncoder);
