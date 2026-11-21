/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpucomputepassencoder

use core::cell::{Cell, RefCell};

use libjs_rust::api::{Gc, GcRefCell, Realm, Trace, Vm};
use webgpu_wire::{Command, ComputePassOp, ObjectId, TimestampWrites};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};
use crate::web_gpu::gpu::GPUObjectDescriptorBase;
use crate::web_gpu::gpu_bind_group::GPUBindGroup;
use crate::web_gpu::gpu_buffer::GPUBuffer;
use crate::web_gpu::gpu_compute_pipeline::GPUComputePipeline;
use crate::web_gpu::gpu_query_set::{GPUQuerySet, timestamp_writes};
use crate::web_gpu::{Label, client};

pub(crate) struct GPUComputePassDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) timestamp_writes: Option<GPUComputePassTimestampWrites>,
}

pub(crate) struct GPUComputePassTimestampWrites {
    pub(crate) query_set: Ref<GPUQuerySet>,
    pub(crate) beginning_of_pass_write_index: Option<u32>,
    pub(crate) end_of_pass_write_index: Option<u32>,
}

/// What a compute pass's recorded commands name only by id, kept alive until the pass is sent.
#[derive(Clone, Copy, Trace)]
#[gc(crate = libjs_rust)]
enum Used {
    Pipeline(Ref<GPUComputePipeline>),
    BindGroup(Ref<GPUBindGroup>),
    Buffer(Ref<GPUBuffer>),
    QuerySet(Ref<GPUQuerySet>),
}

/// A compute pass records its commands here and sends them as one when it ends, as a render pass does.
#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUComputePassEncoder {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    encoder: ObjectId,
    #[gc(untraced)]
    timestamp_writes: Option<TimestampWrites>,
    #[gc(untraced)]
    ops: RefCell<Vec<ComputePassOp>>,
    #[gc(untraced)]
    ended: Cell<bool>,
    used: GcRefCell<Vec<Used>>,
}

impl Wrappable for GPUComputePassEncoder {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPUComputePassEncoder {
    pub(crate) fn create(
        vm: &Vm,
        realm: Gc<Realm>,
        encoder: ObjectId,
        descriptor: &GPUComputePassDescriptor,
    ) -> Ref<Self> {
        let writes = descriptor.timestamp_writes.as_ref();
        let used = writes.map(|writes| Used::QuerySet(writes.query_set));
        allocate(vm, |_| Self {
            label: Label::default(),
            realm,
            encoder,
            timestamp_writes: writes.map(|writes| {
                timestamp_writes(
                    writes.query_set,
                    writes.beginning_of_pass_write_index,
                    writes.end_of_pass_write_index,
                )
            }),
            ops: RefCell::new(Vec::new()),
            ended: Cell::new(false),
            used: GcRefCell::new(used.into_iter().collect()),
        })
    }

    fn record(&self, vm: &Vm, op: ComputePassOp, used: Option<Used>) {
        // FIXME: Commands after end() make the encoder invalid, which the device would report.
        if self.ended.get() {
            return;
        }
        if let Some(used) = used {
            self.used.borrow_mut(vm).push(used);
        }
        self.ops.borrow_mut().push(op);
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucomputepassencoder-setpipeline
    pub(crate) fn set_pipeline(&self, cx: &mut CallContext, pipeline: Ref<GPUComputePipeline>) -> WebIdlResult<()> {
        self.record(
            cx.vm(),
            ComputePassOp::SetPipeline { pipeline: pipeline.id },
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
        let op = ComputePassOp::SetBindGroup {
            index,
            group: bind_group.map(|group| group.id),
            dynamic_offsets,
        };
        self.record(cx.vm(), op, bind_group.map(Used::BindGroup));
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucomputepassencoder-dispatchworkgroups
    pub(crate) fn dispatch_workgroups(
        &self,
        cx: &mut CallContext,
        workgroup_count_x: u32,
        workgroup_count_y: u32,
        workgroup_count_z: u32,
    ) -> WebIdlResult<()> {
        let op = ComputePassOp::DispatchWorkgroups {
            x: workgroup_count_x,
            y: workgroup_count_y,
            z: workgroup_count_z,
        };
        self.record(cx.vm(), op, None);
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucomputepassencoder-dispatchworkgroupsindirect
    pub(crate) fn dispatch_workgroups_indirect(
        &self,
        cx: &mut CallContext,
        indirect_buffer: Ref<GPUBuffer>,
        indirect_offset: u64,
    ) -> WebIdlResult<()> {
        let op = ComputePassOp::DispatchWorkgroupsIndirect {
            buffer: indirect_buffer.id,
            offset: indirect_offset,
        };
        self.record(cx.vm(), op, Some(Used::Buffer(indirect_buffer)));
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucomputepassencoder-end
    pub(crate) fn end(&self, _: &mut CallContext) -> WebIdlResult<()> {
        if self.ended.replace(true) {
            // FIXME: The spec makes ending a pass twice a validation error of the encoder.
            return Ok(());
        }
        client::send(
            self.realm,
            &Command::EncodeComputePass {
                encoder: self.encoder,
                timestamp_writes: self.timestamp_writes.clone(),
                ops: self.ops.take(),
            },
        );
        Ok(())
    }
}

label_attribute!(GPUComputePassEncoder);
