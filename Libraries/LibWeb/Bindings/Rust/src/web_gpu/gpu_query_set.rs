/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpuqueryset

use libjs_rust::api::{Gc, Realm, Trace, Vm};
use webgpu_wire::{Command, ObjectId, TimestampWrites};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};
use crate::web_gpu::gpu::GPUObjectDescriptorBase;
use crate::web_gpu::{Label, client};

string_enum!(GPUQueryType, "gpuquerytype" {
    Occlusion => "occlusion",
    Timestamp => "timestamp",
});

pub(crate) struct GPUQuerySetDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) r#type: GPUQueryType,
    pub(crate) count: u32,
}

/// The members of GPURenderPassTimestampWrites and GPUComputePassTimestampWrites, which are the same.
pub(crate) fn timestamp_writes(
    query_set: Ref<GPUQuerySet>,
    beginning_of_pass_write_index: Option<u32>,
    end_of_pass_write_index: Option<u32>,
) -> TimestampWrites {
    TimestampWrites {
        query_set: query_set.id,
        beginning_of_pass_write_index,
        end_of_pass_write_index,
    }
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUQuerySet {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
    #[gc(untraced)]
    r#type: GPUQueryType,
    #[gc(untraced)]
    count: u32,
}

impl Wrappable for GPUQuerySet {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUQuerySet {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, device: ObjectId, descriptor: &GPUQuerySetDescriptor) -> Ref<Self> {
        let id = client::allocate_object_id();
        client::send(
            realm,
            &Command::CreateQuerySet {
                device,
                id,
                kind: descriptor.r#type.name().to_owned(),
                count: descriptor.count,
            },
        );
        allocate(vm, |_| Self {
            label: Label::default(),
            realm,
            id,
            r#type: descriptor.r#type,
            count: descriptor.count,
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuqueryset-destroy
    pub(crate) fn destroy(&self, _: &mut CallContext) -> WebIdlResult<()> {
        client::send(self.realm, &Command::Destroy { id: self.id });
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuqueryset-type
    pub(crate) fn r#type(&self, _: &mut CallContext) -> WebIdlResult<GPUQueryType> {
        Ok(self.r#type)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuqueryset-count
    pub(crate) fn count(&self, _: &mut CallContext) -> WebIdlResult<u32> {
        Ok(self.count)
    }
}

label_attribute!(GPUQuerySet);
