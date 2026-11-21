/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpupipelinelayout

use libjs_rust::api::{Gc, MarkedVec, Realm, Trace, Vm};
use webgpu_wire::{Command, ObjectId};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::web_gpu::gpu::GPUObjectDescriptorBase;
use crate::web_gpu::gpu_bind_group_layout::GPUBindGroupLayout;
use crate::web_gpu::{Label, client};

pub(crate) struct GPUPipelineLayoutDescriptor<'vm> {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) bind_group_layouts: MarkedVec<'vm, Ref<GPUBindGroupLayout>>,
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUPipelineLayout {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
}

impl Wrappable for GPUPipelineLayout {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUPipelineLayout {
    pub(crate) fn create(
        vm: &Vm,
        realm: Gc<Realm>,
        device: ObjectId,
        descriptor: &GPUPipelineLayoutDescriptor<'_>,
    ) -> Ref<Self> {
        let id = client::allocate_object_id();
        let bind_group_layouts = descriptor
            .bind_group_layouts
            .values()
            .iter()
            .map(|layout| layout.id)
            .collect();
        client::send(
            realm,
            &Command::CreatePipelineLayout {
                device,
                id,
                bind_group_layouts,
            },
        );
        allocate(vm, |_| Self {
            realm,
            id,
            label: Label::default(),
        })
    }
}

label_attribute!(GPUPipelineLayout);
