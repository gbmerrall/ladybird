/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpurenderpipeline

use libjs_rust::api::{Gc, Realm, Trace, Vm};
use webgpu_wire::{Command, ObjectId};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};
use crate::web_gpu::gpu_bind_group_layout::GPUBindGroupLayout;
use crate::web_gpu::{Label, client};

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPURenderPipeline {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
}

impl Wrappable for GPURenderPipeline {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPURenderPipeline {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, id: ObjectId) -> Ref<Self> {
        allocate(vm, |_| Self {
            realm,
            id,
            label: Label::default(),
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpupipelinebase-getbindgrouplayout
    pub(crate) fn get_bind_group_layout(
        &self,
        cx: &mut CallContext,
        index: u32,
    ) -> WebIdlResult<Ref<GPUBindGroupLayout>> {
        let id = client::allocate_object_id();
        client::send(
            self.realm,
            &Command::GetBindGroupLayout {
                pipeline: self.id,
                index,
                id,
            },
        );
        Ok(GPUBindGroupLayout::with_id(cx.vm(), self.realm, id))
    }
}

label_attribute!(GPURenderPipeline);
