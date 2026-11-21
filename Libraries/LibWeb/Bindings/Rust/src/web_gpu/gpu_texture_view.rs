/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gputextureview

use libjs_rust::api::{Gc, Realm, Trace, Vm};
use webgpu_wire::ObjectId;

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::web_gpu::{Label, client};

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUTextureView {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
}

impl Wrappable for GPUTextureView {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUTextureView {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, id: ObjectId) -> Ref<Self> {
        allocate(vm, |_| Self {
            realm,
            id,
            label: Label::default(),
        })
    }
}

label_attribute!(GPUTextureView);
