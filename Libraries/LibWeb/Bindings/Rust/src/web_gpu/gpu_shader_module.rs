/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpushadermodule

use libjs_rust::api::{Gc, Realm, Trace, Value, Vm};
use webgpu_wire::{ObjectId, Reply, Request};

use crate::bindings::interface::{Ref, Wrappable, allocate, wrap};
use crate::bindings::promise::{WebIdlPromise, resolve_promise};
use crate::bindings::{CallContext, WebIdlResult};
use crate::web_gpu::gpu_compilation_info::GPUCompilationInfo;
use crate::web_gpu::{Label, client};

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUShaderModule {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
}

impl Wrappable for GPUShaderModule {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUShaderModule {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, id: ObjectId) -> Ref<Self> {
        allocate(vm, |_| Self {
            realm,
            id,
            label: Label::default(),
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpushadermodule-getcompilationinfo
    pub(crate) fn get_compilation_info(&self, cx: &mut CallContext, promise: WebIdlPromise) -> WebIdlResult<()> {
        fn steps(vm: &Vm, request: &client::PendingRequest, reply: Option<Reply>) {
            let messages = match reply {
                Some(Reply::CompilationInfo { messages }) => messages,
                _ => Vec::new(),
            };
            let info = GPUCompilationInfo::create(vm, request.realm, messages);
            resolve_promise(
                vm,
                request.promise,
                Value::from_object(wrap(&CallContext::new(vm, request.realm), info)),
            );
        }
        client::request(
            cx.vm(),
            self.realm,
            promise,
            self.id,
            Request::GetCompilationInfo { module: self.id },
            steps,
        );
        Ok(())
    }
}

label_attribute!(GPUShaderModule);
