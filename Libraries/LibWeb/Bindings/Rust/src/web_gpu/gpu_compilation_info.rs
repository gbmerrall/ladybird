/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpucompilationinfo

use libjs_rust::api::{Gc, MarkedVec, Realm, Trace, Vm};
use webgpu_wire::CompilationMessage;

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};
use crate::web_gpu::gpu_compilation_message::GPUCompilationMessage;

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUCompilationInfo {
    realm: Gc<Realm>,
    messages: Vec<Ref<GPUCompilationMessage>>,
}

impl Wrappable for GPUCompilationInfo {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPUCompilationInfo {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, messages: Vec<CompilationMessage>) -> Ref<Self> {
        // NB: A Vec would hide the messages from the collector while the next ones are allocated.
        let marked = MarkedVec::with_capacity(vm, messages.len());
        for message in messages {
            marked.push(GPUCompilationMessage::create(vm, realm, message));
        }
        allocate(vm, |_| Self {
            realm,
            messages: marked.to_vec(),
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucompilationinfo-messages
    pub(crate) fn messages<'vm>(
        &self,
        cx: &mut CallContext<'vm>,
    ) -> WebIdlResult<MarkedVec<'vm, Ref<GPUCompilationMessage>>> {
        let messages = MarkedVec::new(cx.vm());
        messages.extend_from_slice(&self.messages);
        Ok(messages)
    }
}
