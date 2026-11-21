/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpuuncapturederrorevent

use core::ffi::c_void;
use core::ptr::NonNull;

use ak::Utf16String;
use libjs_rust::api::{Gc, Realm, Trace, Vm};

use crate::bindings::event_target::CppEvent;
use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};
use crate::dom::event::EventInit;
use crate::web_gpu::gpu_error::AnyGPUError;

pub(crate) struct GPUUncapturedErrorEventInit {
    pub(crate) parent: EventInit,
    pub(crate) error: AnyGPUError,
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUUncapturedErrorEvent {
    event: CppEvent,
    error: AnyGPUError,
}

impl Wrappable for GPUUncapturedErrorEvent {
    fn cpp_parent(&self) -> Option<NonNull<c_void>> {
        self.event.as_cpp_parent()
    }
}

impl GPUUncapturedErrorEvent {
    pub(crate) fn create(
        vm: &Vm,
        realm: Gc<Realm>,
        r#type: &Utf16String,
        event_init: &EventInit,
        error: AnyGPUError,
    ) -> Ref<Self> {
        let event = CppEvent::create_for_owner(
            realm,
            r#type,
            event_init.bubbles,
            event_init.cancelable,
            event_init.composed,
        );
        let object = allocate(vm, |_| Self { event, error });
        object.event.set_owner(object);
        object
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuuncapturederrorevent-gpuuncapturederrorevent
    pub(crate) fn construct(
        cx: &mut CallContext,
        r#type: Utf16String,
        init: GPUUncapturedErrorEventInit,
    ) -> WebIdlResult<Ref<Self>> {
        Ok(Self::create(cx.vm(), cx.realm(), &r#type, &init.parent, init.error))
    }

    pub(crate) fn event(&self) -> &CppEvent {
        &self.event
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuuncapturederrorevent-error
    pub(crate) fn error(&self, _: &mut CallContext) -> WebIdlResult<AnyGPUError> {
        Ok(self.error)
    }
}
