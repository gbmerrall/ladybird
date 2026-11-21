/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpudevicelostinfo

use ak::Utf16String;
use libjs_rust::api::{Gc, Realm, Trace, Vm};
use webgpu_wire::DeviceLostReason;

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPUDeviceLostReason {
    Unknown,
    Destroyed,
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUDeviceLostInfo {
    realm: Gc<Realm>,
    #[gc(untraced)]
    reason: GPUDeviceLostReason,
    #[gc(untraced)]
    message: Utf16String,
}

impl Wrappable for GPUDeviceLostInfo {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPUDeviceLostInfo {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, reason: DeviceLostReason, message: &str) -> Ref<Self> {
        let reason = match reason {
            DeviceLostReason::Unknown => GPUDeviceLostReason::Unknown,
            DeviceLostReason::Destroyed => GPUDeviceLostReason::Destroyed,
        };
        allocate(vm, |_| Self {
            realm,
            reason,
            message: Utf16String::from_utf8(message),
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevicelostinfo-reason
    pub(crate) fn reason(&self, _: &mut CallContext) -> WebIdlResult<GPUDeviceLostReason> {
        Ok(self.reason)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevicelostinfo-message
    pub(crate) fn message(&self, _: &mut CallContext) -> WebIdlResult<Utf16String> {
        Ok(self.message.clone())
    }
}
