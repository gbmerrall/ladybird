/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpuvalidationerror

use ak::Utf16String;
use libjs_rust::api::{Gc, Realm, Trace};

use crate::bindings::interface::{Inherits, Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};
use crate::web_gpu::gpu_error::GPUError;

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUValidationError {
    pub(crate) error: GPUError,
}

impl Wrappable for GPUValidationError {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        self.error.relevant_realm()
    }
}

impl Inherits<GPUError> for GPUValidationError {
    fn parent(&self) -> &GPUError {
        &self.error
    }
}

impl GPUValidationError {
    // https://gpuweb.github.io/gpuweb/#dom-gpuvalidationerror-gpuvalidationerror
    pub(crate) fn construct(cx: &mut CallContext, message: Utf16String) -> WebIdlResult<Ref<Self>> {
        let realm = cx.realm();
        Ok(allocate(cx.vm(), |_| Self {
            error: GPUError::new(realm, message),
        }))
    }
}
