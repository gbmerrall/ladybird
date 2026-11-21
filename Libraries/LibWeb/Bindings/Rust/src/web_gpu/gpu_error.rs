/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpuerror

use core::ops::Deref;

use ak::Utf16String;
use libjs_rust::api::{Gc, Realm, ThrowCompletionOr, Trace, Value, Vm};
use webgpu_wire::{ErrorFilter, GpuError};

use crate::bindings::conversions::ToJs;
use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};
use crate::web_gpu::gpu_internal_error::GPUInternalError;
use crate::web_gpu::gpu_out_of_memory_error::GPUOutOfMemoryError;
use crate::web_gpu::gpu_validation_error::GPUValidationError;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPUErrorFilter {
    Validation,
    OutOfMemory,
    Internal,
}

impl From<GPUErrorFilter> for ErrorFilter {
    fn from(filter: GPUErrorFilter) -> Self {
        match filter {
            GPUErrorFilter::Validation => Self::Validation,
            GPUErrorFilter::OutOfMemory => Self::OutOfMemory,
            GPUErrorFilter::Internal => Self::Internal,
        }
    }
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUError {
    realm: Gc<Realm>,
    #[gc(untraced)]
    message: Utf16String,
}

impl Wrappable for GPUError {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPUError {
    pub(crate) fn new(realm: Gc<Realm>, message: Utf16String) -> Self {
        Self { realm, message }
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuerror-message
    pub(crate) fn message(&self, _: &mut CallContext) -> WebIdlResult<Utf16String> {
        Ok(self.message.clone())
    }
}

/// A GPUError, which is always one of its subclasses.
#[derive(Clone, Copy, PartialEq, Trace)]
#[gc(crate = libjs_rust)]
#[allow(clippy::enum_variant_names, reason = "the variants have the interfaces' names")]
pub(crate) enum AnyGPUError {
    GPUError(Ref<GPUError>),
    GPUValidationError(Ref<GPUValidationError>),
    GPUOutOfMemoryError(Ref<GPUOutOfMemoryError>),
    GPUInternalError(Ref<GPUInternalError>),
}

impl Deref for AnyGPUError {
    type Target = GPUError;

    fn deref(&self) -> &GPUError {
        match self {
            Self::GPUError(error) => error,
            Self::GPUValidationError(error) => &error.error,
            Self::GPUOutOfMemoryError(error) => &error.error,
            Self::GPUInternalError(error) => &error.error,
        }
    }
}

impl ToJs for AnyGPUError {
    fn to_js(self, cx: &CallContext) -> ThrowCompletionOr<Value> {
        match self {
            Self::GPUError(error) => error.to_js(cx),
            Self::GPUValidationError(error) => error.to_js(cx),
            Self::GPUOutOfMemoryError(error) => error.to_js(cx),
            Self::GPUInternalError(error) => error.to_js(cx),
        }
    }
}

impl AnyGPUError {
    /// The error object for an error the compositor reports.
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, error: GpuError) -> Self {
        let error_of = |message: String| GPUError::new(realm, Utf16String::from_utf8(&message));
        match error {
            GpuError::Validation { message } => Self::GPUValidationError(allocate(vm, |_| GPUValidationError {
                error: error_of(message),
            })),
            GpuError::OutOfMemory { message } => Self::GPUOutOfMemoryError(allocate(vm, |_| GPUOutOfMemoryError {
                error: error_of(message),
            })),
            GpuError::Internal { message } => Self::GPUInternalError(allocate(vm, |_| GPUInternalError {
                error: error_of(message),
            })),
        }
    }
}
