/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpucompilationmessage

use ak::Utf16String;
use libjs_rust::api::{Gc, Realm, Trace, Vm};
use webgpu_wire::{CompilationMessage, CompilationMessageType};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPUCompilationMessageType {
    Error,
    Warning,
    Info,
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUCompilationMessage {
    realm: Gc<Realm>,
    #[gc(untraced)]
    message: Utf16String,
    #[gc(untraced)]
    message_type: GPUCompilationMessageType,
    #[gc(untraced)]
    line_num: u64,
    #[gc(untraced)]
    line_pos: u64,
    #[gc(untraced)]
    offset: u64,
    #[gc(untraced)]
    length: u64,
}

impl Wrappable for GPUCompilationMessage {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPUCompilationMessage {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, message: CompilationMessage) -> Ref<Self> {
        allocate(vm, |_| Self {
            realm,
            message: Utf16String::from_utf8(&message.message),
            message_type: match message.message_type {
                CompilationMessageType::Error => GPUCompilationMessageType::Error,
                CompilationMessageType::Warning => GPUCompilationMessageType::Warning,
                CompilationMessageType::Info => GPUCompilationMessageType::Info,
            },
            line_num: message.line_num.into(),
            line_pos: message.line_pos.into(),
            offset: message.offset.into(),
            length: message.length.into(),
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucompilationmessage-message
    pub(crate) fn message(&self, _: &mut CallContext) -> WebIdlResult<Utf16String> {
        Ok(self.message.clone())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucompilationmessage-type
    pub(crate) fn r#type(&self, _: &mut CallContext) -> WebIdlResult<GPUCompilationMessageType> {
        Ok(self.message_type)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucompilationmessage-linenum
    pub(crate) fn line_num(&self, _: &mut CallContext) -> WebIdlResult<u64> {
        Ok(self.line_num)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucompilationmessage-linepos
    pub(crate) fn line_pos(&self, _: &mut CallContext) -> WebIdlResult<u64> {
        Ok(self.line_pos)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucompilationmessage-offset
    pub(crate) fn offset(&self, _: &mut CallContext) -> WebIdlResult<u64> {
        Ok(self.offset)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpucompilationmessage-length
    pub(crate) fn length(&self, _: &mut CallContext) -> WebIdlResult<u64> {
        Ok(self.length)
    }
}
