/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpuadapterinfo

use ak::Utf16String;
use libjs_rust::api::{Gc, Realm, Trace, Vm};
use webgpu_wire::AdapterInfo;

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUAdapterInfo {
    realm: Gc<Realm>,
    #[gc(untraced)]
    vendor: Utf16String,
    #[gc(untraced)]
    architecture: Utf16String,
    #[gc(untraced)]
    device: Utf16String,
    #[gc(untraced)]
    description: Utf16String,
    #[gc(untraced)]
    is_fallback_adapter: bool,
}

impl Wrappable for GPUAdapterInfo {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPUAdapterInfo {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, info: &AdapterInfo) -> Ref<Self> {
        allocate(vm, |_| Self {
            realm,
            vendor: Utf16String::from_utf8(&info.vendor),
            architecture: Utf16String::from_utf8(&info.architecture),
            device: Utf16String::from_utf8(&info.device),
            description: Utf16String::from_utf8(&info.description),
            is_fallback_adapter: info.is_fallback_adapter,
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapterinfo-vendor
    pub(crate) fn vendor(&self, _: &mut CallContext) -> WebIdlResult<Utf16String> {
        Ok(self.vendor.clone())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapterinfo-architecture
    pub(crate) fn architecture(&self, _: &mut CallContext) -> WebIdlResult<Utf16String> {
        Ok(self.architecture.clone())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapterinfo-device
    pub(crate) fn device(&self, _: &mut CallContext) -> WebIdlResult<Utf16String> {
        Ok(self.device.clone())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapterinfo-description
    pub(crate) fn description(&self, _: &mut CallContext) -> WebIdlResult<Utf16String> {
        Ok(self.description.clone())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapterinfo-subgroupminsize
    // FIXME: Subgroups, which wgpu reports for its "subgroups" feature.
    pub(crate) fn subgroup_min_size(&self, _: &mut CallContext) -> WebIdlResult<u32> {
        Ok(4)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapterinfo-subgroupmaxsize
    pub(crate) fn subgroup_max_size(&self, _: &mut CallContext) -> WebIdlResult<u32> {
        Ok(128)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapterinfo-isfallbackadapter
    pub(crate) fn is_fallback_adapter(&self, _: &mut CallContext) -> WebIdlResult<bool> {
        Ok(self.is_fallback_adapter)
    }
}
