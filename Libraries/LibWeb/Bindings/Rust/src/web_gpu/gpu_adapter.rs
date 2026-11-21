/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpuadapter

use ak::Utf16String;
use libjs_rust::api::{Gc, Realm, Trace, Value, Vm};
use webgpu_wire::{AdapterInfo, DeviceRequestError, ObjectId, Reply, Request};

use crate::bindings::interface::{Ref, Wrappable, allocate, wrap};
use crate::bindings::promise::{WebIdlPromise, reject_promise_with_exception, resolve_promise};
use crate::bindings::{CallContext, Exception, WebIdlResult};
use crate::web_gpu::client::{self, RequestSubject};
use crate::web_gpu::gpu::GPUObjectDescriptorBase;
use crate::web_gpu::gpu_adapter_info::GPUAdapterInfo;
use crate::web_gpu::gpu_device::GPUDevice;
use crate::web_gpu::gpu_supported_features::{GPUFeatureName, GPUSupportedFeatures};
use crate::web_gpu::gpu_supported_limits::GPUSupportedLimits;

pub(crate) struct GPUDeviceDescriptor {
    #[allow(
        dead_code,
        reason = "FIXME: the device is made when the compositor answers, and is not labelled yet"
    )]
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) required_features: Vec<GPUFeatureName>,
    pub(crate) required_limits: Option<Vec<(Utf16String, u64)>>,
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUAdapter {
    realm: Gc<Realm>,
    #[gc(untraced)]
    id: ObjectId,
    features: Ref<GPUSupportedFeatures>,
    limits: Ref<GPUSupportedLimits>,
    info: Ref<GPUAdapterInfo>,
}

impl Wrappable for GPUAdapter {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUAdapter {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, id: ObjectId, info: AdapterInfo) -> Ref<Self> {
        let features = GPUSupportedFeatures::create(vm, realm, info.capabilities.features.clone());
        let limits = GPUSupportedLimits::create(vm, realm, info.capabilities.limits.clone());
        let info = GPUAdapterInfo::create(vm, realm, &info);
        allocate(vm, |_| Self {
            realm,
            id,
            features,
            limits,
            info,
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapter-features
    pub(crate) fn features(&self, _: &mut CallContext) -> WebIdlResult<Ref<GPUSupportedFeatures>> {
        Ok(self.features)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapter-limits
    pub(crate) fn limits(&self, _: &mut CallContext) -> WebIdlResult<Ref<GPUSupportedLimits>> {
        Ok(self.limits)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapter-info
    pub(crate) fn info(&self, _: &mut CallContext) -> WebIdlResult<Ref<GPUAdapterInfo>> {
        Ok(self.info)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapter-isfallbackadapter
    pub(crate) fn is_fallback_adapter(&self, cx: &mut CallContext) -> WebIdlResult<bool> {
        self.info.is_fallback_adapter(cx)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpuadapter-requestdevice
    pub(crate) fn request_device(
        &self,
        cx: &mut CallContext,
        promise: WebIdlPromise,
        descriptor: GPUDeviceDescriptor,
    ) -> WebIdlResult<()> {
        // FIXME: The spec makes an adapter "consumed" by its first device, and then fails later requests.
        fn steps(vm: &Vm, request: &client::PendingRequest, reply: Option<Reply>) {
            let reject = |exception| reject_promise_with_exception(vm, request.promise, exception);
            match reply {
                Some(Reply::Device {
                    result: Ok(capabilities),
                }) => {
                    let RequestSubject::AdapterInfo(adapter_info) = request.subject else {
                        unreachable!("a device request carries the adapter's info");
                    };
                    let device = GPUDevice::create(vm, request.realm, request.object, capabilities, adapter_info);
                    let value = Value::from_object(wrap(&CallContext::new(vm, request.realm), device));
                    resolve_promise(vm, request.promise, value);
                }
                Some(Reply::Device {
                    result: Err(DeviceRequestError::Type { message }),
                }) => {
                    reject(Exception::TypeError(Utf16String::from_utf8(&message)));
                }
                Some(Reply::Device {
                    result: Err(DeviceRequestError::Operation { message }),
                }) => {
                    reject(Exception::Dom("OperationError", Utf16String::from_utf8(&message)));
                }
                _ => reject(Exception::Dom(
                    "OperationError",
                    Utf16String::from_utf8("There is no compositor to create a device"),
                )),
            }
        }
        let id = client::allocate_object_id();
        let body = Request::RequestDevice {
            adapter: self.id,
            id,
            required_features: descriptor
                .required_features
                .iter()
                .map(|feature| feature.name().to_owned())
                .collect(),
            required_limits: descriptor
                .required_limits
                .unwrap_or_default()
                .iter()
                .map(|(name, value)| (String::from_utf16_lossy(&name.to_utf16()), *value))
                .collect(),
        };
        client::request_about(
            cx.vm(),
            self.realm,
            promise,
            id,
            body,
            steps,
            RequestSubject::AdapterInfo(self.info),
        );
        Ok(())
    }
}
