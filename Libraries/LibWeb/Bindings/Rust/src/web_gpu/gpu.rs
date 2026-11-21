/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpu-interface

use libjs_rust::api::{Gc, Realm, Trace, Value, Vm};
use webgpu_wire::{PowerPreference, Reply, Request};

use crate::bindings::conversions::UsvString;
use crate::bindings::interface::{Ref, Wrappable, allocate, wrap};
use crate::bindings::promise::{TaskSource, WebIdlPromise, queue_a_global_task, resolve_promise};
use crate::bindings::{CallContext, WebIdlResult};
use crate::web_gpu::client;
use crate::web_gpu::gpu_adapter::GPUAdapter;
use crate::web_gpu::wgsl_language_features::WGSLLanguageFeatures;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPUPowerPreference {
    LowPower,
    HighPerformance,
}

#[derive(Default)]
pub(crate) struct GPURequestAdapterOptions {
    pub(crate) power_preference: Option<GPUPowerPreference>,
    pub(crate) force_fallback_adapter: bool,
}

pub(crate) struct GPUObjectDescriptorBase {
    pub(crate) label: UsvString,
}

string_enum!(GPUTextureFormat, "gputextureformat" {
    R8unorm => "r8unorm",
    R8snorm => "r8snorm",
    R8uint => "r8uint",
    R8sint => "r8sint",
    R16unorm => "r16unorm",
    R16snorm => "r16snorm",
    R16uint => "r16uint",
    R16sint => "r16sint",
    R16float => "r16float",
    Rg8unorm => "rg8unorm",
    Rg8snorm => "rg8snorm",
    Rg8uint => "rg8uint",
    Rg8sint => "rg8sint",
    R32uint => "r32uint",
    R32sint => "r32sint",
    R32float => "r32float",
    Rg16unorm => "rg16unorm",
    Rg16snorm => "rg16snorm",
    Rg16uint => "rg16uint",
    Rg16sint => "rg16sint",
    Rg16float => "rg16float",
    Rgba8unorm => "rgba8unorm",
    Rgba8unormSrgb => "rgba8unorm-srgb",
    Rgba8snorm => "rgba8snorm",
    Rgba8uint => "rgba8uint",
    Rgba8sint => "rgba8sint",
    Bgra8unorm => "bgra8unorm",
    Bgra8unormSrgb => "bgra8unorm-srgb",
    Rgb9e5ufloat => "rgb9e5ufloat",
    Rgb10a2uint => "rgb10a2uint",
    Rgb10a2unorm => "rgb10a2unorm",
    Rg11b10ufloat => "rg11b10ufloat",
    Rg32uint => "rg32uint",
    Rg32sint => "rg32sint",
    Rg32float => "rg32float",
    Rgba16unorm => "rgba16unorm",
    Rgba16snorm => "rgba16snorm",
    Rgba16uint => "rgba16uint",
    Rgba16sint => "rgba16sint",
    Rgba16float => "rgba16float",
    Rgba32uint => "rgba32uint",
    Rgba32sint => "rgba32sint",
    Rgba32float => "rgba32float",
    Stencil8 => "stencil8",
    Depth16unorm => "depth16unorm",
    Depth24plus => "depth24plus",
    Depth24plusStencil8 => "depth24plus-stencil8",
    Depth32float => "depth32float",
    Depth32floatStencil8 => "depth32float-stencil8",
    Bc1RgbaUnorm => "bc1-rgba-unorm",
    Bc1RgbaUnormSrgb => "bc1-rgba-unorm-srgb",
    Bc2RgbaUnorm => "bc2-rgba-unorm",
    Bc2RgbaUnormSrgb => "bc2-rgba-unorm-srgb",
    Bc3RgbaUnorm => "bc3-rgba-unorm",
    Bc3RgbaUnormSrgb => "bc3-rgba-unorm-srgb",
    Bc4RUnorm => "bc4-r-unorm",
    Bc4RSnorm => "bc4-r-snorm",
    Bc5RgUnorm => "bc5-rg-unorm",
    Bc5RgSnorm => "bc5-rg-snorm",
    Bc6hRgbUfloat => "bc6h-rgb-ufloat",
    Bc6hRgbFloat => "bc6h-rgb-float",
    Bc7RgbaUnorm => "bc7-rgba-unorm",
    Bc7RgbaUnormSrgb => "bc7-rgba-unorm-srgb",
    Etc2Rgb8unorm => "etc2-rgb8unorm",
    Etc2Rgb8unormSrgb => "etc2-rgb8unorm-srgb",
    Etc2Rgb8a1unorm => "etc2-rgb8a1unorm",
    Etc2Rgb8a1unormSrgb => "etc2-rgb8a1unorm-srgb",
    Etc2Rgba8unorm => "etc2-rgba8unorm",
    Etc2Rgba8unormSrgb => "etc2-rgba8unorm-srgb",
    EacR11unorm => "eac-r11unorm",
    EacR11snorm => "eac-r11snorm",
    EacRg11unorm => "eac-rg11unorm",
    EacRg11snorm => "eac-rg11snorm",
    Astc4x4Unorm => "astc-4x4-unorm",
    Astc4x4UnormSrgb => "astc-4x4-unorm-srgb",
    Astc5x4Unorm => "astc-5x4-unorm",
    Astc5x4UnormSrgb => "astc-5x4-unorm-srgb",
    Astc5x5Unorm => "astc-5x5-unorm",
    Astc5x5UnormSrgb => "astc-5x5-unorm-srgb",
    Astc6x5Unorm => "astc-6x5-unorm",
    Astc6x5UnormSrgb => "astc-6x5-unorm-srgb",
    Astc6x6Unorm => "astc-6x6-unorm",
    Astc6x6UnormSrgb => "astc-6x6-unorm-srgb",
    Astc8x5Unorm => "astc-8x5-unorm",
    Astc8x5UnormSrgb => "astc-8x5-unorm-srgb",
    Astc8x6Unorm => "astc-8x6-unorm",
    Astc8x6UnormSrgb => "astc-8x6-unorm-srgb",
    Astc8x8Unorm => "astc-8x8-unorm",
    Astc8x8UnormSrgb => "astc-8x8-unorm-srgb",
    Astc10x5Unorm => "astc-10x5-unorm",
    Astc10x5UnormSrgb => "astc-10x5-unorm-srgb",
    Astc10x6Unorm => "astc-10x6-unorm",
    Astc10x6UnormSrgb => "astc-10x6-unorm-srgb",
    Astc10x8Unorm => "astc-10x8-unorm",
    Astc10x8UnormSrgb => "astc-10x8-unorm-srgb",
    Astc10x10Unorm => "astc-10x10-unorm",
    Astc10x10UnormSrgb => "astc-10x10-unorm-srgb",
    Astc12x10Unorm => "astc-12x10-unorm",
    Astc12x10UnormSrgb => "astc-12x10-unorm-srgb",
    Astc12x12Unorm => "astc-12x12-unorm",
    Astc12x12UnormSrgb => "astc-12x12-unorm-srgb",
});

#[expect(
    clippy::upper_case_acronyms,
    reason = "the bindings name the struct after the GPU interface"
)]
#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPU {
    realm: Gc<Realm>,
    wgsl_language_features: Ref<WGSLLanguageFeatures>,
}

impl Wrappable for GPU {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPU {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>) -> Ref<Self> {
        let wgsl_language_features = WGSLLanguageFeatures::create(vm, realm);
        allocate(vm, |_| Self {
            realm,
            wgsl_language_features,
        })
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpu-requestadapter
    pub(crate) fn request_adapter(
        &self,
        cx: &mut CallContext,
        promise: WebIdlPromise,
        options: GPURequestAdapterOptions,
    ) -> WebIdlResult<()> {
        let power_preference = match options.power_preference {
            None => 0,
            Some(GPUPowerPreference::LowPower) => 1,
            Some(GPUPowerPreference::HighPerformance) => 2,
        };
        let deadline = milliseconds_since_epoch() + COMPOSITOR_CONNECTION_TIMEOUT_MS;
        request_adapter_once_connected(
            cx.vm(),
            (
                self.realm,
                promise,
                power_preference,
                options.force_fallback_adapter,
                deadline,
            ),
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpu-wgsllanguagefeatures
    pub(crate) fn wgsl_language_features(&self, _: &mut CallContext) -> WebIdlResult<Ref<WGSLLanguageFeatures>> {
        Ok(self.wgsl_language_features)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpu-getpreferredcanvasformat
    pub(crate) fn get_preferred_canvas_format(&self, _: &mut CallContext) -> WebIdlResult<GPUTextureFormat> {
        // NB: The canvas reaches the screen as a bitmap the compositor copies, which takes either order of channels.
        Ok(GPUTextureFormat::Bgra8unorm)
    }
}

/// How long a request for an adapter waits for the compositor of a new WebContent.
const COMPOSITOR_CONNECTION_TIMEOUT_MS: u64 = 5000;

fn milliseconds_since_epoch() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
}

/// The realm, the promise, the power preference (0 for none, 1 for low power, 2 for high performance), whether to
/// force a fallback adapter, and when to stop waiting for the compositor.
type AdapterRequest = (Gc<Realm>, WebIdlPromise, u8, bool, u64);

// NB: A new WebContent may not be connected to its compositor yet, so until it is, the request tries again in a task
//     of its own.
fn request_adapter_once_connected(vm: &Vm, request: AdapterRequest) {
    let (realm, promise, power_preference, force_fallback_adapter, deadline) = request;
    if !client::has_compositor(realm) && milliseconds_since_epoch() < deadline {
        queue_a_global_task(
            vm,
            realm.global_object(),
            TaskSource::Unspecified,
            request,
            request_adapter_once_connected,
        );
        return;
    }

    let id = client::allocate_object_id();
    let body = Request::RequestAdapter {
        id,
        power_preference: match power_preference {
            1 => PowerPreference::LowPower,
            2 => PowerPreference::HighPerformance,
            _ => PowerPreference::Default,
        },
        force_fallback_adapter,
    };
    fn steps(vm: &Vm, request: &client::PendingRequest, reply: Option<Reply>) {
        let value = match reply {
            Some(Reply::Adapter { info: Some(info) }) => {
                let adapter = GPUAdapter::create(vm, request.realm, request.object, info);
                Value::from_object(wrap(&CallContext::new(vm, request.realm), adapter))
            }
            _ => Value::NULL,
        };
        resolve_promise(vm, request.promise, value);
    }
    client::request(vm, realm, promise, id, body, steps);
}
