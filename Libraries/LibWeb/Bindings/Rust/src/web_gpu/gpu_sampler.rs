/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpusampler

use libjs_rust::api::{Gc, Realm, Trace, Vm};
use webgpu_wire::{Command, ObjectId, SamplerDescriptor};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::web_gpu::gpu::GPUObjectDescriptorBase;
use crate::web_gpu::gpu_device::GPUCompareFunction;
use crate::web_gpu::{Label, client};

string_enum!(GPUAddressMode, "gpuaddressmode" {
    ClampToEdge => "clamp-to-edge",
    Repeat => "repeat",
    MirrorRepeat => "mirror-repeat",
});

string_enum!(GPUFilterMode, "gpufiltermode" {
    Nearest => "nearest",
    Linear => "linear",
});

string_enum!(GPUMipmapFilterMode, "gpumipmapfiltermode" {
    Nearest => "nearest",
    Linear => "linear",
});

pub(crate) struct GPUSamplerDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) address_mode_u: GPUAddressMode,
    pub(crate) address_mode_v: GPUAddressMode,
    pub(crate) address_mode_w: GPUAddressMode,
    pub(crate) mag_filter: GPUFilterMode,
    pub(crate) min_filter: GPUFilterMode,
    pub(crate) mipmap_filter: GPUMipmapFilterMode,
    pub(crate) lod_min_clamp: f32,
    pub(crate) lod_max_clamp: f32,
    pub(crate) compare: Option<GPUCompareFunction>,
    pub(crate) max_anisotropy: u16,
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUSampler {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
}

impl Wrappable for GPUSampler {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUSampler {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, device: ObjectId, descriptor: &GPUSamplerDescriptor) -> Ref<Self> {
        let id = client::allocate_object_id();
        client::send(
            realm,
            &Command::CreateSampler {
                device,
                id,
                descriptor: SamplerDescriptor {
                    address_mode_u: descriptor.address_mode_u.name().to_owned(),
                    address_mode_v: descriptor.address_mode_v.name().to_owned(),
                    address_mode_w: descriptor.address_mode_w.name().to_owned(),
                    mag_filter: descriptor.mag_filter.name().to_owned(),
                    min_filter: descriptor.min_filter.name().to_owned(),
                    mipmap_filter: descriptor.mipmap_filter.name().to_owned(),
                    lod_min_clamp: descriptor.lod_min_clamp,
                    lod_max_clamp: descriptor.lod_max_clamp,
                    compare: descriptor.compare.map(|compare| compare.name().to_owned()),
                    max_anisotropy: u32::from(descriptor.max_anisotropy),
                },
            },
        );
        allocate(vm, |_| Self {
            realm,
            id,
            label: Label::default(),
        })
    }
}

label_attribute!(GPUSampler);
