/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpubindgroup

use libjs_rust::api::{Gc, MarkedVec, Realm, Trace, Vm};
use webgpu_wire::{BindGroupEntry, BindingResource, Command, ObjectId};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::web_gpu::gpu::GPUObjectDescriptorBase;
use crate::web_gpu::gpu_bind_group_layout::GPUBindGroupLayout;
use crate::web_gpu::gpu_buffer::GPUBuffer;
use crate::web_gpu::gpu_sampler::GPUSampler;
use crate::web_gpu::gpu_texture::GPUTexture;
use crate::web_gpu::gpu_texture_view::GPUTextureView;
use crate::web_gpu::{Label, client};

#[derive(Clone, Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUBufferBinding {
    pub(crate) buffer: Ref<GPUBuffer>,
    #[gc(untraced)]
    pub(crate) offset: u64,
    #[gc(untraced)]
    pub(crate) size: Option<u64>,
}

/// The union GPUBindingResource of WebGPU/GPUBindGroup.idl.
#[derive(Clone, Trace)]
#[gc(crate = libjs_rust)]
#[allow(clippy::enum_variant_names, reason = "the variants have the members' names")]
pub(crate) enum GPUBindingResource {
    GPUSampler(Ref<GPUSampler>),
    GPUTexture(Ref<GPUTexture>),
    GPUTextureView(Ref<GPUTextureView>),
    GPUBuffer(Ref<GPUBuffer>),
    GPUBufferBinding(GPUBufferBinding),
}

#[derive(Clone, Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUBindGroupEntry {
    #[gc(untraced)]
    pub(crate) binding: u32,
    pub(crate) resource: GPUBindingResource,
}

pub(crate) struct GPUBindGroupDescriptor<'vm> {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) layout: Ref<GPUBindGroupLayout>,
    pub(crate) entries: MarkedVec<'vm, GPUBindGroupEntry>,
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUBindGroup {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
}

impl Wrappable for GPUBindGroup {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUBindGroup {
    pub(crate) fn create(
        vm: &Vm,
        realm: Gc<Realm>,
        device: ObjectId,
        descriptor: &GPUBindGroupDescriptor<'_>,
    ) -> Ref<Self> {
        let entries = descriptor
            .entries
            .values()
            .iter()
            .map(|entry| BindGroupEntry {
                binding: entry.binding,
                resource: match &entry.resource {
                    GPUBindingResource::GPUSampler(sampler) => BindingResource::Sampler { sampler: sampler.id },
                    GPUBindingResource::GPUTexture(texture) => BindingResource::TextureView {
                        view: texture.default_view(vm).id,
                    },
                    GPUBindingResource::GPUTextureView(view) => BindingResource::TextureView { view: view.id },
                    GPUBindingResource::GPUBuffer(buffer) => BindingResource::Buffer {
                        buffer: buffer.id,
                        offset: 0,
                        size: None,
                    },
                    GPUBindingResource::GPUBufferBinding(binding) => BindingResource::Buffer {
                        buffer: binding.buffer.id,
                        offset: binding.offset,
                        size: binding.size,
                    },
                },
            })
            .collect();
        let id = client::allocate_object_id();
        client::send(
            realm,
            &Command::CreateBindGroup {
                device,
                id,
                layout: descriptor.layout.id,
                entries,
            },
        );
        allocate(vm, |_| Self {
            realm,
            id,
            label: Label::default(),
        })
    }
}

label_attribute!(GPUBindGroup);
