/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpubindgrouplayout

use libjs_rust::api::{Gc, Realm, Trace, Vm};
use webgpu_wire::{BindGroupLayoutEntry, Command, LayoutResource, ObjectId};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{Exception, WebIdlResult};
use crate::web_gpu::gpu::{GPUObjectDescriptorBase, GPUTextureFormat};
use crate::web_gpu::gpu_texture::GPUTextureViewDimension;
use crate::web_gpu::{Label, client};

string_enum!(GPUBufferBindingType, "gpubufferbindingtype" {
    Uniform => "uniform",
    Storage => "storage",
    ReadOnlyStorage => "read-only-storage",
});

string_enum!(GPUSamplerBindingType, "gpusamplerbindingtype" {
    Filtering => "filtering",
    NonFiltering => "non-filtering",
    Comparison => "comparison",
});

string_enum!(GPUTextureSampleType, "gputexturesampletype" {
    Float => "float",
    UnfilterableFloat => "unfilterable-float",
    Depth => "depth",
    Sint => "sint",
    Uint => "uint",
});

string_enum!(GPUStorageTextureAccess, "gpustoragetextureaccess" {
    WriteOnly => "write-only",
    ReadOnly => "read-only",
    ReadWrite => "read-write",
});

pub(crate) struct GPUBufferBindingLayout {
    pub(crate) r#type: GPUBufferBindingType,
    pub(crate) has_dynamic_offset: bool,
    pub(crate) min_binding_size: u64,
}

pub(crate) struct GPUSamplerBindingLayout {
    pub(crate) r#type: GPUSamplerBindingType,
}

pub(crate) struct GPUTextureBindingLayout {
    pub(crate) sample_type: GPUTextureSampleType,
    pub(crate) view_dimension: GPUTextureViewDimension,
    pub(crate) multisampled: bool,
}

pub(crate) struct GPUStorageTextureBindingLayout {
    pub(crate) access: GPUStorageTextureAccess,
    pub(crate) format: GPUTextureFormat,
    pub(crate) view_dimension: GPUTextureViewDimension,
}

pub(crate) struct GPUBindGroupLayoutEntry {
    pub(crate) binding: u32,
    pub(crate) visibility: u32,
    pub(crate) buffer: Option<GPUBufferBindingLayout>,
    pub(crate) sampler: Option<GPUSamplerBindingLayout>,
    pub(crate) texture: Option<GPUTextureBindingLayout>,
    pub(crate) storage_texture: Option<GPUStorageTextureBindingLayout>,
}

pub(crate) struct GPUBindGroupLayoutDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) entries: Vec<GPUBindGroupLayoutEntry>,
}

// https://gpuweb.github.io/gpuweb/#dictdef-gpubindgrouplayoutentry
// "Only one [of buffer, sampler, texture, storageTexture] may be defined", or the device reports a validation error.
fn entry_to_wire(entry: &GPUBindGroupLayoutEntry) -> Option<BindGroupLayoutEntry> {
    let mut resources = Vec::new();
    if let Some(buffer) = &entry.buffer {
        resources.push(LayoutResource::Buffer {
            kind: buffer.r#type.name().to_owned(),
            has_dynamic_offset: buffer.has_dynamic_offset,
            min_binding_size: buffer.min_binding_size,
        });
    }
    if let Some(sampler) = &entry.sampler {
        resources.push(LayoutResource::Sampler {
            kind: sampler.r#type.name().to_owned(),
        });
    }
    if let Some(texture) = &entry.texture {
        resources.push(LayoutResource::Texture {
            sample_type: texture.sample_type.name().to_owned(),
            view_dimension: texture.view_dimension.name().to_owned(),
            multisampled: texture.multisampled,
        });
    }
    if let Some(storage) = &entry.storage_texture {
        resources.push(LayoutResource::StorageTexture {
            access: storage.access.name().to_owned(),
            format: storage.format.name().to_owned(),
            view_dimension: storage.view_dimension.name().to_owned(),
        });
    }
    if resources.len() != 1 {
        return None;
    }
    Some(BindGroupLayoutEntry {
        binding: entry.binding,
        visibility: entry.visibility,
        resource: resources.pop()?,
    })
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUBindGroupLayout {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
}

impl Wrappable for GPUBindGroupLayout {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUBindGroupLayout {
    pub(crate) fn create(
        vm: &Vm,
        realm: Gc<Realm>,
        device: ObjectId,
        descriptor: &GPUBindGroupLayoutDescriptor,
    ) -> WebIdlResult<Ref<Self>> {
        let mut entries = Vec::with_capacity(descriptor.entries.len());
        for entry in &descriptor.entries {
            // FIXME: The spec makes this a validation error of the device, and an invalid layout.
            entries.push(entry_to_wire(entry).ok_or_else(|| {
                Exception::type_error("A bind group layout entry must have exactly one kind of resource")
            })?);
        }
        let id = client::allocate_object_id();
        client::send(realm, &Command::CreateBindGroupLayout { device, id, entries });
        Ok(Self::with_id(vm, realm, id))
    }

    /// A layout whose id the caller has named to the compositor already.
    pub(crate) fn with_id(vm: &Vm, realm: Gc<Realm>, id: ObjectId) -> Ref<Self> {
        allocate(vm, |_| Self {
            realm,
            id,
            label: Label::default(),
        })
    }
}

label_attribute!(GPUBindGroupLayout);
