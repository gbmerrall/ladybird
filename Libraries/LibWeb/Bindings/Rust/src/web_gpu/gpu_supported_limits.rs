/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpusupportedlimits

use libjs_rust::api::{Gc, Realm, Trace, Vm};

use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::{CallContext, WebIdlResult};

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUSupportedLimits {
    realm: Gc<Realm>,
    /// The limits by their names in the spec, as the compositor reports them.
    #[gc(untraced)]
    values: Vec<(String, u64)>,
}

impl Wrappable for GPUSupportedLimits {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

macro_rules! limit_getters {
    ($($getter:ident: $type:ty = $name:literal,)*) => {
        impl GPUSupportedLimits {
            $(
                pub(crate) fn $getter(&self, _: &mut CallContext) -> WebIdlResult<$type> {
                    Ok(self.value($name).try_into().unwrap_or(<$type>::MAX))
                }
            )*
        }
    };
}

limit_getters! {
    max_texture_dimension1d: u32 = "maxTextureDimension1D",
    max_texture_dimension2d: u32 = "maxTextureDimension2D",
    max_texture_dimension3d: u32 = "maxTextureDimension3D",
    max_texture_array_layers: u32 = "maxTextureArrayLayers",
    max_bind_groups: u32 = "maxBindGroups",
    max_bind_groups_plus_vertex_buffers: u32 = "maxBindGroupsPlusVertexBuffers",
    max_bindings_per_bind_group: u32 = "maxBindingsPerBindGroup",
    max_dynamic_uniform_buffers_per_pipeline_layout: u32 = "maxDynamicUniformBuffersPerPipelineLayout",
    max_dynamic_storage_buffers_per_pipeline_layout: u32 = "maxDynamicStorageBuffersPerPipelineLayout",
    max_sampled_textures_per_shader_stage: u32 = "maxSampledTexturesPerShaderStage",
    max_samplers_per_shader_stage: u32 = "maxSamplersPerShaderStage",
    max_storage_buffers_per_shader_stage: u32 = "maxStorageBuffersPerShaderStage",
    max_storage_textures_per_shader_stage: u32 = "maxStorageTexturesPerShaderStage",
    max_storage_buffers_in_vertex_stage: u32 = "maxStorageBuffersInVertexStage",
    max_storage_buffers_in_fragment_stage: u32 = "maxStorageBuffersInFragmentStage",
    max_storage_textures_in_vertex_stage: u32 = "maxStorageTexturesInVertexStage",
    max_storage_textures_in_fragment_stage: u32 = "maxStorageTexturesInFragmentStage",
    max_uniform_buffers_per_shader_stage: u32 = "maxUniformBuffersPerShaderStage",
    max_uniform_buffer_binding_size: u64 = "maxUniformBufferBindingSize",
    max_storage_buffer_binding_size: u64 = "maxStorageBufferBindingSize",
    min_uniform_buffer_offset_alignment: u32 = "minUniformBufferOffsetAlignment",
    min_storage_buffer_offset_alignment: u32 = "minStorageBufferOffsetAlignment",
    max_vertex_buffers: u32 = "maxVertexBuffers",
    max_buffer_size: u64 = "maxBufferSize",
    max_vertex_attributes: u32 = "maxVertexAttributes",
    max_vertex_buffer_array_stride: u32 = "maxVertexBufferArrayStride",
    max_inter_stage_shader_variables: u32 = "maxInterStageShaderVariables",
    max_color_attachments: u32 = "maxColorAttachments",
    max_color_attachment_bytes_per_sample: u32 = "maxColorAttachmentBytesPerSample",
    max_compute_workgroup_storage_size: u32 = "maxComputeWorkgroupStorageSize",
    max_compute_invocations_per_workgroup: u32 = "maxComputeInvocationsPerWorkgroup",
    max_compute_workgroup_size_x: u32 = "maxComputeWorkgroupSizeX",
    max_compute_workgroup_size_y: u32 = "maxComputeWorkgroupSizeY",
    max_compute_workgroup_size_z: u32 = "maxComputeWorkgroupSizeZ",
    max_compute_workgroups_per_dimension: u32 = "maxComputeWorkgroupsPerDimension",
}

impl GPUSupportedLimits {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, values: Vec<(String, u64)>) -> Ref<Self> {
        allocate(vm, |_| Self { realm, values })
    }

    pub(crate) fn value(&self, name: &str) -> u64 {
        self.values
            .iter()
            .find(|(limit, _)| limit == name)
            .map_or(0, |(_, value)| *value)
    }
}
