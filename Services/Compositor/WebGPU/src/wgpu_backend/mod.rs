/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! The Backend implemented with wgpu, on Vulkan.

use std::collections::HashMap;
use std::future::Future;
use std::pin::pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};

use webgpu_wire::{
    AdapterInfo, BindGroupEntry, BindGroupLayoutEntry, BindingResource, Capabilities, ColorAttachment,
    ColorTargetState, CompilationMessage, CompilationMessageType, ComputePassOp, ComputePipelineDescriptor,
    DepthStencilAttachment, DepthStencilState, DeviceLostReason, DeviceRequestError, ErrorFilter, Event, GpuError,
    LayoutResource, LoadOp, ObjectId, PowerPreference, PrimitiveTopology, RenderBundleDescriptor, RenderPassOp,
    RenderPipelineDescriptor, SamplerDescriptor, StencilFaceState, StoreOp, TexelCopyBufferLayout, TexelCopyTexture,
    TextureDescriptor, TextureFormat, TextureViewDescriptor, TimestampWrites,
};

use crate::backend::{Backend, BackendError, BackendResult, CanvasTextureDescriptor, Extent, PresentedFrame, Readback};

mod names;

enum Object {
    Adapter(wgpu::Adapter),
    Device(wgpu::Device, wgpu::Queue),
    /// A device that script destroyed, whose id still names it. wgpu's device is gone, so that a page that makes and
    /// destroys many devices does not run out of them (NVIDIA's driver allows about 64 at once).
    DestroyedDevice,
    ShaderModule(wgpu::ShaderModule),
    RenderPipeline(wgpu::RenderPipeline),
    ComputePipeline(wgpu::ComputePipeline),
    QuerySet(wgpu::QuerySet),
    RenderBundle(wgpu::RenderBundle),
    Texture(wgpu::Texture),
    TextureView(wgpu::TextureView),
    Buffer(wgpu::Buffer),
    BindGroupLayout(wgpu::BindGroupLayout),
    PipelineLayout(wgpu::PipelineLayout),
    BindGroup(wgpu::BindGroup),
    Sampler(wgpu::Sampler),
    CommandEncoder(wgpu::CommandEncoder),
    CommandBuffer(wgpu::CommandBuffer),
}

impl Object {
    fn kind(&self) -> &'static str {
        match self {
            Self::Adapter(_) => "adapter",
            Self::Device(..) => "device",
            Self::DestroyedDevice => "destroyed device",
            Self::ShaderModule(_) => "shader module",
            Self::RenderPipeline(_) => "render pipeline",
            Self::ComputePipeline(_) => "compute pipeline",
            Self::QuerySet(_) => "query set",
            Self::RenderBundle(_) => "render bundle",
            Self::Texture(_) => "texture",
            Self::TextureView(_) => "texture view",
            Self::Buffer(_) => "buffer",
            Self::BindGroupLayout(_) => "bind group layout",
            Self::PipelineLayout(_) => "pipeline layout",
            Self::BindGroup(_) => "bind group",
            Self::Sampler(_) => "sampler",
            Self::CommandEncoder(_) => "command encoder",
            Self::CommandBuffer(_) => "command buffer",
        }
    }
}

/// wgpu's native futures resolve without an executor once the work they wait on is done, so polling in a loop
/// is enough. Only adapter and device requests use this.
fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(value) = future.as_mut().poll(&mut cx) {
            return value;
        }
        std::thread::yield_now();
    }
}

fn texture_format(format: TextureFormat) -> wgpu::TextureFormat {
    match format {
        TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
        TextureFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
        TextureFormat::Bgra8UnormSrgb => wgpu::TextureFormat::Bgra8UnormSrgb,
        TextureFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
    }
}

fn wire_texture_format(format: wgpu::TextureFormat) -> Option<TextureFormat> {
    Some(match format {
        wgpu::TextureFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba8UnormSrgb => TextureFormat::Rgba8UnormSrgb,
        wgpu::TextureFormat::Bgra8Unorm => TextureFormat::Bgra8Unorm,
        wgpu::TextureFormat::Bgra8UnormSrgb => TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgba16Float => TextureFormat::Rgba16Float,
        _ => return None,
    })
}

fn topology(topology: PrimitiveTopology) -> wgpu::PrimitiveTopology {
    match topology {
        PrimitiveTopology::PointList => wgpu::PrimitiveTopology::PointList,
        PrimitiveTopology::LineList => wgpu::PrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip => wgpu::PrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList => wgpu::PrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip => wgpu::PrimitiveTopology::TriangleStrip,
    }
}

fn missing(id: ObjectId, kind: &str) -> BackendError {
    BackendError(format!("no {kind} with id {id}"))
}

#[derive(Default)]
pub struct WgpuBackend {
    instance: Option<wgpu::Instance>,
    objects: HashMap<ObjectId, Object>,
    /// The device each texture and buffer was created on, so that the backend can wait for it.
    texture_devices: HashMap<ObjectId, ObjectId>,
    /// Each device's stack of error scopes, innermost last. wgpu pops a scope through the guard that pushed it.
    error_scopes: HashMap<ObjectId, Vec<wgpu::ErrorScopeGuard>>,
    /// The buffers still mapped from their creation, which wgpu writes into directly.
    mapped_at_creation: std::collections::HashSet<ObjectId>,
    /// Uncaptured errors and lost devices. wgpu reports them from callbacks, possibly on other threads.
    events: Arc<Mutex<Vec<Event>>>,
    /// Canvas frames on their way back from the GPU, oldest first.
    readbacks: Vec<PendingFrame>,
    /// The PCI vendor and device ids of each device's adapter.
    device_ids: HashMap<ObjectId, (u32, u32)>,
}

/// A canvas frame that the GPU is still working on.
struct PendingFrame {
    tag: u64,
    device: ObjectId,
    content: PendingContent,
    /// Set by wgpu's callback once the frame is done, or failed.
    done: Arc<Mutex<Option<Result<(), String>>>>,
}

enum PendingContent {
    Readback(PendingReadback),
    /// The compositor's image, which the frame was rendered into.
    SharedImage(u64),
}

impl PendingFrame {
    fn status(&self) -> Option<Result<(), String>> {
        self.done.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn take(self) -> BackendResult<PresentedFrame> {
        match self.content {
            PendingContent::Readback(readback) => readback.take_pixels().map(PresentedFrame::Pixels),
            PendingContent::SharedImage(image) => Ok(PresentedFrame::SharedImage(image)),
        }
    }
}

/// A texture's pixels being copied into a buffer and mapped.
struct PendingReadback {
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    format: TextureFormat,
    padded_row_bytes: u32,
}

impl PendingReadback {
    /// The pixels, as 4 bytes each: half floats are clamped to 8 bits.
    fn take_pixels(self) -> BackendResult<Readback> {
        let row_bytes = self.width * texel_size(self.format);
        let pixels = {
            let mapped = self
                .buffer
                .slice(..)
                .get_mapped_range()
                .map_err(|error| BackendError(format!("readback not mapped: {error}")))?;
            let mut pixels = Vec::with_capacity((row_bytes * self.height) as usize);
            for row in mapped
                .chunks_exact(self.padded_row_bytes as usize)
                .take(self.height as usize)
            {
                let row = &row[..row_bytes as usize];
                if self.format == TextureFormat::Rgba16Float {
                    pixels.extend(row.as_chunks::<2>().0.iter().map(|half| {
                        let value = half_to_f32(u16::from_le_bytes(*half));
                        (value.clamp(0.0, 1.0) * 255.0).round() as u8
                    }));
                } else {
                    pixels.extend_from_slice(row);
                }
            }
            pixels
        };
        self.buffer.unmap();
        let format = match self.format {
            TextureFormat::Rgba16Float => TextureFormat::Rgba8Unorm,
            format => format,
        };
        Ok(Readback {
            width: self.width,
            height: self.height,
            format,
            pixels,
        })
    }
}

/// The limits of https://gpuweb.github.io/gpuweb/#limits, by their names in the spec, and their fields in wgpu.
macro_rules! for_each_limit {
    ($callback:ident) => {
        $callback! {
            "maxTextureDimension1D" => max_texture_dimension_1d,
            "maxTextureDimension2D" => max_texture_dimension_2d,
            "maxTextureDimension3D" => max_texture_dimension_3d,
            "maxTextureArrayLayers" => max_texture_array_layers,
            "maxBindGroups" => max_bind_groups,
            "maxBindGroupsPlusVertexBuffers" => max_bind_groups_plus_vertex_buffers,
            "maxBindingsPerBindGroup" => max_bindings_per_bind_group,
            "maxDynamicUniformBuffersPerPipelineLayout" => max_dynamic_uniform_buffers_per_pipeline_layout,
            "maxDynamicStorageBuffersPerPipelineLayout" => max_dynamic_storage_buffers_per_pipeline_layout,
            "maxSampledTexturesPerShaderStage" => max_sampled_textures_per_shader_stage,
            "maxSamplersPerShaderStage" => max_samplers_per_shader_stage,
            "maxStorageBuffersPerShaderStage" => max_storage_buffers_per_shader_stage,
            "maxStorageTexturesPerShaderStage" => max_storage_textures_per_shader_stage,
            "maxUniformBuffersPerShaderStage" => max_uniform_buffers_per_shader_stage,
            "maxUniformBufferBindingSize" => max_uniform_buffer_binding_size,
            "maxStorageBufferBindingSize" => max_storage_buffer_binding_size,
            "minUniformBufferOffsetAlignment" => min_uniform_buffer_offset_alignment,
            "minStorageBufferOffsetAlignment" => min_storage_buffer_offset_alignment,
            "maxVertexBuffers" => max_vertex_buffers,
            "maxBufferSize" => max_buffer_size,
            "maxVertexAttributes" => max_vertex_attributes,
            "maxVertexBufferArrayStride" => max_vertex_buffer_array_stride,
            "maxInterStageShaderVariables" => max_inter_stage_shader_variables,
            "maxColorAttachments" => max_color_attachments,
            "maxColorAttachmentBytesPerSample" => max_color_attachment_bytes_per_sample,
            "maxComputeWorkgroupStorageSize" => max_compute_workgroup_storage_size,
            "maxComputeInvocationsPerWorkgroup" => max_compute_invocations_per_workgroup,
            "maxComputeWorkgroupSizeX" => max_compute_workgroup_size_x,
            "maxComputeWorkgroupSizeY" => max_compute_workgroup_size_y,
            "maxComputeWorkgroupSizeZ" => max_compute_workgroup_size_z,
            "maxComputeWorkgroupsPerDimension" => max_compute_workgroups_per_dimension,
        }
    };
}

fn limit_values(limits: &wgpu::Limits) -> Vec<(String, u64)> {
    macro_rules! values {
        ($($name:literal => $field:ident,)*) => {
            vec![$(($name.to_owned(), u64::from(limits.$field)),)*]
        };
    }
    let mut values = for_each_limit!(values);
    values.extend(
        PER_STAGE_LIMITS
            .iter()
            .map(|(name, field)| ((*name).to_owned(), u64::from(per_stage_limit(limits, field)))),
    );
    values
}

/// The limits on storage buffers and textures in the vertex and fragment stages, which wgpu does not separate from
/// the limits per shader stage. Their values are those, as the spec's defaults are.
const PER_STAGE_LIMITS: &[(&str, &str)] = &[
    ("maxStorageBuffersInVertexStage", "maxStorageBuffersPerShaderStage"),
    ("maxStorageBuffersInFragmentStage", "maxStorageBuffersPerShaderStage"),
    ("maxStorageTexturesInVertexStage", "maxStorageTexturesPerShaderStage"),
    ("maxStorageTexturesInFragmentStage", "maxStorageTexturesPerShaderStage"),
];

fn per_stage_limit(limits: &wgpu::Limits, per_shader_stage: &str) -> u32 {
    if per_shader_stage == "maxStorageBuffersPerShaderStage" {
        limits.max_storage_buffers_per_shader_stage
    } else {
        limits.max_storage_textures_per_shader_stage
    }
}

/// Sets a limit by its name in the spec, or returns false if there is no such limit.
fn set_limit(limits: &mut wgpu::Limits, name: &str, value: u64) -> bool {
    // NB: wgpu has no separate limit to set; the adapter's check has already compared the value with it.
    if PER_STAGE_LIMITS.iter().any(|(per_stage, _)| *per_stage == name) {
        return true;
    }
    macro_rules! set {
        ($($name:literal => $field:ident,)*) => {
            match name {
                $($name => {
                    limits.$field = value.try_into().unwrap_or(limits.$field);
                    true
                })*
                _ => false,
            }
        };
    }
    for_each_limit!(set)
}

/// Whether `value` is better than the adapter's: higher for a maximum, lower for a minimum alignment.
fn is_better_than(name: &str, value: u64, adapter: u64) -> bool {
    if name.starts_with("min") {
        value < adapter
    } else {
        value > adapter
    }
}

/// The features of https://gpuweb.github.io/gpuweb/#feature-index that wgpu can have, by their names in the spec.
/// "core-features-and-limits" is every adapter's, since Ladybird offers no compatibility mode.
const FEATURE_NAMES: &[&str] = &[
    "depth-clip-control",
    "depth32float-stencil8",
    "texture-compression-bc",
    "texture-compression-bc-sliced-3d",
    "texture-compression-etc2",
    "texture-compression-astc",
    "texture-compression-astc-sliced-3d",
    "timestamp-query",
    "indirect-first-instance",
    "shader-f16",
    "rg11b10ufloat-renderable",
    "bgra8unorm-storage",
    "float32-filterable",
    "float32-blendable",
    "clip-distances",
    "dual-source-blending",
];

fn feature_names(features: wgpu::Features) -> Vec<String> {
    let mut names = vec!["core-features-and-limits".to_owned()];
    names.extend(
        FEATURE_NAMES
            .iter()
            .filter(|name| {
                name.parse::<wgpu::Features>()
                    .is_ok_and(|feature| features.contains(feature))
            })
            .map(|name| (*name).to_owned()),
    );
    names
}

fn capabilities(features: wgpu::Features, limits: &wgpu::Limits) -> Capabilities {
    Capabilities {
        features: feature_names(features),
        limits: limit_values(limits),
    }
}

fn blend_component(component: &webgpu_wire::BlendComponent) -> BackendResult<wgpu::BlendComponent> {
    Ok(wgpu::BlendComponent {
        src_factor: names::blend_factor(&component.src_factor)?,
        dst_factor: names::blend_factor(&component.dst_factor)?,
        operation: names::blend_operation(&component.operation)?,
    })
}

fn color_target_state(target: &ColorTargetState) -> BackendResult<wgpu::ColorTargetState> {
    Ok(wgpu::ColorTargetState {
        format: names::texture_format(&target.format)?,
        blend: match &target.blend {
            Some(blend) => Some(wgpu::BlendState {
                color: blend_component(&blend.color)?,
                alpha: blend_component(&blend.alpha)?,
            }),
            None => None,
        },
        write_mask: wgpu::ColorWrites::from_bits(target.write_mask)
            .ok_or_else(|| BackendError(format!("{:#x} is not a valid color write mask", target.write_mask)))?,
    })
}

fn stencil_face_state(state: &StencilFaceState) -> BackendResult<wgpu::StencilFaceState> {
    Ok(wgpu::StencilFaceState {
        compare: names::compare_function(&state.compare)?,
        fail_op: names::stencil_operation(&state.fail_op)?,
        depth_fail_op: names::stencil_operation(&state.depth_fail_op)?,
        pass_op: names::stencil_operation(&state.pass_op)?,
    })
}

fn depth_stencil_state(state: &DepthStencilState) -> BackendResult<wgpu::DepthStencilState> {
    Ok(wgpu::DepthStencilState {
        format: names::texture_format(&state.format)?,
        depth_write_enabled: Some(state.depth_write_enabled),
        depth_compare: match &state.depth_compare {
            Some(compare) => Some(names::compare_function(compare)?),
            None => None,
        },
        stencil: wgpu::StencilState {
            front: stencil_face_state(&state.stencil_front)?,
            back: stencil_face_state(&state.stencil_back)?,
            read_mask: state.stencil_read_mask,
            write_mask: state.stencil_write_mask,
        },
        bias: wgpu::DepthBiasState {
            constant: state.depth_bias,
            slope_scale: state.depth_bias_slope_scale,
            clamp: state.depth_bias_clamp,
        },
    })
}

/// The operations on one aspect of a depth-stencil attachment, or None when the pass does not touch it.
fn aspect_operations<T>(
    load_op: Option<&String>,
    store_op: Option<&String>,
    clear_value: T,
) -> BackendResult<Option<wgpu::Operations<T>>> {
    let (Some(load_op), Some(store_op)) = (load_op, store_op) else {
        return Ok(None);
    };
    Ok(Some(wgpu::Operations {
        load: if names::load_op(load_op)? {
            wgpu::LoadOp::Clear(clear_value)
        } else {
            wgpu::LoadOp::Load
        },
        store: names::store_op(store_op)?,
    }))
}

fn gpu_error(error: wgpu::Error) -> GpuError {
    match error {
        wgpu::Error::OutOfMemory { .. } => GpuError::OutOfMemory {
            message: "out of memory".to_owned(),
        },
        wgpu::Error::Validation { description, .. } => GpuError::Validation { message: description },
        wgpu::Error::Internal { description, .. } => GpuError::Internal { message: description },
    }
}

fn push_event(events: &Mutex<Vec<Event>>, event: Event) {
    events.lock().unwrap_or_else(PoisonError::into_inner).push(event);
}

impl WgpuBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn instance(&mut self) -> &wgpu::Instance {
        self.instance.get_or_insert_with(|| {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = if cfg!(target_os = "macos") {
                wgpu::Backends::METAL
            } else {
                wgpu::Backends::VULKAN
            };
            wgpu::Instance::new(descriptor)
        })
    }

    fn insert(&mut self, id: ObjectId, object: Object) -> BackendResult {
        if self.objects.contains_key(&id) {
            return Err(BackendError(format!("id {id} is already in use")));
        }
        self.objects.insert(id, object);
        Ok(())
    }

    fn device(&self, id: ObjectId) -> BackendResult<(&wgpu::Device, &wgpu::Queue)> {
        match self.objects.get(&id) {
            Some(Object::Device(device, queue)) => Ok((device, queue)),
            _ => Err(missing(id, "device")),
        }
    }

    fn shader_module(&self, id: ObjectId) -> BackendResult<&wgpu::ShaderModule> {
        match self.objects.get(&id) {
            Some(Object::ShaderModule(module)) => Ok(module),
            _ => Err(missing(id, "shader module")),
        }
    }

    fn texture_view(&self, id: ObjectId) -> BackendResult<&wgpu::TextureView> {
        match self.objects.get(&id) {
            Some(Object::TextureView(view)) => Ok(view),
            _ => Err(missing(id, "texture view")),
        }
    }

    fn bind_group(&self, id: Option<ObjectId>) -> BackendResult<Option<&wgpu::BindGroup>> {
        match id {
            Some(id) => match self.objects.get(&id) {
                Some(Object::BindGroup(group)) => Ok(Some(group)),
                _ => Err(missing(id, "bind group")),
            },
            None => Ok(None),
        }
    }

    fn pipeline_layout(&self, id: Option<ObjectId>) -> BackendResult<Option<&wgpu::PipelineLayout>> {
        match id {
            Some(id) => match self.objects.get(&id) {
                Some(Object::PipelineLayout(layout)) => Ok(Some(layout)),
                _ => Err(missing(id, "pipeline layout")),
            },
            None => Ok(None),
        }
    }

    fn query_set(&self, id: ObjectId) -> BackendResult<&wgpu::QuerySet> {
        match self.objects.get(&id) {
            Some(Object::QuerySet(query_set)) => Ok(query_set),
            _ => Err(missing(id, "query set")),
        }
    }

    /// The query set of the timestamp writes, and the writes.
    fn timestamp_writes<'a>(
        &self,
        writes: Option<&'a TimestampWrites>,
    ) -> BackendResult<Option<(&wgpu::QuerySet, &'a TimestampWrites)>> {
        match writes {
            Some(writes) => Ok(Some((self.query_set(writes.query_set)?, writes))),
            None => Ok(None),
        }
    }

    fn record_compute_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<&TimestampWrites>,
        ops: &[ComputePassOp],
    ) -> BackendResult {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: self.timestamp_writes(timestamp_writes)?.map(|(query_set, writes)| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: writes.beginning_of_pass_write_index,
                    end_of_pass_write_index: writes.end_of_pass_write_index,
                }
            }),
        });
        for op in ops {
            match op {
                ComputePassOp::SetPipeline { pipeline } => {
                    let Some(Object::ComputePipeline(pipeline)) = self.objects.get(pipeline) else {
                        return Err(missing(*pipeline, "compute pipeline"));
                    };
                    pass.set_pipeline(pipeline);
                }
                ComputePassOp::SetBindGroup {
                    index,
                    group,
                    dynamic_offsets,
                } => pass.set_bind_group(*index, self.bind_group(*group)?, dynamic_offsets),
                ComputePassOp::DispatchWorkgroups { x, y, z } => pass.dispatch_workgroups(*x, *y, *z),
                ComputePassOp::DispatchWorkgroupsIndirect { buffer, offset } => {
                    pass.dispatch_workgroups_indirect(self.buffer(*buffer)?, *offset);
                }
            }
        }
        Ok(())
    }

    fn record_render_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_attachments: &[Option<ColorAttachment>],
        depth_stencil: Option<&DepthStencilAttachment>,
        occlusion_query_set: Option<ObjectId>,
        timestamp_writes: Option<&TimestampWrites>,
        ops: &[RenderPassOp],
    ) -> BackendResult {
        let mut attachments = Vec::with_capacity(color_attachments.len());
        for attachment in color_attachments {
            let Some(attachment) = attachment else {
                attachments.push(None);
                continue;
            };
            let load = match attachment.load {
                LoadOp::Clear([r, g, b, a]) => wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                LoadOp::Load => wgpu::LoadOp::Load,
            };
            let store = match attachment.store {
                StoreOp::Store => wgpu::StoreOp::Store,
                StoreOp::Discard => wgpu::StoreOp::Discard,
            };
            attachments.push(Some(wgpu::RenderPassColorAttachment {
                view: self.texture_view(attachment.view)?,
                depth_slice: attachment.depth_slice,
                resolve_target: match attachment.resolve_target {
                    Some(target) => Some(self.texture_view(target)?),
                    None => None,
                },
                ops: wgpu::Operations { load, store },
            }));
        }
        let depth_stencil_attachment = match depth_stencil {
            Some(attachment) => Some(wgpu::RenderPassDepthStencilAttachment {
                view: self.texture_view(attachment.view)?,
                depth_ops: aspect_operations(
                    attachment.depth_load_op.as_ref(),
                    attachment.depth_store_op.as_ref(),
                    attachment.depth_clear_value,
                )?,
                stencil_ops: aspect_operations(
                    attachment.stencil_load_op.as_ref(),
                    attachment.stencil_store_op.as_ref(),
                    attachment.stencil_clear_value,
                )?,
            }),
            None => None,
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &attachments,
            depth_stencil_attachment,
            timestamp_writes: self.timestamp_writes(timestamp_writes)?.map(|(query_set, writes)| {
                wgpu::RenderPassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: writes.beginning_of_pass_write_index,
                    end_of_pass_write_index: writes.end_of_pass_write_index,
                }
            }),
            occlusion_query_set: match occlusion_query_set {
                Some(query_set) => Some(self.query_set(query_set)?),
                None => None,
            },
            multiview_mask: None,
        });
        for op in ops {
            if self.record_draw_op(&mut pass, op)? {
                continue;
            }
            match op {
                RenderPassOp::SetViewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                } => pass.set_viewport(*x, *y, *width, *height, *min_depth, *max_depth),
                RenderPassOp::SetScissorRect { x, y, width, height } => {
                    pass.set_scissor_rect(*x, *y, *width, *height);
                }
                RenderPassOp::SetBlendConstant { color: [r, g, b, a] } => pass.set_blend_constant(wgpu::Color {
                    r: *r,
                    g: *g,
                    b: *b,
                    a: *a,
                }),
                RenderPassOp::SetStencilReference { reference } => pass.set_stencil_reference(*reference),
                RenderPassOp::BeginOcclusionQuery { index } => pass.begin_occlusion_query(*index),
                RenderPassOp::EndOcclusionQuery => pass.end_occlusion_query(),
                RenderPassOp::ExecuteBundles { bundles } => {
                    let mut render_bundles = Vec::with_capacity(bundles.len());
                    for bundle in bundles {
                        let Some(Object::RenderBundle(bundle)) = self.objects.get(bundle) else {
                            return Err(missing(*bundle, "render bundle"));
                        };
                        render_bundles.push(bundle);
                    }
                    pass.execute_bundles(render_bundles);
                }
                _ => unreachable!("record_draw_op records the other operations"),
            }
        }
        Ok(())
    }

    /// Records an operation that a render pass and a render bundle both have, and returns whether it was one.
    fn record_draw_op<'a, E: wgpu::util::RenderEncoder<'a>>(
        &'a self,
        encoder: &mut E,
        op: &RenderPassOp,
    ) -> BackendResult<bool> {
        let slice = |buffer: &'a wgpu::Buffer, offset: u64, size: Option<u64>| match size {
            Some(size) => buffer.slice(offset..offset.saturating_add(size)),
            None => buffer.slice(offset..),
        };
        match op {
            RenderPassOp::SetPipeline { pipeline } => {
                let Some(Object::RenderPipeline(pipeline)) = self.objects.get(pipeline) else {
                    return Err(missing(*pipeline, "render pipeline"));
                };
                encoder.set_pipeline(pipeline);
            }
            RenderPassOp::Draw {
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            } => encoder.draw(
                *first_vertex..first_vertex.saturating_add(*vertex_count),
                *first_instance..first_instance.saturating_add(*instance_count),
            ),
            RenderPassOp::DrawIndexed {
                index_count,
                instance_count,
                first_index,
                base_vertex,
                first_instance,
            } => encoder.draw_indexed(
                *first_index..first_index.saturating_add(*index_count),
                *base_vertex,
                *first_instance..first_instance.saturating_add(*instance_count),
            ),
            RenderPassOp::SetBindGroup {
                index,
                group,
                dynamic_offsets,
            } => encoder.set_bind_group(*index, self.bind_group(*group)?, dynamic_offsets),
            RenderPassOp::SetVertexBuffer {
                slot,
                buffer,
                offset,
                size,
            } => {
                let buffer = match buffer {
                    Some(buffer) => Some(slice(self.buffer(*buffer)?, *offset, *size)),
                    None => None,
                };
                encoder.set_vertex_buffer(*slot, buffer);
            }
            RenderPassOp::SetIndexBuffer {
                buffer,
                format,
                offset,
                size,
            } => encoder.set_index_buffer(
                slice(self.buffer(*buffer)?, *offset, *size),
                names::index_format(format)?,
            ),
            RenderPassOp::DrawIndirect { buffer, offset } => encoder.draw_indirect(self.buffer(*buffer)?, *offset),
            RenderPassOp::DrawIndexedIndirect { buffer, offset } => {
                encoder.draw_indexed_indirect(self.buffer(*buffer)?, *offset);
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn buffer(&self, id: ObjectId) -> BackendResult<&wgpu::Buffer> {
        match self.objects.get(&id) {
            Some(Object::Buffer(buffer)) => Ok(buffer),
            _ => Err(missing(id, "buffer")),
        }
    }

    /// Drops a device's error scopes, innermost first: wgpu panics when a scope is popped before the ones inside it,
    /// which would take the compositor down.
    fn drop_error_scopes(&mut self, device: ObjectId) {
        if let Some(mut scopes) = self.error_scopes.remove(&device) {
            while let Some(scope) = scopes.pop() {
                drop(scope);
            }
        }
    }

    fn texture(&self, id: ObjectId) -> BackendResult<&wgpu::Texture> {
        match self.objects.get(&id) {
            Some(Object::Texture(texture)) => Ok(texture),
            _ => Err(missing(id, "texture")),
        }
    }

    fn texel_copy_texture(&self, copy: &TexelCopyTexture) -> BackendResult<wgpu::TexelCopyTextureInfo<'_>> {
        let (x, y, z) = copy.origin;
        Ok(wgpu::TexelCopyTextureInfo {
            texture: self.texture(copy.texture)?,
            mip_level: copy.mip_level,
            origin: wgpu::Origin3d { x, y, z },
            aspect: names::texture_aspect(&copy.aspect)?,
        })
    }

    /// Records into the command encoder, which is taken out of the objects meanwhile so that `record` can look up
    /// others.
    fn with_encoder(
        &mut self,
        encoder_id: ObjectId,
        record: impl FnOnce(&Self, &mut wgpu::CommandEncoder) -> BackendResult,
    ) -> BackendResult {
        let Some(Object::CommandEncoder(mut encoder)) = self.objects.remove(&encoder_id) else {
            return Err(missing(encoder_id, "command encoder"));
        };
        let result = record(self, &mut encoder);
        self.objects.insert(encoder_id, Object::CommandEncoder(encoder));
        result
    }
}

/// The constants of a stage as wgpu borrows them.
fn constants(constants: &[(String, f64)]) -> Vec<(&str, f64)> {
    constants.iter().map(|(name, value)| (name.as_str(), *value)).collect()
}

fn compilation_options<'a>(constants: &'a [(&'a str, f64)]) -> wgpu::PipelineCompilationOptions<'a> {
    wgpu::PipelineCompilationOptions {
        constants,
        ..wgpu::PipelineCompilationOptions::default()
    }
}

fn extent((width, height, depth_or_array_layers): Extent) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers,
    }
}

fn texel_copy_buffer_layout(layout: &TexelCopyBufferLayout) -> wgpu::TexelCopyBufferLayout {
    wgpu::TexelCopyBufferLayout {
        offset: layout.offset,
        bytes_per_row: layout.bytes_per_row,
        rows_per_image: layout.rows_per_image,
    }
}

/// The size of a texel of a canvas format.
fn texel_size(format: TextureFormat) -> u32 {
    match format {
        TextureFormat::Rgba16Float => 8,
        _ => 4,
    }
}

/// The value of a half-precision float.
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// The bits of the half-precision float nearest a value from 0 to 1 that is 0 or at least 1/255, as every channel of
/// an 8-bit texel is. Such a value is never subnormal as a half.
fn half_bits(value: f32) -> u16 {
    debug_assert!(value == 0.0 || (1.0 / 256.0..=1.0).contains(&value));
    if value == 0.0 {
        return 0;
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 23) & 0xff) - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    // Rounds to nearest; a carry out of the mantissa correctly moves to the next exponent.
    ((exponent << 10 | mantissa >> 13) + (mantissa >> 12 & 1)) as u16
}

/// RGBA8 texels converted to one of the formats that `copyExternalImageToTexture` may write, and the size of a
/// converted texel. Channels keep their sRGB encoding, as the "srgb" color space of the destination asks.
// https://gpuweb.github.io/gpuweb/#abstract-opdef-validate-gpucopyexternalimagedestinfo
fn convert_rgba8_texels(format: wgpu::TextureFormat, rgba8: &[u8]) -> BackendResult<(Vec<u8>, u32)> {
    use wgpu::TextureFormat as F;
    let texels = rgba8.as_chunks::<4>().0.iter();
    let unit = |channel: u8| f32::from(channel) / 255.0;
    let (bytes, texel_size): (Vec<u8>, u32) = match format {
        F::Rgba8Unorm | F::Rgba8UnormSrgb => (rgba8.to_vec(), 4),
        F::Bgra8Unorm | F::Bgra8UnormSrgb => (texels.flat_map(|t| [t[2], t[1], t[0], t[3]]).collect(), 4),
        F::R8Unorm => (texels.map(|t| t[0]).collect(), 1),
        F::Rg8Unorm => (texels.flat_map(|t| [t[0], t[1]]).collect(), 2),
        F::R16Float | F::Rg16Float | F::Rgba16Float => {
            let channels = match format {
                F::R16Float => 1,
                F::Rg16Float => 2,
                _ => 4,
            };
            let bytes = texels
                .flat_map(|t| t[..channels].iter().flat_map(|&c| half_bits(unit(c)).to_le_bytes()))
                .collect();
            (bytes, 2 * channels as u32)
        }
        F::R32Float | F::Rg32Float | F::Rgba32Float => {
            let channels = match format {
                F::R32Float => 1,
                F::Rg32Float => 2,
                _ => 4,
            };
            let bytes = texels
                .flat_map(|t| t[..channels].iter().flat_map(|&c| unit(c).to_le_bytes()))
                .collect();
            (bytes, 4 * channels as u32)
        }
        F::Rgb10a2Unorm => {
            let ten_bits = |c: u8| (u32::from(c) * 1023 + 127) / 255;
            let bytes = texels
                .flat_map(|t| {
                    let alpha = (u32::from(t[3]) * 3 + 127) / 255;
                    (ten_bits(t[0]) | ten_bits(t[1]) << 10 | ten_bits(t[2]) << 20 | alpha << 30).to_le_bytes()
                })
                .collect();
            (bytes, 4)
        }
        _ => {
            return Err(BackendError(format!(
                "copyExternalImageToTexture cannot write {format:?} textures"
            )));
        }
    };
    Ok((bytes, texel_size))
}

impl Backend for WgpuBackend {
    fn request_adapter(
        &mut self,
        id: ObjectId,
        power_preference: PowerPreference,
        force_fallback_adapter: bool,
        preferred_gpu: Option<(u32, u32)>,
    ) -> Option<AdapterInfo> {
        let options = |power_preference| wgpu::RequestAdapterOptions {
            power_preference,
            force_fallback_adapter,
            compatible_surface: None,
            // Report limits in coarse buckets, as a browser should, so they say less about the machine.
            apply_limit_buckets: true,
        };
        let power_preferences: &[wgpu::PowerPreference] = match power_preference {
            PowerPreference::LowPower => &[wgpu::PowerPreference::LowPower],
            PowerPreference::HighPerformance => &[wgpu::PowerPreference::HighPerformance],
            // NB: wgpu cannot be asked for a particular GPU and still bucket the limits, so the GPU each preference
            //     picks is tried in turn.
            PowerPreference::Default if preferred_gpu.is_some() && !force_fallback_adapter => &[
                wgpu::PowerPreference::HighPerformance,
                wgpu::PowerPreference::LowPower,
                wgpu::PowerPreference::None,
            ],
            PowerPreference::Default => &[wgpu::PowerPreference::None],
        };
        let mut adapter = None;
        for preference in power_preferences {
            let Ok(candidate) = block_on(self.instance().request_adapter(&options(*preference))) else {
                continue;
            };
            let info = candidate.get_info();
            let is_preferred = preferred_gpu == Some((info.vendor, info.device));
            // NB: Without the preferred GPU, the last preference's adapter, which is no preference's, is the page's.
            adapter = Some(candidate);
            if is_preferred {
                break;
            }
        }
        let adapter = adapter?;
        let info = adapter.get_info();
        let adapter_info = AdapterInfo {
            vendor: format!("{:#06x}", info.vendor),
            architecture: String::new(),
            device: format!("{:#06x}", info.device),
            description: info.name.clone(),
            is_fallback_adapter: info.device_type == wgpu::DeviceType::Cpu,
            capabilities: capabilities(adapter.features(), &adapter.limits()),
        };
        self.insert(id, Object::Adapter(adapter)).ok()?;
        Some(adapter_info)
    }

    fn request_device(
        &mut self,
        adapter: ObjectId,
        id: ObjectId,
        required_features: &[String],
        required_limits: &[(String, u64)],
    ) -> Result<Capabilities, DeviceRequestError> {
        let Some(Object::Adapter(adapter)) = self.objects.get(&adapter) else {
            return Err(DeviceRequestError::Operation {
                message: missing(adapter, "adapter").0,
            });
        };

        // https://gpuweb.github.io/gpuweb/#dom-gpuadapter-requestdevice
        let mut features = wgpu::Features::empty();
        for name in required_features {
            if name == "core-features-and-limits" {
                continue;
            }
            let feature = name
                .parse::<wgpu::Features>()
                .ok()
                .filter(|feature| adapter.features().contains(*feature))
                .ok_or_else(|| DeviceRequestError::Type {
                    message: format!("The adapter does not support {name}"),
                })?;
            features |= feature;
        }
        let adapter_limits = limit_values(&adapter.limits());
        let mut limits = wgpu::Limits::default();
        for (name, value) in required_limits {
            let Some((_, adapter_value)) = adapter_limits.iter().find(|(limit, _)| limit == name) else {
                return Err(DeviceRequestError::Operation {
                    message: format!("There is no limit named {name}"),
                });
            };
            if is_better_than(name, *value, *adapter_value) {
                return Err(DeviceRequestError::Operation {
                    message: format!("{name} of {value} is better than the adapter's {adapter_value}"),
                });
            }
            // NB: A requirement worse than the default leaves the default, which is what the device then has.
            if is_better_than(
                name,
                *value,
                limit_values(&limits)
                    .iter()
                    .find(|(limit, _)| limit == name)
                    .map_or(0, |(_, value)| *value),
            ) {
                set_limit(&mut limits, name, *value);
            }
        }

        // NB: Not a WebGPU feature: it lets canvases render into the compositor's images (Phase 8).
        features |= adapter.features() & wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF;
        let adapter_info = adapter.get_info();
        self.device_ids.insert(id, (adapter_info.vendor, adapter_info.device));
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("WebGPU device"),
            required_features: features,
            required_limits: limits,
            ..Default::default()
        };
        let (device, queue) =
            block_on(adapter.request_device(&descriptor)).map_err(|error| DeviceRequestError::Operation {
                message: error.to_string(),
            })?;
        let device_capabilities = capabilities(device.features(), &device.limits());
        // NB: wgpu's default handler panics, which would take the whole compositor down with one bad page.
        let events = Arc::clone(&self.events);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            push_event(
                &events,
                Event::UncapturedError {
                    device: id,
                    error: gpu_error(error),
                },
            );
        }));
        let events = Arc::clone(&self.events);
        device.set_device_lost_callback(move |reason, message| {
            let reason = match reason {
                wgpu::DeviceLostReason::Destroyed => DeviceLostReason::Destroyed,
                wgpu::DeviceLostReason::Unknown => DeviceLostReason::Unknown,
            };
            push_event(
                &events,
                Event::DeviceLost {
                    device: id,
                    reason,
                    message,
                },
            );
        });
        self.insert(id, Object::Device(device, queue))
            .map_err(|error| DeviceRequestError::Operation { message: error.0 })?;
        Ok(device_capabilities)
    }

    fn push_error_scope(&mut self, device_id: ObjectId, filter: ErrorFilter) -> BackendResult {
        let (device, _) = self.device(device_id)?;
        let guard = device.push_error_scope(match filter {
            ErrorFilter::Validation => wgpu::ErrorFilter::Validation,
            ErrorFilter::OutOfMemory => wgpu::ErrorFilter::OutOfMemory,
            ErrorFilter::Internal => wgpu::ErrorFilter::Internal,
        });
        self.error_scopes.entry(device_id).or_default().push(guard);
        Ok(())
    }

    fn pop_error_scope(&mut self, device: ObjectId) -> Result<Option<GpuError>, String> {
        // https://gpuweb.github.io/gpuweb/#dom-gpudevice-poperrorscope: a lost device resolves with null.
        if self.is_destroyed_device(device) {
            return Ok(None);
        }
        let guard = self
            .error_scopes
            .get_mut(&device)
            .and_then(Vec::pop)
            .ok_or_else(|| "There is no error scope to pop".to_owned())?;
        Ok(block_on(guard.pop()).map(gpu_error))
    }

    fn compilation_info(&mut self, module: ObjectId) -> BackendResult<Vec<CompilationMessage>> {
        let Some(Object::ShaderModule(module)) = self.objects.get(&module) else {
            return Err(missing(module, "shader module"));
        };
        let info = block_on(module.get_compilation_info());
        Ok(info
            .messages
            .into_iter()
            .map(|message| {
                let location = message.location.unwrap_or(wgpu::SourceLocation {
                    line_number: 0,
                    line_position: 0,
                    offset: 0,
                    length: 0,
                });
                CompilationMessage {
                    message: message.message,
                    message_type: match message.message_type {
                        wgpu::CompilationMessageType::Error => CompilationMessageType::Error,
                        wgpu::CompilationMessageType::Warning => CompilationMessageType::Warning,
                        wgpu::CompilationMessageType::Info => CompilationMessageType::Info,
                    },
                    line_num: location.line_number,
                    line_pos: location.line_position,
                    offset: location.offset,
                    length: location.length,
                }
            })
            .collect())
    }

    fn take_events(&mut self) -> Vec<Event> {
        // NB: Polling lets wgpu run the callbacks of work that finished, such as that of a destroyed device.
        for object in self.objects.values() {
            if let Object::Device(device, _) = object {
                let _ = device.poll(wgpu::PollType::Poll);
            }
        }
        std::mem::take(&mut *self.events.lock().unwrap_or_else(PoisonError::into_inner))
    }

    fn create_shader_module(&mut self, device: ObjectId, id: ObjectId, code: &str) -> BackendResult {
        let (device, _) = self.device(device)?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        self.insert(id, Object::ShaderModule(module))
    }

    fn create_render_pipeline(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        descriptor: &RenderPipelineDescriptor,
    ) -> BackendResult {
        let (device, _) = self.device(device)?;
        let layout = self.pipeline_layout(descriptor.layout)?;
        let vertex_module = self.shader_module(descriptor.vertex.module)?;

        let mut vertex_attributes = Vec::with_capacity(descriptor.vertex_buffers.len());
        for buffer in &descriptor.vertex_buffers {
            let mut attributes = Vec::new();
            for attribute in buffer.iter().flat_map(|buffer| &buffer.attributes) {
                attributes.push(wgpu::VertexAttribute {
                    format: names::vertex_format(&attribute.format)?,
                    offset: attribute.offset,
                    shader_location: attribute.shader_location,
                });
            }
            vertex_attributes.push(attributes);
        }
        let mut vertex_buffers = Vec::with_capacity(descriptor.vertex_buffers.len());
        for (buffer, attributes) in descriptor.vertex_buffers.iter().zip(&vertex_attributes) {
            vertex_buffers.push(match buffer {
                Some(buffer) => Some(wgpu::VertexBufferLayout {
                    array_stride: buffer.array_stride,
                    step_mode: names::vertex_step_mode(&buffer.step_mode)?,
                    attributes,
                }),
                None => None,
            });
        }

        let vertex_constants = constants(&descriptor.vertex.constants);
        let fragment_constants = descriptor
            .fragment
            .as_ref()
            .map(|fragment| constants(&fragment.stage.constants))
            .unwrap_or_default();
        let fragment = match &descriptor.fragment {
            Some(fragment) => {
                let mut targets = Vec::with_capacity(fragment.targets.len());
                for target in &fragment.targets {
                    targets.push(match target {
                        Some(target) => Some(color_target_state(target)?),
                        None => None,
                    });
                }
                Some((
                    self.shader_module(fragment.stage.module)?,
                    fragment.stage.entry_point.as_deref(),
                    targets,
                ))
            }
            None => None,
        };
        let primitive = &descriptor.primitive;
        let depth_stencil = match &descriptor.depth_stencil {
            Some(state) => Some(depth_stencil_state(state)?),
            None => None,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout,
            vertex: wgpu::VertexState {
                module: vertex_module,
                entry_point: descriptor.vertex.entry_point.as_deref(),
                compilation_options: compilation_options(&vertex_constants),
                buffers: &vertex_buffers,
            },
            fragment: fragment
                .as_ref()
                .map(|(module, entry_point, targets)| wgpu::FragmentState {
                    module,
                    entry_point: *entry_point,
                    compilation_options: compilation_options(&fragment_constants),
                    targets,
                }),
            primitive: wgpu::PrimitiveState {
                topology: topology(primitive.topology),
                strip_index_format: match &primitive.strip_index_format {
                    Some(format) => Some(names::index_format(format)?),
                    None => None,
                },
                front_face: names::front_face(&primitive.front_face)?,
                cull_mode: names::cull_mode(&primitive.cull_mode)?,
                unclipped_depth: primitive.unclipped_depth,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil,
            multisample: wgpu::MultisampleState {
                count: descriptor.multisample.count,
                mask: u64::from(descriptor.multisample.mask),
                alpha_to_coverage_enabled: descriptor.multisample.alpha_to_coverage_enabled,
            },
            multiview_mask: None,
            cache: None,
        });
        self.insert(id, Object::RenderPipeline(pipeline))
    }

    fn create_canvas_texture(
        &mut self,
        device_id: ObjectId,
        id: ObjectId,
        descriptor: CanvasTextureDescriptor,
    ) -> BackendResult {
        let (device, _) = self.device(device_id)?;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("canvas texture"),
            size: wgpu::Extent3d {
                width: descriptor.width.max(1),
                height: descriptor.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format(descriptor.format),
            // NB: The compositor reads the canvas back with a copy.
            usage: wgpu::TextureUsages::from_bits_truncate(descriptor.usage)
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        self.insert(id, Object::Texture(texture))?;
        self.texture_devices.insert(id, device_id);
        Ok(())
    }

    fn can_share_images(&self, device: ObjectId, vendor: u32, device_id: u32) -> bool {
        let Ok((wgpu_device, _)) = self.device(device) else {
            return false;
        };
        self.device_ids.get(&device) == Some(&(vendor, device_id))
            && wgpu_device
                .features()
                .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
    }

    #[cfg(unix)]
    fn import_canvas_texture(
        &mut self,
        device_id: ObjectId,
        id: ObjectId,
        descriptor: CanvasTextureDescriptor,
        image: crate::backend::SharedCanvasImage,
    ) -> BackendResult {
        #[cfg(target_os = "macos")]
        {
            let _ = (device_id, id, descriptor, image);
            Err(BackendError("canvases share IOSurfaces on macOS, not dma-bufs".into()))
        }
        #[cfg(not(target_os = "macos"))]
        {
            use wgpu::hal::api::Vulkan;

            let (device, queue) = self.device(device_id)?;
            let format = texture_format(descriptor.format);
            let size = wgpu::Extent3d {
                width: descriptor.width.max(1),
                height: descriptor.height.max(1),
                depth_or_array_layers: 1,
            };
            let usage = wgpu::TextureUsages::from_bits_truncate(descriptor.usage)
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC;
            // NB: The image was made to be rendered into, sampled and copied, and nothing else.
            let shareable = wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST;
            if !shareable.contains(usage) {
                return Err(BackendError(format!("a shared image cannot have the usage {usage:?}")));
            }
            let mut uses = wgpu::TextureUses::COLOR_TARGET | wgpu::TextureUses::COPY_SRC;
            if usage.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
                uses |= wgpu::TextureUses::RESOURCE;
            }
            if usage.contains(wgpu::TextureUsages::COPY_DST) {
                uses |= wgpu::TextureUses::COPY_DST;
            }
            let hal_descriptor = wgpu::hal::TextureDescriptor {
                label: Some("shared canvas texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: uses,
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: Vec::new(),
            };
            let hal_texture = {
                // SAFETY: The device is a Vulkan device, which is the only backend this build has here.
                let hal_device = unsafe { device.as_hal::<Vulkan>() }
                    .ok_or_else(|| BackendError("the device is not a Vulkan device".into()))?;
                // SAFETY: The compositor made the dma-buf as one plane of this size and format, in the layout that
                //         the modifier, stride and offset describe, and it took the import feature into account.
                unsafe {
                    hal_device.texture_from_dmabuf_fd(
                        image.fd,
                        &hal_descriptor,
                        image.modifier,
                        image.stride,
                        image.offset,
                    )
                }
                .map_err(|error| BackendError(format!("importing a shared canvas image failed: {error}")))?
            };
            let descriptor = wgpu::TextureDescriptor {
                label: Some("shared canvas texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            };
            // SAFETY: The texture is the device's, made for this descriptor; its contents are discarded, as a new
            //         canvas texture's are.
            let texture = unsafe {
                device.create_texture_from_hal::<Vulkan>(hal_texture, &descriptor, wgpu::TextureUses::UNINITIALIZED)
            };
            // A canvas texture starts out transparent black, whatever the image held.
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            }));
            queue.submit([encoder.finish()]);
            self.insert(id, Object::Texture(texture))?;
            self.texture_devices.insert(id, device_id);
            Ok(())
        }
    }

    fn create_texture(&mut self, device_id: ObjectId, id: ObjectId, descriptor: &TextureDescriptor) -> BackendResult {
        let (device, _) = self.device(device_id)?;
        let usage = wgpu::TextureUsages::from_bits(descriptor.usage)
            .ok_or_else(|| BackendError(format!("{:#x} is not a valid texture usage", descriptor.usage)))?;
        let mut view_formats = Vec::with_capacity(descriptor.view_formats.len());
        for format in &descriptor.view_formats {
            view_formats.push(names::texture_format(format)?);
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: descriptor.width,
                height: descriptor.height,
                depth_or_array_layers: descriptor.depth_or_array_layers,
            },
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: names::texture_dimension(&descriptor.dimension)?,
            format: names::texture_format(&descriptor.format)?,
            usage,
            view_formats: &view_formats,
        });
        self.insert(id, Object::Texture(texture))?;
        self.texture_devices.insert(id, device_id);
        Ok(())
    }

    fn create_texture_view(
        &mut self,
        texture: ObjectId,
        id: ObjectId,
        descriptor: &TextureViewDescriptor,
    ) -> BackendResult {
        let view = self.texture(texture)?.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: match &descriptor.format {
                Some(format) => Some(names::texture_format(format)?),
                None => None,
            },
            dimension: match &descriptor.dimension {
                Some(dimension) => Some(names::texture_view_dimension(dimension)?),
                None => None,
            },
            usage: None,
            aspect: names::texture_aspect(&descriptor.aspect)?,
            base_mip_level: descriptor.base_mip_level,
            mip_level_count: descriptor.mip_level_count,
            base_array_layer: descriptor.base_array_layer,
            array_layer_count: descriptor.array_layer_count,
        });
        self.insert(id, Object::TextureView(view))
    }

    fn create_bind_group_layout(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        entries: &[BindGroupLayoutEntry],
    ) -> BackendResult {
        let (device, _) = self.device(device)?;
        let mut layout_entries = Vec::with_capacity(entries.len());
        for entry in entries {
            let ty = match &entry.resource {
                LayoutResource::Buffer {
                    kind,
                    has_dynamic_offset,
                    min_binding_size,
                } => wgpu::BindingType::Buffer {
                    ty: names::buffer_binding_type(kind)?,
                    has_dynamic_offset: *has_dynamic_offset,
                    min_binding_size: wgpu::BufferSize::new(*min_binding_size),
                },
                LayoutResource::Sampler { kind } => wgpu::BindingType::Sampler(names::sampler_binding_type(kind)?),
                LayoutResource::Texture {
                    sample_type,
                    view_dimension,
                    multisampled,
                } => wgpu::BindingType::Texture {
                    sample_type: names::texture_sample_type(sample_type)?,
                    view_dimension: names::texture_view_dimension(view_dimension)?,
                    multisampled: *multisampled,
                },
                LayoutResource::StorageTexture {
                    access,
                    format,
                    view_dimension,
                } => wgpu::BindingType::StorageTexture {
                    access: names::storage_texture_access(access)?,
                    format: names::texture_format(format)?,
                    view_dimension: names::texture_view_dimension(view_dimension)?,
                },
            };
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: entry.binding,
                visibility: wgpu::ShaderStages::from_bits(entry.visibility)
                    .ok_or_else(|| BackendError(format!("{:#x} is not a valid shader stage", entry.visibility)))?,
                ty,
                count: None,
            });
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &layout_entries,
        });
        self.insert(id, Object::BindGroupLayout(layout))
    }

    fn create_pipeline_layout(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        bind_group_layouts: &[ObjectId],
    ) -> BackendResult {
        let (device, _) = self.device(device)?;
        let mut layouts = Vec::with_capacity(bind_group_layouts.len());
        for layout in bind_group_layouts {
            match self.objects.get(layout) {
                Some(Object::BindGroupLayout(layout)) => layouts.push(Some(layout)),
                _ => return Err(missing(*layout, "bind group layout")),
            }
        }
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        self.insert(id, Object::PipelineLayout(layout))
    }

    fn create_bind_group(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        layout: ObjectId,
        entries: &[BindGroupEntry],
    ) -> BackendResult {
        let (device, _) = self.device(device)?;
        let Some(Object::BindGroupLayout(layout)) = self.objects.get(&layout) else {
            return Err(missing(layout, "bind group layout"));
        };
        let mut group_entries = Vec::with_capacity(entries.len());
        for entry in entries {
            let resource = match &entry.resource {
                BindingResource::Buffer { buffer, offset, size } => {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.buffer(*buffer)?,
                        offset: *offset,
                        size: size.and_then(wgpu::BufferSize::new),
                    })
                }
                BindingResource::Sampler { sampler } => match self.objects.get(sampler) {
                    Some(Object::Sampler(sampler)) => wgpu::BindingResource::Sampler(sampler),
                    _ => return Err(missing(*sampler, "sampler")),
                },
                BindingResource::TextureView { view } => match self.objects.get(view) {
                    Some(Object::TextureView(view)) => wgpu::BindingResource::TextureView(view),
                    _ => return Err(missing(*view, "texture view")),
                },
            };
            group_entries.push(wgpu::BindGroupEntry {
                binding: entry.binding,
                resource,
            });
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &group_entries,
        });
        self.insert(id, Object::BindGroup(group))
    }

    fn create_sampler(&mut self, device: ObjectId, id: ObjectId, descriptor: &SamplerDescriptor) -> BackendResult {
        let (device, _) = self.device(device)?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: names::address_mode(&descriptor.address_mode_u)?,
            address_mode_v: names::address_mode(&descriptor.address_mode_v)?,
            address_mode_w: names::address_mode(&descriptor.address_mode_w)?,
            mag_filter: names::filter_mode(&descriptor.mag_filter)?,
            min_filter: names::filter_mode(&descriptor.min_filter)?,
            mipmap_filter: names::mipmap_filter_mode(&descriptor.mipmap_filter)?,
            lod_min_clamp: descriptor.lod_min_clamp,
            lod_max_clamp: descriptor.lod_max_clamp,
            compare: match &descriptor.compare {
                Some(compare) => Some(names::compare_function(compare)?),
                None => None,
            },
            anisotropy_clamp: u16::try_from(descriptor.max_anisotropy).unwrap_or(u16::MAX),
            border_color: None,
        });
        self.insert(id, Object::Sampler(sampler))
    }

    fn create_compute_pipeline(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        descriptor: &ComputePipelineDescriptor,
    ) -> BackendResult {
        let (device, _) = self.device(device)?;
        let compute_constants = constants(&descriptor.compute.constants);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: self.pipeline_layout(descriptor.layout)?,
            module: self.shader_module(descriptor.compute.module)?,
            entry_point: descriptor.compute.entry_point.as_deref(),
            compilation_options: compilation_options(&compute_constants),
            cache: None,
        });
        self.insert(id, Object::ComputePipeline(pipeline))
    }

    fn get_bind_group_layout(&mut self, pipeline: ObjectId, index: u32, id: ObjectId) -> BackendResult {
        let layout = match self.objects.get(&pipeline) {
            Some(Object::RenderPipeline(pipeline)) => pipeline.get_bind_group_layout(index),
            Some(Object::ComputePipeline(pipeline)) => pipeline.get_bind_group_layout(index),
            _ => return Err(missing(pipeline, "pipeline")),
        };
        self.insert(id, Object::BindGroupLayout(layout))
    }

    fn create_command_encoder(&mut self, device: ObjectId, id: ObjectId) -> BackendResult {
        let (device, _) = self.device(device)?;
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.insert(id, Object::CommandEncoder(encoder))
    }

    fn encode_render_pass(
        &mut self,
        encoder_id: ObjectId,
        color_attachments: &[Option<ColorAttachment>],
        depth_stencil: Option<&DepthStencilAttachment>,
        occlusion_query_set: Option<ObjectId>,
        timestamp_writes: Option<&TimestampWrites>,
        ops: &[RenderPassOp],
    ) -> BackendResult {
        // Take the encoder out so that the views and pipelines it uses can be borrowed from the map meanwhile.
        let Some(Object::CommandEncoder(mut encoder)) = self.objects.remove(&encoder_id) else {
            return Err(missing(encoder_id, "command encoder"));
        };
        let result = self.record_render_pass(
            &mut encoder,
            color_attachments,
            depth_stencil,
            occlusion_query_set,
            timestamp_writes,
            ops,
        );
        self.objects.insert(encoder_id, Object::CommandEncoder(encoder));
        result
    }

    fn encode_compute_pass(
        &mut self,
        encoder: ObjectId,
        timestamp_writes: Option<&TimestampWrites>,
        ops: &[ComputePassOp],
    ) -> BackendResult {
        self.with_encoder(encoder, |this, encoder| {
            this.record_compute_pass(encoder, timestamp_writes, ops)
        })
    }

    fn create_render_bundle(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        descriptor: &RenderBundleDescriptor,
        ops: &[RenderPassOp],
    ) -> BackendResult {
        let (device, _) = self.device(device)?;
        let mut color_formats = Vec::with_capacity(descriptor.color_formats.len());
        for format in &descriptor.color_formats {
            color_formats.push(match format {
                Some(format) => Some(names::texture_format(format)?),
                None => None,
            });
        }
        let depth_stencil = match &descriptor.depth_stencil_format {
            Some(format) => Some(wgpu::RenderBundleDepthStencil {
                format: names::texture_format(format)?,
                depth_read_only: descriptor.depth_read_only,
                stencil_read_only: descriptor.stencil_read_only,
            }),
            None => None,
        };
        let mut encoder = device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            label: None,
            color_formats: &color_formats,
            depth_stencil,
            sample_count: descriptor.sample_count,
            multiview: None,
        });
        for op in ops {
            if !self.record_draw_op(&mut encoder, op)? {
                return Err(BackendError("a render bundle cannot record that operation".into()));
            }
        }
        let bundle = encoder.finish(&wgpu::RenderBundleDescriptor { label: None });
        self.insert(id, Object::RenderBundle(bundle))
    }

    fn create_query_set(&mut self, device: ObjectId, id: ObjectId, kind: &str, count: u32) -> BackendResult {
        let (device, _) = self.device(device)?;
        let ty = match kind {
            "occlusion" => wgpu::QueryType::Occlusion,
            "timestamp" => wgpu::QueryType::Timestamp,
            _ => return Err(BackendError(format!("{kind:?} is not a query type"))),
        };
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty, count });
        self.insert(id, Object::QuerySet(query_set))
    }

    fn resolve_query_set(
        &mut self,
        encoder: ObjectId,
        query_set: ObjectId,
        queries: core::ops::Range<u32>,
        destination: ObjectId,
        destination_offset: u64,
    ) -> BackendResult {
        self.with_encoder(encoder, |this, encoder| {
            encoder.resolve_query_set(
                this.query_set(query_set)?,
                queries,
                this.buffer(destination)?,
                destination_offset,
            );
            Ok(())
        })
    }

    fn finish_command_encoder(&mut self, encoder: ObjectId, id: ObjectId) -> BackendResult {
        let Some(Object::CommandEncoder(command_encoder)) = self.objects.remove(&encoder) else {
            return Err(missing(encoder, "command encoder"));
        };
        self.insert(id, Object::CommandBuffer(command_encoder.finish()))
    }

    fn submit(&mut self, device: ObjectId, command_buffers: &[ObjectId]) -> BackendResult {
        let mut buffers = Vec::with_capacity(command_buffers.len());
        for id in command_buffers {
            match self.objects.remove(id) {
                Some(Object::CommandBuffer(buffer)) => buffers.push(buffer),
                Some(other) => {
                    let kind = other.kind();
                    self.objects.insert(*id, other);
                    return Err(BackendError(format!("id {id} is a {kind}, not a command buffer")));
                }
                None => return Err(missing(*id, "command buffer")),
            }
        }
        let (_, queue) = self.device(device)?;
        queue.submit(buffers);
        Ok(())
    }

    fn start_readback(&mut self, texture_id: ObjectId, tag: u64) -> BackendResult {
        let device_id = *self
            .texture_devices
            .get(&texture_id)
            .ok_or_else(|| missing(texture_id, "canvas texture"))?;
        let texture = self.texture(texture_id)?;
        let (device, queue) = self.device(device_id)?;
        let format =
            wire_texture_format(texture.format()).ok_or_else(|| BackendError("unreadable texture format".into()))?;
        let (width, height) = (texture.width(), texture.height());
        let padded_row_bytes = (width * texel_size(format)).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

        // FIXME: Reuse the buffers of finished readbacks of the same size.
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("canvas readback"),
            size: u64::from(padded_row_bytes) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );
        queue.submit([encoder.finish()]);

        let done = Arc::new(Mutex::new(None));
        let callback_done = Arc::clone(&done);
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            *callback_done.lock().unwrap_or_else(PoisonError::into_inner) =
                Some(result.map_err(|error| error.to_string()));
        });
        self.readbacks.push(PendingFrame {
            tag,
            device: device_id,
            content: PendingContent::Readback(PendingReadback {
                buffer,
                width,
                height,
                format,
                padded_row_bytes,
            }),
            done,
        });
        Ok(())
    }

    fn finish_shared_frame(&mut self, texture_id: ObjectId, tag: u64, image: u64) -> BackendResult {
        let device_id = *self
            .texture_devices
            .get(&texture_id)
            .ok_or_else(|| missing(texture_id, "canvas texture"))?;
        let texture = self.texture(texture_id)?;
        let (device, queue) = self.device(device_id)?;
        // NB: The compositor's renderer reads the image in the GENERAL layout, which is where wgpu's Vulkan backend
        //     puts a color texture for more than one read-only use.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.transition_resources(
            core::iter::empty(),
            core::iter::once(wgpu::TextureTransition {
                texture,
                selector: None,
                state: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
            }),
        );
        queue.submit([encoder.finish()]);
        let done = Arc::new(Mutex::new(None));
        let callback_done = Arc::clone(&done);
        queue.on_submitted_work_done(move || {
            *callback_done.lock().unwrap_or_else(PoisonError::into_inner) = Some(Ok(()));
        });
        self.readbacks.push(PendingFrame {
            tag,
            device: device_id,
            content: PendingContent::SharedImage(image),
            done,
        });
        Ok(())
    }

    fn finished_readbacks(&mut self, wait: bool) -> Vec<(u64, BackendResult<PresentedFrame>)> {
        let mut devices: Vec<ObjectId> = self.readbacks.iter().map(|readback| readback.device).collect();
        devices.dedup();
        for device in devices {
            let Ok((device, _)) = self.device(device) else {
                continue;
            };
            let poll = if wait {
                wgpu::PollType::wait_indefinitely()
            } else {
                wgpu::PollType::Poll
            };
            if let Err(error) = device.poll(poll) {
                eprintln!("WebGPU: polling the GPU for readbacks failed: {error}");
            }
        }
        let mut finished = Vec::new();
        let mut index = 0;
        while index < self.readbacks.len() {
            let readback = &self.readbacks[index];
            // NB: A readback whose device is gone will never finish.
            let device_is_gone = self.device(readback.device).is_err();
            match readback.status() {
                None if !device_is_gone => index += 1,
                status => {
                    let readback = self.readbacks.remove(index);
                    let tag = readback.tag;
                    let result = match status {
                        Some(Ok(())) => readback.take(),
                        Some(Err(error)) => Err(BackendError(format!("readback map failed: {error}"))),
                        None => Err(BackendError("the device of the readback is gone".into())),
                    };
                    finished.push((tag, result));
                }
            }
        }
        finished
    }

    fn pending_readbacks(&self, tag: u64) -> usize {
        self.readbacks.iter().filter(|readback| readback.tag == tag).count()
    }

    fn create_buffer(
        &mut self,
        device_id: ObjectId,
        id: ObjectId,
        size: u64,
        usage: u32,
        mapped_at_creation: bool,
    ) -> BackendResult {
        let (device, _) = self.device(device_id)?;
        // NB: GPUBufferUsage has wgpu's bits.
        let usage = wgpu::BufferUsages::from_bits(usage)
            .ok_or_else(|| BackendError(format!("{usage:#x} is not a valid buffer usage")))?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation,
        });
        self.insert(id, Object::Buffer(buffer))?;
        self.texture_devices.insert(id, device_id);
        if mapped_at_creation {
            self.mapped_at_creation.insert(id);
        }
        Ok(())
    }

    fn write_buffer(&mut self, device: ObjectId, buffer: ObjectId, offset: u64, data: &[u8]) -> BackendResult {
        let buffer = self.buffer(buffer)?;
        let (_, queue) = self.device(device)?;
        queue.write_buffer(buffer, offset, data);
        Ok(())
    }

    fn copy_buffer_to_buffer(
        &mut self,
        encoder: ObjectId,
        source: ObjectId,
        source_offset: u64,
        destination: ObjectId,
        destination_offset: u64,
        size: u64,
    ) -> BackendResult {
        self.with_encoder(encoder, |this, encoder| {
            encoder.copy_buffer_to_buffer(
                this.buffer(source)?,
                source_offset,
                this.buffer(destination)?,
                destination_offset,
                size,
            );
            Ok(())
        })
    }

    fn clear_buffer(&mut self, encoder: ObjectId, buffer: ObjectId, offset: u64, size: Option<u64>) -> BackendResult {
        self.with_encoder(encoder, |this, encoder| {
            encoder.clear_buffer(this.buffer(buffer)?, offset, size);
            Ok(())
        })
    }

    fn copy_buffer_to_texture(
        &mut self,
        encoder: ObjectId,
        buffer: ObjectId,
        layout: &TexelCopyBufferLayout,
        texture: &TexelCopyTexture,
        size: Extent,
    ) -> BackendResult {
        self.with_encoder(encoder, |this, encoder| {
            let source = wgpu::TexelCopyBufferInfo {
                buffer: this.buffer(buffer)?,
                layout: texel_copy_buffer_layout(layout),
            };
            encoder.copy_buffer_to_texture(source, this.texel_copy_texture(texture)?, extent(size));
            Ok(())
        })
    }

    fn copy_texture_to_buffer(
        &mut self,
        encoder: ObjectId,
        texture: &TexelCopyTexture,
        buffer: ObjectId,
        layout: &TexelCopyBufferLayout,
        size: Extent,
    ) -> BackendResult {
        self.with_encoder(encoder, |this, encoder| {
            let destination = wgpu::TexelCopyBufferInfo {
                buffer: this.buffer(buffer)?,
                layout: texel_copy_buffer_layout(layout),
            };
            encoder.copy_texture_to_buffer(this.texel_copy_texture(texture)?, destination, extent(size));
            Ok(())
        })
    }

    fn copy_texture_to_texture(
        &mut self,
        encoder: ObjectId,
        source: &TexelCopyTexture,
        destination: &TexelCopyTexture,
        size: Extent,
    ) -> BackendResult {
        self.with_encoder(encoder, |this, encoder| {
            encoder.copy_texture_to_texture(
                this.texel_copy_texture(source)?,
                this.texel_copy_texture(destination)?,
                extent(size),
            );
            Ok(())
        })
    }

    fn write_texture(
        &mut self,
        device: ObjectId,
        texture: &TexelCopyTexture,
        data: &[u8],
        layout: &TexelCopyBufferLayout,
        size: Extent,
    ) -> BackendResult {
        let (_, queue) = self.device(device)?;
        queue.write_texture(
            self.texel_copy_texture(texture)?,
            data,
            texel_copy_buffer_layout(layout),
            extent(size),
        );
        Ok(())
    }

    fn copy_external_image_to_texture(
        &mut self,
        device: ObjectId,
        texture: &TexelCopyTexture,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> BackendResult {
        let (_, queue) = self.device(device)?;
        let destination = self.texel_copy_texture(texture)?;
        if rgba8.len() as u64 != u64::from(width) * u64::from(height) * 4 {
            return Err(BackendError(format!(
                "{} bytes are not {width}×{height} RGBA8 texels",
                rgba8.len()
            )));
        }
        let (texels, texel_size) = convert_rgba8_texels(destination.texture.format(), rgba8)?;
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * texel_size),
            rows_per_image: Some(height),
        };
        queue.write_texture(destination, &texels, layout, extent((width, height, 1)));
        Ok(())
    }

    fn map_buffer(
        &mut self,
        buffer_id: ObjectId,
        for_writing: bool,
        offset: u64,
        size: u64,
    ) -> Result<Vec<u8>, String> {
        let device_id = *self
            .texture_devices
            .get(&buffer_id)
            .ok_or_else(|| missing(buffer_id, "buffer").0)?;
        let buffer = self.buffer(buffer_id).map_err(|error| error.0)?;
        let (device, _) = self.device(device_id).map_err(|error| error.0)?;
        let mode = if for_writing {
            wgpu::MapMode::Write
        } else {
            wgpu::MapMode::Read
        };
        let slice = buffer.slice(offset..offset.saturating_add(size));
        let result = Arc::new(Mutex::new(None));
        let callback_result = Arc::clone(&result);
        slice.map_async(mode, move |mapped| {
            *callback_result.lock().unwrap_or_else(PoisonError::into_inner) = Some(mapped);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| format!("waiting for the GPU failed: {error}"))?;
        let mapped = result.lock().unwrap_or_else(PoisonError::into_inner).take();
        match mapped {
            Some(Ok(())) => {}
            Some(Err(error)) => return Err(error.to_string()),
            None => return Err("the buffer was not mapped".to_owned()),
        }
        let contents = slice.get_mapped_range().map_err(|error| error.to_string())?.to_vec();
        buffer.unmap();
        Ok(contents)
    }

    fn unmap_buffer(&mut self, buffer_id: ObjectId, offset: u64, data: &[u8]) -> BackendResult {
        let range = offset..offset.saturating_add(data.len() as u64);
        // NB: A buffer mapped at creation is still mapped here; one that script mapped for writing is not.
        let was_mapped_at_creation = self.mapped_at_creation.remove(&buffer_id);
        let buffer = self.buffer(buffer_id)?;
        if !was_mapped_at_creation {
            let slice = buffer.slice(range.clone());
            let result = Arc::new(Mutex::new(None));
            let callback_result = Arc::clone(&result);
            slice.map_async(wgpu::MapMode::Write, move |mapped| {
                *callback_result.lock().unwrap_or_else(PoisonError::into_inner) = Some(mapped);
            });
            let device_id = self.texture_devices.get(&buffer_id).copied().unwrap_or_default();
            let (device, _) = self.device(device_id)?;
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| BackendError(format!("waiting for the GPU failed: {error}")))?;
            if !matches!(
                result.lock().unwrap_or_else(PoisonError::into_inner).take(),
                Some(Ok(()))
            ) {
                return Err(BackendError("the buffer could not be mapped for writing".to_owned()));
            }
        }
        let buffer = self.buffer(buffer_id)?;
        buffer
            .slice(range)
            .get_mapped_range_mut()
            .map_err(|error| BackendError(error.to_string()))?
            .copy_from_slice(data);
        buffer.unmap();
        Ok(())
    }

    fn wait_for_submitted_work(&mut self, device: ObjectId) -> BackendResult {
        let (device, _) = self.device(device)?;
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| BackendError(format!("waiting for the GPU failed: {error}")))
    }

    fn destroy(&mut self, id: ObjectId) {
        match self.objects.get(&id) {
            Some(Object::Texture(texture)) => texture.destroy(),
            Some(Object::Buffer(buffer)) => buffer.destroy(),
            Some(Object::Device(device, _)) => {
                device.destroy();
                // NB: wgpu calls the device lost callback when it polls the device, which this does before the device
                //     goes. Objects made on the device keep what they need of it until they are released themselves.
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                self.objects.insert(id, Object::DestroyedDevice);
                self.drop_error_scopes(id);
            }
            _ => {}
        }
    }

    fn is_destroyed_device(&self, id: ObjectId) -> bool {
        matches!(self.objects.get(&id), Some(Object::DestroyedDevice))
    }

    fn release(&mut self, id: ObjectId) {
        self.objects.remove(&id);
        self.texture_devices.remove(&id);
        self.drop_error_scopes(id);
        self.mapped_at_creation.remove(&id);
        self.device_ids.remove(&id);
    }
}
