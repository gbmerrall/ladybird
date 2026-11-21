/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! The seam between Ladybird's WebGPU plumbing and the library that talks to the GPU.
//!
//! Everything above this trait (the page-side API, the wire protocol, canvas bookkeeping and presentation) is
//! Ladybird's own. An implementation only has to execute GPU work on objects named by the ids WebContent chose.
//! Its shape follows webgpu.h, which Dawn and wgpu-native both implement, so either could sit behind it.

use webgpu_wire::{
    AdapterInfo, BindGroupEntry, BindGroupLayoutEntry, Capabilities, ColorAttachment, CompilationMessage,
    ComputePassOp, ComputePipelineDescriptor, DepthStencilAttachment, DeviceRequestError, ErrorFilter, Event, GpuError,
    ObjectId, PowerPreference, RenderBundleDescriptor, RenderPassOp, RenderPipelineDescriptor, SamplerDescriptor,
    TexelCopyBufferLayout, TexelCopyTexture, TextureDescriptor, TextureFormat, TextureViewDescriptor, TimestampWrites,
};

/// The width, height and depth or array layers of a copy.
pub type Extent = (u32, u32, u32);

#[derive(Clone, Copy, Debug)]
pub struct CanvasTextureDescriptor {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    /// The bits of GPUTextureUsage that the canvas's configuration asks for.
    pub usage: u32,
}

/// Pixels read back from a texture, with rows packed tightly, in the texture's own channel order.
pub struct Readback {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub pixels: Vec<u8>,
}

/// An image of the compositor's renderer that a canvas can render into directly, shared as a single-plane dma-buf.
/// `handle` is the compositor's name for it.
#[cfg(unix)]
pub struct SharedCanvasImage {
    pub handle: u64,
    pub fd: std::os::fd::OwnedFd,
    pub modifier: u64,
    pub stride: u64,
    pub offset: u64,
}

/// A canvas frame that is back from the GPU: its pixels, or the shared image it was rendered into.
pub enum PresentedFrame {
    Pixels(Readback),
    SharedImage(u64),
}

/// A failure to find the object an id names, or to execute a call. Callers log it; WebGPU reports most errors
/// asynchronously, so none of these stop the stream of commands.
#[derive(Debug)]
pub struct BackendError(pub String);

pub type BackendResult<T = ()> = Result<T, BackendError>;

pub trait Backend {
    /// Without a power preference, an adapter on the GPU with the preferred PCI vendor and device ids is picked if
    /// there is one, which is the compositor's, so that canvases can render into its images.
    fn request_adapter(
        &mut self,
        id: ObjectId,
        power_preference: PowerPreference,
        force_fallback_adapter: bool,
        preferred_gpu: Option<(u32, u32)>,
    ) -> Option<AdapterInfo>;
    /// Errors on the device that no error scope catches, and the loss of the device, become events that
    /// `take_events` returns.
    /// The device has the required features and limits, and the defaults for every other limit.
    fn request_device(
        &mut self,
        adapter: ObjectId,
        id: ObjectId,
        required_features: &[String],
        required_limits: &[(String, u64)],
    ) -> Result<Capabilities, DeviceRequestError>;

    fn push_error_scope(&mut self, device: ObjectId, filter: ErrorFilter) -> BackendResult;
    /// The error the innermost scope caught, if any, or Err if the device has no scope to pop.
    fn pop_error_scope(&mut self, device: ObjectId) -> Result<Option<GpuError>, String>;
    fn compilation_info(&mut self, module: ObjectId) -> BackendResult<Vec<CompilationMessage>>;

    /// The uncaptured errors and lost devices since the last call.
    fn take_events(&mut self) -> Vec<Event>;

    fn create_shader_module(&mut self, device: ObjectId, id: ObjectId, code: &str) -> BackendResult;
    fn create_render_pipeline(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        descriptor: &RenderPipelineDescriptor,
    ) -> BackendResult;
    /// A texture a canvas renders into: usable as a render attachment and readable back.
    fn create_canvas_texture(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        descriptor: CanvasTextureDescriptor,
    ) -> BackendResult;
    /// Whether the device runs on the GPU with these PCI vendor and device ids, and can render into images it
    /// imports, so that its canvases can render into the compositor's images instead of being read back.
    fn can_share_images(&self, device: ObjectId, vendor: u32, device_id: u32) -> bool;
    /// A canvas texture over a shared image. It starts out transparent black. Fails if the image cannot be imported,
    /// and the canvas is then read back instead.
    #[cfg(unix)]
    fn import_canvas_texture(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        descriptor: CanvasTextureDescriptor,
        image: SharedCanvasImage,
    ) -> BackendResult;
    fn create_texture(&mut self, device: ObjectId, id: ObjectId, descriptor: &TextureDescriptor) -> BackendResult;
    fn create_texture_view(
        &mut self,
        texture: ObjectId,
        id: ObjectId,
        descriptor: &TextureViewDescriptor,
    ) -> BackendResult;
    fn create_bind_group_layout(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        entries: &[BindGroupLayoutEntry],
    ) -> BackendResult;
    fn create_pipeline_layout(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        bind_group_layouts: &[ObjectId],
    ) -> BackendResult;
    fn create_bind_group(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        layout: ObjectId,
        entries: &[BindGroupEntry],
    ) -> BackendResult;
    fn create_sampler(&mut self, device: ObjectId, id: ObjectId, descriptor: &SamplerDescriptor) -> BackendResult;
    /// The bind group layout of a pipeline with an "auto" layout.
    fn create_compute_pipeline(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        descriptor: &ComputePipelineDescriptor,
    ) -> BackendResult;
    /// Of a render or a compute pipeline.
    fn get_bind_group_layout(&mut self, pipeline: ObjectId, index: u32, id: ObjectId) -> BackendResult;
    fn create_command_encoder(&mut self, device: ObjectId, id: ObjectId) -> BackendResult;
    fn encode_render_pass(
        &mut self,
        encoder: ObjectId,
        color_attachments: &[Option<ColorAttachment>],
        depth_stencil: Option<&DepthStencilAttachment>,
        occlusion_query_set: Option<ObjectId>,
        timestamp_writes: Option<&TimestampWrites>,
        ops: &[RenderPassOp],
    ) -> BackendResult;
    fn encode_compute_pass(
        &mut self,
        encoder: ObjectId,
        timestamp_writes: Option<&TimestampWrites>,
        ops: &[ComputePassOp],
    ) -> BackendResult;
    /// A render bundle with the operations it records.
    fn create_render_bundle(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        descriptor: &RenderBundleDescriptor,
        ops: &[RenderPassOp],
    ) -> BackendResult;
    /// `kind` is "occlusion" or "timestamp".
    fn create_query_set(&mut self, device: ObjectId, id: ObjectId, kind: &str, count: u32) -> BackendResult;
    fn resolve_query_set(
        &mut self,
        encoder: ObjectId,
        query_set: ObjectId,
        queries: core::ops::Range<u32>,
        destination: ObjectId,
        destination_offset: u64,
    ) -> BackendResult;
    fn finish_command_encoder(&mut self, encoder: ObjectId, id: ObjectId) -> BackendResult;
    fn submit(&mut self, device: ObjectId, command_buffers: &[ObjectId]) -> BackendResult;

    /// Starts copying a texture's pixels back to the CPU, without waiting for the GPU. `finished_readbacks` hands
    /// them over with the tag once the GPU is done.
    fn start_readback(&mut self, texture: ObjectId, tag: u64) -> BackendResult;
    /// Leaves a texture over a shared image in the layout the compositor's renderer expects once the work submitted
    /// so far is done, and hands the image over with the tag (`finished_readbacks`) when the GPU is.
    fn finish_shared_frame(&mut self, texture: ObjectId, tag: u64, image: u64) -> BackendResult;
    /// The frames that are back, oldest first, with their tags. With `wait`, waits for every one started.
    fn finished_readbacks(&mut self, wait: bool) -> Vec<(u64, BackendResult<PresentedFrame>)>;
    /// How many frames with the tag are not handed over yet.
    fn pending_readbacks(&self, tag: u64) -> usize;

    /// `usage` has the bits of GPUBufferUsage.
    fn create_buffer(
        &mut self,
        device: ObjectId,
        id: ObjectId,
        size: u64,
        usage: u32,
        mapped_at_creation: bool,
    ) -> BackendResult;
    fn write_buffer(&mut self, device: ObjectId, buffer: ObjectId, offset: u64, data: &[u8]) -> BackendResult;
    fn copy_buffer_to_buffer(
        &mut self,
        encoder: ObjectId,
        source: ObjectId,
        source_offset: u64,
        destination: ObjectId,
        destination_offset: u64,
        size: u64,
    ) -> BackendResult;
    fn clear_buffer(&mut self, encoder: ObjectId, buffer: ObjectId, offset: u64, size: Option<u64>) -> BackendResult;
    fn copy_buffer_to_texture(
        &mut self,
        encoder: ObjectId,
        buffer: ObjectId,
        layout: &TexelCopyBufferLayout,
        texture: &TexelCopyTexture,
        size: Extent,
    ) -> BackendResult;
    fn copy_texture_to_buffer(
        &mut self,
        encoder: ObjectId,
        texture: &TexelCopyTexture,
        buffer: ObjectId,
        layout: &TexelCopyBufferLayout,
        size: Extent,
    ) -> BackendResult;
    fn copy_texture_to_texture(
        &mut self,
        encoder: ObjectId,
        source: &TexelCopyTexture,
        destination: &TexelCopyTexture,
        size: Extent,
    ) -> BackendResult;
    fn write_texture(
        &mut self,
        device: ObjectId,
        texture: &TexelCopyTexture,
        data: &[u8],
        layout: &TexelCopyBufferLayout,
        size: Extent,
    ) -> BackendResult;
    /// Writes `width` × `height` RGBA8 texels, converted to the texture's format, at the destination.
    fn copy_external_image_to_texture(
        &mut self,
        device: ObjectId,
        texture: &TexelCopyTexture,
        width: u32,
        height: u32,
        rgba8: &[u8],
    ) -> BackendResult;
    /// Waits for the GPU and returns the contents of the range. The buffer is left unmapped: WebContent holds the
    /// mapping, and writes back what it wrote with `unmap_buffer`.
    fn map_buffer(&mut self, buffer: ObjectId, for_writing: bool, offset: u64, size: u64) -> Result<Vec<u8>, String>;
    /// Writes what script wrote into a mapping for writing, or into the mapping at creation, and ends that mapping.
    fn unmap_buffer(&mut self, buffer: ObjectId, offset: u64, data: &[u8]) -> BackendResult;
    /// Waits for the work submitted so far.
    fn wait_for_submitted_work(&mut self, device: ObjectId) -> BackendResult;

    /// Whether the id names a device that script destroyed.
    fn is_destroyed_device(&self, id: ObjectId) -> bool;
    /// The explicit `destroy()` of a texture or a device. The id goes on naming the destroyed object.
    fn destroy(&mut self, id: ObjectId);

    /// Forgets the id. GPU work that still uses the object keeps it alive until the work is done.
    fn release(&mut self, id: ObjectId);
}
