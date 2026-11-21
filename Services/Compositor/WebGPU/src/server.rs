/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! Executes what one WebContent sends: decodes commands, keeps the state of each WebGPU canvas, and drives
//! the Backend.

use std::collections::HashMap;

use webgpu_wire::{
    Blob, CanvasId, Command, DecodeError, ErrorFilter, Event, GpuError, ObjectId, Reply, Request, TextureFormat,
};

use crate::backend::{Backend, CanvasTextureDescriptor, PresentedFrame};

struct Canvas {
    device: ObjectId,
    format: TextureFormat,
    usage: u32,
    /// The texture script draws into this frame, which the next present shows.
    current_texture: Option<ObjectId>,
    /// The compositor's image under the current texture, if it renders into one.
    current_image: Option<u64>,
    /// Whether the canvas tries to render into the compositor's images, which it stops doing after a failure.
    shares_images: bool,
}

/// Where canvases get the images of the compositor's renderer that they render into directly (zero copy).
pub trait CanvasImageSource {
    /// The PCI vendor and device ids of the GPU the images are on.
    fn gpu(&self) -> (u32, u32);
    /// An image of the size that the compositor is not using, or None if there is none to spare.
    #[cfg(unix)]
    fn allocate(&mut self, canvas: CanvasId, width: u32, height: u32) -> Option<crate::backend::SharedCanvasImage>;
    /// Gives back an image that will not be presented.
    fn release(&mut self, canvas: CanvasId, image: u64);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ServerOptions {
    /// Every adapter is the fallback adapter, which is a software rasterizer (Mesa's lavapipe on Linux), so that
    /// tests render the same without a GPU.
    pub force_fallback_adapter: bool,
    /// Canvases are always read back, instead of rendering into the compositor's images where they can.
    pub readback_only: bool,
}

impl ServerOptions {
    /// Reads `LADYBIRD_WEBGPU_FORCE_FALLBACK_ADAPTER=1` and `LADYBIRD_WEBGPU_READBACK_ONLY=1`.
    #[must_use]
    pub fn from_environment() -> Self {
        let is_set = |name| std::env::var_os(name).is_some_and(|value| value == "1");
        Self {
            force_fallback_adapter: is_set("LADYBIRD_WEBGPU_FORCE_FALLBACK_ADAPTER"),
            readback_only: is_set("LADYBIRD_WEBGPU_READBACK_ONLY"),
        }
    }
}

pub struct Server {
    backend: Box<dyn Backend>,
    options: ServerOptions,
    canvases: HashMap<CanvasId, Canvas>,
    /// The newest frame of each canvas that came back from the GPU and is not handed over yet.
    finished_frames: HashMap<CanvasId, PresentedFrame>,
    image_source: Option<Box<dyn CanvasImageSource>>,
    /// The device each object was made on, so that destroying a device lets go of everything made on it.
    object_devices: HashMap<ObjectId, ObjectId>,
    /// Devices that script destroyed, whose errors nobody hears of any more.
    destroyed_devices: std::collections::HashSet<ObjectId>,
}

/// How many frames of one canvas may be on their way back from the GPU before presenting waits for them. A page that
/// draws faster than the GPU reads back is slowed down rather than queuing without bound.
const MAX_PENDING_FRAMES_PER_CANVAS: usize = 2;

impl Server {
    #[must_use]
    pub fn new(backend: Box<dyn Backend>, options: ServerOptions) -> Self {
        Self {
            backend,
            options,
            canvases: HashMap::new(),
            finished_frames: HashMap::new(),
            image_source: None,
            object_devices: HashMap::new(),
            destroyed_devices: std::collections::HashSet::new(),
        }
    }

    /// Records the device of the object a command makes, which is its parent's for an object made from another.
    fn record_new_object(&mut self, command: &Command) {
        let (id, parent) = match *command {
            Command::CreateShaderModule { device, id, .. }
            | Command::CreateRenderPipeline { device, id, .. }
            | Command::CreateCommandEncoder { device, id }
            | Command::CreateBuffer { device, id, .. }
            | Command::CreateTexture { device, id, .. }
            | Command::CreateBindGroupLayout { device, id, .. }
            | Command::CreatePipelineLayout { device, id, .. }
            | Command::CreateBindGroup { device, id, .. }
            | Command::CreateSampler { device, id, .. }
            | Command::CreateComputePipeline { device, id, .. }
            | Command::CreateQuerySet { device, id, .. }
            | Command::CreateRenderBundle { device, id, .. } => (id, device),
            Command::CreateTextureView { texture, id, .. } => (id, texture),
            Command::FinishCommandEncoder { encoder, id } => (id, encoder),
            Command::GetBindGroupLayout { pipeline, id, .. } => (id, pipeline),
            Command::Request {
                body:
                    Request::CreateRenderPipeline { device, id, .. } | Request::CreateComputePipeline { device, id, .. },
                ..
            } => (id, device),
            _ => return,
        };
        let device = self.object_devices.get(&parent).copied().unwrap_or(parent);
        self.object_devices.insert(id, device);
    }

    /// Lets go of everything made on a device that script destroyed: those objects are invalid now, and would keep
    /// the driver's device alive until script releases them. NVIDIA's driver allows about 64 devices at once.
    fn release_objects_of(&mut self, device: ObjectId) {
        let objects: Vec<ObjectId> = self
            .object_devices
            .iter()
            .filter(|(_, owner)| **owner == device)
            .map(|(object, _)| *object)
            .collect();
        for object in objects {
            self.object_devices.remove(&object);
            self.backend.release(object);
        }
    }

    /// Lets canvases in BGRA render into the compositor's images, instead of being read back.
    pub fn set_image_source(&mut self, source: Box<dyn CanvasImageSource>) {
        if !self.options.readback_only {
            self.image_source = Some(source);
        }
    }

    fn release_image(&mut self, canvas: CanvasId, image: u64) {
        if let Some(source) = self.image_source.as_mut() {
            source.release(canvas, image);
        }
    }

    /// A texture for the canvas over one of the compositor's images, if the canvas can render into one.
    #[cfg(unix)]
    fn import_canvas_texture(&mut self, canvas: CanvasId, id: ObjectId, descriptor: CanvasTextureDescriptor) -> bool {
        let Some(state) = self.canvases.get(&canvas) else {
            return false;
        };
        let device = state.device;
        let Some(source) = self.image_source.as_mut() else {
            return false;
        };
        let (vendor, device_id) = source.gpu();
        if !state.shares_images
            || descriptor.format != TextureFormat::Bgra8Unorm
            || !self.backend.can_share_images(device, vendor, device_id)
        {
            return false;
        }
        let Some(image) = source.allocate(canvas, descriptor.width.max(1), descriptor.height.max(1)) else {
            return false;
        };
        let handle = image.handle;
        match self.backend.import_canvas_texture(device, id, descriptor, image) {
            Ok(()) => {
                if let Some(state) = self.canvases.get_mut(&canvas) {
                    state.current_image = Some(handle);
                }
                true
            }
            Err(error) => {
                eprintln!("WebGPU: canvas {canvas} is read back instead: {}", error.0);
                self.release_image(canvas, handle);
                if let Some(state) = self.canvases.get_mut(&canvas) {
                    state.shares_images = false;
                }
                false
            }
        }
    }

    #[cfg(not(unix))]
    fn import_canvas_texture(&mut self, _: CanvasId, _: ObjectId, _: CanvasTextureDescriptor) -> bool {
        false
    }

    /// Executes a stream of commands and returns the events for WebContent: the replies to its requests, and
    /// errors and lost devices. A malformed stream is the sender misbehaving; the commands before the malformed
    /// one have run.
    pub fn handle_commands(&mut self, bytes: &[u8]) -> Result<Vec<Event>, DecodeError> {
        let mut events = Vec::new();
        let mut result = Ok(());
        match webgpu_wire::decode_commands(bytes) {
            Ok(commands) => {
                for command in commands {
                    self.execute(command, &mut events);
                }
            }
            Err(error) => result = Err(error),
        }
        events.extend(self.backend.take_events());
        result.map(|()| events)
    }

    fn answer(&mut self, request: Request) -> Reply {
        match request {
            Request::RequestAdapter {
                id,
                power_preference,
                force_fallback_adapter,
            } => {
                let force_fallback_adapter = force_fallback_adapter || self.options.force_fallback_adapter;
                let preferred_gpu = self.image_source.as_ref().map(|source| source.gpu());
                Reply::Adapter {
                    info: self
                        .backend
                        .request_adapter(id, power_preference, force_fallback_adapter, preferred_gpu),
                }
            }
            Request::RequestDevice {
                adapter,
                id,
                required_features,
                required_limits,
            } => Reply::Device {
                result: self
                    .backend
                    .request_device(adapter, id, &required_features, &required_limits),
            },
            Request::PopErrorScope { device } => Reply::PopErrorScope {
                result: self.backend.pop_error_scope(device),
            },
            Request::MapBuffer {
                buffer,
                for_writing,
                offset,
                size,
            } => Reply::MapBuffer {
                result: self
                    .backend
                    .map_buffer(buffer, for_writing, offset, size)
                    .map(Blob::from),
            },
            Request::OnSubmittedWorkDone { device } => {
                if let Err(error) = self.backend.wait_for_submitted_work(device) {
                    eprintln!("WebGPU: {}", error.0);
                }
                Reply::WorkDone
            }
            Request::CreateRenderPipeline { device, id, descriptor } => self
                .create_pipeline_in_scopes(device, |backend| {
                    backend.create_render_pipeline(device, id, &descriptor)
                }),
            Request::CreateComputePipeline { device, id, descriptor } => self
                .create_pipeline_in_scopes(device, |backend| {
                    backend.create_compute_pipeline(device, id, &descriptor)
                }),
            Request::GetCompilationInfo { module } => Reply::CompilationInfo {
                messages: self.backend.compilation_info(module).unwrap_or_else(|error| {
                    eprintln!("WebGPU: {}", error.0);
                    Vec::new()
                }),
            },
        }
    }

    /// Makes a pipeline for `create*PipelineAsync()`, whose promise is rejected with the validation or internal error
    /// that making it raises, rather than reporting that error to the device.
    fn create_pipeline_in_scopes(
        &mut self,
        device: ObjectId,
        create: impl FnOnce(&mut dyn Backend) -> crate::backend::BackendResult,
    ) -> Reply {
        let pushed = self.backend.push_error_scope(device, ErrorFilter::Validation).is_ok()
            && self.backend.push_error_scope(device, ErrorFilter::Internal).is_ok();
        let result = create(self.backend.as_mut());
        let (internal, validation) = if pushed {
            (
                self.backend.pop_error_scope(device).ok().flatten(),
                self.backend.pop_error_scope(device).ok().flatten(),
            )
        } else {
            (None, None)
        };
        let error = match result {
            Err(error) => Some(GpuError::Validation { message: error.0 }),
            Ok(()) => validation.or(internal),
        };
        Reply::Pipeline { error }
    }

    /// Starts reading back the texture the canvas drew into this frame, and returns whether it drew one.
    /// `take_frames` hands the pixels over once the GPU is done. The texture is then done with: the next frame draws
    /// into a new one, as getCurrentTexture() requires.
    pub fn present(&mut self, canvas: CanvasId) -> bool {
        let Some(state) = self.canvases.get_mut(&canvas) else {
            return false;
        };
        let Some(texture) = state.current_texture.take() else {
            return false;
        };
        let image = state.current_image.take();
        if self.backend.pending_readbacks(canvas) >= MAX_PENDING_FRAMES_PER_CANVAS {
            self.collect_finished_frames(true);
        }
        let started = match image {
            Some(image) => self.backend.finish_shared_frame(texture, canvas, image),
            None => self.backend.start_readback(texture, canvas),
        };
        if started.is_err()
            && let Some(image) = image
        {
            self.release_image(canvas, image);
        }
        self.retire_canvas_texture(texture);
        started
            .map_err(|error| eprintln!("WebGPU: presenting canvas {canvas}: {}", error.0))
            .is_ok()
    }

    /// The newest frame of each canvas that came back from the GPU since the last call. With `wait`, waits for every
    /// frame on its way.
    pub fn take_frames(&mut self, wait: bool) -> Vec<(CanvasId, PresentedFrame)> {
        self.collect_finished_frames(wait);
        self.finished_frames.drain().collect()
    }

    /// Whether frames are still on their way back from the GPU.
    #[must_use]
    pub fn has_pending_frames(&self) -> bool {
        self.canvases
            .keys()
            .any(|canvas| self.backend.pending_readbacks(*canvas) > 0)
    }

    fn collect_finished_frames(&mut self, wait: bool) {
        for (canvas, frame) in self.backend.finished_readbacks(wait) {
            match frame {
                // NB: Frames finish in order, so a later one replaces an earlier one, whose image goes back.
                Ok(frame) if self.canvases.contains_key(&canvas) => {
                    if let Some(PresentedFrame::SharedImage(image)) = self.finished_frames.insert(canvas, frame) {
                        self.release_image(canvas, image);
                    }
                }
                Ok(PresentedFrame::SharedImage(image)) => self.release_image(canvas, image),
                Ok(PresentedFrame::Pixels(_)) => {}
                Err(error) => eprintln!("WebGPU: presenting canvas {canvas}: {}", error.0),
            }
        }
    }

    fn execute(&mut self, command: Command, events: &mut Vec<Event>) {
        let device = device_of(&command);
        self.record_new_object(&command);
        let result = match command {
            Command::CreateShaderModule { device, id, code } => self.backend.create_shader_module(device, id, &code),
            Command::CreateRenderPipeline { device, id, descriptor } => {
                self.backend.create_render_pipeline(device, id, &descriptor)
            }
            Command::CreateTextureView {
                texture,
                id,
                descriptor,
            } => self.backend.create_texture_view(texture, id, &descriptor),
            Command::CreateTexture { device, id, descriptor } => self.backend.create_texture(device, id, &descriptor),
            Command::CreateBindGroupLayout { device, id, entries } => {
                self.backend.create_bind_group_layout(device, id, &entries)
            }
            Command::CreatePipelineLayout {
                device,
                id,
                bind_group_layouts,
            } => self.backend.create_pipeline_layout(device, id, &bind_group_layouts),
            Command::CreateBindGroup {
                device,
                id,
                layout,
                entries,
            } => self.backend.create_bind_group(device, id, layout, &entries),
            Command::CreateSampler { device, id, descriptor } => self.backend.create_sampler(device, id, &descriptor),
            Command::CreateComputePipeline { device, id, descriptor } => {
                self.backend.create_compute_pipeline(device, id, &descriptor)
            }
            Command::EncodeComputePass {
                encoder,
                timestamp_writes,
                ops,
            } => self
                .backend
                .encode_compute_pass(encoder, timestamp_writes.as_ref(), &ops),
            Command::GetBindGroupLayout { pipeline, index, id } => {
                self.backend.get_bind_group_layout(pipeline, index, id)
            }
            Command::CreateCommandEncoder { device, id } => self.backend.create_command_encoder(device, id),
            Command::EncodeRenderPass {
                encoder,
                color_attachments,
                depth_stencil,
                occlusion_query_set,
                timestamp_writes,
                ops,
            } => self.backend.encode_render_pass(
                encoder,
                &color_attachments,
                depth_stencil.as_ref(),
                occlusion_query_set,
                timestamp_writes.as_ref(),
                &ops,
            ),
            Command::CreateRenderBundle {
                device,
                id,
                descriptor,
                ops,
            } => self.backend.create_render_bundle(device, id, &descriptor, &ops),
            Command::CreateQuerySet {
                device,
                id,
                kind,
                count,
            } => self.backend.create_query_set(device, id, &kind, count),
            Command::ResolveQuerySet {
                encoder,
                query_set,
                first_query,
                query_count,
                destination,
                destination_offset,
            } => self.backend.resolve_query_set(
                encoder,
                query_set,
                first_query..first_query.saturating_add(query_count),
                destination,
                destination_offset,
            ),
            Command::FinishCommandEncoder { encoder, id } => self.backend.finish_command_encoder(encoder, id),
            Command::Submit {
                device,
                command_buffers,
            } => self.backend.submit(device, &command_buffers),
            Command::ConfigureCanvas {
                canvas,
                device,
                format,
                usage,
            } => {
                self.unconfigure(canvas);
                // https://gpuweb.github.io/gpuweb/#dom-gpucanvascontext-configure: the usage is validated as a
                // texture's would be, on the device.
                const TEXTURE_USAGE_BITS: u32 = 0x1f;
                if usage & !TEXTURE_USAGE_BITS != 0 {
                    Err(crate::backend::BackendError(format!(
                        "{usage:#x} is not a valid texture usage for a canvas"
                    )))
                } else {
                    self.canvases.insert(
                        canvas,
                        Canvas {
                            device,
                            format,
                            usage,
                            current_texture: None,
                            current_image: None,
                            shares_images: true,
                        },
                    );
                    Ok(())
                }
            }
            Command::UnconfigureCanvas { canvas } => {
                self.unconfigure(canvas);
                Ok(())
            }
            Command::GetCurrentTexture {
                canvas,
                id,
                width,
                height,
            } => match self.canvases.get_mut(&canvas) {
                Some(state) => {
                    let previous = state.current_texture.replace(id);
                    let previous_image = state.current_image.take();
                    let device = state.device;
                    let descriptor = CanvasTextureDescriptor {
                        width,
                        height,
                        format: state.format,
                        usage: state.usage,
                    };
                    if let Some(previous) = previous {
                        self.retire_canvas_texture(previous);
                    }
                    if let Some(image) = previous_image {
                        self.release_image(canvas, image);
                    }
                    if self.import_canvas_texture(canvas, id, descriptor) {
                        Ok(())
                    } else {
                        self.backend.create_canvas_texture(device, id, descriptor)
                    }
                }
                None => Err(crate::backend::BackendError(format!(
                    "canvas {canvas} is not configured"
                ))),
            },
            Command::Destroy { id } => {
                self.backend.destroy(id);
                if self.backend.is_destroyed_device(id) {
                    self.destroyed_devices.insert(id);
                    self.release_objects_of(id);
                }
                Ok(())
            }
            Command::Release { id } => {
                self.backend.release(id);
                self.object_devices.remove(&id);
                self.destroyed_devices.remove(&id);
                Ok(())
            }
            Command::PushErrorScope { device, filter } => self.backend.push_error_scope(device, filter),
            Command::CreateBuffer {
                device,
                id,
                size,
                usage,
                mapped_at_creation,
            } => self.backend.create_buffer(device, id, size, usage, mapped_at_creation),
            Command::WriteBuffer {
                device,
                buffer,
                offset,
                data,
            } => self.backend.write_buffer(device, buffer, offset, &data.0),
            Command::CopyBufferToBuffer {
                encoder,
                source,
                source_offset,
                destination,
                destination_offset,
                size,
            } => self.backend.copy_buffer_to_buffer(
                encoder,
                source,
                source_offset,
                destination,
                destination_offset,
                size,
            ),
            Command::UnmapBuffer { buffer, offset, data } => self.backend.unmap_buffer(buffer, offset, &data.0),
            Command::ClearBuffer {
                encoder,
                buffer,
                offset,
                size,
            } => self.backend.clear_buffer(encoder, buffer, offset, size),
            Command::CopyBufferToTexture {
                encoder,
                buffer,
                layout,
                texture,
                size,
            } => self
                .backend
                .copy_buffer_to_texture(encoder, buffer, &layout, &texture, size),
            Command::CopyTextureToBuffer {
                encoder,
                texture,
                buffer,
                layout,
                size,
            } => self
                .backend
                .copy_texture_to_buffer(encoder, &texture, buffer, &layout, size),
            Command::CopyTextureToTexture {
                encoder,
                source,
                destination,
                size,
            } => self
                .backend
                .copy_texture_to_texture(encoder, &source, &destination, size),
            Command::WriteTexture {
                device,
                texture,
                data,
                layout,
                size,
            } => self.backend.write_texture(device, &texture, &data.0, &layout, size),
            Command::CopyExternalImageToTexture {
                device,
                texture,
                width,
                height,
                rgba8,
            } => self
                .backend
                .copy_external_image_to_texture(device, &texture, width, height, &rgba8.0),
            Command::Request { request, body } => {
                let reply = self.answer(body);
                events.push(Event::Reply { request, reply });
                Ok(())
            }
        };
        if let Err(error) = result {
            eprintln!("WebGPU: {}", error.0);
            // FIXME: These are Ladybird's own errors, such as an id that names nothing, which wgpu's error scopes do
            //        not see. Report them through the device's scopes too.
            if let Some(device) = device
                && !self.destroyed_devices.contains(&device)
            {
                events.push(Event::UncapturedError {
                    device,
                    error: GpuError::Validation { message: error.0 },
                });
            }
        }
    }

    fn unconfigure(&mut self, canvas: CanvasId) {
        let Some(state) = self.canvases.remove(&canvas) else {
            return;
        };
        if let Some(texture) = state.current_texture {
            self.retire_canvas_texture(texture);
        }
        if let Some(image) = state.current_image {
            self.release_image(canvas, image);
        }
    }

    /// A canvas texture belongs to the canvas, not to script: once presented or replaced, nothing may draw into it.
    /// Its GPUTexture may outlive it, and releases an id that is gone by then, which is harmless.
    fn retire_canvas_texture(&mut self, texture: ObjectId) {
        self.backend.destroy(texture);
        self.backend.release(texture);
    }
}

/// The device a command works on, which its errors are reported to.
fn device_of(command: &Command) -> Option<ObjectId> {
    match *command {
        Command::CreateShaderModule { device, .. }
        | Command::CreateRenderPipeline { device, .. }
        | Command::CreateCommandEncoder { device, .. }
        | Command::Submit { device, .. }
        | Command::ConfigureCanvas { device, .. }
        | Command::PushErrorScope { device, .. }
        | Command::CreateBuffer { device, .. }
        | Command::WriteBuffer { device, .. }
        | Command::CreateTexture { device, .. }
        | Command::CreateBindGroupLayout { device, .. }
        | Command::CreatePipelineLayout { device, .. }
        | Command::CreateBindGroup { device, .. }
        | Command::CreateSampler { device, .. }
        | Command::CreateComputePipeline { device, .. }
        | Command::CreateQuerySet { device, .. }
        | Command::CreateRenderBundle { device, .. }
        | Command::WriteTexture { device, .. }
        | Command::CopyExternalImageToTexture { device, .. } => Some(device),
        _ => None,
    }
}
