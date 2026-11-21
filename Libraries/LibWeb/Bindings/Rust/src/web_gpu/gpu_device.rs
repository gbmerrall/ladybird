/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpudevice

use core::cell::Cell;
use core::ffi::c_void;
use core::ptr::NonNull;

use ak::Utf16String;

use libjs_rust::api::{Gc, Realm, Trace, Value, Vm};
use webgpu_wire::{
    BlendComponent, BlendState, Capabilities, ColorTargetState, Command, DepthStencilState, DeviceLostReason,
    FragmentState, GpuError, MultisampleState, ObjectId, PrimitiveState, PrimitiveTopology, ProgrammableStage,
    RenderPipelineDescriptor, Reply, Request, StencilFaceState, VertexAttribute, VertexBufferLayout,
};

use crate::bindings::conversions::ToJs;
use crate::bindings::conversions::UsvString;
use crate::bindings::event_target::{CppEventTarget, GetTheParent};
use crate::bindings::interface::{Ref, Wrappable, allocate, with_temporary_execution_context, wrap};
use crate::bindings::promise::{
    TaskSource, WebIdlPromise, create_promise, queue_a_global_task, reject_promise_with_exception, resolve_promise,
};
use crate::bindings::{CallContext, Exception, WebIdlResult};
use crate::dom::event::EventInit;
use crate::web_gpu::Label;
use crate::web_gpu::client::{self, RequestSubject};
use crate::web_gpu::gpu::{GPUObjectDescriptorBase, GPUTextureFormat};
use crate::web_gpu::gpu_adapter_info::GPUAdapterInfo;
use crate::web_gpu::gpu_bind_group::{GPUBindGroup, GPUBindGroupDescriptor};
use crate::web_gpu::gpu_bind_group_layout::{GPUBindGroupLayout, GPUBindGroupLayoutDescriptor};
use crate::web_gpu::gpu_buffer::{GPUBuffer, GPUBufferDescriptor};
use crate::web_gpu::gpu_command_encoder::GPUCommandEncoder;
use crate::web_gpu::gpu_compute_pipeline::{GPUComputePipeline, GPUComputePipelineDescriptor};
use crate::web_gpu::gpu_device_lost_info::GPUDeviceLostInfo;
use crate::web_gpu::gpu_error::{AnyGPUError, GPUErrorFilter};
use crate::web_gpu::gpu_pipeline_layout::{GPUPipelineLayout, GPUPipelineLayoutDescriptor};
use crate::web_gpu::gpu_query_set::{GPUQuerySet, GPUQuerySetDescriptor};
use crate::web_gpu::gpu_queue::GPUQueue;
use crate::web_gpu::gpu_render_bundle_encoder::{GPURenderBundleEncoder, GPURenderBundleEncoderDescriptor};
use crate::web_gpu::gpu_render_pipeline::GPURenderPipeline;
use crate::web_gpu::gpu_sampler::{GPUSampler, GPUSamplerDescriptor};
use crate::web_gpu::gpu_shader_module::GPUShaderModule;
use crate::web_gpu::gpu_supported_features::GPUSupportedFeatures;
use crate::web_gpu::gpu_supported_limits::GPUSupportedLimits;
use crate::web_gpu::gpu_texture::{GPUTexture, GPUTextureDescriptor};
use crate::web_gpu::gpu_uncaptured_error_event::GPUUncapturedErrorEvent;
use crate::web_gpu::labelled;

pub(crate) struct GPUShaderModuleDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) code: UsvString,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPUAutoLayoutMode {
    Auto,
}

/// The union (GPUPipelineLayout or GPUAutoLayoutMode) of WebGPU/GPUDevice.idl.
pub(crate) enum GPUPipelineLayoutOrGPUAutoLayoutMode {
    GPUPipelineLayout(Ref<GPUPipelineLayout>),
    GPUAutoLayoutMode(GPUAutoLayoutMode),
}

pub(crate) struct GPUPipelineDescriptorBase {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) layout: GPUPipelineLayoutOrGPUAutoLayoutMode,
}

pub(crate) struct GPUProgrammableStage {
    pub(crate) module: Ref<GPUShaderModule>,
    pub(crate) entry_point: Option<UsvString>,
    pub(crate) constants: Option<Vec<(UsvString, f64)>>,
}

impl GPUPipelineDescriptorBase {
    /// The id of the explicit layout, or None for "auto".
    pub(crate) fn layout_id(&self) -> Option<ObjectId> {
        match &self.layout {
            GPUPipelineLayoutOrGPUAutoLayoutMode::GPUPipelineLayout(layout) => Some(layout.id),
            GPUPipelineLayoutOrGPUAutoLayoutMode::GPUAutoLayoutMode(_) => None,
        }
    }
}

impl GPUProgrammableStage {
    pub(crate) fn to_wire(&self) -> ProgrammableStage {
        ProgrammableStage {
            module: self.module.id,
            entry_point: self.entry_point.as_ref().map(UsvString::to_utf8),
            constants: self
                .constants
                .iter()
                .flatten()
                .map(|(name, value)| (name.to_utf8(), *value))
                .collect(),
        }
    }
}

pub(crate) struct GPUVertexAttribute {
    pub(crate) format: GPUVertexFormat,
    pub(crate) offset: u64,
    pub(crate) shader_location: u32,
}

pub(crate) struct GPUVertexBufferLayout {
    pub(crate) array_stride: u64,
    pub(crate) step_mode: GPUVertexStepMode,
    pub(crate) attributes: Vec<GPUVertexAttribute>,
}

pub(crate) struct GPUVertexState {
    pub(crate) parent: GPUProgrammableStage,
    pub(crate) buffers: Vec<Option<GPUVertexBufferLayout>>,
}

pub(crate) struct GPUBlendComponent {
    pub(crate) operation: GPUBlendOperation,
    pub(crate) src_factor: GPUBlendFactor,
    pub(crate) dst_factor: GPUBlendFactor,
}

impl GPUBlendComponent {
    fn to_wire(&self) -> BlendComponent {
        BlendComponent {
            operation: self.operation.name().to_owned(),
            src_factor: self.src_factor.name().to_owned(),
            dst_factor: self.dst_factor.name().to_owned(),
        }
    }
}

pub(crate) struct GPUBlendState {
    pub(crate) color: GPUBlendComponent,
    pub(crate) alpha: GPUBlendComponent,
}

pub(crate) struct GPUColorTargetState {
    pub(crate) format: GPUTextureFormat,
    pub(crate) blend: Option<GPUBlendState>,
    pub(crate) write_mask: u32,
}

pub(crate) struct GPUFragmentState {
    pub(crate) parent: GPUProgrammableStage,
    pub(crate) targets: Vec<Option<GPUColorTargetState>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPUPrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

pub(crate) struct GPUPrimitiveState {
    pub(crate) topology: GPUPrimitiveTopology,
    pub(crate) strip_index_format: Option<GPUIndexFormat>,
    pub(crate) front_face: GPUFrontFace,
    pub(crate) cull_mode: GPUCullMode,
    pub(crate) unclipped_depth: bool,
}

pub(crate) struct GPUStencilFaceState {
    pub(crate) compare: GPUCompareFunction,
    pub(crate) fail_op: GPUStencilOperation,
    pub(crate) depth_fail_op: GPUStencilOperation,
    pub(crate) pass_op: GPUStencilOperation,
}

impl GPUStencilFaceState {
    fn to_wire(&self) -> StencilFaceState {
        StencilFaceState {
            compare: self.compare.name().to_owned(),
            fail_op: self.fail_op.name().to_owned(),
            depth_fail_op: self.depth_fail_op.name().to_owned(),
            pass_op: self.pass_op.name().to_owned(),
        }
    }
}

pub(crate) struct GPUDepthStencilState {
    pub(crate) format: GPUTextureFormat,
    pub(crate) depth_write_enabled: Option<bool>,
    pub(crate) depth_compare: Option<GPUCompareFunction>,
    pub(crate) stencil_front: GPUStencilFaceState,
    pub(crate) stencil_back: GPUStencilFaceState,
    pub(crate) stencil_read_mask: u32,
    pub(crate) stencil_write_mask: u32,
    pub(crate) depth_bias: i32,
    pub(crate) depth_bias_slope_scale: f32,
    pub(crate) depth_bias_clamp: f32,
}

pub(crate) struct GPUMultisampleState {
    pub(crate) count: u32,
    pub(crate) mask: u32,
    pub(crate) alpha_to_coverage_enabled: bool,
}

pub(crate) struct GPURenderPipelineDescriptor {
    pub(crate) parent: GPUPipelineDescriptorBase,
    pub(crate) vertex: GPUVertexState,
    pub(crate) primitive: GPUPrimitiveState,
    pub(crate) depth_stencil: Option<GPUDepthStencilState>,
    pub(crate) multisample: GPUMultisampleState,
    pub(crate) fragment: Option<GPUFragmentState>,
}

impl GPURenderPipelineDescriptor {
    fn to_wire(&self) -> RenderPipelineDescriptor {
        let primitive = &self.primitive;
        RenderPipelineDescriptor {
            layout: self.parent.layout_id(),
            vertex: self.vertex.parent.to_wire(),
            vertex_buffers: self
                .vertex
                .buffers
                .iter()
                .map(|buffer| {
                    buffer.as_ref().map(|buffer| VertexBufferLayout {
                        array_stride: buffer.array_stride,
                        step_mode: buffer.step_mode.name().to_owned(),
                        attributes: buffer
                            .attributes
                            .iter()
                            .map(|attribute| VertexAttribute {
                                format: attribute.format.name().to_owned(),
                                offset: attribute.offset,
                                shader_location: attribute.shader_location,
                            })
                            .collect(),
                    })
                })
                .collect(),
            fragment: self.fragment.as_ref().map(|fragment| FragmentState {
                stage: fragment.parent.to_wire(),
                targets: fragment
                    .targets
                    .iter()
                    .map(|target| {
                        target.as_ref().map(|target| ColorTargetState {
                            format: target.format.name().to_owned(),
                            blend: target.blend.as_ref().map(|blend| BlendState {
                                color: blend.color.to_wire(),
                                alpha: blend.alpha.to_wire(),
                            }),
                            write_mask: target.write_mask,
                        })
                    })
                    .collect(),
            }),
            primitive: PrimitiveState {
                topology: match primitive.topology {
                    GPUPrimitiveTopology::PointList => PrimitiveTopology::PointList,
                    GPUPrimitiveTopology::LineList => PrimitiveTopology::LineList,
                    GPUPrimitiveTopology::LineStrip => PrimitiveTopology::LineStrip,
                    GPUPrimitiveTopology::TriangleList => PrimitiveTopology::TriangleList,
                    GPUPrimitiveTopology::TriangleStrip => PrimitiveTopology::TriangleStrip,
                },
                strip_index_format: primitive.strip_index_format.map(|format| format.name().to_owned()),
                front_face: primitive.front_face.name().to_owned(),
                cull_mode: primitive.cull_mode.name().to_owned(),
                unclipped_depth: primitive.unclipped_depth,
            },
            depth_stencil: self.depth_stencil.as_ref().map(|state| DepthStencilState {
                format: state.format.name().to_owned(),
                // FIXME: The spec requires depthWriteEnabled and depthCompare for a format with depth, as a
                //        validation error.
                depth_write_enabled: state.depth_write_enabled.unwrap_or(false),
                depth_compare: state.depth_compare.map(|compare| compare.name().to_owned()),
                stencil_front: state.stencil_front.to_wire(),
                stencil_back: state.stencil_back.to_wire(),
                stencil_read_mask: state.stencil_read_mask,
                stencil_write_mask: state.stencil_write_mask,
                depth_bias: state.depth_bias,
                depth_bias_slope_scale: state.depth_bias_slope_scale,
                depth_bias_clamp: state.depth_bias_clamp,
            }),
            multisample: MultisampleState {
                count: self.multisample.count,
                mask: self.multisample.mask,
                alpha_to_coverage_enabled: self.multisample.alpha_to_coverage_enabled,
            },
        }
    }
}

string_enum!(GPUVertexFormat, "gpuvertexformat" {
    Uint8 => "uint8",
    Uint8x2 => "uint8x2",
    Uint8x4 => "uint8x4",
    Sint8 => "sint8",
    Sint8x2 => "sint8x2",
    Sint8x4 => "sint8x4",
    Unorm8 => "unorm8",
    Unorm8x2 => "unorm8x2",
    Unorm8x4 => "unorm8x4",
    Snorm8 => "snorm8",
    Snorm8x2 => "snorm8x2",
    Snorm8x4 => "snorm8x4",
    Uint16 => "uint16",
    Uint16x2 => "uint16x2",
    Uint16x4 => "uint16x4",
    Sint16 => "sint16",
    Sint16x2 => "sint16x2",
    Sint16x4 => "sint16x4",
    Unorm16 => "unorm16",
    Unorm16x2 => "unorm16x2",
    Unorm16x4 => "unorm16x4",
    Snorm16 => "snorm16",
    Snorm16x2 => "snorm16x2",
    Snorm16x4 => "snorm16x4",
    Float16 => "float16",
    Float16x2 => "float16x2",
    Float16x4 => "float16x4",
    Float32 => "float32",
    Float32x2 => "float32x2",
    Float32x3 => "float32x3",
    Float32x4 => "float32x4",
    Uint32 => "uint32",
    Uint32x2 => "uint32x2",
    Uint32x3 => "uint32x3",
    Uint32x4 => "uint32x4",
    Sint32 => "sint32",
    Sint32x2 => "sint32x2",
    Sint32x3 => "sint32x3",
    Sint32x4 => "sint32x4",
    Unorm1010102 => "unorm10-10-10-2",
    Unorm8x4Bgra => "unorm8x4-bgra",
});

string_enum!(GPUVertexStepMode, "gpuvertexstepmode" {
    Vertex => "vertex",
    Instance => "instance",
});

string_enum!(GPUIndexFormat, "gpuindexformat" {
    Uint16 => "uint16",
    Uint32 => "uint32",
});

string_enum!(GPUFrontFace, "gpufrontface" {
    Ccw => "ccw",
    Cw => "cw",
});

string_enum!(GPUCullMode, "gpucullmode" {
    None => "none",
    Front => "front",
    Back => "back",
});

string_enum!(GPUBlendFactor, "gpublendfactor" {
    Zero => "zero",
    One => "one",
    Src => "src",
    OneMinusSrc => "one-minus-src",
    SrcAlpha => "src-alpha",
    OneMinusSrcAlpha => "one-minus-src-alpha",
    Dst => "dst",
    OneMinusDst => "one-minus-dst",
    DstAlpha => "dst-alpha",
    OneMinusDstAlpha => "one-minus-dst-alpha",
    SrcAlphaSaturated => "src-alpha-saturated",
    Constant => "constant",
    OneMinusConstant => "one-minus-constant",
    Src1 => "src1",
    OneMinusSrc1 => "one-minus-src1",
    Src1Alpha => "src1-alpha",
    OneMinusSrc1Alpha => "one-minus-src1-alpha",
});

string_enum!(GPUBlendOperation, "gpublendoperation" {
    Add => "add",
    Subtract => "subtract",
    ReverseSubtract => "reverse-subtract",
    Min => "min",
    Max => "max",
});

string_enum!(GPUCompareFunction, "gpucomparefunction" {
    Never => "never",
    Less => "less",
    Equal => "equal",
    LessEqual => "less-equal",
    Greater => "greater",
    NotEqual => "not-equal",
    GreaterEqual => "greater-equal",
    Always => "always",
});

string_enum!(GPUStencilOperation, "gpustenciloperation" {
    Keep => "keep",
    Zero => "zero",
    Replace => "replace",
    Invert => "invert",
    IncrementClamp => "increment-clamp",
    DecrementClamp => "decrement-clamp",
    IncrementWrap => "increment-wrap",
    DecrementWrap => "decrement-wrap",
});

pub(crate) struct GPUCommandEncoderDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUDevice {
    event_target: CppEventTarget,
    #[gc(untraced)]
    pub(crate) label: Label,
    realm: Gc<Realm>,
    queue: Ref<GPUQueue>,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
    features: Ref<GPUSupportedFeatures>,
    pub(crate) limits: Ref<GPUSupportedLimits>,
    adapter_info: Ref<GPUAdapterInfo>,
    /// https://gpuweb.github.io/gpuweb/#dom-gpudevice-lost
    lost: WebIdlPromise,
    #[gc(untraced)]
    is_lost: Cell<bool>,
}

impl Wrappable for GPUDevice {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::unregister_device(object.id);
        client::release_later(object.id);
    }

    fn cpp_parent(&self) -> Option<NonNull<c_void>> {
        self.event_target.as_cpp_parent()
    }
}

impl GetTheParent for GPUDevice {
    fn get_the_parent(&self) -> Option<CppEventTarget> {
        None
    }
}

impl GPUDevice {
    pub(crate) fn create(
        vm: &Vm,
        realm: Gc<Realm>,
        id: ObjectId,
        capabilities: Capabilities,
        adapter_info: Ref<GPUAdapterInfo>,
    ) -> Ref<Self> {
        let queue = GPUQueue::create(vm, realm, id);
        let features = GPUSupportedFeatures::create(vm, realm, capabilities.features);
        let limits = GPUSupportedLimits::create(vm, realm, capabilities.limits);
        let lost = create_promise(vm, realm);
        let device = allocate(vm, |_| Self {
            label: Label::default(),
            event_target: CppEventTarget::new(),
            realm,
            queue,
            id,
            features,
            limits,
            adapter_info,
            lost,
            is_lost: Cell::new(false),
        });
        device.event_target.set_owner(device);
        device.queue.set_device(device);
        client::register_device(device);
        device
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-features
    pub(crate) fn features(&self, _: &mut CallContext) -> WebIdlResult<Ref<GPUSupportedFeatures>> {
        Ok(self.features)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-limits
    pub(crate) fn limits(&self, _: &mut CallContext) -> WebIdlResult<Ref<GPUSupportedLimits>> {
        Ok(self.limits)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-adapterinfo
    pub(crate) fn adapter_info(&self, _: &mut CallContext) -> WebIdlResult<Ref<GPUAdapterInfo>> {
        Ok(self.adapter_info)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-queue
    pub(crate) fn queue(&self, _: &mut CallContext) -> WebIdlResult<Ref<GPUQueue>> {
        Ok(self.queue)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-destroy
    pub(crate) fn destroy(&self, _: &mut CallContext) -> WebIdlResult<()> {
        // FIXME: Resolve device.lost, as the spec's "lose the device" does.
        client::send(self.realm, &Command::Destroy { id: self.id });
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-lost
    pub(crate) fn lost(&self, _: &mut CallContext) -> WebIdlResult<WebIdlPromise> {
        Ok(self.lost)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-pusherrorscope
    pub(crate) fn push_error_scope(&self, _: &mut CallContext, filter: GPUErrorFilter) -> WebIdlResult<()> {
        // FIXME: A lost device keeps an error scope stack of its own, which the spec still pushes onto.
        client::send(
            self.realm,
            &Command::PushErrorScope {
                device: self.id,
                filter: filter.into(),
            },
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-poperrorscope
    pub(crate) fn pop_error_scope(&self, cx: &mut CallContext, promise: WebIdlPromise) -> WebIdlResult<()> {
        fn steps(vm: &Vm, request: &client::PendingRequest, reply: Option<Reply>) {
            match reply {
                Some(Reply::PopErrorScope { result: Ok(error) }) => {
                    let value = error.map_or(Value::NULL, |error| {
                        let cx = CallContext::new(vm, request.realm);
                        AnyGPUError::create(vm, request.realm, error)
                            .to_js(&cx)
                            .expect("a GPUError converts to a JS value")
                    });
                    resolve_promise(vm, request.promise, value);
                }
                Some(Reply::PopErrorScope { result: Err(message) }) => reject_promise_with_exception(
                    vm,
                    request.promise,
                    Exception::Dom("OperationError", Utf16String::from_utf8(&message)),
                ),
                // NB: Without a compositor the device is lost, and a lost device's scopes caught nothing.
                _ => resolve_promise(vm, request.promise, Value::NULL),
            }
        }
        client::request(
            cx.vm(),
            self.realm,
            promise,
            self.id,
            Request::PopErrorScope { device: self.id },
            steps,
        );
        Ok(())
    }

    /// Fires `uncapturederror` at the device in a task, for an error that no scope caught:
    /// https://gpuweb.github.io/gpuweb/#abstract-opdef-dispatch-error
    pub(crate) fn dispatch_uncaptured_error(device: Ref<Self>, vm: &Vm, error: GpuError) {
        if device.is_lost.get() {
            return;
        }
        let realm = device.realm;
        let error = AnyGPUError::create(vm, realm, error);
        queue_a_global_task(
            vm,
            realm.global_object(),
            TaskSource::Unspecified,
            (device, error),
            |vm, (device, error)| {
                let event_init = EventInit {
                    bubbles: false,
                    cancelable: true,
                    composed: false,
                };
                let event = GPUUncapturedErrorEvent::create(
                    vm,
                    device.realm,
                    &Utf16String::from_utf8("uncapturederror"),
                    &event_init,
                    error,
                );
                device.event_target.dispatch_event(event.event());
            },
        );
    }

    /// Resolves `lost`, once: https://gpuweb.github.io/gpuweb/#lose-the-device
    pub(crate) fn lose(&self, vm: &Vm, reason: DeviceLostReason, message: &str) {
        if self.is_lost.replace(true) {
            return;
        }
        let realm = self.realm;
        let info = GPUDeviceLostInfo::create(vm, realm, reason, message);
        let promise = self.lost;
        queue_a_global_task(
            vm,
            realm.global_object(),
            TaskSource::Unspecified,
            (realm, promise, info),
            |vm, (realm, promise, info)| {
                with_temporary_execution_context(realm, || {
                    let value = Value::from_object(wrap(&CallContext::new(vm, realm), info));
                    resolve_promise(vm, promise, value);
                });
            },
        );
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createbuffer
    pub(crate) fn create_buffer(
        &self,
        cx: &mut CallContext,
        descriptor: GPUBufferDescriptor,
    ) -> WebIdlResult<Ref<GPUBuffer>> {
        GPUBuffer::create(cx.vm(), self.realm, self.id, &descriptor)
            .map(|buffer| labelled(buffer, &descriptor.parent.label))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createshadermodule
    pub(crate) fn create_shader_module(
        &self,
        cx: &mut CallContext,
        descriptor: GPUShaderModuleDescriptor,
    ) -> WebIdlResult<Ref<GPUShaderModule>> {
        let id = client::allocate_object_id();
        client::send(
            self.realm,
            &Command::CreateShaderModule {
                device: self.id,
                id,
                code: descriptor.code.to_utf8(),
            },
        );
        Ok(labelled(
            GPUShaderModule::create(cx.vm(), self.realm, id),
            &descriptor.parent.label,
        ))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createrenderpipeline
    pub(crate) fn create_render_pipeline(
        &self,
        cx: &mut CallContext,
        descriptor: GPURenderPipelineDescriptor,
    ) -> WebIdlResult<Ref<GPURenderPipeline>> {
        let id = client::allocate_object_id();
        client::send(
            self.realm,
            &Command::CreateRenderPipeline {
                device: self.id,
                id,
                descriptor: Box::new(descriptor.to_wire()),
            },
        );
        Ok(labelled(
            GPURenderPipeline::create(cx.vm(), self.realm, id),
            &descriptor.parent.parent.label,
        ))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createcomputepipeline
    pub(crate) fn create_compute_pipeline(
        &self,
        cx: &mut CallContext,
        descriptor: GPUComputePipelineDescriptor,
    ) -> WebIdlResult<Ref<GPUComputePipeline>> {
        let id = client::allocate_object_id();
        client::send(
            self.realm,
            &Command::CreateComputePipeline {
                device: self.id,
                id,
                descriptor: descriptor.to_wire(),
            },
        );
        Ok(labelled(
            GPUComputePipeline::create(cx.vm(), self.realm, id),
            &descriptor.parent.parent.label,
        ))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createcomputepipelineasync
    pub(crate) fn create_compute_pipeline_async(
        &self,
        cx: &mut CallContext,
        promise: WebIdlPromise,
        descriptor: GPUComputePipelineDescriptor,
    ) -> WebIdlResult<()> {
        let id = client::allocate_object_id();
        let pipeline = labelled(
            GPUComputePipeline::create(cx.vm(), self.realm, id),
            &descriptor.parent.parent.label,
        );
        let body = Request::CreateComputePipeline {
            device: self.id,
            id,
            descriptor: descriptor.to_wire(),
        };
        client::request_about(
            cx.vm(),
            self.realm,
            promise,
            id,
            body,
            settle_pipeline_promise,
            RequestSubject::ComputePipeline(pipeline),
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createrenderpipelineasync
    pub(crate) fn create_render_pipeline_async(
        &self,
        cx: &mut CallContext,
        promise: WebIdlPromise,
        descriptor: GPURenderPipelineDescriptor,
    ) -> WebIdlResult<()> {
        let id = client::allocate_object_id();
        let pipeline = labelled(
            GPURenderPipeline::create(cx.vm(), self.realm, id),
            &descriptor.parent.parent.label,
        );
        let body = Request::CreateRenderPipeline {
            device: self.id,
            id,
            descriptor: Box::new(descriptor.to_wire()),
        };
        client::request_about(
            cx.vm(),
            self.realm,
            promise,
            id,
            body,
            settle_pipeline_promise,
            RequestSubject::RenderPipeline(pipeline),
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createrenderbundleencoder
    pub(crate) fn create_render_bundle_encoder(
        &self,
        cx: &mut CallContext,
        descriptor: GPURenderBundleEncoderDescriptor,
    ) -> WebIdlResult<Ref<GPURenderBundleEncoder>> {
        Ok(labelled(
            GPURenderBundleEncoder::create(cx.vm(), self.realm, self.id, &descriptor),
            &descriptor.parent.parent.label,
        ))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createqueryset
    pub(crate) fn create_query_set(
        &self,
        cx: &mut CallContext,
        descriptor: GPUQuerySetDescriptor,
    ) -> WebIdlResult<Ref<GPUQuerySet>> {
        Ok(labelled(
            GPUQuerySet::create(cx.vm(), self.realm, self.id, &descriptor),
            &descriptor.parent.label,
        ))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createtexture
    pub(crate) fn create_texture(
        &self,
        cx: &mut CallContext,
        descriptor: GPUTextureDescriptor,
    ) -> WebIdlResult<Ref<GPUTexture>> {
        GPUTexture::create(cx.vm(), self.realm, self.id, &descriptor)
            .map(|texture| labelled(texture, &descriptor.parent.label))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createsampler
    pub(crate) fn create_sampler(
        &self,
        cx: &mut CallContext,
        descriptor: GPUSamplerDescriptor,
    ) -> WebIdlResult<Ref<GPUSampler>> {
        Ok(labelled(
            GPUSampler::create(cx.vm(), self.realm, self.id, &descriptor),
            &descriptor.parent.label,
        ))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createbindgrouplayout
    pub(crate) fn create_bind_group_layout(
        &self,
        cx: &mut CallContext,
        descriptor: GPUBindGroupLayoutDescriptor,
    ) -> WebIdlResult<Ref<GPUBindGroupLayout>> {
        GPUBindGroupLayout::create(cx.vm(), self.realm, self.id, &descriptor)
            .map(|layout| labelled(layout, &descriptor.parent.label))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createpipelinelayout
    pub(crate) fn create_pipeline_layout(
        &self,
        cx: &mut CallContext,
        descriptor: GPUPipelineLayoutDescriptor<'_>,
    ) -> WebIdlResult<Ref<GPUPipelineLayout>> {
        Ok(labelled(
            GPUPipelineLayout::create(cx.vm(), self.realm, self.id, &descriptor),
            &descriptor.parent.label,
        ))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createbindgroup
    pub(crate) fn create_bind_group(
        &self,
        cx: &mut CallContext,
        descriptor: GPUBindGroupDescriptor<'_>,
    ) -> WebIdlResult<Ref<GPUBindGroup>> {
        Ok(labelled(
            GPUBindGroup::create(cx.vm(), self.realm, self.id, &descriptor),
            &descriptor.parent.label,
        ))
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createcommandencoder
    pub(crate) fn create_command_encoder(
        &self,
        cx: &mut CallContext,
        descriptor: GPUCommandEncoderDescriptor,
    ) -> WebIdlResult<Ref<GPUCommandEncoder>> {
        let id = client::allocate_object_id();
        client::send(self.realm, &Command::CreateCommandEncoder { device: self.id, id });
        Ok(labelled(
            GPUCommandEncoder::create(cx.vm(), self.realm, id),
            &descriptor.parent.label,
        ))
    }
}

/// Resolves the promise of `create*PipelineAsync()` with the pipeline, or rejects it with the error that making the
/// pipeline raised.
fn settle_pipeline_promise(vm: &Vm, request: &client::PendingRequest, reply: Option<Reply>) {
    match reply {
        Some(Reply::Pipeline { error: None }) => {
            let cx = CallContext::new(vm, request.realm);
            let pipeline = match request.subject {
                RequestSubject::RenderPipeline(pipeline) => wrap(&cx, pipeline),
                RequestSubject::ComputePipeline(pipeline) => wrap(&cx, pipeline),
                _ => unreachable!("a pipeline request carries its pipeline"),
            };
            resolve_promise(vm, request.promise, Value::from_object(pipeline));
        }
        // FIXME: Reject with a GPUPipelineError whose reason is "validation" or "internal".
        Some(Reply::Pipeline { error: Some(error) }) => reject_promise_with_exception(
            vm,
            request.promise,
            Exception::Dom("OperationError", Utf16String::from_utf8(error.message())),
        ),
        _ => reject_promise_with_exception(
            vm,
            request.promise,
            Exception::Dom("OperationError", Utf16String::from_utf8("The device is lost")),
        ),
    }
}

label_attribute!(GPUDevice);
