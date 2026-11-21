/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! What crosses the process boundary for WebGPU. WebContent sends commands, some of which are requests that
//! the compositor answers. The compositor sends events back: those answers, errors nothing caught, and lost
//! devices. Both ends are Rust, so this crate is the one definition of the protocol; the encoding is a plain
//! little-endian byte stream.
//!
//! Every object is named by an id that WebContent allocates, so recording never waits for the compositor.
//! Requests travel in the same stream as other commands, so they see the effects of every command before them,
//! as popping an error scope must.
//!
//! Values of WebGPU's enumerations, such as texture formats and blend factors, travel as their names in the spec.
//! The compositor's backend maps them, and an unknown name is a validation error there.

use std::fmt;

pub type ObjectId = u64;
pub type CanvasId = u64;
pub type RequestId = u64;

/// The formats a canvas can be configured with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureFormat {
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    Bgra8UnormSrgb,
    Rgba16Float,
}

impl TextureFormat {
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "rgba8unorm" => Self::Rgba8Unorm,
            "rgba8unorm-srgb" => Self::Rgba8UnormSrgb,
            "bgra8unorm" => Self::Bgra8Unorm,
            "bgra8unorm-srgb" => Self::Bgra8UnormSrgb,
            "rgba16float" => Self::Rgba16Float,
            _ => return None,
        })
    }

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Rgba8Unorm => "rgba8unorm",
            Self::Rgba8UnormSrgb => "rgba8unorm-srgb",
            Self::Bgra8Unorm => "bgra8unorm",
            Self::Bgra8UnormSrgb => "bgra8unorm-srgb",
            Self::Rgba16Float => "rgba16float",
        }
    }

    #[must_use]
    pub fn is_bgra(self) -> bool {
        matches!(self, Self::Bgra8Unorm | Self::Bgra8UnormSrgb)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerPreference {
    Default,
    LowPower,
    HighPerformance,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoadOp {
    Clear([f64; 4]),
    Load,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreOp {
    Store,
    Discard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorFilter {
    Validation,
    OutOfMemory,
    Internal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompilationMessageType {
    Error,
    Warning,
    Info,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceLostReason {
    Unknown,
    Destroyed,
}

/// Declares a struct whose fields are encoded in order.
macro_rules! wire_struct {
    ($(#[$meta:meta])* $name:ident { $($(#[$field_meta:meta])* $field:ident: $type:ty,)* }) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name {
            $($(#[$field_meta])* pub $field: $type,)*
        }

        impl Wire for $name {
            fn encode(&self, w: &mut Writer) {
                $(self.$field.encode(w);)*
            }
            fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
                Ok(Self { $($field: Wire::decode(r)?,)* })
            }
        }
    };
}

/// Declares an enum of struct and unit variants, encoded as a tag and then the fields in order.
macro_rules! wire_enum {
    ($(#[$meta:meta])* $name:ident { $($(#[$variant_meta:meta])* $tag:literal => $variant:ident $({ $($field:ident: $type:ty,)* })?,)* }) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq)]
        pub enum $name {
            $($(#[$variant_meta])* $variant $({ $($field: $type,)* })?,)*
        }

        impl Wire for $name {
            #[allow(unused_variables, reason = "a unit variant encodes no fields")]
            fn encode(&self, w: &mut Writer) {
                match self {
                    $(Self::$variant $({ $($field,)* })? => {
                        w.u8($tag);
                        $($($field.encode(w);)*)?
                    })*
                }
            }
            fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
                Ok(match r.u8()? {
                    $($tag => Self::$variant $({ $($field: Wire::decode(r)?,)* })?,)*
                    _ => return Err(DecodeError(concat!("bad ", stringify!($name)))),
                })
            }
        }
    };
}

wire_struct!(ProgrammableStage {
    module: ObjectId,
    entry_point: Option<String>,
    /// The values of pipeline-overridable constants, by their names or ids.
    constants: Vec<(String, f64)>,
});

wire_struct!(BlendComponent {
    operation: String,
    src_factor: String,
    dst_factor: String,
});

wire_struct!(BlendState {
    color: BlendComponent,
    alpha: BlendComponent,
});

wire_struct!(ColorTargetState {
    format: String,
    blend: Option<BlendState>,
    /// The bits of GPUColorWrite.
    write_mask: u32,
});

wire_struct!(FragmentState {
    stage: ProgrammableStage,
    targets: Vec<Option<ColorTargetState>>,
});

wire_struct!(VertexAttribute {
    format: String,
    offset: u64,
    shader_location: u32,
});

wire_struct!(VertexBufferLayout {
    array_stride: u64,
    step_mode: String,
    attributes: Vec<VertexAttribute>,
});

wire_struct!(PrimitiveState {
    topology: PrimitiveTopology,
    strip_index_format: Option<String>,
    front_face: String,
    cull_mode: String,
    unclipped_depth: bool,
});

wire_struct!(StencilFaceState {
    compare: String,
    fail_op: String,
    depth_fail_op: String,
    pass_op: String,
});

wire_struct!(DepthStencilState {
    format: String,
    depth_write_enabled: bool,
    depth_compare: Option<String>,
    stencil_front: StencilFaceState,
    stencil_back: StencilFaceState,
    stencil_read_mask: u32,
    stencil_write_mask: u32,
    depth_bias: i32,
    depth_bias_slope_scale: f32,
    depth_bias_clamp: f32,
});

wire_struct!(MultisampleState {
    count: u32,
    mask: u32,
    alpha_to_coverage_enabled: bool,
});

wire_struct!(RenderPipelineDescriptor {
    /// None for "auto".
    layout: Option<ObjectId>,
    vertex: ProgrammableStage,
    vertex_buffers: Vec<Option<VertexBufferLayout>>,
    fragment: Option<FragmentState>,
    primitive: PrimitiveState,
    depth_stencil: Option<DepthStencilState>,
    multisample: MultisampleState,
});

wire_struct!(TextureDescriptor {
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    mip_level_count: u32,
    sample_count: u32,
    dimension: String,
    format: String,
    /// The bits of GPUTextureUsage.
    usage: u32,
    view_formats: Vec<String>,
});

wire_struct!(TextureViewDescriptor {
    format: Option<String>,
    dimension: Option<String>,
    aspect: String,
    base_mip_level: u32,
    mip_level_count: Option<u32>,
    base_array_layer: u32,
    array_layer_count: Option<u32>,
});

wire_struct!(SamplerDescriptor {
    address_mode_u: String,
    address_mode_v: String,
    address_mode_w: String,
    mag_filter: String,
    min_filter: String,
    mipmap_filter: String,
    lod_min_clamp: f32,
    lod_max_clamp: f32,
    compare: Option<String>,
    max_anisotropy: u32,
});

wire_enum!(LayoutResource {
    0 => Buffer { kind: String, has_dynamic_offset: bool, min_binding_size: u64, },
    1 => Sampler { kind: String, },
    2 => Texture { sample_type: String, view_dimension: String, multisampled: bool, },
    3 => StorageTexture { access: String, format: String, view_dimension: String, },
});

wire_struct!(BindGroupLayoutEntry {
    binding: u32,
    /// The bits of GPUShaderStage.
    visibility: u32,
    resource: LayoutResource,
});

wire_enum!(BindingResource {
    0 => Buffer { buffer: ObjectId, offset: u64, size: Option<u64>, },
    1 => Sampler { sampler: ObjectId, },
    2 => TextureView { view: ObjectId, },
});

wire_struct!(BindGroupEntry {
    binding: u32,
    resource: BindingResource,
});

wire_struct!(ColorAttachment {
    view: ObjectId,
    /// The slice of a 3D texture to render into.
    depth_slice: Option<u32>,
    resolve_target: Option<ObjectId>,
    load: LoadOp,
    store: StoreOp,
});

wire_struct!(DepthStencilAttachment {
    view: ObjectId,
    /// "load" or "clear", or None when the pass does not touch the aspect.
    depth_load_op: Option<String>,
    depth_store_op: Option<String>,
    depth_clear_value: f32,
    depth_read_only: bool,
    stencil_load_op: Option<String>,
    stencil_store_op: Option<String>,
    stencil_clear_value: u32,
    stencil_read_only: bool,
});

wire_struct!(
    /// GPUTexelCopyTextureInfo: a texture subresource and a point in it.
    TexelCopyTexture {
        texture: ObjectId,
        mip_level: u32,
        origin: (u32, u32, u32),
        aspect: String,
    }
);

wire_struct!(
    /// GPUTexelCopyBufferLayout: where the texels of a copy are in a buffer or in data.
    TexelCopyBufferLayout {
        offset: u64,
        bytes_per_row: Option<u32>,
        rows_per_image: Option<u32>,
    }
);

wire_struct!(
    /// GPURenderPassTimestampWrites or GPUComputePassTimestampWrites.
    TimestampWrites {
        query_set: ObjectId,
        beginning_of_pass_write_index: Option<u32>,
        end_of_pass_write_index: Option<u32>,
    }
);

wire_struct!(ComputePipelineDescriptor {
    /// None for "auto".
    layout: Option<ObjectId>,
    compute: ProgrammableStage,
});

wire_enum!(ComputePassOp {
    0 => SetPipeline { pipeline: ObjectId, },
    1 => SetBindGroup { index: u32, group: Option<ObjectId>, dynamic_offsets: Vec<u32>, },
    2 => DispatchWorkgroups { x: u32, y: u32, z: u32, },
    3 => DispatchWorkgroupsIndirect { buffer: ObjectId, offset: u64, },
});

wire_enum!(RenderPassOp {
    0 => SetPipeline { pipeline: ObjectId, },
    1 => Draw { vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32, },
    2 => DrawIndexed { index_count: u32, instance_count: u32, first_index: u32, base_vertex: i32, first_instance: u32, },
    3 => SetBindGroup { index: u32, group: Option<ObjectId>, dynamic_offsets: Vec<u32>, },
    4 => SetVertexBuffer { slot: u32, buffer: Option<ObjectId>, offset: u64, size: Option<u64>, },
    5 => SetIndexBuffer { buffer: ObjectId, format: String, offset: u64, size: Option<u64>, },
    6 => SetViewport { x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32, },
    7 => SetScissorRect { x: u32, y: u32, width: u32, height: u32, },
    8 => SetBlendConstant { color: [f64; 4], },
    9 => SetStencilReference { reference: u32, },
    10 => DrawIndirect { buffer: ObjectId, offset: u64, },
    11 => DrawIndexedIndirect { buffer: ObjectId, offset: u64, },
    12 => BeginOcclusionQuery { index: u32, },
    13 => EndOcclusionQuery,
    14 => ExecuteBundles { bundles: Vec<ObjectId>, },
});

wire_struct!(
    /// GPURenderBundleEncoderDescriptor: the attachments of the passes the bundle runs in.
    RenderBundleDescriptor {
        color_formats: Vec<Option<String>>,
        depth_stencil_format: Option<String>,
        sample_count: u32,
        depth_read_only: bool,
        stencil_read_only: bool,
    }
);

wire_enum!(Command {
    0 => CreateShaderModule { device: ObjectId, id: ObjectId, code: String, },
    1 => CreateRenderPipeline { device: ObjectId, id: ObjectId, descriptor: Box<RenderPipelineDescriptor>, },
    2 => CreateTextureView { texture: ObjectId, id: ObjectId, descriptor: TextureViewDescriptor, },
    3 => CreateCommandEncoder { device: ObjectId, id: ObjectId, },
    /// A whole render pass, sent when script ends it.
    4 => EncodeRenderPass {
        encoder: ObjectId,
        color_attachments: Vec<Option<ColorAttachment>>,
        depth_stencil: Option<DepthStencilAttachment>,
        occlusion_query_set: Option<ObjectId>,
        timestamp_writes: Option<TimestampWrites>,
        ops: Vec<RenderPassOp>,
    },
    5 => FinishCommandEncoder { encoder: ObjectId, id: ObjectId, },
    6 => Submit { device: ObjectId, command_buffers: Vec<ObjectId>, },
    /// `usage` has the bits of GPUTextureUsage that the configuration asks for.
    7 => ConfigureCanvas { canvas: CanvasId, device: ObjectId, format: TextureFormat, usage: u32, },
    8 => UnconfigureCanvas { canvas: CanvasId, },
    /// Makes `id` the texture the canvas presents next, sized to the canvas.
    9 => GetCurrentTexture { canvas: CanvasId, id: ObjectId, width: u32, height: u32, },
    /// The explicit `destroy()` of a texture, a buffer or a device: it is invalid from now on, though the id still
    /// names it.
    10 => Destroy { id: ObjectId, },
    /// WebContent no longer refers to the object, whose wrapper was collected. GPU work that uses it keeps it alive
    /// for as long as it needs.
    11 => Release { id: ObjectId, },
    12 => PushErrorScope { device: ObjectId, filter: ErrorFilter, },
    /// Asks for an answer, which comes back as an `Event::Reply`.
    13 => Request { request: RequestId, body: Request, },
    /// `usage` has the bits of GPUBufferUsage.
    14 => CreateBuffer { device: ObjectId, id: ObjectId, size: u64, usage: u32, mapped_at_creation: bool, },
    /// `GPUQueue.writeBuffer()`.
    15 => WriteBuffer { device: ObjectId, buffer: ObjectId, offset: u64, data: Blob, },
    16 => CopyBufferToBuffer {
        encoder: ObjectId,
        source: ObjectId,
        source_offset: u64,
        destination: ObjectId,
        destination_offset: u64,
        size: u64,
    },
    /// Ends a mapping for writing, or the mapping at creation, with what script wrote into it, from `offset`.
    17 => UnmapBuffer { buffer: ObjectId, offset: u64, data: Blob, },
    18 => CreateTexture { device: ObjectId, id: ObjectId, descriptor: TextureDescriptor, },
    19 => CreateBindGroupLayout { device: ObjectId, id: ObjectId, entries: Vec<BindGroupLayoutEntry>, },
    20 => CreatePipelineLayout { device: ObjectId, id: ObjectId, bind_group_layouts: Vec<ObjectId>, },
    21 => CreateBindGroup { device: ObjectId, id: ObjectId, layout: ObjectId, entries: Vec<BindGroupEntry>, },
    /// Names the bind group layout of a pipeline with an "auto" layout.
    22 => GetBindGroupLayout { pipeline: ObjectId, index: u32, id: ObjectId, },
    23 => CreateSampler { device: ObjectId, id: ObjectId, descriptor: SamplerDescriptor, },
    24 => CopyBufferToTexture {
        encoder: ObjectId,
        buffer: ObjectId,
        layout: TexelCopyBufferLayout,
        texture: TexelCopyTexture,
        size: (u32, u32, u32),
    },
    25 => CopyTextureToBuffer {
        encoder: ObjectId,
        texture: TexelCopyTexture,
        buffer: ObjectId,
        layout: TexelCopyBufferLayout,
        size: (u32, u32, u32),
    },
    26 => CopyTextureToTexture {
        encoder: ObjectId,
        source: TexelCopyTexture,
        destination: TexelCopyTexture,
        size: (u32, u32, u32),
    },
    /// `GPUQueue.writeTexture()`.
    27 => WriteTexture {
        device: ObjectId,
        texture: TexelCopyTexture,
        data: Blob,
        layout: TexelCopyBufferLayout,
        size: (u32, u32, u32),
    },
    28 => ClearBuffer { encoder: ObjectId, buffer: ObjectId, offset: u64, size: Option<u64>, },
    /// `GPUQueue.copyExternalImageToTexture()`, with the image already read into `width` × `height` RGBA8 texels,
    /// flipped and with its alpha as the destination asks. The compositor converts them to the texture's format.
    29 => CopyExternalImageToTexture {
        device: ObjectId,
        texture: TexelCopyTexture,
        width: u32,
        height: u32,
        rgba8: Blob,
    },
    30 => CreateComputePipeline { device: ObjectId, id: ObjectId, descriptor: ComputePipelineDescriptor, },
    /// A whole compute pass, sent when script ends it.
    31 => EncodeComputePass { encoder: ObjectId, timestamp_writes: Option<TimestampWrites>, ops: Vec<ComputePassOp>, },
    /// `kind` is "occlusion" or "timestamp".
    32 => CreateQuerySet { device: ObjectId, id: ObjectId, kind: String, count: u32, },
    33 => ResolveQuerySet {
        encoder: ObjectId,
        query_set: ObjectId,
        first_query: u32,
        query_count: u32,
        destination: ObjectId,
        destination_offset: u64,
    },
    /// A whole render bundle, sent when script finishes it.
    34 => CreateRenderBundle { device: ObjectId, id: ObjectId, descriptor: RenderBundleDescriptor, ops: Vec<RenderPassOp>, },
});

wire_enum!(
    /// What a `Command::Request` asks for. The answer comes back as an `Event::Reply` with the same request id.
    Request {
        0 => RequestAdapter { id: ObjectId, power_preference: PowerPreference, force_fallback_adapter: bool, },
        1 => RequestDevice {
            adapter: ObjectId,
            id: ObjectId,
            required_features: Vec<String>,
            required_limits: Vec<(String, u64)>,
        },
        2 => PopErrorScope { device: ObjectId, },
        3 => GetCompilationInfo { module: ObjectId, },
        /// Maps a range of a buffer, and answers with its contents. WebContent then holds the mapping.
        4 => MapBuffer { buffer: ObjectId, for_writing: bool, offset: u64, size: u64, },
        5 => OnSubmittedWorkDone { device: ObjectId, },
        /// `createRenderPipelineAsync()`: makes the pipeline, and answers with the error that making it raised.
        6 => CreateRenderPipeline { device: ObjectId, id: ObjectId, descriptor: Box<RenderPipelineDescriptor>, },
        /// `createComputePipelineAsync()`, as above.
        7 => CreateComputePipeline { device: ObjectId, id: ObjectId, descriptor: ComputePipelineDescriptor, },
    }
);

wire_struct!(
    /// What an adapter or a device supports: the names of its features and the values of its limits, both as the
    /// spec names them (`"float32-filterable"`, `"maxBindGroups"`).
    #[derive(Default)]
    Capabilities {
        features: Vec<String>,
        limits: Vec<(String, u64)>,
    }
);

wire_struct!(
    #[derive(Default)]
    AdapterInfo {
        vendor: String,
        architecture: String,
        device: String,
        description: String,
        is_fallback_adapter: bool,
        capabilities: Capabilities,
    }
);

wire_enum!(
    /// Why a device request failed, which decides how its promise is rejected.
    DeviceRequestError {
        /// The adapter lacks a required feature: a TypeError.
        0 => Type { message: String, },
        /// A required limit is unknown or better than the adapter's, or the device could not be created: an
        /// OperationError.
        1 => Operation { message: String, },
    }
);

wire_enum!(GpuError {
    0 => Validation { message: String, },
    1 => OutOfMemory { message: String, },
    2 => Internal { message: String, },
});

impl GpuError {
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            Self::Validation { message } | Self::OutOfMemory { message } | Self::Internal { message } => message,
        }
    }
}

wire_struct!(CompilationMessage {
    message: String,
    message_type: CompilationMessageType,
    /// One-based, or zero when the message has no location, as in the spec.
    line_num: u32,
    line_pos: u32,
    offset: u32,
    length: u32,
});

wire_enum!(Reply {
    0 => Adapter { info: Option<AdapterInfo>, },
    1 => Device { result: Result<Capabilities, DeviceRequestError>, },
    /// The error the scope caught, if any, or Err if the device had no scope to pop.
    2 => PopErrorScope { result: Result<Option<GpuError>, String>, },
    3 => CompilationInfo { messages: Vec<CompilationMessage>, },
    /// The contents of the mapped range, or why it could not be mapped.
    4 => MapBuffer { result: Result<Blob, String>, },
    5 => WorkDone,
    /// The validation or internal error that making a pipeline raised, if any.
    6 => Pipeline { error: Option<GpuError>, },
});

wire_enum!(Event {
    0 => Reply { request: RequestId, reply: Reply, },
    /// An error that no error scope caught: https://gpuweb.github.io/gpuweb/#abstract-opdef-dispatch-error
    1 => UncapturedError { device: ObjectId, error: GpuError, },
    2 => DeviceLost { device: ObjectId, reason: DeviceLostReason, message: String, },
});

/// Bytes that travel with one length, rather than as a sequence of `u8`s.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Blob(pub Vec<u8>);

impl From<Vec<u8>> for Blob {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct DecodeError(pub &'static str);

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "malformed WebGPU message: {}", self.0)
    }
}

pub type DecodeResult<T> = Result<T, DecodeError>;

#[derive(Default)]
pub struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn take(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.bytes)
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn put(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn len(&mut self, len: usize) {
        u32::try_from(len).expect("WebGPU message field too long").encode(self);
    }
}

pub struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    #[must_use]
    pub fn is_at_end(&self) -> bool {
        self.bytes.is_empty()
    }

    fn take(&mut self, count: usize) -> DecodeResult<&'a [u8]> {
        if self.bytes.len() < count {
            return Err(DecodeError("truncated"));
        }
        let (head, tail) = self.bytes.split_at(count);
        self.bytes = tail;
        Ok(head)
    }

    fn array<const N: usize>(&mut self) -> DecodeResult<[u8; N]> {
        let mut array = [0; N];
        array.copy_from_slice(self.take(N)?);
        Ok(array)
    }

    fn u8(&mut self) -> DecodeResult<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn len(&mut self) -> DecodeResult<usize> {
        let len = u32::decode(self)? as usize;
        // Every element takes at least one byte, so a longer count than bytes left is malformed.
        if len > self.bytes.len() {
            return Err(DecodeError("length past end"));
        }
        Ok(len)
    }
}

trait Wire: Sized {
    fn encode(&self, w: &mut Writer);
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self>;
}

macro_rules! wire_number {
    ($($type:ty),*) => {
        $(
            impl Wire for $type {
                fn encode(&self, w: &mut Writer) {
                    w.put(&self.to_le_bytes());
                }
                fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
                    Ok(<$type>::from_le_bytes(r.array()?))
                }
            }
        )*
    };
}

wire_number!(u32, u64, i32, f32, f64);

impl Wire for bool {
    fn encode(&self, w: &mut Writer) {
        w.u8(u8::from(*self));
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        match r.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(DecodeError("bad bool")),
        }
    }
}

impl Wire for String {
    fn encode(&self, w: &mut Writer) {
        w.len(self.len());
        w.put(self.as_bytes());
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        let len = r.len()?;
        String::from_utf8(r.take(len)?.to_vec()).map_err(|_| DecodeError("bad utf-8"))
    }
}

impl Wire for Blob {
    fn encode(&self, w: &mut Writer) {
        (self.0.len() as u64).encode(w);
        w.put(&self.0);
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        let len = usize::try_from(u64::decode(r)?).map_err(|_| DecodeError("length past end"))?;
        Ok(Self(r.take(len)?.to_vec()))
    }
}

impl<T: Wire> Wire for Box<T> {
    fn encode(&self, w: &mut Writer) {
        (**self).encode(w);
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        Ok(Self::new(T::decode(r)?))
    }
}

impl<T: Wire> Wire for Option<T> {
    fn encode(&self, w: &mut Writer) {
        self.is_some().encode(w);
        if let Some(value) = self {
            value.encode(w);
        }
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        if bool::decode(r)? {
            Ok(Some(T::decode(r)?))
        } else {
            Ok(None)
        }
    }
}

impl<T: Wire, E: Wire> Wire for Result<T, E> {
    fn encode(&self, w: &mut Writer) {
        match self {
            Ok(value) => {
                w.u8(0);
                value.encode(w);
            }
            Err(error) => {
                w.u8(1);
                error.encode(w);
            }
        }
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        match r.u8()? {
            0 => Ok(Ok(T::decode(r)?)),
            1 => Ok(Err(E::decode(r)?)),
            _ => Err(DecodeError("bad Result")),
        }
    }
}

impl<T: Wire> Wire for Vec<T> {
    fn encode(&self, w: &mut Writer) {
        w.len(self.len());
        for element in self {
            element.encode(w);
        }
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        let len = r.len()?;
        (0..len).map(|_| T::decode(r)).collect()
    }
}

impl<A: Wire, B: Wire> Wire for (A, B) {
    fn encode(&self, w: &mut Writer) {
        self.0.encode(w);
        self.1.encode(w);
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        Ok((A::decode(r)?, B::decode(r)?))
    }
}

impl<A: Wire, B: Wire, C: Wire> Wire for (A, B, C) {
    fn encode(&self, w: &mut Writer) {
        self.0.encode(w);
        self.1.encode(w);
        self.2.encode(w);
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        Ok((A::decode(r)?, B::decode(r)?, C::decode(r)?))
    }
}

impl Wire for [f64; 4] {
    fn encode(&self, w: &mut Writer) {
        for component in self {
            component.encode(w);
        }
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        Ok([f64::decode(r)?, f64::decode(r)?, f64::decode(r)?, f64::decode(r)?])
    }
}

macro_rules! wire_unit_enum {
    ($type:ident { $($variant:ident = $tag:literal),* $(,)? }) => {
        impl Wire for $type {
            fn encode(&self, w: &mut Writer) {
                w.u8(match self { $(Self::$variant => $tag),* });
            }
            fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
                match r.u8()? {
                    $($tag => Ok(Self::$variant),)*
                    _ => Err(DecodeError(concat!("bad ", stringify!($type)))),
                }
            }
        }
    };
}

wire_unit_enum!(TextureFormat { Rgba8Unorm = 0, Rgba8UnormSrgb = 1, Bgra8Unorm = 2, Bgra8UnormSrgb = 3, Rgba16Float = 4 });
wire_unit_enum!(PrimitiveTopology { PointList = 0, LineList = 1, LineStrip = 2, TriangleList = 3, TriangleStrip = 4 });
wire_unit_enum!(PowerPreference { Default = 0, LowPower = 1, HighPerformance = 2 });
wire_unit_enum!(StoreOp { Store = 0, Discard = 1 });
wire_unit_enum!(ErrorFilter { Validation = 0, OutOfMemory = 1, Internal = 2 });
wire_unit_enum!(CompilationMessageType { Error = 0, Warning = 1, Info = 2 });
wire_unit_enum!(DeviceLostReason { Unknown = 0, Destroyed = 1 });

impl Wire for LoadOp {
    fn encode(&self, w: &mut Writer) {
        match self {
            Self::Clear(color) => {
                w.u8(0);
                color.encode(w);
            }
            Self::Load => w.u8(1),
        }
    }
    fn decode(r: &mut Reader<'_>) -> DecodeResult<Self> {
        match r.u8()? {
            0 => Ok(Self::Clear(<[f64; 4]>::decode(r)?)),
            1 => Ok(Self::Load),
            _ => Err(DecodeError("bad LoadOp")),
        }
    }
}

pub fn encode_command(w: &mut Writer, command: &Command) {
    command.encode(w);
}

/// Decodes a stream of commands, as one `webgpu_commands` message carries.
pub fn decode_commands(bytes: &[u8]) -> DecodeResult<Vec<Command>> {
    let mut r = Reader::new(bytes);
    let mut commands = Vec::new();
    while !r.is_at_end() {
        commands.push(Command::decode(&mut r)?);
    }
    Ok(commands)
}

/// Encodes the events of one `webgpu_events` message.
#[must_use]
pub fn encode_events(events: &[Event]) -> Vec<u8> {
    let mut w = Writer::new();
    for event in events {
        event.encode(&mut w);
    }
    w.take()
}

/// Decodes a stream of events, as one `webgpu_events` message carries.
pub fn decode_events(bytes: &[u8]) -> DecodeResult<Vec<Event>> {
    let mut r = Reader::new(bytes);
    let mut events = Vec::new();
    while !r.is_at_end() {
        events.push(Event::decode(&mut r)?);
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(module: ObjectId, entry_point: Option<&str>) -> ProgrammableStage {
        ProgrammableStage {
            module,
            entry_point: entry_point.map(str::to_owned),
            constants: Vec::new(),
        }
    }

    fn sample_commands() -> Vec<Command> {
        let replace = BlendComponent {
            operation: "add".into(),
            src_factor: "one".into(),
            dst_factor: "zero".into(),
        };
        let keep = StencilFaceState {
            compare: "always".into(),
            fail_op: "keep".into(),
            depth_fail_op: "keep".into(),
            pass_op: "keep".into(),
        };
        vec![
            Command::CreateShaderModule {
                device: 2,
                id: 3,
                code: "@vertex fn main() {}".into(),
            },
            Command::CreateRenderPipeline {
                device: 2,
                id: 4,
                descriptor: Box::new(RenderPipelineDescriptor {
                    layout: Some(30),
                    vertex: stage(3, Some("vs")),
                    vertex_buffers: vec![
                        Some(VertexBufferLayout {
                            array_stride: 40,
                            step_mode: "vertex".into(),
                            attributes: vec![VertexAttribute {
                                format: "float32x4".into(),
                                offset: 0,
                                shader_location: 0,
                            }],
                        }),
                        None,
                    ],
                    fragment: Some(FragmentState {
                        stage: stage(3, None),
                        targets: vec![
                            Some(ColorTargetState {
                                format: "bgra8unorm".into(),
                                blend: Some(BlendState {
                                    color: replace.clone(),
                                    alpha: replace,
                                }),
                                write_mask: 0xF,
                            }),
                            None,
                        ],
                    }),
                    primitive: PrimitiveState {
                        topology: PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: "ccw".into(),
                        cull_mode: "back".into(),
                        unclipped_depth: false,
                    },
                    depth_stencil: Some(DepthStencilState {
                        format: "depth24plus".into(),
                        depth_write_enabled: true,
                        depth_compare: Some("less".into()),
                        stencil_front: keep.clone(),
                        stencil_back: keep,
                        stencil_read_mask: 0xFFFF_FFFF,
                        stencil_write_mask: 0xFFFF_FFFF,
                        depth_bias: -2,
                        depth_bias_slope_scale: 0.5,
                        depth_bias_clamp: 0.0,
                    }),
                    multisample: MultisampleState {
                        count: 1,
                        mask: 0xFFFF_FFFF,
                        alpha_to_coverage_enabled: false,
                    },
                }),
            },
            Command::ConfigureCanvas {
                canvas: 9,
                device: 2,
                format: TextureFormat::Rgba8Unorm,
                usage: 0x10,
            },
            Command::GetCurrentTexture {
                canvas: 9,
                id: 5,
                width: 300,
                height: 150,
            },
            Command::CreateTextureView {
                texture: 5,
                id: 6,
                descriptor: TextureViewDescriptor {
                    format: None,
                    dimension: Some("2d".into()),
                    aspect: "all".into(),
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: Some(1),
                },
            },
            Command::CreateCommandEncoder { device: 2, id: 7 },
            Command::EncodeRenderPass {
                encoder: 7,
                color_attachments: vec![
                    Some(ColorAttachment {
                        view: 6,
                        depth_slice: None,
                        resolve_target: None,
                        load: LoadOp::Clear([0.0, 0.25, 0.5, 1.0]),
                        store: StoreOp::Store,
                    }),
                    None,
                ],
                depth_stencil: Some(DepthStencilAttachment {
                    view: 31,
                    depth_load_op: Some("clear".into()),
                    depth_store_op: Some("store".into()),
                    depth_clear_value: 1.0,
                    depth_read_only: false,
                    stencil_load_op: None,
                    stencil_store_op: None,
                    stencil_clear_value: 0,
                    stencil_read_only: false,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                ops: vec![
                    RenderPassOp::SetPipeline { pipeline: 4 },
                    RenderPassOp::SetBindGroup {
                        index: 0,
                        group: Some(32),
                        dynamic_offsets: vec![256],
                    },
                    RenderPassOp::SetVertexBuffer {
                        slot: 0,
                        buffer: Some(33),
                        offset: 0,
                        size: None,
                    },
                    RenderPassOp::SetIndexBuffer {
                        buffer: 34,
                        format: "uint16".into(),
                        offset: 0,
                        size: Some(72),
                    },
                    RenderPassOp::SetViewport {
                        x: 0.0,
                        y: 0.0,
                        width: 300.0,
                        height: 150.0,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    },
                    RenderPassOp::SetScissorRect {
                        x: 0,
                        y: 0,
                        width: 300,
                        height: 150,
                    },
                    RenderPassOp::SetBlendConstant { color: [1.0; 4] },
                    RenderPassOp::SetStencilReference { reference: 1 },
                    RenderPassOp::Draw {
                        vertex_count: 3,
                        instance_count: 1,
                        first_vertex: 0,
                        first_instance: 0,
                    },
                    RenderPassOp::DrawIndexed {
                        index_count: 36,
                        instance_count: 1,
                        first_index: 0,
                        base_vertex: -1,
                        first_instance: 0,
                    },
                ],
            },
            Command::FinishCommandEncoder { encoder: 7, id: 8 },
            Command::Submit {
                device: 2,
                command_buffers: vec![8],
            },
            Command::UnconfigureCanvas { canvas: 9 },
            Command::Destroy { id: 8 },
            Command::Release { id: 8 },
            Command::PushErrorScope {
                device: 2,
                filter: ErrorFilter::OutOfMemory,
            },
            Command::Request {
                request: 9,
                body: Request::RequestAdapter {
                    id: 10,
                    power_preference: PowerPreference::LowPower,
                    force_fallback_adapter: true,
                },
            },
            Command::Request {
                request: 10,
                body: Request::RequestDevice {
                    adapter: 10,
                    id: 11,
                    required_features: vec!["shader-f16".into()],
                    required_limits: vec![("maxBufferSize".into(), 1 << 30)],
                },
            },
            Command::Request {
                request: 11,
                body: Request::PopErrorScope { device: 2 },
            },
            Command::Request {
                request: 12,
                body: Request::GetCompilationInfo { module: 3 },
            },
            Command::Request {
                request: 13,
                body: Request::MapBuffer {
                    buffer: 20,
                    for_writing: false,
                    offset: 0,
                    size: 16,
                },
            },
            Command::Request {
                request: 14,
                body: Request::OnSubmittedWorkDone { device: 2 },
            },
            Command::CreateBuffer {
                device: 2,
                id: 20,
                size: 256,
                usage: 0x9,
                mapped_at_creation: true,
            },
            Command::WriteBuffer {
                device: 2,
                buffer: 20,
                offset: 16,
                data: vec![1, 2, 3, 4].into(),
            },
            Command::CopyBufferToBuffer {
                encoder: 7,
                source: 20,
                source_offset: 0,
                destination: 21,
                destination_offset: 8,
                size: 64,
            },
            Command::UnmapBuffer {
                buffer: 20,
                offset: 0,
                data: vec![9; 256].into(),
            },
            Command::CreateTexture {
                device: 2,
                id: 31,
                descriptor: TextureDescriptor {
                    width: 300,
                    height: 150,
                    depth_or_array_layers: 1,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: "2d".into(),
                    format: "depth24plus".into(),
                    usage: 0x10,
                    view_formats: Vec::new(),
                },
            },
            Command::CreateBindGroupLayout {
                device: 2,
                id: 35,
                entries: vec![
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: 0x1,
                        resource: LayoutResource::Buffer {
                            kind: "uniform".into(),
                            has_dynamic_offset: false,
                            min_binding_size: 64,
                        },
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: 0x2,
                        resource: LayoutResource::Texture {
                            sample_type: "float".into(),
                            view_dimension: "2d".into(),
                            multisampled: false,
                        },
                    },
                ],
            },
            Command::CreatePipelineLayout {
                device: 2,
                id: 30,
                bind_group_layouts: vec![35],
            },
            Command::CreateBindGroup {
                device: 2,
                id: 32,
                layout: 35,
                entries: vec![
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: 20,
                            offset: 0,
                            size: Some(64),
                        },
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView { view: 6 },
                    },
                ],
            },
            Command::GetBindGroupLayout {
                pipeline: 4,
                index: 0,
                id: 36,
            },
            Command::CopyTextureToTexture {
                encoder: 7,
                source: TexelCopyTexture {
                    texture: 3,
                    mip_level: 1,
                    origin: (1, 2, 3),
                    aspect: "all".into(),
                },
                destination: TexelCopyTexture {
                    texture: 4,
                    mip_level: 0,
                    origin: (0, 0, 0),
                    aspect: "depth-only".into(),
                },
                size: (4, 5, 6),
            },
            Command::WriteTexture {
                device: 2,
                texture: TexelCopyTexture {
                    texture: 3,
                    mip_level: 0,
                    origin: (0, 0, 0),
                    aspect: "all".into(),
                },
                data: Blob(vec![1, 2, 3, 4]),
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4),
                    rows_per_image: None,
                },
                size: (1, 1, 1),
            },
            Command::CreateSampler {
                device: 2,
                id: 37,
                descriptor: SamplerDescriptor {
                    address_mode_u: "repeat".into(),
                    address_mode_v: "clamp-to-edge".into(),
                    address_mode_w: "mirror-repeat".into(),
                    mag_filter: "linear".into(),
                    min_filter: "nearest".into(),
                    mipmap_filter: "linear".into(),
                    lod_min_clamp: 0.0,
                    lod_max_clamp: 32.0,
                    compare: Some("less".into()),
                    max_anisotropy: 4,
                },
            },
        ]
    }

    #[test]
    fn commands_round_trip() {
        let commands = sample_commands();
        let mut w = Writer::new();
        for command in &commands {
            encode_command(&mut w, command);
        }
        assert_eq!(decode_commands(w.bytes()), Ok(commands));
    }

    #[test]
    fn events_round_trip() {
        let events = vec![
            Event::Reply {
                request: 1,
                reply: Reply::Adapter {
                    info: Some(AdapterInfo {
                        vendor: "nvidia".into(),
                        description: "NVIDIA GeForce RTX 3060".into(),
                        capabilities: Capabilities {
                            features: vec!["float32-filterable".into()],
                            limits: vec![("maxBindGroups".into(), 4)],
                        },
                        ..AdapterInfo::default()
                    }),
                },
            },
            Event::Reply {
                request: 2,
                reply: Reply::Adapter { info: None },
            },
            Event::Reply {
                request: 3,
                reply: Reply::Device {
                    result: Err(DeviceRequestError::Type { message: "no".into() }),
                },
            },
            Event::Reply {
                request: 3,
                reply: Reply::Device {
                    result: Err(DeviceRequestError::Operation { message: "no".into() }),
                },
            },
            Event::Reply {
                request: 3,
                reply: Reply::Device {
                    result: Ok(Capabilities::default()),
                },
            },
            Event::Reply {
                request: 4,
                reply: Reply::PopErrorScope { result: Ok(None) },
            },
            Event::Reply {
                request: 5,
                reply: Reply::PopErrorScope {
                    result: Ok(Some(GpuError::OutOfMemory { message: "full".into() })),
                },
            },
            Event::Reply {
                request: 6,
                reply: Reply::PopErrorScope {
                    result: Err("no scope".into()),
                },
            },
            Event::Reply {
                request: 7,
                reply: Reply::CompilationInfo {
                    messages: vec![CompilationMessage {
                        message: "expected ;".into(),
                        message_type: CompilationMessageType::Error,
                        line_num: 3,
                        line_pos: 14,
                        offset: 40,
                        length: 1,
                    }],
                },
            },
            Event::Reply {
                request: 8,
                reply: Reply::MapBuffer {
                    result: Ok(vec![1, 2, 3].into()),
                },
            },
            Event::Reply {
                request: 9,
                reply: Reply::MapBuffer {
                    result: Err("busy".into()),
                },
            },
            Event::Reply {
                request: 10,
                reply: Reply::WorkDone,
            },
            Event::UncapturedError {
                device: 2,
                error: GpuError::Validation { message: "bad".into() },
            },
            Event::DeviceLost {
                device: 2,
                reason: DeviceLostReason::Destroyed,
                message: String::new(),
            },
        ];
        assert_eq!(decode_events(&encode_events(&events)), Ok(events));
    }

    #[test]
    fn truncated_input_is_rejected_not_panicking() {
        let mut w = Writer::new();
        for command in &sample_commands() {
            encode_command(&mut w, command);
        }
        let bytes = w.take();
        for len in 0..bytes.len() {
            let _ = decode_commands(&bytes[..len]);
        }
        assert!(decode_commands(&[200]).is_err());
        assert!(decode_commands(&[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255]).is_err());
    }
}
