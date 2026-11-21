/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! These need a Vulkan device; lavapipe will do. Without one, they say so and pass.

use webgpu_wire::{
    BindGroupEntry, BindGroupLayoutEntry, BindingResource, Blob, ColorAttachment, ColorTargetState, Command,
    CompilationMessageType, ComputePassOp, ComputePipelineDescriptor, DepthStencilAttachment, DepthStencilState,
    DeviceLostReason, ErrorFilter, Event, FragmentState, GpuError, LayoutResource, LoadOp, MultisampleState,
    PowerPreference, PrimitiveState, PrimitiveTopology, ProgrammableStage, RenderPassOp, RenderPipelineDescriptor,
    Reply, Request, StencilFaceState, StoreOp, TexelCopyBufferLayout, TexelCopyTexture, TextureDescriptor,
    TextureFormat, TextureViewDescriptor, VertexAttribute, VertexBufferLayout, Writer,
};

use crate::server::{Server, ServerOptions};
use crate::wgpu_backend::WgpuBackend;

const SHADER: &str = r"
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    var pos = array<vec2f, 3>(vec2f(0.0, 0.5), vec2f(-0.5, -0.5), vec2f(0.5, -0.5));
    return vec4f(pos[i], 0.0, 1.0);
}
@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
";

const ADAPTER: u64 = 1;
const DEVICE: u64 = 2;

fn run(server: &mut Server, commands: &[Command]) -> Vec<Event> {
    let mut w = Writer::new();
    for command in commands {
        webgpu_wire::encode_command(&mut w, command);
    }
    server.handle_commands(w.bytes()).expect("well-formed stream")
}

/// Presents the canvas and waits for its frame.
fn present_now(server: &mut Server, canvas: u64) -> Option<crate::backend::Readback> {
    if !server.present(canvas) {
        return None;
    }
    let frames = server.take_frames(true);
    frames.into_iter().find_map(|(id, frame)| match frame {
        crate::backend::PresentedFrame::Pixels(pixels) if id == canvas => Some(pixels),
        _ => None,
    })
}

fn request(server: &mut Server, body: Request) -> Reply {
    let events = run(server, &[Command::Request { request: 77, body }]);
    events
        .into_iter()
        .find_map(|event| match event {
            Event::Reply { request: 77, reply } => Some(reply),
            _ => None,
        })
        .expect("a reply")
}

/// A server with a device, or None if there is no Vulkan adapter.
fn server_with_device(options: ServerOptions) -> Option<Server> {
    let mut server = Server::new(Box::new(WgpuBackend::new()), options);
    let reply = request(
        &mut server,
        Request::RequestAdapter {
            id: ADAPTER,
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
        },
    );
    let Reply::Adapter { info: Some(info) } = reply else {
        eprintln!("skipping: no Vulkan adapter");
        return None;
    };
    eprintln!("adapter: {} (fallback: {})", info.description, info.is_fallback_adapter);
    assert!(
        !options.force_fallback_adapter || info.is_fallback_adapter,
        "the fallback adapter was forced"
    );
    let Reply::Device {
        result: Ok(capabilities),
    } = request(
        &mut server,
        Request::RequestDevice {
            adapter: ADAPTER,
            id: DEVICE,
            required_features: Vec::new(),
            required_limits: Vec::new(),
        },
    )
    else {
        panic!("a device");
    };
    assert!(
        capabilities.limits.contains(&("maxBindGroups".to_owned(), 4)),
        "the default limits"
    );
    Some(server)
}

fn stage(module: u64, entry_point: Option<&str>) -> ProgrammableStage {
    ProgrammableStage {
        module,
        entry_point: entry_point.map(str::to_owned),
        constants: Vec::new(),
    }
}

fn primitive() -> PrimitiveState {
    PrimitiveState {
        topology: PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: "ccw".into(),
        cull_mode: "none".into(),
        unclipped_depth: false,
    }
}

fn single_sample() -> MultisampleState {
    MultisampleState {
        count: 1,
        mask: !0,
        alpha_to_coverage_enabled: false,
    }
}

fn color_target(format: &str) -> Option<ColorTargetState> {
    Some(ColorTargetState {
        format: format.into(),
        blend: None,
        write_mask: 0xF,
    })
}

fn default_view() -> TextureViewDescriptor {
    TextureViewDescriptor {
        format: None,
        dimension: None,
        aspect: "all".into(),
        base_mip_level: 0,
        mip_level_count: None,
        base_array_layer: 0,
        array_layer_count: None,
    }
}

fn pipeline(id: u64, module: u64, format: TextureFormat) -> Command {
    Command::CreateRenderPipeline {
        device: DEVICE,
        id,
        descriptor: Box::new(RenderPipelineDescriptor {
            layout: None,
            vertex: stage(module, Some("vs_main")),
            vertex_buffers: Vec::new(),
            fragment: Some(FragmentState {
                stage: stage(module, None),
                targets: vec![color_target(format.name())],
            }),
            primitive: primitive(),
            depth_stencil: None,
            multisample: single_sample(),
        }),
    }
}

fn draw_frame(canvas: u64, texture: u64, pipeline: u64, clear: [f64; 4]) -> Vec<Command> {
    vec![
        Command::GetCurrentTexture {
            canvas,
            id: texture,
            width: 300,
            height: 150,
        },
        Command::CreateTextureView {
            texture,
            id: texture + 1,
            descriptor: default_view(),
        },
        Command::CreateCommandEncoder {
            device: DEVICE,
            id: texture + 2,
        },
        Command::EncodeRenderPass {
            encoder: texture + 2,
            color_attachments: vec![Some(ColorAttachment {
                view: texture + 1,
                depth_slice: None,
                resolve_target: None,
                load: LoadOp::Clear(clear),
                store: StoreOp::Store,
            })],
            depth_stencil: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            ops: vec![
                RenderPassOp::SetPipeline { pipeline },
                RenderPassOp::Draw {
                    vertex_count: 3,
                    instance_count: 1,
                    first_vertex: 0,
                    first_instance: 0,
                },
            ],
        },
        Command::FinishCommandEncoder {
            encoder: texture + 2,
            id: texture + 3,
        },
        Command::Submit {
            device: DEVICE,
            command_buffers: vec![texture + 3],
        },
    ]
}

/// The whole success page, as the commands WebContent would send.
fn clear_and_triangle(options: ServerOptions) {
    let Some(mut server) = server_with_device(options) else {
        return;
    };
    let format = TextureFormat::Bgra8Unorm;
    let mut commands = vec![
        Command::CreateShaderModule {
            device: DEVICE,
            id: 3,
            code: SHADER.into(),
        },
        pipeline(4, 3, format),
        Command::ConfigureCanvas {
            canvas: 77,
            device: DEVICE,
            format,
            usage: 0x10,
        },
    ];
    commands.extend(draw_frame(77, 10, 4, [0.0, 0.0, 1.0, 1.0]));
    assert_eq!(run(&mut server, &commands), Vec::new(), "no errors");

    let frame = present_now(&mut server, 77).expect("a frame");
    assert_eq!((frame.width, frame.height), (300, 150));
    let pixel = |x: usize, y: usize| &frame.pixels[(y * 300 + x) * 4..][..4];
    // BGRA: the clear colour is blue, the triangle red.
    assert_eq!(pixel(0, 0), [255, 0, 0, 255]);
    assert_eq!(pixel(150, 75), [0, 0, 255, 255]);
    assert!(present_now(&mut server, 77).is_none(), "a frame is presented once");
}

#[test]
fn clear_and_triangle_through_the_wire() {
    clear_and_triangle(ServerOptions::default());
}

#[test]
fn clear_and_triangle_on_the_fallback_adapter() {
    clear_and_triangle(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    });
}

/// A released id names nothing any more, while an object that GPU work still uses lives on for that work.
#[test]
fn released_objects_are_forgotten() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    let format = TextureFormat::Rgba8Unorm;
    let mut commands = vec![
        Command::CreateShaderModule {
            device: DEVICE,
            id: 3,
            code: SHADER.into(),
        },
        // The pipeline keeps the shader module alive after its id is released.
        pipeline(4, 3, format),
        Command::Release { id: 3 },
        Command::ConfigureCanvas {
            canvas: 1,
            device: DEVICE,
            format,
            usage: 0x10,
        },
    ];
    commands.extend(draw_frame(1, 10, 4, [0.0; 4]));
    assert_eq!(run(&mut server, &commands), Vec::new(), "no errors");
    let frame = present_now(&mut server, 1).expect("the pipeline drew with the released module");
    assert_eq!(&frame.pixels[(75 * 300 + 150) * 4..][..4], [255, 0, 0, 255]);

    // A pipeline from the released module's id is an error the device reports, not a crash.
    let events = run(&mut server, &[Command::Release { id: 4 }, pipeline(5, 3, format)]);
    assert!(
        matches!(
            &events[..],
            [Event::UncapturedError {
                device: DEVICE,
                error: GpuError::Validation { message: _ }
            }]
        ),
        "{events:?}"
    );
}

const BROKEN_SHADER: &str = "@fragment fn main() -> @location(0) vec4f { return vec4f(1.0) }";

#[test]
fn an_error_scope_catches_a_shader_error() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    let events = run(
        &mut server,
        &[
            Command::PushErrorScope {
                device: DEVICE,
                filter: ErrorFilter::Validation,
            },
            Command::CreateShaderModule {
                device: DEVICE,
                id: 3,
                code: BROKEN_SHADER.into(),
            },
        ],
    );
    assert_eq!(
        events,
        Vec::new(),
        "the scope catches the error, so it is not uncaptured"
    );
    let Reply::PopErrorScope {
        result: Ok(Some(GpuError::Validation { message })),
    } = request(&mut server, Request::PopErrorScope { device: DEVICE })
    else {
        panic!("the scope caught a validation error");
    };
    eprintln!("caught: {message}");
    assert_eq!(
        request(&mut server, Request::PopErrorScope { device: DEVICE }),
        Reply::PopErrorScope {
            result: Err("There is no error scope to pop".into())
        }
    );
}

#[test]
fn an_uncaptured_error_becomes_an_event() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    let events = run(
        &mut server,
        &[Command::CreateShaderModule {
            device: DEVICE,
            id: 3,
            code: BROKEN_SHADER.into(),
        }],
    );
    assert!(
        matches!(
            &events[..],
            [Event::UncapturedError {
                device: DEVICE,
                error: GpuError::Validation { message: _ }
            }]
        ),
        "{events:?}"
    );
}

#[test]
fn compilation_info_reports_where_a_shader_is_broken() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    run(
        &mut server,
        &[
            Command::PushErrorScope {
                device: DEVICE,
                filter: ErrorFilter::Validation,
            },
            Command::CreateShaderModule {
                device: DEVICE,
                id: 3,
                code: BROKEN_SHADER.into(),
            },
        ],
    );
    let Reply::CompilationInfo { messages } = request(&mut server, Request::GetCompilationInfo { module: 3 }) else {
        panic!("a compilation info reply");
    };
    eprintln!("{messages:?}");
    let error = messages.first().expect("a message");
    assert_eq!(error.message_type, CompilationMessageType::Error);
    assert_eq!(error.line_num, 1);
    assert!(error.line_pos > 0);
}

#[test]
fn destroying_a_device_loses_it() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    let events = run(&mut server, &[Command::Destroy { id: DEVICE }]);
    assert!(
        matches!(
            &events[..],
            [Event::DeviceLost {
                device: DEVICE,
                reason: DeviceLostReason::Destroyed,
                ..
            }]
        ),
        "{events:?}"
    );
}

#[test]
fn device_requests_check_features_and_limits() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    let mut device_request = |id, required_features: Vec<String>, required_limits: Vec<(String, u64)>| {
        request(
            &mut server,
            Request::RequestDevice {
                adapter: ADAPTER,
                id,
                required_features,
                required_limits,
            },
        )
    };
    assert!(matches!(
        device_request(10, vec!["no-such-feature".into()], Vec::new()),
        Reply::Device {
            result: Err(webgpu_wire::DeviceRequestError::Type { message: _ })
        }
    ));
    assert!(matches!(
        device_request(11, Vec::new(), vec![("maxNoSuchLimit".into(), 1)]),
        Reply::Device {
            result: Err(webgpu_wire::DeviceRequestError::Operation { message: _ })
        }
    ));
    assert!(matches!(
        device_request(12, Vec::new(), vec![("maxBindGroups".into(), u64::from(u32::MAX))]),
        Reply::Device {
            result: Err(webgpu_wire::DeviceRequestError::Operation { message: _ })
        }
    ));
    let Reply::Device {
        result: Ok(capabilities),
    } = device_request(13, Vec::new(), vec![("maxBindGroups".into(), 1)])
    else {
        panic!("a device with a requirement below the default");
    };
    assert!(
        capabilities.limits.contains(&("maxBindGroups".to_owned(), 4)),
        "the default stays"
    );
}

/// Data goes into buffers at creation, through the queue, by a copy and through a mapping for writing, and comes back
/// through a mapping for reading.
#[test]
fn buffers_round_trip() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    const COPY_SRC: u32 = 0x4;
    const COPY_DST: u32 = 0x8;
    const MAP_READ: u32 = 0x1;
    const MAP_WRITE: u32 = 0x2;
    let copy_to_readable = |encoder: u64, source: u64| {
        [
            Command::CreateCommandEncoder {
                device: DEVICE,
                id: encoder,
            },
            Command::CopyBufferToBuffer {
                encoder,
                source,
                source_offset: 0,
                destination: 21,
                destination_offset: 0,
                size: 16,
            },
            Command::FinishCommandEncoder {
                encoder,
                id: encoder + 1,
            },
            Command::Submit {
                device: DEVICE,
                command_buffers: vec![encoder + 1],
            },
        ]
    };
    let read_back = |server: &mut Server| {
        request(
            server,
            Request::MapBuffer {
                buffer: 21,
                for_writing: false,
                offset: 0,
                size: 16,
            },
        )
    };

    let mut commands = vec![
        Command::CreateBuffer {
            device: DEVICE,
            id: 20,
            size: 16,
            usage: COPY_SRC | COPY_DST,
            mapped_at_creation: true,
        },
        Command::UnmapBuffer {
            buffer: 20,
            offset: 0,
            data: Blob((1..=16).collect()),
        },
        Command::WriteBuffer {
            device: DEVICE,
            buffer: 20,
            offset: 4,
            data: Blob(vec![50, 51, 52, 53]),
        },
        Command::CreateBuffer {
            device: DEVICE,
            id: 21,
            size: 16,
            usage: COPY_DST | MAP_READ,
            mapped_at_creation: false,
        },
    ];
    commands.extend(copy_to_readable(30, 20));
    assert_eq!(run(&mut server, &commands), Vec::new(), "no errors");
    assert_eq!(
        request(&mut server, Request::OnSubmittedWorkDone { device: DEVICE }),
        Reply::WorkDone
    );
    assert_eq!(
        read_back(&mut server),
        Reply::MapBuffer {
            result: Ok(Blob(vec![1, 2, 3, 4, 50, 51, 52, 53, 9, 10, 11, 12, 13, 14, 15, 16]))
        }
    );

    // A mapping for writing sees the contents, and what script writes into it lands at unmap.
    let events = run(
        &mut server,
        &[Command::CreateBuffer {
            device: DEVICE,
            id: 22,
            size: 16,
            usage: MAP_WRITE | COPY_SRC,
            mapped_at_creation: false,
        }],
    );
    assert_eq!(events, Vec::new());
    assert_eq!(
        request(
            &mut server,
            Request::MapBuffer {
                buffer: 22,
                for_writing: true,
                offset: 8,
                size: 8
            }
        ),
        Reply::MapBuffer {
            result: Ok(Blob(vec![0; 8]))
        },
        "a new buffer is zeroed"
    );
    let mut commands = vec![Command::UnmapBuffer {
        buffer: 22,
        offset: 8,
        data: Blob(vec![7; 8]),
    }];
    commands.extend(copy_to_readable(40, 22));
    assert_eq!(run(&mut server, &commands), Vec::new());
    assert_eq!(
        read_back(&mut server),
        Reply::MapBuffer {
            result: Ok(Blob(vec![0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7]))
        }
    );

    // Usages that cannot go together are an error the device reports.
    let events = run(
        &mut server,
        &[Command::CreateBuffer {
            device: DEVICE,
            id: 24,
            size: 4,
            usage: MAP_WRITE | COPY_DST,
            mapped_at_creation: false,
        }],
    );
    assert!(
        matches!(&events[..], [Event::UncapturedError { device: DEVICE, .. }]),
        "{events:?}"
    );
}

/// Most of what the rotating cube of the webgpu-samples uses: an explicit layout, a bind group with a uniform
/// buffer, a vertex buffer, an index buffer, a depth texture and an indexed draw. The nearer of two overlapping
/// quads wins the depth test, though it is drawn first.
#[test]
fn indexed_draw_with_a_uniform_and_depth() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    const SHADER: &str = r"
        struct Uniforms { color: vec4f }
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;
        struct Out { @builtin(position) position: vec4f, @location(0) shade: f32 }
        @vertex fn vs_main(@location(0) position: vec4f) -> Out {
            return Out(position, position.z);
        }
        @fragment fn fs_main(in: Out) -> @location(0) vec4f {
            // The near quad has z 0.25 and is drawn in the uniform's colour; the far one, at 0.75, in white.
            if (in.shade < 0.5) { return uniforms.color; }
            return vec4f(1.0);
        }
    ";
    let quad = |z: f32| {
        [
            [-0.5f32, -0.5, z, 1.0],
            [0.5, -0.5, z, 1.0],
            [0.5, 0.5, z, 1.0],
            [-0.5, 0.5, z, 1.0],
        ]
    };
    let vertices: Vec<u8> = quad(0.25)
        .iter()
        .chain(quad(0.75).iter())
        .flat_map(|vertex| vertex.iter().flat_map(|component| component.to_le_bytes()))
        .collect();
    let indices: Vec<u8> = [0u16, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]
        .iter()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    let color: Vec<u8> = [0.0f32, 1.0, 0.0, 1.0].iter().flat_map(|c| c.to_le_bytes()).collect();
    const VERTEX: u32 = 0x20;
    const INDEX: u32 = 0x10;
    const UNIFORM: u32 = 0x40;
    const COPY_DST: u32 = 0x8;
    let keep = StencilFaceState {
        compare: "always".into(),
        fail_op: "keep".into(),
        depth_fail_op: "keep".into(),
        pass_op: "keep".into(),
    };
    let commands = vec![
        Command::CreateShaderModule {
            device: DEVICE,
            id: 3,
            code: SHADER.into(),
        },
        Command::CreateBuffer {
            device: DEVICE,
            id: 40,
            size: 128,
            usage: VERTEX | COPY_DST,
            mapped_at_creation: false,
        },
        Command::WriteBuffer {
            device: DEVICE,
            buffer: 40,
            offset: 0,
            data: Blob(vertices),
        },
        Command::CreateBuffer {
            device: DEVICE,
            id: 41,
            size: 24,
            usage: INDEX | COPY_DST,
            mapped_at_creation: false,
        },
        Command::WriteBuffer {
            device: DEVICE,
            buffer: 41,
            offset: 0,
            data: Blob(indices),
        },
        Command::CreateBuffer {
            device: DEVICE,
            id: 42,
            size: 16,
            usage: UNIFORM | COPY_DST,
            mapped_at_creation: false,
        },
        Command::WriteBuffer {
            device: DEVICE,
            buffer: 42,
            offset: 0,
            data: Blob(color),
        },
        Command::CreateBindGroupLayout {
            device: DEVICE,
            id: 43,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: 0x2,
                resource: LayoutResource::Buffer {
                    kind: "uniform".into(),
                    has_dynamic_offset: false,
                    min_binding_size: 0,
                },
            }],
        },
        Command::CreatePipelineLayout {
            device: DEVICE,
            id: 44,
            bind_group_layouts: vec![43],
        },
        Command::CreateBindGroup {
            device: DEVICE,
            id: 45,
            layout: 43,
            entries: vec![BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: 42,
                    offset: 0,
                    size: None,
                },
            }],
        },
        Command::CreateRenderPipeline {
            device: DEVICE,
            id: 46,
            descriptor: Box::new(RenderPipelineDescriptor {
                layout: Some(44),
                vertex: stage(3, Some("vs_main")),
                vertex_buffers: vec![Some(VertexBufferLayout {
                    array_stride: 16,
                    step_mode: "vertex".into(),
                    attributes: vec![VertexAttribute {
                        format: "float32x4".into(),
                        offset: 0,
                        shader_location: 0,
                    }],
                })],
                fragment: Some(FragmentState {
                    stage: stage(3, Some("fs_main")),
                    targets: vec![color_target("rgba8unorm")],
                }),
                primitive: primitive(),
                depth_stencil: Some(DepthStencilState {
                    format: "depth24plus".into(),
                    depth_write_enabled: true,
                    depth_compare: Some("less".into()),
                    stencil_front: keep.clone(),
                    stencil_back: keep,
                    stencil_read_mask: !0,
                    stencil_write_mask: !0,
                    depth_bias: 0,
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                }),
                multisample: single_sample(),
            }),
        },
        Command::CreateTexture {
            device: DEVICE,
            id: 47,
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
        Command::CreateTextureView {
            texture: 47,
            id: 48,
            descriptor: default_view(),
        },
        Command::ConfigureCanvas {
            canvas: 1,
            device: DEVICE,
            format: TextureFormat::Rgba8Unorm,
            usage: 0x10,
        },
        Command::GetCurrentTexture {
            canvas: 1,
            id: 10,
            width: 300,
            height: 150,
        },
        Command::CreateTextureView {
            texture: 10,
            id: 11,
            descriptor: default_view(),
        },
        Command::CreateCommandEncoder { device: DEVICE, id: 12 },
        Command::EncodeRenderPass {
            encoder: 12,
            color_attachments: vec![Some(ColorAttachment {
                view: 11,
                depth_slice: None,
                resolve_target: None,
                load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                store: StoreOp::Store,
            })],
            depth_stencil: Some(DepthStencilAttachment {
                view: 48,
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
                RenderPassOp::SetPipeline { pipeline: 46 },
                RenderPassOp::SetBindGroup {
                    index: 0,
                    group: Some(45),
                    dynamic_offsets: Vec::new(),
                },
                RenderPassOp::SetVertexBuffer {
                    slot: 0,
                    buffer: Some(40),
                    offset: 0,
                    size: None,
                },
                RenderPassOp::SetIndexBuffer {
                    buffer: 41,
                    format: "uint16".into(),
                    offset: 0,
                    size: None,
                },
                RenderPassOp::DrawIndexed {
                    index_count: 12,
                    instance_count: 1,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance: 0,
                },
            ],
        },
        Command::FinishCommandEncoder { encoder: 12, id: 13 },
        Command::Submit {
            device: DEVICE,
            command_buffers: vec![13],
        },
    ];
    assert_eq!(run(&mut server, &commands), Vec::new(), "no errors");
    let frame = present_now(&mut server, 1).expect("a frame");
    let pixel = |x: usize, y: usize| &frame.pixels[(y * 300 + x) * 4..][..4];
    assert_eq!(
        pixel(150, 75),
        [0, 255, 0, 255],
        "the near green quad wins the depth test"
    );
    assert_eq!(pixel(5, 5), [0, 0, 0, 255], "the clear colour around the quads");
}

fn whole_texture(texture: u64) -> TexelCopyTexture {
    TexelCopyTexture {
        texture,
        mip_level: 0,
        origin: (0, 0, 0),
        aspect: "all".into(),
    }
}

fn texture_2d(id: u64, width: u32, height: u32, format: &str) -> Command {
    const COPY_SRC: u32 = 0x1;
    const COPY_DST: u32 = 0x2;
    Command::CreateTexture {
        device: DEVICE,
        id,
        descriptor: TextureDescriptor {
            width,
            height,
            depth_or_array_layers: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: "2d".into(),
            format: format.into(),
            usage: COPY_SRC | COPY_DST,
            view_formats: Vec::new(),
        },
    }
}

/// Copies a texture into a readable buffer whose rows are 256 bytes apart, and maps it.
fn read_texture_rows(server: &mut Server, texture: u64, width: u32, height: u32) -> Vec<u8> {
    const COPY_DST: u32 = 0x8;
    const MAP_READ: u32 = 0x1;
    let size = 256 * u64::from(height);
    let commands = [
        Command::CreateBuffer {
            device: DEVICE,
            id: 90,
            size,
            usage: COPY_DST | MAP_READ,
            mapped_at_creation: false,
        },
        Command::CreateCommandEncoder { device: DEVICE, id: 91 },
        Command::CopyTextureToBuffer {
            encoder: 91,
            texture: whole_texture(texture),
            buffer: 90,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
            size: (width, height, 1),
        },
        Command::FinishCommandEncoder { encoder: 91, id: 92 },
        Command::Submit {
            device: DEVICE,
            command_buffers: vec![92],
        },
    ];
    assert_eq!(run(server, &commands), Vec::new(), "no errors");
    let reply = request(
        server,
        Request::MapBuffer {
            buffer: 90,
            for_writing: false,
            offset: 0,
            size,
        },
    );
    run(server, &[Command::Release { id: 90 }]);
    match reply {
        Reply::MapBuffer {
            result: Ok(Blob(bytes)),
        } => bytes,
        other => panic!("cannot read the texture back: {other:?}"),
    }
}

/// Texels go into a texture through the queue, to another texture by a copy, and back out through a buffer; an
/// external image's texels are converted to the destination's format.
#[test]
fn textures_round_trip() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    let texels: Vec<u8> = (1..=16).collect();
    let commands = [
        texture_2d(40, 2, 2, "rgba8unorm"),
        texture_2d(41, 2, 2, "rgba8unorm"),
        Command::WriteTexture {
            device: DEVICE,
            texture: whole_texture(40),
            data: Blob(texels.clone()),
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: None,
            },
            size: (2, 2, 1),
        },
        Command::CreateCommandEncoder { device: DEVICE, id: 42 },
        Command::CopyTextureToTexture {
            encoder: 42,
            source: whole_texture(40),
            destination: whole_texture(41),
            size: (2, 2, 1),
        },
        Command::FinishCommandEncoder { encoder: 42, id: 43 },
        Command::Submit {
            device: DEVICE,
            command_buffers: vec![43],
        },
    ];
    assert_eq!(run(&mut server, &commands), Vec::new(), "no errors");
    let rows = read_texture_rows(&mut server, 41, 2, 2);
    assert_eq!(&rows[..8], &texels[..8]);
    assert_eq!(&rows[256..264], &texels[8..]);

    let external = |texture: u64| Command::CopyExternalImageToTexture {
        device: DEVICE,
        texture: whole_texture(texture),
        width: 1,
        height: 1,
        rgba8: Blob(vec![255, 0, 51, 255]),
    };
    let commands = [
        texture_2d(44, 1, 1, "bgra8unorm"),
        external(44),
        texture_2d(45, 1, 1, "rgba16float"),
        external(45),
    ];
    assert_eq!(run(&mut server, &commands), Vec::new(), "no errors");
    assert_eq!(&read_texture_rows(&mut server, 44, 1, 1)[..4], &[51, 0, 255, 255]);
    // 1.0, 0.0, 0.2 (nearest half: 0x3266) and 1.0 as halves.
    assert_eq!(
        &read_texture_rows(&mut server, 45, 1, 1)[..8],
        &[0x00, 0x3c, 0x00, 0x00, 0x66, 0x32, 0x00, 0x3c]
    );
}

/// A compute pass doubles a storage buffer, with an "auto" layout; an asynchronous pipeline with a missing entry point
/// answers with a validation error instead of raising it.
#[test]
fn a_compute_pass_doubles_a_buffer() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    const STORAGE: u32 = 0x80;
    const COPY_SRC: u32 = 0x4;
    const COPY_DST: u32 = 0x8;
    const MAP_READ: u32 = 0x1;
    let compute = |entry_point: &str| ComputePipelineDescriptor {
        layout: None,
        compute: stage(60, Some(entry_point)),
    };
    let commands = [
        Command::CreateShaderModule {
            device: DEVICE,
            id: 60,
            code: "@group(0) @binding(0) var<storage, read_write> data: array<u32>;\n\
                   @compute @workgroup_size(4) fn main(@builtin(global_invocation_id) id: vec3u) {\n\
                   data[id.x] = data[id.x] * 2u;\n}"
                .into(),
        },
        Command::CreateComputePipeline {
            device: DEVICE,
            id: 61,
            descriptor: compute("main"),
        },
        Command::GetBindGroupLayout {
            pipeline: 61,
            index: 0,
            id: 62,
        },
        Command::CreateBuffer {
            device: DEVICE,
            id: 63,
            size: 16,
            usage: STORAGE | COPY_SRC | COPY_DST,
            mapped_at_creation: false,
        },
        Command::WriteBuffer {
            device: DEVICE,
            buffer: 63,
            offset: 0,
            data: Blob([1u32, 2, 3, 4].iter().flat_map(|value| value.to_le_bytes()).collect()),
        },
        Command::CreateBindGroup {
            device: DEVICE,
            id: 64,
            layout: 62,
            entries: vec![BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: 63,
                    offset: 0,
                    size: None,
                },
            }],
        },
        Command::CreateBuffer {
            device: DEVICE,
            id: 65,
            size: 16,
            usage: MAP_READ | COPY_DST,
            mapped_at_creation: false,
        },
        Command::CreateCommandEncoder { device: DEVICE, id: 66 },
        Command::EncodeComputePass {
            encoder: 66,
            timestamp_writes: None,
            ops: vec![
                ComputePassOp::SetPipeline { pipeline: 61 },
                ComputePassOp::SetBindGroup {
                    index: 0,
                    group: Some(64),
                    dynamic_offsets: Vec::new(),
                },
                ComputePassOp::DispatchWorkgroups { x: 1, y: 1, z: 1 },
            ],
        },
        Command::CopyBufferToBuffer {
            encoder: 66,
            source: 63,
            source_offset: 0,
            destination: 65,
            destination_offset: 0,
            size: 16,
        },
        Command::FinishCommandEncoder { encoder: 66, id: 67 },
        Command::Submit {
            device: DEVICE,
            command_buffers: vec![67],
        },
    ];
    assert_eq!(run(&mut server, &commands), Vec::new(), "no errors");
    let doubled: Vec<u8> = [2u32, 4, 6, 8].iter().flat_map(|value| value.to_le_bytes()).collect();
    assert_eq!(
        request(
            &mut server,
            Request::MapBuffer {
                buffer: 65,
                for_writing: false,
                offset: 0,
                size: 16,
            }
        ),
        Reply::MapBuffer {
            result: Ok(Blob(doubled))
        }
    );

    let reply = request(
        &mut server,
        Request::CreateComputePipeline {
            device: DEVICE,
            id: 68,
            descriptor: compute("missing"),
        },
    );
    assert!(
        matches!(
            reply,
            Reply::Pipeline {
                error: Some(GpuError::Validation { .. })
            }
        ),
        "{reply:?}"
    );
    let reply = request(
        &mut server,
        Request::CreateComputePipeline {
            device: DEVICE,
            id: 69,
            descriptor: compute("main"),
        },
    );
    assert_eq!(reply, Reply::Pipeline { error: None });
}

/// A device destroyed or released with error scopes still pushed loses them innermost first, which wgpu requires
/// rather than panicking, and a destroyed device still reports that it was lost.
#[test]
fn a_device_with_error_scopes_can_go() {
    let Some(mut server) = server_with_device(ServerOptions {
        force_fallback_adapter: true,
        ..ServerOptions::default()
    }) else {
        return;
    };
    let push = |filter| Command::PushErrorScope { device: DEVICE, filter };
    let events = run(
        &mut server,
        &[
            push(ErrorFilter::Validation),
            push(ErrorFilter::OutOfMemory),
            push(ErrorFilter::Internal),
            Command::Destroy { id: DEVICE },
        ],
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            Event::DeviceLost {
                reason: DeviceLostReason::Destroyed,
                ..
            }
        )),
        "{events:?}"
    );
    run(&mut server, &[Command::Release { id: DEVICE }]);
}
