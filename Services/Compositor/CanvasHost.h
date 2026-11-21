/*
 * Copyright (c) 2026, Aliaksandr Kalenik <kalenik.aliaksandr@gmail.com>
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <LibCompositing/DisplayList/DisplayListResources.h>

#include <AK/ByteBuffer.h>
#include <AK/Error.h>
#include <AK/HashMap.h>
#include <AK/NonnullOwnPtr.h>
#include <AK/NonnullRefPtr.h>
#include <AK/Optional.h>
#include <AK/OwnPtr.h>
#include <AK/RefPtr.h>
#include <AK/String.h>
#include <AK/Variant.h>
#include <AK/Vector.h>
#include <Compositor/DrawnResources.h>
#include <Compositor/Forward.h>
#include <Compositor/HostWebGLContext.h>
#include <Compositor/PresentableSurface.h>
#include <LibCompositing/DisplayList/DisplayListResourceIds.h>
#include <LibCompositing/Forward.h>
#include <LibCompositing/Types.h>
#include <LibCompositing/WebGL/ImageSourceFrame.h>
#include <LibCompositing/WebGL/Types.h>
#include <LibCompositing/WebGL/WebGLSharedCommandBuffer.h>
#include <LibCore/AnonymousBuffer.h>
#include <LibCore/Forward.h>
#include <LibGfx/Forward.h>
#include <LibGfx/ShareableBitmap.h>
#include <LibMedia/VideoSinkHandle.h>
#include <LibMediaCodecs/Forward.h>

#ifdef USE_VULKAN_DMABUF_IMAGES
#    include <LibGfx/VulkanImage.h>
#endif

class SkImage;

template<typename T>
class sk_sp;

namespace Compositing {

class CanvasSurfaceRegistry;

}

namespace Compositor {

class Canvas2DCommandPlayer;

namespace WebGPUFFI {
struct WebGPUHost;
struct WebGPUSharedImage;
}
class VideoFrameHolds;

class CanvasHost {
public:
    struct CreateWebGLContextResult {
        bool success { false };
        Compositing::CanvasId canvas_id { 0 };
        Vector<String> supported_extensions;
    };

    CanvasHost(RefPtr<Gfx::SkiaBackendContext>, CanvasSurfaceRegistry&, VideoFrameHolds const&);
    ~CanvasHost();

    bool create_2d_context(Compositing::CanvasId, Gfx::IntSize, bool alpha, bool will_read_frequently, Gfx::ColorSpace const&);
    CreateWebGLContextResult create_webgl_context(Compositing::WebGL::WebGLVersion, Gfx::IntSize, bool depth, bool stencil, bool antialias);
    void destroy_context(Compositing::CanvasId);

    [[nodiscard]] bool execute_canvas_2d_stream(Vector<Compositing::Canvas2DCommandStreamSegment> const&, Vector<Compositing::DisplayListFontDescription> const& fonts, Compositing::RegisteredTypefaces const&, Compositing::RegisteredImageFrames const&);
    void execute_webgl_commands(Compositing::CanvasId, ReadonlyBytes, Vector<Compositing::WebGL::ImageSourceFrame> const&);
    void set_webgl_shared_command_buffer(Compositing::CanvasId, Compositing::WebGL::WebGLSharedCommandBuffer);
    [[nodiscard]] bool execute_webgl_commands_from_shared_buffer(Compositing::CanvasId, u64 offset, u64 size_in_bytes, u64 flush_sequence_number, Vector<Compositing::WebGL::ImageSourceFrame> const&);
    ErrorOr<ByteBuffer> execute_webgl_sync_call(Compositing::CanvasId, ByteBuffer request);
    Compositing::WebGL::ReadPixelsResult webgl_read_pixels_robust_angle(Compositing::CanvasId, Compositing::WebGL::GLint x, Compositing::WebGL::GLint y, Compositing::WebGL::GLsizei width, Compositing::WebGL::GLsizei height, Compositing::WebGL::GLenum format, Compositing::WebGL::GLenum type, Compositing::WebGL::GLsizei buf_size, Core::AnonymousBuffer pixels);
    bool webgl_read_buffer_sub_data(Compositing::CanvasId, Compositing::WebGL::GLenum target, Compositing::WebGL::GLintptr offset, Compositing::WebGL::GLintptr size, Core::AnonymousBuffer data);

    // WebGPU, which the Rust crate in Services/Compositor/WebGPU executes. Its canvases present frames read back from
    // the GPU into a surface of their own.
    Compositing::CanvasId create_webgpu_canvas();
    // Returns false if the commands were malformed. The events for WebContent, if any, go in out_events.
    [[nodiscard]] bool execute_webgpu_commands(ReadonlyBytes, ByteBuffer& out_events);
    void present_webgpu_canvas(Compositing::CanvasId);
    // Shows the frames still on their way back from the GPU, as a screenshot needs.
    void wait_for_webgpu_frames();

    Compositing::CanvasSurfaceOwnerId canvas_surface_owner_id() const { return m_canvas_surface_owner_id; }

    void present_webgl_canvas(Compositing::CanvasId, bool preserve_drawing_buffer);
    void clear_webgl_drawing_buffer(Compositing::CanvasId);
    // The pixels come back in the color space.
    Gfx::ShareableBitmap read_back_pixels(Compositing::CanvasId, Gfx::IntRect, Gfx::AlphaType, Gfx::ColorSpace const& = {});
    RefPtr<Gfx::PaintingSurface> presented_surface(Compositing::CanvasId);

    void forget_video_sink(Media::VideoSinkHandle);

    static Gfx::ShareableBitmap read_back_surface(Gfx::PaintingSurface&, Gfx::IntRect, Gfx::AlphaType = Gfx::AlphaType::Premultiplied, Gfx::ColorSpace const& = {});

private:
    struct Canvas2DContext {
        NonnullOwnPtr<Canvas2DCommandPlayer> command_player;
        PresentableSurfacePool presented_surfaces;
        NonnullRefPtr<Gfx::PaintingSurface> presented_surface;
        bool has_uncommitted_commands { false };
    };
    using WebGLContext = NonnullOwnPtr<HostWebGLContext>;
    using Context = Variant<Canvas2DContext, WebGLContext>;

    Context* context(Compositing::CanvasId);
    OwnPtr<Canvas2DCommandPlayer> create_2d_command_player(Gfx::IntSize, bool alpha, bool will_read_frequently, Gfx::ColorSpace const&);
    static HostWebGLContext& as_webgl(Context&);
    void present_canvas_2d_context(Compositing::CanvasId, Canvas2DContext&);
    sk_sp<SkImage> held_video_frame_image(Media::VideoSinkHandle, u64 generation, RefPtr<Gfx::SkiaBackendContext> const& skia_backend_context_of_canvas);
    Vector<HostWebGLContext::UploadSource> resolve_webgl_upload_sources(Vector<Compositing::WebGL::ImageSourceFrame> const&) const;

    RefPtr<Gfx::SkiaBackendContext> m_skia_backend_context;
    CanvasSurfaceRegistry& m_canvas_surface_registry;
    Compositing::CanvasSurfaceOwnerId m_canvas_surface_owner_id;
    HashMap<Compositing::CanvasId, Context> m_contexts;

    WebGPUFFI::WebGPUHost* webgpu_host();
    WebGPUFFI::WebGPUHost* m_webgpu_host { nullptr };
    HashMap<Compositing::CanvasId, PresentableSurfacePool> m_webgpu_canvases;
    void show_finished_webgpu_frames(bool wait);
    void show_webgpu_frame(Compositing::CanvasId, Gfx::Bitmap const&);
#ifdef USE_VULKAN_DMABUF_IMAGES
    // The images of Skia's device that a WebGPU canvas renders into directly, in turn, so that its frames never reach
    // the CPU. The WebGPU host imports each as a dma-buf for one frame, and gives it back to be shown.
    struct WebGPUSharedImage {
        u64 handle { 0 };
        NonnullRefPtr<Gfx::VulkanImage> image;
        // The host is rendering into the image, or has a frame of it on the way back.
        bool is_lent { false };
        // The surface that last showed the image, which Skia may still be reading from.
        RefPtr<PresentableSurface> shown_surface;
        // How many frames of the canvas were shown since this one: Skia's reads of an image happen on its own queue,
        // so an image is only lent again once newer frames have replaced it.
        u32 frames_shown_since { 0 };
    };
    bool lend_webgpu_image(Compositing::CanvasId, Gfx::IntSize, WebGPUFFI::WebGPUSharedImage&);
    void take_back_webgpu_image(Compositing::CanvasId, u64 handle);
    void show_webgpu_image(Compositing::CanvasId, u64 handle);
    HashMap<Compositing::CanvasId, Vector<WebGPUSharedImage>> m_webgpu_shared_images;
    u64 m_next_webgpu_image_handle { 1 };
#endif
    // Checks again for frames that are still on their way back from the GPU.
    RefPtr<Core::Timer> m_webgpu_frame_timer;
    DrawnFonts m_fonts_of_latest_stream_batch;
    Compositing::RegisteredImageFrames const* m_image_frames_registered_while_stream_batch_plays { nullptr };

    VideoFrameHolds const& m_video_frame_holds;
    struct VideoFrameImage;
    HashMap<Media::VideoSinkHandle, NonnullOwnPtr<VideoFrameImage>> m_video_frame_images;
    bool m_drew_video_frame_in_segment { false };
};

}
