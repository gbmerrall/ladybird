/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <AK/ByteBuffer.h>
#include <LibGfx/Bitmap.h>
#include <LibGfx/DecodedImageFrame.h>
#include <LibJS/HostClassBuilder.h>
#include <LibWeb/Bindings/MainThreadVM.h>
#include <LibWeb/Compositor/CompositorHost.h>
#include <LibWeb/HTML/Window.h>
#include <LibWeb/Page/Page.h>
#include <LibWeb/WebGL/RemoteWebGLTransport.h>
#include <LibWeb/WebGPU/WebGPUClient.h>

// What Bindings/Rust/src/web_gpu/client.rs asks of LibWeb: the connection to the compositor of a window's page.

namespace Web::WebGPU {

extern "C" {
// Defined in Bindings/Rust/src/web_gpu/client.rs.
void web_rust_webgpu_handle_events(JSVM*, u8 const* data, size_t length);
void web_rust_webgpu_compositor_lost(JSVM*);
}

void handle_events_from_compositor(ReadonlyBytes events)
{
    web_rust_webgpu_handle_events(JS::HostABI::vm_to_abi(Bindings::main_thread_vm()), events.data(), events.size());
}

void compositor_was_lost()
{
    web_rust_webgpu_compositor_lost(JS::HostABI::vm_to_abi(Bindings::main_thread_vm()));
}

static Compositor::CompositorHost* compositor_host_of(HTML::Window* window)
{
    if (!window)
        return nullptr;
    return &window->page().compositor_host();
}

extern "C" {

void web_webgpu_send_commands(HTML::Window* window, u8 const* data, size_t length);
void web_webgpu_send_commands(HTML::Window* window, u8 const* data, size_t length)
{
    if (auto* host = compositor_host_of(window))
        host->send_webgpu_commands({ data, length });
}

bool web_webgpu_has_compositor(HTML::Window* window);
bool web_webgpu_has_compositor(HTML::Window* window)
{
    auto* host = compositor_host_of(window);
    return host && host->has_compositor_connection();
}

// The pixels of a frame that the WebGL readers of texture image sources return. A video frame has none here: its
// pixels stay in the compositor.
static Gfx::DecodedImageFrame const* decoded_image_frame(void const* frame)
{
    auto const& source_frame = static_cast<WebGL::ImageSourceFrameOwnedByRecorder const*>(frame)->frame;
    if (!source_frame.has<Gfx::DecodedImageFrame>())
        return nullptr;
    return &source_frame.get<Gfx::DecodedImageFrame>();
}

bool web_webgpu_frame_size(void const* frame, u32* out_width, u32* out_height);
bool web_webgpu_frame_size(void const* frame, u32* out_width, u32* out_height)
{
    auto const* image_frame = decoded_image_frame(frame);
    if (!image_frame)
        return false;
    *out_width = image_frame->size().width();
    *out_height = image_frame->size().height();
    return true;
}

// Copies the width × height texels at (x, y) of the frame, after flipping it vertically if asked, into `out` as RGBA8
// with premultiplied or unpremultiplied alpha. The rectangle must be inside the frame.
bool web_webgpu_read_frame_texels(void const* frame, u32 x, u32 y, u32 width, u32 height, bool flip_y, bool premultiplied, u8* out);
bool web_webgpu_read_frame_texels(void const* frame, u32 x, u32 y, u32 width, u32 height, bool flip_y, bool premultiplied, u8* out)
{
    auto const* image_frame = decoded_image_frame(frame);
    if (!image_frame)
        return false;
    auto bitmap = image_frame->bitmap_sharing_pixels();
    auto frame_height = static_cast<u32>(bitmap->height());
    VERIFY(x + width <= static_cast<u32>(bitmap->width()) && y + height <= frame_height);
    bool source_is_premultiplied = bitmap->alpha_type() == Gfx::AlphaType::Premultiplied;
    for (u32 row = 0; row < height; ++row) {
        auto source_row = flip_y ? frame_height - 1 - (y + row) : y + row;
        for (u32 column = 0; column < width; ++column) {
            auto color = bitmap->get_pixel(static_cast<int>(x + column), static_cast<int>(source_row));
            u32 red = color.red(), green = color.green(), blue = color.blue(), alpha = color.alpha();
            if (source_is_premultiplied && !premultiplied) {
                if (alpha == 0) {
                    red = green = blue = 0;
                } else {
                    red = min(255u, (red * 255 + alpha / 2) / alpha);
                    green = min(255u, (green * 255 + alpha / 2) / alpha);
                    blue = min(255u, (blue * 255 + alpha / 2) / alpha);
                }
            } else if (!source_is_premultiplied && premultiplied) {
                red = (red * alpha + 127) / 255;
                green = (green * alpha + 127) / 255;
                blue = (blue * alpha + 127) / 255;
            }
            *out++ = red;
            *out++ = green;
            *out++ = blue;
            *out++ = alpha;
        }
    }
    return true;
}
}

}
