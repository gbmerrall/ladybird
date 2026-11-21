/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/Types.h>

// Defined in Services/Compositor/WebGPU/src/ffi.rs, which compositor_core links in.

namespace Compositor::WebGPUFFI {

struct WebGPUHost;

extern "C" {
WebGPUHost* webgpu_host_create();
void webgpu_host_destroy(WebGPUHost*);
bool webgpu_host_handle_commands(WebGPUHost*, u8 const* data, size_t length, void (*events)(void* context, u8 const* data, size_t length), void* context);
bool webgpu_host_present(WebGPUHost*, u64 canvas_id);
// The image is the compositor's name for the image a frame was rendered into, or 0 for a frame of pixels.
bool webgpu_host_take_frames(WebGPUHost*, bool wait, void (*frame)(void* context, u64 canvas_id, u64 image, u32 width, u32 height, bool is_bgra, u8 const* data, size_t length), void* context);

// An image of the compositor's renderer, shared as a single-plane dma-buf, whose fd the host takes over.
struct WebGPUSharedImage {
    u64 handle;
    i32 fd;
    u64 modifier;
    u64 stride;
    u64 offset;
};
void webgpu_host_set_image_source(WebGPUHost*, u32 vendor, u32 device, bool (*allocate)(void* context, u64 canvas_id, u32 width, u32 height, WebGPUSharedImage* out_image), void (*release)(void* context, u64 canvas_id, u64 image), void* context);
}

}
