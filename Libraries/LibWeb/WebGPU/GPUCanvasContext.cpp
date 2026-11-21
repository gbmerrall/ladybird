/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <LibJS/HostClassBuilder.h>
#include <LibJS/Runtime/Realm.h>
#include <LibWeb/Compositor/CompositorHost.h>
#include <LibWeb/DOM/Document.h>
#include <LibWeb/HTML/HTMLCanvasElement.h>
#include <LibWeb/Page/Page.h>
#include <LibWeb/WebGPU/GPUCanvasContext.h>

namespace Web::WebGPU {

extern "C" {
// Defined in Bindings/Rust/src/web_gpu/exports.rs.
void* web_rust_webgpu_create_canvas_context(JSVM*, void* realm, void* canvas, void* canvas_host, u64 canvas_id);
u64 web_rust_webgpu_canvas_context_canvas_id(void* context);
bool web_rust_webgpu_canvas_context_take_current_texture(void* context);
}

extern "C" {
// The compositor's name for a new WebGPU canvas, or 0 without a compositor.
u64 web_webgpu_create_compositor_canvas(HTML::HTMLCanvasElement* canvas);
u64 web_webgpu_create_compositor_canvas(HTML::HTMLCanvasElement* canvas)
{
    auto canvas_id = canvas->document().page().compositor_host().create_webgpu_canvas();
    return canvas_id.has_value() ? canvas_id->value() : 0;
}
}

GC::Ref<GPUCanvasContext> GPUCanvasContext::create(JS::Realm& realm, HTML::HTMLCanvasElement& canvas)
{
    // NB: The context asks the compositor for a canvas when it is first configured, since a page can make it before
    //     its compositor is connected.
    HTML::CanvasHost& canvas_host = canvas;
    auto* context = web_rust_webgpu_create_canvas_context(JS::HostABI::vm_to_abi(realm.vm()), &realm, &canvas, &canvas_host, 0);
    return *static_cast<GPUCanvasContext*>(context);
}

Optional<Compositing::CanvasId> GPUCanvasContext::canvas_id() const
{
    auto canvas_id = web_rust_webgpu_canvas_context_canvas_id(const_cast<GPUCanvasContext*>(this));
    if (canvas_id == 0)
        return {};
    return Compositing::CanvasId { canvas_id };
}

void GPUCanvasContext::prepare_for_compositing(HTML::HTMLCanvasElement& canvas)
{
    if (!web_rust_webgpu_canvas_context_take_current_texture(this))
        return;
    if (auto canvas_id = this->canvas_id(); canvas_id.has_value())
        canvas.document().page().compositor_host().present_webgpu_canvas(*canvas_id);
}

}
