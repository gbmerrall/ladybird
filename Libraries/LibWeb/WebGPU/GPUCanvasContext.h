/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/Optional.h>
#include <LibCompositing/DisplayList/DisplayListResourceIds.h>
#include <LibGC/Ptr.h>
#include <LibJS/Forward.h>
#include <LibWeb/Bindings/GPUCanvasContext.h>
#include <LibWeb/Bindings/RustInterface.h>
#include <LibWeb/Forward.h>

namespace Web::WebGPU {

// https://gpuweb.github.io/gpuweb/#gpucanvascontext
class GPUCanvasContext : public Bindings::RustInterfaceCell {
    WEB_RUST_INTERFACE(GPUCanvasContext, gpu_canvas_context)

public:
    static GC::Ref<GPUCanvasContext> create(JS::Realm&, HTML::HTMLCanvasElement&);

    // The compositor's name for the canvas, which it has once it is configured.
    Optional<Compositing::CanvasId> canvas_id() const;

    // Presents the texture script drew into since the last rendering update, if it drew into one.
    void prepare_for_compositing(HTML::HTMLCanvasElement&);
};

}
