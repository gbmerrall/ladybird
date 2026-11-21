/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <LibGC/Ptr.h>
#include <LibJS/Forward.h>
#include <LibWeb/Bindings/GPU.h>
#include <LibWeb/Bindings/RustInterface.h>
#include <LibWeb/Forward.h>

namespace Web::WebGPU {

// https://gpuweb.github.io/gpuweb/#gpu-interface
// NB: WebGPU is implemented in Rust, in Bindings/Rust/src/web_gpu. This names its objects for Navigator.
class GPU : public Bindings::RustInterfaceCell {
    WEB_RUST_INTERFACE(GPU, gpu)

public:
    static GC::Ref<GPU> create(JS::Realm&);
};

}
