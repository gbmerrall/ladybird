/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <LibJS/HostClassBuilder.h>
#include <LibJS/Runtime/Realm.h>
#include <LibWeb/WebGPU/GPU.h>

namespace Web::WebGPU {

extern "C" {
// Defined in Bindings/Rust/src/web_gpu/exports.rs.
void* web_rust_webgpu_create_gpu(JSVM*, void* realm);
}

GC::Ref<GPU> GPU::create(JS::Realm& realm)
{
    return *static_cast<GPU*>(web_rust_webgpu_create_gpu(JS::HostABI::vm_to_abi(realm.vm()), &realm));
}

}
