/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#pragma once

#include <AK/Span.h>

namespace Web::WebGPU {

// What the compositor sends WebContent about WebGPU, which Bindings/Rust/src/web_gpu handles: replies to requests,
// uncaptured errors and lost devices, as webgpu_wire events.
void handle_events_from_compositor(ReadonlyBytes events);

// Loses every device and fails every pending request, since no compositor will answer them.
void compositor_was_lost();

}
