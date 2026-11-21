/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! The compositor's side of WebGPU. See `backend` for the seam behind which wgpu sits.

pub mod backend;
pub mod ffi;
pub mod server;
pub mod wgpu_backend;

#[cfg(test)]
mod tests;
