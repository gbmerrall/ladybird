/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! The entry points through which LibWeb's C++ code creates and drives WebGPU objects, declared in
//! Libraries/LibWeb/WebGPU/GPU.cpp and GPUCanvasContext.cpp.

use core::ffi::c_void;
use core::ptr::NonNull;

use crate::bindings::interface::{cell_from_abi, cell_into_abi, realm_arguments};
use crate::web_gl::canvas::Canvas;
use crate::web_gpu::gpu::GPU;
use crate::web_gpu::gpu_canvas_context::GPUCanvasContext;

/// # Safety
///
/// The pointers must be the live VM and a live realm.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_rust_webgpu_create_gpu(vm: *mut c_void, realm: *mut c_void) -> *mut c_void {
    // SAFETY: The caller passes the live VM and a live realm.
    let (vm, realm) = unsafe { realm_arguments(vm, realm) };
    cell_into_abi(GPU::create(vm, realm))
}

/// # Safety
///
/// The pointers must be the live VM, a live realm, and a live HTMLCanvasElement with the CanvasHost it is.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_rust_webgpu_create_canvas_context(
    vm: *mut c_void,
    realm: *mut c_void,
    canvas: *mut c_void,
    canvas_host: *mut c_void,
    canvas_id: u64,
) -> *mut c_void {
    // SAFETY: The caller passes the live VM and a live realm.
    let (vm, realm) = unsafe { realm_arguments(vm, realm) };
    let canvas = Canvas {
        cell: NonNull::new(canvas).expect("C++ passes the canvas"),
        host: NonNull::new(canvas_host).expect("C++ passes the canvas"),
        is_offscreen: false,
    };
    cell_into_abi(GPUCanvasContext::create(vm, realm, canvas, canvas_id))
}

/// # Safety
///
/// The context must be a live GPUCanvasContext.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_rust_webgpu_canvas_context_canvas_id(context: *mut c_void) -> u64 {
    // SAFETY: The caller passes a live GPUCanvasContext.
    unsafe { cell_from_abi::<GPUCanvasContext>(context) }.canvas_id()
}

/// # Safety
///
/// The context must be a live GPUCanvasContext.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_rust_webgpu_canvas_context_take_current_texture(context: *mut c_void) -> bool {
    // SAFETY: The caller passes a live GPUCanvasContext.
    unsafe { cell_from_abi::<GPUCanvasContext>(context) }.take_current_texture_for_presentation()
}
