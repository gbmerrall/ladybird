/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! The C ABI that Services/Compositor/CanvasHost.cpp calls. Results are handed to callbacks, so no buffer
//! changes hands across the boundary.

use std::ffi::c_void;

use crate::backend::PresentedFrame;
use crate::server::{CanvasImageSource, Server, ServerOptions};
use crate::wgpu_backend::WgpuBackend;

/// An image of the compositor's renderer, shared as a single-plane dma-buf, whose `fd` the host takes over.
#[repr(C)]
pub struct WebGPUSharedImage {
    /// The compositor's name for the image, never 0.
    pub handle: u64,
    pub fd: i32,
    pub modifier: u64,
    pub stride: u64,
    pub offset: u64,
}

type AllocateImage = unsafe extern "C" fn(
    context: *mut c_void,
    canvas: u64,
    width: u32,
    height: u32,
    out_image: *mut WebGPUSharedImage,
) -> bool;
type ReleaseImage = unsafe extern "C" fn(context: *mut c_void, canvas: u64, image: u64);

/// The compositor's callbacks for its images.
struct ForeignImageSource {
    vendor: u32,
    device: u32,
    allocate: AllocateImage,
    release: ReleaseImage,
    context: *mut c_void,
}

impl CanvasImageSource for ForeignImageSource {
    fn gpu(&self) -> (u32, u32) {
        (self.vendor, self.device)
    }

    #[cfg(unix)]
    fn allocate(&mut self, canvas: u64, width: u32, height: u32) -> Option<crate::backend::SharedCanvasImage> {
        use std::os::fd::{FromRawFd, OwnedFd};

        let mut image = WebGPUSharedImage {
            handle: 0,
            fd: -1,
            modifier: 0,
            stride: 0,
            offset: 0,
        };
        // SAFETY: The compositor passed the callback and its context, which outlive the host.
        if !unsafe { (self.allocate)(self.context, canvas, width, height, &raw mut image) } || image.fd < 0 {
            return None;
        }
        Some(crate::backend::SharedCanvasImage {
            handle: image.handle,
            // SAFETY: The compositor hands over a dma-buf fd that nothing else owns.
            fd: unsafe { OwnedFd::from_raw_fd(image.fd) },
            modifier: image.modifier,
            stride: image.stride,
            offset: image.offset,
        })
    }

    fn release(&mut self, canvas: u64, image: u64) {
        // SAFETY: As for allocate.
        unsafe { (self.release)(self.context, canvas, image) };
    }
}

pub struct WebGPUHost {
    server: Server,
}

/// # Safety
/// `data` must point to `len` readable bytes, or be null with `len` zero.
unsafe fn bytes<'a>(data: *const u8, len: usize) -> &'a [u8] {
    // SAFETY: The caller guarantees `data` points to `len` bytes, or is null with `len` zero.
    unsafe { ak::slice_from_raw_parts(data, len) }
}

#[unsafe(no_mangle)]
pub extern "C" fn webgpu_host_create() -> *mut WebGPUHost {
    let server = Server::new(Box::new(WgpuBackend::new()), ServerOptions::from_environment());
    Box::into_raw(Box::new(WebGPUHost { server }))
}

/// # Safety
/// `host` must come from `webgpu_host_create` and not be used afterwards.
/// Lets canvases render into images of the compositor's renderer, on the GPU with these PCI vendor and device ids.
///
/// # Safety
/// `host` must be live, and the callbacks and their context must outlive it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn webgpu_host_set_image_source(
    host: *mut WebGPUHost,
    vendor: u32,
    device: u32,
    allocate: AllocateImage,
    release: ReleaseImage,
    context: *mut c_void,
) {
    // SAFETY: The caller guarantees the pointer.
    let host = unsafe { &mut *host };
    host.server.set_image_source(Box::new(ForeignImageSource {
        vendor,
        device,
        allocate,
        release,
        context,
    }));
}

/// # Safety
/// `host` must come from `webgpu_host_create` and not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn webgpu_host_destroy(host: *mut WebGPUHost) {
    if !host.is_null() {
        // SAFETY: The caller passes ownership of a host from webgpu_host_create.
        drop(unsafe { Box::from_raw(host) });
    }
}

/// Executes a stream of commands, and calls `events` with the encoded events for WebContent, if there are any.
/// Returns false if the stream was malformed.
///
/// # Safety
/// `host` must be live; `data` must point to `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn webgpu_host_handle_commands(
    host: *mut WebGPUHost,
    data: *const u8,
    len: usize,
    events: unsafe extern "C" fn(context: *mut c_void, data: *const u8, len: usize),
    context: *mut c_void,
) -> bool {
    // SAFETY: The caller guarantees both pointers.
    let (host, data) = unsafe { (&mut *host, bytes(data, len)) };
    match host.server.handle_commands(data) {
        Ok(produced) => {
            if !produced.is_empty() {
                let encoded = webgpu_wire::encode_events(&produced);
                // SAFETY: `events` is the caller's callback, called with bytes that live for the call.
                unsafe { events(context, encoded.as_ptr(), encoded.len()) };
            }
            true
        }
        Err(error) => {
            eprintln!("WebGPU: {error}");
            false
        }
    }
}

/// Starts reading back the canvas's frame, and returns true, or returns false if the canvas drew nothing since it
/// was last presented. `webgpu_host_take_frames` hands the frame over once the GPU is done.
///
/// # Safety
/// `host` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn webgpu_host_present(host: *mut WebGPUHost, canvas: u64) -> bool {
    // SAFETY: The caller guarantees the pointer.
    let host = unsafe { &mut *host };
    host.server.present(canvas)
}

/// Calls `frame` with the newest frame of each canvas that came back from the GPU, and returns whether frames are
/// still on their way. A frame is either the compositor's image it was rendered into (`image` is not 0), or pixels
/// (tightly packed rows of 4-byte pixels). With `wait`, waits for every frame on its way.
///
/// # Safety
/// `host` must be live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn webgpu_host_take_frames(
    host: *mut WebGPUHost,
    wait: bool,
    frame: unsafe extern "C" fn(
        context: *mut c_void,
        canvas: u64,
        image: u64,
        width: u32,
        height: u32,
        is_bgra: bool,
        data: *const u8,
        len: usize,
    ),
    context: *mut c_void,
) -> bool {
    // SAFETY: The caller guarantees the pointer.
    let host = unsafe { &mut *host };
    for (canvas, presented) in host.server.take_frames(wait) {
        // SAFETY: `frame` is the caller's callback, called with bytes that live for the call.
        unsafe {
            match presented {
                PresentedFrame::Pixels(readback) => frame(
                    context,
                    canvas,
                    0,
                    readback.width,
                    readback.height,
                    readback.format.is_bgra(),
                    readback.pixels.as_ptr(),
                    readback.pixels.len(),
                ),
                PresentedFrame::SharedImage(image) => frame(context, canvas, image, 0, 0, true, core::ptr::null(), 0),
            }
        }
    }
    host.server.has_pending_frames()
}
