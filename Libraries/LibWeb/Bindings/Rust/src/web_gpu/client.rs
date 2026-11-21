/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! The connection to the compositor, through LibWeb's C++ (Libraries/LibWeb/WebGPU/WebGPUClient.cpp): commands go
//! out, events come back. A request is a command with an id, whose reply event settles a promise.

use core::cell::RefCell;
use core::ffi::c_void;
use core::ptr;
use core::sync::atomic::{AtomicU64, Ordering};
use std::collections::HashMap;

use libjs_rust::api::{CellRoot, EmbedderCell, EmbedderClass, EmbedderValue, Gc, Realm, Trace, Vm};
use webgpu_wire::{Command, DeviceLostReason, Event, ObjectId, Reply, Request, RequestId, Writer};

use crate::bindings::global_object::window_from_global_object;
use crate::bindings::interface::{Ref, with_temporary_execution_context};
use crate::bindings::promise::{TaskSource, WebIdlPromise, queue_a_global_task};
use crate::web_gpu::gpu_adapter_info::GPUAdapterInfo;
use crate::web_gpu::gpu_buffer::GPUBuffer;
use crate::web_gpu::gpu_compute_pipeline::GPUComputePipeline;
use crate::web_gpu::gpu_device::GPUDevice;
use crate::web_gpu::gpu_render_pipeline::GPURenderPipeline;

unsafe extern "C" {
    fn web_webgpu_send_commands(window: *mut c_void, data: *const u8, len: usize);
    fn web_webgpu_has_compositor(window: *mut c_void) -> bool;
}

/// Ids name objects for the compositor's connection to this WebContent, so one counter for the process will do.
static NEXT_OBJECT_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// What to do with the reply to a request, or with its absence when the compositor is lost (None). It runs in a
/// task, in the realm of the request.
pub(crate) type ReplySteps = fn(vm: &Vm, request: &PendingRequest, reply: Option<Reply>);

/// What the reply to a request needs besides the promise.
#[derive(Clone, Copy, Trace)]
#[gc(crate = libjs_rust)]
pub(crate) enum RequestSubject {
    None,
    /// The info of the adapter that a device request asks, which its device reports too.
    AdapterInfo(Ref<GPUAdapterInfo>),
    /// The buffer to map, whether for writing, and the offset of the range.
    Buffer(Ref<GPUBuffer>, bool, u64),
    /// The pipeline that `create*PipelineAsync()` makes.
    RenderPipeline(Ref<GPURenderPipeline>),
    ComputePipeline(Ref<GPUComputePipeline>),
}

/// A request waiting for its reply. The pending list roots it, so its promise lives until then.
#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct PendingRequest {
    pub(crate) realm: Gc<Realm>,
    pub(crate) promise: WebIdlPromise,
    /// The object the request makes or asks about, such as the id of the adapter it requests.
    #[gc(untraced)]
    pub(crate) object: ObjectId,
    /// The object whose request this is, which its reply needs.
    pub(crate) subject: RequestSubject,
    #[gc(untraced)]
    steps: ReplySteps,
    #[gc(untraced)]
    reply: RefCell<Option<Reply>>,
}

static PENDING_REQUEST_CLASS: EmbedderClass<PendingRequest> = EmbedderClass::new("WebGPUPendingRequest");

impl EmbedderValue for PendingRequest {
    const CLASS: &'static EmbedderClass<Self> = &PENDING_REQUEST_CLASS;
}

thread_local! {
    /// The ids of objects whose wrappers were collected, which the next message to the compositor releases.
    /// Finalizers run during a collection, which is no time to send a message. WebGPU lives on the main thread.
    static PENDING_RELEASES: RefCell<Vec<ObjectId>> = const { RefCell::new(Vec::new()) };

    static PENDING_REQUESTS: RefCell<HashMap<RequestId, CellRoot<EmbedderCell<PendingRequest>>>> =
        RefCell::new(HashMap::new());

    /// The live devices by id, for the events about them. A device leaves the map when it is finalized, so the map
    /// does not keep it alive.
    static DEVICES: RefCell<HashMap<ObjectId, Ref<GPUDevice>>> = RefCell::new(HashMap::new());
}

/// Releases the object in the compositor with the next message, for the finalizer of its wrapper.
pub(crate) fn release_later(id: ObjectId) {
    PENDING_RELEASES.with_borrow_mut(|ids| ids.push(id));
}

pub(crate) fn register_device(device: Ref<GPUDevice>) {
    DEVICES.with_borrow_mut(|devices| devices.insert(device.id, device));
}

/// For the finalizer of a device.
pub(crate) fn unregister_device(id: ObjectId) {
    DEVICES.with_borrow_mut(|devices| devices.remove(&id));
}

fn device_by_id(id: ObjectId) -> Option<Ref<GPUDevice>> {
    DEVICES.with_borrow(|devices| devices.get(&id).copied())
}

fn encode_pending_releases(writer: &mut Writer) {
    for id in PENDING_RELEASES.take() {
        webgpu_wire::encode_command(writer, &Command::Release { id });
    }
}

/// The Window of the realm, whose page has the connection to the compositor, or null in a worker.
// FIXME: WebGPU in workers.
fn window_of(realm: Gc<Realm>) -> *mut c_void {
    window_from_global_object(realm.global_object()).map_or(ptr::null_mut(), |window| window.as_ptr())
}

/// Whether the realm's page is connected to its compositor yet, which a new WebContent is only after the compositor
/// starts.
pub(crate) fn has_compositor(realm: Gc<Realm>) -> bool {
    // SAFETY: The window is live or null.
    unsafe { web_webgpu_has_compositor(window_of(realm)) }
}

pub(crate) fn allocate_object_id() -> ObjectId {
    NEXT_OBJECT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Sends a command to the compositor of the realm's page. LibWeb batches them (CompositorHost::send_webgpu_commands).
pub(crate) fn send(realm: Gc<Realm>, command: &Command) {
    let mut writer = Writer::new();
    encode_pending_releases(&mut writer);
    webgpu_wire::encode_command(&mut writer, command);
    let bytes = writer.bytes();
    // SAFETY: The window is live or null, and the bytes live for the call.
    unsafe { web_webgpu_send_commands(window_of(realm), bytes.as_ptr(), bytes.len()) };
}

/// Sends a request, whose reply runs `steps` in a task. Without a compositor, they run with no reply.
pub(crate) fn request(
    vm: &Vm,
    realm: Gc<Realm>,
    promise: WebIdlPromise,
    object: ObjectId,
    body: Request,
    steps: ReplySteps,
) {
    request_about(vm, realm, promise, object, body, steps, RequestSubject::None);
}

/// A request whose reply needs the subject.
pub(crate) fn request_about(
    vm: &Vm,
    realm: Gc<Realm>,
    promise: WebIdlPromise,
    object: ObjectId,
    body: Request,
    steps: ReplySteps,
    subject: RequestSubject,
) {
    let pending = PENDING_REQUEST_CLASS.allocate(vm, |_| PendingRequest {
        realm,
        promise,
        object,
        subject,
        steps,
        reply: RefCell::new(None),
    });
    if !has_compositor(realm) {
        settle_in_a_task(vm, pending, None);
        return;
    }
    let request = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    PENDING_REQUESTS.with_borrow_mut(|requests| requests.insert(request, CellRoot::new(pending)));
    send(realm, &Command::Request { request, body });
}

fn settle_in_a_task(vm: &Vm, pending: Gc<EmbedderCell<PendingRequest>>, reply: Option<Reply>) {
    *pending.reply.borrow_mut() = reply;
    queue_a_global_task(
        vm,
        pending.realm.global_object(),
        TaskSource::Unspecified,
        pending,
        |vm, pending| {
            let reply = pending.reply.borrow_mut().take();
            with_temporary_execution_context(pending.realm, || (pending.steps)(vm, &pending, reply));
        },
    );
}

/// The events of one message from the compositor.
///
/// # Safety
///
/// `vm` must be the live main thread VM, and `data` must point to `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_rust_webgpu_handle_events(vm: *mut c_void, data: *const u8, len: usize) {
    // SAFETY: The caller passes the live VM and the bytes of the message.
    let (vm, bytes) = unsafe { (&*vm.cast::<Vm>(), ak::slice_from_raw_parts(data, len)) };
    let events = match webgpu_wire::decode_events(bytes) {
        Ok(events) => events,
        Err(error) => {
            eprintln!("WebGPU: {error}");
            return;
        }
    };
    for event in events {
        match event {
            Event::Reply { request, reply } => {
                let Some(pending) = PENDING_REQUESTS.with_borrow_mut(|requests| requests.remove(&request)) else {
                    continue;
                };
                settle_in_a_task(vm, pending.cell(), Some(reply));
            }
            Event::UncapturedError { device: id, error } => {
                if let Some(device) = device_by_id(id) {
                    GPUDevice::dispatch_uncaptured_error(device, vm, error);
                }
            }
            Event::DeviceLost {
                device: id,
                reason,
                message,
            } => {
                if let Some(device) = device_by_id(id) {
                    device.lose(vm, reason, &message);
                }
            }
        }
    }
}

/// Fails the pending requests and loses every device, since no compositor will answer.
///
/// # Safety
///
/// `vm` must be the live main thread VM.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn web_rust_webgpu_compositor_lost(vm: *mut c_void) {
    // SAFETY: The caller passes the live VM.
    let vm = unsafe { &*vm.cast::<Vm>() };
    let pending: Vec<_> = PENDING_REQUESTS.with_borrow_mut(|requests| requests.drain().map(|(_, root)| root).collect());
    for root in pending {
        settle_in_a_task(vm, root.cell(), None);
    }
    let devices: Vec<_> = DEVICES.with_borrow(|devices| devices.values().copied().collect());
    for device in devices {
        device.lose(vm, DeviceLostReason::Unknown, "The compositor was lost");
    }
}
