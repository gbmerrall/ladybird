/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpubuffer
//!
//! The contents of a buffer live in the compositor. Mapping one asks for the contents of the range, which WebContent
//! keeps while the buffer is mapped; unmapping a buffer mapped for writing sends back what script wrote.

use core::cell::{Cell, RefCell};

use ak::Utf16String;
use libjs_rust::api::{ArrayBuffer, ArrayBufferShared, Gc, GcRefCell, Realm, Trace, Value, Vm};
use webgpu_wire::{Blob, Command, ObjectId, Reply, Request};

use crate::bindings::conversions::copy_of_bytes_of_buffer;
use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::promise::{WebIdlPromise, reject_promise_with_exception, resolve_promise};
use crate::bindings::{CallContext, Exception, WebIdlResult};
use crate::web_gpu::Label;
use crate::web_gpu::client::{self, RequestSubject};
use crate::web_gpu::gpu::GPUObjectDescriptorBase;

pub(crate) struct GPUBufferDescriptor {
    pub(crate) parent: GPUObjectDescriptorBase,
    pub(crate) size: u64,
    pub(crate) usage: u32,
    pub(crate) mapped_at_creation: bool,
}

// https://gpuweb.github.io/gpuweb/#enumdef-gpubuffermapstate
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum GPUBufferMapState {
    Unmapped,
    Pending,
    Mapped,
}

const MAP_READ: u32 = 0x1;
const MAP_WRITE: u32 = 0x2;

/// The range of a mapped buffer, and its contents as script sees them.
struct Mapping {
    for_writing: bool,
    offset: u64,
    contents: Vec<u8>,
}

fn operation_error(message: &str) -> Exception {
    Exception::Dom("OperationError", Utf16String::from_utf8(message))
}

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUBuffer {
    realm: Gc<Realm>,
    #[gc(untraced)]
    pub(crate) label: Label,
    #[gc(untraced)]
    pub(crate) id: ObjectId,
    #[gc(untraced)]
    size: u64,
    #[gc(untraced)]
    usage: u32,
    #[gc(untraced)]
    map_state: Cell<GPUBufferMapState>,
    #[gc(untraced)]
    mapping: RefCell<Option<Mapping>>,
    /// The promise of the mapping that is pending, which unmapping rejects.
    pending_map: GcRefCell<Option<WebIdlPromise>>,
    /// What getMappedRange() returned, with the offsets in the buffer of the ranges, which unmapping detaches.
    mapped_ranges: GcRefCell<Vec<(Gc<ArrayBuffer>, u64)>>,
    /// The buffer itself, for the reply to a request to map it.
    this: Cell<Option<Ref<GPUBuffer>>>,
}

impl Wrappable for GPUBuffer {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }

    const FINALIZES: bool = true;

    fn finalize(object: Ref<Self>) {
        client::release_later(object.id);
    }
}

impl GPUBuffer {
    pub(crate) fn create(
        vm: &Vm,
        realm: Gc<Realm>,
        device: ObjectId,
        descriptor: &GPUBufferDescriptor,
    ) -> WebIdlResult<Ref<Self>> {
        // https://gpuweb.github.io/gpuweb/#dom-gpudevice-createbuffer
        // 2. If descriptor.mappedAtCreation is true and descriptor.size is not a multiple of 4, throw a RangeError.
        if descriptor.mapped_at_creation && !descriptor.size.is_multiple_of(4) {
            return Err(Exception::range_error(
                "A buffer mapped at creation must have a size that is a multiple of 4",
            ));
        }
        let id = client::allocate_object_id();
        client::send(
            realm,
            &Command::CreateBuffer {
                device,
                id,
                size: descriptor.size,
                usage: descriptor.usage,
                mapped_at_creation: descriptor.mapped_at_creation,
            },
        );
        let mapping = descriptor.mapped_at_creation.then(|| Mapping {
            for_writing: true,
            offset: 0,
            contents: vec![0; usize::try_from(descriptor.size).unwrap_or(0)],
        });
        let map_state = if mapping.is_some() {
            GPUBufferMapState::Mapped
        } else {
            GPUBufferMapState::Unmapped
        };
        let buffer = allocate(vm, |_| Self {
            label: Label::default(),
            realm,
            id,
            size: descriptor.size,
            usage: descriptor.usage,
            map_state: Cell::new(map_state),
            mapping: RefCell::new(mapping),
            pending_map: GcRefCell::default(),
            mapped_ranges: GcRefCell::default(),
            this: Cell::new(None),
        });
        buffer.this.set(Some(buffer));
        Ok(buffer)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpubuffer-size
    pub(crate) fn size(&self, _: &mut CallContext) -> WebIdlResult<u64> {
        Ok(self.size)
    }

    pub(crate) fn size_in_bytes(&self) -> u64 {
        self.size
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpubuffer-usage
    pub(crate) fn usage(&self, _: &mut CallContext) -> WebIdlResult<u32> {
        Ok(self.usage)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpubuffer-mapstate
    pub(crate) fn map_state(&self, _: &mut CallContext) -> WebIdlResult<GPUBufferMapState> {
        Ok(self.map_state.get())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpubuffer-mapasync
    pub(crate) fn map_async(
        &self,
        cx: &mut CallContext,
        promise: WebIdlPromise,
        mode: u32,
        offset: u64,
        size: Option<u64>,
    ) -> WebIdlResult<()> {
        let vm = cx.vm();
        // 4. If this.mapState is not "unmapped", issue an error and reject with an OperationError.
        if self.map_state.get() != GPUBufferMapState::Unmapped {
            reject_promise_with_exception(
                vm,
                promise,
                operation_error("The buffer is already mapped or being mapped"),
            );
            return Ok(());
        }
        let size = size.unwrap_or_else(|| self.size.saturating_sub(offset));
        let for_writing = mode == MAP_WRITE;
        // NB: The spec's checks of the range and the mode are on the device timeline; their failure rejects the same.
        let valid = (mode == MAP_READ || mode == MAP_WRITE)
            && self.usage & mode != 0
            && offset.is_multiple_of(8)
            && size.is_multiple_of(4)
            && offset.checked_add(size).is_some_and(|end| end <= self.size);
        if !valid {
            reject_promise_with_exception(vm, promise, operation_error("The buffer cannot be mapped that way"));
            return Ok(());
        }

        self.map_state.set(GPUBufferMapState::Pending);
        *self.pending_map.borrow_mut(vm) = Some(promise);
        fn steps(vm: &Vm, request: &client::PendingRequest, reply: Option<Reply>) {
            let RequestSubject::Buffer(buffer, for_writing, offset) = request.subject else {
                return;
            };
            // NB: Unmapping or destroying the buffer while the mapping was pending rejected the promise already.
            if buffer.pending_map.borrow_mut(vm).take().is_none() {
                return;
            }
            match reply {
                Some(Reply::MapBuffer { result: Ok(contents) }) => {
                    *buffer.mapping.borrow_mut() = Some(Mapping {
                        for_writing,
                        offset,
                        contents: contents.0,
                    });
                    buffer.map_state.set(GPUBufferMapState::Mapped);
                    resolve_promise(vm, request.promise, Value::UNDEFINED);
                }
                Some(Reply::MapBuffer { result: Err(message) }) => {
                    buffer.map_state.set(GPUBufferMapState::Unmapped);
                    reject_promise_with_exception(vm, request.promise, operation_error(&message));
                }
                _ => {
                    buffer.map_state.set(GPUBufferMapState::Unmapped);
                    reject_promise_with_exception(vm, request.promise, operation_error("The device is lost"));
                }
            }
        }
        let this = self.this.get().expect("a buffer knows itself once it is allocated");
        client::request_about(
            vm,
            self.realm,
            promise,
            self.id,
            Request::MapBuffer {
                buffer: self.id,
                for_writing,
                offset,
                size,
            },
            steps,
            RequestSubject::Buffer(this, for_writing, offset),
        );
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpubuffer-getmappedrange
    pub(crate) fn get_mapped_range(
        &self,
        cx: &mut CallContext,
        offset: u64,
        size: Option<u64>,
    ) -> WebIdlResult<Gc<ArrayBuffer>> {
        let mapping = self.mapping.borrow();
        let Some(mapping) = mapping
            .as_ref()
            .filter(|_| self.map_state.get() == GPUBufferMapState::Mapped)
        else {
            return Err(operation_error("The buffer is not mapped"));
        };
        let mapping_end = mapping.offset + mapping.contents.len() as u64;
        let size = size.unwrap_or_else(|| mapping_end.saturating_sub(offset));
        let end = offset.checked_add(size);
        let in_range = offset.is_multiple_of(8)
            && size.is_multiple_of(4)
            && offset >= mapping.offset
            && end.is_some_and(|end| end <= mapping_end);
        if !in_range {
            return Err(operation_error("The range is not in the mapped range"));
        }
        let end = end.unwrap_or(offset);
        let overlaps = self.mapped_ranges.borrow().iter().any(|(range, range_offset)| {
            let range_end = range_offset + range.byte_length() as u64;
            offset < range_end && *range_offset < end
        });
        if overlaps {
            return Err(operation_error("The range overlaps a range that was returned already"));
        }
        let start = usize::try_from(offset - mapping.offset).unwrap_or(0);
        let bytes = &mapping.contents[start..start + usize::try_from(size).unwrap_or(0)];
        let array_buffer = ArrayBuffer::create_from_bytes(cx.vm(), cx.realm(), bytes, ArrayBufferShared::No);
        self.mapped_ranges.borrow_mut(cx.vm()).push((array_buffer, offset));
        Ok(array_buffer)
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpubuffer-unmap
    pub(crate) fn unmap(&self, cx: &mut CallContext) -> WebIdlResult<()> {
        self.end_mapping(cx.vm());
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#dom-gpubuffer-destroy
    pub(crate) fn destroy(&self, cx: &mut CallContext) -> WebIdlResult<()> {
        self.end_mapping(cx.vm());
        client::send(self.realm, &Command::Destroy { id: self.id });
        Ok(())
    }

    // https://gpuweb.github.io/gpuweb/#abstract-opdef-unmap
    fn end_mapping(&self, vm: &Vm) {
        // 1. If buffer.[[pending_map]] is not null, reject it with an AbortError.
        if let Some(promise) = self.pending_map.borrow_mut(vm).take() {
            reject_promise_with_exception(
                vm,
                promise,
                Exception::Dom(
                    "AbortError",
                    Utf16String::from_utf8("The buffer was unmapped while it was being mapped"),
                ),
            );
        }
        self.map_state.set(GPUBufferMapState::Unmapped);
        let Some(mut mapping) = self.mapping.borrow_mut().take() else {
            return;
        };
        let ranges = core::mem::take(&mut *self.mapped_ranges.borrow_mut(vm));
        for (array_buffer, range_offset) in &ranges {
            if mapping.for_writing && !array_buffer.is_detached() {
                let length = array_buffer.byte_length();
                if let Some(bytes) = copy_of_bytes_of_buffer(array_buffer, 0..length) {
                    let start = usize::try_from(range_offset - mapping.offset).unwrap_or(0);
                    mapping.contents[start..start + bytes.len()].copy_from_slice(&bytes);
                }
            }
            // 3. Detach each ArrayBuffer of buffer.[[mapping]].views.
            array_buffer.detach_buffer(vm);
        }
        if mapping.for_writing {
            client::send(
                self.realm,
                &Command::UnmapBuffer {
                    buffer: self.id,
                    offset: mapping.offset,
                    data: Blob(mapping.contents),
                },
            );
        }
    }
}

label_attribute!(GPUBuffer);
