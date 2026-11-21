/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#wgsllanguagefeatures

use ak::Utf16String;
use libjs_rust::api::{Gc, PrimitiveString, Realm, Set, Trace, Value, Vm};

use crate::bindings::CallContext;
use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::setlike::{Setlike, WorldEntries};

/// The WGSL language extensions that the compositor's shader compiler implements. `navigator.gpu` reports them before
/// there is an adapter to ask, so they are listed here.
// FIXME: Ask the compositor once it is connected, so that this cannot drift from the compiler it runs.
const NAMES: &[&str] = &[
    "readonly_and_readwrite_storage_textures",
    "packed_4x8_integer_dot_product",
    "pointer_composite_access",
];

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct WGSLLanguageFeatures {
    realm: Gc<Realm>,
    set_entries: WorldEntries<Set>,
}

impl Wrappable for WGSLLanguageFeatures {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl WGSLLanguageFeatures {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>) -> Ref<Self> {
        allocate(vm, |_| Self {
            realm,
            set_entries: WorldEntries::default(),
        })
    }
}

impl Setlike for WGSLLanguageFeatures {
    fn set_entries(&self, cx: &CallContext) -> Gc<Set> {
        self.set_entries.get_or_create(cx, || {
            let set = Set::create(cx.vm(), cx.realm());
            for name in NAMES {
                let string = PrimitiveString::create(cx.vm(), Utf16String::from_utf8(name));
                set.set_add(cx.vm(), Value::from_string(string));
            }
            set
        })
    }

    fn set_entries_size(&self) -> usize {
        NAMES.len()
    }

    fn set_entries_contain(&self, value: Value) -> bool {
        let name = value.as_string().to_utf8();
        NAMES.contains(&name.as_str())
    }
}
