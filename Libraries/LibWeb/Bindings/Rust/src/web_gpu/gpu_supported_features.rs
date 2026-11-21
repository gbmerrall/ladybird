/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! https://gpuweb.github.io/gpuweb/#gpusupportedfeatures

use ak::Utf16String;
use libjs_rust::api::{Gc, PrimitiveString, Realm, Set, Trace, Value, Vm};

use crate::bindings::CallContext;
use crate::bindings::interface::{Ref, Wrappable, allocate};
use crate::bindings::setlike::{Setlike, WorldEntries};

macro_rules! for_each_feature {
    ($callback:ident) => {
        $callback! {
            CoreFeaturesAndLimits => "core-features-and-limits",
            DepthClipControl => "depth-clip-control",
            Depth32floatStencil8 => "depth32float-stencil8",
            TextureCompressionBc => "texture-compression-bc",
            TextureCompressionBcSliced3d => "texture-compression-bc-sliced-3d",
            TextureCompressionEtc2 => "texture-compression-etc2",
            TextureCompressionAstc => "texture-compression-astc",
            TextureCompressionAstcSliced3d => "texture-compression-astc-sliced-3d",
            TimestampQuery => "timestamp-query",
            IndirectFirstInstance => "indirect-first-instance",
            ShaderF16 => "shader-f16",
            Rg11b10ufloatRenderable => "rg11b10ufloat-renderable",
            Bgra8unormStorage => "bgra8unorm-storage",
            Float32Filterable => "float32-filterable",
            Float32Blendable => "float32-blendable",
            ClipDistances => "clip-distances",
            DualSourceBlending => "dual-source-blending",
        }
    };
}

macro_rules! define_feature_names {
    ($($variant:ident => $name:literal,)*) => {
        // https://gpuweb.github.io/gpuweb/#enumdef-gpufeaturename
        #[derive(Clone, Copy, PartialEq, Eq)]
        pub(crate) enum GPUFeatureName {
            $($variant,)*
        }

        impl GPUFeatureName {
            pub(crate) fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)*
                }
            }
        }
    };
}

for_each_feature!(define_feature_names);

#[derive(Trace)]
#[gc(crate = libjs_rust)]
pub(crate) struct GPUSupportedFeatures {
    realm: Gc<Realm>,
    #[gc(untraced)]
    names: Vec<String>,
    set_entries: WorldEntries<Set>,
}

impl Wrappable for GPUSupportedFeatures {
    fn relevant_realm(&self) -> Option<Gc<Realm>> {
        Some(self.realm)
    }
}

impl GPUSupportedFeatures {
    pub(crate) fn create(vm: &Vm, realm: Gc<Realm>, names: Vec<String>) -> Ref<Self> {
        allocate(vm, |_| Self {
            realm,
            names,
            set_entries: WorldEntries::default(),
        })
    }
}

impl Setlike for GPUSupportedFeatures {
    fn set_entries(&self, cx: &CallContext) -> Gc<Set> {
        self.set_entries.get_or_create(cx, || {
            let set = Set::create(cx.vm(), cx.realm());
            for name in &self.names {
                let string = PrimitiveString::create(cx.vm(), Utf16String::from_utf8(name));
                set.set_add(cx.vm(), Value::from_string(string));
            }
            set
        })
    }

    fn set_entries_size(&self) -> usize {
        self.names.len()
    }

    fn set_entries_contain(&self, value: Value) -> bool {
        let name = value.as_string().to_utf8();
        self.names.contains(&name)
    }
}
