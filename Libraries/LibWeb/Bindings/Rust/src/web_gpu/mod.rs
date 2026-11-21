/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! WebGPU, https://gpuweb.github.io/gpuweb/
//!
//! Nothing here touches a GPU. Every call is recorded as a webgpu_wire::Command naming objects by ids allocated here,
//! and the compositor executes the commands (Services/Compositor/WebGPU). Requests, such as for an adapter or for
//! the error an error scope caught, are commands too, whose replies come back as events and settle promises.

/// Declares the enum of a WebGPU enumeration, whose variants the bindings name after its values, and the name of
/// each value in the spec, which the protocol carries.
macro_rules! string_enum {
    ($name:ident, $anchor:literal { $($variant:ident => $value:literal,)* }) => {
        #[doc = concat!("https://gpuweb.github.io/gpuweb/#enumdef-", $anchor)]
        #[derive(Clone, Copy, PartialEq, Eq)]
        #[allow(non_camel_case_types, reason = "the bindings name the variants after the values")]
        pub(crate) enum $name {
            $($variant,)*
        }

        impl $name {
            pub(crate) fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $value,)*
                }
            }
        }
    };
}
/// The label of https://gpuweb.github.io/gpuweb/#gpuobjectbase, which script can read and write.
// FIXME: Send labels to the compositor, so that its error messages name the objects.
pub(crate) struct Label(core::cell::RefCell<crate::bindings::conversions::UsvString>);

impl Default for Label {
    fn default() -> Self {
        Self(core::cell::RefCell::new(
            crate::bindings::conversions::UsvString::from_ascii(b""),
        ))
    }
}

impl Label {
    pub(crate) fn set(&self, label: &crate::bindings::conversions::UsvString) {
        self.0.replace(label.clone());
    }
}

/// An object that includes GPUObjectBase.
pub(crate) trait HasLabel {
    fn label_slot(&self) -> &Label;
}

/// The object, labelled as its descriptor asks: https://gpuweb.github.io/gpuweb/#abstract-opdef-gpuobjectbase-label
pub(crate) fn labelled<T: HasLabel + crate::bindings::interface::Interface>(
    object: crate::bindings::interface::Ref<T>,
    label: &crate::bindings::conversions::UsvString,
) -> crate::bindings::interface::Ref<T> {
    object.label_slot().set(label);
    object
}

/// The `label` attribute of the interfaces that include GPUObjectBase, for a struct with a `label: Label` field.
macro_rules! label_attribute {
    ($type:ident) => {
        impl crate::web_gpu::HasLabel for $type {
            fn label_slot(&self) -> &crate::web_gpu::Label {
                &self.label
            }
        }

        impl $type {
            // https://gpuweb.github.io/gpuweb/#dom-gpuobjectbase-label
            pub(crate) fn label(
                &self,
                _: &mut crate::bindings::CallContext,
            ) -> crate::bindings::WebIdlResult<crate::bindings::conversions::UsvString> {
                Ok(self.label.0.borrow().clone())
            }

            pub(crate) fn set_label(
                &self,
                _: &mut crate::bindings::CallContext,
                value: crate::bindings::conversions::UsvString,
            ) -> crate::bindings::WebIdlResult<()> {
                self.label.0.replace(value);
                Ok(())
            }
        }
    };
}

pub(crate) mod client;
pub(crate) mod exports;
pub(crate) mod gpu;
pub(crate) mod gpu_adapter;
pub(crate) mod gpu_adapter_info;
pub(crate) mod gpu_bind_group;
pub(crate) mod gpu_bind_group_layout;
pub(crate) mod gpu_buffer;
pub(crate) mod gpu_buffer_usage;
pub(crate) mod gpu_canvas_context;
pub(crate) mod gpu_color_write;
pub(crate) mod gpu_command_buffer;
pub(crate) mod gpu_command_encoder;
pub(crate) mod gpu_compilation_info;
pub(crate) mod gpu_compilation_message;
pub(crate) mod gpu_compute_pass_encoder;
pub(crate) mod gpu_compute_pipeline;
pub(crate) mod gpu_device;
pub(crate) mod gpu_device_lost_info;
pub(crate) mod gpu_error;
pub(crate) mod gpu_internal_error;
pub(crate) mod gpu_map_mode;
pub(crate) mod gpu_out_of_memory_error;
pub(crate) mod gpu_pipeline_layout;
pub(crate) mod gpu_query_set;
pub(crate) mod gpu_queue;
pub(crate) mod gpu_render_bundle;
pub(crate) mod gpu_render_bundle_encoder;
pub(crate) mod gpu_render_pass_encoder;
pub(crate) mod gpu_render_pipeline;
pub(crate) mod gpu_sampler;
pub(crate) mod gpu_shader_module;
pub(crate) mod gpu_shader_stage;
pub(crate) mod gpu_supported_features;
pub(crate) mod gpu_supported_limits;
pub(crate) mod gpu_texture;
pub(crate) mod gpu_texture_usage;
pub(crate) mod gpu_texture_view;
pub(crate) mod gpu_uncaptured_error_event;
pub(crate) mod gpu_validation_error;
pub(crate) mod wgsl_language_features;
