/*
 * Copyright (c) 2026-present, the Ladybird developers.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

//! The values of WebGPU's enumerations, by their names in the spec, as wgpu's types. The protocol carries the names;
//! an unknown one is a validation error, as an invalid descriptor is.

use crate::backend::{BackendError, BackendResult};

/// Declares a function from the names of an enumeration to the values of a wgpu type.
macro_rules! names {
    ($function:ident -> $type:ty, $what:literal { $($name:literal => $value:expr,)* }) => {
        pub(crate) fn $function(name: &str) -> BackendResult<$type> {
            Ok(match name {
                $($name => $value,)*
                _ => return Err(BackendError(format!(concat!("\"{}\" is not a ", $what), name))),
            })
        }
    };
}

names!(texture_format -> wgpu::TextureFormat, "texture format" {
    "r8unorm" => wgpu::TextureFormat::R8Unorm,
    "r8snorm" => wgpu::TextureFormat::R8Snorm,
    "r8uint" => wgpu::TextureFormat::R8Uint,
    "r8sint" => wgpu::TextureFormat::R8Sint,
    "r16unorm" => wgpu::TextureFormat::R16Unorm,
    "r16snorm" => wgpu::TextureFormat::R16Snorm,
    "r16uint" => wgpu::TextureFormat::R16Uint,
    "r16sint" => wgpu::TextureFormat::R16Sint,
    "r16float" => wgpu::TextureFormat::R16Float,
    "rg8unorm" => wgpu::TextureFormat::Rg8Unorm,
    "rg8snorm" => wgpu::TextureFormat::Rg8Snorm,
    "rg8uint" => wgpu::TextureFormat::Rg8Uint,
    "rg8sint" => wgpu::TextureFormat::Rg8Sint,
    "r32uint" => wgpu::TextureFormat::R32Uint,
    "r32sint" => wgpu::TextureFormat::R32Sint,
    "r32float" => wgpu::TextureFormat::R32Float,
    "rg16unorm" => wgpu::TextureFormat::Rg16Unorm,
    "rg16snorm" => wgpu::TextureFormat::Rg16Snorm,
    "rg16uint" => wgpu::TextureFormat::Rg16Uint,
    "rg16sint" => wgpu::TextureFormat::Rg16Sint,
    "rg16float" => wgpu::TextureFormat::Rg16Float,
    "rgba8unorm" => wgpu::TextureFormat::Rgba8Unorm,
    "rgba8unorm-srgb" => wgpu::TextureFormat::Rgba8UnormSrgb,
    "rgba8snorm" => wgpu::TextureFormat::Rgba8Snorm,
    "rgba8uint" => wgpu::TextureFormat::Rgba8Uint,
    "rgba8sint" => wgpu::TextureFormat::Rgba8Sint,
    "bgra8unorm" => wgpu::TextureFormat::Bgra8Unorm,
    "bgra8unorm-srgb" => wgpu::TextureFormat::Bgra8UnormSrgb,
    "rgb9e5ufloat" => wgpu::TextureFormat::Rgb9e5Ufloat,
    "rgb10a2uint" => wgpu::TextureFormat::Rgb10a2Uint,
    "rgb10a2unorm" => wgpu::TextureFormat::Rgb10a2Unorm,
    "rg11b10ufloat" => wgpu::TextureFormat::Rg11b10Ufloat,
    "rg32uint" => wgpu::TextureFormat::Rg32Uint,
    "rg32sint" => wgpu::TextureFormat::Rg32Sint,
    "rg32float" => wgpu::TextureFormat::Rg32Float,
    "rgba16unorm" => wgpu::TextureFormat::Rgba16Unorm,
    "rgba16snorm" => wgpu::TextureFormat::Rgba16Snorm,
    "rgba16uint" => wgpu::TextureFormat::Rgba16Uint,
    "rgba16sint" => wgpu::TextureFormat::Rgba16Sint,
    "rgba16float" => wgpu::TextureFormat::Rgba16Float,
    "rgba32uint" => wgpu::TextureFormat::Rgba32Uint,
    "rgba32sint" => wgpu::TextureFormat::Rgba32Sint,
    "rgba32float" => wgpu::TextureFormat::Rgba32Float,
    "stencil8" => wgpu::TextureFormat::Stencil8,
    "depth16unorm" => wgpu::TextureFormat::Depth16Unorm,
    "depth24plus" => wgpu::TextureFormat::Depth24Plus,
    "depth24plus-stencil8" => wgpu::TextureFormat::Depth24PlusStencil8,
    "depth32float" => wgpu::TextureFormat::Depth32Float,
    "depth32float-stencil8" => wgpu::TextureFormat::Depth32FloatStencil8,
    // The compressed formats, which the device validates against its features.
    "bc1-rgba-unorm" => wgpu::TextureFormat::Bc1RgbaUnorm,
    "bc1-rgba-unorm-srgb" => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
    "bc2-rgba-unorm" => wgpu::TextureFormat::Bc2RgbaUnorm,
    "bc2-rgba-unorm-srgb" => wgpu::TextureFormat::Bc2RgbaUnormSrgb,
    "bc3-rgba-unorm" => wgpu::TextureFormat::Bc3RgbaUnorm,
    "bc3-rgba-unorm-srgb" => wgpu::TextureFormat::Bc3RgbaUnormSrgb,
    "bc4-r-unorm" => wgpu::TextureFormat::Bc4RUnorm,
    "bc4-r-snorm" => wgpu::TextureFormat::Bc4RSnorm,
    "bc5-rg-unorm" => wgpu::TextureFormat::Bc5RgUnorm,
    "bc5-rg-snorm" => wgpu::TextureFormat::Bc5RgSnorm,
    "bc6h-rgb-ufloat" => wgpu::TextureFormat::Bc6hRgbUfloat,
    "bc6h-rgb-float" => wgpu::TextureFormat::Bc6hRgbFloat,
    "bc7-rgba-unorm" => wgpu::TextureFormat::Bc7RgbaUnorm,
    "bc7-rgba-unorm-srgb" => wgpu::TextureFormat::Bc7RgbaUnormSrgb,
    "etc2-rgb8unorm" => wgpu::TextureFormat::Etc2Rgb8Unorm,
    "etc2-rgb8unorm-srgb" => wgpu::TextureFormat::Etc2Rgb8UnormSrgb,
    "etc2-rgb8a1unorm" => wgpu::TextureFormat::Etc2Rgb8A1Unorm,
    "etc2-rgb8a1unorm-srgb" => wgpu::TextureFormat::Etc2Rgb8A1UnormSrgb,
    "etc2-rgba8unorm" => wgpu::TextureFormat::Etc2Rgba8Unorm,
    "etc2-rgba8unorm-srgb" => wgpu::TextureFormat::Etc2Rgba8UnormSrgb,
    "eac-r11unorm" => wgpu::TextureFormat::EacR11Unorm,
    "eac-r11snorm" => wgpu::TextureFormat::EacR11Snorm,
    "eac-rg11unorm" => wgpu::TextureFormat::EacRg11Unorm,
    "eac-rg11snorm" => wgpu::TextureFormat::EacRg11Snorm,
    "astc-4x4-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::Unorm },
    "astc-4x4-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B4x4, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-5x4-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::Unorm },
    "astc-5x4-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x4, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-5x5-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::Unorm },
    "astc-5x5-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B5x5, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-6x5-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::Unorm },
    "astc-6x5-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x5, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-6x6-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::Unorm },
    "astc-6x6-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B6x6, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-8x5-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::Unorm },
    "astc-8x5-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x5, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-8x6-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::Unorm },
    "astc-8x6-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x6, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-8x8-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::Unorm },
    "astc-8x8-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B8x8, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-10x5-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::Unorm },
    "astc-10x5-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x5, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-10x6-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::Unorm },
    "astc-10x6-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x6, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-10x8-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::Unorm },
    "astc-10x8-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x8, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-10x10-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::Unorm },
    "astc-10x10-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B10x10, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-12x10-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::Unorm },
    "astc-12x10-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x10, channel: wgpu::AstcChannel::UnormSrgb },
    "astc-12x12-unorm" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::Unorm },
    "astc-12x12-unorm-srgb" => wgpu::TextureFormat::Astc { block: wgpu::AstcBlock::B12x12, channel: wgpu::AstcChannel::UnormSrgb },
});

names!(vertex_format -> wgpu::VertexFormat, "vertex format" {
    "uint8" => wgpu::VertexFormat::Uint8,
    "uint8x2" => wgpu::VertexFormat::Uint8x2,
    "uint8x4" => wgpu::VertexFormat::Uint8x4,
    "sint8" => wgpu::VertexFormat::Sint8,
    "sint8x2" => wgpu::VertexFormat::Sint8x2,
    "sint8x4" => wgpu::VertexFormat::Sint8x4,
    "unorm8" => wgpu::VertexFormat::Unorm8,
    "unorm8x2" => wgpu::VertexFormat::Unorm8x2,
    "unorm8x4" => wgpu::VertexFormat::Unorm8x4,
    "snorm8" => wgpu::VertexFormat::Snorm8,
    "snorm8x2" => wgpu::VertexFormat::Snorm8x2,
    "snorm8x4" => wgpu::VertexFormat::Snorm8x4,
    "uint16" => wgpu::VertexFormat::Uint16,
    "uint16x2" => wgpu::VertexFormat::Uint16x2,
    "uint16x4" => wgpu::VertexFormat::Uint16x4,
    "sint16" => wgpu::VertexFormat::Sint16,
    "sint16x2" => wgpu::VertexFormat::Sint16x2,
    "sint16x4" => wgpu::VertexFormat::Sint16x4,
    "unorm16" => wgpu::VertexFormat::Unorm16,
    "unorm16x2" => wgpu::VertexFormat::Unorm16x2,
    "unorm16x4" => wgpu::VertexFormat::Unorm16x4,
    "snorm16" => wgpu::VertexFormat::Snorm16,
    "snorm16x2" => wgpu::VertexFormat::Snorm16x2,
    "snorm16x4" => wgpu::VertexFormat::Snorm16x4,
    "float16" => wgpu::VertexFormat::Float16,
    "float16x2" => wgpu::VertexFormat::Float16x2,
    "float16x4" => wgpu::VertexFormat::Float16x4,
    "float32" => wgpu::VertexFormat::Float32,
    "float32x2" => wgpu::VertexFormat::Float32x2,
    "float32x3" => wgpu::VertexFormat::Float32x3,
    "float32x4" => wgpu::VertexFormat::Float32x4,
    "uint32" => wgpu::VertexFormat::Uint32,
    "uint32x2" => wgpu::VertexFormat::Uint32x2,
    "uint32x3" => wgpu::VertexFormat::Uint32x3,
    "uint32x4" => wgpu::VertexFormat::Uint32x4,
    "sint32" => wgpu::VertexFormat::Sint32,
    "sint32x2" => wgpu::VertexFormat::Sint32x2,
    "sint32x3" => wgpu::VertexFormat::Sint32x3,
    "sint32x4" => wgpu::VertexFormat::Sint32x4,
    "unorm10-10-10-2" => wgpu::VertexFormat::Unorm10_10_10_2,
    "unorm8x4-bgra" => wgpu::VertexFormat::Unorm8x4Bgra,
});

names!(vertex_step_mode -> wgpu::VertexStepMode, "vertex step mode" {
    "vertex" => wgpu::VertexStepMode::Vertex,
    "instance" => wgpu::VertexStepMode::Instance,
});

names!(index_format -> wgpu::IndexFormat, "index format" {
    "uint16" => wgpu::IndexFormat::Uint16,
    "uint32" => wgpu::IndexFormat::Uint32,
});

names!(compare_function -> wgpu::CompareFunction, "compare function" {
    "never" => wgpu::CompareFunction::Never,
    "less" => wgpu::CompareFunction::Less,
    "equal" => wgpu::CompareFunction::Equal,
    "less-equal" => wgpu::CompareFunction::LessEqual,
    "greater" => wgpu::CompareFunction::Greater,
    "not-equal" => wgpu::CompareFunction::NotEqual,
    "greater-equal" => wgpu::CompareFunction::GreaterEqual,
    "always" => wgpu::CompareFunction::Always,
});

names!(front_face -> wgpu::FrontFace, "front face" {
    "ccw" => wgpu::FrontFace::Ccw,
    "cw" => wgpu::FrontFace::Cw,
});

names!(cull_mode -> Option<wgpu::Face>, "cull mode" {
    "none" => None,
    "front" => Some(wgpu::Face::Front),
    "back" => Some(wgpu::Face::Back),
});

names!(blend_factor -> wgpu::BlendFactor, "blend factor" {
    "zero" => wgpu::BlendFactor::Zero,
    "one" => wgpu::BlendFactor::One,
    "src" => wgpu::BlendFactor::Src,
    "one-minus-src" => wgpu::BlendFactor::OneMinusSrc,
    "src-alpha" => wgpu::BlendFactor::SrcAlpha,
    "one-minus-src-alpha" => wgpu::BlendFactor::OneMinusSrcAlpha,
    "dst" => wgpu::BlendFactor::Dst,
    "one-minus-dst" => wgpu::BlendFactor::OneMinusDst,
    "dst-alpha" => wgpu::BlendFactor::DstAlpha,
    "one-minus-dst-alpha" => wgpu::BlendFactor::OneMinusDstAlpha,
    "src-alpha-saturated" => wgpu::BlendFactor::SrcAlphaSaturated,
    "constant" => wgpu::BlendFactor::Constant,
    "one-minus-constant" => wgpu::BlendFactor::OneMinusConstant,
    "src1" => wgpu::BlendFactor::Src1,
    "one-minus-src1" => wgpu::BlendFactor::OneMinusSrc1,
    "src1-alpha" => wgpu::BlendFactor::Src1Alpha,
    "one-minus-src1-alpha" => wgpu::BlendFactor::OneMinusSrc1Alpha,
});

names!(blend_operation -> wgpu::BlendOperation, "blend operation" {
    "add" => wgpu::BlendOperation::Add,
    "subtract" => wgpu::BlendOperation::Subtract,
    "reverse-subtract" => wgpu::BlendOperation::ReverseSubtract,
    "min" => wgpu::BlendOperation::Min,
    "max" => wgpu::BlendOperation::Max,
});

names!(stencil_operation -> wgpu::StencilOperation, "stencil operation" {
    "keep" => wgpu::StencilOperation::Keep,
    "zero" => wgpu::StencilOperation::Zero,
    "replace" => wgpu::StencilOperation::Replace,
    "invert" => wgpu::StencilOperation::Invert,
    "increment-clamp" => wgpu::StencilOperation::IncrementClamp,
    "decrement-clamp" => wgpu::StencilOperation::DecrementClamp,
    "increment-wrap" => wgpu::StencilOperation::IncrementWrap,
    "decrement-wrap" => wgpu::StencilOperation::DecrementWrap,
});

names!(texture_dimension -> wgpu::TextureDimension, "texture dimension" {
    "1d" => wgpu::TextureDimension::D1,
    "2d" => wgpu::TextureDimension::D2,
    "3d" => wgpu::TextureDimension::D3,
});

names!(texture_view_dimension -> wgpu::TextureViewDimension, "texture view dimension" {
    "1d" => wgpu::TextureViewDimension::D1,
    "2d" => wgpu::TextureViewDimension::D2,
    "2d-array" => wgpu::TextureViewDimension::D2Array,
    "cube" => wgpu::TextureViewDimension::Cube,
    "cube-array" => wgpu::TextureViewDimension::CubeArray,
    "3d" => wgpu::TextureViewDimension::D3,
});

names!(texture_aspect -> wgpu::TextureAspect, "texture aspect" {
    "all" => wgpu::TextureAspect::All,
    "stencil-only" => wgpu::TextureAspect::StencilOnly,
    "depth-only" => wgpu::TextureAspect::DepthOnly,
});

names!(buffer_binding_type -> wgpu::BufferBindingType, "buffer binding type" {
    "uniform" => wgpu::BufferBindingType::Uniform,
    "storage" => wgpu::BufferBindingType::Storage { read_only: false },
    "read-only-storage" => wgpu::BufferBindingType::Storage { read_only: true },
});

names!(sampler_binding_type -> wgpu::SamplerBindingType, "sampler binding type" {
    "filtering" => wgpu::SamplerBindingType::Filtering,
    "non-filtering" => wgpu::SamplerBindingType::NonFiltering,
    "comparison" => wgpu::SamplerBindingType::Comparison,
});

names!(texture_sample_type -> wgpu::TextureSampleType, "texture sample type" {
    "float" => wgpu::TextureSampleType::Float { filterable: true },
    "unfilterable-float" => wgpu::TextureSampleType::Float { filterable: false },
    "depth" => wgpu::TextureSampleType::Depth,
    "sint" => wgpu::TextureSampleType::Sint,
    "uint" => wgpu::TextureSampleType::Uint,
});

names!(storage_texture_access -> wgpu::StorageTextureAccess, "storage texture access" {
    "write-only" => wgpu::StorageTextureAccess::WriteOnly,
    "read-only" => wgpu::StorageTextureAccess::ReadOnly,
    "read-write" => wgpu::StorageTextureAccess::ReadWrite,
});

names!(load_op -> bool, "load operation" {
    "load" => false,
    "clear" => true,
});

names!(store_op -> wgpu::StoreOp, "store operation" {
    "store" => wgpu::StoreOp::Store,
    "discard" => wgpu::StoreOp::Discard,
});

names!(address_mode -> wgpu::AddressMode, "address mode" {
    "clamp-to-edge" => wgpu::AddressMode::ClampToEdge,
    "repeat" => wgpu::AddressMode::Repeat,
    "mirror-repeat" => wgpu::AddressMode::MirrorRepeat,
});

names!(filter_mode -> wgpu::FilterMode, "filter mode" {
    "nearest" => wgpu::FilterMode::Nearest,
    "linear" => wgpu::FilterMode::Linear,
});

names!(mipmap_filter_mode -> wgpu::MipmapFilterMode, "mipmap filter mode" {
    "nearest" => wgpu::MipmapFilterMode::Nearest,
    "linear" => wgpu::MipmapFilterMode::Linear,
});
