#![allow(
    clippy::needless_range_loop,
    reason = "keep the scalar code comparable to the vectorized code"
)]
/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Saturday, March 22, 2025
 * Author: Yumechi <yume@yumechi.jp>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use core::{
    fmt::{Debug, Display},
    ops::Add,
};

use generic_array::typenum::U3;
#[allow(unused_imports)]
use generic_array::{
    ArrayLength, GenericArray,
    sequence::Flatten,
    typenum::{B1, U0, U1, U2, U4, U16, U127, U128, U256, U512, Unsigned},
};

#[cfg_attr(not(feature = "reference-rug"), allow(unused_imports))]
use num_traits::{FromPrimitive, NumCast, Signed, ToPrimitive, float::FloatCore};

pub use generic_array;
use sealing::Sealed;
use type_traits::{DivisibleBy8, EvaluateHardwareFeature, SquareOf, Term};

use crate::alignment::DefaultPaddedArray;

/// Constants for conversions.
pub mod constants;

/// Kernels based on x86-64 intrinsics.
#[cfg(target_arch = "x86_64")]
pub mod x86;

/// A fallback router for kernels.
pub mod router;

#[cfg(feature = "reference-rug")]
/// 128-bit floating point type.
pub mod float128;

#[cfg(feature = "portable-simd")]
/// Portable SIMD implementation.
pub mod portable_simd;

// potentially incorrect, don't use it for now
// pub mod dihedral;

/// Threshold/quantization methods.
pub mod threshold;

/// Type traits.
pub mod type_traits;

include!(concat!(env!("OUT_DIR"), "/dct_matrix.rs"));
include!(concat!(env!("OUT_DIR"), "/tent_filter_weights.rs"));
include!(concat!(env!("OUT_DIR"), "/convolution_offset.rs"));

#[cfg(feature = "portable-simd")]
/// The worst case fallback kernel.
pub type FallbackKernel = portable_simd::PortableSimdF32Kernel<8>;

#[cfg(all(not(feature = "portable-simd"), not(target_arch = "x86_64")))]
/// The worst case fallback kernel.
pub type FallbackKernel = DefaultKernelNoPadding;

#[cfg(all(not(feature = "portable-simd"), target_arch = "x86_64"))]
/// The worst case fallback kernel.
pub type FallbackKernel = DefaultKernelPadXYTo128;

/// Return an opaque kernel object that is likely what you want. (based on your feature flags)
///
/// Generally we will make every effort to make new kernels available through this function so you don't need to care about the underlying implementation.
#[inline(always)]
#[must_use]
pub fn smart_kernel() -> impl Kernel<
    RequiredHardwareFeature = impl EvaluateHardwareFeature<EnabledStatic = B1>,
    InputDimension = U512,
    OutputDimension = U16,
    InternalFloat = f32,
> + Clone {
    smart_kernel_impl()
}

#[cfg(target_arch = "x86_64")]
#[cfg(feature = "avx512")]
#[cfg(any(feature = "prefer-x86-intrinsics", not(feature = "portable-simd")))]
pub(crate) type SmartKernelConcreteType = router::KernelRouter<
    x86::Avx512F32Kernel,
    router::KernelRouter<x86::Avx2F32Kernel, FallbackKernel>,
>;

#[cfg(target_arch = "x86_64")]
#[cfg(feature = "avx512")]
#[cfg(all(not(feature = "prefer-x86-intrinsics"), feature = "portable-simd"))]
pub(crate) type SmartKernelConcreteType =
    router::KernelRouter<x86::Avx512F32Kernel, portable_simd::PortableSimdF32Kernel<8>>;

#[cfg(target_arch = "x86_64")]
#[cfg(feature = "avx512")]
#[inline(always)]
pub(crate) fn smart_kernel_impl() -> SmartKernelConcreteType {
    #[allow(unused_imports)]
    use router::KernelRouter;

    #[cfg(any(feature = "prefer-x86-intrinsics", not(feature = "portable-simd")))]
    {
        KernelRouter::new(x86::Avx2F32Kernel, FallbackKernel::default())
            .layer_on_top(x86::Avx512F32Kernel)
    }

    #[cfg(all(not(feature = "prefer-x86-intrinsics"), feature = "portable-simd"))]
    {
        KernelRouter::new(
            x86::Avx512F32Kernel,
            portable_simd::PortableSimdF32Kernel::<8>,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[cfg(not(feature = "avx512"))]
#[cfg(any(feature = "prefer-x86-intrinsics", not(feature = "portable-simd")))]
pub(crate) type SmartKernelConcreteType = router::KernelRouter<x86::Avx2F32Kernel, FallbackKernel>;

#[cfg(target_arch = "x86_64")]
#[cfg(not(feature = "avx512"))]
#[cfg(all(not(feature = "prefer-x86-intrinsics"), feature = "portable-simd"))]
pub(crate) type SmartKernelConcreteType = portable_simd::PortableSimdF32Kernel<8>;

#[cfg(target_arch = "x86_64")]
#[cfg(not(feature = "avx512"))]
#[inline(always)]
pub(crate) fn smart_kernel_impl() -> SmartKernelConcreteType {
    #[allow(unused_imports)]
    use router::KernelRouter;

    #[cfg(any(feature = "prefer-x86-intrinsics", not(feature = "portable-simd")))]
    {
        KernelRouter::new(x86::Avx2F32Kernel, FallbackKernel::default())
    }

    #[cfg(all(not(feature = "prefer-x86-intrinsics"), feature = "portable-simd"))]
    {
        portable_simd::PortableSimdF32Kernel::<8>::default()
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) type SmartKernelConcreteType = FallbackKernel;

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn smart_kernel_impl() -> SmartKernelConcreteType {
    FallbackKernel::default()
}

mod sealing {
    /// Private sealing trait
    pub trait Sealed {}
}

/// Extension trait for square `GenericArray`'s.
///
/// This trait is defined for all non-empty square `GenericArray`'s up to 1024x1024.
pub trait SquareGenericArrayExt<I, L: SquareOf>: Sized + Sealed {
    /// Unflatten a [`GenericArray`] into a square matrix.
    fn unflatten_square(self) -> GenericArray<GenericArray<I, L>, L> {
        unsafe { generic_array::const_transmute(self) }
    }

    /// Unflatten a [`GenericArray`] into a square matrix by reference.
    fn unflatten_square_ref(&self) -> &GenericArray<GenericArray<I, L>, L> {
        unsafe { generic_array::const_transmute(self) }
    }

    /// Unflatten a [`GenericArray`] into a square matrix by mutable reference.
    fn unflatten_square_mut(&mut self) -> &mut GenericArray<GenericArray<I, L>, L> {
        unsafe { generic_array::const_transmute(self) }
    }
}

impl<I, L: ArrayLength> Sealed for GenericArray<I, L> {}

impl<I, L: SquareOf> SquareGenericArrayExt<I, L> for GenericArray<I, <L as SquareOf>::Output> {}

/// Marker trait to indicate that this kernel produces precise results (i.e. no lossy LUTs).
pub trait PreciseKernel: Kernel {}

// Copied verbatim from [darwinium-com/pdqhash](https://github.com/darwinium-com/pdqhash/blob/main/src/lib.rs).
#[allow(clippy::all, clippy::pedantic)]
pub(crate) fn torben_median<F: FloatCore>(m: &GenericArray<GenericArray<F, U16>, U16>) -> F {
    let mut min = m.iter().flatten().cloned().reduce(F::min).unwrap();
    let mut max = m.iter().flatten().cloned().reduce(F::max).unwrap();

    let half = (16 * 16 + 1) / 2;
    loop {
        let guess = (min + max) / F::from(2).unwrap();
        let mut less = 0;
        let mut greater = 0;
        let mut equal = 0;
        let mut maxltguess = min;
        let mut mingtguess = max;
        for val in m.into_iter().flatten() {
            if *val < guess {
                less += 1;
                if *val > maxltguess {
                    maxltguess = *val;
                }
            } else if *val > guess {
                greater += 1;
                if *val < mingtguess {
                    mingtguess = *val;
                }
            } else {
                equal += 1;
            }
        }
        if less <= half && greater <= half {
            return if less >= half {
                maxltguess
            } else if less + equal >= half {
                guess
            } else {
                mingtguess
            };
        } else if less > greater {
            max = maxltguess;
        } else {
            min = mingtguess;
        }
    }
}

/// The divisor for the quality adjustment offset.
/// cbindgen:ignore
pub const QUALITY_ADJUST_DIVISOR: usize = 180;

// reference: https://raw.githubusercontent.com/facebook/ThreatExchange/main/hashing/hashing.pdf

/// Compute kernel for doing heavy-duty transformations.
///
/// Kernels not marked with [`PreciseKernel`] uses LUTs extensively and are thus less accurate, albeit still well within the official docs' tolerance of "correct" implementation (10 bits different).
///
/// A typical matching threshold is distance <=31 bits out of 256, well within the error margin.
///
/// A scalar (auto-vectorized) implementation is provided in [`DefaultKernel`].
pub trait Kernel {
    /// The width of the first stage (compression) buffer.
    type Buffer1WidthX: ArrayLength;
    /// The length of the first stage (compression) buffer.
    type Buffer1LengthY: ArrayLength;
    /// The width and height of the input image.
    type InputDimension: ArrayLength + SquareOf;
    /// The width and height of the output hash.
    type OutputDimension: ArrayLength + SquareOf;
    /// The hardware features required to run this kernel
    type RequiredHardwareFeature: EvaluateHardwareFeature;

    /// Identification token.
    type Ident: Debug + Display + Clone + Copy + 'static + PartialEq;

    /// The internal floating point type used for intermediate calculations.
    type InternalFloat: num_traits::float::TotalOrder
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + num_traits::bounds::Bounded
        + num_traits::NumCast
        + num_traits::identities::Zero
        + num_traits::identities::One
        + num_traits::Signed
        + PartialOrd
        + Clone
        + Display
        + Debug
        + Default
        + Send
        + Sync;

    /// Convert one row of input from RGB8 to LUMA8 floating point.
    ///
    /// A scalar implementation is provided by default.
    fn cvt_rgb8_to_luma8f<const R_COEFF: u32, const G_COEFF: u32, const B_COEFF: u32>(
        &mut self,
        input: &GenericArray<GenericArray<u8, U3>, Self::InputDimension>,
        output: &mut GenericArray<f32, Self::InputDimension>,
    ) where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
    {
        for (input_pixel, output_pixel) in input.iter().zip(output.iter_mut()) {
            let r = input_pixel[0] as f32;
            let g = input_pixel[1] as f32;
            let b = input_pixel[2] as f32;
            let luma = f32::from_ne_bytes(R_COEFF.to_ne_bytes()) * r
                + f32::from_ne_bytes(G_COEFF.to_ne_bytes()) * g
                + f32::from_ne_bytes(B_COEFF.to_ne_bytes()) * b;
            *output_pixel = luma;
        }
    }

    /// Convert RGBA8 to LUMA8.
    ///
    /// A scalar implementation is provided by default.
    fn cvt_rgba8_to_luma8f<const R_COEFF: u32, const G_COEFF: u32, const B_COEFF: u32>(
        &mut self,
        input: &GenericArray<GenericArray<u8, U4>, Self::InputDimension>,
        output: &mut GenericArray<f32, Self::InputDimension>,
    ) where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
    {
        for (input_pixel, output_pixel) in input.iter().zip(output.iter_mut()) {
            let r = input_pixel[0] as f32;
            let g = input_pixel[1] as f32;
            let b = input_pixel[2] as f32;
            let luma = f32::from_ne_bytes(R_COEFF.to_ne_bytes()) * r
                + f32::from_ne_bytes(G_COEFF.to_ne_bytes()) * g
                + f32::from_ne_bytes(B_COEFF.to_ne_bytes()) * b;
            *output_pixel = luma;
        }
    }

    /// Transpose a PDQF matrix in place. Output is equivalent to PDQF(t(image)).
    ///
    /// A scalar implementation is provided by default.
    fn pdqf_t(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
    {
        for i in 0..Self::OutputDimension::USIZE {
            for j in 0..Self::OutputDimension::USIZE {
                if i < j {
                    (input[j][i], input[i][j]) = (input[i][j].clone(), input[j][i].clone());
                }
            }
        }
    }

    /// Negate alternative columns of PDQF matrix in place. Equivalent to PDQF(flop(image)).
    ///
    /// NEGATE means start from odd-indices (even columns) instead, useful for continuous flipping.
    ///
    /// A scalar implementation is provided by default.
    fn pdqf_negate_alt_cols<const NEGATE: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
    {
        for i in 0..Self::OutputDimension::USIZE {
            for j in ((if NEGATE { 1 } else { 0 })..Self::OutputDimension::USIZE).step_by(2) {
                input[i][j] = -input[i][j].clone();
            }
        }
    }

    /// Negate alternative rows of PDQF matrix in place. Equivalent to PDQF(flip(image)).
    ///
    /// NEGATE means start from odd-indices (even rows) instead, useful for continuous flipping.
    ///
    /// A scalar implementation is provided by default.
    ///
    /// You usually should not have to override this as it is trivially vectorizable.
    fn pdqf_negate_alt_rows<const NEGATE: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
    {
        for i in ((if NEGATE { 1 } else { 0 })..Self::OutputDimension::USIZE).step_by(2) {
            for j in 0..Self::OutputDimension::USIZE {
                input[i][j] = -input[i][j].clone();
            }
        }
    }

    /// Negate off-diagonals of PDQF matrix in place. Equivalent to PDQF(rotate180(image)).
    ///
    /// If you need the intermediate result,
    /// it is usually less efficient than just doing [`Self::pdqf_negate_alt_cols`] and [`Self::pdqf_negate_alt_rows`] inverted in sequence.
    ///
    /// A scalar implementation is provided by default.
    fn pdqf_negate_off_diagonals(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        for i in 0..Self::OutputDimension::USIZE {
            for j in 0..Self::OutputDimension::USIZE {
                if j.wrapping_sub(i) % 2 == 1 {
                    input[i][j] = -input[i][j].clone();
                }
            }
        }
    }

    /// Return an identification of the kernel. For composite kernels that route to multiple kernels, report the kernel that would be executed.
    fn ident(&self) -> Self::Ident;

    /// Shorthand to `<<K as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::met_runtime()`
    ///
    /// Usually downstream applications should be generic over this trait and finally a runtime check is done to dispatch application logic with the suitable kernel.
    #[must_use]
    fn required_hardware_features_met() -> bool {
        Self::RequiredHardwareFeature::met_runtime()
    }

    /// Apply a tent-filter average to every 8x8 sub-block of the input buffer and write the result of each sub-block to the output buffer.
    fn jarosz_compress(
        &mut self,
        _buffer: &GenericArray<GenericArray<f32, Self::InputDimension>, Self::InputDimension>,
        _output: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
    ) where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>;

    /// Convert input to binary by thresholding at median
    fn quantize(
        &mut self,
        _input: &GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
        _threshold: &mut Self::InternalFloat,
        _output: &mut GenericArray<
            GenericArray<u8, <Self::OutputDimension as DivisibleBy8>::Output>,
            Self::OutputDimension,
        >,
    ) where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
        <Self as Kernel>::OutputDimension: DivisibleBy8;

    /// Compute the sum of gradients of the input buffer in both horizontal and vertical directions.
    #[must_use]
    fn sum_of_gradients(
        &mut self,
        _input: &GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> Self::InternalFloat
    where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>;

    /// Adjust the quality metric to be between 0 and 1.
    #[must_use]
    fn adjust_quality(_input: Self::InternalFloat) -> f32;

    /// Apply a 2D DCT-II transformation to the input buffer write the result to the output buffer.
    fn dct2d(
        &mut self,
        _buffer: &GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
        _tmp_row_buffer: &mut GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
        _output: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>;
}

/// A pure-Rust implementation of the `Kernel` trait.
#[derive(Clone, Copy)]
pub struct DefaultKernel<
    PadIntermediateX: ArrayLength + Add<U127> = U0,
    PadIntermediateY: ArrayLength + Add<U127> = U0,
> where
    <PadIntermediateX as Add<U127>>::Output: ArrayLength,
    <PadIntermediateY as Add<U127>>::Output: ArrayLength,
{
    _marker: core::marker::PhantomData<(PadIntermediateX, PadIntermediateY)>,
}

/// A default kernel with no padding on the intermediate 127x127 matrix.
pub type DefaultKernelNoPadding = DefaultKernel<U0, U0>;
/// A default kernel with padding 1 row on the intermediate 127x127 matrix on the X axis.
/// This does not make it faster but only useful for sharing buffers with vectorized kernels.
pub type DefaultKernelPadXTo128 = DefaultKernel<U1, U0>;
/// A default kernel with padding 1 column on the intermediate 127x127 matrix on the Y axis.
/// This does not make it faster but only useful for sharing buffers with vectorized kernels.
pub type DefaultKernelPadYTo128 = DefaultKernel<U0, U1>;
/// A default kernel with padding 1 row and 1 column on the intermediate 127x127 matrix on both the X and Y axes.
/// This does not make it faster but only useful for sharing buffers with vectorized kernels.
pub type DefaultKernelPadXYTo128 = DefaultKernel<U1, U1>;

impl<PadIntermediateX: ArrayLength + Add<U127>, PadIntermediateY: ArrayLength + Add<U127>> Default
    for DefaultKernel<PadIntermediateX, PadIntermediateY>
where
    <PadIntermediateX as Add<U127>>::Output: ArrayLength,
    <PadIntermediateY as Add<U127>>::Output: ArrayLength,
{
    fn default() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<PadIntermediateX: ArrayLength + Add<U127>, PadIntermediateY: ArrayLength + Add<U127>>
    DefaultKernel<PadIntermediateX, PadIntermediateY>
where
    <PadIntermediateX as Add<U127>>::Output: ArrayLength,
    <PadIntermediateY as Add<U127>>::Output: ArrayLength,
{
    #[inline(always)]
    pub(crate) fn dct2d_impl<P: ArrayLength>(
        dct_matrix_rmajor: &DefaultPaddedArray<f32, DctMatrixNumElements, P>,
        buffer: &GenericArray<
            GenericArray<f32, <PadIntermediateX as Add<U127>>::Output>,
            <PadIntermediateY as Add<U127>>::Output,
        >,
        tmp_row_buffer: &mut GenericArray<f32, <PadIntermediateX as Add<U127>>::Output>,
        output: &mut GenericArray<GenericArray<f32, U16>, U16>,
    ) {
        // crate::testing::dump_image("dct2d_input_scalar.ppm", buffer);
        for k in 0..16 {
            for j in 0..127 {
                let mut sumks = [0.0; 4];
                for (k2, sumk) in (0..127).zip((0..4).cycle()) {
                    sumks[sumk] +=
                        dct_matrix_rmajor[k * DctMatrixNumCols::USIZE + k2] * buffer[k2][j];
                }

                tmp_row_buffer[j] = sumks[0] + sumks[1] + sumks[2] + sumks[3];
            }

            for j in 0..(DctMatrixNumRows::USIZE) {
                let mut sumks = [0.0; 4];
                for (m, sumk) in (0..DctMatrixNumCols::USIZE).zip((0..4).cycle()) {
                    sumks[sumk] +=
                        tmp_row_buffer[m] * dct_matrix_rmajor[j * DctMatrixNumCols::USIZE + m];
                }
                output[k][j] = sumks[0] + sumks[1] + sumks[2] + sumks[3];
            }
        }
        // crate::testing::dump_image("dct2d_output_scalar.ppm", output);
    }
}

impl<PadIntermediateX: ArrayLength + Add<U127>, PadIntermediateY: ArrayLength + Add<U127>> Kernel
    for DefaultKernel<PadIntermediateX, PadIntermediateY>
where
    <PadIntermediateX as Add<U127>>::Output: ArrayLength,
    <PadIntermediateY as Add<U127>>::Output: ArrayLength,
{
    type Buffer1WidthX = <PadIntermediateX as Add<U127>>::Output;
    type Buffer1LengthY = <PadIntermediateY as Add<U127>>::Output;
    type InternalFloat = f32;
    type InputDimension = U512;
    type OutputDimension = U16;
    type RequiredHardwareFeature = Term;
    type Ident = &'static str;

    fn ident(&self) -> &'static str {
        "default_scalar_autovectorized_f32"
    }

    fn quantize(
        &mut self,
        input: &GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
        threshold: &mut Self::InternalFloat,
        output: &mut GenericArray<GenericArray<u8, U2>, U16>,
    ) {
        let median = torben_median(input);
        *threshold = median;
        let output = output.flatten();
        output.fill(0);
        for (i, j) in input.iter().flatten().enumerate() {
            output[32 - 1 - i / 8] += if *j > median { 1 << (i % 8) } else { 0 };
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input.to_f32().unwrap() / (QUALITY_ADJUST_DIVISOR as f32);

        scaled.min(1.0)
    }

    fn sum_of_gradients(
        &mut self,
        input: &GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
    ) -> Self::InternalFloat {
        let mut gradient_sum = Default::default();

        for i in 0..(16 - 1) {
            for j in 0..16 {
                let u = input[i][j];
                let v = input[i + 1][j];
                let d = u - v;
                gradient_sum += d.abs();
            }
        }

        for i in 0..16 {
            for j in 0..(16 - 1) {
                let u = input[i][j];
                let v = input[i][j + 1];
                let d = u - v;
                gradient_sum += d.abs();
            }
        }

        gradient_sum
    }

    fn jarosz_compress(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, U512>, U512>,
        output: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
    ) {
        for outi in 0..127 {
            let in_i = CONVOLUTION_OFFSET_512_TO_127[outi] - TENT_FILTER_COLUMN_OFFSET;
            for outj in 0..127 {
                let in_j = CONVOLUTION_OFFSET_512_TO_127[outj] - TENT_FILTER_COLUMN_OFFSET;
                let mut sum = 0.0;
                for di in 0..TENT_FILTER_EFFECTIVE_ROWS {
                    for dj in 0..TENT_FILTER_EFFECTIVE_COLS {
                        sum += TENT_FILTER_WEIGHTS[di * TENT_FILTER_EFFECTIVE_COLS + dj]
                            * buffer[in_i + di][in_j + dj];
                    }
                }
                output[outi][outj] = sum;
            }
        }
    }

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        tmp_row_buffer: &mut GenericArray<f32, Self::Buffer1WidthX>,
        output: &mut GenericArray<GenericArray<f32, U16>, U16>,
    ) {
        Self::dct2d_impl(&DCT_MATRIX_RMAJOR, buffer, tmp_row_buffer, output);
    }
}

/// A reference implementation of the `Kernel` trait, copied verbatim from the officially-endorsed implementation by [darwinium-com](https://github.com/darwinium-com/pdqhash).
#[cfg(any(feature = "std", test))]
#[derive(Default)]
pub struct ReferenceKernel<N: num_traits::FromPrimitive = f32> {
    _marker: core::marker::PhantomData<N>,
}

#[cfg(any(feature = "std", test))]
impl PreciseKernel for ReferenceKernel<f32> {}

#[cfg(any(feature = "std", test))]
impl PreciseKernel for ReferenceKernel<f64> {}
#[cfg(any(feature = "std", test))]
impl Kernel for ReferenceKernel<f32> {
    type Buffer1WidthX = U127;
    type Buffer1LengthY = U127;
    type InternalFloat = f32;
    type InputDimension = U512;
    type OutputDimension = U16;
    type RequiredHardwareFeature = Term;
    type Ident = &'static str;

    fn ident(&self) -> Self::Ident {
        "reference_scalar_autovectorized"
    }

    #[allow(clippy::cast_precision_loss)]
    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input / (QUALITY_ADJUST_DIVISOR as f32);

        scaled.min(1.0)
    }

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        tmp_row_buffer: &mut GenericArray<f32, Self::Buffer1WidthX>,
        output: &mut GenericArray<GenericArray<f32, U16>, U16>,
    ) {
        for i in 0..16 {
            for j in 0..127 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk += DCT_MATRIX_RMAJOR[i * DctMatrixNumCols::USIZE + k] * buffer[k][j];
                }

                tmp_row_buffer[j] = sumk;
            }

            for j in 0..16 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk += tmp_row_buffer[k] * DCT_MATRIX_RMAJOR[j * DctMatrixNumCols::USIZE + k];
                }
                output[i][j] = sumk;
            }
        }
    }

    fn sum_of_gradients(
        &mut self,
        input: &GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
    ) -> Self::InternalFloat {
        let mut gradient_sum = Default::default();

        for i in 0..(16 - 1) {
            for j in 0..16 {
                let u = input[i][j];
                let v = input[i + 1][j];
                let d = u - v;
                gradient_sum += d.abs();
            }
        }

        for i in 0..16 {
            for j in 0..(16 - 1) {
                let u = input[i][j];
                let v = input[i][j + 1];
                let d = u - v;
                gradient_sum += d.abs();
            }
        }

        gradient_sum
    }

    fn quantize(
        &mut self,
        input: &GenericArray<GenericArray<f32, U16>, U16>,
        threshold: &mut Self::InternalFloat,
        output: &mut GenericArray<GenericArray<u8, U2>, U16>,
    ) {
        let median = torben_median(input);
        *threshold = median;

        let output = output.flatten();

        for i in 0..32 {
            let mut byte = 0;
            for j in 0..8 {
                let offset = i * 8 + j;
                let offset_i = offset / 16;
                let offset_j = offset % 16;
                let val = input[offset_i][offset_j];
                if val > median {
                    byte |= 1 << j;
                }
            }
            output[32 - i - 1] = byte;
        }
    }

    fn jarosz_compress(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, U512>, U512>,
        output: &mut GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
    ) {
        #[allow(missing_docs, dead_code)]
        mod reference {
            include!("ref.rs");
        }

        let window_size = reference::compute_jarosz_filter_window_size(512, 127);
        let mut buffer = buffer.flatten().to_vec();
        reference::jarosz_filter_float(
            buffer.as_mut_slice().try_into().unwrap(),
            512,
            512,
            window_size,
            window_size,
            2,
        );

        for outi in 0..127 {
            let ini = ((outi * 2 + 1) * 512) / (127 * 2);

            for outj in 0..127 {
                let inj = ((outj * 2 + 1) * 512) / (127 * 2);

                output[outi][outj] = buffer[ini * 512 + inj];
            }
        }
    }
}

#[cfg(any(feature = "std", test))]
impl Kernel for ReferenceKernel<f64> {
    type Buffer1WidthX = U127;
    type Buffer1LengthY = U127;
    type InternalFloat = f64;
    type InputDimension = U512;
    type OutputDimension = U16;
    type RequiredHardwareFeature = Term;
    type Ident = &'static str;

    fn ident(&self) -> Self::Ident {
        "reference_scalar_autovectorized_f64"
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input / (QUALITY_ADJUST_DIVISOR as f64);

        scaled.min(1.0) as f32
    }

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f64, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        tmp_row_buffer: &mut GenericArray<f64, Self::Buffer1WidthX>,
        output: &mut GenericArray<GenericArray<f64, U16>, U16>,
    ) {
        for i in 0..16 {
            for j in 0..127 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk += DCT_MATRIX_RMAJOR_64[i * DctMatrixNumCols::USIZE + k] * buffer[k][j];
                }

                tmp_row_buffer[j] = sumk;
            }

            for j in 0..16 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk +=
                        tmp_row_buffer[k] * DCT_MATRIX_RMAJOR_64[j * DctMatrixNumCols::USIZE + k];
                }
                output[i][j] = sumk;
            }
        }
    }

    fn sum_of_gradients(
        &mut self,
        input: &GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
    ) -> Self::InternalFloat {
        let mut gradient_sum = Default::default();

        for i in 0..(16 - 1) {
            for j in 0..16 {
                let u = input[i][j];
                let v = input[i + 1][j];
                let d = u - v;
                gradient_sum += d.abs();
            }
        }

        for i in 0..16 {
            for j in 0..(16 - 1) {
                let u = input[i][j];
                let v = input[i][j + 1];
                let d = u - v;
                gradient_sum += d.abs();
            }
        }

        gradient_sum
    }

    fn quantize(
        &mut self,
        input: &GenericArray<GenericArray<f64, U16>, U16>,
        threshold: &mut Self::InternalFloat,
        output: &mut GenericArray<GenericArray<u8, U2>, U16>,
    ) {
        let median = torben_median(input);
        *threshold = median;

        let output = output.flatten();

        for i in 0..32 {
            let mut byte = 0;
            for j in 0..8 {
                let offset = i * 8 + j;
                let offset_i = offset / 16;
                let offset_j = offset % 16;
                let val = input[offset_i][offset_j];
                if val > median {
                    byte |= 1 << j;
                }
            }
            output[32 - i - 1] = byte;
        }
    }

    fn jarosz_compress(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, U512>, U512>,
        output: &mut GenericArray<GenericArray<f64, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
    ) {
        #[allow(missing_docs, dead_code)]
        mod reference {
            include!("ref.rs");
        }

        let window_size = reference::compute_jarosz_filter_window_size(512, 127);
        let mut buffer = buffer
            .flatten()
            .into_iter()
            .map(|s| From::from(*s))
            .collect::<Vec<f64>>();
        reference::jarosz_filter_float(
            buffer.as_mut_slice().try_into().unwrap(),
            512,
            512,
            window_size,
            window_size,
            2,
        );

        for outi in 0..127 {
            let ini = ((outi * 2 + 1) * 512) / (127 * 2);

            for outj in 0..127 {
                let inj = ((outj * 2 + 1) * 512) / (127 * 2);

                output[outi][outj] = buffer[ini * 512 + inj];
            }
        }
    }
}

#[cfg(feature = "reference-rug")]
impl<const C: u32> Kernel for ReferenceKernel<float128::ArbFloat<C>> {
    type Buffer1WidthX = U127;
    type Buffer1LengthY = U127;
    type InternalFloat = float128::ArbFloat<C>;
    type InputDimension = U512;
    type OutputDimension = U16;
    type RequiredHardwareFeature = Term;
    type Ident = &'static str;

    fn ident(&self) -> &'static str {
        "reference_scalar_autovectorized_arb_float"
    }

    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input / (float128::ArbFloat::from_usize(QUALITY_ADJUST_DIVISOR).unwrap());

        let one = float128::ArbFloat::from_f32(1.0).unwrap();
        if scaled > one { 1.0 } else { scaled.to_f32() }
    }

    fn sum_of_gradients(
        &mut self,
        input: &GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
    ) -> Self::InternalFloat {
        let mut gradient_sum = float128::ArbFloat::default();

        for i in 0..(16 - 1) {
            for j in 0..16 {
                let u = input[i][j].clone();
                let v = input[i + 1][j].clone();
                let d = u - v;
                gradient_sum += d.abs();
            }
        }

        for i in 0..16 {
            for j in 0..(16 - 1) {
                let u = input[i][j].clone();
                let v = input[i][j + 1].clone();
                let d = u - v;
                gradient_sum += d.abs();
            }
        }

        gradient_sum
    }

    fn dct2d(
        &mut self,
        buffer: &GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
        _tmp_row_buffer: &mut GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
        output: &mut GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
    ) {
        let d_value = |i: i32, j: i32, n: i32| {
            let n1: Self::InternalFloat = NumCast::from(n).unwrap();
            let i1: Self::InternalFloat = NumCast::from(i).unwrap();
            let j1: Self::InternalFloat = NumCast::from(j).unwrap();
            let one: Self::InternalFloat = NumCast::from(1).unwrap();
            let two: Self::InternalFloat = one.clone() + one.clone();
            let pi: Self::InternalFloat = float128::ArbFloat::<C>::pi();

            (two.clone() / n1.clone()).sqrt()
                * (pi / (two.clone() * n1) * i1 * (two * j1 + one)).cos()
        };

        let mut d_table = Vec::with_capacity(127 * 16);
        for i in 1..=16 {
            for j in 0..127 {
                d_table.push(d_value(i, j, 127));
            }
        }

        let mut intermediate_matrix = Vec::<Self::InternalFloat>::with_capacity(127);
        intermediate_matrix.resize(127, Self::InternalFloat::default());

        for i in 0..16 {
            for j in 0..127 {
                let mut sumk = Self::InternalFloat::default();
                for k in 0..127 {
                    let d = d_table[i * 127 + k].clone();
                    sumk += d * buffer[k][j].clone();
                }

                intermediate_matrix[j] = sumk;
            }

            for j in 0..16 {
                let mut sumk = Self::InternalFloat::default();
                for k in 0..127 {
                    let d = d_table[j * 127 + k].clone();
                    sumk += intermediate_matrix[k].clone() * d;
                }
                output[i][j] = sumk;
            }
        }
    }

    fn quantize(
        &mut self,
        input: &GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
        threshold: &mut Self::InternalFloat,
        output: &mut GenericArray<GenericArray<u8, U2>, U16>,
    ) {
        let half = Self::InternalFloat::from_f64(0.5).unwrap();
        let mut flattened = GenericArray::<Self::InternalFloat, U256>::default();
        for i in 0..16 {
            for j in 0..16 {
                flattened[i * 16 + j] = input[i][j].clone();
            }
        }
        flattened.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let median = (flattened[127].clone() + flattened[128].clone()) * half;
        *threshold = median.clone();

        let output = output.flatten();
        for i in 0..32 {
            let mut byte = 0;
            for j in 0..8 {
                let offset = i * 8 + j;
                let offset_i = offset / 16;
                let offset_j = offset % 16;
                let val = input[offset_i][offset_j].clone();
                if val > median {
                    byte |= 1 << j;
                }
            }
            output[32 - i - 1] = byte;
        }
    }

    fn jarosz_compress(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, U512>, U512>,
        output: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
    ) {
        #[allow(missing_docs, dead_code)]
        mod reference {
            include!("ref.rs");
        }

        let window_size = reference::compute_jarosz_filter_window_size(512, 127);
        let mut buffer = buffer
            .flatten()
            .iter()
            .map(|s| float128::ArbFloat::<C>::from_f32(*s).unwrap())
            .collect::<Vec<_>>();
        reference::jarosz_filter_float(
            buffer.as_mut_slice().try_into().unwrap(),
            512,
            512,
            window_size,
            window_size,
            2,
        );

        // target centers not corners:

        for outi in 0..127 {
            let ini = ((outi * 2 + 1) * 512) / (127 * 2);

            for outj in 0..127 {
                let inj = ((outj * 2 + 1) * 512) / (127 * 2);

                output[outi][outj] = buffer[ini * 512 + inj].clone();
            }
        }
    }
}

#[cfg(feature = "reference-rug")]
impl<const C: u32> PreciseKernel for ReferenceKernel<float128::ArbFloat<C>> {}

#[cfg(test)]
mod tests {
    use core::ops::Mul;

    use num_traits::NumCast;
    use rand::Rng;

    use super::*;

    extern crate alloc;

    #[allow(dead_code)]
    fn test_gradient_impl<OD: ArrayLength, K: Kernel<OutputDimension = OD>>(kernel: &mut K)
    where
        ReferenceKernel<<K as Kernel>::InternalFloat>:
            Kernel<InternalFloat = <K as Kernel>::InternalFloat, OutputDimension = OD>,
        OD: Mul<OD>,
        <OD as Mul<OD>>::Output: ArrayLength,
        K::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
        ReferenceKernel<K::InternalFloat>:
            Kernel<InternalFloat = K::InternalFloat, OutputDimension = OD>,
        <ReferenceKernel<K::InternalFloat> as Kernel>::RequiredHardwareFeature:
            EvaluateHardwareFeature<EnabledStatic = B1>,
    {
        let mut rng = rand::rng();
        let mut input: GenericArray<GenericArray<<K as Kernel>::InternalFloat, OD>, OD> =
            GenericArray::default();
        input.iter_mut().for_each(|row| {
            row.iter_mut()
                .for_each(|val| *val = NumCast::from(rng.random_range(0.0..1.0)).unwrap());
        });

        let gradient_ref = ReferenceKernel::<K::InternalFloat>::default().sum_of_gradients(&input);
        let gradient = kernel.sum_of_gradients(&input);

        let diff = (gradient.clone() - NumCast::from(gradient_ref.clone()).unwrap()).abs();
        let diff_ref = diff / NumCast::from(gradient_ref.clone()).unwrap();

        assert!(
            diff_ref < NumCast::from(0.05).unwrap(),
            "gradient: {}, gradient_ref: {}, diff: {} %",
            gradient,
            gradient_ref,
            diff_ref * NumCast::from(100.0).unwrap(),
        );
    }

    fn test_dct64_impl<K: Kernel>(kernel: &mut K, eps: K::InternalFloat)
    where
        <K as Kernel>::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
        ReferenceKernel<K::InternalFloat>: Kernel<InternalFloat = K::InternalFloat>,
        <ReferenceKernel<K::InternalFloat> as Kernel>::RequiredHardwareFeature:
            EvaluateHardwareFeature<EnabledStatic = B1>,
    {
        let mut rng = rand::rng();
        let mut input_ref: GenericArray<GenericArray<<K as Kernel>::InternalFloat, _>, _> =
            GenericArray::default();
        input_ref.iter_mut().enumerate().for_each(|(i, row)| {
            row.iter_mut().enumerate().for_each(|(j, val)| {
                *val = if (i + 2 * j) % 20 < 10 {
                    NumCast::from(rng.random_range(0.0..1.0)).unwrap()
                } else {
                    NumCast::from(rng.random_range(-1.0..0.0)).unwrap()
                };
            });
        });
        let mut input: GenericArray<
            GenericArray<K::InternalFloat, K::Buffer1WidthX>,
            K::Buffer1LengthY,
        > = GenericArray::default();
        for i in 0..127 {
            for j in 0..127 {
                input[i][j] = input_ref[i][j].clone();
            }
        }

        let mut output = GenericArray::default();
        let mut output_ref = GenericArray::default();
        ReferenceKernel::<K::InternalFloat>::default().dct2d(
            &input_ref,
            &mut GenericArray::default(),
            &mut output_ref,
        );

        kernel.dct2d(&input, &mut GenericArray::default(), &mut output);

        output
            .iter()
            .flatten()
            .zip(output_ref.iter().flatten())
            .for_each(|(a, b)| {
                let diff = (a.clone() - NumCast::from(b.clone()).unwrap()).abs();
                assert!(diff <= eps, "difference: {:?} (tolerance: {:?})", diff, eps);
            });
    }

    #[test]
    fn test_dct64_impl_base() {
        let mut kernel = DefaultKernelNoPadding::default();
        test_dct64_impl(&mut kernel, NumCast::from(f32::EPSILON * 32.00).unwrap());
    }

    #[test]
    fn test_dct64_impl_smart_kernel() {
        let mut kernel = smart_kernel();
        test_dct64_impl(&mut kernel, NumCast::from(f32::EPSILON * 32.00).unwrap());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dct64_impl_avx2_equivalence() {
        // do an empirical equivalence check between the default and AVX2 implementations

        use generic_array::typenum::U128;

        let mut dct_matrix =
            DefaultPaddedArray::<_, _, U16>::new(*GenericArray::from_slice(&[1.0; 16 * 127]));

        let mut input_buffer_127: Box<GenericArray<GenericArray<f32, U127>, U127>> = Box::default();
        let mut input_buffer_128: Box<GenericArray<GenericArray<f32, U128>, U128>> = Box::default();

        // part 1: hold the DCT matrix constant as 1 and fill the input buffer sequentially
        input_buffer_127
            .iter_mut()
            .enumerate()
            .for_each(|(i, row)| {
                row.iter_mut().enumerate().for_each(|(j, val)| {
                    input_buffer_128[i][j] = (i * 127 + j) as f32 * 0.00001;
                    *val = (i * 127 + j) as f32 * 0.00001;
                });
            });
        let mut output = GenericArray::default();
        let mut output_ref = GenericArray::default();
        DefaultKernelNoPadding::dct2d_impl(
            &dct_matrix,
            &input_buffer_127,
            &mut GenericArray::default(),
            &mut output_ref,
        );

        #[allow(unused_unsafe)]
        unsafe {
            x86::Avx2F32Kernel::dct2d_impl(
                &dct_matrix,
                &input_buffer_128,
                &mut GenericArray::default(),
                &mut output,
            );
        }

        for i in 0..16 {
            for j in 0..16 {
                let diff = (output_ref[i][j] - output[i][j]).abs();
                let diff_ref = diff / output_ref[i][j];
                // accept up to 0.00004% rounding error (4e-7), almost f32::EPSILON
                assert!(
                    diff_ref < 4e-7,
                    "difference: {:?} (relative: {:?})",
                    diff,
                    diff_ref
                );
            }
        }

        // part 2: hold the input buffer constant and fill the DCT matrix sequentially
        input_buffer_127
            .iter_mut()
            .enumerate()
            .for_each(|(i, row)| {
                row.iter_mut().enumerate().for_each(|(j, val)| {
                    input_buffer_128[i][j] = 0.00001;
                    *val = 0.00001;
                });
            });

        dct_matrix
            .iter_mut()
            .enumerate()
            .for_each(|(idx, val)| *val = idx as f32);

        DefaultKernelNoPadding::dct2d_impl(
            &dct_matrix,
            &input_buffer_127,
            &mut GenericArray::default(),
            &mut output_ref,
        );

        #[allow(unused_unsafe)]
        unsafe {
            x86::Avx2F32Kernel::dct2d_impl(
                &dct_matrix,
                &input_buffer_128,
                &mut GenericArray::default(),
                &mut output,
            );
        }

        for i in 0..16 {
            for j in 0..16 {
                let diff = (output_ref[i][j] - output[i][j]).abs();
                let diff_ref = diff / output_ref[i][j];
                // accept up to 0.00008% rounding error (8e-7), almost f32::EPSILON
                assert!(
                    diff_ref < 8e-7,
                    "difference: {:?} (relative: {:?})",
                    diff,
                    diff_ref
                );
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512", target_feature = "avx512f"))]
    #[test]
    fn test_dct64_impl_avx512() {
        let mut kernel = x86::Avx512F32Kernel;
        test_gradient_impl(&mut kernel);
        test_dct64_impl(&mut kernel, NumCast::from(5e-6).unwrap());
    }
}
