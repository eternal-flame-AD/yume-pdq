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

use core::fmt::{Debug, Display};

#[cfg_attr(not(feature = "reference-rug"), allow(unused_imports))]
use generic_array::{
    ArrayLength, GenericArray,
    typenum::{U16, U127, U256},
};
use generic_array::{
    sequence::Flatten,
    typenum::{U2, U512},
};

#[cfg_attr(not(feature = "reference-rug"), allow(unused_imports))]
use num_traits::{FromPrimitive, NumCast, Signed, ToPrimitive, float::FloatCore};

pub use generic_array;
use sealing::Sealed;
use type_traits::{DivisibleBy8, SquareOf};

/// Kernels based on x86-64 intrinsics.
#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(feature = "reference-rug")]
/// 128-bit floating point type.
pub mod float128;

/// Dihedral flips.
#[cfg(feature = "unstable")]
pub mod dihedral;

/// Thresholding methods.
pub mod threshold;

/// Type traits.
pub mod type_traits;

include!(concat!(env!("OUT_DIR"), "/dct_matrix.rs"));
include!(concat!(env!("OUT_DIR"), "/tent_filter_weights.rs"));
include!(concat!(env!("OUT_DIR"), "/convolution_offset.rs"));

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

/// Marker trait to indicate that this kernel produces precise results.
pub trait PreciseKernel: Kernel {}

// Copied verbatim from [darwinium-com/pdqhash](https://github.com/darwinium-com/pdqhash/blob/main/src/lib.rs).
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
pub const QUALITY_ADJUST_DIVISOR: usize = 256 * 2048 * 64;

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

    /// Apply a tent-filter average to every 8x8 sub-block of the input buffer and write the result of each sub-block to the output buffer.
    fn jarosz_compress(
        &mut self,
        _buffer: &GenericArray<GenericArray<f32, Self::InputDimension>, Self::InputDimension>,
        _output: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
    );

    /// Convert input to binary by thresholding at median
    ///
    /// # Parameters
    ///
    /// * `input`: The input buffer.
    /// * `threshold`: The threshold value.
    /// * `output`: The output buffer.
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
        <Self as Kernel>::OutputDimension: DivisibleBy8;

    /// Compute the sum of gradients of the input buffer in both horizontal and vertical directions.
    fn sum_of_gradients(
        &mut self,
        _input: &GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> Self::InternalFloat;

    /// Adjust the quality metric to be between 0 and 1.
    fn adjust_quality(_input: Self::InternalFloat) -> f32;

    /// Apply a 2D DCT-II transformation to the input buffer write the result to the output buffer.
    fn dct2d(
        &mut self,
        _buffer: &GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
        _output: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    );
}

/// A pure-Rust implementation of the `Kernel` trait.
pub struct DefaultKernel;

impl DefaultKernel {
    #[inline(always)]
    pub(crate) fn dct2d_impl(
        dct_matrix_rmajor: &[f32; 16 * 127],
        buffer: &GenericArray<GenericArray<f32, U127>, U127>,
        output: &mut GenericArray<GenericArray<f32, U16>, U16>,
    ) {
        for k in 0..16 {
            let mut tmp = [0.0; 127];
            for j in 0..127 {
                let mut sumks = [0.0; 4];
                for (k2, sumk) in (0..127).zip((0..4).cycle()) {
                    sumks[sumk] += dct_matrix_rmajor[k * DCT_MATRIX_NCOLS + k2] * buffer[k2][j];
                }

                tmp[j] = sumks[0] + sumks[1] + sumks[2] + sumks[3];
            }

            for j in 0..DCT_MATRIX_NROWS {
                let mut sumks = [0.0; 4];
                for (m, sumk) in (0..DCT_MATRIX_NCOLS).zip((0..4).cycle()) {
                    sumks[sumk] += tmp[m] * dct_matrix_rmajor[j * DCT_MATRIX_NCOLS + m];
                }
                output[k][j] = sumks[0] + sumks[1] + sumks[2] + sumks[3];
            }
        }
    }
}

impl Kernel for DefaultKernel {
    type Buffer1WidthX = U127;
    type Buffer1LengthY = U127;
    type InternalFloat = f32;
    type InputDimension = U512;
    type OutputDimension = U16;

    fn quantize(
        &mut self,
        input: &GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
        threshold: &mut Self::InternalFloat,
        output: &mut GenericArray<GenericArray<u8, U2>, U16>,
    ) {
        let median = torben_median(input);
        *threshold = median;
        let output = output.flatten();
        for (i, j) in input.iter().flatten().enumerate() {
            output[32 - 1 - i / 8] += if *j > median { 1 << (i % 8) } else { 0 };
        }
    }

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
        output: &mut GenericArray<GenericArray<f32, U16>, U16>,
    ) {
        DefaultKernel::dct2d_impl(&DCT_MATRIX_RMAJOR, buffer, output);
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
impl Kernel for ReferenceKernel<f32> {
    type Buffer1WidthX = U127;
    type Buffer1LengthY = U127;
    type InternalFloat = f32;
    type InputDimension = U512;
    type OutputDimension = U16;

    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input / (QUALITY_ADJUST_DIVISOR as f32);

        scaled.min(1.0)
    }

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        output: &mut GenericArray<GenericArray<f32, U16>, U16>,
    ) {
        let mut intermediate_matrix = [0.0; 127];
        for i in 0..16 {
            for j in 0..127 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk += DCT_MATRIX_RMAJOR[i * DCT_MATRIX_NCOLS + k] * buffer[k][j];
                }

                intermediate_matrix[j] = sumk;
            }

            for j in 0..16 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk += intermediate_matrix[k] * DCT_MATRIX_RMAJOR[j * DCT_MATRIX_NCOLS + k];
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

    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input / (QUALITY_ADJUST_DIVISOR as f64);

        scaled.min(1.0) as f32
    }

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f64, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        output: &mut GenericArray<GenericArray<f64, U16>, U16>,
    ) {
        let mut intermediate_matrix = [0.0; 127];
        for i in 0..16 {
            for j in 0..127 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk += DCT_MATRIX_RMAJOR_64[i * DCT_MATRIX_NCOLS + k] * buffer[k][j];
                }

                intermediate_matrix[j] = sumk;
            }

            for j in 0..16 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk += intermediate_matrix[k] * DCT_MATRIX_RMAJOR_64[j * DCT_MATRIX_NCOLS + k];
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
            .map(|s| *s as f64)
            .collect::<Vec<_>>();
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

    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input / (float128::ArbFloat::from_usize(QUALITY_ADJUST_DIVISOR).unwrap());

        let one = float128::ArbFloat::from_f32(1.0).unwrap();
        if scaled > one { 1.0 } else { scaled.to_f32() }
    }

    fn sum_of_gradients(
        &mut self,
        input: &GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
    ) -> Self::InternalFloat {
        let mut gradient_sum = Default::default();

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

#[cfg(test)]
mod tests {
    use num_traits::NumCast;
    use rand::Rng;

    use super::*;

    extern crate alloc;

    #[allow(dead_code)]
    fn test_gradient_impl<OD: ArrayLength, K: Kernel<OutputDimension = OD>>(kernel: &mut K)
    where
        ReferenceKernel<<K as Kernel>::InternalFloat>:
            Kernel<InternalFloat = <K as Kernel>::InternalFloat, OutputDimension = OD>,
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
        ReferenceKernel<K::InternalFloat>: Kernel<InternalFloat = K::InternalFloat>,
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
        ReferenceKernel::<K::InternalFloat>::default().dct2d(&input_ref, &mut output_ref);

        kernel.dct2d(&input, &mut output);

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
        let mut kernel = DefaultKernel;
        test_dct64_impl(&mut kernel, NumCast::from(f32::EPSILON * 32.00).unwrap());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dct64_impl_avx2_equivalence() {
        // do an empirical equivalence check between the default and AVX2 implementations

        use generic_array::typenum::U128;

        let mut dct_matrix = [1.0; 16 * 127];

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

        DefaultKernel::dct2d_impl(&dct_matrix, &input_buffer_127, &mut output_ref);

        x86::Avx2F32Kernel::dct2d_impl(&dct_matrix, &input_buffer_128, &mut output);

        for i in 0..16 {
            print!("(reference, actual) = [");
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
                print!("({} {}) ", output_ref[i][j], output[i][j]);
            }
            println!("]");
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

        DefaultKernel::dct2d_impl(&dct_matrix, &input_buffer_127, &mut output_ref);
        x86::Avx2F32Kernel::dct2d_impl(&dct_matrix, &input_buffer_128, &mut output);

        for i in 0..16 {
            print!("(reference, actual) = [");
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
                print!("({} {}) ", output_ref[i][j], output[i][j]);
            }
            println!("]");
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_dct64_impl_avx512() {
        let mut kernel = x86::Avx512F32Kernel;
        test_gradient_impl(&mut kernel);
        test_dct64_impl(&mut kernel, NumCast::from(5e-6).unwrap());
    }
}
