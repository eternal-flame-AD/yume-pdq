/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Tuesday, April 8, 2025
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

use core::simd::{LaneCount, Simd, Swizzle, cmp::SimdPartialOrd, num::SimdFloat};
use core::{
    fmt::{Debug, Display},
    ops::BitAnd,
};
use std::simd::num::{SimdInt, SimdUint};

use generic_array::{
    GenericArray,
    sequence::Flatten,
    typenum::{U16, U128, U512, Unsigned},
};

use super::{
    CONVOLUTION_OFFSET_512_TO_127, DCT_MATRIX_RMAJOR, Kernel, QUALITY_ADJUST_DIVISOR,
    TENT_FILTER_COLUMN_OFFSET, TENT_FILTER_EFFECTIVE_ROWS, TENT_FILTER_WEIGHTS_X8,
    type_traits::Term,
};

#[cfg(not(any(target_endian = "little", target_endian = "big")))]
compile_error!("sanity check: Platform must be either little-endian or big-endian");

#[cfg(all(target_endian = "little", target_endian = "big"))]
compile_error!(
    "sanity check: rustc reported target platform as both little-endian and big-endian!"
);

#[cfg(not(feature = "portable-simd-fma"))]
macro_rules! fma {
    (($a:expr) * ($b:expr) + ($c:expr)) => {
        ($a * $b) + $c
    };
    ($acc:ident += ($a:expr) * ($b:expr)) => {
        $acc = ($a * $b) + $acc;
    };
    (($acc:expr) += ($a:expr) * ($b:expr)) => {
        $acc = ($a * $b) + $acc;
    };
}

#[cfg(feature = "portable-simd-fma")]
macro_rules! fma {
    (($a:expr) * ($b:expr) + ($c:expr)) => {{
        use std::simd::StdFloat;
        $a.mul_add($b, $c)
    }};
    ($acc:ident += ($a:expr) * ($b:expr)) => {{
        use std::simd::StdFloat;
        $acc = $a.mul_add($b, $acc);
    }};
    (($acc:expr) += ($a:expr) * ($b:expr)) => {{
        use std::simd::StdFloat;
        $acc = $a.mul_add($b, $acc);
    }};
}

mod sealing {
    pub trait Sealed {}
}

type SimdPS<const N: usize> = Simd<f32, N>;

/// A private refinement trait of [`core::simd::SupportedLaneCount`] to further restrict the lane count supported.
pub trait SupportedLaneCount: core::simd::SupportedLaneCount + sealing::Sealed {}

macro_rules! supported_lane_count {
    ($($lanes:literal),+) => {
        $(
            impl sealing::Sealed for LaneCount<$lanes> {}
            impl SupportedLaneCount for LaneCount<$lanes> {}
        )+
    };
}

struct Pick<const I: usize, const N: usize>;

impl<const I: usize, const N: usize> Swizzle<N> for Pick<I, N> {
    const INDEX: [usize; N] = [I; N];
}

// 32, 64, 128, 256, 512 bit registers
supported_lane_count!(1, 2, 4, 8, 16);

#[derive(Clone, Copy, Default, PartialEq, Eq)]
/// Identification token for the portable SIMD kernel.
pub struct PortableSimdF32KernelIIdent<const N: usize> {
    _private: (),
}

impl<const N: usize> Debug for PortableSimdF32KernelIIdent<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PortableSimd<f32x{N}>")
    }
}

impl<const N: usize> Display for PortableSimdF32KernelIIdent<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("portable-simd (guided vectorization)")
    }
}

/// A kernel based on the currently nightly portable SIMD API.
///
/// This is optimized for an 8-lane f32 system but some stages can be optimized for narrower lanes.
///
/// Whether one specific stage is optimized for the custom lane counts in the const generic parameter is not part of the API guarantee.
#[derive(Clone, Copy, Default)]
pub struct PortableSimdF32Kernel<const N: usize>
where
    LaneCount<N>: SupportedLaneCount;

impl<const N: usize> Kernel for PortableSimdF32Kernel<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Buffer1WidthX = U128;
    type Buffer1LengthY = U128;
    type InternalFloat = f32;
    type InputDimension = U512;
    type OutputDimension = U16;
    type RequiredHardwareFeature = Term;
    type Ident = PortableSimdF32KernelIIdent<N>;

    fn ident(&self) -> Self::Ident {
        PortableSimdF32KernelIIdent { _private: () }
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input / (QUALITY_ADJUST_DIVISOR as f32);

        scaled.min(1.0)
    }

    fn cvt_rgba8_to_luma8f<const R_COEFF: u32, const G_COEFF: u32, const B_COEFF: u32>(
        &mut self,
        input: &GenericArray<GenericArray<u8, generic_array::typenum::U4>, Self::InputDimension>,
        output: &mut GenericArray<f32, Self::InputDimension>,
    ) where
        Self::RequiredHardwareFeature:
            super::type_traits::EvaluateHardwareFeature<EnabledStatic = generic_array::typenum::B1>,
    {
        unsafe {
            let mut in_offset = 0;
            let mut out_offset = 0;

            let coeff_r = SimdPS::<8>::splat(f32::from_ne_bytes(R_COEFF.to_ne_bytes()));
            let coeff_g = SimdPS::<8>::splat(f32::from_ne_bytes(G_COEFF.to_ne_bytes()));
            let coeff_b = SimdPS::<8>::splat(f32::from_ne_bytes(B_COEFF.to_ne_bytes()));

            macro_rules! do_loop {
                (8) => {
                    // 8 pixels in 32 bytes, place into exactly 8 output elements (1 lane)
                    // note this is little endian, so the ordering is pretty unintuitive
                    let data = input
                        .as_ptr()
                        .cast::<u8>()
                        .add(in_offset)
                        .cast::<Simd<u32, 8>>()
                        .read_unaligned();
                    let mask32 = Simd::<u32, 8>::splat(0x000000FF);
                    #[cfg(target_endian = "little")]
                    let output_r = (data & mask32).cast();
                    #[cfg(target_endian = "big")]
                    let output_r = (data >> 24 & mask32).cast();
                    #[cfg(target_endian = "little")]
                    let output_g = ((data >> 8) & mask32).cast();
                    #[cfg(target_endian = "big")]
                    let output_g = ((data >> 16) & mask32).cast();
                    #[cfg(target_endian = "little")]
                    let output_b = ((data >> 16) & mask32).cast();
                    #[cfg(target_endian = "big")]
                    let output_b = ((data >> 8) & mask32).cast();
                    let mut output_v: SimdPS<8> = output_r * coeff_r;
                    fma!(output_v += (output_g) * (coeff_g));
                    fma!(output_v += (output_b) * (coeff_b));
                    output
                        .as_mut_ptr()
                        .add(out_offset)
                        .cast::<SimdPS<8>>()
                        .write_unaligned(output_v);
                };
                (16) => {
                    do_loop!(8);
                    in_offset += 32;
                    out_offset += 8;
                    do_loop!(8);
                };
                (32) => {
                    do_loop!(16);
                    in_offset += 32;
                    out_offset += 8;
                    do_loop!(16);
                };
                (64) => {
                    do_loop!(32);
                    in_offset += 32;
                    out_offset += 8;
                    do_loop!(32);
                };
                (128) => {
                    do_loop!(64);
                    in_offset += 32;
                    out_offset += 8;
                    do_loop!(64);
                };
                (256) => {
                    do_loop!(128);
                    in_offset += 32;
                    out_offset += 8;
                    do_loop!(128);
                };
                (512) => {
                    do_loop!(256);
                    in_offset += 32;
                    out_offset += 8;
                    do_loop!(256);
                };
            }

            do_loop!(512);
        }
    }

    fn jarosz_compress(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::InputDimension>, Self::InputDimension>,
        output: &mut GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
    ) {
        for outi in 0..127 {
            let in_i = CONVOLUTION_OFFSET_512_TO_127[outi] - TENT_FILTER_COLUMN_OFFSET;
            for outj in 0..127 {
                let in_j = CONVOLUTION_OFFSET_512_TO_127[outj] - TENT_FILTER_COLUMN_OFFSET;
                let mut sum = SimdPS::<8>::splat(0.0);
                for di in 0..TENT_FILTER_EFFECTIVE_ROWS {
                    let mut offset = (in_i + di) * 512 + in_j;

                    // rewind one element to avoid out of bounds access
                    if di == 6 && outi == 126 && outj == 126 {
                        offset -= 1;
                    }

                    debug_assert!(
                        offset <= 512 * 512 - 8,
                        "offset out of bounds: {} is invalid, last valid offset is {}, in_i: {}, in_j: {}, di: {}, outi: {}, outj: {}",
                        offset,
                        512 * 512 - 8,
                        in_i,
                        in_j,
                        di,
                        outi,
                        outj
                    );

                    let mut buffer = unsafe {
                        buffer
                            .flatten()
                            .as_ptr()
                            .add(offset)
                            .cast::<SimdPS<8>>()
                            .read_unaligned()
                    };

                    // shift back one element to avoid out of bounds access
                    // [-1, 0, 1, 2, 3, 4, 5, 6] -> [0, 1, 2, 3, 4, 5, 6, 7]
                    if di == 6 && outi == 126 && outj == 126 {
                        buffer = buffer.shift_elements_left::<1>(0.0);
                    }

                    #[cfg(debug_assertions)]
                    let weights = SimdPS::<8>::from_slice(&TENT_FILTER_WEIGHTS_X8[di * 8..]);

                    #[cfg(not(debug_assertions))]
                    let weights = unsafe {
                        TENT_FILTER_WEIGHTS_X8
                            .as_ptr()
                            .add(di * 8)
                            .cast::<SimdPS<8>>()
                            .read_unaligned()
                    };

                    fma!(sum += (buffer) * (weights));
                }

                output[outi][outj] = sum.reduce_sum();
            }
        }

        /*
        crate::testing::dump_image("step_by_step/compress/portable_simd/output.ppm", output);
        */
    }

    fn pdqf_negate_off_diagonals(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        let mul_for_even = SimdPS::<16>::from_array([
            1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        ]);
        let mul_for_odd = -mul_for_even;
        let mut offset = 0;

        macro_rules! do_loop {
            (2) => {
                let row_0_0_16 = SimdPS::<16>::from_array(input[offset].into_array());
                let negated_0_0_16 = row_0_0_16 * mul_for_even;
                let row_1_0_16 = SimdPS::<16>::from_array(input[offset + 16].into_array());
                let negated_1_0_16 = row_1_0_16 * mul_for_odd;
                negated_0_0_16.copy_to_slice(&mut input[offset]);
                negated_1_0_16.copy_to_slice(&mut input[offset + 16]);
            };
            (4) => {
                do_loop!(2);
                offset += 32;
                do_loop!(2);
            };
            (8) => {
                do_loop!(4);
                offset += 32;
                do_loop!(4);
            };
            (16) => {
                do_loop!(8);
                offset += 32;
                do_loop!(8);
            };
        }

        do_loop!(16);
    }

    fn pdqf_negate_alt_cols<const INVERT: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) where
        Self::RequiredHardwareFeature:
            super::type_traits::EvaluateHardwareFeature<EnabledStatic = generic_array::typenum::B1>,
    {
        let coefficients = SimdPS::<16>::from_array(if INVERT {
            [
                1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                -1.0,
            ]
        } else {
            [
                -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
                1.0,
            ]
        });

        for i in 0..16 {
            let row = input[i];
            let row = SimdPS::<16>::from_array(row.into_array());
            let negated = row * coefficients;
            negated.copy_to_slice(&mut input[i]);
        }
    }

    fn pdqf_negate_alt_rows<const NEGATE: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) where
        Self::RequiredHardwareFeature:
            super::type_traits::EvaluateHardwareFeature<EnabledStatic = generic_array::typenum::B1>,
    {
        let mut offset = if NEGATE { 0 } else { 1 };

        macro_rules! do_loop {
            (2) => {
                let mut row_0_0_16 = SimdPS::<16>::from_array(input[offset].into_array());
                row_0_0_16 = -row_0_0_16;
                row_0_0_16.copy_to_slice(&mut input[offset]);
            };
            (4) => {
                do_loop!(2);
                offset += 1;
                do_loop!(2);
            };
            (8) => {
                do_loop!(4);
                offset += 1;
                do_loop!(4);
            };
            (16) => {
                do_loop!(8);
                offset += 1;
                do_loop!(8);
            };
        }

        do_loop!(16);
    }

    fn sum_of_gradients(
        &mut self,
        input: &generic_array::GenericArray<
            generic_array::GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> f32 {
        let mut gradient_sum_v = SimdPS::<16>::splat(0.0);

        for i in 0..(16 - 1) {
            let u_v = SimdPS::<16>::from_slice(&input[i][..]);
            let v_v = SimdPS::<16>::from_slice(&input[i + 1][..]);
            let d_v = u_v - v_v;
            let abs_d_v = d_v.abs();
            gradient_sum_v += abs_d_v;
        }

        let mut gradient_sum_v_16 = SimdPS::<16>::splat(0.0);

        for i in 0..16 {
            let row = SimdPS::<16>::from_slice(&input[i]);
            let shl1 = row.shift_elements_left::<1>(0.0);

            let d_v = row - shl1;
            let abs_d_v = d_v.abs();
            gradient_sum_v_16 += abs_d_v;
        }

        gradient_sum_v.reduce_sum() + gradient_sum_v_16.reduce_sum()
    }

    fn dct2d(
        &mut self,
        buffer: &generic_array::GenericArray<
            generic_array::GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
        tmp: &mut generic_array::GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
        output: &mut generic_array::GenericArray<
            generic_array::GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        #[cfg(debug_assertions)]
        #[allow(clippy::float_cmp)]
        {
            for i in 0..128 {
                assert_eq!(
                    buffer[i][127], 0.0,
                    "padding in buffer is not zero (i: {i}, j: 127)",
                );
            }
            for j in 0..128 {
                assert_eq!(
                    buffer[127][j], 0.0,
                    "padding in buffer is not zero (i: 127, j: {j})",
                );
            }
        }
        for k in 0..16 {
            for j_by_8 in (0..128usize).step_by(8) {
                let mut sumks = [SimdPS::<8>::splat(0.0); 2];

                for outer in (0..128).step_by(8) {
                    #[allow(unsafe_code, reason = "zero padded, so this is safe, intentional OOB")]
                    let dct_row = unsafe {
                        DCT_MATRIX_RMAJOR
                            .as_ptr()
                            .add(k * super::DctMatrixNumCols::USIZE + outer)
                            .cast::<SimdPS<8>>()
                            .read_unaligned()
                    };

                    let dct_row_swizzled = [
                        Pick::<0, 8>::swizzle::<f32, 8>(dct_row),
                        Pick::<1, 8>::swizzle::<f32, 8>(dct_row),
                        Pick::<2, 8>::swizzle::<f32, 8>(dct_row),
                        Pick::<3, 8>::swizzle::<f32, 8>(dct_row),
                        Pick::<4, 8>::swizzle::<f32, 8>(dct_row),
                        Pick::<5, 8>::swizzle::<f32, 8>(dct_row),
                        Pick::<6, 8>::swizzle::<f32, 8>(dct_row),
                        Pick::<7, 8>::swizzle::<f32, 8>(dct_row),
                    ];

                    let mut inner = 0;

                    let mut sum_target = true;

                    // some unrolling encouragement
                    macro_rules! do_loop {
                        (1) => {
                            let k2 = outer + inner;

                            #[cfg(debug_assertions)]
                            {
                                let offset = k2 * Self::Buffer1LengthY::USIZE + j_by_8;
                                if offset > 512 * 512 - 8 {
                                    panic!(
                                        "offset out of bounds: {} is invalid, last valid offset is {}, k: {}, j_by_8: {}, outer: {}, inner: {}",
                                        offset, 512 * 512 - 8, k, j_by_8, outer, inner
                                    );
                                }
                            }


                            let buf_row = unsafe {
                                buffer[k2][j_by_8..]
                                    .as_ptr()
                                    .cast::<SimdPS<8>>()
                                    .read_unaligned()
                            };
                            let dct = dct_row_swizzled[inner];
                            fma!((sumks[sum_target as usize]) += (buf_row) * (dct));
                        };
                        (2) => {
                            do_loop!(1);
                            inner += 1;
                            sum_target = !sum_target;
                            do_loop!(1);
                        };
                        (4) => {
                            do_loop!(2);
                            inner += 1;
                            do_loop!(2);
                        };
                        (8) => {
                            do_loop!(4);
                            inner += 1;
                            do_loop!(4);
                        };
                    }

                    do_loop!(8);
                }

                let sum0 = sumks[0];

                let sum0_nan_mask = sum0.is_nan();

                let sum0_nan_zeroed = sum0_nan_mask.select(SimdPS::<8>::splat(0.0), sum0);

                let sum1 = sumks[1];

                let sum1_nan_mask = sum1.is_nan();

                let sum1_nan_zeroed = sum1_nan_mask.select(SimdPS::<8>::splat(0.0), sum1);

                let final_sum = sum0_nan_zeroed + sum1_nan_zeroed;

                final_sum.copy_to_slice(&mut tmp[j_by_8..]);
            }

            for j in 0..16 {
                let mut sumkh = [SimdPS::<8>::splat(0.0); 2];
                let mut sum_target = true;

                let dct_ptr = DCT_MATRIX_RMAJOR.as_ptr();
                let mut j2 = 0;

                macro_rules! do_loop {
                    (8) => {
                        let tmp_row = SimdPS::<8>::from_slice(&tmp[j2..]);
                        let dct_row = unsafe {
                            dct_ptr
                                .add(j * super::DctMatrixNumCols::USIZE + j2)
                                .cast::<SimdPS<8>>()
                                .read_unaligned()
                        };
                        fma!((sumkh[sum_target as usize]) += (tmp_row) * (dct_row));
                    };
                    (16) => {
                        do_loop!(8);
                        sum_target = !sum_target;
                        j2 += 8;
                        do_loop!(8);
                    };
                    (32) => {
                        do_loop!(16);
                        j2 += 8;
                        do_loop!(16);
                    };
                    (64) => {
                        do_loop!(32);
                        j2 += 8;
                        do_loop!(32);
                    };
                    (128) => {
                        do_loop!(64);
                        j2 += 8;
                        do_loop!(64);
                    };
                }

                do_loop!(128);

                let sum0 = sumkh[0];
                let sum1 = sumkh[1];

                let sum0_nan_mask = sum0.is_nan();
                let sum1_nan_mask = sum1.is_nan();

                let sum0_nan_zeroed = sum0_nan_mask.select(SimdPS::<8>::splat(0.0), sum0);
                let sum1_nan_zeroed = sum1_nan_mask.select(SimdPS::<8>::splat(0.0), sum1);

                let final_sum = sum0_nan_zeroed + sum1_nan_zeroed;

                output[k][j] = final_sum.reduce_sum();
            }
        }

        /*
        crate::testing::dump_image("step_by_step/dct2d/portable_simd/output.ppm", output);
        */
    }

    fn quantize(
        &mut self,
        input: &generic_array::GenericArray<
            generic_array::GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
        threshold: &mut Self::InternalFloat,
        output: &mut generic_array::GenericArray<
            generic_array::GenericArray<
                u8,
                <Self::OutputDimension as super::type_traits::DivisibleBy8>::Output,
            >,
            Self::OutputDimension,
        >,
    ) {
        let mut min_v = SimdPS::<16>::splat(f32::MAX);
        let mut max_v = SimdPS::<16>::splat(f32::MIN);

        let output_flattened = output.flatten();

        for chunk in input.flatten().chunks(16) {
            let vec = SimdPS::<16>::from_slice(chunk);
            min_v = min_v.simd_min(vec);
            max_v = max_v.simd_max(vec);
        }

        let mut min = min_v.reduce_min();
        let mut max = max_v.reduce_max();
        let half_point = 16 * 16 / 2;

        let mut guess = (min + max) * 0.5;
        let mut num_over = 0; // how many elements are beyond the search max?
        let mut num_under = 0; // how many elements are below the search min?

        // Binary search with SIMD with inter-searchspace statistics for reducing oscillations
        #[allow(clippy::items_after_statements)]
        const MAX_ITER: usize = 32;
        #[cfg_attr(not(debug_assertions), allow(unused))]
        let mut converged = false;
        for iter in 0..MAX_ITER {
            debug_assert!(
                guess.is_finite(),
                "guess is not finite but {guess:?} (iter: {iter})",
            );
            let guess_v = SimdPS::<8>::splat(guess);
            let mut gt_count = 0;
            let mut lt_count = 0;
            let mut resid_sum_gt_v = SimdPS::<8>::splat(0.0);
            let mut resid_sum_lt_v = SimdPS::<8>::splat(0.0);
            let min_v = SimdPS::<8>::splat(min);
            let max_v = SimdPS::<8>::splat(max);

            let mut output_iter_u8 = output_flattened.iter_mut().rev();
            for chunk in input.flatten().chunks(8) {
                let row = SimdPS::<8>::from_slice(chunk);
                let cmp_gt = row.simd_gt(guess_v);
                let mask = cmp_gt.to_bitmask();
                *output_iter_u8.next().unwrap() = (mask & 0xFF) as u8;

                let cmp_lt = row.simd_lt(guess_v);
                let cmp_max_lt = row.simd_le(max_v);
                let cmp_min_gt = row.simd_ge(min_v);

                let mask_resid_gt = cmp_gt.bitand(cmp_max_lt);
                let mask_resid_lt = cmp_lt.bitand(cmp_min_gt);

                // Count elements and sum values
                gt_count += mask_resid_gt.to_bitmask().count_ones();
                lt_count += mask_resid_lt.to_bitmask().count_ones();

                let row_bits = row.to_bits();
                let mask_gt_bits = mask_resid_gt.to_int().cast();
                let mask_lt_bits = mask_resid_lt.to_int().cast();

                resid_sum_gt_v += SimdPS::<8>::from_bits(row_bits & mask_gt_bits);
                resid_sum_lt_v += SimdPS::<8>::from_bits(row_bits & mask_lt_bits);
            }

            /*
            crate::testing::dump_thresholding_diagnostic(
                &format!("step_by_step/quantize/portable_simd/iter_{_iter}.ppm"),
                input,
                |&p: _| {
                    if p < guess {
                        Some(false)
                    } else if p > guess {
                        Some(true)
                    } else {
                        None
                    }
                },
            );

            */

            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            if gt_count + num_over > half_point {
                num_under += lt_count;
                min = guess;
                guess = resid_sum_gt_v.reduce_sum() / gt_count as f32;
            } else if lt_count + num_under > half_point {
                num_over += gt_count;
                max = guess;
                guess = resid_sum_lt_v.reduce_sum() / lt_count as f32;
            } else {
                converged = true;
                break;
            }
        }
        debug_assert!(converged, "quantization did not converge");

        *threshold = guess;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portable_simd_pick_swizzle() {
        let value = SimdPS::<8>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        [
            Pick::<0, 8>::swizzle::<f32, 8>(value),
            Pick::<1, 8>::swizzle::<f32, 8>(value),
            Pick::<2, 8>::swizzle::<f32, 8>(value),
            Pick::<3, 8>::swizzle::<f32, 8>(value),
            Pick::<4, 8>::swizzle::<f32, 8>(value),
            Pick::<5, 8>::swizzle::<f32, 8>(value),
            Pick::<6, 8>::swizzle::<f32, 8>(value),
        ]
        .into_iter()
        .enumerate()
        .for_each(|(i, x)| {
            assert_eq!(x, SimdPS::<8>::splat(i as f32 + 1.0));
        });
    }

    #[test]
    fn test_portable_simd_mask_endianness() {
        let value = SimdPS::<8>::from_slice(&[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        let is_positive = value.simd_gt(SimdPS::<8>::splat(0.0));
        let mask = is_positive.to_bitmask() as u8;
        assert_eq!(mask, 0b00001111);
    }
}
