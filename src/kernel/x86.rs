/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Sunday, March 23, 2025
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

use core::arch::x86_64::*;

use generic_array::typenum::{IsGreaterOrEqual, U8, Unsigned};
#[allow(unused_imports)]
use generic_array::{
    ArrayLength, GenericArray,
    sequence::Flatten,
    typenum::{B0, B1, U2, U16, U128, U512},
};

use crate::{
    alignment::{Align32, DefaultPaddedArray},
    kernel::Kernel,
};

use super::{
    CONVOLUTION_OFFSET_512_TO_127, DCT_MATRIX_RMAJOR, QUALITY_ADJUST_DIVISOR,
    TENT_FILTER_COLUMN_OFFSET, TENT_FILTER_EFFECTIVE_ROWS, TENT_FILTER_WEIGHTS_X8,
    type_traits::{
        EvaluateHardwareFeature, RequireCompilerTimeHardwareFeature, kernel_sealing::KernelSealed,
    },
};

const CPUID_REG_EBX: u8 = 0x00;
const CPUID_REG_ECX: u8 = 0x01;

/// Type-level CPUID flags for x86_64
pub struct X8664CPUID<const LEAF: u32, const SUB_LEAF: u32, const REG: u8, const BIT: u8> {
    _private: (),
}

impl<const LEAF: u32, const SUB_LEAF: u32, const REG: u8, const BIT: u8> KernelSealed
    for X8664CPUID<LEAF, SUB_LEAF, REG, BIT>
{
}

impl<const LEAF: u32, const SUB_LEAF: u32, const REG: u8, const BIT: u8>
    X8664CPUID<LEAF, SUB_LEAF, REG, BIT>
{
    #[inline]
    fn test_runtime() -> bool {
        // use intrinsics to check if the feature is supported instead of the is_x86_feature_detected macro
        // because it does not work well with -Zsanitize options
        unsafe {
            let res = match REG {
                CPUID_REG_EBX => __cpuid_count(LEAF, SUB_LEAF).ebx,
                CPUID_REG_ECX => __cpuid_count(LEAF, SUB_LEAF).ecx,
                _ => unreachable!(),
            };
            (1 << BIT) & res != 0
        }
    }
}

macro_rules! define_x8664_cpuid {
    ($name:literal, $leaf:expr, $sub_leaf:expr, $reg:expr, $bit:expr) => {
        impl EvaluateHardwareFeature for X8664CPUID<$leaf, $sub_leaf, $reg, $bit> {
            type Name = &'static str;

            #[cfg(target_feature = $name)]
            type EnabledStatic = B1;

            #[cfg(not(target_feature = $name))]
            type EnabledStatic = B0;

            type MustCheck = B1;

            fn name() -> Self::Name {
                $name
            }

            fn met_runtime() -> bool {
                Self::test_runtime()
            }
        }
    };
}

define_x8664_cpuid!("fma", 1, 0, CPUID_REG_ECX, 12);

/// Type alias for FMA feature.
pub type CpuIdFma = X8664CPUID<1, 0, CPUID_REG_ECX, 12>;

define_x8664_cpuid!("avx2", 7, 0, CPUID_REG_EBX, 5);

/// Type alias for AVX2 feature.
pub type CpuIdAvx2 = X8664CPUID<7, 0, CPUID_REG_EBX, 5>;

define_x8664_cpuid!("avx512f", 7, 0, CPUID_REG_EBX, 16);

/// Type alias for AVX512F feature.
pub type CpuIdAvx512f = X8664CPUID<7, 0, CPUID_REG_EBX, 16>;

/// Compute kernel using hand-written AVX2 intrinsics.
///
/// Note: This would produce a slightly different numeric result than scalar implementation due to:
///
/// - different rounding rules in the intrinsics by generally within 5 times epsilon (~5e-7) when tested on a random input.
///
/// Testing shows that the difference is usually within 1 bit (out of 256) on test images.
///
/// Benchmark shows a ~10x improvement throughput over the generic (auto-vectorized) implementation, ~100x over the reference implementation.
///
/// Note: You MUST compile with `target-feature=+avx2` (or equivalent) to use this kernel, otherwise a very slow
/// fallback will be emitted by LLVM.
#[derive(Clone, Copy, Default)]
pub struct Avx2F32Kernel;

impl Avx2F32Kernel {
    #[inline(always)]
    pub(crate) fn dct2d_impl<P: ArrayLength + IsGreaterOrEqual<U8>>(
        dct_matrix_rmajor: &DefaultPaddedArray<f32, super::DctMatrixNumElements, P>,
        buffer: &GenericArray<GenericArray<f32, U128>, U128>,
        tmp: &mut GenericArray<f32, U128>,
        output: &mut GenericArray<GenericArray<f32, U16>, U16>,
    ) {
        let mut out_buffer = Align32([0.0; 8]);
        unsafe {
            // crate::testing::dump_image("step_by_step/dct2d/avx2/input.ppm", buffer);

            let shift1 = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
            let shift2 = _mm256_set_epi32(1, 0, 7, 6, 5, 4, 3, 2);
            let shift4 = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);

            macro_rules! horizontal {
                (max($a:expr)) => {{
                    let pairwise = _mm256_max_ps($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 =
                        _mm256_max_ps(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = _mm256_max_ps(every4, _mm256_permutevar8x32_ps(every4, shift4));
                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
                (min($a:expr)) => {{
                    let pairwise = _mm256_min_ps($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 =
                        _mm256_min_ps(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = _mm256_min_ps(every4, _mm256_permutevar8x32_ps(every4, shift4));

                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
                (sum($a:expr)) => {{
                    let pairwise = _mm256_add_ps($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 =
                        _mm256_add_ps(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = _mm256_add_ps(every4, _mm256_permutevar8x32_ps(every4, shift4));

                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
            }

            for k in 0..16 {
                // vectorize j by 32x8 lanes, unroll inner loop by 8 w.r.t. DCT loading
                for j_by_8 in (0..128).step_by(8) {
                    let mut sumk = [_mm256_setzero_ps(); 2];

                    let mut sum_target = true;

                    // ~~it is very important to skip the last iteration as tmp has unknown padding~~
                    // now we clear NaN instead of explicitly skipping the last iteration
                    // which handles edge case images return NaN quality for some situation as well for user experience

                    for k2_base in (0..128).step_by(8) {
                        let mut k2 = k2_base;

                        macro_rules! do_loop {
                            (1) => {
                                let buf_row = _mm256_loadu_ps(&buffer[k2][j_by_8]);
                                // Broadcast the i-th column of D^t, multiply with the j-th row of the image
                                let dct = _mm256_set1_ps(
                                    *dct_matrix_rmajor
                                        .as_ptr()
                                        .add(k * super::DctMatrixNumCols::USIZE + k2),
                                );
                                // add the result to the sum
                                sumk[sum_target as usize] = _mm256_fmadd_ps(buf_row, dct, sumk[sum_target as usize]);
                            };
                            (2) => {
                                do_loop!(1);
                                sum_target = !sum_target;
                                k2 += 1;
                                do_loop!(1);
                            };
                            (4) => {
                                do_loop!(2);
                                k2 += 1;
                                do_loop!(2);
                            };
                            (8) => {
                                do_loop!(4);
                                k2 += 1;
                                do_loop!(4);
                            };
                        }

                        do_loop!(8);
                    }

                    let sumk0 = sumk[0];
                    let sumk1 = sumk[1];

                    let sumk0_not_nan_mask = _mm256_cmp_ps(sumk0, _mm256_setzero_ps(), _CMP_ORD_S);

                    let sumk0_nan_zeroed =
                        _mm256_blendv_ps(_mm256_setzero_ps(), sumk0, sumk0_not_nan_mask);

                    let sumk1_not_nan_mask = _mm256_cmp_ps(sumk1, _mm256_setzero_ps(), _CMP_ORD_S);

                    let sumk1_nan_zeroed =
                        _mm256_blendv_ps(_mm256_setzero_ps(), sumk1, sumk1_not_nan_mask);

                    _mm256_storeu_ps(
                        tmp.as_mut_ptr().add(j_by_8),
                        _mm256_add_ps(sumk0_nan_zeroed, sumk1_nan_zeroed),
                    );
                }

                for j in 0..16 {
                    let tmp_ptr = tmp.as_ptr();
                    let mut sumkh = [_mm256_setzero_ps(); 2];
                    let mut sum_target = true;
                    let mut j2 = 0;

                    let dct_ptr = dct_matrix_rmajor
                        .as_ptr()
                        .add(j * super::DctMatrixNumCols::USIZE);

                    macro_rules! do_loop {
                        (8) => {
                            let tmp_row = _mm256_loadu_ps(tmp_ptr.add(j2));
                            let dct_row = _mm256_loadu_ps(dct_ptr.add(j2));
                            sumkh[sum_target as usize] =
                                _mm256_fmadd_ps(tmp_row, dct_row, sumkh[sum_target as usize]);
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

                    let sumkh = _mm256_add_ps(sumkh[0], sumkh[1]);

                    output[k][j] = horizontal!(sum(sumkh));

                    // crate::testing::dump_image(
                    //     &format!("step_by_step/dct2d/avx2/round/{k}.ppm"),
                    //     output,
                    // );
                }
            }
            // crate::testing::dump_image("step_by_step/dct2d/avx2/output.ppm", output);
        }
    }
}

unsafe fn jarosz_compress_avx2<Buffer1WidthX: ArrayLength, Buffer1LengthY: ArrayLength>(
    buffer: &GenericArray<GenericArray<f32, U512>, U512>,
    output: &mut GenericArray<GenericArray<f32, Buffer1WidthX>, Buffer1LengthY>,
) {
    unsafe {
        // crate::testing::dump_image("step_by_step/compress/avx2/input.ppm", buffer);
        let mut out_buffer = Align32([0.0; 8]);

        // little endian:                               [7] [6] [5] [4] [3] [2] [1] [0]
        // intended index: [-1, 0, 1, 2, 3, 4, 5, 6] -> [0,  1,  2,  3,  4,  5,  6,  7]
        // the last one is not important (padded into zero by FMA)
        let shiftl1 = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);

        for outi in 0..127 {
            let in_i = CONVOLUTION_OFFSET_512_TO_127[outi] - TENT_FILTER_COLUMN_OFFSET;
            for outj in 0..127 {
                let in_j = CONVOLUTION_OFFSET_512_TO_127[outj] - TENT_FILTER_COLUMN_OFFSET;
                let mut sum = _mm256_setzero_ps();
                for di in 0..TENT_FILTER_EFFECTIVE_ROWS {
                    let mut offset = (in_i + di) * 512 + in_j;

                    // rewind one element to avoid out of bounds access
                    if di == 6 && outi == 126 && outj == 126 {
                        offset -= 1;
                    }

                    #[cfg(debug_assertions)]
                    {
                        if offset > 512 * 512 - 8 {
                            panic!(
                                "offset out of bounds: {} is invalid, last valid offset is {}, in_i: {}, in_j: {}, di: {}, outi: {}, outj: {}",
                                offset,
                                512 * 512 - 8,
                                in_i,
                                in_j,
                                di,
                                outi,
                                outj
                            );
                        }
                    }

                    let mut buffer = _mm256_loadu_ps(buffer.flatten().as_ptr().add(offset));

                    // shift back one element
                    if di == 6 && outi == 126 && outj == 126 {
                        buffer = _mm256_permutevar8x32_ps(buffer, shiftl1);
                    }

                    let weights = _mm256_loadu_ps(TENT_FILTER_WEIGHTS_X8.as_ptr().add(di * 8));
                    sum = _mm256_fmadd_ps(buffer, weights, sum);
                }
                sum = _mm256_hadd_ps(sum, _mm256_setzero_ps());
                sum = _mm256_hadd_ps(sum, _mm256_setzero_ps());
                _mm256_store_ps(out_buffer.0.as_mut_ptr(), sum);
                output[outi][outj] = out_buffer.0[0] + out_buffer.0[4];
            }
        }
        // crate::testing::dump_image("step_by_step/compress/avx2/output.ppm", output);
    }
}

#[inline]
fn cvt_rgba8_to_luma8f_avx2<const R_COEFF: u32, const G_COEFF: u32, const B_COEFF: u32>(
    input: &GenericArray<GenericArray<u8, generic_array::typenum::U4>, U512>,
    output: &mut GenericArray<f32, U512>,
) {
    unsafe {
        let mut in_offset = 0;
        let mut out_offset = 0;
        let coeff_r = _mm256_set1_ps(f32::from_ne_bytes(R_COEFF.to_ne_bytes()));
        let coeff_g = _mm256_set1_ps(f32::from_ne_bytes(G_COEFF.to_ne_bytes()));
        let coeff_b = _mm256_set1_ps(f32::from_ne_bytes(B_COEFF.to_ne_bytes()));

        macro_rules! do_loop {
            (8) => {
                // 8 pixels in 32 bytes, place into exactly 8 output elements (1 lane)
                // note this is little endian, so the ordering is pretty unintuitive
                let data = _mm256_loadu_si256(input.as_ptr().cast::<u8>().add(in_offset).cast());
                let mask32 = _mm256_set1_epi32(0x000000FF);
                let output_r = _mm256_cvtepi32_ps(_mm256_and_si256(data, mask32));
                let output_g =
                    _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32::<8>(data), mask32));
                let output_b =
                    _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32::<16>(data), mask32));
                let mut output_v = _mm256_mul_ps(output_r, coeff_r);
                output_v = _mm256_fmadd_ps(output_g, coeff_g, output_v);
                output_v = _mm256_fmadd_ps(output_b, coeff_b, output_v);
                _mm256_storeu_ps(output.as_mut_ptr().add(out_offset), output_v);
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

impl Kernel for Avx2F32Kernel {
    type Buffer1WidthX = U128;
    type Buffer1LengthY = U128;
    type InternalFloat = f32;
    type InputDimension = U512;
    type OutputDimension = U16;
    type RequiredHardwareFeature = RequireCompilerTimeHardwareFeature<CpuIdAvx2, CpuIdFma>;
    type Ident = &'static str;

    fn ident(&self) -> &'static str {
        "avx2_f32"
    }

    #[inline]
    fn cvt_rgba8_to_luma8f<const R_COEFF: u32, const G_COEFF: u32, const B_COEFF: u32>(
        &mut self,
        input: &GenericArray<GenericArray<u8, generic_array::typenum::U4>, Self::InputDimension>,
        output: &mut GenericArray<f32, Self::InputDimension>,
    ) {
        cvt_rgba8_to_luma8f_avx2::<R_COEFF, G_COEFF, B_COEFF>(input, output);
    }

    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input / (QUALITY_ADJUST_DIVISOR as f32);

        scaled.min(1.0)
    }

    fn jarosz_compress(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::InputDimension>, Self::InputDimension>,
        output: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
    ) {
        unsafe {
            jarosz_compress_avx2(buffer, output);
        }
    }

    fn pdqf_negate_alt_rows<const NEGATE: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        unsafe {
            let neg = _mm256_set1_ps(-1.0);
            let mut offset = if NEGATE { 0 } else { 16 };
            macro_rules! do_loop {
                (2) => {
                    let row_0_8 = _mm256_loadu_ps(input.as_ptr().cast::<f32>().add(offset));
                    let row_8_16 = _mm256_loadu_ps(input.as_ptr().cast::<f32>().add(offset + 8));
                    let negated_0_8 = _mm256_mul_ps(row_0_8, neg);
                    let negated_8_16 = _mm256_mul_ps(row_8_16, neg);
                    _mm256_storeu_ps(input.as_mut_ptr().cast::<f32>().add(offset), negated_0_8);
                    _mm256_storeu_ps(
                        input.as_mut_ptr().cast::<f32>().add(offset + 8),
                        negated_8_16,
                    );
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
    }

    fn pdqf_negate_alt_cols<const NEGATE: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        unsafe {
            let coefficients = if NEGATE {
                _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
            } else {
                _mm256_setr_ps(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
            };
            let mut offset = 0;
            macro_rules! do_loop {
                (1) => {
                    let row_0_8 = _mm256_loadu_ps(input.as_ptr().cast::<f32>().add(offset));
                    let negated_0_8 = _mm256_mul_ps(row_0_8, coefficients);
                    _mm256_storeu_ps(input.as_mut_ptr().cast::<f32>().add(offset), negated_0_8);
                };
                (2) => {
                    do_loop!(1);
                    offset += 16;
                    do_loop!(1);
                };
                (4) => {
                    do_loop!(2);
                    offset += 16;
                    do_loop!(2);
                };
                (8) => {
                    do_loop!(4);
                    offset += 16;
                    do_loop!(4);
                };
                (16) => {
                    do_loop!(8);
                    offset += 16;
                    do_loop!(8);
                };
            }

            debug_assert!(offset <= 15 * 16, "final offset may be out of bounds");

            do_loop!(16);
        }
    }

    fn pdqf_negate_off_diagonals(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        unsafe {
            let mul_for_even_rows = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
            let mul_for_odd_rows = _mm256_mul_ps(_mm256_set1_ps(-1.0), mul_for_even_rows);
            let mut offset = 0;
            macro_rules! do_loop {
                (2) => {
                    let row_0_0_8 = _mm256_loadu_ps(input.as_ptr().cast::<f32>().add(offset));
                    let row_0_8_16 = _mm256_loadu_ps(input.as_ptr().cast::<f32>().add(offset + 8));
                    let row_1_0_8 = _mm256_loadu_ps(input.as_ptr().cast::<f32>().add(offset + 16));
                    let row_1_8_16 = _mm256_loadu_ps(input.as_ptr().cast::<f32>().add(offset + 24));
                    let negated_0_0_8 = _mm256_mul_ps(row_0_0_8, mul_for_even_rows);
                    let negated_0_8_16 = _mm256_mul_ps(row_0_8_16, mul_for_even_rows);
                    let negated_1_0_8 = _mm256_mul_ps(row_1_0_8, mul_for_odd_rows);
                    let negated_1_8_16 = _mm256_mul_ps(row_1_8_16, mul_for_odd_rows);
                    _mm256_storeu_ps(input.as_mut_ptr().cast::<f32>().add(offset), negated_0_0_8);
                    _mm256_storeu_ps(
                        input.as_mut_ptr().cast::<f32>().add(offset + 8),
                        negated_0_8_16,
                    );
                    _mm256_storeu_ps(
                        input.as_mut_ptr().cast::<f32>().add(offset + 16),
                        negated_1_0_8,
                    );
                    _mm256_storeu_ps(
                        input.as_mut_ptr().cast::<f32>().add(offset + 24),
                        negated_1_8_16,
                    );
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

            debug_assert!(offset <= 14 * 16, "final offset may be out of bounds");

            do_loop!(16);
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
        let mut out_buffer = Align32([0.0; 8]);
        // crate::testing::dump_image("step_by_step/quantize/avx2/input.ppm", input);
        unsafe {
            let shift1 = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
            let shift2 = _mm256_set_epi32(1, 0, 7, 6, 5, 4, 3, 2);
            let shift4 = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);

            macro_rules! horizontal {
                (max($a:expr)) => {{
                    let pairwise = _mm256_max_ps($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 =
                        _mm256_max_ps(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = _mm256_max_ps(every4, _mm256_permutevar8x32_ps(every4, shift4));
                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
                (min($a:expr)) => {{
                    let pairwise = _mm256_min_ps($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 =
                        _mm256_min_ps(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = _mm256_min_ps(every4, _mm256_permutevar8x32_ps(every4, shift4));

                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
                (sum($a:expr)) => {{
                    let pairwise = _mm256_add_ps($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 =
                        _mm256_add_ps(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = _mm256_add_ps(every4, _mm256_permutevar8x32_ps(every4, shift4));

                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
            }

            let mut max_v = _mm256_set1_ps(f32::MIN);
            let mut min_v = _mm256_set1_ps(f32::MAX);
            for i in (0..(16 * 16)).step_by(8) {
                let row = _mm256_loadu_ps(input.as_ptr().cast::<f32>().add(i));
                min_v = _mm256_min_ps(min_v, row);
                max_v = _mm256_max_ps(max_v, row);
            }
            let mut max = horizontal!(max(max_v));
            let mut min = horizontal!(min(min_v));
            let half_point = 16 * 16 / 2;

            let mut guess = (min + max) * 0.5;
            let mut num_over = 0; // how many elements are beyond the search max?
            let mut num_under = 0; // how many elements are below the search min?

            const MAX_ITER: usize = 32;
            #[cfg_attr(not(debug_assertions), allow(unused))]
            let mut converged = false;

            // Binary search with SIMD with inter-searchspace statistics for reducing oscillations
            //
            // The reason we need to keep this many inter-searchspace comparisons and averages is the issue that
            // classic binary search expects a "perfect" answer, and it will require exhaustively narrowing down to an exact
            // floating point number and unable to prove that the current guess already is the perfect thresholding.
            //
            // This provides a more informed new guess that always set the next guess to be the mean of the new search space
            // and thus either guarantee elimination of at least one element,
            // or if the input is extreme, a new search space that is extremely small (floating point error territory)
            for _iter in 0..MAX_ITER {
                debug_assert!(guess.is_finite(), "guess is NaN");
                let guess_v = _mm256_set1_ps(guess);
                let mut resid_sum_lt_v = _mm256_setzero_ps();
                let mut resid_sum_gt_v = _mm256_setzero_ps();
                let min_v = _mm256_set1_ps(min);
                let max_v = _mm256_set1_ps(max);
                let mut gt_count = 0;
                let mut lt_count = 0;

                let mut row_ptr = input.as_ptr().cast::<f32>();
                let mut output_ptr = output.flatten().as_mut_ptr().add(32 - 1);
                macro_rules! do_loop {
                    (1) => {
                        let row = _mm256_loadu_ps(row_ptr);
                        let cmp_gt = _mm256_cmp_ps(row, guess_v, _CMP_GT_OQ);
                        *output_ptr = _mm256_movemask_ps(cmp_gt) as u8;
                        let cmp_lt = _mm256_cmp_ps(row, guess_v, _CMP_LT_OQ);
                        let cmp_max_lt = _mm256_cmp_ps(row, max_v, _CMP_LE_OQ);
                        let cmp_min_gt = _mm256_cmp_ps(row, min_v, _CMP_GE_OQ);
                        let mask_resid_gt = _mm256_and_ps(cmp_gt, cmp_max_lt);
                        let mask_resid_lt = _mm256_and_ps(cmp_lt, cmp_min_gt);
                        gt_count += _mm256_movemask_ps(mask_resid_gt).count_ones();
                        lt_count += _mm256_movemask_ps(mask_resid_lt).count_ones();

                        resid_sum_gt_v =
                            _mm256_add_ps(resid_sum_gt_v, _mm256_and_ps(mask_resid_gt, row));
                        resid_sum_lt_v =
                            _mm256_add_ps(resid_sum_lt_v, _mm256_and_ps(mask_resid_lt, row));
                    };
                    (2) => {
                        do_loop!(1);
                        row_ptr = row_ptr.add(8);
                        output_ptr = output_ptr.sub(1);
                        do_loop!(1);
                    };
                    (4) => {
                        do_loop!(2);
                        row_ptr = row_ptr.add(8);
                        output_ptr = output_ptr.sub(1);
                        do_loop!(2);
                    };
                    (8) => {
                        do_loop!(4);
                        row_ptr = row_ptr.add(8);
                        output_ptr = output_ptr.sub(1);
                        do_loop!(4);
                    };
                    (16) => {
                        do_loop!(8);
                        row_ptr = row_ptr.add(8);
                        output_ptr = output_ptr.sub(1);
                        do_loop!(8);
                    };
                    (32) => {
                        do_loop!(16);
                        row_ptr = row_ptr.add(8);
                        output_ptr = output_ptr.sub(1);
                        do_loop!(16);
                    };
                }

                do_loop!(32);

                /*
                crate::testing::dump_thresholding_diagnostic(
                    &format!("step_by_step/quantize/avx2/iter_{_iter}.ppm"),
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

                if gt_count + num_over > half_point {
                    num_under += lt_count;
                    min = guess;
                    debug_assert!(
                        gt_count != 0,
                        "gt_count is 0 somehow when gt_count + num_over > half_point"
                    );
                    guess = horizontal!(sum(resid_sum_gt_v)) / gt_count as f32;
                } else if lt_count + num_under > half_point {
                    num_over += gt_count;
                    max = guess;
                    debug_assert!(
                        lt_count != 0,
                        "lt_count is 0 somehow when lt_count + num_under > half_point"
                    );
                    guess = horizontal!(sum(resid_sum_lt_v)) / lt_count as f32;
                } else {
                    converged = true;
                    break;
                }
            }
            debug_assert!(converged, "quantization did not converge");

            *threshold = guess;
        } // unsafe
        // crate::testing::dump_image("step_by_step/quantize/avx2/output.ppm", output);
    }

    // sum of gradients is a pretty auto-vectorizable operation (and benchmark shows the same performance), so we will let it do what it thinks is best for the target architecture

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        _tmp_row_buffer: &mut GenericArray<f32, Self::Buffer1WidthX>,
        output: &mut GenericArray<GenericArray<f32, U16>, U16>,
    ) {
        Avx2F32Kernel::dct2d_impl(&DCT_MATRIX_RMAJOR, buffer, _tmp_row_buffer, output);
    }
}

/// Compute kernel using AVX512 intrinstics ported from AVX2 implementation.
///
/// Benchmark shows it is only slightly faster (~20%) than AVX2 for DCT2D and marginal overall and requires a less common CPU flag+nightly Rust compiler thus feature gated.
#[cfg(feature = "avx512")]
#[derive(Clone, Copy, Default)]
pub struct Avx512F32Kernel;

#[cfg(feature = "avx512")]
impl Kernel for Avx512F32Kernel {
    type Buffer1WidthX = U128;
    type Buffer1LengthY = U128;
    type InternalFloat = f32;
    type InputDimension = U512;
    type OutputDimension = U16;
    type RequiredHardwareFeature = RequireCompilerTimeHardwareFeature<
        CpuIdAvx512f,
        <Avx2F32Kernel as Kernel>::RequiredHardwareFeature,
    >;
    type Ident = &'static str;

    fn ident(&self) -> &'static str {
        "avx512_f32"
    }

    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input / (QUALITY_ADJUST_DIVISOR as f32);

        scaled.min(1.0)
    }

    fn cvt_rgba8_to_luma8f<const R_COEFF: u32, const G_COEFF: u32, const B_COEFF: u32>(
        &mut self,
        input: &GenericArray<GenericArray<u8, generic_array::typenum::U4>, Self::InputDimension>,
        output: &mut GenericArray<f32, Self::InputDimension>,
    ) {
        cvt_rgba8_to_luma8f_avx2::<R_COEFF, G_COEFF, B_COEFF>(input, output);
    }

    fn pdqf_negate_off_diagonals(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        unsafe {
            let mul_for_even_rows = _mm512_setr_ps(
                1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                -1.0,
            );
            let mul_for_odd_rows = _mm512_mul_ps(_mm512_set1_ps(-1.0), mul_for_even_rows);
            let mut offset = 0;
            macro_rules! do_loop {
                (2) => {
                    let row_0_0_16 = _mm512_loadu_ps(input.as_ptr().cast::<f32>().add(offset));
                    let negated_0_0_16 = _mm512_mul_ps(row_0_0_16, mul_for_even_rows);
                    let row_1_0_16 = _mm512_loadu_ps(input.as_ptr().cast::<f32>().add(offset + 16));
                    let negated_1_0_16 = _mm512_mul_ps(row_1_0_16, mul_for_odd_rows);
                    _mm512_storeu_ps(input.as_mut_ptr().cast::<f32>().add(offset), negated_0_0_16);
                    _mm512_storeu_ps(
                        input.as_mut_ptr().cast::<f32>().add(offset + 16),
                        negated_1_0_16,
                    );
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

            debug_assert!(offset <= 14 * 16, "final offset may be out of bounds");

            do_loop!(16);
        }
    }

    fn pdqf_negate_alt_rows<const NEGATE: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        unsafe {
            let neg = _mm512_set1_ps(-1.0);
            let mut offset = if NEGATE { 0 } else { 16 };
            macro_rules! do_loop {
                (2) => {
                    let row_0_0_16 = _mm512_loadu_ps(input.as_ptr().cast::<f32>().add(offset));
                    let negated_0_0_16 = _mm512_mul_ps(row_0_0_16, neg);
                    _mm512_storeu_ps(input.as_mut_ptr().cast::<f32>().add(offset), negated_0_0_16);
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
    }

    fn pdqf_negate_alt_cols<const NEGATE: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        unsafe {
            let coefficients = if NEGATE {
                _mm512_setr_ps(
                    1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
                    1.0, -1.0,
                )
            } else {
                _mm512_setr_ps(
                    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                    -1.0, 1.0,
                )
            };

            let mut offset = 0;
            macro_rules! do_loop {
                (1) => {
                    let row_0_0_16 = _mm512_loadu_ps(input.as_ptr().cast::<f32>().add(offset));
                    let negated_0_0_16 = _mm512_mul_ps(row_0_0_16, coefficients);
                    _mm512_storeu_ps(input.as_mut_ptr().cast::<f32>().add(offset), negated_0_0_16);
                };
                (2) => {
                    do_loop!(1);
                    offset += 16;
                    do_loop!(1);
                };
                (4) => {
                    do_loop!(2);
                    offset += 16;
                    do_loop!(2);
                };
                (8) => {
                    do_loop!(4);
                    offset += 16;
                    do_loop!(4);
                };
                (16) => {
                    do_loop!(8);
                    offset += 16;
                    do_loop!(8);
                };
            }

            debug_assert!(offset <= 15 * 16, "final offset may be out of bounds");

            do_loop!(16);
        }
    }

    fn jarosz_compress(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, U512>, U512>,
        output: &mut GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
    ) {
        // this part requires serious unrolling to overcome suboptimal memory access patterns
        // so we will not vectorize it further
        unsafe {
            jarosz_compress_avx2(buffer, output);
        }
    }

    fn quantize(
        &mut self,
        input: &GenericArray<GenericArray<f32, U16>, U16>,
        threshold: &mut Self::InternalFloat,
        output: &mut GenericArray<GenericArray<u8, U2>, U16>,
    ) {
        unsafe {
            let mut max_v = _mm512_set1_ps(f32::MIN);
            let mut min_v = _mm512_set1_ps(f32::MAX);
            for i in (0..(16 * 16)).step_by(16) {
                let row = _mm512_loadu_ps(input.as_ptr().cast::<f32>().add(i));
                min_v = _mm512_min_ps(min_v, row);
                max_v = _mm512_max_ps(max_v, row);
            }
            let mut max = _mm512_reduce_max_ps(max_v);
            let mut min = _mm512_reduce_min_ps(min_v);
            let half_point = 16 * 16 / 2;

            let mut guess = (min + max) * 0.5;
            let mut num_over = 0; // how many elements are beyond the search max?
            let mut num_under = 0; // how many elements are below the search min?

            // Binary search with SIMD with inter-searchspace statistics for reducing oscillations
            //
            // The reason we need to keep this many inter-searchspace comparisons and averages is the issue that
            // classic binary search expects a "perfect" answer, and it will require exhaustively narrowing down to an exact
            // floating point number and unable to prove that the current guess already is the perfect thresholding.
            //
            // This provides a more informed new guess that always set the next guess to be the mean of the new search space
            // and thus either guarantee elimination of at least one element,
            // or if the input is extreme, a new search space that is extremely small (floating point error territory)
            const MAX_ITER: usize = 32;
            #[cfg_attr(not(debug_assertions), allow(unused))]
            let mut converged = false;
            for _iter in 0..MAX_ITER {
                assert!(guess.is_finite(), "guess is NaN");
                let guess_v = _mm512_set1_ps(guess);
                let mut resid_sum_lt_v = _mm512_setzero_ps();
                let mut resid_sum_gt_v = _mm512_setzero_ps();
                let min_v = _mm512_set1_ps(min);
                let max_v = _mm512_set1_ps(max);
                let mut gt_count = 0;
                let mut lt_count = 0;

                let mut row_ptr = input.as_ptr().cast::<f32>();
                let mut output_ptr = output.flatten().as_mut_ptr().add(32 - 1);
                macro_rules! do_loop {
                    (1) => {
                        let row = _mm512_loadu_ps(row_ptr);
                        let cmp_gt = _mm512_cmp_ps_mask(row, guess_v, _CMP_GT_OQ);
                        *output_ptr = (cmp_gt & 0xff) as u8;
                        output_ptr = output_ptr.sub(1);
                        *output_ptr = ((cmp_gt >> 8) & 0xff) as u8;
                        let cmp_lt = _mm512_cmp_ps_mask(row, guess_v, _CMP_LT_OQ);
                        let cmp_max_lt = _mm512_cmp_ps_mask(row, max_v, _CMP_LE_OQ);
                        let cmp_min_gt = _mm512_cmp_ps_mask(row, min_v, _CMP_GE_OQ);
                        gt_count += (cmp_gt & cmp_max_lt).count_ones();
                        lt_count += (cmp_lt & cmp_min_gt).count_ones();

                        resid_sum_gt_v = _mm512_mask_add_ps(
                            resid_sum_gt_v,
                            cmp_gt & cmp_max_lt,
                            resid_sum_gt_v,
                            row,
                        );
                        resid_sum_lt_v = _mm512_mask_add_ps(
                            resid_sum_lt_v,
                            cmp_lt & cmp_min_gt,
                            resid_sum_lt_v,
                            row,
                        );
                    };
                    (2) => {
                        do_loop!(1);
                        row_ptr = row_ptr.add(16);
                        output_ptr = output_ptr.sub(1);
                        do_loop!(1);
                    };
                    (4) => {
                        do_loop!(2);
                        row_ptr = row_ptr.add(16);
                        output_ptr = output_ptr.sub(1);
                        do_loop!(2);
                    };
                    (8) => {
                        do_loop!(4);
                        row_ptr = row_ptr.add(16);
                        output_ptr = output_ptr.sub(1);
                        do_loop!(4);
                    };
                    (16) => {
                        do_loop!(8);
                        row_ptr = row_ptr.add(16);
                        output_ptr = output_ptr.sub(1);
                        do_loop!(8);
                    };
                }

                do_loop!(16);

                if gt_count + num_over > half_point {
                    num_under += lt_count;
                    min = guess;
                    debug_assert!(
                        gt_count != 0,
                        "gt_count is 0 somehow when gt_count + num_over > half_point"
                    );
                    guess = _mm512_reduce_add_ps(resid_sum_gt_v) / gt_count as f32;
                } else if lt_count + num_under > half_point {
                    num_over += gt_count;
                    max = guess;
                    debug_assert!(
                        lt_count != 0,
                        "lt_count is 0 somehow when lt_count + num_under > half_point"
                    );
                    guess = _mm512_reduce_add_ps(resid_sum_lt_v) / lt_count as f32;
                } else {
                    converged = true;
                    break;
                }
            }
            debug_assert!(converged, "quantization did not converge");

            *threshold = guess;
        } // unsafe
    }

    fn sum_of_gradients(&mut self, input: &GenericArray<GenericArray<f32, U16>, U16>) -> f32 {
        unsafe {
            let mut sum_v = _mm512_setzero_ps();
            let mut cur_row_0_16 = _mm512_loadu_ps(input.as_ptr().cast::<f32>());
            for i in 1..16 {
                let next_row_0_16 = _mm512_loadu_ps(input.as_ptr().cast::<f32>().add(i * 16));

                let diff_0 = _mm512_sub_ps(next_row_0_16, cur_row_0_16);
                let diff_0_abs = _mm512_abs_ps(diff_0);
                sum_v = _mm512_add_ps(sum_v, diff_0_abs);

                cur_row_0_16 = next_row_0_16;
            }

            let mut input_ptr = input.as_ptr().cast::<f32>();

            macro_rules! do_loop {
                (1) => {
                    let shiftr1 =
                        _mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0);
                    let row0_0 = _mm512_loadu_ps(input_ptr.cast::<f32>());
                    let row0_0r = _mm512_permutexvar_ps(shiftr1, row0_0);
                    let diff = _mm512_sub_ps(row0_0r, row0_0);
                    let diff_abs = _mm512_abs_ps(diff);
                    sum_v = _mm512_add_ps(sum_v, diff_abs);
                };
                (2) => {
                    do_loop!(1);
                    input_ptr = input_ptr.add(16);
                    do_loop!(1);
                };
                (4) => {
                    do_loop!(2);
                    input_ptr = input_ptr.add(16);
                    do_loop!(2);
                };
                (8) => {
                    do_loop!(4);
                    input_ptr = input_ptr.add(16);
                    do_loop!(4);
                };
                (16) => {
                    do_loop!(8);
                    input_ptr = input_ptr.add(16);
                    do_loop!(8);
                };
            }

            do_loop!(16);

            _mm512_reduce_add_ps(sum_v)
        }
    }

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        tmp: &mut GenericArray<f32, Self::Buffer1WidthX>,
        output: &mut GenericArray<GenericArray<f32, U16>, U16>,
    ) {
        unsafe {
            for k in 0..16 {
                // vectorize j by 32x16 lanes, unroll inner loop by 16 w.r.t. DCT loading
                for j_by_16 in (0..128).step_by(16) {
                    let mut sumks = [_mm512_setzero_ps(); 2];

                    let mut sum_target = true;

                    for k2_base in (0..128).step_by(16) {
                        let mut k2 = k2_base;

                        macro_rules! do_loop {
                            (1) => {
                                let buf_row = _mm512_loadu_ps(&buffer[k2][j_by_16]);
                                let dct = _mm512_set1_ps(
                                    *DCT_MATRIX_RMAJOR
                                        .as_ptr()
                                        .add(k * super::DctMatrixNumCols::USIZE + k2),
                                );
                                sumks[sum_target as usize] =
                                    _mm512_fmadd_ps(buf_row, dct, sumks[sum_target as usize]);
                            };
                            (2) => {
                                do_loop!(1);
                                sum_target = !sum_target;
                                k2 += 1;
                                do_loop!(1);
                            };
                            (4) => {
                                do_loop!(2);
                                k2 += 1;
                                do_loop!(2);
                            };
                            (8) => {
                                do_loop!(4);
                                k2 += 1;
                                do_loop!(4);
                            };
                            (16) => {
                                do_loop!(8);
                                k2 += 1;
                                do_loop!(8);
                            };
                        }

                        do_loop!(16);
                    }
                    let sumk0 = sumks[0];
                    let sumk1 = sumks[1];

                    let sumk0_nan_mask = _mm512_cmp_ps_mask(sumk0, _mm512_setzero_ps(), _CMP_ORD_S);
                    let sumk1_nan_mask = _mm512_cmp_ps_mask(sumk1, _mm512_setzero_ps(), _CMP_ORD_S);

                    let sumk0_nan_zeroed =
                        _mm512_mask_blend_ps(sumk0_nan_mask, _mm512_setzero_ps(), sumk0);
                    let sumk1_nan_zeroed =
                        _mm512_mask_blend_ps(sumk1_nan_mask, _mm512_setzero_ps(), sumk1);

                    let sumk = _mm512_add_ps(sumk0_nan_zeroed, sumk1_nan_zeroed);

                    _mm512_storeu_ps(tmp.as_mut_ptr().add(j_by_16), sumk);
                }

                for j in 0..16 {
                    let mut tmp_ptr = tmp.as_ptr();
                    let mut sumkhs = [_mm512_setzero_ps(); 2];
                    let mut sum_target = true;

                    let mut dct_ptr = DCT_MATRIX_RMAJOR
                        .as_ptr()
                        .add(j * super::DctMatrixNumCols::USIZE);

                    macro_rules! do_loop {
                        (16) => {
                            sumkhs[sum_target as usize] = _mm512_fmadd_ps(
                                _mm512_loadu_ps(tmp_ptr),
                                _mm512_loadu_ps(dct_ptr),
                                sumkhs[sum_target as usize],
                            );
                        };
                        (32) => {
                            do_loop!(16);
                            tmp_ptr = tmp_ptr.add(16);
                            dct_ptr = dct_ptr.add(16);
                            sum_target = !sum_target;
                            do_loop!(16);
                        };
                        (64) => {
                            do_loop!(32);
                            tmp_ptr = tmp_ptr.add(16);
                            dct_ptr = dct_ptr.add(16);
                            do_loop!(32);
                        };
                        (128) => {
                            do_loop!(64);
                            tmp_ptr = tmp_ptr.add(16);
                            dct_ptr = dct_ptr.add(16);
                            do_loop!(64);
                        };
                    }

                    do_loop!(128);

                    output[k][j] =
                        _mm512_reduce_add_ps(sumkhs[0]) + _mm512_reduce_add_ps(sumkhs[1]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use generic_array::typenum::U4;

    use crate::{alignment::Align32, kernel::constants};

    use super::*;

    #[test]
    fn test_avx2_shift_elements_left() {
        // test the rewinding paradigm
        unsafe {
            // we have rewound the buffer pointer by 1 element, so 0.0 should be get rid of, NAN is out of bounds
            let fake_buffer = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, f32::NAN];
            let mut buffer = _mm256_loadu_ps(fake_buffer.as_ptr());
            let shiftr1 = _mm256_set_epi32(7, 7, 6, 5, 4, 3, 2, 1);
            buffer = _mm256_permutevar8x32_ps(buffer, shiftr1);
            let mut read = [0.0; 8];
            _mm256_storeu_ps(read.as_mut_ptr(), buffer);
            assert_eq!(read, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0]);
        }
    }

    #[test]
    fn test_avx2_horizontal_max_min_sum_ps() {
        let mut out_buffer = Align32([0.0; 8]);
        unsafe {
            let shift1 = _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1);
            let shift2 = _mm256_set_epi32(1, 0, 7, 6, 5, 4, 3, 2);
            let shift4 = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);

            macro_rules! horizontal {
                (max($a:expr)) => {{
                    let pairwise = _mm256_max_ps($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 =
                        _mm256_max_ps(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = _mm256_max_ps(every4, _mm256_permutevar8x32_ps(every4, shift4));
                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
                (min($a:expr)) => {{
                    let pairwise = _mm256_min_ps($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 =
                        _mm256_min_ps(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = _mm256_min_ps(every4, _mm256_permutevar8x32_ps(every4, shift4));

                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
                (sum($a:expr)) => {{
                    let pairwise = _mm256_add_ps($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 =
                        _mm256_add_ps(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = _mm256_add_ps(every4, _mm256_permutevar8x32_ps(every4, shift4));

                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
            }

            let case_0 = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            let case_1 = _mm256_set_ps(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0);
            let case_2 = _mm256_set_ps(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0);
            let case_3 = _mm256_set_ps(4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0);
            let case_4 = _mm256_set_ps(5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0);
            let case_5 = _mm256_set_ps(6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0);
            let case_6 = _mm256_set_ps(7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
            let case_7 = _mm256_set_ps(8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);

            assert_eq!(horizontal!(max(case_0)), 8.0);
            assert_eq!(horizontal!(max(case_1)), 8.0);
            assert_eq!(horizontal!(max(case_2)), 8.0);
            assert_eq!(horizontal!(max(case_3)), 8.0);
            assert_eq!(horizontal!(max(case_4)), 8.0);
            assert_eq!(horizontal!(max(case_5)), 8.0);
            assert_eq!(horizontal!(max(case_6)), 8.0);
            assert_eq!(horizontal!(max(case_7)), 8.0);
            assert_eq!(horizontal!(min(case_0)), 1.0);
            assert_eq!(horizontal!(min(case_1)), 1.0);
            assert_eq!(horizontal!(min(case_2)), 1.0);
            assert_eq!(horizontal!(min(case_3)), 1.0);
            assert_eq!(horizontal!(min(case_4)), 1.0);
            assert_eq!(horizontal!(min(case_5)), 1.0);
            assert_eq!(horizontal!(min(case_6)), 1.0);
            assert_eq!(horizontal!(min(case_7)), 1.0);
            assert_eq!(horizontal!(sum(case_0)), 36.0);
            assert_eq!(horizontal!(sum(case_1)), 36.0);
            assert_eq!(horizontal!(sum(case_2)), 36.0);
            assert_eq!(horizontal!(sum(case_3)), 36.0);
            assert_eq!(horizontal!(sum(case_4)), 36.0);
            assert_eq!(horizontal!(sum(case_5)), 36.0);
            assert_eq!(horizontal!(sum(case_6)), 36.0);
            assert_eq!(horizontal!(sum(case_7)), 36.0);
        }
    }

    #[test]
    fn test_avx2_cvt_rgba8_to_luma8f() {
        let red = GenericArray::from_array([255, 0, 0, 128]);
        let green = GenericArray::from_array([0, 255, 0, 128]);
        let blue = GenericArray::from_array([0, 0, 255, 128]);
        let white = GenericArray::from_array([255, 255, 255, 128]);
        let mut row = GenericArray::<GenericArray<u8, U4>, U512>::default();
        row.iter_mut()
            .zip([red, green, blue, white].into_iter().cycle())
            .for_each(|(a, b)| {
                *a = b;
            });

        let mut output = GenericArray::<f32, U512>::default();

        let red_expect = constants::RGB8_TO_LUMA8_TABLE_ITU[0] as f32 * 255.0;
        let green_expect = constants::RGB8_TO_LUMA8_TABLE_ITU[1] as f32 * 255.0;
        let blue_expect = constants::RGB8_TO_LUMA8_TABLE_ITU[2] as f32 * 255.0;
        let white_expect = 255.0;

        cvt_rgba8_to_luma8f_avx2::<
            { u32::from_ne_bytes(constants::RGB8_TO_LUMA8_TABLE_ITU[0].to_ne_bytes()) },
            { u32::from_ne_bytes(constants::RGB8_TO_LUMA8_TABLE_ITU[1].to_ne_bytes()) },
            { u32::from_ne_bytes(constants::RGB8_TO_LUMA8_TABLE_ITU[2].to_ne_bytes()) },
        >(&row, &mut output);

        output
            .iter()
            .enumerate()
            .zip(
                [red_expect, green_expect, blue_expect, white_expect]
                    .into_iter()
                    .cycle(),
            )
            .for_each(|((i, a), b)| {
                let diff = (a - b).abs();
                assert!(
                    diff < 0.0001,
                    "difference is too large at position {i}: expected {} but got {}",
                    b,
                    a
                );
            });
    }
}
