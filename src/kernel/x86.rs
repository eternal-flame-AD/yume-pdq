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

use core::{
    arch::x86_64::*,
    ops::{Deref, DerefMut},
};

use crate::kernel::Kernel;

use super::{
    CONVOLUTION_LOOKUP_TABLE_INDEX, CONVOLUTION_OFFSET_512_TO_64, DCT_MATRIX_NCOLS,
    DCT_MATRIX_RMAJOR, TENT_FILTER_COLUMN_OFFSET, TENT_FILTER_EFFECTIVE_ROWS,
    TENT_FILTER_WEIGHTS_X8,
};

/// Compute kernel using hand-written AVX2 intrinsics.
///
/// Note: This would produce a slightly different numeric result than reference implementation due to:
///
/// - different rounding rules in the intrinsics by generally within 5 times epsilon (~5e-7) when tested on a random input.
/// - rounding errors when convoluting using a pre-computed lookup table.
///
/// On a test image usually we can do 11-15 bits different than reference, when the official docs consider <10 bits different as "correct".
/// This is still useful nevertheless for high-throughput matching as a matching threshold is <=31 by official docs, well within this error margin.
///
/// Benchmark shows a ~10x improvement in the core DCT2D operation throughput (and ~4x for a whole workflow due to hitting memory bandwidth limits) over the generic or reference implementation.
///
/// Note: You MUST compile with `target-feature=+avx2` (or equivalent) to use this kernel, otherwise a very slow
/// fallback will be emitted by LLVM.
pub struct Avx2F32Kernel;

#[repr(align(32))]
struct Align32<T>(T);

#[repr(align(64))]
#[cfg_attr(not(feature = "avx512"), expect(unused))]
struct Align64<T>(T);

impl<T> Deref for Align32<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align32<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Deref for Align64<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align64<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Kernel for Avx2F32Kernel {
    fn jarosz_compress(&mut self, buffer: &[f32; 512 * 512], output: &mut [f32; 64 * 64]) {
        unsafe {
            let mut out_buffer = Align32([0.0; 8]);

            let shift1 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 0);
            let shift2 = _mm256_set_epi32(2, 3, 4, 5, 6, 7, 0, 1);
            let shift4 = _mm256_set_epi32(4, 5, 6, 7, 0, 1, 2, 3);

            macro_rules! horizontal {
                ($combine:ident($a:expr)) => {{
                    let pairwise = $combine($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 = $combine(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = $combine(every4, _mm256_permutevar8x32_ps(every4, shift4));

                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
            }

            for outi in 0..64 {
                let in_i = CONVOLUTION_OFFSET_512_TO_64[outi] - TENT_FILTER_COLUMN_OFFSET;
                for outj in 0..64 {
                    let in_j = CONVOLUTION_OFFSET_512_TO_64[outj] - TENT_FILTER_COLUMN_OFFSET;
                    let mut sum = _mm256_setzero_ps();
                    let weight_table_idx = CONVOLUTION_LOOKUP_TABLE_INDEX[outi * 64 + outj];
                    let weights = &TENT_FILTER_WEIGHTS_X8[weight_table_idx as usize];
                    for di in 0..TENT_FILTER_EFFECTIVE_ROWS {
                        let buffer = _mm256_loadu_ps(buffer.as_ptr().add((in_i + di) * 512 + in_j));
                        let weights = _mm256_loadu_ps(weights.as_ptr().add(di * 8));
                        sum = _mm256_fmadd_ps(buffer, weights, sum);
                    }
                    output[outi * 64 + outj] = horizontal!(_mm256_hadd_ps(sum));
                }
            }
        }
    }

    fn quantize(&mut self, input: &[f32; 16 * 16], output: &mut [u8; 2 * 16]) {
        let mut out_buffer = Align32([0.0; 8]);
        unsafe {
            let shift1 = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 0);
            let shift2 = _mm256_set_epi32(2, 3, 4, 5, 6, 7, 0, 1);
            let shift4 = _mm256_set_epi32(4, 5, 6, 7, 0, 1, 2, 3);

            macro_rules! horizontal {
                ($combine:ident($a:expr)) => {{
                    let pairwise = $combine($a, _mm256_permutevar8x32_ps($a, shift1));
                    let every4 = $combine(pairwise, _mm256_permutevar8x32_ps(pairwise, shift2));
                    let every8 = $combine(every4, _mm256_permutevar8x32_ps(every4, shift4));

                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), every8);
                    out_buffer.0[0]
                }};
            }

            let mut max_v = _mm256_set1_ps(f32::MIN);
            let mut min_v = _mm256_set1_ps(f32::MAX);
            for i in (0..(16 * 16)).step_by(8) {
                let row = _mm256_loadu_ps(input.as_ptr().add(i));
                min_v = _mm256_min_ps(min_v, row);
                max_v = _mm256_max_ps(max_v, row);
            }
            let mut max = horizontal!(_mm256_max_ps(max_v));
            let mut min = horizontal!(_mm256_min_ps(min_v));
            let half_point = (16 * 16 / 2) as f32;

            let mut guess = (min + max) * 0.5;
            let mut num_over = 0.0f32; // how many elements are beyond the search max?
            let mut num_under = 0.0f32; // how many elements are below the search min?
            for _ in 0..8 {
                let guess_v = _mm256_set1_ps(guess);
                let mut resid_gt_count_v = _mm256_setzero_ps();
                let mut resid_lt_count_v = _mm256_setzero_ps();
                let mut resid_sum_lt_v = _mm256_setzero_ps();
                let mut resid_sum_gt_v = _mm256_setzero_ps();
                let min_v = _mm256_set1_ps(min);
                let max_v = _mm256_set1_ps(max);

                let mut row_ptr = input.as_ptr();
                let mut output_ptr = output.as_mut_ptr().add(32 - 1);
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
                        resid_gt_count_v = _mm256_add_ps(
                            resid_gt_count_v,
                            _mm256_and_ps(mask_resid_gt, _mm256_set1_ps(1.0)),
                        );
                        resid_lt_count_v = _mm256_add_ps(
                            resid_lt_count_v,
                            _mm256_and_ps(mask_resid_lt, _mm256_set1_ps(1.0)),
                        );

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

                let gt_count = horizontal!(_mm256_add_ps(resid_gt_count_v));
                let lt_count = horizontal!(_mm256_add_ps(resid_lt_count_v));

                if gt_count + num_over > half_point {
                    num_under += lt_count;
                    min = guess;
                    guess = horizontal!(_mm256_add_ps(resid_sum_gt_v)) / gt_count;
                } else if lt_count + num_under > half_point {
                    num_over += gt_count;
                    max = guess;
                    guess = horizontal!(_mm256_add_ps(resid_sum_lt_v)) / lt_count;
                } else {
                    break;
                }
            }
        }
    }

    // sum of gradients is a pretty auto-vectorizable operation (and benchmark shows the same performance), so we will let it do what it thinks is best for the target architecture

    fn dct2d(&mut self, buffer: &[f32; 64 * 64], output: &mut [f32; 16 * 16]) {
        unsafe {
            let mut out_buffer = Align32([0.0; 8]);

            for k in 0..16 {
                let mut tmp = [0.0; 64];

                // vectorize j by 32x8 lanes, unroll inner loop by 8 w.r.t. DCT loading
                for j_by_8 in (0..64).step_by(8) {
                    let mut sumk = _mm256_setzero_ps();

                    let mut k2 = 0;

                    macro_rules! do_loop {
                        (1, $dct_row:expr, $dct_idx:expr) => {
                            let buf_row = _mm256_loadu_ps(buffer.as_ptr().add(k2 * 64 + j_by_8));
                            let dct =
                                _mm256_permutevar8x32_ps($dct_row, _mm256_set1_epi32($dct_idx));
                            sumk = _mm256_fmadd_ps(buf_row, dct, sumk);
                        };
                        (2, $dct_row:expr, $dct_idx:expr) => {
                            do_loop!(1, $dct_row, $dct_idx);
                            $dct_idx += 1;
                            k2 += 1;
                            do_loop!(1, $dct_row, $dct_idx);
                        };
                        (4, $dct_row:expr, $dct_idx:expr) => {
                            do_loop!(2, $dct_row, $dct_idx);
                            $dct_idx += 1;
                            k2 += 1;
                            do_loop!(2, $dct_row, $dct_idx);
                        };
                        (8) => {
                            let dct_row = _mm256_loadu_ps(
                                DCT_MATRIX_RMAJOR.as_ptr().add(k * DCT_MATRIX_NCOLS + k2),
                            );
                            let mut dct_idx = 0;
                            do_loop!(4, dct_row, dct_idx);
                            dct_idx += 1;
                            k2 += 1;
                            do_loop!(4, dct_row, dct_idx);
                        };
                        (16) => {
                            do_loop!(8);
                            k2 += 1;
                            do_loop!(8);
                        };
                        (32) => {
                            do_loop!(16);
                            k2 += 1;
                            do_loop!(16);
                        };
                        (64) => {
                            do_loop!(32);
                            k2 += 1;
                            do_loop!(32);
                        };
                    }

                    do_loop!(64);

                    _mm256_storeu_ps(tmp.as_mut_ptr().add(j_by_8), sumk);
                }

                for j in 0..16 {
                    let mut tmp_ptr = tmp.as_ptr();
                    let zero = _mm256_setzero_ps();
                    let mut sumkh = zero;

                    let mut dct_ptr = DCT_MATRIX_RMAJOR.as_ptr().add(j * DCT_MATRIX_NCOLS);

                    macro_rules! do_loop {
                        (8) => {
                            sumkh = _mm256_fmadd_ps(
                                _mm256_loadu_ps(tmp_ptr),
                                _mm256_loadu_ps(dct_ptr),
                                sumkh,
                            );
                        };
                        (16) => {
                            do_loop!(8);
                            tmp_ptr = tmp_ptr.add(8);
                            dct_ptr = dct_ptr.add(8);
                            do_loop!(8);
                        };
                        (32) => {
                            do_loop!(16);
                            tmp_ptr = tmp_ptr.add(8);
                            dct_ptr = dct_ptr.add(8);
                            do_loop!(16);
                        };
                        (64) => {
                            do_loop!(32);
                            tmp_ptr = tmp_ptr.add(8);
                            dct_ptr = dct_ptr.add(8);
                            do_loop!(32);
                        };
                    }

                    do_loop!(64);

                    // 0, 1, 2, 3, 4, 5, 6, 7
                    sumkh = _mm256_hadd_ps(sumkh, zero);
                    // 01, 23, _, _, 45, 67, _, _
                    sumkh = _mm256_hadd_ps(sumkh, zero);
                    // 0123, _, _, _, 4567, _, _, _

                    _mm256_store_ps(out_buffer.0.as_mut_ptr(), sumkh);
                    output[k * 16 + j] = out_buffer.0[0] + out_buffer.0[4];
                }
            }
        }
    }
}

/// Compute kernel using AVX512 intrinstics ported from AVX2 implementation.
///
/// Benchmark shows it is only slightly faster (~20%) than AVX2 for DCT2D and marginal overall and requires a less common CPU flag+nightly Rust compiler thus feature gated.
#[cfg(feature = "avx512")]
pub struct Avx512F32Kernel;

#[cfg(feature = "avx512")]
impl Kernel for Avx512F32Kernel {
    fn jarosz_compress(&mut self, buffer: &[f32; 512 * 512], output: &mut [f32; 64 * 64]) {
        // this part requires serious unrolling to overcome suboptimal memory access patterns
        // so we will not vectorize it further
        Avx2F32Kernel.jarosz_compress(buffer, output);
    }

    fn quantize(&mut self, input: &[f32; 16 * 16], output: &mut [u8; 2 * 16]) {
        let mut out_buffer = Align64([0.0; 16]);
        unsafe {
            let shift1 = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);
            let shift2 = _mm512_set_epi32(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
            let shift4 = _mm512_set_epi32(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3);
            let shift8 = _mm512_set_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);

            macro_rules! horizontal {
                ($combine:ident($a:expr)) => {{
                    let pairwise = $combine($a, _mm512_permutexvar_ps(shift1, $a));
                    let every4 = $combine(pairwise, _mm512_permutexvar_ps(shift2, pairwise));
                    let every8 = $combine(every4, _mm512_permutexvar_ps(shift4, every4));
                    let every16 = $combine(every8, _mm512_permutexvar_ps(shift8, every8));

                    _mm512_store_ps(out_buffer.0.as_mut_ptr(), every16);
                    out_buffer.0[0]
                }};
            }

            let mut max_v = _mm512_set1_ps(f32::MIN);
            let mut min_v = _mm512_set1_ps(f32::MAX);
            for i in (0..(16 * 16)).step_by(16) {
                let row = _mm512_loadu_ps(input.as_ptr().add(i));
                min_v = _mm512_min_ps(min_v, row);
                max_v = _mm512_max_ps(max_v, row);
            }
            let mut max = horizontal!(_mm512_max_ps(max_v));
            let mut min = horizontal!(_mm512_min_ps(min_v));
            let half_point = (16 * 16 / 2) as f32;

            let mut guess = (min + max) * 0.5;
            let mut num_over = 0.0f32; // how many elements are beyond the search max?
            let mut num_under = 0.0f32; // how many elements are below the search min?
            for _ in 0..8 {
                let guess_v = _mm512_set1_ps(guess);
                let mut resid_gt_count_v = _mm512_setzero_ps();
                let mut resid_lt_count_v = _mm512_setzero_ps();
                let mut resid_sum_lt_v = _mm512_setzero_ps();
                let mut resid_sum_gt_v = _mm512_setzero_ps();
                let min_v = _mm512_set1_ps(min);
                let max_v = _mm512_set1_ps(max);

                let mut row_ptr = input.as_ptr();
                let mut output_ptr = output.as_mut_ptr().add(32 - 1);
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
                        resid_gt_count_v = _mm512_mask_add_ps(
                            resid_gt_count_v,
                            cmp_gt & cmp_max_lt,
                            resid_gt_count_v,
                            _mm512_set1_ps(1.0),
                        );
                        resid_lt_count_v = _mm512_mask_add_ps(
                            resid_lt_count_v,
                            cmp_lt & cmp_min_gt,
                            resid_lt_count_v,
                            _mm512_set1_ps(1.0),
                        );

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

                let gt_count = horizontal!(_mm512_add_ps(resid_gt_count_v));
                let lt_count = horizontal!(_mm512_add_ps(resid_lt_count_v));

                if gt_count + num_over > half_point {
                    num_under += lt_count;
                    min = guess;
                    guess = horizontal!(_mm512_add_ps(resid_sum_gt_v)) / gt_count;
                } else if lt_count + num_under > half_point {
                    num_over += gt_count;
                    max = guess;
                    guess = horizontal!(_mm512_add_ps(resid_sum_lt_v)) / lt_count;
                } else {
                    break;
                }
            }
        }
    }

    fn sum_of_gradients(&mut self, input: &[f32; 16 * 16]) -> f32 {
        unsafe {
            let mut out_buffer = Align64([0.0; 16]);
            let shift1 = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);
            let shift2 = _mm512_set_epi32(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
            let shift4 = _mm512_set_epi32(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3);
            let shift8 = _mm512_set_epi32(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);

            macro_rules! horizontal {
                ($combine:ident($a:expr)) => {{
                    let pairwise = $combine($a, _mm512_permutexvar_ps(shift1, $a));
                    let every4 = $combine(pairwise, _mm512_permutexvar_ps(shift2, pairwise));
                    let every8 = $combine(every4, _mm512_permutexvar_ps(shift4, every4));
                    let every16 = $combine(every8, _mm512_permutexvar_ps(shift8, every8));

                    _mm512_store_ps(out_buffer.0.as_mut_ptr(), every16);
                    out_buffer.0[0]
                }};
            }

            let mut sum_v = _mm512_setzero_ps();
            let mut cur_row_0_16 = _mm512_loadu_ps(input.as_ptr());
            for i in 1..16 {
                let next_row_0_16 = _mm512_loadu_ps(input.as_ptr().add(i * 16));

                let diff_0 = _mm512_sub_ps(next_row_0_16, cur_row_0_16);
                let diff_0_abs = _mm512_abs_ps(diff_0);
                sum_v = _mm512_add_ps(sum_v, diff_0_abs);

                cur_row_0_16 = next_row_0_16;
            }

            let mut input_ptr = input.as_ptr();

            macro_rules! do_loop {
                (1) => {
                    let shiftr1 =
                        _mm512_set_epi32(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
                    let row0_0 = _mm512_loadu_ps(input_ptr);
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

            horizontal!(_mm512_add_ps(sum_v))
        }
    }

    fn dct2d(&mut self, buffer: &[f32; 64 * 64], output: &mut [f32; 16 * 16]) {
        unsafe {
            for k in 0..16 {
                let mut tmp = [0.0; 64];

                // vectorize j by 32x16 lanes, unroll inner loop by 16 w.r.t. DCT loading
                for j_by_16 in (0..64).step_by(16) {
                    let mut sumk = _mm512_setzero_ps();

                    let mut k2 = 0;

                    macro_rules! do_loop {
                        (1, $dct_row:expr, $dct_idx:expr) => {
                            let buf_row = _mm512_loadu_ps(buffer.as_ptr().add(k2 * 64 + j_by_16));
                            let dct = _mm512_permutexvar_ps(_mm512_set1_epi32($dct_idx), $dct_row);
                            sumk = _mm512_fmadd_ps(buf_row, dct, sumk);
                        };
                        (2, $dct_row:expr, $dct_idx:expr) => {
                            do_loop!(1, $dct_row, $dct_idx);
                            $dct_idx += 1;
                            k2 += 1;
                            do_loop!(1, $dct_row, $dct_idx);
                        };
                        (4, $dct_row:expr, $dct_idx:expr) => {
                            do_loop!(2, $dct_row, $dct_idx);
                            $dct_idx += 1;
                            k2 += 1;
                            do_loop!(2, $dct_row, $dct_idx);
                        };
                        (8, $dct_row:expr, $dct_idx:expr) => {
                            do_loop!(4, $dct_row, $dct_idx);
                            $dct_idx += 1;
                            k2 += 1;
                            do_loop!(4, $dct_row, $dct_idx);
                        };
                        (16) => {
                            let dct_row = _mm512_loadu_ps(
                                DCT_MATRIX_RMAJOR.as_ptr().add(k * DCT_MATRIX_NCOLS + k2),
                            );
                            let mut dct_idx = 0;
                            do_loop!(8, dct_row, dct_idx);
                            dct_idx += 1;
                            k2 += 1;
                            do_loop!(8, dct_row, dct_idx);
                        };
                        (32) => {
                            do_loop!(16);
                            k2 += 1;
                            do_loop!(16);
                        };
                        (64) => {
                            do_loop!(32);
                            k2 += 1;
                            do_loop!(32);
                        };
                    }

                    do_loop!(64);

                    _mm512_storeu_ps(tmp.as_mut_ptr().add(j_by_16), sumk);
                }

                for j in 0..16 {
                    let mut tmp_ptr = tmp.as_ptr();
                    let zero = _mm512_setzero_ps();
                    let mut sumkh = zero;

                    let mut dct_ptr = DCT_MATRIX_RMAJOR.as_ptr().add(j * DCT_MATRIX_NCOLS);

                    macro_rules! do_loop {
                        (16) => {
                            sumkh = _mm512_fmadd_ps(
                                _mm512_loadu_ps(tmp_ptr),
                                _mm512_loadu_ps(dct_ptr),
                                sumkh,
                            );
                        };
                        (32) => {
                            do_loop!(16);
                            tmp_ptr = tmp_ptr.add(16);
                            dct_ptr = dct_ptr.add(16);
                            do_loop!(16);
                        };
                        (64) => {
                            do_loop!(32);
                            tmp_ptr = tmp_ptr.add(16);
                            dct_ptr = dct_ptr.add(16);
                            do_loop!(32);
                        };
                    }

                    do_loop!(64);

                    output[k * 16 + j] = _mm512_reduce_add_ps(sumkh);
                }
            }
        }
    }
}
