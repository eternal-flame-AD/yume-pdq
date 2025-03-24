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

use generic_array::{ArrayLength, GenericArray, typenum::U127};

/// Kernels based on x86-64 intrinsics.
#[cfg(target_arch = "x86_64")]
pub mod x86;

include!(concat!(env!("OUT_DIR"), "/dct_matrix.rs"));
include!(concat!(env!("OUT_DIR"), "/tent_filter_weights.rs"));
include!(concat!(env!("OUT_DIR"), "/convolution_offset.rs"));

/// Marker trait to indicate that this kernel produces precise results.
pub trait PreciseKernel: Kernel {}

// Copied verbatim from [darwinium-com/pdqhash](https://github.com/darwinium-com/pdqhash/blob/main/src/lib.rs).
pub(crate) fn torben_median(m: &[f32]) -> f32 {
    let mut min = m.iter().cloned().reduce(f32::min).unwrap();
    let mut max = m.iter().cloned().reduce(f32::max).unwrap();

    let half = (m.len() + 1) / 2;
    loop {
        let guess = (min + max) / 2.0;
        let mut less = 0;
        let mut greater = 0;
        let mut equal = 0;
        let mut maxltguess = min;
        let mut mingtguess = max;
        for val in m {
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

// reference: https://raw.githubusercontent.com/facebook/ThreatExchange/main/hashing/hashing.pdf

/// Compute kernel for doing heavy-duty transformations.
///
/// A scalar (auto-vectorized) implementation is provided in `DefaultKernel`.
pub trait Kernel: Send + Sized {
    /// The width of the first stage (compression) buffer.
    type Buffer1WidthX: ArrayLength;
    /// The length of the first stage (compression) buffer.
    type Buffer1LengthY: ArrayLength;

    /// Apply a tent-filter average to every 8x8 sub-block of the input buffer and write the result of each sub-block to the output buffer.
    fn jarosz_compress(
        &mut self,
        buffer: &[f32; 512 * 512],
        output: &mut GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
    ) {
        for outi in 0..127 {
            let in_i = CONVOLUTION_OFFSET_512_TO_127[outi] - TENT_FILTER_COLUMN_OFFSET;
            for outj in 0..127 {
                let in_j = CONVOLUTION_OFFSET_512_TO_127[outj] - TENT_FILTER_COLUMN_OFFSET;
                let mut sum = 0.0;
                for di in 0..TENT_FILTER_EFFECTIVE_ROWS {
                    for dj in 0..TENT_FILTER_EFFECTIVE_COLS {
                        sum += TENT_FILTER_WEIGHTS[di * TENT_FILTER_EFFECTIVE_COLS + dj]
                            * buffer[(in_i + di) * 512 + (in_j + dj)];
                    }
                }
                output[outi][outj] = sum;
            }
        }
    }

    /// Convert input to binary by thresholding at median
    fn quantize(&mut self, input: &[f32; 16 * 16], output: &mut [u8; 2 * 16]) {
        let median = torben_median(input);

        for (i, j) in input.iter().enumerate() {
            output[32 - 1 - i / 8] += if *j > median { 1 << (i % 8) } else { 0 };
        }
    }

    /// Compute the sum of gradients of the input buffer in both horizontal and vertical directions.
    fn sum_of_gradients(&mut self, input: &[f32; 16 * 16]) -> f32 {
        let mut gradient_sum = 0.0;

        for i in 0..(16 - 1) {
            for j in 0..16 {
                let u = input[i * 16 + j];
                let v = input[(i + 1) * 16 + j];
                let d = u - v;
                gradient_sum += d.abs();
            }
        }

        for i in 0..16 {
            for j in 0..(16 - 1) {
                let u = input[i * 16 + j];
                let v = input[i * 16 + (j + 1)];
                let d = u - v;
                gradient_sum += d.abs();
            }
        }

        gradient_sum
    }

    /// Adjust the quality metric to be between 0 and 1.
    fn adjust_quality(input: f32) -> f32 {
        (input / (255.0 * 90.0)).min(1.0)
    }

    /// Apply a 2D DCT-II transformation to the input buffer write the result to the output buffer.
    fn dct2d(
        &mut self,
        _buffer: &GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        _output: &mut [f32; 16 * 16],
    );
}

/// A pure-Rust implementation of the `Kernel` trait.
pub struct DefaultKernel;

impl Kernel for DefaultKernel {
    type Buffer1WidthX = U127;
    type Buffer1LengthY = U127;

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        output: &mut [f32; 16 * 16],
    ) {
        for k in 0..16 {
            let mut tmp = [0.0; 127];
            for j in 0..127 {
                let mut sumk = 0.0;
                for k2 in 0..127 {
                    sumk += DCT_MATRIX_RMAJOR[k * DCT_MATRIX_NCOLS + k2] * buffer[k2][j];
                }

                tmp[j] = sumk;
            }

            for j in 0..DCT_MATRIX_NROWS {
                let mut sumk = 0.0;
                for m in 0..DCT_MATRIX_NCOLS {
                    sumk += tmp[m] * DCT_MATRIX_RMAJOR[j * DCT_MATRIX_NCOLS + m];
                }
                output[k * 16 + j] = sumk;
            }
        }
    }
}

/// A reference implementation of the `Kernel` trait, copied verbatim from the officially-endorsed implementation by [darwinium-com](https://github.com/darwinium-com/pdqhash).
#[cfg(any(feature = "std", test))]
pub struct ReferenceKernel;

#[cfg(any(feature = "std", test))]
impl PreciseKernel for ReferenceKernel {}

#[cfg(any(feature = "std", test))]
impl Kernel for ReferenceKernel {
    type Buffer1WidthX = U127;
    type Buffer1LengthY = U127;

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        output: &mut [f32; 16 * 16],
    ) {
        let mut intermediate_matrix = [[0.0; 127]; 16];
        for i in 0..16 {
            for j in 0..127 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk += DCT_MATRIX_RMAJOR[i * DCT_MATRIX_NCOLS + k] * buffer[k][j];
                }

                intermediate_matrix[i][j] = sumk;
            }
        }

        for i in 0..16 {
            for j in 0..16 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk += intermediate_matrix[i][k] * DCT_MATRIX_RMAJOR[j * DCT_MATRIX_NCOLS + k];
                }
                output[i * 16 + j] = sumk;
            }
        }
    }

    fn quantize(&mut self, input: &[f32; 16 * 16], output: &mut [u8; 2 * 16]) {
        let median = torben_median(input);

        for i in 0..32 {
            let mut byte = 0;
            for j in 0..8 {
                let val = input[i * 8 + j];
                if val > median {
                    byte |= 1 << j;
                }
            }
            output[32 - i - 1] = byte;
        }
    }

    fn jarosz_compress(
        &mut self,
        buffer: &[f32; 512 * 512],
        output: &mut GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
    ) {
        #[allow(missing_docs, dead_code)]
        mod reference {
            include!("ref.rs");
        }

        let window_size = reference::compute_jarosz_filter_window_size(512, 127);
        let mut buffer = buffer.to_vec();
        reference::jarosz_filter_float(
            buffer.as_mut_slice().try_into().unwrap(),
            512,
            512,
            window_size,
            window_size,
            2,
        );
        reference::decimate_float::<64, 64>(
            buffer.as_slice().try_into().unwrap(),
            64,
            64,
            unsafe { std::mem::transmute::<_, &mut [[f32; 64]; 64]>(output) },
        );
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    extern crate alloc;

    fn test_gradient_impl<K: Kernel>(kernel: &mut K) {
        let mut rng = rand::rng();
        let input = core::array::from_fn(|_| rng.random_range(0.0..1.0));
        let gradient_ref = ReferenceKernel.sum_of_gradients(&input);
        let gradient = kernel.sum_of_gradients(&input);

        let diff = (gradient - gradient_ref).abs();
        let diff_ref = diff / gradient_ref;

        assert!(
            diff_ref < 0.05,
            "gradient: {}, gradient_ref: {}, diff: {} %",
            gradient,
            gradient_ref,
            diff_ref * 100.0,
        );
    }

    fn test_dct64_impl<K: Kernel>(kernel: &mut K) {
        let mut rng = rand::rng();
        let mut input_ref: GenericArray<GenericArray<f32, U127>, U127> = GenericArray::default();
        input_ref.iter_mut().for_each(|row| {
            row.iter_mut()
                .for_each(|val| *val = rng.random_range(0.0..1.0));
        });
        let mut input: GenericArray<GenericArray<f32, K::Buffer1WidthX>, K::Buffer1LengthY> =
            GenericArray::default();
        for i in 0..127 {
            for j in 0..127 {
                input[i][j] = input_ref[i][j];
            }
        }

        let mut output = [0.0; 16 * 16];
        let mut output_ref = [0.0; 16 * 16];
        ReferenceKernel.dct2d(&input_ref, &mut output_ref);

        kernel.dct2d(&input, &mut output);

        output
            .iter()
            .zip(output_ref.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < f32::EPSILON * 5.0));
    }

    #[test]
    fn test_dct64_impl_base() {
        let mut kernel = DefaultKernel;
        test_dct64_impl(&mut kernel);
    }

    #[test]
    fn test_dct64_impl_ref() {
        let mut kernel = ReferenceKernel;
        test_dct64_impl(&mut kernel);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dct64_impl_avx2() {
        let mut kernel = x86::Avx2F32Kernel;
        test_gradient_impl(&mut kernel);
        test_dct64_impl(&mut kernel);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_dct64_impl_avx512() {
        let mut kernel = x86::Avx512F32Kernel;
        test_gradient_impl(&mut kernel);
        test_dct64_impl(&mut kernel);
    }
}
