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

use generic_array::{
    ArrayLength, GenericArray,
    typenum::{U16, U127},
};

use num_traits::{AsPrimitive, float::FloatCore};

/// Kernels based on x86-64 intrinsics.
#[cfg(target_arch = "x86_64")]
pub mod x86;

/// Dihedral flips.
pub mod dihedral;

include!(concat!(env!("OUT_DIR"), "/dct_matrix.rs"));
include!(concat!(env!("OUT_DIR"), "/tent_filter_weights.rs"));
include!(concat!(env!("OUT_DIR"), "/convolution_offset.rs"));

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

// reference: https://raw.githubusercontent.com/facebook/ThreatExchange/main/hashing/hashing.pdf

/// Compute kernel for doing heavy-duty transformations.
///
/// A scalar (auto-vectorized) implementation is provided in `DefaultKernel`.
pub trait Kernel: Send + Sized {
    /// The width of the first stage (compression) buffer.
    type Buffer1WidthX: ArrayLength;
    /// The length of the first stage (compression) buffer.
    type Buffer1LengthY: ArrayLength;

    /// The internal floating point type used for intermediate calculations.
    type InternalFloat: num_traits::float::FloatCore
        + num_traits::float::TotalOrder
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + num_traits::AsPrimitive<f32>
        + num_traits::bounds::Bounded
        + num_traits::NumCast
        + num_traits::Signed
        + num_traits::Zero
        + num_traits::One
        + Display
        + Debug
        + Default
        + Send
        + Sync;

    /// Apply a tent-filter average to every 8x8 sub-block of the input buffer and write the result of each sub-block to the output buffer.
    fn jarosz_compress(
        &mut self,
        _buffer: &[f32; 512 * 512],
        _output: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
    );

    /// Convert input to binary by thresholding at median
    fn quantize(
        &mut self,
        input: &GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
        output: &mut [u8; 2 * 16],
    ) {
        let median = torben_median(input);

        for (i, j) in input.iter().flatten().enumerate() {
            output[32 - 1 - i / 8] += if *j > median { 1 << (i % 8) } else { 0 };
        }
    }

    /// Compute the sum of gradients of the input buffer in both horizontal and vertical directions.
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

    /// Adjust the quality metric to be between 0 and 1.
    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        let scaled = input.as_() / (256.0f32 * 2048.0 * 64.0);

        scaled.min(1.0)
    }

    /// Apply a 2D DCT-II transformation to the input buffer write the result to the output buffer.
    fn dct2d(
        &mut self,
        _buffer: &GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
        _output: &mut GenericArray<GenericArray<Self::InternalFloat, U16>, U16>,
    );
}

/// A pure-Rust implementation of the `Kernel` trait.
pub struct DefaultKernel;

impl Kernel for DefaultKernel {
    type Buffer1WidthX = U127;
    type Buffer1LengthY = U127;
    type InternalFloat = f32;

    fn jarosz_compress(
        &mut self,
        buffer: &[f32; 512 * 512],
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
                            * buffer[(in_i + di) * 512 + (in_j + dj)];
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
                output[k][j] = sumk;
            }
        }
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

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        output: &mut GenericArray<GenericArray<f32, U16>, U16>,
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
                output[i][j] = sumk;
            }
        }
    }

    fn quantize(
        &mut self,
        input: &GenericArray<GenericArray<f32, U16>, U16>,
        output: &mut [u8; 2 * 16],
    ) {
        let median = torben_median(input);

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
        reference::decimate_float::<f32, 127, 127>(
            buffer.as_slice().try_into().unwrap(),
            512,
            512,
            unsafe { std::mem::transmute::<_, &mut [[f32; 127]; 127]>(output) },
        );
    }
}

impl Kernel for ReferenceKernel<f64> {
    type Buffer1WidthX = U127;
    type Buffer1LengthY = U127;
    type InternalFloat = f64;

    fn dct2d(
        &mut self,
        buffer: &GenericArray<GenericArray<f64, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
        output: &mut GenericArray<GenericArray<f64, U16>, U16>,
    ) {
        let mut intermediate_matrix = [[0.0; 127]; 16];
        for i in 0..16 {
            for j in 0..127 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk += DCT_MATRIX_RMAJOR_64[i * DCT_MATRIX_NCOLS + k] * buffer[k][j];
                }

                intermediate_matrix[i][j] = sumk;
            }
        }

        for i in 0..16 {
            for j in 0..16 {
                let mut sumk = 0.0;
                for k in 0..127 {
                    sumk +=
                        intermediate_matrix[i][k] * DCT_MATRIX_RMAJOR_64[j * DCT_MATRIX_NCOLS + k];
                }
                output[i][j] = sumk;
            }
        }
    }

    fn quantize(
        &mut self,
        input: &GenericArray<GenericArray<f64, U16>, U16>,
        output: &mut [u8; 2 * 16],
    ) {
        let median = torben_median(input);

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
        buffer: &[f32; 512 * 512],
        output: &mut GenericArray<GenericArray<f64, Self::Buffer1WidthX>, Self::Buffer1LengthY>,
    ) {
        #[allow(missing_docs, dead_code)]
        mod reference {
            include!("ref.rs");
        }

        let cb = || {
            let window_size = reference::compute_jarosz_filter_window_size(512, 127);
            let mut buffer = buffer.map(|s| s as f64).to_vec();
            reference::jarosz_filter_float(
                buffer.as_mut_slice().try_into().unwrap(),
                512,
                512,
                window_size,
                window_size,
                2,
            );
            reference::decimate_float::<f64, 127, 127>(
                buffer.as_slice().try_into().unwrap(),
                512,
                512,
                unsafe { std::mem::transmute::<_, &mut [[f64; 127]; 127]>(output) },
            );
        };

        #[cfg(any(feature = "reference-stacker", test))]
        stacker::grow(core::mem::size_of::<f64>() * 512 * 512 * 8, cb);

        #[cfg(not(any(feature = "reference-stacker", test)))]
        cb();
    }
}

#[cfg(test)]
mod tests {
    use num_traits::NumCast;
    use rand::Rng;

    use super::*;

    extern crate alloc;

    fn test_gradient_impl<K: Kernel>(kernel: &mut K)
    where
        ReferenceKernel<<K as Kernel>::InternalFloat>:
            Kernel<InternalFloat = <K as Kernel>::InternalFloat>,
    {
        let mut rng = rand::rng();
        let mut input: GenericArray<GenericArray<<K as Kernel>::InternalFloat, U16>, U16> =
            GenericArray::default();
        input.iter_mut().for_each(|row| {
            row.iter_mut()
                .for_each(|val| *val = NumCast::from(rng.random_range(0.0..1.0)).unwrap());
        });

        let gradient_ref = ReferenceKernel::<K::InternalFloat>::default().sum_of_gradients(&input);
        let gradient = kernel.sum_of_gradients(&input);

        let diff = (gradient - NumCast::from(gradient_ref).unwrap()).abs();
        let diff_ref = diff / NumCast::from(gradient_ref).unwrap();

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
                input[i][j] = input_ref[i][j];
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
                let diff = (*a - NumCast::from(*b).unwrap()).abs();
                assert!(diff < eps, "difference: {:?}", diff);
            });
    }

    #[test]
    fn test_dct64_impl_base() {
        let mut kernel = DefaultKernel;
        test_dct64_impl(&mut kernel, NumCast::from(f32::EPSILON * 2.00).unwrap());
    }

    #[test]
    fn test_dct64_impl_ref() {
        let mut kernel = ReferenceKernel::<f32>::default();
        test_dct64_impl(&mut kernel, NumCast::from(5e-6).unwrap());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_dct64_impl_avx2() {
        let mut kernel = x86::Avx2F32Kernel;
        test_gradient_impl(&mut kernel);
        test_dct64_impl(&mut kernel, NumCast::from(5e-6).unwrap());
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_dct64_impl_avx512() {
        let mut kernel = x86::Avx512F32Kernel;
        test_gradient_impl(&mut kernel);
        test_dct64_impl(&mut kernel, NumCast::from(5e-6).unwrap());
    }
}
