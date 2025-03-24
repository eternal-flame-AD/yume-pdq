//! An optimized implementation of the PDQ hash function.
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

#![cfg_attr(all(not(test), not(feature = "std")), no_std)]
#![cfg_attr(feature = "avx512", feature(stdarch_x86_avx512))]
#![warn(missing_docs)]
use kernel::Kernel;

/// PDQ compression kernel
pub mod kernel;

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel.
pub fn hash<K: Kernel>(
    kernel: &mut K,
    input: &[f32; 512 * 512],
    output: &mut [u8; 2 * 16],
    buf1: &mut [f32; 64 * 64],
    buf2: &mut [f32; 16 * 16],
) -> f32 {
    kernel.jarosz_compress(input, buf1);
    kernel.dct2d(buf1, buf2);
    let gradient = kernel.sum_of_gradients(buf2);
    let quality = K::adjust_quality(gradient);
    kernel.quantize(buf2, output);
    quality
}

#[cfg(test)]
mod tests {
    use pdqhash::image::{self, DynamicImage, ImageBuffer, Luma};

    use super::*;

    fn test_hash_impl<K: Kernel>(kernel: &mut K) {
        let input = image::load_from_memory(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test-data/music.png"
        )))
        .unwrap();

        let input = input.resize_exact(512, 512, image::imageops::FilterType::Triangle);

        let input_image = input.to_luma8();

        let input_image_f = ImageBuffer::<Luma<f32>, _>::from_fn(512, 512, |i, j| {
            Luma([input_image.get_pixel(i, j)[0] as f32])
        });

        let output_expected =
            pdqhash::generate_pdq_full_size(&DynamicImage::ImageLuma8(input_image));

        let mut output = [0; 2 * 16];
        let mut buf1 = [0.0; 64 * 64];
        let mut buf2 = [0.0; 16 * 16];
        let quality = hash(
            kernel,
            input_image_f.as_raw().as_slice().try_into().unwrap(),
            &mut output,
            &mut buf1,
            &mut buf2,
        );
        let mut distance = 0;
        for (a, b) in output.iter().zip(output_expected.0.iter()) {
            let bits_diff = (a ^ b).count_ones();
            distance += bits_diff;
        }

        let quality_expected = output_expected.1;
        let quality_diff = (quality - quality_expected).abs();
        let quality_diff_ref = quality_diff / quality_expected;

        assert!(quality_diff_ref < 0.05);

        println!(
            "Distance: {}/{} (Q={})",
            distance,
            16 * 16,
            output_expected.1
        );
        assert!(distance < 15);
    }

    #[test]
    fn test_hash_impl_base() {
        let mut kernel = kernel::DefaultKernel;
        test_hash_impl(&mut kernel);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_impl_avx2() {
        let mut kernel = kernel::x86::Avx2F32Kernel;
        test_hash_impl(&mut kernel);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    fn test_hash_impl_avx512() {
        let mut kernel = kernel::x86::Avx512F32Kernel;
        test_hash_impl(&mut kernel);
    }
}
