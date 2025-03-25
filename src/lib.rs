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
pub use generic_array::GenericArray;
use kernel::Kernel;

use generic_array::typenum::U16;

/// PDQ compression kernel
pub mod kernel;

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel.
pub fn hash<K: Kernel>(
    kernel: &mut K,
    input: &[f32; 512 * 512],
    output: &mut [u8; 2 * 16],
    buf1: &mut GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>,
    // the floating point version of the input image
    pdqf: &mut GenericArray<GenericArray<K::InternalFloat, U16>, U16>,
) -> f32 {
    kernel.jarosz_compress(input, buf1);
    kernel.dct2d(buf1, pdqf);
    let gradient = kernel.sum_of_gradients(pdqf);
    let quality = K::adjust_quality(gradient);
    kernel.quantize(pdqf, output);
    quality
}

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel without quantization.
///
/// This is called PDQF in the original paper.
pub fn hash_float<K: Kernel>(
    kernel: &mut K,
    input: &[f32; 512 * 512],
    output: &mut GenericArray<GenericArray<K::InternalFloat, U16>, U16>,
    buf1: &mut GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>,
) -> f32 {
    kernel.jarosz_compress(input, buf1);
    kernel.dct2d(buf1, output);
    let gradient = kernel.sum_of_gradients(output);
    let quality = K::adjust_quality(gradient);
    quality
}

#[cfg(test)]
mod tests {
    use num_traits::FromPrimitive;
    use pdqhash::image::{self, DynamicImage, ImageBuffer, Luma};

    use crate::kernel::ReferenceKernel;

    use super::*;

    const TEST_IMAGE_AAA_ORIG: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-data/aaa-orig.jpg"
    ));
    const TEST_IMAGE_ANIME: &[u8] =
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/test-data/anime.png"));
    const TEST_IMAGE_MUSIC: &[u8] =
        include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/test-data/music.png"));
    const TEST_IMAGE_NEOFETCH: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/test-data/neofetch.png"
    ));

    fn test_hash_impl_lib<K: Kernel>(kernel: &mut K) {
        for (name, image_data) in [
            ("aaa-orig.jpg", TEST_IMAGE_AAA_ORIG),
            ("anime.png", TEST_IMAGE_ANIME),
            ("music.png", TEST_IMAGE_MUSIC),
            ("neofetch.png", TEST_IMAGE_NEOFETCH),
        ] {
            let input = image::load_from_memory(image_data).unwrap();

            let input = input.resize_exact(512, 512, image::imageops::FilterType::Triangle);

            let input_image = input.to_luma16();

            let input_image_f = ImageBuffer::<Luma<f32>, _>::from_fn(512, 512, |i, j| {
                Luma([input_image.get_pixel(i, j)[0] as f32])
            });

            let output_expected =
                pdqhash::generate_pdq_full_size(&DynamicImage::ImageLuma16(input_image));

            let mut output = [0; 2 * 16];
            let mut buf1 = Box::default();
            let mut buf2 = Box::default();
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

            println!(
                "[{} ({})] {}: Distance vs. library: {}/{} (Qin={}, Qout={})",
                std::any::type_name::<K>(),
                std::any::type_name::<K::InternalFloat>(),
                name,
                distance,
                16 * 16,
                output_expected.1,
                quality
            );
            assert!(distance <= 31);
        }
    }

    fn test_hash_impl_ref<K: Kernel>(kernel: &mut K)
    where
        ReferenceKernel<K::InternalFloat>: Kernel<InternalFloat = K::InternalFloat>,
    {
        for (name, image_data) in [
            ("aaa-orig.jpg", TEST_IMAGE_AAA_ORIG),
            ("anime.png", TEST_IMAGE_ANIME),
            ("music.png", TEST_IMAGE_MUSIC),
        ] {
            let input = image::load_from_memory(image_data).unwrap();

            let input = input.resize_exact(512, 512, image::imageops::FilterType::Triangle);

            let input_image = input
                .to_luma16()
                .into_iter()
                .map(|p| FromPrimitive::from_u16(*p).unwrap())
                .collect::<Vec<_>>();

            let mut output = [0; 2 * 16];
            let mut output_ref = [0; 2 * 16];
            let mut buf1 = Box::default();
            let mut buf1a = Box::default();
            let mut buf2 = Box::default();
            let quality = hash(
                kernel,
                input_image.as_slice().try_into().unwrap(),
                &mut output,
                &mut buf1,
                &mut buf2,
            );
            let mut ref_kernel = ReferenceKernel::<K::InternalFloat>::default();
            let quality_ref = hash(
                &mut ref_kernel,
                input_image.as_slice().try_into().unwrap(),
                &mut output_ref,
                &mut buf1a,
                &mut buf2,
            );
            let mut distance = 0;
            for (a, b) in output.iter().zip(output_ref.iter()) {
                let bits_diff = (a ^ b).count_ones();
                distance += bits_diff;
            }

            println!(
                "[{} ({})] {}: Distance vs. reference: {}/{} (Q={})",
                std::any::type_name::<K>(),
                std::any::type_name::<K::InternalFloat>(),
                name,
                distance,
                16 * 16,
                quality_ref
            );
            assert!(distance <= 31);
        }
    }

    #[test]
    fn test_hash_impl_base() {
        let mut kernel = kernel::DefaultKernel;
        test_hash_impl_lib(&mut kernel);
        test_hash_impl_ref(&mut kernel);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_impl_avx2() {
        let mut kernel = kernel::x86::Avx2F32Kernel;
        test_hash_impl_lib(&mut kernel);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    fn test_hash_impl_avx512() {
        let mut kernel = kernel::x86::Avx512F32Kernel;
        test_hash_impl_lib(&mut kernel);
    }
}
