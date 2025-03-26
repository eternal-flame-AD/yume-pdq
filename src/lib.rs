#![doc = include_str!("../README.md")]
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
#![warn(missing_docs, clippy::pedantic)]

pub use generic_array::GenericArray;
use kernel::{
    Kernel,
    type_traits::{DivisibleBy8, SquareOf},
};

use generic_array::{
    ArrayLength,
    typenum::{PartialDiv, U8, U16},
};

/// PDQ compression kernel
pub mod kernel;

/// PDQ hash type
pub type PDQHash<L = U16> = GenericArray<GenericArray<u8, <L as PartialDiv<U8>>::Output>, L>;

/// Unquantized PDQ hash ("PDQF" in the original paper)
pub type PDQHashF<N = f32, L = U16> = GenericArray<GenericArray<N, L>, L>;

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel.
///
/// This is a convenience wrapper function and just calls [`hash_get_threshold`] with a dummy output location.
///
/// # Safety
///
/// Some vectorized kernels may read out of bounds by at most 1 element to the right.
///
/// They do not affect the final result but if your buffer is right at the edge of a page boundary
/// you may want to use a padding struct to avoid segmentation faults.
pub fn hash<K: Kernel>(
    kernel: &mut K,
    input: &GenericArray<GenericArray<f32, K::InputDimension>, K::InputDimension>,
    output: &mut GenericArray<
        GenericArray<u8, <K::OutputDimension as DivisibleBy8>::Output>,
        K::OutputDimension,
    >,
    buf1: &mut GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>,
    // the floating point version of the input image
    pdqf: &mut PDQHashF<K::InternalFloat, K::OutputDimension>,
) -> f32
where
    <K as Kernel>::OutputDimension: DivisibleBy8,
    <K as Kernel>::InputDimension: SquareOf,
    <<K as Kernel>::InputDimension as SquareOf>::Output: ArrayLength,
    <K as Kernel>::OutputDimension: SquareOf,
    <<K as Kernel>::OutputDimension as SquareOf>::Output: ArrayLength,
{
    hash_get_threshold(kernel, input, &mut Default::default(), output, buf1, pdqf)
}

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel, obtaining the threshold value useful for [`kernel::threshold::threshold_2d_f32`].
///
/// # Safety
///
/// Some vectorized kernels may read out of bounds by at most 1 element to the right.
///
/// They do not affect the final result but if your buffer is right at the edge of a page boundary
/// you may want to use a padding struct to avoid segmentation faults.
#[inline]
pub fn hash_get_threshold<K: Kernel>(
    kernel: &mut K,
    input: &GenericArray<GenericArray<f32, K::InputDimension>, K::InputDimension>,
    threshold: &mut K::InternalFloat,
    output: &mut GenericArray<
        GenericArray<u8, <K::OutputDimension as DivisibleBy8>::Output>,
        K::OutputDimension,
    >,
    buf1: &mut GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>,
    // the floating point version of the input image
    pdqf: &mut PDQHashF<K::InternalFloat, K::OutputDimension>,
) -> f32
where
    <K as Kernel>::InputDimension: SquareOf,
    <<K as Kernel>::InputDimension as SquareOf>::Output: ArrayLength,
    <K as Kernel>::OutputDimension: SquareOf,
    <<K as Kernel>::OutputDimension as SquareOf>::Output: ArrayLength,
    <K as Kernel>::OutputDimension: DivisibleBy8,
{
    kernel.jarosz_compress(input, buf1);
    kernel.dct2d(buf1, pdqf);
    let gradient = kernel.sum_of_gradients(pdqf);
    let quality = K::adjust_quality(gradient);

    kernel.quantize(pdqf, threshold, output);
    quality
}

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel without quantization.
///
/// This is called PDQF in the original paper.
///
/// # Safety
///
/// Some vectorized kernels may read out of bounds by at most 1 element to the right.
///
/// They do not affect the final result but if your buffer is right at the edge of a page boundary
/// you may want to use a padding struct to avoid segmentation faults.
pub fn hash_float<K: Kernel>(
    kernel: &mut K,
    input: &GenericArray<GenericArray<f32, K::InputDimension>, K::InputDimension>,
    output: &mut PDQHashF<K::InternalFloat, K::OutputDimension>,
    buf1: &mut GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>,
) -> f32 {
    kernel.jarosz_compress(input, buf1);
    kernel.dct2d(buf1, output);
    let gradient = kernel.sum_of_gradients(output);

    K::adjust_quality(gradient)
}

#[cfg(test)]
mod tests {

    use generic_array::{
        sequence::Flatten,
        typenum::{U2, U512},
    };
    use num_traits::FromPrimitive;
    use pdqhash::image::{self, DynamicImage, ImageBuffer, Luma};

    use crate::kernel::{
        DefaultKernel, ReferenceKernel, SquareGenericArrayExt,
        type_traits::{DivisibleBy8, SquareOf},
    };

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

    fn test_hash_impl_lib<K: Kernel>(kernel: &mut K)
    where
        K: Kernel<OutputDimension = U16, InputDimension = U512>,
    {
        let mut buf1 = Box::default();
        let mut buf2 = Box::default();

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

            let mut output = GenericArray::default();

            let quality = hash(
                kernel,
                GenericArray::<_, _>::from_slice(input_image_f.as_raw().as_slice())
                    .unflatten_square_ref(),
                &mut output,
                &mut buf1,
                &mut buf2,
            );

            let mut distance = 0;
            for (a, b) in output.flatten().iter().zip(output_expected.0.iter()) {
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

    fn test_hash_impl_ref<
        ID: ArrayLength + SquareOf,
        OD: ArrayLength + SquareOf,
        K: Kernel<InputDimension = ID, OutputDimension = OD>,
    >(
        kernel: &mut K,
    ) where
        OD: DivisibleBy8,
        <ID as SquareOf>::Output: ArrayLength,
        <OD as SquareOf>::Output: ArrayLength,
        ReferenceKernel<K::InternalFloat>:
            Kernel<InputDimension = ID, InternalFloat = K::InternalFloat, OutputDimension = OD>,
    {
        let mut buf1 = Box::default();
        let mut buf1a = Box::default();
        let mut buf2 = Box::default();

        for (name, image_data) in [
            ("aaa-orig.jpg", TEST_IMAGE_AAA_ORIG),
            ("anime.png", TEST_IMAGE_ANIME),
            ("music.png", TEST_IMAGE_MUSIC),
            ("neofetch.png", TEST_IMAGE_NEOFETCH),
        ] {
            let input = image::load_from_memory(image_data).unwrap();

            let input = input.resize_exact(
                ID::USIZE as _,
                ID::USIZE as _,
                image::imageops::FilterType::Triangle,
            );

            let input_image = input
                .to_luma16()
                .iter()
                .map(|p| FromPrimitive::from_u16(*p).unwrap())
                .collect::<Vec<_>>();

            let mut output = GenericArray::default();
            let mut output_ref = GenericArray::default();

            hash(
                kernel,
                GenericArray::from_slice(input_image.as_slice()).unflatten_square_ref(),
                &mut output,
                &mut buf1,
                &mut buf2,
            );
            let mut ref_kernel = ReferenceKernel::<K::InternalFloat>::default();
            let mut thres = K::InternalFloat::default();
            let quality_ref = hash_get_threshold(
                &mut ref_kernel,
                GenericArray::from_slice(input_image.as_slice()).unflatten_square_ref(),
                &mut thres,
                &mut output_ref,
                &mut buf1a,
                &mut buf2,
            );
            let mut distance = 0;
            for (a, b) in output.iter().flatten().zip(output_ref.iter().flatten()) {
                let bits_diff = (a ^ b).count_ones();
                distance += bits_diff;
            }

            println!(
                "[{} ({})] {}: Distance vs. ref32: {}/{} (Q={})",
                std::any::type_name::<K>(),
                std::any::type_name::<K::InternalFloat>(),
                name,
                distance,
                16 * 16,
                quality_ref
            );
        }
    }

    #[test]
    fn test_hash_rethreshold() {
        let input = image::load_from_memory(TEST_IMAGE_AAA_ORIG).unwrap();
        let input = input.resize_exact(512, 512, image::imageops::FilterType::Triangle);
        let input_image = input
            .to_luma16()
            .iter()
            .map(|p| FromPrimitive::from_u16(*p).unwrap())
            .collect::<Vec<_>>();
        let mut output = GenericArray::default();
        let mut output_rethres = GenericArray::<GenericArray<u8, U2>, U16>::default();
        let mut buf1 = Box::default();
        let mut thres = 0.0f32;
        let mut pdqf = GenericArray::<GenericArray<f32, U16>, U16>::default();

        hash_get_threshold(
            &mut DefaultKernel,
            GenericArray::from_slice(input_image.as_slice()).unflatten_square_ref(),
            &mut thres,
            &mut output,
            &mut buf1,
            &mut pdqf,
        );

        kernel::threshold::threshold_2d_f32::<U16>(&pdqf, &mut output_rethres, thres);

        assert_eq!(output_rethres, output);
    }

    #[cfg(feature = "reference-rug")]
    fn test_hash_impl_ref_arb<
        ID: ArrayLength + SquareOf,
        OD: ArrayLength + SquareOf + DivisibleBy8,
        K: Kernel<InputDimension = ID, OutputDimension = OD>,
    >(
        kernel: &mut K,
    ) where
        ReferenceKernel<crate::kernel::float128::ArbFloat<96>>: Kernel<
                InputDimension = ID,
                InternalFloat = crate::kernel::float128::ArbFloat<96>,
                OutputDimension = OD,
            >,
    {
        use generic_array::typenum::U127;

        use crate::kernel::float128::ArbFloat;
        let mut buf1 = Box::default();
        let mut buf1a_uninit =
            Box::<GenericArray<GenericArray<ArbFloat<96>, U127>, U127>>::new_uninit();
        for i in 0..127 {
            for j in 0..127 {
                let value = ArbFloat::<96>::default();
                unsafe {
                    buf1a_uninit
                        .as_mut()
                        .assume_init_mut()
                        .get_unchecked_mut(i)
                        .as_mut_ptr()
                        .add(j)
                        .write(value);
                }
            }
        }
        let mut buf1a = unsafe { buf1a_uninit.assume_init() };
        let mut buf2 = Box::default();
        let mut buf2a = Box::default();

        for (name, image_data) in [
            ("aaa-orig.jpg", TEST_IMAGE_AAA_ORIG),
            ("anime.png", TEST_IMAGE_ANIME),
            ("music.png", TEST_IMAGE_MUSIC),
            ("neofetch.png", TEST_IMAGE_NEOFETCH),
        ] {
            let input = image::load_from_memory(image_data).unwrap();

            let input = input.resize_exact(512, 512, image::imageops::FilterType::Triangle);

            let input_image = input
                .to_luma16()
                .iter()
                .map(|p| FromPrimitive::from_u16(*p).unwrap())
                .collect::<Vec<_>>();

            let mut output = GenericArray::default();
            let mut output_ref = GenericArray::default();

            hash(
                kernel,
                GenericArray::from_slice(input_image.as_slice()).unflatten_square_ref(),
                &mut output,
                &mut buf1,
                &mut buf2,
            );
            let mut ref_kernel = ReferenceKernel::<ArbFloat<96>>::default();
            let quality_ref = hash(
                &mut ref_kernel,
                GenericArray::from_slice(input_image.as_slice()).unflatten_square_ref(),
                &mut output_ref,
                &mut buf1a,
                &mut buf2a,
            );
            let mut distance = 0;
            for (a, b) in output.iter().flatten().zip(output_ref.iter().flatten()) {
                let bits_diff = (a ^ b).count_ones();
                distance += bits_diff;
            }

            println!(
                "[{} ({})] {}: Distance vs. ref96: {}/{} (Q={})",
                std::any::type_name::<K>(),
                std::any::type_name::<K::InternalFloat>(),
                name,
                distance,
                16 * 16,
                quality_ref
            );
        }
    }

    #[test]
    fn test_hash_impl_base() {
        let mut kernel = kernel::DefaultKernel;
        test_hash_impl_lib(&mut kernel);
        test_hash_impl_ref(&mut kernel);
    }

    #[cfg(feature = "reference-rug")]
    #[test]
    fn test_hash_impl_base_arb() {
        let mut kernel = kernel::DefaultKernel;
        test_hash_impl_ref_arb(&mut kernel);
    }

    #[cfg(feature = "reference-rug")]
    #[test]
    fn test_hash_impl_ref_arb_rug() {
        let mut kernel = kernel::ReferenceKernel::<f32>::default();
        test_hash_impl_ref_arb(&mut kernel);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_hash_impl_avx2() {
        let mut kernel = kernel::x86::Avx2F32Kernel;
        test_hash_impl_lib(&mut kernel);
    }

    #[cfg(all(target_arch = "x86_64", feature = "reference-rug"))]
    #[test]
    fn test_hash_impl_avx2_arb() {
        let mut kernel = kernel::x86::Avx2F32Kernel;
        test_hash_impl_ref_arb(&mut kernel);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    fn test_hash_impl_avx512() {
        let mut kernel = kernel::x86::Avx512F32Kernel;
        test_hash_impl_lib(&mut kernel);
    }

    #[cfg(all(target_arch = "x86_64", feature = "reference-rug", feature = "avx512"))]
    #[test]
    fn test_hash_impl_avx512_arb() {
        let mut kernel = kernel::x86::Avx512F32Kernel;
        test_hash_impl_ref_arb(&mut kernel);
    }
}
