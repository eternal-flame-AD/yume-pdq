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
    type_traits::{DivisibleBy8, EvaluateHardwareFeature, SquareOf},
};

use generic_array::{
    ArrayLength,
    typenum::{B1, U16},
};

/// PDQ compression kernel
pub mod kernel;

pub use kernel::smart_kernel;

/// Memory alignment utilities.
pub mod alignment;

/// Diagnostic utilities for debugging, integrating developers, or generally for fun inspecting internals. Not part of the stable API. Correctness is only checked empirically.
#[cfg(any(test, all(feature = "unstable", feature = "std")))]
pub mod testing;

#[cfg(feature = "ffi")]
pub mod ffi {
    //! Foreign function interface binding for the PDQ hash function.
    use generic_array::{sequence::Unflatten, typenum::U32};

    use super::*;
    use crate::kernel::{SmartKernelConcreteType, SquareGenericArrayExt, smart_kernel_impl};
    use std::sync::LazyLock;

    const SMART_KERNEL: LazyLock<SmartKernelConcreteType> = LazyLock::new(smart_kernel_impl);

    #[unsafe(export_name = "yume_pdq_hash_smart_kernel")]
    /// Compute the PDQ hash of a 512x512 single-channel image using [`kernel::smart_kernel`].
    ///
    /// # Safety
    ///
    /// - `input` must be a pointer to a 512x512 single-channel image in float32 format, row-major order.
    /// - `threshold` must be a valid aligned pointer to a f32 value or NULL.
    /// - `output` must be a pointer to a 2x16 array of u8 to receive the final 256-bit hash.
    /// - `buf1` must be a pointer to a 128x128 array of f32 values to receive the intermediate results of the DCT transform.
    /// - `tmp` must be a pointer to a 128x1 array of f32 values as scratch space for the DCT transform.
    /// - `pdqf` must be a pointer to a 16x16 array of f32 values to receive PDQF (unquantized) hash values.
    ///
    /// # Returns
    ///
    /// The quality of the hash as a f32 value between 0.0 and 1.0. You are responsible for checking whether quality is acceptable.
    pub unsafe extern "C" fn hash_smart_kernel(
        input: &[f32; 512 * 512],
        threshold: *mut f32,
        output: &mut [u8; 2 * 16],
        buf1: &mut [f32; 128 * 128],
        tmp: &mut [f32; 128],
        pdqf: &mut [f32; 16 * 16],
    ) -> f32 {
        let mut kernel = SMART_KERNEL.clone();
        let input = GenericArray::from_slice(input).unflatten_square_ref();
        let output = GenericArray::<_, U32>::from_mut_slice(output).unflatten();
        let buf1 = GenericArray::from_mut_slice(buf1).unflatten_square_mut();
        let pdqf = GenericArray::from_mut_slice(pdqf).unflatten_square_mut();

        let mut dummy_threshold = 0.0;

        crate::hash_get_threshold(
            &mut kernel,
            &input,
            unsafe { threshold.as_mut().unwrap_or(&mut dummy_threshold) },
            output,
            buf1,
            tmp.into(),
            pdqf,
        )
    }
}

#[cfg(feature = "lut-utils")]
/// Some miscellaneous lookup tables that might be helpful for downstream applications.
///
/// I reserve the right to remove or break API of this module in the future.
pub mod lut_utils;

/// PDQ hash type
pub type PDQHash<L = U16> = GenericArray<GenericArray<u8, <L as DivisibleBy8>::Output>, L>;

/// Unquantized PDQ hash ("PDQF" in the original paper)
pub type PDQHashF<N = f32, L = U16> = GenericArray<GenericArray<N, L>, L>;

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel.
///
/// This is a convenience wrapper function and just calls [`hash_get_threshold`] with a dummy output location.
///
/// # TLDR how to use this contraption
///
/// ```rust,no_run
/// use yume_pdq::{smart_kernel, GenericArray};
///
/// // Create a 512x512 input image
/// //
/// // values 0.0-255.0 if you want the quality for be accurate, otherwise scale is not important
/// // this is a known limitation and will be fixed in the future
/// let input: GenericArray<GenericArray<f32, _>, _> = GenericArray::default();
///
/// // Get the optimal kernel for your CPU
/// let mut kernel = smart_kernel();
///
/// // Allocate output and temporary buffers (make sure your stack is big enough or allocate on the heap)
/// let mut output = GenericArray::default();  // Will contain the final 256-bit hash
/// let mut buf1 = GenericArray::default();    // Temporary buffer
/// let mut row_tmp = GenericArray::default();    // Temporary buffer
/// let mut pdqf = GenericArray::default();    // Temporary buffer (PDQF unquantized hash)
///
/// // Compute the hash
/// let quality = yume_pdq::hash(&mut kernel, &input, &mut output, &mut buf1, &mut row_tmp, &mut pdqf);
///
pub fn hash<K: Kernel>(
    kernel: &mut K,
    input: &GenericArray<GenericArray<f32, K::InputDimension>, K::InputDimension>,
    output: &mut GenericArray<
        GenericArray<u8, <K::OutputDimension as DivisibleBy8>::Output>,
        K::OutputDimension,
    >,
    buf1: &mut GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>,
    tmp: &mut GenericArray<K::InternalFloat, K::Buffer1WidthX>,
    // the floating point version of the input image
    pdqf: &mut PDQHashF<K::InternalFloat, K::OutputDimension>,
) -> f32
where
    <K as Kernel>::OutputDimension: DivisibleBy8,
    <K as Kernel>::InputDimension: SquareOf,
    <<K as Kernel>::InputDimension as SquareOf>::Output: ArrayLength,
    <K as Kernel>::OutputDimension: SquareOf,
    <<K as Kernel>::OutputDimension as SquareOf>::Output: ArrayLength,
    <K as Kernel>::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
{
    hash_get_threshold(
        kernel,
        input,
        &mut Default::default(),
        output,
        buf1,
        tmp,
        pdqf,
    )
}

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel, obtaining the threshold value useful for [`kernel::threshold::threshold_2d_f32`].
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
    tmp: &mut GenericArray<K::InternalFloat, K::Buffer1WidthX>,
    // the floating point version of the input image
    pdqf: &mut PDQHashF<K::InternalFloat, K::OutputDimension>,
) -> f32
where
    <K as Kernel>::InputDimension: SquareOf,
    <<K as Kernel>::InputDimension as SquareOf>::Output: ArrayLength,
    <K as Kernel>::OutputDimension: SquareOf,
    <<K as Kernel>::OutputDimension as SquareOf>::Output: ArrayLength,
    <K as Kernel>::OutputDimension: DivisibleBy8,
    <K as Kernel>::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
{
    kernel.jarosz_compress(input, buf1);
    kernel.dct2d(buf1, tmp, pdqf);
    let gradient = kernel.sum_of_gradients(pdqf);
    let quality = K::adjust_quality(gradient);

    kernel.quantize(pdqf, threshold, output);
    quality
}

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel without quantization.
///
/// This is called PDQF in the original paper.
///
pub fn hash_float<K: Kernel>(
    kernel: &mut K,
    input: &GenericArray<GenericArray<f32, K::InputDimension>, K::InputDimension>,
    output: &mut PDQHashF<K::InternalFloat, K::OutputDimension>,
    buf1: &mut GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>,
    tmp: &mut GenericArray<K::InternalFloat, K::Buffer1WidthX>,
) -> f32
where
    <K as Kernel>::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
{
    kernel.jarosz_compress(input, buf1);
    kernel.dct2d(buf1, tmp, output);
    let gradient = kernel.sum_of_gradients(output);

    K::adjust_quality(gradient)
}

#[cfg(test)]
mod tests {

    use core::ops::Mul;

    use generic_array::{
        sequence::Flatten,
        typenum::{U2, U512},
    };
    use pdqhash::image::{self, DynamicImage};

    use crate::kernel::{
        ReferenceKernel, SquareGenericArrayExt,
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
        K::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
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

            let input_image = input.to_luma8();

            let input_image_f = input_image
                .as_raw()
                .iter()
                .map(|p| *p as f32)
                .collect::<Vec<_>>();

            let output_expected =
                pdqhash::generate_pdq_full_size(&DynamicImage::ImageLuma8(input_image));

            let mut output = GenericArray::default();

            let quality = hash(
                kernel,
                GenericArray::<_, _>::from_slice(input_image_f.as_slice()).unflatten_square_ref(),
                &mut output,
                &mut buf1,
                &mut GenericArray::default(),
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
        OD: ArrayLength + SquareOf + Mul<OD>,
        K: Kernel<InputDimension = ID, OutputDimension = OD>,
    >(
        kernel: &mut K,
    ) where
        OD: DivisibleBy8,
        <ID as SquareOf>::Output: ArrayLength,
        <OD as SquareOf>::Output: ArrayLength,
        <OD as Mul<OD>>::Output: ArrayLength,
        ReferenceKernel<K::InternalFloat>:
            Kernel<InputDimension = ID, InternalFloat = K::InternalFloat, OutputDimension = OD>,
        <ReferenceKernel<K::InternalFloat> as Kernel>::RequiredHardwareFeature:
            EvaluateHardwareFeature<EnabledStatic = B1>,
        K::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
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
                .to_luma8()
                .iter()
                .map(|p| *p as f32)
                .collect::<Vec<_>>();

            let mut output = GenericArray::default();
            let mut output_ref = GenericArray::default();

            hash(
                kernel,
                GenericArray::from_slice(input_image.as_slice()).unflatten_square_ref(),
                &mut output,
                &mut buf1,
                &mut GenericArray::default(),
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
                &mut GenericArray::default(),
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
            .to_luma8()
            .iter()
            .map(|p| *p as f32)
            .collect::<Vec<_>>();
        let mut output = GenericArray::default();
        let mut output_rethres = GenericArray::<GenericArray<u8, U2>, U16>::default();
        let mut buf1 = Box::default();
        let mut thres = 0.0f32;
        let mut pdqf = GenericArray::<GenericArray<f32, U16>, U16>::default();

        hash_get_threshold(
            &mut kernel::DefaultKernelNoPadding::default(),
            GenericArray::from_slice(input_image.as_slice()).unflatten_square_ref(),
            &mut thres,
            &mut output,
            &mut buf1,
            &mut GenericArray::default(),
            &mut pdqf,
        );

        kernel::threshold::threshold_2d_f32::<U16>(&pdqf, &mut output_rethres, thres);

        assert_eq!(output_rethres, output);
    }

    #[cfg(feature = "reference-rug")]
    fn test_hash_impl_ref_arb<
        ID: ArrayLength + SquareOf,
        OD: ArrayLength + SquareOf + DivisibleBy8 + Mul<OD>,
        K: Kernel<InputDimension = ID, OutputDimension = OD>,
    >(
        kernel: &mut K,
    ) where
        K::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
        ReferenceKernel<crate::kernel::float128::ArbFloat<96>>: Kernel<
                InputDimension = ID,
                InternalFloat = crate::kernel::float128::ArbFloat<96>,
                OutputDimension = OD,
            >,
        <OD as Mul<OD>>::Output: ArrayLength,
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
                .to_luma8()
                .iter()
                .map(|p| *p as f32)
                .collect::<Vec<_>>();

            let mut output = GenericArray::default();
            let mut output_ref = GenericArray::default();

            hash(
                kernel,
                GenericArray::from_slice(input_image.as_slice()).unflatten_square_ref(),
                &mut output,
                &mut buf1,
                &mut GenericArray::default(),
                &mut buf2,
            );
            let mut ref_kernel = ReferenceKernel::<ArbFloat<96>>::default();
            let quality_ref = hash(
                &mut ref_kernel,
                GenericArray::from_slice(input_image.as_slice()).unflatten_square_ref(),
                &mut output_ref,
                &mut buf1a,
                &mut GenericArray::default(),
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
        let mut kernel = kernel::DefaultKernelNoPadding::default();
        test_hash_impl_lib(&mut kernel);
        test_hash_impl_ref(&mut kernel);
    }

    #[cfg(feature = "reference-rug")]
    #[test]
    fn test_hash_impl_base_arb() {
        let mut kernel = kernel::DefaultKernelNoPadding::default();
        test_hash_impl_ref_arb(&mut kernel);
    }

    #[cfg(feature = "reference-rug")]
    #[test]
    fn test_hash_impl_ref_arb_rug() {
        let mut kernel = kernel::ReferenceKernel::<f32>::default();
        test_hash_impl_ref_arb(&mut kernel);
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        all(target_feature = "avx2", target_feature = "fma")
    ))]
    fn test_hash_impl_avx2() {
        let mut kernel = kernel::x86::Avx2F32Kernel;
        test_hash_impl_lib(&mut kernel);
    }

    #[cfg(all(
        target_arch = "x86_64",
        all(target_feature = "avx2", target_feature = "fma"),
        feature = "reference-rug"
    ))]
    #[test]
    fn test_hash_impl_avx2_arb() {
        let mut kernel = kernel::x86::Avx2F32Kernel;
        test_hash_impl_ref_arb(&mut kernel);
    }

    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        feature = "avx512",
        all(target_feature = "avx512f")
    ))]
    fn test_hash_impl_avx512() {
        let mut kernel = kernel::x86::Avx512F32Kernel;
        test_hash_impl_lib(&mut kernel);
    }

    #[cfg(all(
        target_arch = "x86_64",
        feature = "reference-rug",
        target_feature = "avx512f",
        feature = "avx512"
    ))]
    #[test]
    fn test_hash_impl_avx512_arb() {
        let mut kernel = kernel::x86::Avx512F32Kernel;
        test_hash_impl_ref_arb(&mut kernel);
    }
}
