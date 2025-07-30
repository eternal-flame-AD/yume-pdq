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
#![cfg_attr(feature = "portable-simd", feature(portable_simd))]
#![warn(missing_docs, clippy::pedantic)]
#![allow(
    clippy::type_complexity,
    clippy::missing_errors_doc,
    clippy::doc_markdown,
    clippy::similar_names,
    clippy::cast_lossless
)]
#![allow(clippy::inline_always)]
#![allow(
    clippy::bool_to_int_with_if,
    reason = "I don't know, I think it's more readable"
)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub use const_default::{self, ConstDefault};
pub use generic_array::{self, GenericArray};
pub use num_traits;
pub use zeroize;

use kernel::{
    Kernel,
    threshold::threshold_2d_f32,
    type_traits::{DivisibleBy8, EvaluateHardwareFeature, SquareOf},
};

use generic_array::{
    ArrayLength,
    typenum::{B1, IsLessOrEqual, U16, U32},
};

/// PDQ compression kernel
pub mod kernel;

/// PDQ matching solution
///
/// Currently all solutions are exact linear-scan nearest neighbor thresholding and are expected to continue to be so
///
/// Metric-tree based solutions such as BK-tree and KD-tree are not efficient due to unique characteristics of PDQ hash and dihedral invariance necessitating all screens to match 8 hashes at once. See [TECHNICAL.md](TECHNICAL.md) for more details.
///
/// ANN will lead to significant, guaranteed false negatives (unlike my DISC21 benchmark shows 2 outliers (still well within threshold) does not mean guaranteed <98% recall).
/// Experiment using Facebook(R) Faiss IndexBinaryHNSW on real NEMEC PDQ data shows 90% recall with nearing 10ms per query single-threaded.
/// Even if one can accept this recall (one shouldn't), performance is still not competitive with any optimized matcher here.
pub mod matching;

pub use kernel::smart_kernel;

/// Memory alignment utilities.
pub mod alignment;

/// Diagnostic utilities for debugging, integrating developers, or generally for fun inspecting internals. Not part of the stable API. Correctness is only checked empirically.
#[cfg(any(test, all(feature = "unstable", feature = "std")))]
pub mod testing;

#[cfg(target_arch = "wasm32")]
/// WASM bindings.
/// cbindgen:ignore
pub mod wasm;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
/// A packed representation of a matrix for dihedral transformations.
pub struct Dihedrals {
    /// The packed representation of the dihedral matrix.
    ///
    /// Ordering is first x-to-x, then x-to-y, then y-to-x, then y-to-y. Big-endian signed 8-bit integers packed into a u32.
    pub packed: u32,
}

impl Dihedrals {
    /// Create a new dihedral from a tuple of tuples.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::identity_op,
        clippy::cast_sign_loss
    )]
    #[must_use]
    pub const fn from_tuples(dx: (i8, i8), dy: (i8, i8)) -> Self {
        Self {
            packed: u32::from_be_bytes([(dx.0 as u8), (dx.1 as u8), (dy.0 as u8), (dy.1 as u8)]),
        }
    }

    /// Convert the dihedral to a tuple of tuples.
    #[allow(clippy::cast_possible_truncation, clippy::identity_op)]
    #[must_use]
    pub const fn into_tuples(self) -> ((i8, i8), (i8, i8)) {
        let (dx0, dx1) = (self.packed >> 24 & 0xFF, self.packed >> 16 & 0xFF);
        let (dy0, dy1) = (self.packed >> 8 & 0xFF, self.packed >> 0 & 0xFF);
        ((dx0 as i8, dx1 as i8), (dy0 as i8, dy1 as i8))
    }

    /// The normal dihedral transformation.
    pub const NORMAL: Self = Self::from_tuples((1, 0), (0, 1));
    /// The flipped dihedral transformation.
    pub const FLIPPED: Self = Self::from_tuples((1, 0), (0, -1));
    /// The flopped dihedral transformation.
    pub const FLOPPED: Self = Self::from_tuples((-1, 0), (0, 1));
    /// The 180-degree rotated dihedral transformation.
    pub const ROTATED_180: Self = Self::from_tuples((-1, 0), (0, -1));
    /// The 90-degree rotated dihedral transformation.
    pub const ROTATED_90: Self = Self::from_tuples((0, 1), (-1, 0));
    /// The 270-degree rotated dihedral transformation.
    pub const ROTATED_270: Self = Self::from_tuples((0, 1), (1, 0));
    /// The 90-degree flopped dihedral transformation.
    pub const ROTATED_90_FLOPPED: Self = Self::from_tuples((0, -1), (-1, 0));
    /// The 270-degree flopped dihedral transformation.
    pub const FLOPPED_ROTATED_270: Self = Self::from_tuples((0, -1), (1, 0));
}

#[cfg(feature = "ffi")]
#[allow(clippy::transmute_ptr_to_ptr, clippy::transmute_ptr_to_ref)]
pub mod ffi {
    //! Foreign function interface binding for the PDQ hash function.
    //!
    //! There is no guarantee of Rust-level API compatibility in this module.
    use generic_array::{sequence::Unflatten, typenum::U32};

    #[allow(clippy::wildcard_imports)]
    use super::*;
    use crate::kernel::{SmartKernelConcreteType, SquareGenericArrayExt, smart_kernel_impl};
    use core::ffi::c_void;
    use std::sync::LazyLock;

    include!(concat!(env!("OUT_DIR"), "/version_ffi.rs"));

    static SMART_KERNEL: LazyLock<SmartKernelConcreteType> = LazyLock::new(smart_kernel_impl);

    /// re-exported constants for the dihedrals
    #[unsafe(no_mangle)]
    pub static YUME_PDQ_DIHEDRAL_NORMAL: Dihedrals = Dihedrals::NORMAL;
    /// re-exported constants for the dihedrals
    #[unsafe(no_mangle)]
    pub static YUME_PDQ_DIHEDRAL_FLIPPED: Dihedrals = Dihedrals::FLIPPED;
    /// re-exported constants for the dihedrals
    #[unsafe(no_mangle)]
    pub static YUME_PDQ_DIHEDRAL_FLOPPED: Dihedrals = Dihedrals::FLOPPED;
    /// re-exported constants for the dihedrals
    #[unsafe(no_mangle)]
    pub static YUME_PDQ_DIHEDRAL_ROTATED_180: Dihedrals = Dihedrals::ROTATED_180;
    /// re-exported constants for the dihedrals
    #[unsafe(no_mangle)]
    pub static YUME_PDQ_DIHEDRAL_ROTATED_90: Dihedrals = Dihedrals::ROTATED_90;
    /// re-exported constants for the dihedrals
    #[unsafe(no_mangle)]
    pub static YUME_PDQ_DIHEDRAL_ROTATED_270: Dihedrals = Dihedrals::ROTATED_270;
    /// re-exported constants for the dihedrals
    #[unsafe(no_mangle)]
    pub static YUME_PDQ_DIHEDRAL_ROTATED_90_FLOPPED: Dihedrals = Dihedrals::ROTATED_90_FLOPPED;
    /// re-exported constants for the dihedrals
    #[unsafe(no_mangle)]
    pub static YUME_PDQ_DIHEDRAL_FLOPPED_ROTATED_270: Dihedrals = Dihedrals::FLOPPED_ROTATED_270;

    /// A callback function for visiting all dihedrals.
    ///
    /// The threshold, PDQF and quantized output will be available to the caller via the provided buffers ONLY before the callback returns.
    ///
    /// Return true to continue, false to stop.
    ///
    /// The function must not modify the buffers, and must copy them out before returning if they need to keep them.
    pub type DihedralCallback =
        extern "C" fn(ctx: *mut c_void, dihedral: u32, threshold: f32, quality: f32) -> bool;

    #[unsafe(export_name = "yume_pdq_visit_dihedrals_smart_kernel")]
    /// Visit the 7 alternative dihedrals of the PDQF hash.
    ///
    /// # Safety
    ///
    /// - `ctx` is transparently passed to the callback function.
    /// - `threshold` must be a valid threshold value for the provided PDQF input received from [`hash_smart_kernel`].
    /// - `output` is out only, must be a pointer to a 2x16 array of u8 to receive any intermediate 256-bit hash. It does not have to be initialized to any particular value.
    /// - `pdqf` is in/out, must be a pointer to a 16x16 array of f32 values of the initial PDQF data, and be writable to receive derived PDQF (unquantized) hash values.
    /// - `callback` must be a valid callback function that will be called for each dihedral.
    ///
    /// No buffer should overlap.
    ///
    /// # Returns
    ///
    /// - `true` if all dihedrals were visited, `false` if the callback returned false for any dihedral.
    pub unsafe extern "C" fn visit_dihedrals_smart_kernel(
        ctx: *mut c_void,
        threshold: f32,
        output: *mut u8,
        pdqf: *mut f32,
        callback: DihedralCallback,
    ) -> bool {
        let output = unsafe { core::mem::transmute::<*mut u8, &mut [u8; 2 * 16]>(output) };
        let pdqf = unsafe { core::mem::transmute::<*mut f32, &mut [f32; 16 * 16]>(pdqf) };

        let pdqf = GenericArray::from_mut_slice(pdqf).unflatten_square_mut();
        let output = GenericArray::<_, U32>::from_mut_slice(output).unflatten();

        crate::visit_dihedrals(
            &mut SMART_KERNEL.clone(),
            pdqf,
            output,
            threshold,
            |dihedral, _, (quality, _pdqf, _output)| {
                if callback(ctx, dihedral.packed, threshold, quality) {
                    Ok(())
                } else {
                    Err(())
                }
            },
        )
        .is_ok()
    }

    #[unsafe(export_name = "yume_pdq_hash_smart_kernel")]
    /// Compute the PDQ hash of a 512x512 single-channel image using [`kernel::smart_kernel`].
    ///
    /// # Safety
    ///
    /// - `input` is in only, must be a pointer to a 512x512 single-channel image in float32 format, row-major order.
    /// - `threshold` is out only, must be a valid aligned pointer to a f32 value or NULL.
    /// - `output` is out only, must be a pointer to a 2x16 array of u8 to receive the final 256-bit hash.
    /// - `buf1` is in/out, must be a pointer to a 128x128 array of f32 values to receive the intermediate results of the DCT transform.
    /// - `tmp` is in/out, must be a pointer to a 128x1 array of f32 values as scratch space for the DCT transform.
    /// - `pdqf` is out only, must be a pointer to a 16x16 array of f32 values to receive PDQF (unquantized) hash values.
    ///
    /// No buffer should overlap.
    ///
    /// # Returns
    ///
    /// The quality of the hash as a f32 value between 0.0 and 1.0. You are responsible for checking whether quality is acceptable.
    pub unsafe extern "C" fn hash_smart_kernel(
        input: *const f32,
        threshold: *mut f32,
        output: *mut u8,
        buf1: *mut f32,
        tmp: *mut f32,
        pdqf: *mut f32,
    ) -> f32 {
        let input = unsafe { core::mem::transmute::<*const f32, &[f32; 512 * 512]>(input) };
        let output = unsafe { core::mem::transmute::<*mut u8, &mut [u8; 2 * 16]>(output) };
        let buf1 = unsafe { core::mem::transmute::<*mut f32, &mut [f32; 128 * 128]>(buf1) };
        let tmp = unsafe { core::mem::transmute::<*mut f32, &mut [f32; 128]>(tmp) };
        let pdqf = unsafe { core::mem::transmute::<*mut f32, &mut [f32; 16 * 16]>(pdqf) };

        #[allow(clippy::clone_on_copy)]
        let mut kernel = SMART_KERNEL.clone();
        let input = GenericArray::from_slice(input).unflatten_square_ref();
        let output = GenericArray::<_, U32>::from_mut_slice(output).unflatten();
        let buf1 = GenericArray::from_mut_slice(buf1).unflatten_square_mut();
        let pdqf = GenericArray::from_mut_slice(pdqf).unflatten_square_mut();

        let mut dummy_threshold = 0.0;

        crate::hash_get_threshold(
            &mut kernel,
            input,
            unsafe { threshold.as_mut().unwrap_or(&mut dummy_threshold) },
            output,
            buf1,
            tmp.into(),
            pdqf,
        )
    }
}

/// PDQ hash type
pub type PDQHash<L = U16> = GenericArray<GenericArray<u8, <L as DivisibleBy8>::Output>, L>;

/// Unquantized PDQ hash ("PDQF" in the original paper)
pub type PDQHashF<N = f32, L = U16> = GenericArray<GenericArray<N, L>, L>;

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel.
///
/// This is a convenience wrapper function and just calls [`hash_get_threshold`] with a dummy output location.
///
/// **Warning**: While it may be tempting, DO NOT pass uninitialized memory into any parameter of this function.
/// While the contents are not important, the padding must be zero-initialized otherwise subtly incorrect results will be returned.
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
#[inline]
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

/// Visit the 7 alternative dihedrals of the PDQF hash.
///
/// The callback function is called with first the matrix of the dihedral, then the threshold, then the quality, then the PDQF hash and finally the output hash.
///
/// The PDQF hash and output hash are guaranteed to point to the same buffer as the input hash, it is just to make the borrow-checker happy.
pub fn visit_dihedrals<
    K: Kernel<InternalFloat = f32>,
    E,
    F: FnMut(
        Dihedrals,
        f32,
        (
            f32,
            &mut PDQHashF<K::InternalFloat, K::OutputDimension>,
            &mut PDQHash<K::OutputDimension>,
        ),
    ) -> Result<(), E>,
>(
    kernel: &mut K,
    pdqf: &mut PDQHashF<K::InternalFloat, K::OutputDimension>,
    output: &mut PDQHash<K::OutputDimension>,
    threshold: K::InternalFloat,
    mut f: F,
) -> Result<(), E>
where
    K::OutputDimension: DivisibleBy8 + IsLessOrEqual<U32, Output = B1>,
    <K as Kernel>::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
{
    macro_rules! callback {
        ($dihedral:expr, $threshold:expr) => {
            let gradient = kernel.sum_of_gradients(pdqf);
            let quality = K::adjust_quality(gradient);
            f($dihedral, $threshold, (quality, pdqf, output))?;
        };
    }

    let mut threshold_negate_alt_cols = threshold;
    let mut threshold_negate_alt_rows = threshold;
    let mut threshold_negate_off_diagonals = threshold;
    kernel.pdqf_negate_alt_cols::<false>(pdqf); // first negate by columns
    kernel.quantize(pdqf, &mut threshold_negate_alt_cols, output);
    callback!(Dihedrals::FLOPPED, threshold_negate_alt_cols);
    kernel.pdqf_negate_alt_rows::<true>(pdqf); // then negate by rows, getting the negate-by off-diagonals
    kernel.quantize(pdqf, &mut threshold_negate_off_diagonals, output);
    callback!(Dihedrals::ROTATED_180, threshold_negate_off_diagonals);
    kernel.pdqf_negate_alt_cols::<false>(pdqf); // then negate by columns again, getting the negate-by alt-rows
    kernel.quantize(pdqf, &mut threshold_negate_alt_rows, output);
    callback!(Dihedrals::FLIPPED, threshold_negate_alt_rows);
    // undo all negations, transpose
    kernel.pdqf_negate_alt_rows::<true>(pdqf);
    kernel.pdqf_t(pdqf);
    threshold_2d_f32(pdqf, output, threshold);
    callback!(Dihedrals::ROTATED_90_FLOPPED, threshold);
    // now undo the original transformations to get back to the other 3 hashes that require transposition
    kernel.pdqf_negate_alt_rows::<true>(pdqf);
    threshold_2d_f32(pdqf, output, threshold_negate_alt_cols);
    callback!(Dihedrals::ROTATED_270, threshold_negate_alt_cols);
    kernel.pdqf_negate_alt_cols::<false>(pdqf);
    threshold_2d_f32(pdqf, output, threshold_negate_off_diagonals);
    callback!(
        Dihedrals::FLOPPED_ROTATED_270,
        threshold_negate_off_diagonals
    );
    kernel.pdqf_negate_alt_rows::<true>(pdqf);
    threshold_2d_f32(pdqf, output, threshold_negate_alt_rows);
    callback!(Dihedrals::ROTATED_90, threshold_negate_alt_rows);
    Ok(())
}

/// Compute the PDQ hash of a 512x512 single-channel image using the given kernel, obtaining the threshold value useful for [`kernel::threshold::threshold_2d_f32`].
///
/// **Warning**: While it may be tempting, DO NOT pass uninitialized memory into any parameter of this function.
/// While the contents are not important, the padding must be zero-initialized otherwise subtly incorrect results will be returned.
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
/// **Warning**: While it may be tempting, DO NOT pass uninitialized memory into any parameter of this function.
/// While the contents are not important, the padding must be zero-initialized otherwise subtly incorrect results will be returned.
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

            // this is demo "bad" picture for hashing, highly malleable output is expected
            // when there is any non-prescribed preprocessing happening
            if name != "neofetch.png" {
                // half of the matching threshold
                assert!(distance <= 16);
            }
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

    #[test]
    #[cfg(feature = "portable-simd")]
    fn test_hash_impl_portable_simd() {
        let mut kernel = kernel::portable_simd::PortableSimdF32Kernel::<8>;
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
    #[cfg(all(target_arch = "x86_64", feature = "avx512", target_feature = "avx512f"))]
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
