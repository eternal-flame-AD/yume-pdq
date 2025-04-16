/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Monday, April 14, 2025
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

use generic_array::sequence::Flatten;
#[allow(unused_imports)]
use generic_array::{
    ArrayLength, GenericArray,
    typenum::{B0, B1, Bit, U4, U8, U32, U2048},
};

#[cfg(all(feature = "avx512", target_feature = "avx512vpopcntdq"))]
#[allow(clippy::wildcard_imports)]
use core::arch::x86_64::*;
use core::{
    fmt::{Debug, Display},
    ops::Mul,
};

use crate::{alignment::Align8, kernel::type_traits::EvaluateHardwareFeature};

#[cfg(feature = "vulkan")]
/// Vulkan-based accelerator-based matcher
pub mod vulkan;

/// A trait for PDQ matchers
///
/// Some matchers (mainly CPU-based) are initialized with the needles and "queried" with the haystack.
/// Some (mainly accelerator-based) are initialized with the haystack and "queried" with the needles due to the memory transfer overhead.
pub trait PDQMatcher {
    /// Number of hashes per single query
    type BatchSize: ArrayLength;
    /// The hardware features required to run this kernel
    type RequiredHardwareFeature: EvaluateHardwareFeature;
    /// Input vector dimension, in bytes
    type InputDimension: ArrayLength;
    /// Aligner type
    type Aligner<T>;

    /// Identification token.
    type Ident: Debug + Display + Clone + Copy + 'static + PartialEq;

    /// Get the identification token.
    fn ident() -> Self::Ident;

    /// Marker type to indicate the matcher may return false negatives for downstream code to constrain.
    ///
    /// Matchers that are only guaranteed to find at least one match if one exists but not all still count as no false negatives.
    ///
    /// Currently all matchers are exact (i.e. [`generic_array::typenum::B1`]).
    type NoFalseNegative: Bit;
    /// Marker type to indicate the matcher may return false positives for downstream code to constrain.
    ///
    /// Currently all matchers are exact (i.e. [`generic_array::typenum::B1`]).
    type NoFalsePositive: Bit;

    /// Whether the matcher can report all matches or only report at least one match if one exists (usually on accelerator-based matchers).
    type FindsAllMatches: Bit;

    /// Perform a single scan for a boolean match/no match result.
    ///
    /// It is very likely for many cases this is not actually faster than just find the match.
    /// But you should benchmark on your machine.
    fn scan(
        &mut self,
        query: &Self::Aligner<
            GenericArray<GenericArray<u8, Self::InputDimension>, Self::BatchSize>,
        >,
    ) -> bool
    where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>;

    /// Find the first match and return the indices of both the needle and the haystack.
    ///
    /// The scan may be aborted early if the callback returns `Some(R)`.
    fn find<R, F: FnMut(usize, usize) -> Option<R>>(
        &mut self,
        query: &Self::Aligner<
            GenericArray<GenericArray<u8, Self::InputDimension>, Self::BatchSize>,
        >,
        f: F,
    ) -> Option<R>
    where
        Self::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>;
}

/// A CPU-based matcher that uses population count.
///
/// Expect about 250-350 million haystack vectors per second without vectorization and 600-750 million with AVX-512.
///
/// It will use the AVX-512 vectorized vpopcntq if available and compile-time enabled (enable the "avx512" feature and "-C target-feature=+avx512vpopcntdq").
pub struct CpuMatcher {
    #[cfg(all(feature = "avx512", target_feature = "avx512vpopcntdq"))]
    needles_comp: [__m512i; 4],
    #[cfg(not(all(feature = "avx512", target_feature = "avx512vpopcntdq")))]
    needles: GenericArray<GenericArray<u64, U4>, U8>,
    threshold: u64,
}

impl CpuMatcher {
    #[cfg(not(all(feature = "avx512", target_feature = "avx512vpopcntdq")))]
    #[inline]
    #[must_use]
    /// Create a new CPU matcher from needles.
    ///
    /// # Panics
    ///
    /// Panics if the threshold is not in the range of [0, 256].
    pub fn new(needles: &GenericArray<GenericArray<u8, U32>, U8>, threshold: u64) -> Self {
        assert!(
            threshold <= 256,
            "threshold must be in the range of [0, 256]"
        );
        Self {
            needles: unsafe {
                needles
                    .as_ptr()
                    .cast::<GenericArray<_, U8>>()
                    .read_unaligned()
            },
            threshold,
        }
    }

    /// Create a new CPU matcher from a nested array of PDQ hashes.
    ///
    /// # Panics
    ///
    /// Panics if the threshold is not in the range of [0, 256].
    #[must_use]
    pub fn new_nested<
        T,
        N: ArrayLength + Mul<M>,
        M: ArrayLength,
        V: Flatten<T, N, M, Output = GenericArray<u8, U32>>,
    >(
        needles: &GenericArray<V, U8>,
        threshold: u64,
    ) -> Self
    where
        <N as Mul<M>>::Output: ArrayLength,
    {
        #[allow(clippy::missing_transmute_annotations, clippy::transmute_ptr_to_ptr)]
        unsafe {
            Self::new(
                core::mem::transmute::<_, &GenericArray<GenericArray<u8, U32>, U8>>(needles),
                threshold,
            )
        }
    }

    #[cfg(all(feature = "avx512", target_feature = "avx512vpopcntdq"))]
    #[inline]
    #[must_use]
    /// Create a new CPU matcher from needles.
    ///
    /// # Panics
    ///
    /// Panics if the threshold is not in the range of [0, 256].
    pub fn new(needles: &GenericArray<GenericArray<u8, U32>, U8>, threshold: u64) -> Self {
        assert!(
            threshold <= 256,
            "threshold must be in the range of [0, 256]"
        );
        use crate::alignment::Align64;
        let mut tmp = Align64::<[u64; 8]>::default();
        let ones = unsafe { _mm512_set1_epi64(!0) };

        let regs = core::array::from_fn(|reg| unsafe {
            #[allow(clippy::cast_ptr_alignment)]
            for i in 0..8 {
                tmp[i] = needles[i].as_ptr().cast::<u64>().add(reg).read_unaligned();
            }
            _mm512_xor_si512(ones, _mm512_load_si512(tmp.as_ptr().cast()))
        });
        Self {
            needles_comp: regs,
            threshold,
        }
    }
}

impl PDQMatcher for CpuMatcher {
    type BatchSize = U2048;
    #[cfg(all(feature = "avx512", target_feature = "avx512vpopcntdq"))]
    type RequiredHardwareFeature = crate::kernel::x86::CpuIdAvx512Vpopcntdq;
    #[cfg(not(all(feature = "avx512", target_feature = "avx512vpopcntdq")))]
    type RequiredHardwareFeature = crate::kernel::type_traits::Term;
    type InputDimension = U32;
    type Aligner<T> = Align8<T>;
    type NoFalseNegative = B1;
    type NoFalsePositive = B1;
    type FindsAllMatches = B1;
    type Ident = &'static str;

    #[cfg(not(all(feature = "avx512", target_feature = "avx512vpopcntdq")))]
    fn ident() -> Self::Ident {
        "CpuMatcher (u64 scalar)"
    }

    #[cfg(all(feature = "avx512", target_feature = "avx512vpopcntdq"))]
    fn ident() -> Self::Ident {
        "CpuMatcher (AVX-512)"
    }

    #[cfg(not(all(feature = "avx512", target_feature = "avx512vpopcntdq")))]
    fn scan(
        &mut self,
        query: &Self::Aligner<
            GenericArray<GenericArray<u8, Self::InputDimension>, Self::BatchSize>,
        >,
    ) -> bool {
        let mut matched = false;
        for q in &**query {
            for n in &self.needles {
                let dist = unsafe {
                    #[allow(clippy::transmute_ptr_to_ptr)]
                    core::mem::transmute::<&GenericArray<u8, U32>, &GenericArray<u64, U4>>(q)
                }
                .iter()
                .zip(n.iter())
                .map(|(a, b)| a ^ b)
                .map(|x| x.count_ones() as u64)
                .sum::<u64>();
                // benchmark shows it's not faster to be too smart and save a branch here
                if dist <= self.threshold {
                    matched = true;
                }
            }
        }
        matched
    }

    #[cfg(all(feature = "avx512", target_feature = "avx512vpopcntdq"))]
    fn scan(
        &mut self,
        query: &Self::Aligner<
            GenericArray<GenericArray<u8, Self::InputDimension>, Self::BatchSize>,
        >,
    ) -> bool {
        use core::arch::x86_64::_mm512_store_epi64;

        use crate::alignment::Align64;
        unsafe {
            #[allow(clippy::cast_possible_wrap)]
            let addend = _mm512_set1_epi64(self.threshold as i64);

            // defer the result of the match by not branching in the middle
            let mut results = _mm512_setzero_si512();

            // [^A0|^B0|^C0|^D0|^E0|^F0|^G0|^H0]  // First 64 bits of each needle
            // [^A1|^B1|^C1|^D1|^E1|^F1|^G1|^H1]  // Second 64 bits
            // [^A2|^B2|^C2|^D2|^E2|^F2|^G2|^H2]  // Third 64 bits
            // [^A3|^B3|^C3|^D3|^E3|^F3|^G3|^H3]  // Fourth 64 bits

            // we are zipping it with
            // [Q0|Q0|Q0|Q0|Q0|Q0|Q0|Q0]
            // [Q1|Q1|Q1|Q1|Q1|Q1|Q1|Q1]
            // [Q2|Q2|Q2|Q2|Q2|Q2|Q2|Q2]
            // [Q3|Q3|Q3|Q3|Q3|Q3|Q3|Q3]

            for i in 0..2048 {
                // we forced alignment on type level, this is guaranteed to be aligned
                #[allow(clippy::cast_ptr_alignment)]
                let queries = [
                    _mm512_set1_epi64(query[i].as_ptr().cast::<i64>().add(0).read()),
                    _mm512_set1_epi64(query[i].as_ptr().cast::<i64>().add(1).read()),
                    _mm512_set1_epi64(query[i].as_ptr().cast::<i64>().add(2).read()),
                    _mm512_set1_epi64(query[i].as_ptr().cast::<i64>().add(3).read()),
                ];

                // LLVM can be smarter and shave off 1 or 2 instructions here,
                // uses vpternlogq, we don't have to be too smart here
                let differences = [
                    _mm512_xor_si512(self.needles_comp[0], queries[0]),
                    _mm512_xor_si512(self.needles_comp[1], queries[1]),
                    _mm512_xor_si512(self.needles_comp[2], queries[2]),
                    _mm512_xor_si512(self.needles_comp[3], queries[3]),
                ];

                let counts = [
                    _mm512_popcnt_epi64(differences[0]),
                    _mm512_popcnt_epi64(differences[1]),
                    _mm512_popcnt_epi64(differences[2]),
                    _mm512_popcnt_epi64(differences[3]),
                ];

                let count1 = _mm512_add_epi64(counts[0], counts[1]);
                let count2 = _mm512_add_epi64(counts[2], counts[3]);
                let count3 = _mm512_add_epi64(count1, count2);

                results = _mm512_or_si512(results, _mm512_add_epi64(count3, addend));
            }

            let mut output: Align64<[u64; 8]> = Align64::default();

            _mm512_store_epi64(output.as_mut_ptr().cast(), results);

            output.iter().any(|x| (*x & 256) != 0)
        }
    }

    #[cfg(not(all(feature = "avx512", target_feature = "avx512vpopcntdq")))]
    fn find<R, F: FnMut(usize, usize) -> Option<R>>(
        &mut self,
        query: &Self::Aligner<
            GenericArray<GenericArray<u8, Self::InputDimension>, Self::BatchSize>,
        >,
        mut f: F,
    ) -> Option<R> {
        for (i, q) in query.iter().enumerate() {
            for (j, n) in self.needles.iter().enumerate() {
                #[allow(clippy::transmute_ptr_to_ptr)]
                let dist = unsafe {
                    core::mem::transmute::<&GenericArray<u8, U32>, &GenericArray<u64, U4>>(q)
                }
                .iter()
                .zip(n.iter())
                .map(|(a, b)| a ^ b)
                .map(|x| x.count_ones() as u64)
                .sum::<u64>();
                if dist <= self.threshold {
                    if let Some(r) = f(i, j) {
                        return Some(r);
                    }
                }
            }
        }
        None
    }

    #[cfg(all(feature = "avx512", target_feature = "avx512vpopcntdq"))]
    fn find<R, F: FnMut(usize, usize) -> Option<R>>(
        &mut self,
        query: &Self::Aligner<
            GenericArray<GenericArray<u8, Self::InputDimension>, Self::BatchSize>,
        >,
        mut f: F,
    ) -> Option<R> {
        unsafe {
            #[allow(clippy::cast_possible_wrap)]
            let threshold = _mm512_set1_epi64((256 - self.threshold) as i64);

            // the needles are pre-pivot and pre-complemented, i.e:
            // [^A0|^B0|^C0|^D0|^E0|^F0|^G0|^H0]  // First 64 bits of each needle
            // [^A1|^B1|^C1|^D1|^E1|^F1|^G1|^H1]  // Second 64 bits
            // [^A2|^B2|^C2|^D2|^E2|^F2|^G2|^H2]  // Third 64 bits
            // [^A3|^B3|^C3|^D3|^E3|^F3|^G3|^H3]  // Fourth 64 bits

            // we are zipping it with
            // [Q0|Q0|Q0|Q0|Q0|Q0|Q0|Q0]
            // [Q1|Q1|Q1|Q1|Q1|Q1|Q1|Q1]
            // [Q2|Q2|Q2|Q2|Q2|Q2|Q2|Q2]
            // [Q3|Q3|Q3|Q3|Q3|Q3|Q3|Q3]

            // The query
            #[allow(clippy::cast_ptr_alignment)]
            for i in 0..2048 {
                let decomposed: [u64; 4] = query[i].as_ptr().cast::<[u64; 4]>().read();

                #[allow(clippy::cast_possible_wrap)]
                let queries = [
                    _mm512_set1_epi64(decomposed[0] as i64),
                    _mm512_set1_epi64(decomposed[1] as i64),
                    _mm512_set1_epi64(decomposed[2] as i64),
                    _mm512_set1_epi64(decomposed[3] as i64),
                ];

                let differences = [
                    _mm512_xor_si512(self.needles_comp[0], queries[0]),
                    _mm512_xor_si512(self.needles_comp[1], queries[1]),
                    _mm512_xor_si512(self.needles_comp[2], queries[2]),
                    _mm512_xor_si512(self.needles_comp[3], queries[3]),
                ];

                let counts = [
                    _mm512_popcnt_epi64(differences[0]),
                    _mm512_popcnt_epi64(differences[1]),
                    _mm512_popcnt_epi64(differences[2]),
                    _mm512_popcnt_epi64(differences[3]),
                ];

                let count1 = _mm512_add_epi64(counts[0], counts[1]);
                let count2 = _mm512_add_epi64(counts[2], counts[3]);
                let count3 = _mm512_add_epi64(count1, count2);

                let mut matched = _mm512_cmpge_epi64_mask(count3, threshold);
                while matched != 0 {
                    let needle_idx = _tzcnt_u16(matched as u16);
                    if let Some(result) = f(i, needle_idx as usize) {
                        return Some(result);
                    }
                    matched ^= 1 << needle_idx;
                }
            }

            None
        }
    }
}

#[cfg(test)]
mod tests {
    use generic_array::typenum::{U32, U2048};
    use rand::prelude::*;

    use super::*;

    #[test]
    fn test_scan_cpu() {
        let mut rng = rand::rng();

        // Generate random needles
        let mut needles_data = GenericArray::<GenericArray<u8, U32>, U8>::default();
        for needle in needles_data.iter_mut() {
            rng.fill_bytes(needle);
        }

        // Generate random batch data
        let mut haystack_data = Align8::<GenericArray<GenericArray<u8, U32>, U2048>>::default();
        for vector in haystack_data.iter_mut() {
            rng.fill_bytes(vector);
        }

        fn generate_positive_control<L: ArrayLength, R: RngCore>(
            rng: &mut R,
            reference: &GenericArray<u8, L>,
            output: &mut GenericArray<u8, L>,
            distance: u32,
        ) {
            for (i, needle) in output.iter_mut().enumerate() {
                *needle = reference[i];
            }

            for _ in 0..distance {
                let position_byte = rng.random_range(0..L::USIZE);
                let position_bit = rng.random_range(0..8);
                output[position_byte] ^= 1 << position_bit;
            }
        }

        // Compare results, initially there are no matches
        let mut kernel = CpuMatcher::new(&needles_data, 31);
        assert_eq!(kernel.scan(&haystack_data), false);

        kernel.find(&haystack_data, |_, i| {
            panic!("Found match at index {}", i);
            #[allow(unreachable_code)]
            Some(i)
        });

        // Insert a few matches to ensure we're testing the matching logic
        // Copy a few needles into the batch at random positions
        // Try 1000x8 times just to make sure we didn't have a blindspot or something
        for _test in 0..1000 {
            for i in 0..8 {
                let pos = rng.random_range(0..2048);
                generate_positive_control(&mut rng, &haystack_data[pos], &mut needles_data[i], 20);

                let mut kernel = CpuMatcher::new(&needles_data, 31);
                // Compare results again
                assert_eq!(kernel.scan(&haystack_data), true);

                let result = kernel.find(&haystack_data, |j, i| {
                    assert_eq!(pos as usize, j);
                    Some(i)
                });

                assert_eq!(result, Some(i));

                // reset the needle
                rng.fill_bytes(&mut needles_data[i]);
            }
        }

        // Compare results, make sure finally there are no matches
        let mut kernel = CpuMatcher::new(&needles_data, 31);
        assert_eq!(kernel.scan(&haystack_data), false);

        kernel.find(&haystack_data, |_, i| {
            panic!("Found match at index {}", i);
            #[allow(unreachable_code)]
            Some(i)
        });
    }
}
