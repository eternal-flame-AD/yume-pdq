/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Thursday, March 27, 2025
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
    fmt::Debug,
    marker::PhantomData,
    ops::{BitAnd, BitOr, Mul},
};

use generic_array::{
    ArrayLength,
    typenum::{B0, B1, Bit, UInt},
};
use kernel_sealing::KernelSealed;

mod sealing {
    pub trait Sealed {}
}

impl<U, B: Bit> sealing::Sealed for UInt<U, B> {}

/// A type-level LUT for squaring a number.
///
/// Currently it is defined for up to 1024x1024 (the implementors are hidden from rustdoc).
pub trait SquareOf: ArrayLength + sealing::Sealed {
    /// The Squared result type.
    type Output: ArrayLength;
}

include!(concat!(env!("OUT_DIR"), "/square_generic_array.rs"));

/// Whether a number is divisible by 8.
pub trait DivisibleBy8: ArrayLength + sealing::Sealed
where
    <Self::Output as Mul<Self>>::Output: ArrayLength,
{
    /// The result after dividing by 8.
    type Output: ArrayLength + Mul<Self>;
}

impl<U: ArrayLength> DivisibleBy8 for UInt<UInt<UInt<U, B0>, B0>, B0>
where
    U: Mul<UInt<UInt<UInt<U, B0>, B0>, B0>>,
    <U as Mul<UInt<UInt<UInt<U, B0>, B0>, B0>>>::Output: ArrayLength,
{
    type Output = U;
}

pub(crate) mod kernel_sealing {
    pub trait KernelSealed {}
}

/// Type level struct to represent the compiler time requirements for a kernel.
pub struct RequireCompilerTimeHardwareFeature<Cur, Next> {
    _cur: PhantomData<Cur>,
    _next: PhantomData<Next>,
}

impl<S: KernelSealed, N: KernelSealed> KernelSealed for RequireCompilerTimeHardwareFeature<S, N> {}

/// Type level struct to represent a hardware feature guarded by a fallback that is guaranteed to be available.
pub struct FallbackRequirements<S: KernelSealed, N: KernelSealed> {
    _preferred: PhantomData<S>,
    _fallback: PhantomData<N>,
}

impl<S: KernelSealed, N: KernelSealed> KernelSealed for FallbackRequirements<S, N> {}

/// Type level trait to represent the runtime requirements for a kernel.
pub trait EvaluateHardwareFeature: KernelSealed {
    /// Whether the feature is possible to use at compile time.
    type EnabledStatic: Bit + BitOr<B0> + BitAnd<B1> + BitOr<B1> + BitAnd<B0>;
    /// Whether the feature must be checked before execution at runtime.
    type MustCheck: Bit;
    /// The name of the feature.
    type Name: Debug + Clone + Copy + 'static + PartialEq;

    /// Get the name of the requirement.
    fn name() -> Self::Name;

    /// Check if the feature is available at runtime.
    fn met_runtime() -> bool;
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// A type that represents the intersection of two names.
pub struct AndName<A, B>(A, B);

impl<F: EvaluateHardwareFeature, N: EvaluateHardwareFeature> EvaluateHardwareFeature
    for RequireCompilerTimeHardwareFeature<F, N>
where
    <F as EvaluateHardwareFeature>::EnabledStatic:
        BitAnd<<N as EvaluateHardwareFeature>::EnabledStatic>,
    <F::EnabledStatic as BitAnd<N::EnabledStatic>>::Output: Bit,
    <F as EvaluateHardwareFeature>::MustCheck: BitOr<<N as EvaluateHardwareFeature>::MustCheck>,
    <F::MustCheck as BitOr<N::MustCheck>>::Output: Bit,
    <<F as EvaluateHardwareFeature>::EnabledStatic as BitAnd<
        <N as EvaluateHardwareFeature>::EnabledStatic,
    >>::Output: BitAnd<B0>,
    <<F as EvaluateHardwareFeature>::EnabledStatic as BitAnd<
        <N as EvaluateHardwareFeature>::EnabledStatic,
    >>::Output: BitAnd<B0>,
    <<F as EvaluateHardwareFeature>::EnabledStatic as BitAnd<
        <N as EvaluateHardwareFeature>::EnabledStatic,
    >>::Output: BitAnd<B0>,
    <<F as EvaluateHardwareFeature>::EnabledStatic as BitAnd<
        <N as EvaluateHardwareFeature>::EnabledStatic,
    >>::Output: BitOr<B1>,
    <<F as EvaluateHardwareFeature>::EnabledStatic as BitAnd<
        <N as EvaluateHardwareFeature>::EnabledStatic,
    >>::Output: BitAnd<B1>,
    <<F as EvaluateHardwareFeature>::EnabledStatic as BitAnd<
        <N as EvaluateHardwareFeature>::EnabledStatic,
    >>::Output: BitOr<B0>,
{
    type Name = AndName<F::Name, N::Name>;
    type EnabledStatic = <F::EnabledStatic as BitAnd<N::EnabledStatic>>::Output;
    type MustCheck = <F::MustCheck as BitOr<N::MustCheck>>::Output;

    fn name() -> Self::Name {
        AndName(F::name(), N::name())
    }

    fn met_runtime() -> bool {
        F::met_runtime() && N::met_runtime()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// A type that represents the union of two names.
pub struct FallbackName<A, B>(A, B);

impl<P: EvaluateHardwareFeature, F: EvaluateHardwareFeature<EnabledStatic = B1>>
    EvaluateHardwareFeature for FallbackRequirements<P, F>
{
    type Name = FallbackName<P::Name, F::Name>;
    type EnabledStatic = B1;
    type MustCheck = B0;

    fn name() -> Self::Name {
        FallbackName(P::name(), F::name())
    }

    fn met_runtime() -> bool {
        P::met_runtime()
    }
}

impl EvaluateHardwareFeature for Term {
    type Name = &'static str;
    type EnabledStatic = B1;
    type MustCheck = B0;

    fn name() -> Self::Name {
        "."
    }

    fn met_runtime() -> bool {
        true
    }
}

/// A type that represents a kernel that is always available.
pub struct Term {
    _private: (),
}

impl KernelSealed for Term {}

#[cfg(test)]
mod tests {
    #![allow(dead_code, unused)]

    use generic_array::typenum::{U7, U9, U56, U81};

    use super::*;

    type TestSquareOf9 = <U9 as SquareOf>::Output;
    type ExpectedSquareOf9 = U81;
    type Test56DivisibleBy8 = <U56 as DivisibleBy8>::Output;
    type Expected56DivisibleBy8 = U7;
    const ASSERT_SQUARE_OF_9_IS_U81: PhantomData<ExpectedSquareOf9> = PhantomData::<TestSquareOf9>;
    const ASSERT_56_DIVISIBLE_BY_8_IS_U7: PhantomData<Expected56DivisibleBy8> =
        PhantomData::<Test56DivisibleBy8>;
}
