/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Wednesday, March 26, 2025
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
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::Mul,
};

use generic_array::{
    ArrayLength, GenericArray,
    typenum::{B0, B1, Bit, U3, U4},
};

use super::{
    Kernel,
    type_traits::{DivisibleBy8, EvaluateHardwareFeature, FallbackRequirements, SquareOf},
};

#[derive(Debug, Clone, Copy, PartialEq)]
/// A token that has both a compile-time bit and a runtime-bit indicating whether a decision fell through to the fallback kernel.
pub struct MaybeFellThroughToken<EnabledStatically: Bit> {
    _private: PhantomData<EnabledStatically>,
    /// Whether the decision fell through to the fallback kernel at runtime.
    pub fell_through_runtime: bool,
}

trait KernelFallthrough<
    EnabledStatically: Bit,
    InputDimension: ArrayLength,
    Buffer1WidthX: ArrayLength,
    Buffer1LengthY: ArrayLength,
    OutputDimension: ArrayLength + DivisibleBy8,
    InternalFloat: num_traits::float::TotalOrder
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + num_traits::bounds::Bounded
        + num_traits::NumCast,
>:
    Kernel<
        Buffer1WidthX = Buffer1WidthX,
        Buffer1LengthY = Buffer1LengthY,
        InputDimension = InputDimension,
        OutputDimension = OutputDimension,
        InternalFloat = InternalFloat,
    >
{
    fn ident_opt(&self) -> MaybeFellThroughToken<EnabledStatically>;

    fn would_run(&self) -> bool {
        false
    }

    fn pdqf_t_opt<const CHECKED: bool>(
        &mut self,
        _input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> bool {
        false
    }

    fn pdqf_negate_alt_cols_opt<const NEGATE: bool, const CHECKED: bool>(
        &mut self,
        _input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> bool {
        false
    }

    fn pdqf_negate_alt_rows_opt<const NEGATE: bool, const CHECKED: bool>(
        &mut self,
        _input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> bool {
        false
    }

    fn pdqf_negate_off_diagonals_opt<const CHECKED: bool>(
        &mut self,
        _input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> bool {
        false
    }

    fn cvt_rgb8_to_luma8f_opt<
        const CHECKED: bool,
        const R_COEFF: u32,
        const G_COEFF: u32,
        const B_COEFF: u32,
    >(
        &mut self,
        _input: &GenericArray<GenericArray<u8, U3>, InputDimension>,
        _output: &mut GenericArray<f32, InputDimension>,
    ) -> bool {
        false
    }

    fn cvt_rgba8_to_luma8f_opt<
        const CHECKED: bool,
        const R_COEFF: u32,
        const G_COEFF: u32,
        const B_COEFF: u32,
    >(
        &mut self,
        _input: &GenericArray<GenericArray<u8, U4>, InputDimension>,
        _output: &mut GenericArray<f32, InputDimension>,
    ) -> bool {
        false
    }

    fn jarosz_compress_opt<const CHECKED: bool>(
        &mut self,
        _buffer: &GenericArray<GenericArray<f32, InputDimension>, InputDimension>,
        _output: &mut GenericArray<GenericArray<InternalFloat, Buffer1WidthX>, Buffer1LengthY>,
    ) -> bool {
        false
    }

    fn quantize_opt<const CHECKED: bool>(
        &mut self,
        _input: &GenericArray<GenericArray<InternalFloat, OutputDimension>, OutputDimension>,
        _threshold: &mut InternalFloat,
        _output: &mut GenericArray<
            GenericArray<u8, <OutputDimension as DivisibleBy8>::Output>,
            OutputDimension,
        >,
    ) -> bool {
        false
    }

    fn dct2d_opt<const CHECKED: bool>(
        &mut self,
        _buffer: &GenericArray<GenericArray<InternalFloat, Buffer1WidthX>, Buffer1LengthY>,
        _tmp_row_buffer: &mut GenericArray<InternalFloat, Buffer1WidthX>,
        _output: &mut GenericArray<GenericArray<InternalFloat, OutputDimension>, OutputDimension>,
    ) -> bool {
        false
    }

    fn sum_of_gradients_opt<const CHECKED: bool>(
        &mut self,
        _input: &GenericArray<GenericArray<InternalFloat, OutputDimension>, OutputDimension>,
    ) -> Option<InternalFloat> {
        None
    }
}

impl<M: EvaluateHardwareFeature<EnabledStatic = B0>, P: Kernel<RequiredHardwareFeature = M>>
    KernelFallthrough<
        B0,
        <P as Kernel>::InputDimension,
        <P as Kernel>::Buffer1WidthX,
        <P as Kernel>::Buffer1LengthY,
        <P as Kernel>::OutputDimension,
        <P as Kernel>::InternalFloat,
    > for P
where
    <P as Kernel>::OutputDimension: DivisibleBy8,
{
    fn ident_opt(&self) -> MaybeFellThroughToken<B0> {
        MaybeFellThroughToken {
            _private: PhantomData,
            fell_through_runtime: true,
        }
    }
}

impl<M: EvaluateHardwareFeature<EnabledStatic = B1>, P: Kernel<RequiredHardwareFeature = M>>
    KernelFallthrough<
        B1,
        <P as Kernel>::InputDimension,
        <P as Kernel>::Buffer1WidthX,
        <P as Kernel>::Buffer1LengthY,
        <P as Kernel>::OutputDimension,
        <P as Kernel>::InternalFloat,
    > for P
where
    <P as Kernel>::OutputDimension: DivisibleBy8,
{
    fn would_run(&self) -> bool {
        // must be enabled statically and either checking was statically not required or runtime check passed
        <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL &&
            (!<<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::MustCheck::BOOL
                || <P as Kernel>::RequiredHardwareFeature::met_runtime())
    }

    fn cvt_rgb8_to_luma8f_opt<
        const CHECKED: bool,
        const R_COEFF: u32,
        const G_COEFF: u32,
        const B_COEFF: u32,
    >(
        &mut self,
        input: &GenericArray<GenericArray<u8, U3>, <P as Kernel>::InputDimension>,
        output: &mut GenericArray<f32, <P as Kernel>::InputDimension>,
    ) -> bool {
        if CHECKED && !self.would_run() {
            return false;
        }

        self.cvt_rgb8_to_luma8f::<R_COEFF, G_COEFF, B_COEFF>(input, output);
        true
    }

    fn cvt_rgba8_to_luma8f_opt<
        const CHECKED: bool,
        const R_COEFF: u32,
        const G_COEFF: u32,
        const B_COEFF: u32,
    >(
        &mut self,
        input: &GenericArray<GenericArray<u8, U4>, <P as Kernel>::InputDimension>,
        output: &mut GenericArray<f32, <P as Kernel>::InputDimension>,
    ) -> bool {
        if CHECKED && !self.would_run() {
            return false;
        }

        self.cvt_rgba8_to_luma8f::<R_COEFF, G_COEFF, B_COEFF>(input, output);
        true
    }

    fn pdqf_negate_alt_cols_opt<const NEGATE: bool, const CHECKED: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> bool {
        if CHECKED && !self.would_run() {
            return false;
        }

        self.pdqf_negate_alt_cols::<NEGATE>(input);
        true
    }

    fn pdqf_negate_alt_rows_opt<const NEGATE: bool, const CHECKED: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> bool {
        if CHECKED && !self.would_run() {
            return false;
        }

        self.pdqf_negate_alt_rows::<NEGATE>(input);
        true
    }

    fn pdqf_negate_off_diagonals_opt<const CHECKED: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> bool {
        if CHECKED && !self.would_run() {
            return false;
        }

        self.pdqf_negate_off_diagonals(input);
        true
    }

    fn pdqf_t_opt<const CHECKED: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> bool {
        if CHECKED && !self.would_run() {
            return false;
        }

        self.pdqf_t(input);
        true
    }

    fn jarosz_compress_opt<const CHECKED: bool>(
        &mut self,
        buffer: &GenericArray<
            GenericArray<f32, <P as Kernel>::InputDimension>,
            <P as Kernel>::InputDimension,
        >,
        output: &mut GenericArray<
            GenericArray<<P as Kernel>::InternalFloat, <P as Kernel>::Buffer1WidthX>,
            <P as Kernel>::Buffer1LengthY,
        >,
    ) -> bool {
        if CHECKED && !self.would_run() {
            return false;
        }

        self.jarosz_compress(buffer, output);
        true
    }

    fn ident_opt(
        &self,
    ) -> MaybeFellThroughToken<
        <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic,
    > {
        if !self.would_run() {
            return MaybeFellThroughToken {
                _private: PhantomData,
                fell_through_runtime: true,
            };
        }

        MaybeFellThroughToken {
            _private: PhantomData,
            fell_through_runtime: false,
        }
    }

    fn quantize_opt<const CHECKED: bool>(
        &mut self,
        input: &GenericArray<
            GenericArray<<P as Kernel>::InternalFloat, <P as Kernel>::OutputDimension>,
            <P as Kernel>::OutputDimension,
        >,
        threshold: &mut <P as Kernel>::InternalFloat,
        output: &mut GenericArray<
            GenericArray<u8, <<P as Kernel>::OutputDimension as DivisibleBy8>::Output>,
            <P as Kernel>::OutputDimension,
        >,
    ) -> bool {
        if CHECKED && !self.would_run() {
            return false;
        }

        self.quantize(input, threshold, output);
        true
    }

    fn dct2d_opt<const CHECKED: bool>(
        &mut self,
        buffer: &GenericArray<
            GenericArray<<P as Kernel>::InternalFloat, <P as Kernel>::Buffer1WidthX>,
            <P as Kernel>::Buffer1LengthY,
        >,
        tmp_row_buffer: &mut GenericArray<
            <P as Kernel>::InternalFloat,
            <P as Kernel>::Buffer1WidthX,
        >,
        output: &mut GenericArray<
            GenericArray<<P as Kernel>::InternalFloat, <P as Kernel>::OutputDimension>,
            <P as Kernel>::OutputDimension,
        >,
    ) -> bool {
        if CHECKED && !self.would_run() {
            return false;
        }

        self.dct2d(buffer, tmp_row_buffer, output);
        true
    }

    fn sum_of_gradients_opt<const CHECKED: bool>(
        &mut self,
        input: &GenericArray<
            GenericArray<<P as Kernel>::InternalFloat, <P as Kernel>::OutputDimension>,
            <P as Kernel>::OutputDimension,
        >,
    ) -> Option<P::InternalFloat> {
        if CHECKED && !self.would_run() {
            return None;
        }

        Some(self.sum_of_gradients(input))
    }
}

/// A static fallback router for composing kernels.
#[derive(Debug, Clone, Copy, Default)]
pub struct KernelRouter<P, F> {
    materialized_decision: bool,
    preferred: P,
    fallback: F,
}

impl<
    Buffer1WidthX: ArrayLength,
    Buffer1LengthY: ArrayLength,
    InputDimension: ArrayLength + SquareOf,
    OutputDimension: ArrayLength + SquareOf + Mul<OutputDimension> + DivisibleBy8,
    InternalFloat: num_traits::float::TotalOrder
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + num_traits::bounds::Bounded
        + num_traits::NumCast
        + num_traits::identities::Zero
        + num_traits::identities::One
        + num_traits::Signed
        + PartialOrd
        + Clone
        + Display
        + Debug
        + Default
        + Send
        + Sync,
    P: Kernel<
            Buffer1WidthX = Buffer1WidthX,
            Buffer1LengthY = Buffer1LengthY,
            InputDimension = InputDimension,
            OutputDimension = OutputDimension,
            InternalFloat = InternalFloat,
        >,
    F: Kernel<
            Buffer1WidthX = Buffer1WidthX,
            Buffer1LengthY = Buffer1LengthY,
            InputDimension = InputDimension,
            OutputDimension = OutputDimension,
            InternalFloat = InternalFloat,
        >,
> KernelRouter<P, F>
where
    P::RequiredHardwareFeature: EvaluateHardwareFeature,
    <OutputDimension as Mul<OutputDimension>>::Output: ArrayLength,
    <P as Kernel>::OutputDimension: DivisibleBy8,
    <F as Kernel>::OutputDimension: DivisibleBy8,
{
    /// Create a new kernel router.
    ///
    /// Fallback kernel must be guaranteed to be available at runtime.
    pub fn new(preferred: P, fallback: F) -> Self
    where
        F::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1, MustCheck = B0>,
    {
        // must be enabled statically and either checking was statically not required or runtime check passed
        let decision = <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL &&
            (!<<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::MustCheck::BOOL
                || <P as Kernel>::RequiredHardwareFeature::met_runtime());

        Self {
            materialized_decision: decision,
            preferred,
            fallback,
        }
    }

    /// Layer a new kernel on top of the current kernel router.
    ///
    /// The new kernel will be used as the preferred kernel, and the current kernel router will be used as the secondary fallback kernel.
    pub fn layer_on_top<
        P2: Kernel<
                Buffer1WidthX = Buffer1WidthX,
                Buffer1LengthY = Buffer1LengthY,
                InputDimension = InputDimension,
                OutputDimension = OutputDimension,
                InternalFloat = InternalFloat,
            >,
    >(
        self,
        kernel: P2,
    ) -> KernelRouter<P2, Self> {
        let decision =
            !<<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::MustCheck::BOOL
                || <P as Kernel>::RequiredHardwareFeature::met_runtime();

        KernelRouter {
            materialized_decision: decision,
            preferred: kernel,
            fallback: self,
        }
    }
}

#[derive(Clone, Copy)]
/// An identification token for a runtime decision to use a fallback kernel.
pub struct FallbackToken<
    PE: Bit,
    PIdent: Debug + Display + Clone + Copy + 'static + PartialEq,
    FIdent: Debug + Display + Clone + Copy + 'static + PartialEq,
> {
    preferred: PIdent,
    fall_through: MaybeFellThroughToken<PE>,
    fallback: FIdent,
}

impl<
    PE: Bit,
    PIdent: Debug + Display + Clone + Copy + 'static + PartialEq,
    FIdent: Debug + Display + Clone + Copy + 'static + PartialEq,
> Debug for FallbackToken<PE, PIdent, FIdent>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if PE::BOOL {
            if self.fall_through.fell_through_runtime {
                write!(
                    f,
                    "FallbackIdent(decision: fallback (reason: runtime hardware feature not met) {:?}, preferred was {:?})",
                    self.fallback, self.preferred
                )
            } else {
                write!(
                    f,
                    "FallbackIdent(decision: using preferred {:?}, fallback was {:?})",
                    self.preferred, self.fallback
                )
            }
        } else {
            write!(
                f,
                "FallbackIdent(decision: fallback (reason: compile time flag not met) {:?}, preferred was {:?})",
                self.fallback, self.preferred
            )
        }
    }
}

impl<
    PE: Bit,
    PIdent: Debug + Display + Clone + Copy + 'static + PartialEq,
    FIdent: Debug + Display + Clone + Copy + 'static + PartialEq,
> Display for FallbackToken<PE, PIdent, FIdent>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if PE::BOOL && !self.fall_through.fell_through_runtime {
            write!(f, "{}", self.preferred)
        } else {
            write!(f, "{}", self.fallback)
        }
    }
}

impl<
    PE: Bit,
    PIdent: Debug + Display + Clone + Copy + 'static + PartialEq,
    FIdent: Debug + Display + Clone + Copy + 'static + PartialEq,
> PartialEq for FallbackToken<PE, PIdent, FIdent>
{
    fn eq(&self, other: &Self) -> bool {
        self.preferred == other.preferred
            && self.fallback == other.fallback
            && self.fall_through.fell_through_runtime == other.fall_through.fell_through_runtime
    }
}

impl<
    Buffer1WidthX: ArrayLength,
    Buffer1LengthY: ArrayLength,
    InputDimension: ArrayLength + SquareOf,
    OutputDimension: ArrayLength + SquareOf + Mul<OutputDimension> + DivisibleBy8,
    InternalFloat: num_traits::float::TotalOrder
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + num_traits::bounds::Bounded
        + num_traits::NumCast
        + num_traits::identities::Zero
        + num_traits::identities::One
        + num_traits::Signed
        + PartialOrd
        + Clone
        + Display
        + Debug
        + Default
        + Send
        + Sync,
    P: Kernel<
            Buffer1WidthX = Buffer1WidthX,
            Buffer1LengthY = Buffer1LengthY,
            InputDimension = InputDimension,
            OutputDimension = OutputDimension,
            InternalFloat = InternalFloat,
        > + KernelFallthrough<
            <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic,
            InputDimension,
            Buffer1WidthX,
            Buffer1LengthY,
            OutputDimension,
            InternalFloat,
        >,
    F: Kernel<
            Buffer1WidthX = Buffer1WidthX,
            Buffer1LengthY = Buffer1LengthY,
            InputDimension = InputDimension,
            OutputDimension = OutputDimension,
            InternalFloat = InternalFloat,
        >,
> Kernel for KernelRouter<P, F>
where
    P::RequiredHardwareFeature: EvaluateHardwareFeature,
    F::RequiredHardwareFeature: EvaluateHardwareFeature,
    <OutputDimension as Mul<OutputDimension>>::Output: ArrayLength,
    P::RequiredHardwareFeature: EvaluateHardwareFeature,
    F::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1, MustCheck = B0>,
{
    type InternalFloat = InternalFloat;
    type Buffer1WidthX = Buffer1WidthX;
    type Buffer1LengthY = Buffer1LengthY;
    type InputDimension = InputDimension;
    type OutputDimension = OutputDimension;
    type RequiredHardwareFeature =
        FallbackRequirements<P::RequiredHardwareFeature, F::RequiredHardwareFeature>;
    type Ident = FallbackToken<
        <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic,
        <P as Kernel>::Ident,
        <F as Kernel>::Ident,
    >;

    fn ident(&self) -> Self::Ident {
        let token = self.preferred.ident_opt();
        Self::Ident {
            preferred: self.preferred.ident(),
            fall_through: token,
            fallback: self.fallback.ident(),
        }
    }

    fn required_hardware_features_met() -> bool {
        Self::RequiredHardwareFeature::met_runtime()
    }

    fn cvt_rgb8_to_luma8f<const R_COEFF: u32, const G_COEFF: u32, const B_COEFF: u32>(
        &mut self,
        input: &GenericArray<GenericArray<u8, U3>, Self::InputDimension>,
        output: &mut GenericArray<f32, Self::InputDimension>,
    ) {
        if self.materialized_decision && <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL && self.preferred.cvt_rgb8_to_luma8f_opt::<false, R_COEFF, G_COEFF, B_COEFF>(input, output) {
            return;
        }

        self.fallback
            .cvt_rgb8_to_luma8f::<R_COEFF, G_COEFF, B_COEFF>(input, output);
    }

    fn cvt_rgba8_to_luma8f<const R_COEFF: u32, const G_COEFF: u32, const B_COEFF: u32>(
        &mut self,
        input: &GenericArray<GenericArray<u8, U4>, Self::InputDimension>,
        output: &mut GenericArray<f32, Self::InputDimension>,
    ) {
        if self.materialized_decision && <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL && self.preferred.cvt_rgba8_to_luma8f_opt::<false, R_COEFF, G_COEFF, B_COEFF>(input, output) {
            return;
        }

        self.fallback
            .cvt_rgba8_to_luma8f::<R_COEFF, G_COEFF, B_COEFF>(input, output);
    }

    fn pdqf_negate_alt_cols<const NEGATE: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        if self.materialized_decision && <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL && self.preferred.pdqf_negate_alt_cols_opt::<NEGATE, false>(input) {
            return;
        }

        self.fallback.pdqf_negate_alt_cols::<NEGATE>(input);
    }

    fn pdqf_negate_alt_rows<const NEGATE: bool>(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        if self.materialized_decision && <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL && self.preferred.pdqf_negate_alt_rows_opt::<NEGATE, false>(input) {
            return;
        }

        self.fallback.pdqf_negate_alt_rows::<NEGATE>(input);
    }

    fn pdqf_negate_off_diagonals(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        if self.materialized_decision && <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL && self.preferred.pdqf_negate_off_diagonals_opt::<false>(input) {
            return;
        }

        self.fallback.pdqf_negate_off_diagonals(input);
    }

    fn pdqf_t(
        &mut self,
        input: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        if self.materialized_decision && <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL && self.preferred.pdqf_t_opt::<false>(input) {
            return;
        }

        self.fallback.pdqf_t(input);
    }

    fn jarosz_compress(
        &mut self,
        buffer: &GenericArray<GenericArray<f32, Self::InputDimension>, Self::InputDimension>,
        output: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
    ) {
        if self.materialized_decision && <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL && self.preferred.jarosz_compress_opt::<false>(buffer, output) {
            return;
        }

        self.fallback.jarosz_compress(buffer, output);
    }

    fn quantize(
        &mut self,
        input: &GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
        threshold: &mut Self::InternalFloat,
        output: &mut GenericArray<
            GenericArray<u8, <Self::OutputDimension as DivisibleBy8>::Output>,
            Self::OutputDimension,
        >,
    ) {
        if self.materialized_decision && <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL && self
                .preferred
                .quantize_opt::<false>(input, threshold, output) {
            return;
        }

        self.fallback.quantize(input, threshold, output);
    }

    /// Compute the sum of gradients of the input buffer in both horizontal and vertical directions.
    fn sum_of_gradients(
        &mut self,
        input: &GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) -> Self::InternalFloat {
        if self.materialized_decision && <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL {
            if let Some(sum) = self.preferred.sum_of_gradients_opt::<false>(input) {
                return sum;
            }
        }

        self.fallback.sum_of_gradients(input)
    }

    fn adjust_quality(input: Self::InternalFloat) -> f32 {
        <F as Kernel>::adjust_quality(input)
    }

    fn dct2d(
        &mut self,
        buffer: &GenericArray<
            GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
            Self::Buffer1LengthY,
        >,
        tmp_row_buffer: &mut GenericArray<Self::InternalFloat, Self::Buffer1WidthX>,
        output: &mut GenericArray<
            GenericArray<Self::InternalFloat, Self::OutputDimension>,
            Self::OutputDimension,
        >,
    ) {
        if self.materialized_decision && <<P as Kernel>::RequiredHardwareFeature as EvaluateHardwareFeature>::EnabledStatic::BOOL {
            self.preferred.dct2d_opt::<false>(buffer, tmp_row_buffer, output);
            return;
        }

        self.fallback.dct2d(buffer, tmp_row_buffer, output);
    }
}
