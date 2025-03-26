use core::marker::PhantomData;

use generic_array::{
    ArrayLength,
    typenum::{B0, Bit, U7, U9, U81, UInt},
};

mod sealing {
    pub trait Sealed {}
}

impl<U, B: Bit> sealing::Sealed for UInt<U, B> {}

/// A type-level LUT for squaring a number.
///
/// Currently it is defined for up to 1024x1024.
pub trait SquareOf: ArrayLength + sealing::Sealed {
    /// The Squared result type.
    type Output: ArrayLength;
}

include!(concat!(env!("OUT_DIR"), "/square_generic_array.rs"));

/// Whether a number is divisible by 8.
pub trait DivisibleBy8: ArrayLength + sealing::Sealed {
    /// The result after dividing by 8.
    type Output: ArrayLength;
}

impl<U: ArrayLength> DivisibleBy8 for UInt<UInt<UInt<U, B0>, B0>, B0> {
    type Output = U;
}

mod tests {
    #![allow(dead_code, unused)]

    use generic_array::typenum::U56;

    use super::*;

    type TestSquareOf9 = <U9 as SquareOf>::Output;
    type ExpectedSquareOf9 = U81;
    type Test56DivisibleBy8 = <U56 as DivisibleBy8>::Output;
    type Expected56DivisibleBy8 = U7;
    const ASSERT_SQUARE_OF_9_IS_U81: PhantomData<ExpectedSquareOf9> = PhantomData::<TestSquareOf9>;
    const ASSERT_56_DIVISIBLE_BY_8_IS_U7: PhantomData<Expected56DivisibleBy8> =
        PhantomData::<Test56DivisibleBy8>;
}
