/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Tuesday, March 25, 2025
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
    cmp::Ordering,
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};
use num_traits::{Bounded, FromPrimitive, Num, NumCast, One, ToPrimitive, float::TotalOrder};
use rug::Assign;

#[cfg(feature = "reference-rug")]
#[derive(Debug, Clone, PartialEq, PartialOrd)]
/// arbitrary-precision floating point type.
pub struct ArbFloat<const C: u32 = 96>(rug::Float);

impl<const C: u32> TotalOrder for ArbFloat<C> {
    fn total_cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl<const C: u32> Add<ArbFloat<C>> for ArbFloat<C> {
    type Output = Self;
    fn add(self, rhs: ArbFloat<C>) -> Self::Output {
        Self(self.0.clone() + rhs.0)
    }
}

impl<const C: u32> AddAssign<ArbFloat<C>> for ArbFloat<C> {
    fn add_assign(&mut self, rhs: ArbFloat<C>) {
        self.0 += rhs.0;
    }
}

impl<const C: u32> Sub<ArbFloat<C>> for ArbFloat<C> {
    type Output = Self;
    fn sub(self, rhs: ArbFloat<C>) -> Self::Output {
        Self(self.0.clone() - rhs.0)
    }
}

impl<const C: u32> SubAssign<ArbFloat<C>> for ArbFloat<C> {
    fn sub_assign(&mut self, rhs: ArbFloat<C>) {
        self.0 -= rhs.0;
    }
}

impl<const C: u32> NumCast for ArbFloat<C> {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        let mut ret = rug::Float::new(C);
        ret.assign(n.to_f64().unwrap());
        Some(Self(ret))
    }
}

impl<const C: u32> ArbFloat<C> {
    /// Convert to 32-bit floating point type.
    #[must_use]
    pub fn to_f32(&self) -> f32 {
        self.0.to_f32()
    }

    /// Convert to 64-bit floating point type.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }

    /// Square root.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        Self(self.0.clone().sqrt())
    }

    /// Cosine.
    #[must_use]
    pub fn cos(&self) -> Self {
        Self(self.0.clone().cos())
    }

    /// Pi.
    #[must_use]
    pub fn pi() -> Self {
        let mut pi = rug::Float::new(C);
        pi.acos_mut();
        pi *= 2;
        debug_assert!(
            pi.to_f64() == std::f64::consts::PI,
            "pi: {:?}, std::f64::consts::PI: {:?}",
            pi.to_f64(),
            std::f64::consts::PI
        );
        Self(pi)
    }
}

macro_rules! impl_from_primitive {
    ($($name:ident(n: $t:ty)),*) => {
        $(
            fn $name(n: $t) -> Option<Self> {
                let mut ret = rug::Float::new(C);
                ret.assign(n);
                Some(Self(ret))
            }
        )*
    };
}

macro_rules! impl_to_primitive {
    ($($name:ident as $t:ty),*) => {
        $(
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            fn $name(&self) -> Option<$t> {
                let val = self.0.to_f64();
                if val.is_nan() {
                    None
                } else {
                    Some(val as $t)
                }
            }
        )*
    };
}

impl<const C: u32> FromPrimitive for ArbFloat<C> {
    impl_from_primitive!(
        from_f32(n: f32), from_f64(n: f64),
        from_i8(n: i8), from_i16(n: i16),
        from_i32(n: i32), from_i64(n: i64), from_i128(n: i128), from_isize(n: isize), from_u8(n: u8), from_u16(n: u16), from_u32(n: u32), from_u64(n: u64), from_u128(n: u128));
}

impl<const C: u32> ToPrimitive for ArbFloat<C> {
    impl_to_primitive!(
        to_f32 as f32,
        to_f64 as f64,
        to_i8 as i8,
        to_i16 as i16,
        to_i32 as i32,
        to_i64 as i64,
        to_i128 as i128,
        to_isize as isize,
        to_u8 as u8,
        to_u16 as u16,
        to_u32 as u32,
        to_u64 as u64,
        to_u128 as u128
    );
}

impl<const C: u32> Bounded for ArbFloat<C> {
    fn min_value() -> Self {
        let mut ret = rug::Float::new(C);
        ret -= 1.0;
        loop {
            let new_ret = ret.clone() + ret.clone();
            if new_ret.is_finite() {
                ret = new_ret;
            } else {
                return Self(ret);
            }
        }
    }
    fn max_value() -> Self {
        let mut ret = rug::Float::new(C);
        ret += 1.0;
        loop {
            let new_ret = ret.clone() + ret.clone();
            if new_ret.is_finite() {
                ret = new_ret;
            } else {
                return Self(ret);
            }
        }
    }
}

impl<const C: u32> Default for ArbFloat<C> {
    fn default() -> Self {
        Self(rug::Float::new(C))
    }
}

impl<const C: u32> Display for ArbFloat<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<const C: u32> Neg for ArbFloat<C> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<const C: u32> Mul for ArbFloat<C> {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self(self.0 * other.0)
    }
}

impl<const C: u32> MulAssign for ArbFloat<C> {
    fn mul_assign(&mut self, other: Self) {
        self.0 *= other.0;
    }
}

impl<const C: u32> Div for ArbFloat<C> {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self(self.0 / other.0)
    }
}

impl<const C: u32> DivAssign for ArbFloat<C> {
    fn div_assign(&mut self, other: Self) {
        self.0 /= other.0;
    }
}

impl<const C: u32> One for ArbFloat<C> {
    fn one() -> Self {
        let mut ret = rug::Float::new(C);
        ret += 1;
        Self(ret)
    }
}

impl<const C: u32> Rem for ArbFloat<C> {
    type Output = Self;

    fn rem(self, other: Self) -> Self::Output {
        Self(self.0 % other.0)
    }
}

impl<const C: u32> RemAssign for ArbFloat<C> {
    fn rem_assign(&mut self, other: Self) {
        self.0 %= other.0;
    }
}

impl<const C: u32> num_traits::identities::Zero for ArbFloat<C> {
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn zero() -> Self {
        Self::default()
    }
}

impl<const C: u32> Num for ArbFloat<C> {
    type FromStrRadixErr = rug::float::ParseFloatError;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let f = rug::Float::parse_radix(str, radix.try_into().unwrap())?;
        Ok(Self(rug::Float::with_val(C, f)))
    }
}

impl<const C: u32> num_traits::Signed for ArbFloat<C> {
    fn signum(&self) -> Self {
        Self(self.0.clone().signum())
    }

    fn abs(&self) -> Self {
        Self(self.0.clone().abs())
    }

    fn abs_sub(&self, other: &Self) -> Self {
        Self(self.0.clone().positive_diff(&other.0))
    }

    fn is_negative(&self) -> bool {
        self.0.clone().is_sign_negative()
    }

    fn is_positive(&self) -> bool {
        self.0.clone().is_sign_positive()
    }
}
