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

use core::ops::{Deref, DerefMut};

use const_default::ConstDefault;
use generic_array::{ArrayLength, GenericArray};

#[repr(align(8))]
#[derive(Debug)]
/// Align the item to 8 bytes.
pub struct Align8<T>(pub T);

impl<T: Default> Default for Align8<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Deref for Align8<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align8<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Align8<T> {
    /// Convert a pointer to a `Align8<T>` to a reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 8 bytes, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_unchecked(ptr: *const T) -> *const Self {
        ptr as *const Self
    }

    /// Convert a reference to a `T` to a reference to a `Align8<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the reference is not aligned to 8 bytes.  
    pub fn from_raw(input: &T) -> &Self {
        let ptr = input as *const T;
        assert_eq!(ptr.align_offset(8), 0, "pointer is not aligned to 8 bytes");
        unsafe { &*(ptr as *const Self) }
    }

    /// Convert a pointer to a `Align8<T>` to a mutable reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 8 bytes, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_mut_unchecked(ptr: *mut T) -> *mut Self {
        ptr as *mut Self
    }

    /// Convert a reference to a `T` to a mutable reference to a `Align8<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the reference is not aligned to 8 bytes.
    pub fn from_raw_mut(input: &mut T) -> &mut Self {
        let ptr = input as *mut T;
        assert_eq!(ptr.align_offset(8), 0, "pointer is not aligned to 8 bytes");
        unsafe { &mut *(ptr as *mut Self) }
    }
}

#[repr(align(32))]
#[derive(Debug)]
/// Align the item to 32 bytes.
pub struct Align32<T>(pub T);

impl<T: Default> Default for Align32<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Deref for Align32<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align32<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Align32<T> {
    /// Convert a pointer to a `Align32<T>` to a reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 32 bytes, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_unchecked(ptr: *const T) -> *const Self {
        ptr as *const Self
    }

    /// Convert a reference to a `T` to a reference to a `Align32<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the reference is not aligned to 32 bytes.
    pub fn from_raw(input: &T) -> &Self {
        let ptr = input as *const T;
        assert_eq!(
            ptr.align_offset(32),
            0,
            "pointer is not aligned to 32 bytes"
        );
        unsafe { &*(ptr as *const Self) }
    }

    /// Convert a pointer to a `Align32<T>` to a mutable reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 32 bytes, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_mut_unchecked(ptr: *mut T) -> *mut Self {
        ptr as *mut Self
    }

    /// Convert a reference to a `T` to a mutable reference to a `Align32<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the reference is not aligned to 32 bytes.
    pub fn from_raw_mut(input: &mut T) -> &mut Self {
        let ptr = input as *mut T;
        assert_eq!(
            ptr.align_offset(32),
            0,
            "pointer is not aligned to 32 bytes"
        );
        unsafe { &mut *(ptr as *mut Self) }
    }
}

#[repr(align(64))]
/// Align the item to 64 bytes.
#[derive(Debug)]
pub struct Align64<T>(pub T);

impl<T: Default> Default for Align64<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Deref for Align64<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align64<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[repr(C)]
/// A double-ended padded array.
pub struct DefaultPaddedArray<E, L: ArrayLength, P: ArrayLength> {
    _pad0: GenericArray<E, P>,
    inner: GenericArray<E, L>,
    _pad1: GenericArray<E, P>,
}

impl<E, L: ArrayLength, P: ArrayLength> DefaultPaddedArray<E, L, P> {
    /// Get a reference to the inner array.
    pub fn as_ref(&self) -> &GenericArray<E, L> {
        &self.inner
    }

    /// Get a mutable reference to the inner array.
    pub fn as_mut(&mut self) -> &mut GenericArray<E, L> {
        &mut self.inner
    }

    /// Get a pointer to the inner array.
    pub fn as_ptr(&self) -> *const E {
        self.inner.as_ptr()
    }

    /// Get a mutable pointer to the inner array.
    pub fn as_mut_ptr(&mut self) -> *mut E {
        self.inner.as_mut_ptr()
    }
}

impl<E: ConstDefault, L: ArrayLength, P: ArrayLength> DefaultPaddedArray<E, L, P>
where
    <P as ArrayLength>::ArrayType<E>: ConstDefault,
{
    /// Create a new `DefaultPaddedArray` with the given inner array.
    pub const fn new(inner: GenericArray<E, L>) -> Self {
        Self {
            _pad0: GenericArray::const_default(),
            inner,
            _pad1: GenericArray::const_default(),
        }
    }
}

impl<E: ConstDefault, L: ArrayLength, P: ArrayLength> Deref for DefaultPaddedArray<E, L, P> {
    type Target = GenericArray<E, L>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<E: ConstDefault, L: ArrayLength, P: ArrayLength> DerefMut for DefaultPaddedArray<E, L, P> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<E: Default, L: ArrayLength, P: ArrayLength> Default for DefaultPaddedArray<E, L, P> {
    fn default() -> Self {
        Self {
            _pad0: GenericArray::default(),
            inner: GenericArray::default(),
            _pad1: GenericArray::default(),
        }
    }
}
