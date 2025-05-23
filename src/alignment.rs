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
use generic_array::{
    ArrayLength, GenericArray,
    typenum::{B0, B1, IsGreater, PowerOfTwo, U0, U1, U8, U32, U64, UInt, Unsigned},
};
use zeroize::DefaultIsZeroes;

/// A trait for a trivial aligner that has no function except setting the alignment and transparently holding a value of type `T`.
///
/// # Safety
///
/// Implementor must ensure their memory layout is valid.
pub unsafe trait AlignerTo<T>: Deref<Target = T> + DerefMut<Target = T> + From<T> {
    /// The alignment of the aligner.
    type Alignment: Unsigned + PowerOfTwo + IsGreater<U0, Output = B1>;
    /// The type of the aligner.
    type Output;

    /// Create a `core::alloc::Layout` for the aligner.
    ///
    /// # Panics
    ///
    /// Panics if the request memory size is rejected by the allocator API.
    fn create_layout() -> core::alloc::Layout;
}

unsafe impl<T> AlignerTo<T> for Align1<T> {
    type Alignment = U1;
    type Output = Align1<T>;

    fn create_layout() -> core::alloc::Layout {
        core::alloc::Layout::from_size_align(core::mem::size_of::<T>(), 1)
            .expect("request memory size is too large for allocation")
    }
}

unsafe impl<T> AlignerTo<T> for Align8<T> {
    type Alignment = U8;
    type Output = Align8<T>;

    fn create_layout() -> core::alloc::Layout {
        core::alloc::Layout::from_size_align(core::mem::size_of::<T>(), 8)
            .expect("request memory size is too large for allocation")
    }
}

unsafe impl<T> AlignerTo<T> for Align32<T> {
    type Alignment = U32;
    type Output = Align32<T>;
    fn create_layout() -> core::alloc::Layout {
        core::alloc::Layout::from_size_align(core::mem::size_of::<T>(), 32)
            .expect("request memory size is too large for allocation")
    }
}

unsafe impl<T> AlignerTo<T> for Align64<T> {
    type Alignment = U64;
    type Output = Align64<T>;

    fn create_layout() -> core::alloc::Layout {
        core::alloc::Layout::from_size_align(core::mem::size_of::<T>(), 64)
            .expect("request memory size is too large for allocation")
    }
}

/// Align the item to 1 byte (a dummy aligner).
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Align1<T>(pub T);

impl<T: DefaultIsZeroes + Copy> DefaultIsZeroes for Align1<T> {}

impl<T: Default> Default for Align1<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Deref for Align1<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align1<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> From<T> for Align1<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T> Align1<T> {
    /// Create a new `Align1<T>` from a `T`.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Convert a pointer to a `Align1<T>` to a reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 1 byte, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_unchecked(ptr: *const T) -> *const Self {
        ptr.cast::<Self>()
    }

    /// Convert a reference to a `T` to a reference to a `Align1<T>`.
    ///
    /// # Panics
    ///
    pub fn from_raw(input: &T) -> &Self {
        unsafe { &*(input as *const T).cast::<Self>() }
    }

    /// Convert a pointer to a `Align1<T>` to a mutable reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 1 byte, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_mut_unchecked(ptr: *mut T) -> *mut Self {
        ptr.cast::<Self>()
    }

    /// Convert a reference to a `T` to a mutable reference to a `Align1<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the reference is not aligned to 1 byte.
    pub fn from_raw_mut(input: &mut T) -> &mut Self {
        unsafe { &mut *(input as *mut T).cast::<Self>() }
    }
}

#[repr(align(8))]
#[derive(Debug, Clone, Copy)]
/// Align the item to 8 bytes.
pub struct Align8<T>(pub T);

impl<T: DefaultIsZeroes + Copy> DefaultIsZeroes for Align8<T> {}

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

impl<T> From<T> for Align8<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

type Times8<T> = UInt<UInt<UInt<T, B0>, B0>, B0>;

type Times32<T> = UInt<UInt<Times8<T>, B0>, B0>;

type Times64<T> = UInt<Times32<T>, B0>;

impl<T, L: ArrayLength> Align8<GenericArray<T, Times8<L>>> {
    /// Convert a `GenericArray<Align8<GenericArray<T, L>>, U>` to a `Align8<GenericArray<GenericArray<T, L>, U>>`, if L is a multiple of 8.
    #[must_use]
    pub const fn lift<U: ArrayLength>(
        input: GenericArray<Self, U>,
    ) -> Align8<GenericArray<GenericArray<T, Times8<L>>, U>> {
        // SAFETY: if L is a multiple of 8, then GenericArray<T, L> is either a ZST or at least a multiple of 8 bytes, guaranteeing a zero-padding layout
        unsafe { generic_array::const_transmute(input) }
    }

    /// Convert a `Box<GenericArray<Align8<GenericArray<T, L>>, U>>` to a `Box<Align8<GenericArray<GenericArray<T, L>, U>>>`, if L is a multiple of 8.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub const fn lift_boxed<U: ArrayLength>(
        input: alloc::boxed::Box<GenericArray<Self, U>>,
    ) -> alloc::boxed::Box<Align8<GenericArray<GenericArray<T, Times8<L>>, U>>> {
        // SAFETY: if L is a multiple of 8, then GenericArray<T, L> is either a ZST or at least a multiple of 8 bytes, guaranteeing a zero-padding layout
        unsafe { generic_array::const_transmute(input) }
    }
}

impl<T> Align8<T> {
    /// Create a new `Align8<T>` from a `T`.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Convert a pointer to a `Align8<T>` to a reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 8 bytes, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_unchecked(ptr: *const T) -> *const Self {
        ptr.cast::<Self>()
    }

    /// Convert a reference to a `T` to a reference to a `Align8<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the reference is not aligned to 8 bytes.  
    pub fn from_raw(input: &T) -> &Self {
        let ptr = input as *const T;
        assert_eq!(ptr.align_offset(8), 0, "pointer is not aligned to 8 bytes");
        unsafe { &*(ptr.cast::<Self>()) }
    }

    /// Convert a pointer to a `Align8<T>` to a mutable reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 8 bytes, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_mut_unchecked(ptr: *mut T) -> *mut Self {
        ptr.cast::<Self>()
    }

    /// Convert a reference to a `T` to a mutable reference to a `Align8<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the reference is not aligned to 8 bytes.
    pub fn from_raw_mut(input: &mut T) -> &mut Self {
        let ptr = input as *mut T;
        assert_eq!(ptr.align_offset(8), 0, "pointer is not aligned to 8 bytes");
        unsafe { &mut *(ptr.cast::<Self>()) }
    }
}

#[repr(align(32))]
#[derive(Debug, Clone, Copy)]
/// Align the item to 32 bytes.
pub struct Align32<T>(pub T);

impl<T: DefaultIsZeroes + Copy> DefaultIsZeroes for Align32<T> {}

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

impl<T> From<T> for Align32<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T, L: ArrayLength> Align32<GenericArray<T, Times32<L>>> {
    /// Convert a `GenericArray<Align32<GenericArray<T, L>>, U>` to a `Align32<GenericArray<GenericArray<T, L>, U>>`, if L is a multiple of 32.
    #[must_use]
    pub const fn lift<U: ArrayLength>(
        input: GenericArray<Self, U>,
    ) -> Align32<GenericArray<GenericArray<T, Times32<L>>, U>> {
        // SAFETY: if L is a multiple of 32, then GenericArray<T, L> is either a ZST or at least a multiple of 32 bytes, guaranteeing a zero-padding layout
        unsafe { generic_array::const_transmute(input) }
    }

    /// Convert a `Box<GenericArray<Align32<GenericArray<T, L>>, U>>` to a `Box<Align32<GenericArray<GenericArray<T, L>, U>>>`, if L is a multiple of 32.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub const fn lift_boxed<U: ArrayLength>(
        input: alloc::boxed::Box<GenericArray<Self, U>>,
    ) -> alloc::boxed::Box<Align32<GenericArray<GenericArray<T, Times32<L>>, U>>> {
        // SAFETY: if L is a multiple of 32, then GenericArray<T, L> is either a ZST or at least a multiple of 32 bytes, guaranteeing a zero-padding layout
        unsafe { generic_array::const_transmute(input) }
    }
}

impl<T> Align32<T> {
    /// Create a new `Align32<T>` from a `T`.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Convert a pointer to a `Align32<T>` to a reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 32 bytes, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_unchecked(ptr: *const T) -> *const Self {
        ptr.cast::<Self>()
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
        unsafe { &*(ptr.cast::<Self>()) }
    }

    /// Convert a pointer to a `Align32<T>` to a mutable reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 32 bytes, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_mut_unchecked(ptr: *mut T) -> *mut Self {
        ptr.cast::<Self>()
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
        unsafe { &mut *(ptr.cast::<Self>()) }
    }
}

#[repr(align(64))]
/// Align the item to 64 bytes.
#[derive(Debug, Clone, Copy)]
pub struct Align64<T>(pub T);

impl<T: DefaultIsZeroes + Copy> DefaultIsZeroes for Align64<T> {}

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

impl<T> From<T> for Align64<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T> Align64<T> {
    /// Create a new `Align64<T>` from a `T`.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Convert a pointer to a `Align64<T>` to a reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 64 bytes, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_unchecked(ptr: *const T) -> *const Self {
        ptr.cast::<Self>()
    }

    /// Convert a reference to a `T` to a reference to a `Align64<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the reference is not aligned to 64 bytes.
    pub fn from_raw(input: &T) -> &Self {
        let ptr = input as *const T;
        assert_eq!(
            ptr.align_offset(64),
            0,
            "pointer is not aligned to 64 bytes"
        );
        unsafe { &*(ptr.cast::<Self>()) }
    }

    /// Convert a pointer to a `Align64<T>` to a mutable reference.
    ///
    /// # Safety
    ///
    /// The pointer was not checked to be aligned to 64 bytes, so it is the caller's responsibility to ensure it is.
    pub const unsafe fn from_raw_mut_unchecked(ptr: *mut T) -> *mut Self {
        ptr.cast::<Self>()
    }

    /// Convert a reference to a `T` to a mutable reference to a `Align64<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the reference is not aligned to 64 bytes.
    pub fn from_raw_mut(input: &mut T) -> &mut Self {
        let ptr = input as *mut T;
        assert_eq!(
            ptr.align_offset(64),
            0,
            "pointer is not aligned to 64 bytes"
        );
        unsafe { &mut *(ptr.cast::<Self>()) }
    }
}

impl<T, L: ArrayLength> Align64<GenericArray<T, Times64<L>>> {
    /// Convert a `GenericArray<Align64<GenericArray<T, L>>, U>` to a `Align64<GenericArray<GenericArray<T, L>, U>>`, if L is a multiple of 64.
    #[must_use]
    pub const fn lift<U: ArrayLength>(
        input: GenericArray<Self, U>,
    ) -> Align64<GenericArray<GenericArray<T, Times64<L>>, U>> {
        // SAFETY: if L is a multiple of 64, then GenericArray<T, L> is either a ZST or at least a multiple of 64 bytes, guaranteeing a zero-padding layout
        unsafe { generic_array::const_transmute(input) }
    }

    /// Convert a `Box<GenericArray<Align64<GenericArray<T, L>>, U>>` to a `Box<Align64<GenericArray<GenericArray<T, L>, U>>>`, if L is a multiple of 64.
    #[cfg(feature = "alloc")]
    #[must_use]
    pub const fn lift_boxed<U: ArrayLength>(
        input: alloc::boxed::Box<GenericArray<Self, U>>,
    ) -> alloc::boxed::Box<Align64<GenericArray<GenericArray<T, Times64<L>>, U>>> {
        // SAFETY: if L is a multiple of 64, then GenericArray<T, L> is either a ZST or at least a multiple of 64 bytes, guaranteeing a zero-padding layout
        unsafe { generic_array::const_transmute(input) }
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
    /// Get a pointer to the inner array.
    pub fn as_ptr(&self) -> *const E {
        self.inner.as_ptr()
    }

    /// Get a mutable pointer to the inner array.
    pub fn as_mut_ptr(&mut self) -> *mut E {
        self.inner.as_mut_ptr()
    }
}

impl<E: ConstDefault, L: ArrayLength, P: ArrayLength> AsRef<GenericArray<E, L>>
    for DefaultPaddedArray<E, L, P>
{
    fn as_ref(&self) -> &GenericArray<E, L> {
        &self.inner
    }
}

impl<E: ConstDefault, L: ArrayLength, P: ArrayLength> AsMut<GenericArray<E, L>>
    for DefaultPaddedArray<E, L, P>
{
    fn as_mut(&mut self) -> &mut GenericArray<E, L> {
        &mut self.inner
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

#[cfg(feature = "alloc")]
/// Allocate a 1D generic array on the heap and zeroize it, without creating intermediates on the stack.
pub fn calloc_generic_array_1d<
    T: DefaultIsZeroes + Copy,
    A: AlignerTo<GenericArray<T, L>>,
    L: ArrayLength,
>() -> alloc::boxed::Box<<A as AlignerTo<GenericArray<T, L>>>::Output> {
    // SAFETY:
    // - T must be non-drop (constrained by Copy)
    // - Default must be the natural zero value for T (constrained by DefaultIsZeroes)

    use core::mem::MaybeUninit;
    unsafe {
        let template = T::default();
        let layout = A::create_layout();
        let memory = alloc::alloc::alloc(layout);
        let mut_slice = core::slice::from_raw_parts_mut(memory.cast::<MaybeUninit<T>>(), L::USIZE);
        mut_slice.iter_mut().for_each(|item| {
            item.as_mut_ptr().write(template);
        });
        alloc::boxed::Box::from_raw(memory.cast())
    }
}

#[cfg(feature = "alloc")]
/// Allocate a 2D generic array on the heap and zeroize it, without creating intermediates on the stack.
pub fn calloc_generic_array_2d<
    T: DefaultIsZeroes + Copy,
    A: AlignerTo<GenericArray<GenericArray<T, M>, L>>,
    L: ArrayLength,
    M: ArrayLength,
>() -> alloc::boxed::Box<<A as AlignerTo<GenericArray<GenericArray<T, M>, L>>>::Output> {
    // SAFETY:
    // - T must be non-drop (constrained by Copy)
    // - Default must be the natural zero value for T (constrained by DefaultIsZeroes)

    use core::mem::MaybeUninit;
    unsafe {
        let template = T::default();
        let layout = A::create_layout();
        let memory = alloc::alloc::alloc(layout);
        let mut_slice =
            core::slice::from_raw_parts_mut(memory.cast::<MaybeUninit<T>>(), L::USIZE * M::USIZE);

        mut_slice.iter_mut().for_each(|item| {
            item.as_mut_ptr().write(template);
        });
        alloc::boxed::Box::from_raw(memory.cast())
    }
}

#[cfg(feature = "alloc")]
/// Allocate a 3D generic array on the heap and zeroize it, without creating intermediates on the stack.
pub fn calloc_generic_array_3d<
    T: DefaultIsZeroes + Copy,
    A: AlignerTo<GenericArray<GenericArray<GenericArray<T, M>, N>, L>>,
    L: ArrayLength,
    M: ArrayLength,
    N: ArrayLength,
>()
-> alloc::boxed::Box<<A as AlignerTo<GenericArray<GenericArray<GenericArray<T, M>, N>, L>>>::Output>
{
    // SAFETY:
    // - T must be non-drop (constrained by Copy)
    // - Default must be the natural zero value for T (constrained by DefaultIsZeroes)

    use core::mem::MaybeUninit;

    unsafe {
        let template = T::default();
        let layout = A::create_layout();
        let memory = alloc::alloc::alloc(layout);
        let mut_slice = core::slice::from_raw_parts_mut(
            memory.cast::<MaybeUninit<T>>(),
            L::USIZE * M::USIZE * N::USIZE,
        );
        mut_slice.iter_mut().for_each(|item| {
            item.as_mut_ptr().write(template);
        });
        alloc::boxed::Box::from_raw(memory.cast())
    }
}
