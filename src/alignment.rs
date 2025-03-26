use core::ops::{Deref, DerefMut};

#[repr(align(32))]
/// Align the item to 32 bytes.
pub struct Align32<T>(pub T);

impl<T: Default> Default for Align32<T> {
    fn default() -> Self {
        Self(Default::default())
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
pub struct Align64<T>(pub T);

impl<T: Default> Default for Align64<T> {
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
