use yume_pdq::{
    GenericArray,
    generic_array::typenum::{U8, U16, U512},
    kernel::type_traits::SquareOf,
};

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

unsafe extern "C" {
    /// Adaptor function for the official PDQ hashing implementation.
    ///
    /// # Safety
    ///
    /// - `image_rgba_in` in (maybe? official API is confusing and declared as mut pointer) must be a pointer to a 512x512 image in RGBA format.
    /// - `tmp_512x512` must be a pointer to a 512x512 buffer.
    /// - `hash_all_dihedrals` out, must be a pointer to a 8x16 array of u16 or one of higher alignment and length.
    #[link_name = "yumepdq_official_512x512_hash_adapator"]
    unsafe fn yumepdq_official_512x512_hash_adapator(
        image_rgba_in: *mut f32,
        tmp_512x512: *mut f32,
        hash_all_dihedrals: *mut u16,
    ) -> i32;
}

/// Adaptor function for the official PDQ hashing implementation in facebook/ThreatExchange.
///
/// Input is declared as mut reference, but may be constant (official API is confusing).
pub fn hash_threat_exchange_512x512(
    image_rgba_in: &mut GenericArray<GenericArray<f32, U512>, U512>,
    tmp_512x512: &mut GenericArray<f32, <U512 as SquareOf>::Output>,
    hash_all_dihedrals: &mut GenericArray<GenericArray<u16, U16>, U8>,
) -> i32 {
    unsafe {
        yumepdq_official_512x512_hash_adapator(
            image_rgba_in.as_mut_ptr().cast(),
            tmp_512x512.as_mut_ptr(),
            hash_all_dihedrals.as_mut_ptr().cast(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
