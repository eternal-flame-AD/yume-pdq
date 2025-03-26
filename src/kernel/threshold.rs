use core::ops::{BitAnd, Div};

use generic_array::{
    ArrayLength, GenericArray,
    typenum::{IsLessOrEqual, U0, U7, U8, U32},
};

/// Threshold the 2D array.
///
/// You do not have to call this function yourself if you just want the hash, the quantization kernel will threshold the
/// data without you having to explicitly call this function.
///
/// Instead this function is useful for recomputing hashes after applying dihedral flips to the DCT-II output.
///
/// This function is generic up to 32x32.
pub const fn threshold_2d_f32<L: ArrayLength>(
    input: &GenericArray<GenericArray<f32, L>, L>,
    output: &mut GenericArray<GenericArray<u8, <L as Div<U8>>::Output>, L>,
    threshold: f32,
) where
    L: BitAnd<U7, Output = U0> + Div<U8> + IsLessOrEqual<U32>,
    <L as Div<U8>>::Output: ArrayLength,
{
    unsafe {
        let mut ptr = input.as_slice().as_ptr().cast::<f32>();
        let mut countdown = L::USIZE * L::USIZE;
        let mut output_ptr = output
            .as_mut_slice()
            .as_mut_ptr()
            .cast::<u8>()
            .add(L::USIZE * L::USIZE / 8 - 1);

        macro_rules! pack1 {
            ($write:expr) => {
                if *ptr > threshold {
                    $write |= 0x80;
                }
                ptr = ptr.add(1);
                countdown -= 1;
            };
        }

        macro_rules! do_loop {
            ($msb:literal * 32 + inner 1) => {{
                let mut write = 0u8;
                pack1!(write);
                write >>= 1;
                pack1!(write);
                write >>= 1;
                pack1!(write);
                write >>= 1;
                pack1!(write);
                write >>= 1;
                pack1!(write);
                write >>= 1;
                pack1!(write);
                write >>= 1;
                pack1!(write);
                write >>= 1;
                pack1!(write);
                *output_ptr = write;
                if countdown == 0 {
                    return;
                }
                output_ptr = output_ptr.sub(1);
            }};
            ($msb:literal * 32 + inner 2) => {{
                do_loop!($msb * 32 + inner 1);
                do_loop!($msb * 32 + inner 1);
            }};
            ($msb:literal * 32 + inner 4) => {{
                do_loop!($msb * 32 + inner 2);
                do_loop!($msb * 32 + inner 2);
            }};
            ($msb:literal * 32 + inner 8) => {{
                do_loop!($msb * 32 + inner 4);
                do_loop!($msb * 32 + inner 4);
            }};
            ($msb:literal * 32 + inner 16) => {{
                do_loop!($msb * 32 + inner 8);
                do_loop!($msb * 32 + inner 8);
            }};
            ($msb:literal * 32 + inner 32) => {{
                do_loop!($msb * 32 + inner 16);
                do_loop!($msb * 32 + inner 16);
            }};
            ([$($msb:literal),*] * 32) => {
                $(
                    do_loop!($msb * 32 + inner 32);
                )*
            };
        }

        do_loop!([0, 1, 2, 3] * 32);

        let _ = (output_ptr, ptr);
    }
}
