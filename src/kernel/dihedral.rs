#![allow(dead_code, reason = "Experimental code")]
use generic_array::{
    ArrayLength, GenericArray,
    typenum::{IsLessOrEqual, U32},
};

use crate::PDQHashF;

include!(concat!(env!("OUT_DIR"), "/dihedral.rs"));

/// Flip the hash horizontally (i.e. mirroring on the left-right axis, a dihedral flip using [[-1, 0], [0, 1]]).
///
/// It is recommended that you store both the regular hash and a horizontally flipped hash in your vector database.
pub const fn hash_hflip<N: Copy>(output: &mut PDQHashF<N>) {
    let mut i = 0;
    let output_as_slice = unsafe { core::mem::transmute::<_, &mut [[N; 16]; 16]>(output) };
    macro_rules! do_loop {
        (1) => {{
            let target = &mut output_as_slice[i];
            (target[0], target[15]) = (target[15], target[0]);
            (target[1], target[14]) = (target[14], target[1]);
            (target[2], target[13]) = (target[13], target[2]);
            (target[3], target[12]) = (target[12], target[3]);
            (target[4], target[11]) = (target[11], target[4]);
            (target[5], target[10]) = (target[10], target[5]);
            (target[6], target[9]) = (target[9], target[6]);
            (target[7], target[8]) = (target[8], target[7]);
        }};
        (2) => {
            do_loop!(1);
            i += 1;
            do_loop!(1);
        };
        (4) => {
            do_loop!(2);
            i += 1;
            do_loop!(2);
        };
        (8) => {
            do_loop!(4);
            i += 1;
            do_loop!(4);
        };
        (16) => {
            do_loop!(8);
            i += 1;
            do_loop!(8);
        };
    }
    do_loop!(16);
}

/// Flip the hash vertically (i.e. mirroring on the top-bottom axis, a dihedral flip using [[1, 0], [0, -1]]).
pub const fn hash_vflip<N: Copy>(output: &mut PDQHashF<N>) {
    let output_as_slice = unsafe { core::mem::transmute::<_, &mut [[N; 16]; 16]>(output) };
    output_as_slice.swap(0, 15);
    output_as_slice.swap(1, 14);
    output_as_slice.swap(2, 13);
    output_as_slice.swap(3, 12);
    output_as_slice.swap(4, 11);
    output_as_slice.swap(5, 10);
    output_as_slice.swap(6, 9);
    output_as_slice.swap(7, 8);
}

const fn swap_anti_diagonal<N: Copy, const START_I: usize, L: ArrayLength>(
    output: &mut GenericArray<GenericArray<N, L>, L>,
) {
    let mut left_i = START_I;
    let mut left_j = 0;
    let mut right_i = 0;
    let mut right_j = START_I;

    let output_ptr = output.as_mut_slice().as_mut_ptr().cast::<N>();

    macro_rules! do_loop {
        (1) => {
            if left_j > right_j {
                return;
            }
            unsafe {
                core::ptr::swap(
                    output_ptr.add(left_i * L::USIZE + left_j),
                    output_ptr.add(right_i * L::USIZE + right_j),
                );
            }
        };
        (2) => {
            do_loop!(1);
            left_j += 1;
            if left_i > 0 {
                left_i -= 1;
            }
            if right_j > 0 {
                right_j -= 1;
            }
            right_i += 1;
            do_loop!(1);
        };
        (4) => {
            do_loop!(2);
            left_j += 1;
            if left_i > 0 {
                left_i -= 1;
            }
            if right_j > 0 {
                right_j -= 1;
            }
            right_i += 1;
            do_loop!(2);
        };
        (8) => {
            do_loop!(4);
            left_j += 1;
            if left_i > 0 {
                left_i -= 1;
            }
            if right_j > 0 {
                right_j -= 1;
            }
            right_i += 1;
            do_loop!(4);
        };
        (16) => {
            do_loop!(8);
            left_j += 1;
            if left_i > 0 {
                left_i -= 1;
            }
            if right_j > 0 {
                right_j -= 1;
            }
            right_i += 1;
            do_loop!(8);
        };
    }

    do_loop!(16);
}
