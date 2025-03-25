include!(concat!(env!("OUT_DIR"), "/dihedral.rs"));

/// Flip the hash horizontally (i.e. mirroring on the left-right axis, a dihedral flip using [[-1, 0], [0, 1]]).
///
/// It is recommended that you store both the regular hash and a horizontally flipped hash in your vector database.
pub const fn hash_hflip(output: &mut [u8; 2 * 16]) {
    let mut i = 0;
    macro_rules! do_loop {
        (1) => {
            (output[i * 2 + 1], output[i * 2]) = (
                FLIP_U8[output[i * 2] as usize],
                FLIP_U8[output[i * 2 + 1] as usize],
            );
        };
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
pub const fn hash_vflip(output: &mut [u8; 2 * 16]) {
    let mut i = 0;
    macro_rules! do_loop {
        (1) => {
            (output[i * 2], output[(16 - 1 - i) * 2]) = (output[(16 - 1 - i) * 2], output[i * 2]);
            (output[i * 2 + 1], output[(16 - 1 - i) * 2 + 1]) =
                (output[(16 - 1 - i) * 2 + 1], output[i * 2 + 1]);
        };
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
    }
    do_loop!(8);
}

#[cfg(not(target_feature = "avx2"))]
/// Flip the hash diagonally (i.e. mirroring on the top-left to bottom-right axis, a dihedral flip using [[0, 1], [1, 0]]).
pub const fn hash_diagflip(output: &mut [u8; 2 * 16]) {
    // a lookup table version when avx2 is not available
    macro_rules! pack8 {
        ($e0:expr, $e1:expr, $e2:expr, $e3:expr, $e4:expr, $e5:expr, $e6:expr, $e7:expr) => {
            if $e0 { 128 } else { 0 }
                | if $e1 { 64 } else { 0 }
                | if $e2 { 32 } else { 0 }
                | if $e3 { 16 } else { 0 }
                | if $e4 { 8 } else { 0 }
                | if $e5 { 4 } else { 0 }
                | if $e6 { 2 } else { 0 }
                | if $e7 { 1 } else { 0 }
        };
    }

    macro_rules! do_quadrant {
        (($basei_in:literal, $basej_in:literal) | ($basei_out:literal, $basej_out:literal)) => {{
            let row0_bits = EXTRACT_BITS[output[($basei_in + 0) * 2 + $basej_in] as usize];
            let row1_bits = EXTRACT_BITS[output[($basei_in + 1) * 2 + $basej_in] as usize];
            let row2_bits = EXTRACT_BITS[output[($basei_in + 2) * 2 + $basej_in] as usize];
            let row3_bits = EXTRACT_BITS[output[($basei_in + 3) * 2 + $basej_in] as usize];
            let row4_bits = EXTRACT_BITS[output[($basei_in + 4) * 2 + $basej_in] as usize];
            let row5_bits = EXTRACT_BITS[output[($basei_in + 5) * 2 + $basej_in] as usize];
            let row6_bits = EXTRACT_BITS[output[($basei_in + 6) * 2 + $basej_in] as usize];
            let row7_bits = EXTRACT_BITS[output[($basei_in + 7) * 2 + $basej_in] as usize];
            let row0_bits_r = EXTRACT_BITS[output[($basei_out + 0) * 2 + $basej_out] as usize];
            let row1_bits_r = EXTRACT_BITS[output[($basei_out + 1) * 2 + $basej_out] as usize];
            let row2_bits_r = EXTRACT_BITS[output[($basei_out + 2) * 2 + $basej_out] as usize];
            let row3_bits_r = EXTRACT_BITS[output[($basei_out + 3) * 2 + $basej_out] as usize];
            let row4_bits_r = EXTRACT_BITS[output[($basei_out + 4) * 2 + $basej_out] as usize];
            let row5_bits_r = EXTRACT_BITS[output[($basei_out + 5) * 2 + $basej_out] as usize];
            let row6_bits_r = EXTRACT_BITS[output[($basei_out + 6) * 2 + $basej_out] as usize];
            let row7_bits_r = EXTRACT_BITS[output[($basei_out + 7) * 2 + $basej_out] as usize];
            output[($basei_out + 0) * 2 + $basej_out] = pack8!(
                row0_bits[0],
                row1_bits[0],
                row2_bits[0],
                row3_bits[0],
                row4_bits[0],
                row5_bits[0],
                row6_bits[0],
                row7_bits[0]
            );
            output[($basei_out + 1) * 2 + $basej_out] = pack8!(
                row0_bits[1],
                row1_bits[1],
                row2_bits[1],
                row3_bits[1],
                row4_bits[1],
                row5_bits[1],
                row6_bits[1],
                row7_bits[1]
            );
            output[($basei_out + 2) * 2 + $basej_out] = pack8!(
                row0_bits[2],
                row1_bits[2],
                row2_bits[2],
                row3_bits[2],
                row4_bits[2],
                row5_bits[2],
                row6_bits[2],
                row7_bits[2]
            );
            output[($basei_out + 3) * 2 + $basej_out] = pack8!(
                row0_bits[3],
                row1_bits[3],
                row2_bits[3],
                row3_bits[3],
                row4_bits[3],
                row5_bits[3],
                row6_bits[3],
                row7_bits[3]
            );
            output[($basei_out + 4) * 2 + $basej_out] = pack8!(
                row0_bits[4],
                row1_bits[4],
                row2_bits[4],
                row3_bits[4],
                row4_bits[4],
                row5_bits[4],
                row6_bits[4],
                row7_bits[4]
            );
            output[($basei_out + 5) * 2 + $basej_out] = pack8!(
                row0_bits[5],
                row1_bits[5],
                row2_bits[5],
                row3_bits[5],
                row4_bits[5],
                row5_bits[5],
                row6_bits[5],
                row7_bits[5]
            );
            output[($basei_out + 6) * 2 + $basej_out] = pack8!(
                row0_bits[6],
                row1_bits[6],
                row2_bits[6],
                row3_bits[6],
                row4_bits[6],
                row5_bits[6],
                row6_bits[6],
                row7_bits[6]
            );
            output[($basei_out + 7) * 2 + $basej_out] = pack8!(
                row0_bits[7],
                row1_bits[7],
                row2_bits[7],
                row3_bits[7],
                row4_bits[7],
                row5_bits[7],
                row6_bits[7],
                row7_bits[7]
            );
            output[($basei_in + 0) * 2 + $basej_in] = pack8!(
                row0_bits_r[0],
                row1_bits_r[0],
                row2_bits_r[0],
                row3_bits_r[0],
                row4_bits_r[0],
                row5_bits_r[0],
                row6_bits_r[0],
                row7_bits_r[0]
            );
            output[($basei_in + 1) * 2 + $basej_in] = pack8!(
                row0_bits_r[1],
                row1_bits_r[1],
                row2_bits_r[1],
                row3_bits_r[1],
                row4_bits_r[1],
                row5_bits_r[1],
                row6_bits_r[1],
                row7_bits_r[1]
            );
            output[($basei_in + 2) * 2 + $basej_in] = pack8!(
                row0_bits_r[2],
                row1_bits_r[2],
                row2_bits_r[2],
                row3_bits_r[2],
                row4_bits_r[2],
                row5_bits_r[2],
                row6_bits_r[2],
                row7_bits_r[2]
            );
            output[($basei_in + 3) * 2 + $basej_in] = pack8!(
                row0_bits_r[3],
                row1_bits_r[3],
                row2_bits_r[3],
                row3_bits_r[3],
                row4_bits_r[3],
                row5_bits_r[3],
                row6_bits_r[3],
                row7_bits_r[3]
            );
            output[($basei_in + 4) * 2 + $basej_in] = pack8!(
                row0_bits_r[4],
                row1_bits_r[4],
                row2_bits_r[4],
                row3_bits_r[4],
                row4_bits_r[4],
                row5_bits_r[4],
                row6_bits_r[4],
                row7_bits_r[4]
            );
            output[($basei_in + 5) * 2 + $basej_in] = pack8!(
                row0_bits_r[5],
                row1_bits_r[5],
                row2_bits_r[5],
                row3_bits_r[5],
                row4_bits_r[5],
                row5_bits_r[5],
                row6_bits_r[5],
                row7_bits_r[5]
            );
            output[($basei_in + 6) * 2 + $basej_in] = pack8!(
                row0_bits_r[6],
                row1_bits_r[6],
                row2_bits_r[6],
                row3_bits_r[6],
                row4_bits_r[6],
                row5_bits_r[6],
                row6_bits_r[6],
                row7_bits_r[6]
            );
        }};
        (($basei_in:literal, $basej_in:literal)) => {{
            let row0_bits = EXTRACT_BITS[output[($basei_in + 0) * 2 + $basej_in] as usize];
            let row1_bits = EXTRACT_BITS[output[($basei_in + 1) * 2 + $basej_in] as usize];
            let row2_bits = EXTRACT_BITS[output[($basei_in + 2) * 2 + $basej_in] as usize];
            let row3_bits = EXTRACT_BITS[output[($basei_in + 3) * 2 + $basej_in] as usize];
            let row4_bits = EXTRACT_BITS[output[($basei_in + 4) * 2 + $basej_in] as usize];
            let row5_bits = EXTRACT_BITS[output[($basei_in + 5) * 2 + $basej_in] as usize];
            let row6_bits = EXTRACT_BITS[output[($basei_in + 6) * 2 + $basej_in] as usize];
            let row7_bits = EXTRACT_BITS[output[($basei_in + 7) * 2 + $basej_in] as usize];
            output[($basei_in + 0) * 2 + $basej_in] = pack8!(
                row0_bits[0],
                row1_bits[0],
                row2_bits[0],
                row3_bits[0],
                row4_bits[0],
                row5_bits[0],
                row6_bits[0],
                row7_bits[0]
            );
            output[($basei_in + 1) * 2 + $basej_in] = pack8!(
                row0_bits[1],
                row1_bits[1],
                row2_bits[1],
                row3_bits[1],
                row4_bits[1],
                row5_bits[1],
                row6_bits[1],
                row7_bits[1]
            );
            output[($basei_in + 2) * 2 + $basej_in] = pack8!(
                row0_bits[2],
                row1_bits[2],
                row2_bits[2],
                row3_bits[2],
                row4_bits[2],
                row5_bits[2],
                row6_bits[2],
                row7_bits[2]
            );
            output[($basei_in + 3) * 2 + $basej_in] = pack8!(
                row0_bits[3],
                row1_bits[3],
                row2_bits[3],
                row3_bits[3],
                row4_bits[3],
                row5_bits[3],
                row6_bits[3],
                row7_bits[3]
            );
            output[($basei_in + 4) * 2 + $basej_in] = pack8!(
                row0_bits[4],
                row1_bits[4],
                row2_bits[4],
                row3_bits[4],
                row4_bits[4],
                row5_bits[4],
                row6_bits[4],
                row7_bits[4]
            );
            output[($basei_in + 5) * 2 + $basej_in] = pack8!(
                row0_bits[5],
                row1_bits[5],
                row2_bits[5],
                row3_bits[5],
                row4_bits[5],
                row5_bits[5],
                row6_bits[5],
                row7_bits[5]
            );
            output[($basei_in + 6) * 2 + $basej_in] = pack8!(
                row0_bits[6],
                row1_bits[6],
                row2_bits[6],
                row3_bits[6],
                row4_bits[6],
                row5_bits[6],
                row6_bits[6],
                row7_bits[6]
            );
            output[($basei_in + 7) * 2 + $basej_in] = pack8!(
                row0_bits[7],
                row1_bits[7],
                row2_bits[7],
                row3_bits[7],
                row4_bits[7],
                row5_bits[7],
                row6_bits[7],
                row7_bits[7]
            );
        }};
    }
    do_quadrant!((0, 0));
    do_quadrant!((0, 1) | (8, 0));
    do_quadrant!((8, 1));
}

#[cfg(target_feature = "avx2")]
/// Flip the hash diagonally (i.e. mirroring on the top-left to bottom-right axis, a dihedral flip using [[0, 1], [1, 0]]).
pub const fn hash_diagflip(output: &mut [u8; 2 * 16]) {
    // a runtime version when avx2 is available and auto-vectorized to optimize performance
    const M0: u8 = 0b10000000;
    const M1: u8 = 0b01000000;
    const M2: u8 = 0b00100000;
    const M3: u8 = 0b00010000;
    const M4: u8 = 0b00001000;
    const M5: u8 = 0b00000100;
    const M6: u8 = 0b00000010;
    const M7: u8 = 0b00000001;

    macro_rules! pivot8 {
        ([$row0:expr, $row1:expr, $row2:expr, $row3:expr, $row4:expr, $row5:expr, $row6:expr, $row7:expr]) => {{
            let output0 = (($row0 & M0) >> 0)
                | (($row1 & M0) >> 1)
                | (($row2 & M0) >> 2)
                | (($row3 & M0) >> 3)
                | (($row4 & M0) >> 4)
                | (($row5 & M0) >> 5)
                | (($row6 & M0) >> 6)
                | (($row7 & M0) >> 7);
            let output1 = (($row0 & M1) << 1)
                | (($row1 & M1) >> 0)
                | (($row2 & M1) >> 1)
                | (($row3 & M1) >> 2)
                | (($row4 & M1) >> 3)
                | (($row5 & M1) >> 4)
                | (($row6 & M1) >> 5)
                | (($row7 & M1) >> 6);
            let output2 = (($row0 & M2) << 2)
                | (($row1 & M2) << 1)
                | (($row2 & M2) >> 0)
                | (($row3 & M2) >> 1)
                | (($row4 & M2) >> 2)
                | (($row5 & M2) >> 3)
                | (($row6 & M2) >> 4)
                | (($row7 & M2) >> 5);
            let output3 = (($row0 & M3) << 3)
                | (($row1 & M3) << 2)
                | (($row2 & M3) << 1)
                | (($row3 & M3) >> 0)
                | (($row4 & M3) >> 1)
                | (($row5 & M3) >> 2)
                | (($row6 & M3) >> 3)
                | (($row7 & M3) >> 4);
            let output4 = (($row0 & M4) << 4)
                | (($row1 & M4) << 3)
                | (($row2 & M4) << 2)
                | (($row3 & M4) << 1)
                | (($row4 & M4) >> 0)
                | (($row5 & M4) >> 1)
                | (($row6 & M4) >> 2)
                | (($row7 & M4) >> 3);
            let output5 = (($row0 & M5) << 5)
                | (($row1 & M5) << 4)
                | (($row2 & M5) << 3)
                | (($row3 & M5) << 2)
                | (($row4 & M5) << 1)
                | (($row5 & M5) >> 0)
                | (($row6 & M5) >> 1)
                | (($row7 & M5) >> 2);
            let output6 = (($row0 & M6) << 6)
                | (($row1 & M6) << 5)
                | (($row2 & M6) << 4)
                | (($row3 & M6) << 3)
                | (($row4 & M6) << 2)
                | (($row5 & M6) << 1)
                | (($row6 & M6) >> 0)
                | (($row7 & M6) >> 1);
            let output7 = (($row0 & M7) << 7)
                | (($row1 & M7) << 6)
                | (($row2 & M7) << 5)
                | (($row3 & M7) << 4)
                | (($row4 & M7) << 3)
                | (($row5 & M7) << 2)
                | (($row6 & M7) << 1)
                | (($row7 & M7) >> 0);
            (
                output0, output1, output2, output3, output4, output5, output6, output7,
            )
        }};
    }

    macro_rules! do_quadrant {
        (($basei_in:literal, $basej_in:literal) | ($basei_out:literal, $basej_out:literal)) => {{
            let row0 = output[($basei_in + 0) * 2 + $basej_in];
            let row1 = output[($basei_in + 1) * 2 + $basej_in];
            let row2 = output[($basei_in + 2) * 2 + $basej_in];
            let row3 = output[($basei_in + 3) * 2 + $basej_in];
            let row4 = output[($basei_in + 4) * 2 + $basej_in];
            let row5 = output[($basei_in + 5) * 2 + $basej_in];
            let row6 = output[($basei_in + 6) * 2 + $basej_in];
            let row7 = output[($basei_in + 7) * 2 + $basej_in];
            let row_r0 = output[($basei_out + 0) * 2 + $basej_out];
            let row_r1 = output[($basei_out + 1) * 2 + $basej_out];
            let row_r2 = output[($basei_out + 2) * 2 + $basej_out];
            let row_r3 = output[($basei_out + 3) * 2 + $basej_out];
            let row_r4 = output[($basei_out + 4) * 2 + $basej_out];
            let row_r5 = output[($basei_out + 5) * 2 + $basej_out];
            let row_r6 = output[($basei_out + 6) * 2 + $basej_out];
            let row_r7 = output[($basei_out + 7) * 2 + $basej_out];
            // parenthesize to avoid formatter getting too excited
            let pivoted = pivot8!([row0, row1, row2, row3, row4, row5, row6, row7]);
            let pivoted_r = pivot8!([
                row_r0, row_r1, row_r2, row_r3, row_r4, row_r5, row_r6, row_r7
            ]);
            output[($basei_out + 0) * 2 + $basej_out] = pivoted.0;
            output[($basei_out + 1) * 2 + $basej_out] = pivoted.1;
            output[($basei_out + 2) * 2 + $basej_out] = pivoted.2;
            output[($basei_out + 3) * 2 + $basej_out] = pivoted.3;
            output[($basei_out + 4) * 2 + $basej_out] = pivoted.4;
            output[($basei_out + 5) * 2 + $basej_out] = pivoted.5;
            output[($basei_out + 6) * 2 + $basej_out] = pivoted.6;
            output[($basei_out + 7) * 2 + $basej_out] = pivoted.7;
            output[($basei_in + 0) * 2 + $basej_in] = pivoted_r.0;
            output[($basei_in + 1) * 2 + $basej_in] = pivoted_r.1;
            output[($basei_in + 2) * 2 + $basej_in] = pivoted_r.2;
            output[($basei_in + 3) * 2 + $basej_in] = pivoted_r.3;
            output[($basei_in + 4) * 2 + $basej_in] = pivoted_r.4;
            output[($basei_in + 5) * 2 + $basej_in] = pivoted_r.5;
            output[($basei_in + 6) * 2 + $basej_in] = pivoted_r.6;
            output[($basei_in + 7) * 2 + $basej_in] = pivoted_r.7;
        }};
        (($basei_in:literal, $basej_in:literal)) => {{
            let row0 = output[($basei_in + 0) * 2 + $basej_in];
            let row1 = output[($basei_in + 1) * 2 + $basej_in];
            let row2 = output[($basei_in + 2) * 2 + $basej_in];
            let row3 = output[($basei_in + 3) * 2 + $basej_in];
            let row4 = output[($basei_in + 4) * 2 + $basej_in];
            let row5 = output[($basei_in + 5) * 2 + $basej_in];
            let row6 = output[($basei_in + 6) * 2 + $basej_in];
            let row7 = output[($basei_in + 7) * 2 + $basej_in];
            let pivoted = pivot8!([row0, row1, row2, row3, row4, row5, row6, row7]);
            output[($basei_in + 0) * 2 + $basej_in] = pivoted.0;
            output[($basei_in + 1) * 2 + $basej_in] = pivoted.1;
            output[($basei_in + 2) * 2 + $basej_in] = pivoted.2;
            output[($basei_in + 3) * 2 + $basej_in] = pivoted.3;
            output[($basei_in + 4) * 2 + $basej_in] = pivoted.4;
            output[($basei_in + 5) * 2 + $basej_in] = pivoted.5;
            output[($basei_in + 6) * 2 + $basej_in] = pivoted.6;
            output[($basei_in + 7) * 2 + $basej_in] = pivoted.7;
        }};
    }
    do_quadrant!((0, 0));
    do_quadrant!((0, 1) | (8, 0));
    do_quadrant!((8, 1));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_diagflip() {
        let mut hash = [0; 2 * 16];
        hash[1] = 0b00000100;
        hash_diagflip(&mut hash);
        assert_eq!(
            hash,
            [
                0, 0, // row0
                0, 0, // row1
                0, 0, // row2
                0, 0, // row3
                0, 0, // row4
                0, 0, // row5
                0, 0, // row6
                0, 0, // row7
                0, 0, // row8
                0, 0, // row9
                0, 0, // row10
                0, 0, // row11
                0, 0, // row12
                0b10000000, 0, // row13
                0, 0, // row14
                0, 0, // row15
            ]
        );

        // make sure we didn't miss any positions
        hash.fill(!0);
        hash_diagflip(&mut hash);
        assert_eq!(hash, [!0; 2 * 16]);
    }

    #[test]
    fn test_hash_hflip() {
        let mut hash = [0; 2 * 16];
        hash[4] = 0b00000010;
        hash_hflip(&mut hash);
        assert_eq!(
            hash,
            [
                0, 0, // row0
                0, 0, // row1
                0, 0b01000000, // row2
                0, 0, // row3
                0, 0, // row4
                0, 0, // row5
                0, 0, // row6
                0, 0, // row7
                0, 0, // row8
                0, 0, // row9
                0, 0, // row10
                0, 0, // row11
                0, 0, // row12
                0, 0, // row13
                0, 0, // row14
                0, 0, // row15
            ]
        );

        // make sure we didn't miss any positions
        hash.fill(!0);
        hash_hflip(&mut hash);
        assert_eq!(hash, [!0; 2 * 16]);
    }

    #[test]
    fn test_hash_vflip() {
        let mut hash = [0; 2 * 16];
        hash[2] = 0b00000010;
        hash_vflip(&mut hash);
        assert_eq!(
            hash,
            [
                0, 0, // row0
                0, 0, // row1
                0, 0, // row2
                0, 0, // row3
                0, 0, // row4
                0, 0, // row5
                0, 0, // row6
                0, 0, // row7
                0, 0, // row8
                0, 0, // row9
                0, 0, // row10
                0, 0, // row11
                0, 0, // row12
                0, 0, // row13
                0b00000010, 0, // row14
                0, 0, // row15
            ]
        );

        // make sure we didn't miss any positions
        hash.fill(!0);
        hash_vflip(&mut hash);
        assert_eq!(hash, [!0; 2 * 16]);
    }
}
