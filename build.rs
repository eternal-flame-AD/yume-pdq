/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Sunday, March 23, 2025
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

use std::{
    fmt::Debug,
    fs::File,
    io::Write,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
    path::PathBuf,
};

use rug::Float as RFloat;

trait MiniFloat:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + Neg
    + PartialOrd<Self>
    + Clone
    + Debug
    + Sized
{
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(self) -> Self;
    fn pi() -> Self;
    fn cos(self) -> Self;
    fn from_i32(i: i32) -> Self;
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
}

#[derive(
    Debug,
    Clone,
    PartialEq,
    PartialOrd,
    derive_more::Add,
    derive_more::Sub,
    derive_more::AddAssign,
    derive_more::SubAssign,
    derive_more::MulAssign,
    derive_more::DivAssign,
)]
struct MultiPrecF<const PREC: u32 = 128>(RFloat);

impl<const PREC: u32> Mul for MultiPrecF<PREC> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(self.0.mul(rhs.0))
    }
}

impl<const PREC: u32> Div for MultiPrecF<PREC> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self(self.0.div(rhs.0))
    }
}

impl<const PREC: u32> Neg for MultiPrecF<PREC> {
    type Output = Self;

    fn neg(self) -> Self {
        Self(self.0.neg())
    }
}

impl<const PREC: u32> MiniFloat for MultiPrecF<PREC> {
    fn zero() -> Self {
        Self(RFloat::new(PREC))
    }

    fn one() -> Self {
        let f = RFloat::with_val(PREC, 1);
        Self(f)
    }

    fn pi() -> Self {
        let mut pi = RFloat::new(PREC);
        pi.acos_mut();
        pi *= 2;
        assert!(
            pi.to_f64() == std::f64::consts::PI,
            "pi: {:?}, std::f64::consts::PI: {:?}",
            pi.to_f64(),
            std::f64::consts::PI
        );
        Self(pi)
    }

    fn cos(self) -> Self {
        Self(self.0.cos())
    }

    fn from_i32(i: i32) -> Self {
        Self(RFloat::with_val(PREC, i))
    }

    fn to_f32(self) -> f32 {
        self.0.to_f32()
    }

    fn to_f64(self) -> f64 {
        self.0.to_f64()
    }

    fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }
}

// taken from an officially-endorsed reference implementation with some slight modifications to prefer accuracy over speed
//
// https://github.com/darwinium-com/pdqhash/blob/main/src/lib.rs#L139
mod mean_box {
    use crate::MiniFloat;

    // This is called from two places, one has a constant stride, the other a variable stride
    // It should compile a version for each.
    #[inline]
    pub fn box_one_d_float<F: MiniFloat>(
        invec: &[F],
        in_start_offset: usize,
        outvec: &mut [F],
        vector_length: usize,
        stride: usize,
        full_window_size: usize,
    ) {
        let half_window_size = (full_window_size + 2) / 2; // 7->4, 8->5

        let phase_1_nreps = half_window_size - 1;
        let phase_2_nreps = full_window_size - half_window_size + 1;

        let oi_off = phase_1_nreps * stride;
        let li_off = phase_2_nreps * stride;

        let mut sum = F::zero();
        let mut current_window_size = F::zero();

        let phase_1_end = oi_off + in_start_offset;

        // PHASE 1: ACCUMULATE FIRST SUM NO WRITES
        for ri in (in_start_offset..phase_1_end).step_by(stride) {
            let value = invec[ri].clone();
            sum += value;
            current_window_size += F::one();
        }

        let phase_2_end = full_window_size * stride + in_start_offset;
        // PHASE 2: INITIAL WRITES WITH SMALL WINDOW
        for ri in (phase_1_end..phase_2_end).step_by(stride) {
            let oi = ri - oi_off;
            sum += invec[ri].clone();
            current_window_size += F::one();
            outvec[oi] = sum.clone() / current_window_size.clone();
        }

        let phase_3_end = vector_length * stride + in_start_offset;
        // PHASE 3: WRITES WITH FULL WINDOW
        for ri in (phase_2_end..phase_3_end).step_by(stride) {
            let oi = ri - oi_off;
            let li = oi - li_off;
            sum += invec[ri].clone();
            sum -= invec[li].clone();
            outvec[oi] = sum.clone() / current_window_size.clone();
        }

        let phase_4_start = (vector_length - half_window_size + 1) * stride + in_start_offset;
        // PHASE 4: FINAL WRITES WITH SMALL WINDOW
        for oi in (phase_4_start..phase_3_end).step_by(stride) {
            let li = oi - li_off;
            sum -= invec[li].clone();
            current_window_size -= F::one();
            outvec[oi] = sum.clone() / current_window_size.clone();
        }
    }

    // ----------------------------------------------------------------
    pub fn box_along_rows_float<F: MiniFloat>(
        input: &[F],      // matrix as num_rows x num_cols in row-major order
        output: &mut [F], // matrix as num_rows x num_cols in row-major order
        n_rows: usize,
        n_cols: usize,
        window_size: usize,
    ) {
        for i in 0..n_rows {
            box_one_d_float(input, i * n_cols, output, n_cols, 1, window_size);
        }
    }

    // ----------------------------------------------------------------
    pub fn box_along_cols_float<F: MiniFloat>(
        input: &[F],      // matrix as num_rows x num_cols in row-major order
        output: &mut [F], // matrix as num_rows x num_cols in row-major order
        n_rows: usize,
        n_cols: usize,
        window_size: usize,
    ) {
        for j in 0..n_cols {
            box_one_d_float(input, j, output, n_rows, n_cols, window_size);
        }
    }

    pub fn jarosz_filter_float<F: MiniFloat>(
        buffer1: &mut [F], // matrix as num_rows x num_cols in row-major order
        num_rows: usize,
        num_cols: usize,
        window_size_along_rows: usize,
        window_size_along_cols: usize,
        nreps: usize,
    ) {
        let mut temp_buf = Vec::new();
        temp_buf.resize(buffer1.len(), F::zero());
        for _ in 0..nreps {
            box_along_rows_float(
                buffer1,
                temp_buf.as_mut_slice(),
                num_rows,
                num_cols,
                window_size_along_rows,
            );
            box_along_cols_float(
                temp_buf.as_slice(),
                buffer1,
                num_rows,
                num_cols,
                window_size_along_cols,
            );
        }
    }

    pub fn compute_jarosz_filter_window_size(old_dimension: usize, new_dimension: usize) -> usize {
        (old_dimension + 2 * new_dimension - 1) / (2 * new_dimension)
    }
}

// d matrix for DCT, compared to be identical with the reference implementation lookup table
fn d_value<R: MiniFloat>(i: i32, j: i32, n: i32) -> R {
    let n1 = R::from_i32(n);
    let i1 = R::from_i32(i);
    let j1 = R::from_i32(j);
    let one = R::one();
    let two = one.clone() + one.clone();
    let pi = R::pi();

    (two.clone() / n1.clone()).sqrt() * (pi / (two.clone() * n1) * i1 * (two * j1 + one)).cos()
}

// precompute a tent filter weights matrix using impulse response on a reference implementation
//
// it is intended to be symmetric so we don't need two
// the return is (offset, (effective_rows, effective_cols), weights)
// where weights[di][dj] is the effect of the pixel at (i - offset + di, j - offset + dj) on the output pixel located at (i, j)
//
// it can be padded to SIMD register width to enable vectorized processing
fn tent_filter_weights<R: MiniFloat>(
    old_dimension: usize,
    new_dimension: usize,
) -> (usize, (usize, usize), Vec<R>) {
    let window_size = mean_box::compute_jarosz_filter_window_size(old_dimension, new_dimension);
    assert!(window_size > 0);
    let effective_rows = window_size * 2 + 1;
    let effective_cols = window_size * 2 + 1;
    let mut tmp = Vec::with_capacity(effective_rows * effective_cols);
    tmp.resize(effective_rows * effective_cols, R::zero());

    let mut output = Vec::with_capacity(effective_rows * effective_cols * 4);
    output.resize(effective_rows * effective_cols * 4, R::zero());
    for di in 0..effective_rows {
        for dj in 0..effective_cols {
            tmp.fill_with(|| R::zero());

            tmp[di * effective_cols + dj] = R::one();

            let mut sum = R::zero();
            for x in tmp.iter() {
                sum += x.clone();
            }
            tmp.iter_mut().for_each(|x| *x = x.clone() / sum.clone());

            mean_box::jarosz_filter_float(
                tmp.as_mut_slice(),
                effective_rows,
                effective_cols,
                window_size,
                window_size,
                2,
            );

            let middle = &tmp[window_size * effective_cols + window_size];

            if middle.clone().to_f32() > 0.0 {
                output[di * effective_cols + dj] = middle.clone();
            }
        }
    }

    (
        (effective_cols - 1) / 2,
        (effective_rows, effective_cols),
        output,
    )
}

fn lerp(a: f64, b: f64, p: u32, n: u32) -> f64 {
    let spaces = n + 1;
    let step = (b - a) / spaces as f64;
    a + step * (p + 1) as f64
}

fn main() {
    let mut out_dct_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    out_dct_path.push("dct_matrix.rs");
    let mut out_tent_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    out_tent_path.push("tent_filter_weights.rs");
    let mut out_convolution_offset_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    out_convolution_offset_path.push("convolution_offset.rs");
    let mut dihedral_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    dihedral_path.push("dihedral.rs");

    let mut dihedral_file = File::create(dihedral_path).unwrap();
    dihedral_file.set_len(0).unwrap();
    writeln!(
        dihedral_file,
        "/// Lookup table for flipping a byte horizontally"
    )
    .unwrap();
    writeln!(dihedral_file, "const FLIP_U8: [u8; 256] = [").unwrap();
    for input in 0..=u8::MAX {
        let mut out = 0;
        for j in 0..8 {
            let on = (1 << (7 - j)) & input;
            if on != 0 {
                out |= 1 << j;
            }
        }
        write!(dihedral_file, "0b{:08b},", out).unwrap();
    }
    writeln!(dihedral_file, "];").unwrap();
    writeln!(dihedral_file, "").unwrap();
    writeln!(
        dihedral_file,
        "/// Lookup table for expanding a byte value into 8 masks"
    )
    .unwrap();
    writeln!(dihedral_file, "#[allow(unused)]").unwrap();
    writeln!(dihedral_file, "const EXTRACT_BITS: [[bool; 8]; 256] = [").unwrap();
    for j in 0..=u8::MAX {
        writeln!(dihedral_file, "[").unwrap();
        for i in 0..8 {
            let mask = 0b10000000 >> i;
            write!(dihedral_file, "{},", (j & mask) != 0).unwrap();
        }
        writeln!(dihedral_file, "], ").unwrap();
    }
    writeln!(dihedral_file, "];").unwrap();
    writeln!(dihedral_file, "").unwrap();

    dihedral_file.flush().unwrap();

    let mut file = File::create(out_dct_path).unwrap();
    file.set_len(0).unwrap();
    let nrows = 16;
    let ncols = 127;
    let nelems = nrows * ncols;
    writeln!(file, "/// The DCT matrix number of rows").unwrap();
    writeln!(file, "pub const DCT_MATRIX_NROWS: usize = {};", nrows).unwrap();
    writeln!(file, "/// The DCT matrix number of columns").unwrap();
    writeln!(file, "pub const DCT_MATRIX_NCOLS: usize = {};", ncols).unwrap();
    writeln!(file, "/// The DCT matrix number of elements").unwrap();
    writeln!(file, "pub const DCT_MATRIX_NELEMS: usize = {};", nelems).unwrap();
    writeln!(file, "/// The DCT matrix in row-major order").unwrap();
    writeln!(file, "pub static DCT_MATRIX_RMAJOR: [f32; {}] = [", nelems).unwrap();
    for i in 1..=16 {
        for j in 0..127 {
            let v: MultiPrecF = d_value(i, j, ncols);
            writeln!(
                file,
                "    f32::from_bits({}), // {}",
                (v.clone().to_f32()).to_bits(),
                v.to_f32()
            )
            .unwrap();
        }
    }
    writeln!(file, "];").unwrap();

    writeln!(file, "").unwrap();
    writeln!(file, "/// The DCT matrix in row-major order (f64)").unwrap();
    writeln!(
        file,
        "pub static DCT_MATRIX_RMAJOR_64: [f64; {}] = [",
        nelems
    )
    .unwrap();
    for i in 1..=16 {
        for j in 0..127 {
            let v: MultiPrecF = d_value(i, j, ncols);
            writeln!(
                file,
                "    f64::from_bits({}),  // {}",
                (v.clone().to_f64()).to_bits(),
                v.clone().to_f64()
            )
            .unwrap();
        }
    }
    writeln!(file, "];").unwrap();

    writeln!(file, "").unwrap();

    file.flush().unwrap();

    let (offset, (effective_rows, effective_cols), impulse_response) =
        tent_filter_weights::<MultiPrecF>(512, 127);

    assert_eq!((offset, (effective_rows, effective_cols)), (3, (7, 7)));

    // now generate 2 variants, a compact version and a version padded to 32x8 pipelines (AVX2, etc)

    let mut tent_file = File::create(out_tent_path).unwrap();
    tent_file.set_len(0).unwrap();
    writeln!(
        tent_file,
        "/// The tent filter column offset (subtract this before applying the filter)"
    )
    .unwrap();
    writeln!(
        tent_file,
        "pub const TENT_FILTER_COLUMN_OFFSET: usize = {};",
        offset
    )
    .unwrap();
    writeln!(
        tent_file,
        "/// The tent filter effective rows (the number of rows in the filter)"
    )
    .unwrap();
    writeln!(
        tent_file,
        "pub const TENT_FILTER_EFFECTIVE_ROWS: usize = {};",
        effective_rows
    )
    .unwrap();
    writeln!(
        tent_file,
        "/// The tent filter effective columns (the number of columns in the filter)"
    )
    .unwrap();
    writeln!(
        tent_file,
        "pub const TENT_FILTER_EFFECTIVE_COLS: usize = {};",
        effective_cols
    )
    .unwrap();

    writeln!(
        tent_file,
        "/// The tent filter impulse response lookup table",
    )
    .unwrap();
    writeln!(
        tent_file,
        "pub static TENT_FILTER_WEIGHTS: [f32; {}] = [",
        effective_rows * effective_cols
    )
    .unwrap();

    for i in 0..effective_rows {
        let row = &impulse_response[i * effective_cols..];
        for j in 0..effective_cols {
            writeln!(
                tent_file,
                "f32::from_bits({}), // {}",
                (row[j].clone().to_f32()).to_bits(),
                row[j].clone().to_f32()
            )
            .unwrap();
        }
    }

    writeln!(tent_file, "];").unwrap();

    writeln!(tent_file, "").unwrap();

    writeln!(
        tent_file,
        "/// The tent filter weights in row-major order padded with zeroes to 8-elements wide",
    )
    .unwrap();
    writeln!(
        tent_file,
        "pub static TENT_FILTER_WEIGHTS_X8: [f32; {}] = [",
        effective_rows * 8,
    )
    .unwrap();
    let padding = 8 - effective_cols;

    for i in 0..effective_rows {
        let row = &impulse_response[i * effective_cols..];
        for j in 0..effective_cols {
            writeln!(
                tent_file,
                "f32::from_bits({}),  // {}",
                (row[j].clone().to_f32()).to_bits(),
                row[j].clone().to_f32()
            )
            .unwrap();
        }
        for _ in 0..padding {
            write!(tent_file, "0.0, ").unwrap();
        }
        writeln!(tent_file).unwrap();
    }
    writeln!(tent_file, "];").unwrap();

    tent_file.flush().unwrap();

    let mut convolution_offset_file = File::create(out_convolution_offset_path).unwrap();
    convolution_offset_file.set_len(0).unwrap();
    writeln!(
        convolution_offset_file,
        "/// A lookup table for linear interpolation of 512 into 64 steps rounded to integer boundaries"
    )
    .unwrap();
    writeln!(
        convolution_offset_file,
        "const CONVOLUTION_OFFSET_512_TO_127: [usize; 127] = [",
    )
    .unwrap();

    for out_i in 0..127 {
        let in_i = lerp(0.0, 512.0, out_i, 127);
        writeln!(
            convolution_offset_file,
            "    {}, // {} -> {}",
            (in_i.round() as usize).max(0).min(511),
            out_i,
            in_i
        )
        .unwrap();
    }
    writeln!(convolution_offset_file, "];").unwrap();

    convolution_offset_file.flush().unwrap();
}
