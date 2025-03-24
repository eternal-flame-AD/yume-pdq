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
    fn is_pos(&self) -> bool;
    fn zero() -> Self;
    fn one() -> Self;
    fn sqrt(self) -> Self;
    fn pi() -> Self;
    fn cos(self) -> Self;
    fn from_u32(u: u32) -> Self;
    fn to_f32(self) -> f32;
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
    fn is_pos(&self) -> bool {
        self.0.is_sign_positive()
    }

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

    fn from_u32(u: u32) -> Self {
        Self(RFloat::with_val(PREC, u))
    }

    fn to_f32(self) -> f32 {
        self.0.to_f32()
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

fn d_value<R: MiniFloat>(i: u32, j: u32, n: u32) -> R {
    let n1 = R::from_u32(n);
    let i1 = R::from_u32(i);
    let j1 = R::from_u32(j);
    let one = R::one();
    let two = one.clone() + one.clone();
    let pi = R::pi();

    (two.clone() / n1.clone()).sqrt() * (pi / (two.clone() * n1) * i1 * (two * j1 + one)).cos()
}

/*
Brainstorm sub-pixel tent filter:

We can create sub-pixel lookup tables:

Originally we have this and lookup tables are only accurate at exact convolution centers that map dead on an input pixel:

#---------#
|         |
|         |
|         |
|         |
|         |
#---------#

We can divide the unit square 3x3 sub-pixels, 1 lookup table -> 9 lookup tables (+~114k when padded to 32x8 SIMD) but much more precision:

Finally add the index of the lookup table to use in the convolution lookup table

#--*--*--#
|         |
*--*--*--*
|         |
*--*--*--*
|         |
#--*--*--#

*/

// The offset of the impulse for each lookup table. x goes to the right and y goes down
//
// 0 1 2
// 3 4 5
// 6 7 8
#[derive(Debug, Clone)]
struct LerpPad<R: MiniFloat> {
    pad: [R; 9],
}

impl<R: MiniFloat> LerpPad<R> {
    fn zeros() -> Self {
        Self {
            pad: std::array::from_fn(|_| R::zero()),
        }
    }

    fn compute_2d_lerp(impulse_offset: (R, R)) -> Self {
        let mut lerp_pad = Self::zeros();
        let one = R::one();
        let (p_11, p_12, p_21, p_22) = match (impulse_offset.0.is_pos(), impulse_offset.1.is_pos())
        {
            (true, true) => (4, 5, 7, 8),
            (true, false) => (1, 2, 4, 5),
            (false, true) => (3, 4, 6, 7),
            (false, false) => (0, 1, 3, 4),
        };
        let dx;
        let dy;
        if impulse_offset.0.is_pos() {
            dx = impulse_offset.0;
        } else {
            dx = impulse_offset.0 + one.clone();
        }
        if impulse_offset.1.is_pos() {
            dy = impulse_offset.1;
        } else {
            dy = impulse_offset.1 + one.clone();
        }
        lerp_pad.pad[p_11] = (R::one() - dx.clone()) * (R::one() - dy.clone());
        lerp_pad.pad[p_12] = dx.clone() * (R::one() - dy.clone());
        lerp_pad.pad[p_21] = (R::one() - dx.clone()) * dy.clone();
        lerp_pad.pad[p_22] = dx.clone() * dy.clone();

        let all_positive = lerp_pad.pad.iter().all(|x| x.is_pos() || *x == R::zero());
        assert!(all_positive, "lerp_pad: {:?}", lerp_pad);

        lerp_pad
    }

    fn compute(
        &self,
        x00: &R,
        x01: &R,
        x02: &R,
        x10: &R,
        x11: &R,
        x12: &R,
        x20: &R,
        x21: &R,
        x22: &R,
    ) -> R {
        let mut sum = R::zero();
        sum += x00.clone() * self.pad[0].clone();
        sum += x01.clone() * self.pad[1].clone();
        sum += x02.clone() * self.pad[2].clone();
        sum += x10.clone() * self.pad[3].clone();
        sum += x11.clone() * self.pad[4].clone();
        sum += x12.clone() * self.pad[5].clone();
        sum += x20.clone() * self.pad[6].clone();
        sum += x21.clone() * self.pad[7].clone();
        sum += x22.clone() * self.pad[8].clone();
        sum
    }
}

// precompute a padded tent filter weights matrix using impulse response on a reference implementation
//
// it is intended to be symmetric so we don't need two
// the return is (offset, (effective_rows, effective_cols), weights)
// where weights[di][dj] is the effect of the pixel at (i - offset + di, j - offset + dj) on the output pixel located at (i, j)
//
// it can be padded to SIMD register width to enable vectorized processing
fn tent_filter_weights<R: MiniFloat>(
    old_dimension: usize,
    new_dimension: usize,
    // the sub-pixel offset of the impulse for the lookup table
    // say we want to convolve on (7.2, 6.8) with a radius of 2 (ie. 5x5 convolution where the center is slightly top-left, effective window is (5.2..=9.2, 4.8..=8.8))
    // then the impulse offset is (0.2, -0.2), and the impulse intended to be delivered to (0, 0) w.r.t. the output will now be delivered at (0.2, -0.2) and distributed to concrete pixels by 2D lerping
    impulse_offset: (R, R),
) -> (usize, (usize, usize), Vec<R>) {
    let window_size = mean_box::compute_jarosz_filter_window_size(old_dimension, new_dimension);
    let effective_rows = window_size * 2 + 3;
    let effective_cols = window_size * 2 + 3;
    let lerp_pad = LerpPad::compute_2d_lerp(impulse_offset);
    let mut tmp = Vec::with_capacity((window_size * 2 + 3) * (window_size * 2 + 3));
    tmp.resize((window_size * 2 + 3) * (window_size * 2 + 3), R::zero());

    let mut output = Vec::with_capacity(effective_rows * effective_cols * 4);
    output.resize(effective_rows * effective_cols * 4, R::zero());
    for di in 0..effective_rows {
        for dj in 0..effective_cols {
            tmp.fill_with(|| R::zero());

            tmp[di * (window_size * 2 + 3) + dj] = R::one();

            let mut sum = R::zero();
            for x in tmp.iter() {
                sum += x.clone();
            }
            tmp.iter_mut().for_each(|x| *x = x.clone() / sum.clone());

            mean_box::jarosz_filter_float(
                tmp.as_mut_slice(),
                window_size * 2 + 3,
                window_size * 2 + 3,
                window_size,
                window_size,
                2,
            );

            let middle = lerp_pad.compute(
                &tmp[(window_size - 1) * (window_size * 2 + 3) + window_size - 1],
                &tmp[window_size * (window_size * 2 + 3) + window_size],
                &tmp[(window_size + 1) * (window_size * 2 + 3) + window_size + 1],
                &tmp[(window_size - 1) * (window_size * 2 + 3) + window_size - 1],
                &tmp[window_size * (window_size * 2 + 3) + window_size],
                &tmp[(window_size + 1) * (window_size * 2 + 3) + window_size + 1],
                &tmp[(window_size - 1) * (window_size * 2 + 3) + window_size - 1],
                &tmp[window_size * (window_size * 2 + 3) + window_size],
                &tmp[(window_size + 1) * (window_size * 2 + 3) + window_size + 1],
            );

            if middle.clone().to_f32() > 0.0 {
                output[di * effective_cols + dj] = middle;
            }
        }
    }

    let trim = output
        .chunks_exact(effective_cols)
        .take_while(|x| x.iter().all(|y| y.clone().to_f32() == 0.0))
        .count();

    let mut output_trimmed =
        Vec::with_capacity((effective_rows - trim * 2) * (effective_cols - trim * 2));
    output_trimmed.resize(
        (effective_rows - trim * 2) * (effective_cols - trim * 2),
        R::zero(),
    );

    for i in 0..effective_rows - trim * 2 {
        for j in 0..effective_cols - trim * 2 {
            output_trimmed[i * (effective_cols - trim * 2) + j] =
                output[(i + trim) * effective_cols + j + trim].clone();
        }
    }

    (
        (effective_cols - trim * 2 - 1) / 2,
        (effective_rows - trim * 2, effective_cols - trim * 2),
        output_trimmed,
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

    let mut file = File::create(out_dct_path).unwrap();
    file.set_len(0).unwrap();
    let nrows = 16;
    let ncols = 64;
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
        for j in 0..64 {
            let v: MultiPrecF = d_value(i, j, ncols);
            writeln!(file, "    f32::from_bits({}),", (v.to_f32()).to_bits()).unwrap();
        }
    }
    writeln!(file, "];").unwrap();
    file.flush().unwrap();

    let (offset, (effective_rows, effective_cols), _) =
        tent_filter_weights::<MultiPrecF<192>>(512, 64, (MultiPrecF::zero(), MultiPrecF::zero()));

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
        "/// The tent filter weights in row-major order padded with zeroes to 8-elements wide",
    )
    .unwrap();
    writeln!(
        tent_file,
        "pub static TENT_FILTER_WEIGHTS_X8: [[f32; {}]; 9] = [",
        effective_rows * 8,
    )
    .unwrap();
    let padding = 8 - effective_cols;
    for subtype in 0..9 {
        writeln!(tent_file, "[").unwrap();
        let zero = MultiPrecF::zero();
        let one = MultiPrecF::one();
        let two = one.clone() + one.clone();
        let half = one.clone() / two.clone();
        let three = two.clone() + one.clone();
        let six = three.clone() + three.clone();
        let sixth = one.clone() / six.clone();
        let neg_threshold = half.clone().neg() + sixth.clone();
        let pos_threshold = half - sixth;
        assert_eq!(neg_threshold.clone().neg(), pos_threshold);
        let offset = &[
            (neg_threshold.clone(), neg_threshold.clone()),
            (zero.clone(), neg_threshold.clone()),
            (pos_threshold.clone(), neg_threshold.clone()),
            (neg_threshold.clone(), zero.clone()),
            (zero.clone(), zero.clone()),
            (pos_threshold.clone(), zero.clone()),
            (neg_threshold.clone(), pos_threshold.clone()),
            (zero.clone(), pos_threshold.clone()),
            (pos_threshold.clone(), pos_threshold.clone()),
        ][subtype];

        let (_, (_, _), weights) = tent_filter_weights::<MultiPrecF<192>>(
            512,
            64,
            (offset.0.clone().neg(), offset.1.clone().neg()),
        );

        for i in 0..effective_rows {
            let row = &weights[i * effective_cols..];
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
        writeln!(tent_file, "],").unwrap();
    }
    writeln!(tent_file, "];").unwrap();
    writeln!(
        tent_file,
        "/// The tent filter impulse response lookup table",
    )
    .unwrap();
    writeln!(
        tent_file,
        "pub static TENT_FILTER_WEIGHTS: [[f32; {}]; 9] = [",
        effective_rows * effective_cols
    )
    .unwrap();

    for subtype in 0..9 {
        writeln!(tent_file, "[").unwrap();
        let zero = MultiPrecF::zero();
        let one = MultiPrecF::one();
        let two = one.clone() + one.clone();
        let half = one.clone() / two.clone();
        let three = two.clone() + one.clone();
        let six = three.clone() + three.clone();
        let sixth = one.clone() / six.clone();
        let neg_threshold = half.clone().neg() + sixth.clone();
        let pos_threshold = half - sixth;
        assert_eq!(neg_threshold.clone().neg(), pos_threshold);

        let offset = &[
            (neg_threshold.clone(), neg_threshold.clone()),
            (zero.clone(), neg_threshold.clone()),
            (pos_threshold.clone(), neg_threshold.clone()),
            (neg_threshold.clone(), zero.clone()),
            (zero.clone(), zero.clone()),
            (pos_threshold.clone(), zero.clone()),
            (neg_threshold.clone(), pos_threshold.clone()),
            (zero.clone(), pos_threshold.clone()),
            (pos_threshold.clone(), pos_threshold.clone()),
        ][subtype];

        let (_, (_, _), weights) = tent_filter_weights::<MultiPrecF<192>>(
            512,
            64,
            (offset.0.clone().neg(), offset.1.clone().neg()),
        );

        for i in 0..effective_rows {
            let row = &weights[i * effective_cols..];
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
        writeln!(tent_file, "],").unwrap();
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
        "const CONVOLUTION_OFFSET_512_TO_64: [usize; 64] = [",
    )
    .unwrap();

    for out_i in 0..64 {
        let in_i = lerp(0.0, 512.0, out_i, 64);
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
    writeln!(
        convolution_offset_file,
        "const CONVOLUTION_LOOKUP_TABLE_INDEX: [u8; 64 * 64] = ["
    )
    .unwrap();
    for i in 0..64 {
        for j in 0..64 {
            const POS_THRESHOLD: f64 = 1.0 / 2.0 - 1.0 / 3.0;
            const NEG_THRESHOLD: f64 = -1.0 / 2.0 + 1.0 / 3.0;

            // 0 ....... 0.5 ...... 1
            // within 0 - 0.5, we split 0.5 - 1/3 to the intermediate spate
            let in_i = lerp(0.0, 512.0, i, 64);
            let in_i_int = (in_i.round() as usize).max(0).min(511);
            let in_i_offset = in_i - in_i_int as f64;
            let in_j = lerp(0.0, 512.0, j, 64);
            let in_j_int = (in_j.round() as usize).max(0).min(511);
            let in_j_offset = in_j - in_j_int as f64;
            let idx_i = if in_i_offset > POS_THRESHOLD && i != 63 {
                0
            } else if in_i_offset < NEG_THRESHOLD && i != 0 {
                2
            } else {
                1
            };
            let idx_j = if in_j_offset > POS_THRESHOLD && j != 63 {
                0
            } else if in_j_offset < NEG_THRESHOLD && j != 0 {
                2
            } else {
                1
            };
            writeln!(
                convolution_offset_file,
                "{}, // {:?} -> {:?} ~> {:?}",
                idx_i * 3 + idx_j,
                (i, j),
                (in_i, in_j),
                (in_i_int, in_j_int),
            )
            .unwrap();
        }
    }
    writeln!(convolution_offset_file, "];").unwrap();
    convolution_offset_file.flush().unwrap();
}
