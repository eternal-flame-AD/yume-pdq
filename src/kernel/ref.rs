// readapted straight from pdqhash crate, not part of a user-facing build, mainly used for testing kernels

pub fn compute_jarosz_filter_window_size(old_dimension: usize, new_dimension: usize) -> usize {
    old_dimension.div_ceil(2 * new_dimension)
}

pub fn jarosz_filter_float<
    F: num_traits::FromPrimitive
        + std::ops::Div<Output = F>
        + std::ops::AddAssign<F>
        + std::ops::SubAssign<F>
        + Clone
        + Default,
>(
    buffer1: &mut [F; 512 * 512], // matrix as num_rows x num_cols in row-major order

    num_rows: usize,

    num_cols: usize,

    window_size_along_rows: usize,

    window_size_along_cols: usize,

    nreps: usize,
) {
    let mut temp_buf = Vec::<F>::with_capacity(512 * 512);
    temp_buf.resize(512 * 512, F::default());

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

// This is called from two places, one has a constant stride, the other a variable stride

// It should compile a version for each.

#[inline(always)]
#[allow(clippy::manual_midpoint)]
fn box_one_d_float<
    F: num_traits::FromPrimitive
        + std::ops::Div<Output = F>
        + std::ops::AddAssign<F>
        + std::ops::SubAssign<F>
        + Clone
        + Default,
>(
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

    let mut sum = F::default();

    let mut current_window_size = F::default();

    let phase_1_end = oi_off + in_start_offset;

    // PHASE 1: ACCUMULATE FIRST SUM NO WRITES

    for ri in (in_start_offset..phase_1_end).step_by(stride) {
        let value = invec[ri].clone();

        sum += value;

        current_window_size += F::from_f32(1.0).unwrap();
    }

    let phase_2_end = full_window_size * stride + in_start_offset;

    // PHASE 2: INITIAL WRITES WITH SMALL WINDOW

    for ri in (phase_1_end..phase_2_end).step_by(stride) {
        let oi = ri - oi_off;

        sum += invec[ri].clone();

        current_window_size += F::from_f32(1.0).unwrap();

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

        current_window_size += F::from_f32(-1.0).unwrap();

        outvec[oi] = sum.clone() / current_window_size.clone();
    }
}

// ----------------------------------------------------------------

fn box_along_rows_float<
    F: num_traits::FromPrimitive
        + std::ops::Div<Output = F>
        + std::ops::AddAssign<F>
        + std::ops::SubAssign<F>
        + Clone
        + Default,
>(
    input: &[F], // matrix as num_rows x num_cols in row-major order

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

fn box_along_cols_float<
    F: num_traits::FromPrimitive
        + std::ops::Div<Output = F>
        + std::ops::AddAssign<F>
        + std::ops::SubAssign<F>
        + Clone
        + Default,
>(
    input: &[F], // matrix as num_rows x num_cols in row-major order

    output: &mut [F], // matrix as num_rows x num_cols in row-major order

    n_rows: usize,

    n_cols: usize,

    window_size: usize,
) {
    for j in 0..n_cols {
        box_one_d_float(input, j, output, n_rows, n_cols, window_size);
    }
}

// ----------------------------------------------------------------

// This is all heuristic (see the PDQ hashing doc). Quantization matters since

// we want to count *significant* gradients, not just the some of many small

// ones. The constants are all manually selected, and tuned as described in the

// document.

fn pdq_image_domain_quality_metric<const OUT_NUM_ROWS: usize, const OUT_NUM_COLS: usize>(
    buffer64x64: &[[f32; OUT_NUM_COLS]; OUT_NUM_ROWS],
) -> f32 {
    let mut gradient_sum = 0.0;

    for i in 0..(OUT_NUM_ROWS - 1) {
        for j in 0..OUT_NUM_COLS {
            let u = buffer64x64[i][j];

            let v = buffer64x64[i + 1][j];

            let d = (u - v) / 255.;

            gradient_sum += d.abs();
        }
    }

    for i in 0..OUT_NUM_ROWS {
        for j in 0..(OUT_NUM_COLS - 1) {
            let u = buffer64x64[i][j];

            let v = buffer64x64[i][j + 1];

            let d = (u - v) / 255.;

            gradient_sum += d.abs();
        }
    }

    // Heuristic scaling factor.

    let quality = gradient_sum / 90.;

    if quality > 1.0 { 1.0 } else { quality }
}

const BUFFER_W_H: usize = 64;

const DCT_OUTPUT_W_H: usize = 16;

const DCT_OUTPUT_MATRIX_SIZE: usize = DCT_OUTPUT_W_H * DCT_OUTPUT_W_H;

const HASH_LENGTH: usize = DCT_OUTPUT_MATRIX_SIZE / 8;
