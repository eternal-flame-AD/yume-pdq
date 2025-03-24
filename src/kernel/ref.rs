///! Compute PDQ hash of an image.

///! The PDQ algorithm was developed and open-sourced by Facebook (now Meta) in 2019.

///! It specifies a transformation which converts images into a binary format ('PDQ Hash') whereby 'perceptually similar’ images produce similar outputs.

///! It was designed to offer an industry standard for representing images to collaborate on threat mitigation.

//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

pub fn compute_jarosz_filter_window_size(old_dimension: usize, new_dimension: usize) -> usize {
    (old_dimension + 2 * new_dimension - 1) / (2 * new_dimension)
}

pub fn jarosz_filter_float(
    buffer1: &mut [f32; 512 * 512], // matrix as num_rows x num_cols in row-major order

    num_rows: usize,

    num_cols: usize,

    window_size_along_rows: usize,

    window_size_along_cols: usize,

    nreps: usize,
) {
    let mut temp_buf = [0.0; 512 * 512];

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

fn box_one_d_float(
    invec: &[f32],

    in_start_offset: usize,

    outvec: &mut [f32],

    vector_length: usize,

    stride: usize,

    full_window_size: usize,
) {
    let half_window_size = (full_window_size + 2) / 2; // 7->4, 8->5

    let phase_1_nreps = half_window_size - 1;

    let phase_2_nreps = full_window_size - half_window_size + 1;

    let oi_off = phase_1_nreps * stride;

    let li_off = phase_2_nreps * stride;

    let mut sum = 0.0;

    let mut current_window_size = 0.0;

    let phase_1_end = oi_off + in_start_offset;

    // PHASE 1: ACCUMULATE FIRST SUM NO WRITES

    for ri in (in_start_offset..phase_1_end).step_by(stride) {
        let value = invec[ri];

        sum += value;

        current_window_size += 1.0;
    }

    let phase_2_end = full_window_size * stride + in_start_offset;

    // PHASE 2: INITIAL WRITES WITH SMALL WINDOW

    for ri in (phase_1_end..phase_2_end).step_by(stride) {
        let oi = ri - oi_off;

        sum += invec[ri];

        current_window_size += 1.0;

        outvec[oi] = sum / current_window_size;
    }

    let phase_3_end = vector_length * stride + in_start_offset;

    // PHASE 3: WRITES WITH FULL WINDOW

    for ri in (phase_2_end..phase_3_end).step_by(stride) {
        let oi = ri - oi_off;

        let li = oi - li_off;

        sum += invec[ri];

        sum -= invec[li];

        outvec[oi] = sum / (current_window_size);
    }

    let phase_4_start = (vector_length - half_window_size + 1) * stride + in_start_offset;

    // PHASE 4: FINAL WRITES WITH SMALL WINDOW

    for oi in (phase_4_start..phase_3_end).step_by(stride) {
        let li = oi - li_off;

        sum -= invec[li];

        current_window_size -= 1.0;

        outvec[oi] = sum / current_window_size;
    }
}

// ----------------------------------------------------------------

fn box_along_rows_float(
    input: &[f32], // matrix as num_rows x num_cols in row-major order

    output: &mut [f32], // matrix as num_rows x num_cols in row-major order

    n_rows: usize,

    n_cols: usize,

    window_size: usize,
) {
    for i in 0..n_rows {
        box_one_d_float(input, i * n_cols, output, n_cols, 1, window_size);
    }
}

// ----------------------------------------------------------------

fn box_along_cols_float(
    input: &[f32], // matrix as num_rows x num_cols in row-major order

    output: &mut [f32], // matrix as num_rows x num_cols in row-major order

    n_rows: usize,

    n_cols: usize,

    window_size: usize,
) {
    for j in 0..n_cols {
        box_one_d_float(input, j, output, n_rows, n_cols, window_size);
    }
}

// ----------------------------------------------------------------

pub fn decimate_float<const OUT_NUM_ROWS: usize, const OUT_NUM_COLS: usize>(
    input: &[f32], // matrix as in_num_rows x in_num_cols in row-major order

    in_num_rows: usize,

    in_num_cols: usize,

    output: &mut [[f32; OUT_NUM_COLS]; OUT_NUM_ROWS],
) {
    // target centers not corners:

    for outi in 0..OUT_NUM_ROWS {
        let ini = ((outi * 2 + 1) * in_num_rows) / (OUT_NUM_ROWS * 2);

        for outj in 0..OUT_NUM_COLS {
            let inj = ((outj * 2 + 1) * in_num_cols) / (OUT_NUM_COLS * 2);

            output[outi][outj] = input[ini * in_num_cols + inj];
        }
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

// Quickly find the median

fn torben_median(m: &[f32]) -> Option<f32> {
    let mut min = m.iter().cloned().reduce(f32::min)?;

    let mut max = m.iter().cloned().reduce(f32::max)?;

    let half = (m.len() + 1) / 2;

    loop {
        let guess = (min + max) / 2.0;

        let mut less = 0;

        let mut greater = 0;

        let mut equal = 0;

        let mut maxltguess = min;

        let mut mingtguess = max;

        for val in m {
            if *val < guess {
                less += 1;

                if *val > maxltguess {
                    maxltguess = *val;
                }
            } else if *val > guess {
                greater += 1;

                if *val < mingtguess {
                    mingtguess = *val;
                }
            } else {
                equal += 1;
            }
        }

        if less <= half && greater <= half {
            return Some(if less >= half {
                maxltguess
            } else if less + equal >= half {
                guess
            } else {
                mingtguess
            });
        } else if less > greater {
            max = maxltguess;
        } else {
            min = mingtguess;
        }
    }
}

fn pdq_buffer16x16_to_bits(input: &[f32; DCT_OUTPUT_MATRIX_SIZE]) -> [u8; HASH_LENGTH] {
    let dct_median = torben_median(input).unwrap();

    let mut hash = [0; HASH_LENGTH];

    for i in 0..HASH_LENGTH {
        let mut byte = 0;

        for j in 0..8 {
            let val = input[i * 8 + j];

            if val > dct_median {
                byte |= 1 << j;
            }
        }

        hash[HASH_LENGTH - i - 1] = byte;
    }

    hash
}
