use core::{cmp::Ordering, fmt::Debug};
use std::path::Path;

use generic_array::{ArrayLength, GenericArray, sequence::GenericSequence};
use num_traits::ToPrimitive;

/// Dump an image to a PPM file, scale is 0-255
///
/// Please rescale your image to 0-255 before dumping.
pub fn dump_image<P: AsRef<Path>, W: ArrayLength, H: ArrayLength, F: ToPrimitive + Debug>(
    filename: P,
    input: &GenericArray<GenericArray<F, W>, H>,
) -> std::io::Result<()> {
    use std::io::Write;

    let path = filename.as_ref();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }

    let mut of = std::fs::File::create(path)?;
    writeln!(of, "P6")?;
    writeln!(of, "{} {}", W::USIZE, H::USIZE)?;
    writeln!(of, "255")?;

    let mut min = f64::MAX;
    let mut max = f64::MIN;

    for i in 0..H::USIZE {
        for j in 0..W::USIZE {
            let val = input[i][j].to_f64().unwrap();
            if val < min && !val.is_nan() {
                min = val;
            }
            if val > max && !val.is_nan() {
                max = val;
            }
        }
    }

    let scale_by = (max - min) / 255.0;
    let offset_by = min;

    for i in 0..H::USIZE {
        for j in 0..W::USIZE {
            let val = input[i][j].to_f64().unwrap();
            if !val.is_finite() {
                of.write_all(if val.is_nan() {
                    &[255, 0, 0]
                } else if val.is_infinite() {
                    if val.is_sign_positive() {
                        &[0, 255, 0]
                    } else {
                        &[0, 0, 255]
                    }
                } else {
                    &[0, 0, 0]
                })?;
                continue;
            }
            let scaled_val = (val - offset_by) / scale_by;
            of.write_all(&[scaled_val as u8, scaled_val as u8, scaled_val as u8])?;
        }
    }
    Ok(())
}

/// Dump a thresholding diagnostic image
pub fn dump_thresholding_diagnostic<
    P: AsRef<Path>,
    W: ArrayLength,
    H: ArrayLength,
    F: PartialOrd,
    // yes, no, unsure
    T: Fn(&F) -> Option<bool>,
>(
    filename: P,
    input: &GenericArray<GenericArray<F, W>, H>,
    is_one: T,
) -> std::io::Result<()> {
    use std::io::Write;

    // find the median independently by ourselves
    let mut ranking: GenericArray<GenericArray<usize, W>, H> =
        GenericArray::generate(|i| GenericArray::generate(|j| i * W::USIZE + j));

    let ranking_flat = unsafe {
        std::slice::from_raw_parts_mut(ranking.as_mut_ptr() as *mut usize, W::USIZE * H::USIZE)
    };

    unsafe {
        ranking_flat.sort_by(|a, b| {
            let a_val = input.as_ptr().cast::<F>().add(*a).read();
            let b_val = input.as_ptr().cast::<F>().add(*b).read();
            a_val.partial_cmp(&b_val).unwrap()
        });
    }

    let num_elements = W::USIZE * H::USIZE;
    let (actual_median_ib_idx, actual_median_ub_idx) = if num_elements % 2 == 0 {
        (
            ranking_flat[num_elements / 2 - 1],
            ranking_flat[num_elements / 2],
        )
    } else {
        (
            ranking_flat[num_elements / 2],
            ranking_flat[num_elements / 2],
        )
    };

    let actual_median_ib = &input[actual_median_ib_idx / W::USIZE][actual_median_ib_idx % W::USIZE];
    let actual_median_ub = &input[actual_median_ub_idx / W::USIZE][actual_median_ub_idx % W::USIZE];

    let path = filename.as_ref();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }

    let mut of = std::fs::File::create(path)?;
    writeln!(of, "P6")?;
    writeln!(of, "{} {}", W::USIZE, H::USIZE)?;
    writeln!(of, "255")?;

    // according to AI this is more or less colorblind friendly and still intuitive
    const RED: [u8; 3] = [230, 25, 75]; // invalid number (bright red - universally indicates error)
    const ORANGE: [u8; 3] = [245, 130, 48]; // overshoot by claiming above median when it's below (warm orange - indicates "hot"/over)
    const BLUE: [u8; 3] = [0, 130, 200]; // correctly identified as above median (blue - positive)
    const YELLOW: [u8; 3] = [255, 225, 25]; // correctly identified as below median (yellow - caution)
    const BROWN: [u8; 3] = [170, 110, 40]; // undershoot by claiming below median when it's above (cool brown - indicates "cold"/under)
    const GRAY: [u8; 3] = [128, 128, 128]; // refused to identify (neutral gray - no decision made)
    const WHITE: [u8; 3] = [255, 255, 255]; // this element is right on the median

    for i in 0..H::USIZE {
        for j in 0..W::USIZE {
            let val = &input[i][j];
            let input_thresholding = is_one(&val);
            match (
                val.partial_cmp(&actual_median_ib),
                val.partial_cmp(&actual_median_ub),
                input_thresholding,
            ) {
                (Some(Ordering::Equal), _, _) => {
                    of.write_all(&WHITE)?;
                }
                (_, Some(Ordering::Equal), _) => {
                    of.write_all(&WHITE)?;
                }
                (Some(Ordering::Greater), Some(Ordering::Less), _) => {
                    of.write_all(&WHITE)?;
                }
                (Some(Ordering::Less), Some(Ordering::Greater), _) => {
                    of.write_all(&WHITE)?;
                }
                (None, _, _) | (_, None, _) => {
                    // uncomparable
                    of.write_all(&RED)?;
                }
                (_, _, None) => {
                    // refused to identify
                    of.write_all(&GRAY)?;
                }
                (_, Some(Ordering::Greater), Some(true)) => {
                    of.write_all(&BLUE)?;
                }
                (Some(Ordering::Less), _, Some(false)) => {
                    of.write_all(&YELLOW)?;
                }
                (Some(Ordering::Less), _, Some(true)) => {
                    of.write_all(&ORANGE)?;
                }
                (_, Some(Ordering::Greater), Some(false)) => {
                    of.write_all(&BROWN)?;
                }
            }
        }
    }

    Ok(())
}
