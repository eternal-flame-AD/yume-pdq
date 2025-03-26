use generic_array::{ArrayLength, GenericArray};
use num_traits::ToPrimitive;

/// Dump an image to a PPM file, scale is 0-255
///
/// Please rescale your image to 0-255 before dumping.
pub fn dump_image<W: ArrayLength, H: ArrayLength, F: ToPrimitive>(
    filename: &str,
    input: &GenericArray<GenericArray<F, W>, H>,
) -> std::io::Result<()> {
    use std::io::Write;

    let mut of = std::fs::File::create(filename)?;
    writeln!(of, "P6")?;
    writeln!(of, "{} {}", W::USIZE, H::USIZE)?;
    writeln!(of, "255")?;

    for i in 0..W::USIZE {
        for j in 0..H::USIZE {
            let val = input[i][j].to_u8().unwrap();
            of.write_all(&[val, val, val])?;
        }
    }
    Ok(())
}
