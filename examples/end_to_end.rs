//! this demo will demonstrate the robustness of perceptual hash matching, compatibility of yume-pdq hash with officially endorsed implementation,
//! and a real-demo of matching a mutated and dihedrally transformed image from a synthetic database using the
//! dihedral derivation feature of yume-pdq and the CPU matching backend.
use generic_array::{
    GenericArray,
    sequence::Flatten,
    typenum::{U32, U512, U2048},
};
use image::{DynamicImage, imageops};
use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};
use yume_pdq::{
    PDQHash,
    alignment::Align8,
    kernel::{Kernel, SquareGenericArrayExt},
    matching::{CpuMatcher, PDQMatcher},
};

const AAA_ORIG_JPG_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test-data/aaa-orig.jpg"
));

type All8Dihedrals = [PDQHash; 8];

fn main() {
    let mut rng = SmallRng::seed_from_u64(u64::from_be_bytes(*b"yumeYUME"));
    let test_image = image::load_from_memory(AAA_ORIG_JPG_DATA).expect("failed to load test image");
    let test_image_512x512 = test_image
        .resize_exact(512, 512, image::imageops::FilterType::Triangle)
        .to_luma8()
        .into_flat_samples()
        .as_slice()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<_>>();
    let pixels_512x512: &GenericArray<GenericArray<f32, U512>, U512> =
        GenericArray::from_slice(test_image_512x512.as_slice()).unflatten_square_ref();

    let test_image2 =
        pdqhash::image::load_from_memory(AAA_ORIG_JPG_DATA).expect("failed to load test image");

    // hypothetically, someone put this hash in a database, using an exact match
    // and we are not sure how they did the initial 512x512 luma8, so the hash will certainly be different but comparable
    let (official_hash, official_quality) =
        pdqhash::generate_pdq(&test_image2).expect("failed to generate hash");
    println!("official hash: {:02x?}", official_hash);

    assert_eq!(official_quality, 1.0);

    // now let's try to put the original, untransformed, official hash in a "database" and get it back
    // the cpu matcher lookup in 2048 chunks, in reality you would repeat the last entry until you have whole chunks
    // (this also allows you to do in-place insertion or deletion by substitution)
    let mut database = Vec::new();
    let insertion_index = rng.random_range(0..4096);
    for chunk_index in 0..2 {
        let mut chunk = unsafe {
            let mut x = Box::<Align8<GenericArray<GenericArray<u8, U32>, U2048>>>::new_uninit();
            x.assume_init_mut().flatten().fill(0);
            x.assume_init()
        };
        for i in 0..2048 {
            if i + chunk_index * 2048 == insertion_index {
                chunk[i] = GenericArray::from_array(official_hash);
            } else {
                let mut random = GenericArray::default();
                rng.fill_bytes(&mut random);
                chunk[i] = random;
            }
        }
        database.push(chunk);
    }

    // let's generate our own hash
    let mut kernel = yume_pdq::kernel::smart_kernel();
    println!("kernel: {:?}", kernel.ident());

    let mut threshold = 0.0;
    let mut buf1 = Box::default();
    let mut pdqf = GenericArray::default();
    let mut hash = GenericArray::default();
    let quality = yume_pdq::hash_get_threshold(
        &mut kernel,
        pixels_512x512,
        &mut threshold,
        &mut hash,
        &mut buf1,
        &mut GenericArray::default(),
        &mut pdqf,
    );

    println!("our hash: {:02x?}", hash);
    assert_eq!(quality, 1.0);

    // first do a trivial comparison
    let distance = hash
        .flatten()
        .iter()
        .zip(official_hash.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum::<u32>();

    // we will add 2 to the tolerance of a "precise" implementation because the initial scaling down to 512x512is not exactly the same
    // using 10 passes on my machine but adding 2 to make sure it doesn't fail on other people's machines
    assert!(distance < 12, "distance is too large");

    let mut all_dihedrals = All8Dihedrals::default();
    all_dihedrals[0] = hash;
    let mut i = 1;
    yume_pdq::visit_dihedrals(
        &mut kernel,
        &mut pdqf,
        &mut hash,
        threshold,
        |dihedral, _, (_, _, hash)| {
            println!("hash for dihedral {:?}: {:02x?}", dihedral, hash);
            all_dihedrals[i] = hash.clone();
            i += 1;
            Ok::<(), ()>(())
        },
    )
    .expect("failed to visit dihedrals");

    // now let's rotate the image 90 degrees, flip it horizontally, crop it by 1%, blur, add some noise, and see if we can find it
    let mut new_image = imageops::blur(
        &imageops::flip_horizontal(&imageops::rotate90(&test_image)),
        5.0,
    );
    let new_image_pdqhash = pdqhash::image::imageops::flip_horizontal(
        &pdqhash::image::imageops::rotate90(&test_image2),
    );
    // the pdq hash crate doesn't support dihedral derivation, so we need to generate the hash again
    let new_image_pdqhash =
        pdqhash::generate_pdq(&pdqhash::image::DynamicImage::ImageRgba8(new_image_pdqhash))
            .expect("failed to generate hash")
            .0;
    let original_width = new_image.width();
    let original_height = new_image.height();
    let mut new_image = imageops::crop(
        &mut new_image,
        5,
        5,
        original_width - 10,
        original_height - 10,
    )
    .to_image();
    // 1% of the pixels are replaced with random noise
    for _noise in 0..(new_image.width() * new_image.height() / 100) {
        let x = rng.random_range(0..new_image.width());
        let y = rng.random_range(0..new_image.height());
        let pixel = image::Rgba([
            rng.random_range(0..=255),
            rng.random_range(0..=255),
            rng.random_range(0..=255),
            255,
        ]);
        new_image.put_pixel(x, y, pixel);
    }
    let new_image = DynamicImage::ImageRgba8(new_image)
        .resize_exact(512, 512, image::imageops::FilterType::Triangle)
        .to_luma8()
        .into_flat_samples()
        .as_slice()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<_>>();
    let new_image: &GenericArray<GenericArray<f32, U512>, U512> =
        GenericArray::from_slice(new_image.as_slice()).unflatten_square_ref();

    let mut hash = GenericArray::default();
    let quality = yume_pdq::hash(
        &mut kernel,
        new_image,
        &mut hash,
        &mut buf1,
        &mut GenericArray::default(),
        &mut pdqf,
    );
    assert_eq!(quality, 1.0);
    println!("hash for rotated image: {:02x?}", hash);

    let hash_compared_to_official = hash
        .flatten()
        .iter()
        .zip(new_image_pdqhash.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum::<u32>();
    println!("hash compared to official: {:?}", hash_compared_to_official);
    // compared to the official hash before cropping and adding noise, the distance is <15 :)
    assert!(
        hash_compared_to_official < 15,
        "hash compared to official is too large"
    );
    let mut found_it = false;
    for i in 0..8 {
        let distance = hash
            .flatten()
            .iter()
            .zip(all_dihedrals[i].flatten().iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum::<u32>();
        // using official match threshold
        if distance < 31 {
            println!("found it at dihedral {:?}", i);
            found_it = true;
            break;
        }
    }
    assert!(found_it, "failed to find rotated image");

    // now hypothetically we received that rotated, cropped, and noisy image, and we want to still match it
    // from our synthetic database that contains the official hash on the original image
    // this is what is typical of what to do for a real scenario and how yume-pdq helps you achieve that
    let mut threshold = 0.0;
    let quality = yume_pdq::hash_get_threshold(
        &mut kernel,
        new_image,
        &mut threshold,
        &mut hash,
        &mut buf1,
        &mut GenericArray::default(),
        &mut pdqf,
    );
    // in reality you should discard the hash if the quality is too low not panic
    assert!(quality > 0.8, "quality is too low");
    yume_pdq::hash_get_threshold(
        &mut kernel,
        new_image,
        &mut threshold,
        &mut hash,
        &mut buf1,
        &mut GenericArray::default(),
        &mut pdqf,
    );
    all_dihedrals[0] = hash;
    let mut i = 1;
    yume_pdq::visit_dihedrals(
        &mut kernel,
        &mut pdqf,
        &mut hash,
        threshold,
        |_dihedral, _, (_, _, hash)| {
            all_dihedrals[i] = hash.clone();
            i += 1;
            Ok::<(), ()>(())
        },
    )
    .expect("failed to visit dihedrals");
    assert_eq!(i, 8, "failed to visit all dihedrals");

    // now we can use the matcher to find the hash, the official threshold is 31 but we use 15 here to be stricter
    let mut matcher =
        CpuMatcher::<15>::new_nested(GenericArray::from_slice(all_dihedrals.as_slice()));
    let mut found_at = None;
    for (chunk_index, chunk) in database.iter().enumerate() {
        matcher.find(chunk, |haystack_index, needle_index| {
            found_at = Some((haystack_index + chunk_index * 2048, needle_index));
            Some(())
        });
    }
    assert!(found_at.is_some(), "failed to find hash");
    let (found_haystack_index, found_needle_index) = found_at.unwrap();
    assert_eq!(found_haystack_index, insertion_index);
    // it should be one of the dihedrals, not the first one
    assert!(
        found_needle_index > 0 && found_needle_index < 8,
        "found needle index is out of range"
    );
}
