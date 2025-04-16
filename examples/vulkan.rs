//! this demo will show how to use the vulkan backend to match hashes in 10 million vector databases
#![forbid(unsafe_code)]
use generic_array::{
    GenericArray,
    sequence::Flatten,
    typenum::{U8, U32, U512},
};
use pdqhash::image::{DynamicImage, imageops};
use rand::{Rng, RngCore, SeedableRng, rngs::SmallRng};
use yume_pdq::{
    kernel::{Kernel, SquareGenericArrayExt},
    matching::{
        PDQMatcher,
        vulkan::{VulkanMatcher, VulkanVectorDatabase},
    },
    smart_kernel,
};
const AAA_ORIG_JPG_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test-data/aaa-orig.jpg"
));

fn main() {
    let mut rng = SmallRng::seed_from_u64(u64::from_be_bytes(*b"yumeYUME"));

    // the core setup is the similar to the end_to_end example
    let test_image =
        pdqhash::image::load_from_memory(AAA_ORIG_JPG_DATA).expect("failed to load test image");

    // hypothetically, someone put this hash in a database, using an exact match
    // and we are not sure how they did the initial 512x512 luma8, so the hash will certainly be different but comparable
    let (official_hash, official_quality) =
        pdqhash::generate_pdq(&test_image).expect("failed to generate hash");
    println!("official hash: {:02x?}", official_hash);

    assert_eq!(official_quality, 1.0);

    // now let's initialize the device and load a synthetic database which includes this hash
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();

    eprintln!("adapter: {:?}", adapter.get_info());

    let mut features = wgpu::Features::empty();
    features.insert(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);

    let mut limits = wgpu::Limits::default();
    // you may need to adjust this value based on your GPU's limits, but it should work for most GPUs you can get from cloud providers
    let chunk_size = 5_000_000u32.next_power_of_two();
    limits.max_storage_buffer_binding_size = 32 * chunk_size;

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_features: features,
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))
    .unwrap();

    let insertion_index = rng.random_range(0..10_000_000);
    let mut inserted = false;

    let databases: [VulkanVectorDatabase<U32>; 2] = core::array::from_fn(|chunk_index| {
        let mut data = vec![GenericArray::default(); chunk_size as usize];
        for (i, hash) in data
            .iter_mut()
            .enumerate()
            .map(|(i, hash)| (i + chunk_index * chunk_size as usize, hash))
        {
            if i == insertion_index {
                hash.copy_from_slice(&official_hash);
                inserted = true;
            } else {
                rng.fill_bytes(hash.as_mut_slice());
            }
        }
        VulkanVectorDatabase::<U32>::new(&device, None, data.as_slice())
    });

    assert!(inserted, "failed to insert hash");

    let mut matchers: [VulkanMatcher<U8, U32>; 2] = core::array::from_fn(|chunk_index| {
        // we will put a lower threshold here to make the match more strict
        VulkanMatcher::new(device.clone(), queue.clone(), &databases[chunk_index], 12)
    });

    let mut buf1 = Box::default();
    // now we see a weird picture
    let test_image_blur_rotated = DynamicImage::ImageRgba8(imageops::huerotate(
        &imageops::blur(
            &imageops::flip_horizontal(&imageops::rotate90(&test_image)),
            5.0,
        ),
        23,
    ));

    // pdqhash crate does not support dihedral derivation, so we have to help it out
    let test_image_blur_rotated_back = DynamicImage::ImageRgba8(imageops::rotate270(
        &imageops::flip_horizontal(&test_image_blur_rotated),
    ));

    let new_official_hash = pdqhash::generate_pdq(&test_image_blur_rotated_back)
        .unwrap()
        .0;

    let official_distance = new_official_hash
        .iter()
        .zip(official_hash.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum::<u32>();
    println!("official distance: {}", official_distance);

    // rotate the image 90 degrees and blur it by a significant amount
    let test_image_512x512 = test_image_blur_rotated
        .resize_exact(512, 512, imageops::FilterType::Triangle)
        .to_luma8()
        .into_flat_samples()
        .as_slice()
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<_>>();
    let pixels_512x512: &GenericArray<GenericArray<f32, U512>, U512> =
        GenericArray::from_slice(test_image_512x512.as_slice()).unflatten_square_ref();

    let mut kernel = smart_kernel();
    println!("kernel: {:?}", kernel.ident());

    let mut threshold = 0.0;
    let mut yume_pdq_hash = GenericArray::default();
    let mut pdqf = GenericArray::default();

    yume_pdq::hash_get_threshold(
        &mut kernel,
        pixels_512x512,
        &mut threshold,
        &mut yume_pdq_hash,
        &mut buf1,
        &mut GenericArray::default(),
        &mut pdqf,
    );

    let mut dihedrals = GenericArray::<GenericArray<u8, U32>, U8>::default();
    dihedrals[0] = yume_pdq_hash.flatten().clone();
    let mut i = 1;
    yume_pdq::visit_dihedrals(
        &mut kernel,
        &mut pdqf,
        &mut yume_pdq_hash,
        threshold,
        |_dihedral, _index, (_, _, hash)| {
            dihedrals[i] = hash.flatten().clone();
            i += 1;
            Ok::<(), ()>(())
        },
    )
    .expect("failed to visit dihedrals");
    println!("dihedrals: {:02x?}", dihedrals);

    let distance = dihedrals.iter().fold(u32::MAX, |acc, hash| {
        acc.min(
            hash.iter()
                .zip(official_hash.iter())
                .map(|(a, b)| (a ^ b).count_ones())
                .sum::<u32>(),
        )
    });
    println!("yume_pdq distance after transform: {}", distance);
    assert!(
        distance <= 10,
        "the hash was further than expected, distance is {}",
        distance
    );

    // some device can have a cold-start penalty of up to 100ms on the first invocation of the shader
    // so in production you might want to warm-up the program first by running some random queries
    // (or better yet: use it as a self-test in the mean time, like we do here)
    let mut found_at = None;
    for (chunk_index, matcher) in matchers.iter_mut().enumerate() {
        matcher.find(&dihedrals, |haystack_index, needle_index| {
            found_at = Some((
                haystack_index + chunk_index * chunk_size as usize,
                needle_index,
            ));
            Some(())
        });
    }
    assert!(found_at.is_some(), "failed to find hash");
    let (found_haystack_index, found_needle_index) = found_at.unwrap();
    assert_eq!(found_haystack_index, insertion_index);
    assert!(
        found_needle_index > 0 && found_needle_index < 8,
        "found needle index is out of range"
    );

    // let's clear the hash to see how long it takes to scan the entire database
    dihedrals.iter_mut().for_each(|hash| {
        rng.fill_bytes(hash.as_mut_slice());
    });

    let start = std::time::Instant::now();
    let mut found_at = None;
    for (chunk_index, matcher) in matchers.iter_mut().enumerate() {
        matcher.find(&dihedrals, |haystack_index, needle_index| {
            found_at = Some((
                haystack_index + chunk_index * chunk_size as usize,
                needle_index,
            ));
            Some(())
        });
    }
    println!("time taken: {:?}", start.elapsed());
    assert!(found_at.is_none(), "found hash match out of random data");
}
