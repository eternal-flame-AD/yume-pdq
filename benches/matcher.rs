use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use generic_array::{
    GenericArray,
    typenum::{U1, U2, U3, U4, U8, U16, U24, U32, U2048},
};
use rand::{RngCore, SeedableRng, rngs::SmallRng};

use yume_pdq::{
    alignment::Align8,
    matching::{CpuMatcher, PDQMatcher},
};

#[cfg(feature = "vulkan")]
use yume_pdq::matching::vulkan::{VulkanMatcher, VulkanVectorDatabase};

fn bench_pdq_popcnt(c: &mut Criterion) {
    // about 10m vectors
    const NUM_BATCHES: usize = 10_000_000 / 2048 + 1;
    println!(
        "NUM_BATCHES: {} ({} vectors, {} MiB)",
        NUM_BATCHES,
        NUM_BATCHES * 2048,
        NUM_BATCHES * 2048 * (256 / 8) / (1024 * 1024)
    );

    let mut rng = SmallRng::from_os_rng();

    // Generate random batch data
    let batches = (0..NUM_BATCHES)
        .map(|_| {
            let mut batch_data = GenericArray::<GenericArray<u8, U32>, U2048>::default();
            for vector in batch_data.iter_mut() {
                rng.fill_bytes(vector);
            }
            Align8(batch_data)
        })
        .collect::<Vec<_>>();

    // Generate random needles
    let mut needles_data = Align8(GenericArray::<GenericArray<u8, U32>, U8>::default());
    for needle in needles_data.iter_mut() {
        rng.fill_bytes(needle);
    }

    let mut needles16_data = Align8(GenericArray::<GenericArray<u8, U32>, U16>::default());
    for needle in needles16_data.iter_mut() {
        rng.fill_bytes(needle);
    }

    let mut needles24_data = Align8(GenericArray::<GenericArray<u8, U32>, U24>::default());
    for needle in needles24_data.iter_mut() {
        rng.fill_bytes(needle);
    }

    let mut needles32_data = Align8(GenericArray::<GenericArray<u8, U32>, U32>::default());
    for needle in needles32_data.iter_mut() {
        rng.fill_bytes(needle);
    }

    {
        let mut group = c.benchmark_group("8needles/scan10mil");

        // the kernel matches 1024 vectors at a time, we will only benchmark the full batch case not the remainder case
        group.throughput(Throughput::Elements(NUM_BATCHES as u64 * 2048 as u64));

        group.bench_function(
            if cfg!(all(feature = "avx512", target_feature = "avx512vpopcntdq")) {
                "avx512 popcnt"
            } else {
                "cpu popcnt"
            },
            |b| {
                b.iter(|| {
                    let mut kernel = CpuMatcher::<U2048, U1>::new(&needles_data, 31);
                    let mut matched = false;
                    for batch in batches.iter() {
                        matched |= kernel.scan(batch);
                    }
                    assert!(!matched);
                    matched
                });
            },
        );

        #[cfg(feature = "vulkan")]
        {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN,
                ..Default::default()
            });
            let adapter = pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
            )
            .unwrap();

            eprintln!("adapter: {:?}", adapter.get_info());

            let mut features = wgpu::Features::empty();
            features.insert(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);

            let mut limits = wgpu::Limits::default();
            limits.max_storage_buffer_binding_size = 160006144;

            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                }))
                .unwrap();

            let haystack0 = VulkanVectorDatabase::<U32>::new(&device, None, unsafe {
                core::slice::from_raw_parts::<GenericArray<u8, U32>>(
                    batches[0].as_ptr().cast(),
                    NUM_BATCHES * 1024,
                )
            });
            let haystack1 = VulkanVectorDatabase::<U32>::new(&device, None, unsafe {
                core::slice::from_raw_parts::<GenericArray<u8, U32>>(
                    batches[NUM_BATCHES / 2].as_ptr().cast(),
                    NUM_BATCHES * 1024,
                )
            });

            let mut gpu_kernel0 =
                VulkanMatcher::<U8, U32>::new(device.clone(), queue.clone(), &haystack0, 31);
            let mut gpu_kernel1 =
                VulkanMatcher::<U8, U32>::new(device.clone(), queue.clone(), &haystack1, 31);

            group.bench_function("vulkan", |b| {
                b.iter(|| {
                    let mut matched = false;
                    matched |= gpu_kernel0.scan(&needles_data);
                    matched |= gpu_kernel1.scan(&needles_data);
                    assert!(!matched);
                });
            });
        }
    }

    {
        let mut group = c.benchmark_group("16needles/scan10mil");

        group.throughput(Throughput::Elements(NUM_BATCHES as u64 * 2048 * 2));

        group.bench_function(
            if cfg!(all(feature = "avx512", target_feature = "avx512vpopcntdq")) {
                "avx512 popcnt"
            } else {
                "scalar popcnt"
            },
            |b| {
                b.iter(|| {
                    let mut kernel = CpuMatcher::<U2048, U2>::new(&needles16_data, 31);
                    let mut matched = false;
                    for batch in batches.iter() {
                        matched |= kernel.scan(batch);
                    }
                    assert!(!matched);
                    matched
                });
            },
        );
    }

    {
        let mut group = c.benchmark_group("24needles/scan10mil");

        group.throughput(Throughput::Elements(NUM_BATCHES as u64 * 2048 * 3));

        group.bench_function(
            if cfg!(all(feature = "avx512", target_feature = "avx512vpopcntdq")) {
                "avx512 popcnt"
            } else {
                "scalar popcnt"
            },
            |b| {
                b.iter(|| {
                    let mut kernel = CpuMatcher::<U2048, U3>::new(&needles24_data, 31);
                    let mut matched = false;
                    for batch in batches.iter() {
                        matched |= kernel.scan(batch);
                    }
                    assert!(!matched);
                    matched
                });
            },
        );
    }

    {
        let mut group = c.benchmark_group("32needles/scan10mil");

        group.throughput(Throughput::Elements(NUM_BATCHES as u64 * 2048 * 4));

        group.bench_function(
            if cfg!(all(feature = "avx512", target_feature = "avx512vpopcntdq")) {
                "avx512 popcnt"
            } else {
                "scalar popcnt"
            },
            |b| {
                b.iter(|| {
                    let mut kernel = CpuMatcher::<U2048, U4>::new(&needles32_data, 31);
                    let mut matched = false;
                    for batch in batches.iter() {
                        matched |= kernel.scan(batch);
                    }
                    assert!(!matched);
                    matched
                });
            },
        );
    }

    {
        let mut group = c.benchmark_group("8needles/find10mil");

        group.throughput(Throughput::Elements(NUM_BATCHES as u64 * 2048));

        group.bench_function(
            if cfg!(all(feature = "avx512", target_feature = "avx512vpopcntdq")) {
                "avx512 popcnt"
            } else {
                "scalar popcnt"
            },
            |b| {
                b.iter(|| {
                    let mut kernel = CpuMatcher::<U2048, U1>::new(&needles_data, 31);
                    let mut sum = 0;
                    for batch in batches.iter() {
                        kernel.find(batch, |i, j| {
                            sum += i + j;
                            // it is very important to return Some(()) here,
                            // to make sure the compiler don't collapse the branch in the hot loop
                            Some(())
                        });
                    }
                    sum
                });
            },
        );

        #[cfg(feature = "vulkan")]
        {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: wgpu::Backends::VULKAN,
                ..Default::default()
            });
            let adapter = pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
            )
            .unwrap();

            eprintln!("adapter: {:?}", adapter.get_info());

            let mut features = wgpu::Features::empty();
            features.insert(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);

            let mut limits = wgpu::Limits::default();
            limits.max_storage_buffer_binding_size = 160006144;

            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: features,
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                }))
                .unwrap();

            let haystack0 = VulkanVectorDatabase::<U32>::new(&device, None, unsafe {
                core::slice::from_raw_parts::<GenericArray<u8, U32>>(
                    batches[0].as_ptr().cast(),
                    NUM_BATCHES * 1024,
                )
            });
            let haystack1 = VulkanVectorDatabase::<U32>::new(&device, None, unsafe {
                core::slice::from_raw_parts::<GenericArray<u8, U32>>(
                    batches[NUM_BATCHES / 2].as_ptr().cast(),
                    NUM_BATCHES * 1024,
                )
            });

            let mut gpu_kernel0 =
                VulkanMatcher::<U8, U32>::new(device.clone(), queue.clone(), &haystack0, 31);
            let mut gpu_kernel1 =
                VulkanMatcher::<U8, U32>::new(device.clone(), queue.clone(), &haystack1, 31);

            group.bench_function("vulkan", |b| {
                b.iter(|| {
                    let mut sum = 0;
                    gpu_kernel0.find(&needles_data, |i, j| {
                        sum += i + j;
                        // it is very important to return Some(()) here,
                        // to make sure the compiler don't collapse the branch in the hot loop
                        Some(())
                    });
                    gpu_kernel1.find(&needles_data, |i, j| {
                        sum += i + j;
                        // it is very important to return Some(()) here,
                        // to make sure the compiler don't collapse the branch in the hot loop
                        Some(())
                    });
                    assert_eq!(sum, 0);
                    sum
                });
            });
        }
    }

    {
        let mut group = c.benchmark_group("16needles/find10mil");

        group.throughput(Throughput::Elements(NUM_BATCHES as u64 * 2048 * 2));

        group.bench_function(
            if cfg!(all(feature = "avx512", target_feature = "avx512vpopcntdq")) {
                "avx512 popcnt"
            } else {
                "scalar popcnt"
            },
            |b| {
                b.iter(|| {
                    let mut kernel = CpuMatcher::<U2048, U2>::new(&needles16_data, 31);
                    let mut sum = 0;
                    for batch in batches.iter() {
                        kernel.find(batch, |i, j| {
                            sum += i + j;
                            // it is very important to return Some(()) here,
                            // to make sure the compiler don't collapse the branch in the hot loop
                            Some(())
                        });
                    }
                    sum
                });
            },
        );
    }

    {
        let mut group = c.benchmark_group("24needles/find10mil");

        group.throughput(Throughput::Elements(NUM_BATCHES as u64 * 2048 * 3));

        group.bench_function(
            if cfg!(all(feature = "avx512", target_feature = "avx512vpopcntdq")) {
                "avx512 popcnt"
            } else {
                "scalar popcnt"
            },
            |b| {
                b.iter(|| {
                    let mut kernel = CpuMatcher::<U2048, U3>::new(&needles24_data, 31);
                    let mut sum = 0;
                    for batch in batches.iter() {
                        kernel.find(batch, |i, j| {
                            sum += i + j;
                            // it is very important to return Some(()) here,
                            // to make sure the compiler don't collapse the branch in the hot loop
                            Some(())
                        });
                    }
                    sum
                });
            },
        );
    }

    {
        let mut group = c.benchmark_group("32needles/find10mil");

        group.throughput(Throughput::Elements(NUM_BATCHES as u64 * 2048 * 4));

        group.bench_function(
            if cfg!(all(feature = "avx512", target_feature = "avx512vpopcntdq")) {
                "avx512 popcnt"
            } else {
                "scalar popcnt"
            },
            |b| {
                b.iter(|| {
                    let mut kernel = CpuMatcher::<U2048, U4>::new(&needles32_data, 31);
                    let mut sum = 0;
                    for batch in batches.iter() {
                        kernel.find(batch, |i, j| {
                            sum += i + j;
                            // it is very important to return Some(()) here,
                            // to make sure the compiler don't collapse the branch in the hot loop
                            Some(())
                        });
                    }
                    sum
                });
            },
        );
    }
}

criterion_group!(benches, bench_pdq_popcnt);
criterion_main!(benches);
