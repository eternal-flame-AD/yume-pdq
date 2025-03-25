use criterion::{Criterion, black_box, criterion_group, criterion_main};
use generic_array::typenum::U16;
use rand::Rng;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use yume_pdq::{
    GenericArray,
    kernel::{
        DefaultKernel, Kernel,
        dihedral::{hash_diagflip, hash_hflip, hash_vflip},
        x86,
    },
};

fn bench_dct2d(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("dct2d");

    group.throughput(criterion::Throughput::Bytes(127 * 127));

    group.bench_function("scalar", |b| {
        let mut input: GenericArray<
            GenericArray<f32, <DefaultKernel as Kernel>::Buffer1WidthX>,
            <DefaultKernel as Kernel>::Buffer1LengthY,
        > = GenericArray::default();
        for i in 0..127 {
            for j in 0..127 {
                input[i][j] = rng.random_range(0.0..1.0);
            }
        }
        let mut kernel = DefaultKernel;
        b.iter(|| {
            let mut output = GenericArray::default();
            kernel.dct2d(&input, &mut output);
            output
        });
    });

    #[cfg(feature = "std")]
    group.bench_function("reference", |b| {
        let mut input: GenericArray<
            GenericArray<f32, <yume_pdq::kernel::ReferenceKernel as Kernel>::Buffer1WidthX>,
            <yume_pdq::kernel::ReferenceKernel as Kernel>::Buffer1LengthY,
        > = GenericArray::default();
        for i in 0..127 {
            for j in 0..127 {
                input[i][j] = rng.random_range(0.0..1.0);
            }
        }
        let mut kernel = yume_pdq::kernel::ReferenceKernel::<f32>::default();
        b.iter(|| {
            let mut output = GenericArray::default();
            kernel.dct2d(&input, &mut output);
            output
        });
    });

    group.bench_function("avx2", |b| {
        let mut input: GenericArray<
            GenericArray<f32, <x86::Avx2F32Kernel as Kernel>::Buffer1WidthX>,
            <x86::Avx2F32Kernel as Kernel>::Buffer1LengthY,
        > = GenericArray::default();
        for i in 0..127 {
            for j in 0..127 {
                input[i][j] = rng.random_range(0.0..1.0);
            }
        }
        let mut kernel = x86::Avx2F32Kernel;
        b.iter(|| {
            let mut output = GenericArray::default();
            kernel.dct2d(&input, &mut output);
            output
        });
    });

    #[cfg(feature = "avx512")]
    group.bench_function("avx512", |b| {
        let mut input: GenericArray<
            GenericArray<f32, <x86::Avx512F32Kernel as Kernel>::Buffer1WidthX>,
            <x86::Avx512F32Kernel as Kernel>::Buffer1LengthY,
        > = GenericArray::default();
        for i in 0..127 {
            for j in 0..127 {
                input[i][j] = rng.random_range(0.0..1.0);
            }
        }
        let mut kernel = x86::Avx512F32Kernel;
        b.iter(|| {
            let mut output = GenericArray::default();
            kernel.dct2d(&input, &mut output);
            output
        });
    });
}

fn bench_jarosz_compress(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("jarosz_compress");

    group.throughput(criterion::Throughput::Bytes(
        512 * 512 * std::mem::size_of::<f32>() as u64,
    ));

    group.bench_function("reference", |b| {
        let input = std::array::from_fn(|_| rng.random_range(0.0..1.0));
        let mut kernel = yume_pdq::kernel::ReferenceKernel::<f32>::default();
        b.iter(|| {
            let mut output = GenericArray::default();
            kernel.jarosz_compress(&input, &mut output);
            output
        });
    });

    group.bench_function("scalar", |b| {
        let input = std::array::from_fn(|_| rng.random_range(0.0..1.0));
        let mut kernel = DefaultKernel;
        b.iter(|| {
            let mut output = GenericArray::default();
            kernel.jarosz_compress(&input, &mut output);
            output
        });
    });

    group.bench_function("avx2", |b| {
        let input = std::array::from_fn(|_| rng.random_range(0.0..1.0));
        let mut kernel = x86::Avx2F32Kernel;
        b.iter(|| {
            let mut output = GenericArray::default();
            kernel.jarosz_compress(&input, &mut output);
            output
        });
    });
}

fn bench_quantize(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("quantize");

    group.throughput(criterion::Throughput::Bytes(
        16 * 16 * std::mem::size_of::<f32>() as u64,
    ));

    group.bench_function("scalar", |b| {
        let mut input = GenericArray::<GenericArray<f32, U16>, U16>::default();
        for i in 0..16 {
            for j in 0..16 {
                input[i][j] = rng.random_range(0.0..1.0);
            }
        }
        let mut kernel = DefaultKernel;
        b.iter(|| {
            let mut output = [0; 2 * 16];
            kernel.quantize(&input, &mut output);
            output
        });
    });

    group.bench_function("avx2", |b| {
        let mut input = GenericArray::<GenericArray<f32, U16>, U16>::default();
        for i in 0..16 {
            for j in 0..16 {
                input[i][j] = rng.random_range(0.0..1.0);
            }
        }
        let mut kernel = x86::Avx2F32Kernel;
        b.iter(|| {
            let mut output = [0; 2 * 16];
            kernel.quantize(&input, &mut output);
            output
        });
    });
}

fn bench_sum_of_gradients(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("sum_of_gradients");

    group.throughput(criterion::Throughput::Bytes(
        16 * 16 * std::mem::size_of::<f32>() as u64,
    ));

    group.bench_function("scalar", |b| {
        let mut input = GenericArray::<GenericArray<f32, U16>, U16>::default();
        for i in 0..16 {
            for j in 0..16 {
                input[i][j] = rng.random_range(0.0..1.0);
            }
        }
        let mut kernel = DefaultKernel;
        b.iter(|| kernel.sum_of_gradients(&input));
    });

    #[cfg(feature = "avx512")]
    group.bench_function("avx512", |b| {
        let mut input = GenericArray::<GenericArray<f32, U16>, U16>::default();
        for i in 0..16 {
            for j in 0..16 {
                input[i][j] = rng.random_range(0.0..1.0);
            }
        }
        let mut kernel = x86::Avx512F32Kernel;
        b.iter(|| kernel.sum_of_gradients(&input));
    });
}

fn bench_hash_diagflip(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("hash_diagflip");

    group.throughput(criterion::Throughput::Bytes(
        2 * 16 * std::mem::size_of::<u8>() as u64,
    ));

    group.bench_function("scalar", |b| {
        let mut input = [0; 2 * 16];
        rng.fill(&mut input);
        b.iter(|| {
            hash_diagflip(&mut input);
            black_box(&input);
            input
        });
    });
}

fn bench_hash_flip8(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("hash_flip8");

    group.throughput(criterion::Throughput::Bytes(
        2 * 16 * std::mem::size_of::<u8>() as u64,
    ));

    group.bench_function("scalar", |b| {
        let mut input = [0; 2 * 16];
        rng.fill(&mut input);
        b.iter(|| {
            // flip horizontally to get a horizontally flipped version
            hash_hflip(&mut input);
            black_box(&input);
            // flip vertically to get a vertically & horizontally flipped version
            hash_vflip(&mut input);
            black_box(&input);
            // flip horizontally again to get the vertically flipped version
            hash_hflip(&mut input);
            black_box(&input);
            // then flip diagonally to get a diagonally flipped version
            hash_diagflip(&mut input);
            black_box(&input);
            // repeat after flipping diagonally
            hash_hflip(&mut input);
            black_box(&input);
            hash_vflip(&mut input);
            black_box(&input);
            hash_hflip(&mut input);
            black_box(&input);

            input
        });
    });

    group.bench_function("avx2", |b| {
        let mut input = [0; 2 * 16];
        rng.fill(&mut input);
        b.iter(|| {
            hash_hflip(&mut input);
            black_box(&input);
            hash_vflip(&mut input);
            black_box(&input);
            hash_hflip(&mut input);
            black_box(&input);
            hash_diagflip(&mut input);
            black_box(&input);
            hash_hflip(&mut input);
            black_box(&input);
            hash_vflip(&mut input);
            black_box(&input);
            hash_hflip(&mut input);
            black_box(&input);
            input
        });
    });

    #[cfg(feature = "avx512")]
    group.bench_function("avx512", |b| {
        let mut input = [0; 2 * 16];
        rng.fill(&mut input);
        b.iter(|| {
            hash_hflip(&mut input);
            black_box(&input);
            hash_vflip(&mut input);
            black_box(&input);
            hash_hflip(&mut input);
            black_box(&input);
            hash_diagflip(&mut input);
            black_box(&input);
            hash_hflip(&mut input);
            black_box(&input);
            hash_vflip(&mut input);
            black_box(&input);
            hash_hflip(&mut input);
            black_box(&input);
            input
        });
    });
}

fn bench_hash(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("hash");

    group.throughput(criterion::Throughput::Bytes(
        512 * 512 * std::mem::size_of::<f32>() as u64,
    ));

    group.bench_function("reference", |b| {
        let mut input = Vec::with_capacity(512 * 512);
        for _ in 0..(512 * 512) {
            input.push(rng.random_range(0.0..1.0));
        }
        let mut kernel = yume_pdq::kernel::ReferenceKernel::<f32>::default();
        b.iter(|| {
            let mut output = [0; 2 * 16];
            let mut buf1 = GenericArray::default();
            let mut buf2 = GenericArray::default();
            yume_pdq::hash(
                &mut kernel,
                &input.as_slice().try_into().unwrap(),
                &mut output,
                &mut buf1,
                &mut buf2,
            );
            output
        });
    });

    group.bench_function("scalar", |b| {
        let mut input = Vec::with_capacity(512 * 512);
        for _ in 0..(512 * 512) {
            input.push(rng.random_range(0.0..1.0));
        }
        let mut kernel = DefaultKernel;
        b.iter(|| {
            let mut output = [0; 2 * 16];
            let mut buf1 = GenericArray::default();
            let mut buf2 = GenericArray::default();
            yume_pdq::hash(
                &mut kernel,
                &input.as_slice().try_into().unwrap(),
                &mut output,
                &mut buf1,
                &mut buf2,
            );
            output
        });
    });

    group.bench_function("avx2", |b| {
        let mut input = Vec::with_capacity(512 * 512);
        for _ in 0..(512 * 512) {
            input.push(rng.random_range(0.0..1.0));
        }
        let mut kernel = x86::Avx2F32Kernel;
        b.iter(|| {
            let mut output = [0; 2 * 16];
            let mut buf1 = GenericArray::default();
            let mut buf2 = GenericArray::default();
            yume_pdq::hash(
                &mut kernel,
                &input.as_slice().try_into().unwrap(),
                &mut output,
                &mut buf1,
                &mut buf2,
            );
            output
        });
    });

    #[cfg(feature = "avx512")]
    group.bench_function("avx512", |b| {
        let mut input = Vec::with_capacity(512 * 512);
        for _ in 0..(512 * 512) {
            input.push(rng.random_range(0.0..1.0));
        }
        let mut kernel = x86::Avx512F32Kernel;
        b.iter(|| {
            let mut output = [0; 2 * 16];
            let mut buf1 = GenericArray::default();
            let mut buf2 = GenericArray::default();
            yume_pdq::hash(
                &mut kernel,
                &input.as_slice().try_into().unwrap(),
                &mut output,
                &mut buf1,
                &mut buf2,
            );
            output
        });
    });
}

fn bench_hash_par(c: &mut Criterion) {
    let mut rng = rand::rng();

    const INPUT_PER_ROUND: usize = 48;

    for (nt, name) in [(4, "hashx4"), (8, "hashx8")] {
        let mut group = c.benchmark_group(name);

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .stack_size(8 << 20)
            .build()
            .unwrap();

        group.throughput(criterion::Throughput::Bytes(
            INPUT_PER_ROUND as u64 * 512 * 512 * std::mem::size_of::<f32>() as u64,
        ));

        group.bench_function("reference", |b| {
            let inputs: [Vec<f32>; INPUT_PER_ROUND] = std::array::from_fn(|_| {
                let mut input = Vec::with_capacity(512 * 512);
                for _ in 0..(512 * 512) {
                    input.push(rng.random_range(0.0..1.0));
                }
                input
            });

            b.iter(|| {
                let mut outputs: [Box<[u8; 2 * 16]>; INPUT_PER_ROUND] =
                    std::array::from_fn(|_| Box::new([0; 2 * 16]));
                pool.install(|| {
                    inputs
                        .par_iter()
                        .zip(outputs.par_iter_mut())
                        .for_each(|(input, output)| {
                            let mut kernel = yume_pdq::kernel::ReferenceKernel::<f32>::default();
                            let mut buf1 = GenericArray::default();
                            let mut buf2 = GenericArray::default();
                            yume_pdq::hash(
                                &mut kernel,
                                &input.as_slice().try_into().unwrap(),
                                output,
                                &mut buf1,
                                &mut buf2,
                            );
                        });
                });
                outputs
            });
        });

        group.bench_function("scalar", |b| {
            let inputs: [Vec<f32>; INPUT_PER_ROUND] = std::array::from_fn(|_| {
                let mut input = Vec::with_capacity(512 * 512);
                for _ in 0..(512 * 512) {
                    input.push(rng.random_range(0.0..1.0));
                }
                input
            });

            b.iter(|| {
                let mut outputs: [Box<[u8; 2 * 16]>; INPUT_PER_ROUND] =
                    std::array::from_fn(|_| Box::new([0; 2 * 16]));
                pool.install(|| {
                    inputs
                        .par_iter()
                        .zip(outputs.par_iter_mut())
                        .for_each(|(input, output)| {
                            let mut kernel = DefaultKernel;
                            let mut buf1 = GenericArray::default();
                            let mut buf2 = GenericArray::default();
                            yume_pdq::hash(
                                &mut kernel,
                                &input.as_slice().try_into().unwrap(),
                                output,
                                &mut buf1,
                                &mut buf2,
                            );
                        });
                });
                outputs
            });
        });

        group.bench_function("avx2", |b| {
            let inputs: [Vec<f32>; INPUT_PER_ROUND] = std::array::from_fn(|_| {
                let mut input = Vec::with_capacity(512 * 512);
                for _ in 0..(512 * 512) {
                    input.push(rng.random_range(0.0..1.0));
                }
                input
            });

            b.iter(|| {
                let mut outputs: [Box<[u8; 2 * 16]>; INPUT_PER_ROUND] =
                    std::array::from_fn(|_| Box::new([0; 2 * 16]));
                pool.install(|| {
                    inputs
                        .par_iter()
                        .zip(outputs.par_iter_mut())
                        .for_each(|(input, output)| {
                            let mut kernel = x86::Avx2F32Kernel;
                            let mut buf1 = GenericArray::default();
                            let mut buf2 = GenericArray::default();
                            yume_pdq::hash(
                                &mut kernel,
                                &input.as_slice().try_into().unwrap(),
                                output,
                                &mut buf1,
                                &mut buf2,
                            );
                        });
                });
                outputs
            });
        });

        #[cfg(feature = "avx512")]
        group.bench_function("avx512", |b| {
            let inputs: [Vec<f32>; INPUT_PER_ROUND] = std::array::from_fn(|_| {
                let mut input = Vec::with_capacity(512 * 512);
                for _ in 0..(512 * 512) {
                    input.push(rng.random_range(0.0..1.0));
                }
                input
            });

            b.iter(|| {
                let mut outputs: [Box<[u8; 2 * 16]>; INPUT_PER_ROUND] =
                    std::array::from_fn(|_| Box::new([0; 2 * 16]));
                pool.install(|| {
                    inputs
                        .par_iter()
                        .zip(outputs.par_iter_mut())
                        .for_each(|(input, output)| {
                            let mut kernel = x86::Avx512F32Kernel;
                            let mut buf1 = GenericArray::default();
                            let mut buf2 = GenericArray::default();
                            yume_pdq::hash(
                                &mut kernel,
                                &input.as_slice().try_into().unwrap(),
                                output,
                                &mut buf1,
                                &mut buf2,
                            );
                        });
                });
                outputs
            });
        });
    }
}

criterion_group!(
    benches,
    bench_dct2d,
    bench_jarosz_compress,
    bench_quantize,
    bench_sum_of_gradients,
    bench_hash,
    bench_hash_par,
    bench_hash_flip8,
    bench_hash_diagflip
);
criterion_main!(benches);
