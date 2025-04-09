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

use clap::{Arg, ArgAction, Command, value_parser};
use rand::SeedableRng;
use std::{
    hash::RandomState,
    io::{Read, Write},
    ops::Div,
    ptr::addr_of,
    sync::atomic::AtomicU64,
    thread::{self},
};

use generic_array::{
    ArrayLength,
    sequence::{Flatten, GenericSequence},
    typenum::{B1, U4, Unsigned},
};
#[allow(unused_imports)]
use yume_pdq::{
    GenericArray, PDQHash, PDQHashF,
    alignment::Align32,
    kernel::{
        FallbackKernel, Kernel,
        router::KernelRouter,
        type_traits::{DivisibleBy8, EvaluateHardwareFeature, SquareOf},
    },
    lut_utils,
};

fn build_cli() -> Command {
    Command::new("yume-pdq")
        .about("Fast PDQ perceptual image hashing implementation")
        .long_about(concat!(
r#"
A high-performance implementation of the PDQ perceptual image hashing algorithm. Supports various input/output formats and hardware acceleration.

"#, env!("TARGET_SPECIFIC_CLI_MESSAGE"), r#"

Build Facts:
  Version: "#,env!("CARGO_PKG_VERSION"),r#"
  Optimization: -O "#, env!("BUILD_OPT_LEVEL"),r#"
  Build time CPU flag support: "#, env!("BUILD_CFG_TARGET_FEATURES")
        ))
        .version(env!("CARGO_PKG_VERSION"))
        .flatten_help(true)
        .subcommand(
            Command::new("pipe")
                .about("Process image stream and output hashes, see 'pipe --help' usage examples")
                .long_about(
r#"
Reads a stream of 512x512 grayscale images and outputs their PDQ hashes.

Usage examples with common tools:

 * Emit a single image, return the hash in ASCII hex format, prefix by the quality score, pad by a line feed:

    >ffmpeg -loglevel error -hide_banner -i test-data/aaa-orig.jpg \
      -vf "scale=512:512:force_original_aspect_ratio=disable" \
      -frames:v 1 -pix_fmt gray8  -f rawvideo - | yume-pdq pipe -f q+hex+lf

       Output: 100.000:58f8f0cee0f4a84f06370a32038f67f0b36e2ed596623e1d33e6b39c4e9c9b22


 * Process every frame of a video stream, return the hash in ASCII binary format with no padding in between frames: 

   >ffmpeg -f lavfi -i testsrc=size=512x512:rate=1  -pix_fmt gray  -f rawvideo - | yume-pdq pipe -f bin

       Output: <BINARY_HASH>, expect to see thousands of FPS reported by ffmpeg!

 * Process an arbitrary list of images, return the hash in ASCII hex format, pad by a line feed:

   > for i in (seq 1 1000); ln -s (realpath test-data/aaa-orig.jpg) /tmp/test/$i.jpg; end
   >  time convert 'test-data/*' -resize 512x512! -colorspace gray -depth 8 gray:- \
   >    | yume-pdq pipe -f 'hex+lf'

       Output: d8f8f0cee0f4a84f06370a32038f67f0b36e2ed596621e1d33e6b39c4e9c9b22 (*1000 lines)
       Executed in   26.01 secs    fish           external
       usr time   47.76 secs    0.00 millis   47.76 secs
       sys time    9.43 secs    2.77 millis    9.43 secs
"#)
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .help("Input source (- for stdin)")
                        .long_help(
                            "Source of input images. Use '-' for stdin or provide a file path. \
                             Expects 512x512 grayscale images in raw format.",
                        )
                        .default_value("-"),
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .help("Output destination (- for stdout)")
                        .long_help("Destination for hash output. Use '-' for stdout or provide a file path.")
                        .default_value("-"),
                )
                .arg(
                    Arg::new("output_buffer")
                        .long("output-buffer")
                        .long_help(
                            "This flag is no longer used and have no effect.",
                        )
                        .value_parser(value_parser!(u32))
                        .hide(true),
                )
                .arg(
                    Arg::new("stats")
                        .long("stats")
                        .help("Show processing statistics")
                        .long_help("Display periodic statistics about processing speed and throughput to stderr.")
                        .action(ArgAction::SetTrue),
                )
                .arg(
                    Arg::new("core0")
                        .long("core0")
                        .help("Pin processing to core 0")
                        .value_parser(value_parser!(usize))
                        .hide(!cfg!(feature = "hpc")),
                )
                .arg(
                    Arg::new("core1")
                        .long("core1")
                        .help("Pin processing to core 1")
                        .value_parser(value_parser!(usize))
                        .hide(!cfg!(feature = "hpc")),
                )
                .arg(
                    Arg::new("format")
                        .short('f')
                        .long("format")
                        .help("Output format specification")
                        .long_help(
                            "Specify the output format for hashes. Available formats:\n\
                             - raw/RAW: Binary output\n\
                             - hex/HEX: Hexadecimal output (lowercase/uppercase)\n\
                             - bin/BIN: Binary string output\n\
                             Modifiers:\n\
                             - q+: Prefix with quality score\n\
                             - +lf/+cr/+crlf: Add line ending\n\
                             Examples: q+hex+lf, HEX+crlf, q+bin",
                        )
                        .default_value("bin"),
                ),
        )
        .subcommand(
            Command::new("busyloop")
                .about("Catch in an infinite loop by feeding random data, for profiling usage")
                .long_about(
                    "Catch in an infinite loop by feeding random data, for profiling usage",
                )
                .arg(
                    Arg::new("core0")
                        .long("core0")
                        .help("Pin processing to core 0")
                        .value_parser(value_parser!(usize))
                        .hide(!cfg!(feature = "hpc")),
                )
                .arg(
                    Arg::new("core1")
                        .long("core1")
                        .help("Pin processing to core 1")
                        .value_parser(value_parser!(usize))
                        .hide(!cfg!(feature = "hpc")),
                )
        )
        .subcommand(
            Command::new("random-stream")
                .about("Generate random byte stream")
                .long_about(
                    "Generates a continuous stream of random bytes (0-255) to stdout. \
                     Useful for testing and benchmarking.",
                ),
        )
        .subcommand(
            Command::new("bench")
                .hide(!cfg!(feature = "cli-bench"))
                .about("Run a formal benchmark")
                .long_about(
                    "Run a formal benchmark using Criterion.rs using an internal synthetic image source.",
                )
                .arg(
                    Arg::new("core0")
                        .long("core0")
                        .help("Core ID to run the benchmark on")
                        .required(false)
                        .hide(!cfg!(feature = "hpc")),
                )
                .arg(
                    Arg::new("core1")
                        .long("core1")
                        .help("Core ID to run the benchmark on")
                        .required(false)
                        .hide(!cfg!(feature = "hpc")),
                ),
        )
        .subcommand(
            Command::new("vectorization-info")
                .about("Display vectorization information")
                .long_about(
                    "Displays diagnostic information about the vectorization capabilities of the current CPU. \
                     HIGHLY RECOMMENDED to run this command before deploying on a new micro-architecture.",
                ),
        )
        .subcommand(
            Command::new("list-cores")
                .about("List cores IDs")
                .long_about("Lists the core IDs of the current CPU.")
                .hide(!cfg!(feature = "hpc")),
        )
}

#[repr(C)]
struct BufferPad {
    #[cfg(feature = "avx512")]
    _pad: [f32; 32],
    #[cfg(not(feature = "avx512"))]
    _pad: [f32; 16],
}

const OUTPUT_FLAG_UPPER: u8 = 128;
const OUTPUT_TYPE_RAW: u8 = 0;
const OUTPUT_TYPE_RAW_PREFIX_QUALITY: u8 = 1;
const OUTPUT_TYPE_ASCII_HEX: u8 = 2;
const OUTPUT_TYPE_ASCII_HEX_UPPER: u8 = OUTPUT_FLAG_UPPER | OUTPUT_TYPE_ASCII_HEX;
const OUTPUT_TYPE_ASCII_BINARY: u8 = 3;
const OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY: u8 = 4;
const OUTPUT_TYPE_ASCII_HEX_UPPER_PREFIX_QUALITY: u8 =
    OUTPUT_FLAG_UPPER | OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY;
const OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY: u8 = 5;

#[repr(C)]
struct PairBuffer<K: Kernel>
where
    K::OutputDimension: DivisibleBy8,
{
    // defensive padding at least 2 times register width to:
    //  - reduce false sharing
    //  - ensure if there was a buffer overrun, it's not catastrophic
    half_frames_processed: AtomicU64,
    buf1_input: Align32<GenericArray<GenericArray<f32, K::InputDimension>, K::InputDimension>>,
    _pad1: BufferPad,
    buf1_intermediate:
        Align32<GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>>,
    _pad2: BufferPad,
    buf1_pdqf: Align32<PDQHashF<K::InternalFloat, K::OutputDimension>>,
    buf1_tmp: Align32<GenericArray<K::InternalFloat, K::Buffer1WidthX>>,
    _pad3: BufferPad,
    buf1_output: PDQHash<K::OutputDimension>,
    _pad4: BufferPad,
    buf2_input: Align32<GenericArray<GenericArray<f32, K::InputDimension>, K::InputDimension>>,
    _pad5: BufferPad,
    buf2_intermediate:
        Align32<GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>>,
    _pad6: BufferPad,
    buf2_pdqf: Align32<PDQHashF<K::InternalFloat, K::OutputDimension>>,
    buf2_tmp: Align32<GenericArray<K::InternalFloat, K::Buffer1WidthX>>,
    _pad7: BufferPad,
    buf2_output: PDQHash<K::OutputDimension>,
}

struct PairProcessor<
    K: Kernel + Send + Sync,
    R: Read + Send + Sync,
    W: Write + Send + Sync,
    const OUTPUT_TYPE: u8,
> where
    K::OutputDimension: DivisibleBy8,
{
    barrier: std::sync::Barrier,
    buffers: Box<PairBuffer<K>>,
    reader: R,
    writer: W,
}

impl<K: Kernel + Send + Sync, R: Read + Send + Sync, W: Write + Send + Sync, const OUTPUT_TYPE: u8>
    PairProcessor<K, R, W, OUTPUT_TYPE>
where
    K::OutputDimension: DivisibleBy8,
    <K as Kernel>::RequiredHardwareFeature: EvaluateHardwareFeature<EnabledStatic = B1>,
    <K::OutputDimension as SquareOf>::Output: ArrayLength + Div<U4>,
    <<K::OutputDimension as SquareOf>::Output as Div<U4>>::Output: ArrayLength,
{
    /// Initialize fast by filling with default values.
    pub fn new_fast(reader: R, writer: W) -> Self
    where
        <K as Kernel>::InternalFloat: Default + Copy,
    {
        let mut buffers = Box::new_uninit();
        unsafe {
            let buffers: &mut PairBuffer<K> = buffers.assume_init_mut();
            buffers.buf1_input.fill_with(Default::default);
            buffers.buf1_intermediate.fill_with(Default::default);
            buffers.buf1_pdqf.fill_with(Default::default);
            buffers.buf1_output.fill_with(Default::default);
            buffers.buf2_input.fill_with(Default::default);
            buffers.buf2_intermediate.fill_with(Default::default);
            buffers.buf2_pdqf.fill_with(Default::default);
            buffers.buf2_output.fill_with(Default::default);
            buffers
                .half_frames_processed
                .store(0, std::sync::atomic::Ordering::SeqCst);
        };
        Self {
            barrier: std::sync::Barrier::new(2),
            buffers: unsafe { buffers.assume_init() },
            reader,
            writer,
        }
    }

    /// Loop for one of the threads.
    ///
    /// One thread must start with i_am_reading = true, and the other with i_am_reading = false.
    pub unsafe fn loop_thread<const I_AM_READING_INITIALLY: bool, const STATS: bool>(
        &self,
        mut kernel: K,
        separator: &'static [u8],
        #[cfg(feature = "hpc")] pin_core: Option<usize>,
    ) -> Result<(), std::io::Error> {
        #[cfg(feature = "hpc")]
        if let Some(core_id) = pin_core {
            if !core_affinity::set_for_current(core_affinity::CoreId { id: core_id }) {
                eprintln!(
                    "Failed to pin processing to core {}, continuing without pinning",
                    core_id
                );
            }
        }

        let mut have_data = false;
        let mut i_am_reading = I_AM_READING_INITIALLY;
        loop {
            if i_am_reading {
                // since the other thread start to the last branch here (no data), we do the same here, and synchronize right before side-effects
                self.barrier.wait();
                unsafe {
                    let reader_mut = addr_of!(self.reader).cast_mut().as_mut().unwrap();
                    let input_buf_mut = if I_AM_READING_INITIALLY {
                        addr_of!(self.buffers.buf1_input.0)
                    } else {
                        addr_of!(self.buffers.buf2_input.0)
                    }
                    .cast_mut()
                    .as_mut()
                    .unwrap();
                    for i in 0..K::InputDimension::USIZE {
                        let mut row_buf = Align32::<GenericArray<u8, K::InputDimension>>::default();
                        let mut ptr = 0;
                        while ptr < K::InputDimension::USIZE {
                            match reader_mut.read(&mut row_buf[ptr..]) {
                                Ok(0) => {
                                    if ptr > 0 {
                                        return Err(std::io::Error::new(
                                            std::io::ErrorKind::UnexpectedEof,
                                            "Unexpected EOF while reading the middle of a frame",
                                        ));
                                    } else {
                                        return Ok(());
                                    }
                                }
                                Ok(s) => {
                                    ptr += s;
                                }
                                Err(e) => {
                                    return Err(e);
                                }
                            }
                        }

                        for j in 0..K::InputDimension::USIZE {
                            input_buf_mut[i][j] = row_buf[j] as f32;
                        }
                    }
                }

                have_data = true;
            } else if have_data {
                unsafe {
                    let mut threshold = Default::default();

                    let quality = if I_AM_READING_INITIALLY {
                        yume_pdq::hash_get_threshold(
                            &mut kernel,
                            &self.buffers.buf1_input.0,
                            &mut threshold,
                            addr_of!(self.buffers.buf1_output)
                                .cast_mut()
                                .as_mut()
                                .unwrap(),
                            addr_of!(self.buffers.buf1_intermediate.0)
                                .cast_mut()
                                .as_mut()
                                .unwrap(),
                            addr_of!(self.buffers.buf1_tmp.0)
                                .cast_mut()
                                .as_mut()
                                .unwrap(),
                            addr_of!(self.buffers.buf1_pdqf.0)
                                .cast_mut()
                                .as_mut()
                                .unwrap(),
                        )
                    } else {
                        yume_pdq::hash_get_threshold(
                            &mut kernel,
                            &self.buffers.buf2_input.0,
                            &mut threshold,
                            addr_of!(self.buffers.buf2_output)
                                .cast_mut()
                                .as_mut()
                                .unwrap(),
                            addr_of!(self.buffers.buf2_intermediate.0)
                                .cast_mut()
                                .as_mut()
                                .unwrap(),
                            addr_of!(self.buffers.buf2_tmp.0)
                                .cast_mut()
                                .as_mut()
                                .unwrap(),
                            addr_of!(self.buffers.buf2_pdqf.0)
                                .cast_mut()
                                .as_mut()
                                .unwrap(),
                        )
                    };

                    let writer_mut = addr_of!(self.writer).cast_mut().as_mut().unwrap();
                    let output_ref: &GenericArray<GenericArray<u8, _>, K::OutputDimension> =
                        if I_AM_READING_INITIALLY {
                            &self.buffers.buf1_output
                        } else {
                            &self.buffers.buf2_output
                        };
                    let output_flattened: &GenericArray<u8, _> = Flatten::flatten(output_ref);

                    // we are about to cause side effects, so we have to synchronize (do this as late as possible)
                    self.barrier.wait();

                    match OUTPUT_TYPE {
                        OUTPUT_TYPE_RAW => {
                            writer_mut.write_all(output_flattened.as_slice())?;
                        }
                        OUTPUT_TYPE_RAW_PREFIX_QUALITY => {
                            writer_mut.write_all(&quality.to_le_bytes())?;
                            writer_mut.write_all(output_flattened.as_slice())?;
                        }
                        OUTPUT_TYPE_ASCII_HEX | OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY => {
                            if OUTPUT_TYPE == OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY {
                                write!(writer_mut, "{0:02.3}:", quality * 100.0)?;
                            }

                            let mut buf: GenericArray<
                                u8,
                                <<K::OutputDimension as SquareOf>::Output as Div<U4>>::Output,
                            > = GenericArray::generate(|_| b'0');
                            buf.iter_mut()
                                .zip(
                                    output_flattened
                                        .iter()
                                        .flat_map(|x| [(x >> 4) as u8, x & 0x0f as u8]),
                                )
                                .for_each(|(a, b): (&mut u8, u8)| {
                                    *a += b;
                                    if b >= 10 {
                                        const OFFSET_UPPER: u8 = b'A' - (b'9' + 1);
                                        const OFFSET_LOWER: u8 = b'a' - (b'9' + 1);
                                        *a += if OUTPUT_TYPE & OUTPUT_FLAG_UPPER != 0 {
                                            OFFSET_UPPER
                                        } else {
                                            OFFSET_LOWER
                                        };
                                    }
                                });

                            writer_mut.write_all(buf.as_slice())?;
                        }
                        OUTPUT_TYPE_ASCII_BINARY | OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY => {
                            if OUTPUT_TYPE == OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY {
                                write!(writer_mut, "{0:02.3}:", quality * 100.0)?;
                            }

                            for i in 0..K::OutputDimension::USIZE {
                                let data_iter =
                                    (0..(K::OutputDimension::USIZE / 8)).flat_map(|j| {
                                        lut_utils::BINARY_PRINTING[output_ref[i][j] as usize]
                                    });

                                let row_buf: GenericArray<u8, K::OutputDimension> =
                                    GenericArray::from_iter(data_iter);

                                writer_mut.write_all(row_buf.as_slice())?;
                            }
                        }
                        _ => unreachable!(),
                    }

                    if !separator.is_empty() {
                        writer_mut.write_all(separator)?;
                    }

                    writer_mut.flush()?;

                    if I_AM_READING_INITIALLY && STATS {
                        self.buffers
                            .half_frames_processed
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            } else {
                self.barrier.wait();
            }
            i_am_reading = !i_am_reading;
        }
    }
}

fn open_reader(spec: &str) -> Result<std::io::BufReader<std::fs::File>, std::io::Error> {
    use std::os::fd::{AsFd, FromRawFd, IntoRawFd};

    let reader = if spec == "-" {
        let stdin = std::io::stdin();
        let fd = stdin.as_fd().try_clone_to_owned()?;
        unsafe { std::fs::File::from_raw_fd(fd.into_raw_fd()) }
    } else {
        std::fs::File::open(spec)?
    };

    Ok(std::io::BufReader::new(reader))
}

fn open_writer(spec: &str) -> Result<std::io::BufWriter<std::fs::File>, std::io::Error> {
    use std::os::fd::{AsFd, FromRawFd, IntoRawFd};
    const BUFFER_SIZE: usize = 512; // big enough for all outputs, we are flushing anyways
    let writer = if spec == "-" {
        let stdout = std::io::stdout();
        let fd = stdout.as_fd().try_clone_to_owned()?;
        unsafe { std::fs::File::from_raw_fd(fd.into_raw_fd()) }
    } else {
        std::fs::File::create(spec)?
    };

    Ok(std::io::BufWriter::with_capacity(BUFFER_SIZE, writer))
}

fn type_name_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}

struct RandReadAdaptor<R: rand::RngCore + Send + Sync> {
    rng: R,
}

impl<R: rand::RngCore + Send + Sync> RandReadAdaptor<R> {
    fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: rand::RngCore + Send + Sync> std::io::Read for RandReadAdaptor<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.rng.fill_bytes(buf);
        Ok(buf.len())
    }
}

fn main() {
    let matches = build_cli().get_matches();

    match matches.subcommand() {
        #[cfg(feature = "hpc")]
        Some(("list-cores", _)) => {
            if let Some(core_ids) = core_affinity::get_core_ids() {
                for core_id in core_ids {
                    println!("{}", core_id.id);
                }
            } else {
                eprintln!("Failed to get core IDs");
            }
        }
        Some(("vectorization-info", _)) => {
            println!("=== Feature flag information ===\n");
            println!(
                "  Capability of this binary: {}",
                env!("TARGET_SPECIFIC_CLI_MESSAGE")
            );

            println!(
                "  Supported CPU features: {}",
                env!("BUILD_CFG_TARGET_FEATURES")
            );

            println!("\n=== Runtime Routing Information ===\n");

            let kernel = yume_pdq::kernel::smart_kernel();

            let ident = kernel.ident();

            println!("  Runtime decision: {}", ident);
            println!();
            println!("  Runtime decision details: {:?}", ident);
            println!();
            println!("  Router type: {}", type_name_of(&kernel));
        }
        Some(("busyloop", sub_matches)) => {
            let _ = sub_matches;
            use std::hash::BuildHasher;
            let key = RandomState::new().hash_one(0);
            let seeded = rand::rngs::SmallRng::seed_from_u64(key);
            let rng = seeded;

            let (kernel0, kernel1) = {
                (
                    yume_pdq::kernel::smart_kernel(),
                    yume_pdq::kernel::smart_kernel(),
                )
            };

            #[cfg(feature = "hpc")]
            let arg_core0 = sub_matches.get_one::<usize>("core0").cloned();
            #[cfg(feature = "hpc")]
            let arg_core1 = sub_matches.get_one::<usize>("core1").cloned();

            let processor = PairProcessor::<_, _, _, OUTPUT_TYPE_RAW>::new_fast(
                std::io::BufReader::new(RandReadAdaptor::new(rng)),
                std::io::sink(),
            );
            thread::scope(|s| unsafe {
                let j1 = thread::Builder::new()
                    .stack_size(8 << 20)
                    .name(String::from("worker0"))
                    .spawn_scoped(s, || {
                        processor.loop_thread::<true, true>(
                            kernel0,
                            b"",
                            #[cfg(feature = "hpc")]
                            arg_core0,
                        )
                    })
                    .expect("Failed to spawn worker thread 0");

                let j2 = thread::Builder::new()
                    .stack_size(8 << 20)
                    .name(String::from("worker1"))
                    .spawn_scoped(s, || {
                        processor.loop_thread::<true, true>(
                            kernel1,
                            b"",
                            #[cfg(feature = "hpc")]
                            arg_core1,
                        )
                    })
                    .expect("Failed to spawn worker thread 1");

                let mut time_since_last_stat = std::time::Instant::now();
                let mut elapsed = std::time::Duration::ZERO;
                let mut last_frames_processed_half = 0;

                loop {
                    std::thread::park_timeout(std::time::Duration::from_millis(1000));
                    let now = std::time::Instant::now();
                    let delta_time = now.duration_since(time_since_last_stat);
                    if delta_time > std::time::Duration::from_secs(1) {
                        elapsed += delta_time;
                        time_since_last_stat = now;
                        let new_frames_processed_half = processor
                            .buffers
                            .half_frames_processed
                            .load(std::sync::atomic::Ordering::Relaxed);
                        // assuming 100k frames a second (more than 10 times my maximum possible benchmark speed, only achievable by feeding with /dev/zero or PRNG like Xorshift)
                        let delta_frames =
                            (new_frames_processed_half - last_frames_processed_half) * 2;
                        last_frames_processed_half = new_frames_processed_half;
                        let delta_time_us = delta_time.as_micros() as u64;
                        eprintln!(
                            "{} new frames processed ({} fps), {} total frames processed ({} fps overall)",
                            delta_frames,
                            1_000_000 * delta_frames / delta_time_us,
                            new_frames_processed_half * 2,
                            // this LHS is likely to be the first to overflow, ( 1_000_000 * 2 ) < 2^21, so we have at least 2^43 * 2 frames to work with
                            // it takes ~218.15 days to overflow
                            1_000_000 * 2 * last_frames_processed_half / elapsed.as_micros() as u64
                        );
                    }
                    if j1.is_finished() {
                        match j1.join() {
                            Ok(r) => match r {
                                Ok(_) => {
                                    std::process::exit(0);
                                }
                                Err(e) => {
                                    eprintln!("IO Error in worker thread 0: {:?}", e);
                                    std::process::exit(5);
                                }
                            },
                            Err(e) => {
                                eprintln!("Fatal Error in worker thread 0: {:?}", e);
                                std::process::exit(128);
                            }
                        }
                    }
                    if j2.is_finished() {
                        match j2.join() {
                            Ok(r) => match r {
                                Ok(_) => {
                                    std::process::exit(0);
                                }
                                Err(e) => {
                                    eprintln!("IO Error in worker thread 1: {:?}", e);
                                    std::process::exit(5);
                                }
                            },
                            Err(e) => {
                                eprintln!("Fatal Error in worker thread 1: {:?}", e);
                                std::process::exit(128);
                            }
                        }
                    }
                }
            });
        }
        Some(("random-stream", _)) => {
            use rand::RngCore;
            use std::hash::BuildHasher;
            let key = RandomState::new().hash_one(0);
            let seeded = rand::rngs::SmallRng::seed_from_u64(key);
            let mut rng = seeded;
            let mut buf = [0; 8192];
            let mut output = std::io::BufWriter::new(std::io::stdout());
            loop {
                rng.fill_bytes(&mut buf);
                output.write_all(&buf).expect("Failed to write to stdout");
            }
        }

        Some(("pipe", sub_matches)) => {
            let arg_input = sub_matches.get_one::<String>("input").unwrap().clone();
            let arg_output = sub_matches.get_one::<String>("output").unwrap().clone();
            #[cfg(feature = "hpc")]
            let arg_core0 = sub_matches.get_one::<usize>("core0").cloned();
            #[cfg(feature = "hpc")]
            let arg_core1 = sub_matches.get_one::<usize>("core1").cloned();
            let arg_stats = sub_matches.get_flag("stats");
            let mut arg_output_format = sub_matches.get_one::<String>("format").unwrap().clone();

            let osep = arg_output_format.split('+').last().and_then(|s| match s {
                #[cfg(all(target_family = "unix", not(target_os = "macos")))]
                "nl" | "NL" => Some(b"\n".as_slice()),
                #[cfg(target_os = "macos")]
                "nl" | "NL" => Some(b"\r".as_slice()),
                #[cfg(all(not(target_family = "unix"), not(target_os = "macos")))]
                "nl" | "NL" => Some(b"\r\n".as_slice()),
                "lf" | "LF" => Some(b"\n".as_slice()),
                "crlf" | "CRLF" => Some(b"\r\n".as_slice()),
                "cr" | "CR" => Some(b"\r".as_slice()),
                "nul" | "NUL" | "null" | "NULL" => Some(b"\0".as_slice()),
                _ => None,
            });

            if osep.is_some() {
                let mut cuts = arg_output_format
                    .split('+')
                    .rev()
                    .skip(1)
                    .collect::<Vec<_>>();
                cuts.reverse();
                arg_output_format = cuts.join("+");
            }

            let osep = osep.unwrap_or(b"");

            let (kernel0, kernel1) = {
                (
                    yume_pdq::kernel::smart_kernel(),
                    yume_pdq::kernel::smart_kernel(),
                )
            };

            let reader = open_reader(&arg_input).unwrap();
            let writer = open_writer(&arg_output).unwrap();

            macro_rules! match_format {
                ($($spec:pat => $otype:ident),* $(,)?) => {
                    match arg_output_format.as_str() {
                        $($spec => {
                            let processor = PairProcessor::<_, _, _, $otype>::new_fast(reader, writer);
                            thread::scope(|s| {
                                let j1 = thread::Builder::new().stack_size(8 << 20).name(String::from("worker0")).spawn_scoped(s, || {
                                    if arg_stats {
                                        unsafe { processor.loop_thread::<true, true>(kernel0, osep, #[cfg(feature = "hpc")] arg_core0) }
                                    } else {
                                        unsafe { processor.loop_thread::<true, false>(kernel0, osep, #[cfg(feature = "hpc")] arg_core0) }
                                    }
                                }).expect("Failed to spawn worker thread 0");
                                let j2 = thread::Builder::new().stack_size(8 << 20).name(String::from("worker1")).spawn_scoped(s, || {
                                    if arg_stats {
                                        unsafe { processor.loop_thread::<false, true>(kernel1, osep, #[cfg(feature = "hpc")] arg_core1) }
                                    } else {
                                        unsafe { processor.loop_thread::<false, false>(kernel1, osep, #[cfg(feature = "hpc")] arg_core1) }
                                    }
                                }).expect("Failed to spawn worker thread 1");

                                let mut time_since_last_stat = std::time::Instant::now();
                                let mut elapsed = std::time::Duration::ZERO;
                                let mut last_frames_processed_half = 0;
                                loop {
                                    std::thread::park_timeout(std::time::Duration::from_millis(1000));
                                    if arg_stats {
                                        let now = std::time::Instant::now();
                                        let delta_time = now.duration_since(time_since_last_stat);
                                        if delta_time > std::time::Duration::from_secs(1) {
                                            elapsed += delta_time;
                                            time_since_last_stat = now;
                                            let new_frames_processed_half = processor.buffers.half_frames_processed.load(std::sync::atomic::Ordering::Relaxed);
                                            // assuming 100k frames a second (more than 10 times my maximum possible benchmark speed, only achievable by feeding with /dev/zero or PRNG like Xorshift)
                                            let delta_frames = (new_frames_processed_half - last_frames_processed_half) * 2;
                                            last_frames_processed_half = new_frames_processed_half;
                                            let delta_time_us = delta_time.as_micros() as u64;
                                            eprintln!(
                                                "{} new frames processed ({} fps), {} total frames processed ({} fps overall)",
                                                delta_frames,
                                                1_000_000 * delta_frames / delta_time_us ,
                                                new_frames_processed_half * 2,
                                                // this LHS is likely to be the first to overflow, ( 1_000_000 * 2 ) < 2^21, so we have at least 2^43 * 2 frames to work with
                                                // it takes ~218.15 days to overflow
                                                1_000_000 * 2 * last_frames_processed_half  / elapsed.as_micros() as u64
                                            );
                                        }
                                    }
                                    if j1.is_finished() {
                                        match j1.join() {
                                            Ok(r) => match r {
                                                Ok(_) => {
                                                    std::process::exit(0);
                                                }
                                                Err(e) => {
                                                    eprintln!("IO Error in worker thread 0: {:?}", e);
                                                    std::process::exit(5);
                                                }
                                            }
                                            Err(e) => {
                                                eprintln!("Fatal Error in worker thread 0: {:?}", e);
                                                std::process::exit(128);
                                            }
                                        }
                                    }
                                    if j2.is_finished() {
                                        match j2.join() {
                                            Ok(r) => match r {
                                                Ok(_) => {
                                                    std::process::exit(0);
                                                }
                                                Err(e) => {
                                                    eprintln!("IO Error in worker thread 1: {:?}", e);
                                                    std::process::exit(5);
                                                }
                                            }
                                            Err(e) => {
                                                eprintln!("Fatal Error in worker thread 1: {:?}", e);
                                                std::process::exit(128);
                                            }
                                        }
                                    }
                                }
                            });
                        }
                        )*
                        _ => {
                            eprintln!(r#"invalid output format: '{}',
                            
                            please try one of the following, you can append (+nl, +lf, +crlf, +cr, +nul) to the format to add a separator:

                            'raw', 'RAW': raw binary output
                            'q+raw', 'q+RAW': raw binary output with quality score prefix in float format little endian

                            'hex', 'HEX': hex output (case-sensitive)
                            'q+hex', 'q+HEX': hex output with ASCII decimal quality score prefix separated by a colon

                            'bin', 'BIN': binary output
                            'q+bin', 'q+BIN': binary output with ASCII decimal quality score prefix separated by a colon
                            "#, arg_output_format);
                            std::process::exit(255);
                        }
                    }
                }
            }

            match_format!(
                "raw" | "Raw" | "RAW" => OUTPUT_TYPE_RAW,
                "q+raw" | "q+Raw" | "q+RAW" => OUTPUT_TYPE_RAW_PREFIX_QUALITY,

                "hex" => OUTPUT_TYPE_ASCII_HEX,
                "HEX" => OUTPUT_TYPE_ASCII_HEX_UPPER,
                "bin" | "BIN" => OUTPUT_TYPE_ASCII_BINARY,

                "q+hex" => OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY,
                "q+HEX" => OUTPUT_TYPE_ASCII_HEX_UPPER_PREFIX_QUALITY,

                "q+bin" => OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY,
                "q+BIN" => OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY,
            );
        }
        #[cfg(feature = "cli-bench")]
        Some(("bench", sub_matches)) => {
            use std::hash::BuildHasher;
            let num = RandomState::new().hash_one(0);
            let rng = rand::rngs::SmallRng::seed_from_u64(num);
            // replicate the indirection on the real version
            let reader = std::io::BufReader::new(RandReadAdaptor::new(rng));
            let rng2 = rand::rngs::SmallRng::seed_from_u64(num);
            let reader2 = std::io::BufReader::new(RandReadAdaptor::new(rng2));
            let rng3 = rand::rngs::SmallRng::seed_from_u64(num);
            let reader3 = std::io::BufReader::new(RandReadAdaptor::new(rng3));
            let mut crit = criterion::Criterion::default().without_plots();
            #[cfg_attr(not(feature = "hpc"), expect(unused))]
            let core0 = sub_matches.get_one::<usize>("core0").map(|s| *s);
            #[cfg_attr(not(feature = "hpc"), expect(unused))]
            let core1 = sub_matches.get_one::<usize>("core1").map(|s| *s);

            let (mut pipe_rx, pipe_tx) = std::io::pipe().expect("Failed to create loopback pipe");
            let (mut pipe_rx_hex, pipe_tx_hex) =
                std::io::pipe().expect("Failed to create loopback pipe");
            let (mut pipe_rx_bin, pipe_tx_bin) =
                std::io::pipe().expect("Failed to create loopback pipe");

            let processor = PairProcessor::<_, _, _, OUTPUT_TYPE_RAW>::new_fast(reader, pipe_tx);

            let processor_hex =
                PairProcessor::<_, _, _, OUTPUT_TYPE_ASCII_HEX>::new_fast(reader2, pipe_tx_hex);

            let processor_bin =
                PairProcessor::<_, _, _, OUTPUT_TYPE_ASCII_BINARY>::new_fast(reader3, pipe_tx_bin);

            thread::scope(|s| {
                let kernel0 = yume_pdq::kernel::smart_kernel();
                let kernel1 = yume_pdq::kernel::smart_kernel();
                let kernel2 = yume_pdq::kernel::smart_kernel();
                let kernel3 = yume_pdq::kernel::smart_kernel();
                let kernel4 = yume_pdq::kernel::smart_kernel();
                let kernel5 = yume_pdq::kernel::smart_kernel();
                s.spawn(|| unsafe {
                    processor
                        .loop_thread::<true, false>(
                            kernel0,
                            b"",
                            #[cfg(feature = "hpc")]
                            core0,
                        )
                        .expect("Failed to spawn worker thread 0");
                });

                s.spawn(|| unsafe {
                    processor
                        .loop_thread::<false, false>(
                            kernel1,
                            b"",
                            #[cfg(feature = "hpc")]
                            core1,
                        )
                        .expect("Failed to spawn worker thread 1");
                });

                s.spawn(|| unsafe {
                    processor_hex
                        .loop_thread::<true, false>(
                            kernel2,
                            b"",
                            #[cfg(feature = "hpc")]
                            core0,
                        )
                        .expect("Failed to spawn worker thread 2");
                });

                s.spawn(|| unsafe {
                    processor_hex
                        .loop_thread::<false, false>(
                            kernel3,
                            b"",
                            #[cfg(feature = "hpc")]
                            core1,
                        )
                        .expect("Failed to spawn worker thread 3");
                });

                s.spawn(|| unsafe {
                    processor_bin
                        .loop_thread::<true, false>(
                            kernel4,
                            b"",
                            #[cfg(feature = "hpc")]
                            core0,
                        )
                        .expect("Failed to spawn worker thread 4");
                });

                s.spawn(|| unsafe {
                    processor_bin
                        .loop_thread::<false, false>(
                            kernel5,
                            b"",
                            #[cfg(feature = "hpc")]
                            core1,
                        )
                        .expect("Failed to spawn worker thread 5");
                });

                let mut group = crit.benchmark_group("pdq_pingpong");
                group.throughput(criterion::Throughput::Bytes(512 * 512));
                group.measurement_time(std::time::Duration::from_secs(15));
                // make sure we drained everything already processed
                group.warm_up_time(std::time::Duration::from_secs(10));
                group.bench_function("hash", |b| {
                    b.iter(|| {
                        let mut hash: [u8; 256 / 8] = [0; 256 / 8];
                        pipe_rx.read_exact(&mut hash).unwrap();
                        hash
                    });
                });

                drop(group);

                let mut group = crit.benchmark_group("pdq_pingpong_hex");
                group.throughput(criterion::Throughput::Bytes(512 * 512));
                group.measurement_time(std::time::Duration::from_secs(15));
                // make sure we drained everything already processed
                group.warm_up_time(std::time::Duration::from_secs(10));
                group.bench_function("hash", |b| {
                    b.iter(|| {
                        let mut hash: [u8; 256 / 8 * 2] = [0; 256 / 8 * 2];
                        pipe_rx_hex.read_exact(&mut hash).unwrap();
                        hash
                    });
                });

                drop(group);

                let mut group = crit.benchmark_group("pdq_pingpong_bin");
                group.throughput(criterion::Throughput::Bytes(512 * 512));
                group.measurement_time(std::time::Duration::from_secs(15));
                // make sure we drained everything already processed
                group.warm_up_time(std::time::Duration::from_secs(10));
                group.bench_function("hash", |b| {
                    b.iter(|| {
                        let mut hash: [u8; 256 / 8 * 8] = [0; 256 / 8 * 8];
                        pipe_rx_bin.read_exact(&mut hash).unwrap();
                        hash
                    });
                });

                drop(group);

                std::process::exit(0);
            });
        }
        _ => {
            eprintln!("Invalid subcommand, try --help for usage");
            std::process::exit(255);
        }
    }
}
