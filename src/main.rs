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
                        .help("Output buffer size in bytes")
                        .long_help(
                            "Size of the output buffer in bytes. Larger buffers may improve performance \
                             when writing to files or pipes.",
                        )
                        .value_parser(value_parser!(u32)),
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
            Command::new("random-stream")
                .about("Generate random byte stream")
                .long_about(
                    "Generates a continuous stream of random bytes (0-255) to stdout. \
                     Useful for testing and benchmarking.",
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

const OUTPUT_TYPE_RAW: u8 = 0;
const OUTPUT_TYPE_RAW_PREFIX_QUALITY: u8 = 1;
const OUTPUT_TYPE_ASCII_HEX: u8 = 2;
const OUTPUT_TYPE_ASCII_BINARY: u8 = 3;
const OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY: u8 = 4;
const OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY: u8 = 5;

const OUTPUT_SEPARATOR_NONE: u8 = 0;
const OUTPUT_SEPARATOR_CR: u8 = 1;
const OUTPUT_SEPARATOR_LF: u8 = 2;
const OUTPUT_SEPARATOR_CRLF: u8 = 3;

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
    const OUTPUT_SEPARATOR: u8,
    const OUTPUT_UPPER: bool,
> where
    K::OutputDimension: DivisibleBy8,
{
    barrier: std::sync::Barrier,
    buffers: Box<PairBuffer<K>>,
    reader: R,
    writer: W,
}

impl<
    K: Kernel + Send + Sync,
    R: Read + Send + Sync,
    W: Write + Send + Sync,
    const OUTPUT_TYPE: u8,
    const OUTPUT_SEPARATOR: u8,
    const OUTPUT_UPPER: bool,
> PairProcessor<K, R, W, OUTPUT_TYPE, OUTPUT_SEPARATOR, OUTPUT_UPPER>
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
                                        *a += if OUTPUT_UPPER {
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

                    match OUTPUT_SEPARATOR {
                        OUTPUT_SEPARATOR_NONE => {}
                        OUTPUT_SEPARATOR_CR => {
                            writer_mut.write_all(b"\r")?;
                        }
                        OUTPUT_SEPARATOR_LF => {
                            writer_mut.write_all(b"\n")?;
                        }
                        OUTPUT_SEPARATOR_CRLF => {
                            writer_mut.write_all(b"\r\n")?;
                        }
                        _ => unreachable!(),
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

fn open_reader(spec: &str) -> Result<Box<dyn Read + Send + Sync>, std::io::Error> {
    if spec == "-" {
        Ok(Box::new(std::io::stdin()))
    } else {
        println!("Taking input from {}", spec);
        let file = std::fs::File::open(spec)?;
        Ok(Box::new(file))
    }
}

fn open_writer(
    spec: &str,
    buffer: Option<u32>,
) -> Result<Box<dyn Write + Send + Sync>, std::io::Error> {
    if spec == "-" {
        if let Some(buffer) = buffer {
            Ok(Box::new(std::io::BufWriter::with_capacity(
                buffer as usize,
                std::io::stdout(),
            )))
        } else {
            Ok(Box::new(std::io::stdout()))
        }
    } else {
        if let Some(buffer) = buffer {
            let file = std::fs::File::create(spec)?;
            Ok(Box::new(std::io::BufWriter::with_capacity(
                buffer as usize,
                file,
            )))
        } else {
            let file = std::fs::File::create(spec)?;
            Ok(Box::new(file))
        }
    }
}

fn type_name_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
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
            println!("=== Feature flag infomation ===\n");
            println!(
                "  Capability of this binary: {}",
                env!("TARGET_SPECIFIC_CLI_MESSAGE")
            );

            println!(
                "  Supported CPU features: {}",
                env!("BUILD_CFG_TARGET_FEATURES")
            );

            println!("\n=== Runtime Routing Infomation ===\n");

            let kernel = yume_pdq::kernel::smart_kernel();

            let ident = kernel.ident();

            println!("  Runtime decision: {}", ident);
            println!();
            println!("  Runtime decision details: {:?}", ident);
            println!();
            println!("  Router type: {}", type_name_of(&kernel));
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
            let arg_output_buffer = sub_matches.get_one::<u32>("output_buffer").cloned();
            #[cfg(feature = "hpc")]
            let arg_core0 = sub_matches.get_one::<usize>("core0").cloned();
            #[cfg(feature = "hpc")]
            let arg_core1 = sub_matches.get_one::<usize>("core1").cloned();
            let arg_stats = sub_matches.get_flag("stats");
            let arg_output_format = sub_matches.get_one::<String>("format").unwrap().clone();

            let (kernel0, kernel1) = {
                (
                    yume_pdq::kernel::smart_kernel(),
                    yume_pdq::kernel::smart_kernel(),
                )
            };

            let reader = open_reader(&arg_input).unwrap();
            let writer = open_writer(&arg_output, arg_output_buffer).unwrap();

            macro_rules! match_format {
                ($($spec:pat => ($otype:ident, $osep:ident, $oupper:literal)),* $(,)?) => {
                    match arg_output_format.as_str() {
                        $($spec => {

                            let processor = PairProcessor::<_, _, _, $otype, $osep, $oupper>::new_fast(reader, writer);
                            thread::scope(|s| {
                                let j1 = thread::Builder::new().stack_size(8 << 20).name(String::from("worker0")).spawn_scoped(s, || {
                                    if arg_stats {
                                        unsafe { processor.loop_thread::<true, true>(kernel0, #[cfg(feature = "hpc")] arg_core0) }
                                    } else {
                                        unsafe { processor.loop_thread::<true, false>(kernel0, #[cfg(feature = "hpc")] arg_core0) }
                                    }
                                }).expect("Failed to spawn worker thread 0");
                                let j2 = thread::Builder::new().stack_size(8 << 20).name(String::from("worker1")).spawn_scoped(s, || {
                                    if arg_stats {
                                        unsafe { processor.loop_thread::<false, true>(kernel1, #[cfg(feature = "hpc")] arg_core1) }
                                    } else {
                                        unsafe { processor.loop_thread::<false, false>(kernel1, #[cfg(feature = "hpc")] arg_core1) }
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
                            
                            please try one of the following:

                            'raw', 'RAW': raw binary output
                            'q+raw', 'q+RAW': raw binary output with quality score prefix in float format little endian

                            'hex', 'HEX': hex output
                            'q+hex', 'q+HEX': hex output with ASCII decimal quality score prefix separated by a colon

                            'bin', 'BIN': binary output
                            'q+bin', 'q+BIN': binary output with ASCII decimal quality score prefix separated by a colon

                            'hex+lf', 'HEX+LF': hex output with line feed separator
                            'hex+crlf', 'HEX+CRLF': hex output with carriage return line feed separator
                            'hex+cr', 'HEX+CR': hex output with carriage return separator
                            
                            'bin+lf', 'BIN+LF': binary output with line feed separator
                            'bin+crlf', 'BIN+CRLF': binary output with carriage return line feed separator
                            'bin+cr', 'BIN+CR': binary output with carriage return separator    
                            
                            'q+hex', 'q+HEX': hex output with ASCII decimal quality score prefix separated by a colon
                            'q+bin', 'q+BIN': binary output with ASCII decimal quality score prefix separated by a colon

                            'hex+lf', 'HEX+LF': hex output with line feed separator
                            'hex+crlf', 'HEX+CRLF': hex output with carriage return line feed separator
                            'hex+cr', 'HEX+CR': hex output with carriage return separator
                            
                            'q+hex+lf', 'q+HEX+LF': hex output with ASCII decimal quality score prefix separated by a colon and line feed separator     
                            'q+hex+crlf', 'q+HEX+CRLF': hex output with ASCII decimal quality score prefix separated by a colon and carriage return line feed separator
                            'q+hex+cr', 'q+HEX+CR': hex output with ASCII decimal quality score prefix separated by a colon and carriage return separator

                            'q+bin+lf', 'q+BIN+LF': binary output with ASCII decimal quality score prefix separated by a colon and line feed separator
                            'q+bin+crlf', 'q+BIN+CRLF': binary output with ASCII decimal quality score prefix separated by a colon and carriage return line feed separator
                            'q+bin+cr', 'q+BIN+CR': binary output with ASCII decimal quality score prefix separated by a colon and carriage return separator

                            "#, arg_output_format);
                            std::process::exit(255);
                        }
                    }
                }
            }

            match_format!(
                "raw" | "Raw" | "RAW" => (OUTPUT_TYPE_RAW, OUTPUT_SEPARATOR_NONE, false),
                "q+raw" | "q+Raw" | "q+RAW" => (OUTPUT_TYPE_RAW_PREFIX_QUALITY, OUTPUT_SEPARATOR_NONE, false),

                "hex" => (OUTPUT_TYPE_ASCII_HEX, OUTPUT_SEPARATOR_NONE, false),
                "HEX" => (OUTPUT_TYPE_ASCII_HEX, OUTPUT_SEPARATOR_NONE, true),
                "bin" | "BIN" => (OUTPUT_TYPE_ASCII_BINARY, OUTPUT_SEPARATOR_NONE, false),

                "hex+lf" | "hex+LF" => (OUTPUT_TYPE_ASCII_HEX, OUTPUT_SEPARATOR_LF, false),
                "HEX+lf" | "HEX+LF" => (OUTPUT_TYPE_ASCII_HEX, OUTPUT_SEPARATOR_LF, true),
                "hex+crlf" | "hex+CRLF" => (OUTPUT_TYPE_ASCII_HEX, OUTPUT_SEPARATOR_CRLF, false),
                "HEX+crlf" | "HEX+CRLF" => (OUTPUT_TYPE_ASCII_HEX, OUTPUT_SEPARATOR_CRLF, true),
                "hex+cr" | "hex+CR" => (OUTPUT_TYPE_ASCII_HEX, OUTPUT_SEPARATOR_CR, false),
                "HEX+cr" | "HEX+CR" => (OUTPUT_TYPE_ASCII_HEX, OUTPUT_SEPARATOR_CR, true),

                "bin+lf" | "BIN+LF" => (OUTPUT_TYPE_ASCII_BINARY, OUTPUT_SEPARATOR_LF, false),
                "bin+crlf" | "BIN+CRLF" => (OUTPUT_TYPE_ASCII_BINARY, OUTPUT_SEPARATOR_CRLF, false),
                "bin+cr" | "BIN+CR" => (OUTPUT_TYPE_ASCII_BINARY, OUTPUT_SEPARATOR_CR, false),

                "q+hex" => (OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY, OUTPUT_SEPARATOR_NONE, false),
                "q+HEX" => (OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY, OUTPUT_SEPARATOR_NONE, true),
                "q+hex+lf" | "q+hex+LF" => (OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY, OUTPUT_SEPARATOR_LF, false),
                "q+HEX+lf" | "q+HEX+LF" => (OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY, OUTPUT_SEPARATOR_LF, true),
                "q+hex+crlf" | "q+hex+CRLF" => (OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY, OUTPUT_SEPARATOR_CRLF, false),
                "q+HEX+crlf" | "q+HEX+CRLF" => (OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY, OUTPUT_SEPARATOR_CRLF, true),
                "q+hex+cr" | "q+hex+CR" => (OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY, OUTPUT_SEPARATOR_CR, false),
                "q+HEX+cr" | "q+HEX+CR" => (OUTPUT_TYPE_ASCII_HEX_PREFIX_QUALITY, OUTPUT_SEPARATOR_CR, true),

                "q+bin" => (OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY, OUTPUT_SEPARATOR_NONE, false),
                "q+BIN" => (OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY, OUTPUT_SEPARATOR_NONE, true),
                "q+bin+lf" | "q+bin+LF" => (OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY, OUTPUT_SEPARATOR_LF, false),
                "q+BIN+lf" | "q+BIN+LF" => (OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY, OUTPUT_SEPARATOR_LF, true),
                "q+bin+crlf" | "q+bin+CRLF" => (OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY, OUTPUT_SEPARATOR_CRLF, false),
                "q+BIN+crlf" | "q+BIN+CRLF" => (OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY, OUTPUT_SEPARATOR_CRLF, true),
                "q+bin+cr" | "q+bin+CR" => (OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY, OUTPUT_SEPARATOR_CR, false),
                "q+BIN+cr" | "q+BIN+CR" => (OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY, OUTPUT_SEPARATOR_CR, true),

            );
        }
        _ => {
            eprintln!("Invalid subcommand, try --help for usage");
            std::process::exit(255);
        }
    }
}
