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

use clap::{Parser, Subcommand};
use rand::SeedableRng;
use std::{
    hash::RandomState, io::{Read, Write}, ops::{Div, Mul}, ptr::addr_of, thread::{self}
};

use generic_array::{
    ArrayLength,
    sequence::{Flatten, GenericSequence},
    typenum::{U4, U8, Unsigned},
};
use yume_pdq::{
    alignment::Align32, kernel::{self, type_traits::DivisibleBy8, Kernel }, lut_utils, GenericArray, PDQHash, PDQHashF
};

#[derive(Parser)]
#[command(
    name = "yume-pdq",
    about = "Fast PDQ perceptual image hashing implementation",
    long_about = concat!(
        "A high-performance implementation of the PDQ perceptual image hashing algorithm. \
         Supports various input/output formats and hardware acceleration.",
        "\n\n",
        env!("TARGET_SPECIFIC_CLI_MESSAGE"),
        "\n\n",
        "Build Facts:",
        "\n",
        "  Version: ", env!("CARGO_PKG_VERSION"),
        " built on ", env!("VERGEN_BUILD_DATE"),
        "\n",
        "  Target & Optimization: ", env!("VERGEN_CARGO_TARGET_TRIPLE"),
        " -O", env!("VERGEN_CARGO_OPT_LEVEL"),
        "\n",
        "  Feature Flags: ", env!("VERGEN_CARGO_FEATURES"),
    ),
    version = env!("CARGO_PKG_VERSION"),
)]
struct Args {
    #[command(subcommand)]
    subcommand: SubCmd,
}

#[derive(Subcommand)]
enum SubCmd {
    #[command(
        name = "pipe",
        about = "Process image stream and output hashes, see 'pipe --help' usage examples",
        long_about = concat!(
            "Reads a stream of 512x512 grayscale images and outputs their PDQ hashes.",
            "\n\n",
            "Usage examples with common tools:",
            "\n\n",
            " * Emit a single image, return the hash in ASCII hex format, prefix by the quality score, pad by a line feed:",
            "\n\n",
            "    >ffmpeg -loglevel error -hide_banner -i test-data/aaa-orig.jpg \\",
            "\n",
            "      -vf \"scale=512:512:force_original_aspect_ratio=disable\" \\",
            "\n",
            "      -frames:v 1 -pix_fmt gray8  -f rawvideo - | yume-pdq pipe -f q+hex+lf",
            "\n\n",
            "       Output: 100.000:58f8f0cee0f4a84f06370a32038f67f0b36e2ed596623e1d33e6b39c4e9c9b22",
            "\n\n\n",
            " * Process every frame of a video stream, return the hash in raw binary format with no padding in between frames: ",
            "\n\n",
            "   >ffmpeg -f lavfi -i testsrc=size=512x512:rate=1  -pix_fmt gray  -f rawvideo - | yume-pdq pipe -f bin",
            "\n\n",
            "       Output: <BINARY_HASH>, expect to see thousands of FPS reported by ffmpeg!",
            "\n\n",
            " * Process an arbitrary list of images, return the hash in ASCII hex format, pad by a line feed:",
            "\n\n",
            "   > for i in (seq 1 1000); ln -s (realpath test-data/aaa-orig.jpg) /tmp/test/$i.jpg; end",
            "\n",
            "   >  time convert 'test-data/*' -resize 512x512! -colorspace gray -depth 8 gray:- \\",
            "\n",
            "   >    | yume-pdq pipe -f 'hex+lf'",
            "\n\n",
            "       Output: d8f8f0cee0f4a84f06370a32038f67f0b36e2ed596621e1d33e6b39c4e9c9b22 (*1000 lines)",
            "\n",
            "       Executed in   26.01 secs    fish           external",
            "\n",
            "       usr time   47.76 secs    0.00 millis   47.76 secs",
            "\n",
            "       sys time    9.43 secs    2.77 millis    9.43 secs",
        )
    )]
    Pipe(PipeArgs),

    #[command(
        name = "random-stream",
        about = "Generate random byte stream",
        long_about = "Generates a continuous stream of random bytes (0-255) to stdout. \
                      Useful for testing and benchmarking."
    )]
    RandomStream,

    #[command(
        name = "vectorization-info",
        about = "Display vectorization information",
        long_about = "Displays diagnostic information about the vectorization capabilities of the current CPU. HIGHLY RECOMMENDED to run this command before deploying on a new micro-architecture."
    )]
    VectorizationInfo,
}

#[derive(Parser)]
struct PipeArgs {
    #[arg(
        short,
        long,
        default_value = "-",
        help = "Input source (- for stdin)",
        long_help = "Source of input images. Use '-' for stdin or provide a file path. \
                     Expects 512x512 grayscale images in raw format."
    )]
    input: String,

    #[arg(
        short,
        long,
        default_value = "-",
        help = "Output destination (- for stdout)",
        long_help = "Destination for hash output. Use '-' for stdout or provide a file path."
    )]
    output: String,

    #[arg(
        long,
        help = "Output buffer size in bytes",
        long_help = "Size of the output buffer in bytes. Larger buffers may improve performance \
                     when writing to files or pipes."
    )]
    output_buffer: Option<u32>,

    #[arg(
        long,
        help = "Force scalar processing",
        long_help = "Disable SIMD acceleration and use scalar processing only. \
                     Useful for testing or when encountering issues with vectorized code."
    )]
    force_scalar: bool,

    #[arg(
        long,
        help = "Show processing statistics",
        long_help = "Display periodic statistics about processing speed and throughput to stderr."
    )]
    stats: bool,

    #[arg(
        short = 'f',
        long = "format",
        default_value = "bin",
        help = "Output format specification",
        long_help = "Specify the output format for hashes. Available formats:\n\
                     - raw/RAW: Binary output\n\
                     - hex/HEX: Hexadecimal output (lowercase/uppercase)\n\
                     - bin/BIN: Binary string output\n\
                     Modifiers:\n\
                     - q+: Prefix with quality score\n\
                     - +lf/+cr/+crlf: Add line ending\n\
                     Examples: q+hex+lf, HEX+crlf, q+bin"
    )]
    output_format: String,
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
    // defensive padding at least 2 times register width to ensure if there was a buffer overrun, it's not catastrophic
    _pad0: BufferPad,
    buf1_input: Align32<GenericArray<GenericArray<f32, K::InputDimension>, K::InputDimension>>,
    _pad1: BufferPad,
    buf1_intermediate:
        Align32<GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>>,
    _pad2: BufferPad,
    buf1_pdqf: Align32<PDQHashF<K::InternalFloat, K::OutputDimension>>,
    _pad3: BufferPad,
    buf1_output: PDQHash<K::OutputDimension>,
    _pad4: BufferPad,
    buf2_input: Align32<GenericArray<GenericArray<f32, K::InputDimension>, K::InputDimension>>,
    _pad5: BufferPad,
    buf2_intermediate:
        Align32<GenericArray<GenericArray<K::InternalFloat, K::Buffer1WidthX>, K::Buffer1LengthY>>,
    _pad6: BufferPad,
    buf2_pdqf: Align32<PDQHashF<K::InternalFloat, K::OutputDimension>>,
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
    K::OutputDimension: DivisibleBy8 + Div<U4>,
    <K::OutputDimension as Div<U4>>::Output: ArrayLength,
    U8: Mul<K::OutputDimension> + Mul<<<K as Kernel>::OutputDimension as DivisibleBy8>::Output>,
    <U8 as Mul<K::OutputDimension>>::Output: ArrayLength,
    <U8 as Mul<<<K as Kernel>::OutputDimension as DivisibleBy8>::Output>>::Output: ArrayLength,
    <K as Kernel>::OutputDimension:
        Mul<<<K as Kernel>::OutputDimension as Div<U4>>::Output>,
    <<K as Kernel>::OutputDimension as Mul<<<K as Kernel>::OutputDimension as Div<U4>>::Output>>::Output: ArrayLength,

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
        }
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
    ) -> Result<(), std::io::Error> {
        let mut have_data = false;
        let mut i_am_reading = I_AM_READING_INITIALLY;
        let mut frames_processed = 0;
        let mut last_checkpoint = std::time::Instant::now();
        let mut last_checkpoint_frames = 0u64;
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
                                        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "Unexpected EOF while reading the middle of a frame"));
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
                                GenericArray<u8, K::OutputDimension>,
                                <K::OutputDimension as Div<U4>>::Output,
                            > = GenericArray::generate(|_| GenericArray::generate(|_| b'0'));
                            buf.iter_mut()
                                .flatten()
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
                            writer_mut.write_all(Flatten::flatten(buf).as_slice())?;
                        }
                        OUTPUT_TYPE_ASCII_BINARY | OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY => {
                            if OUTPUT_TYPE == OUTPUT_TYPE_ASCII_BINARY_PREFIX_QUALITY {
                                write!(writer_mut, "{0:02.3}:", quality * 100.0)?;
                            }

                            for i in 0..K::OutputDimension::USIZE {
                                let row_buf: GenericArray<
                                    GenericArray<u8, U8>,
                                    <K::OutputDimension as DivisibleBy8>::Output,
                                > = GenericArray::generate(|j| {
                                    GenericArray::from(
                                        lut_utils::BINARY_PRINTING[output_ref[i][j] as usize],
                                    )
                                });

                                writer_mut.write_all(Flatten::flatten(&row_buf).as_slice())?;
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
                        frames_processed += 1;
                        if frames_processed & (1 << 10) == 0 {
                            let now = std::time::Instant::now();
                            let delta_time = now.duration_since(last_checkpoint);
                            let delta_frames = (frames_processed - last_checkpoint_frames) * 2;
                            if delta_time > std::time::Duration::from_secs(1) {
                                eprintln!("{} frames processed in {:?} ({} fps)", delta_frames, delta_time, delta_frames as f64 / delta_time.as_secs_f64());
                                last_checkpoint = now;
                                last_checkpoint_frames = frames_processed;
                            }
                        }
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

fn open_writer(spec: &str, buffer: Option<u32>) -> Result<Box<dyn Write + Send + Sync>, std::io::Error> {
    if spec == "-" {
        if let Some(buffer) = buffer {
            Ok(Box::new(std::io::BufWriter::with_capacity(buffer as usize, std::io::stdout())))
        } else {
            Ok(Box::new(std::io::stdout()))
        }
    } else {
        if let Some(buffer) = buffer {
            let file = std::fs::File::create(spec)?;
            Ok(Box::new(std::io::BufWriter::with_capacity(buffer as usize, file)))
        } else {
            let file = std::fs::File::create(spec)?;
            Ok(Box::new(file))
        }
    }
}

fn main() {
    let args = Args::parse();

    #[cfg(target_arch = "x86_64")]
    let has_avx2_runtime = is_x86_feature_detected!("avx2");
    #[cfg(target_arch = "x86_64")]
    let has_avx512f_runtime = is_x86_feature_detected!("avx512f");

    match args.subcommand {
        #[cfg(target_arch = "x86_64")]
        SubCmd::VectorizationInfo => {

            println!("Capability of this binary: {}", env!("TARGET_SPECIFIC_CLI_MESSAGE"));
            println!();
            println!("On your processor, this build can use the following features:");

            #[cfg(target_feature = "avx2")]
            println!("AVX2: {}", if has_avx2_runtime { if cfg!(feature = "avx512") { "Detected but this build uses AVX512 kernels" } else { "Yes" } } else { "No" });

            #[cfg(not(target_feature = "avx2"))]
            if has_avx2_runtime {
                println!("AVX2: SSE only, did you set RUSTFLAGS=-Ctarget-feature=+avx2 or -Ctarget-cpu=native ?");
            }

            #[cfg(feature = "avx512")]
            println!("AVX512F: {}", if has_avx512f_runtime { "Yes" } else { "No" });

            #[cfg(not(feature = "avx512"))]
            if has_avx512f_runtime {
                println!("AVX512F: Auto-vectorization only, optimized kernel disabled by feature flag");
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        SubCmd::VectorizationInfo => {
            eprintln!("Vectorization info is currently only available on x86_64. ARM NEON support is planned.");
        }
        SubCmd::RandomStream => {
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
        
        SubCmd::Pipe(mut args) => {

            #[cfg(target_arch = "x86_64")]
            if !args.force_scalar && !has_avx2_runtime {
                eprintln!("Warning: AVX2 is not available on this CPU, using scalar code.");
                args.force_scalar = true;
            }

            #[cfg(target_arch = "x86_64")]
            let (kernel0, kernel1) = {
                #[cfg(all(feature = "avx512", target_feature = "avx512f"))]
                {
                    (kernel::x86::Avx512F32Kernel, kernel::x86::Avx512F32Kernel)
                }
                #[cfg(all(
                    target_feature = "avx2",
                    not(all(feature = "avx512", target_feature = "avx512f"))
                ))]
                {
                    (kernel::x86::Avx2F32Kernel, kernel::x86::Avx2F32Kernel)
                }
                #[cfg(not(target_feature = "avx2"))]
                {
                    (kernel::DefaultKernel, kernel::DefaultKernel)
                }
            };

            #[cfg(not(target_arch = "x86_64"))]
            let (kernel0, kernel1) = (kernel::DefaultKernel(), kernel::DefaultKernel());

            let reader = open_reader(&args.input).unwrap();
            let writer = open_writer(&args.output, args.output_buffer).unwrap();

            macro_rules! match_format {
                ($($spec:pat => ($otype:ident, $osep:ident, $oupper:literal)),* $(,)?) => {
                    match args.output_format.as_str() {
                        $($spec => {
                            if args.force_scalar {
                                let processor = PairProcessor::<_, _, _, $otype, $osep, $oupper>::new_fast(reader, writer);
                                thread::scope(|s| {
                                    let j1 = thread::Builder::new().stack_size(8 << 20).name(String::from("worker0")).spawn_scoped(s, || {
                                        if args.stats {
                                            unsafe { processor.loop_thread::<true, true>(kernel::DefaultKernel) }
                                        } else {
                                            unsafe { processor.loop_thread::<true, false>(kernel::DefaultKernel) }
                                        }
                                    }).expect("Failed to spawn worker thread 0");
                                    let j2 = thread::Builder::new().stack_size(8 << 20).name(String::from("worker1")).spawn_scoped(s, || {
                                        if args.stats {
                                            unsafe { processor.loop_thread::<false, true>(kernel::DefaultKernel) }
                                        } else {
                                            unsafe { processor.loop_thread::<false, false>(kernel::DefaultKernel) }
                                        }
                                    }).expect("Failed to spawn worker thread 1");

                                    loop {
                                        std::thread::park_timeout(std::time::Duration::from_secs(1));
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
                            } else {
                                let processor = PairProcessor::<_, _, _, $otype, $osep, $oupper>::new_fast(reader, writer);
                                thread::scope(|s| {
                                    let j1 = thread::Builder::new().stack_size(8 << 20).name(String::from("worker0")).spawn_scoped(s, || {
                                        if args.stats {
                                            unsafe { processor.loop_thread::<true, true>(kernel0) }
                                        } else {
                                            unsafe { processor.loop_thread::<true, false>(kernel0) }
                                        }
                                    }).expect("Failed to spawn worker thread 0");
                                    let j2 = thread::Builder::new().stack_size(8 << 20).name(String::from("worker1")).spawn_scoped(s, || {
                                        if args.stats {
                                            unsafe { processor.loop_thread::<false, true>(kernel1) }
                                        } else {
                                            unsafe { processor.loop_thread::<false, false>(kernel1) }
                                        }
                                    }).expect("Failed to spawn worker thread 1");

                                    loop {
                                        std::thread::park_timeout(std::time::Duration::from_secs(1));
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
                        }
                        )*
                        _ => {
                            panic!("invalid output format: {}", args.output_format);
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
    }
}
