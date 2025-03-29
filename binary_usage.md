
### Microarchitecture diagnostics

```
> cargo build --release --features cli -- vectorization-info

Capability of this binary: This yume-pdq kernel has no vectorized superpowers.

On your processor, this build can use the following features:
AVX2: SSE only, did you set RUSTFLAGS=-Ctarget-feature=+avx2 or -Ctarget-cpu=native ?
AVX512F: Auto-vectorization only, optimized kernel disabled by feature flag

> RUSTFLAGS="-Ctarget-cpu=native" cargo run --release --features "cli" -- vectorization-info

=== Feature flag information ===

  Capability of this binary: This yume-pdq kernel has AVX2 yumemi power.
  Supported CPU features: avx,avx2,fma,fxsr,sse,sse2,sse3,sse4.1,sse4.2,ssse3,x87

=== Runtime Routing Information ===

  Runtime decision: avx2_f32

  Runtime decision details: FallbackIdent(decision: using preferred "avx2_f32", fallback was "default_scalar_autovectorized_f32")

  Router type: yume_pdq::kernel::router::KernelRouter<yume_pdq::kernel::x86::Avx2F32Kernel, yume_pdq::kernel::DefaultKernel<typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>, typenum::uint::UInt<typenum::uint::UTerm, typenum::bit::B1>>>
```

Example response on aarch64 sve2:

```
=== Feature flag information ===

  Capability of this binary: This yume-pdq kernel uses LLVM-IR guided SIMD (portable-simd). Check the supported CPU features for your vectorization backend.
  Supported CPU features: neon,sve,sve2

=== Runtime Routing Information ===

  Runtime decision: portable-simd (guided vectorization)

  Runtime decision details: PortableSimd<f32x8>

  Router type: yume_pdq::kernel::portable_simd::PortableSimdF32Kernel<8>
```

### Pipeline Processing


```
> yume-pdq pipe --help

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


Usage: yume-pdq pipe [OPTIONS]

Options:
  -i, --input <INPUT>
          Source of input images. Use '-' for stdin or provide a file path. Expects 512x512 grayscale images in raw format.
          
          [default: -]

  -o, --output <OUTPUT>
          Destination for hash output. Use '-' for stdout or provide a file path.
          
          [default: -]

      --output-buffer <OUTPUT_BUFFER>
          Size of the output buffer in bytes. Larger buffers may improve performance when writing to files or pipes.

      --force-scalar
          Disable SIMD acceleration and use scalar processing only. Useful for testing or when encountering issues with vectorized code.

      --stats
          Display periodic statistics about processing speed and throughput to stderr.

  -f, --format <OUTPUT_FORMAT>
          Specify the output format for hashes. Available formats:
          - raw/RAW: Binary output
          - hex/HEX: Hexadecimal output (lowercase/uppercase)
          - bin/BIN: Binary string output
          Modifiers:
          - q+: Prefix with quality score
          - +lf/+cr/+crlf: Add line ending
          Examples: q+hex+lf, HEX+crlf, q+bin
          
          [default: bin]
```

## Hardware Topology based optimizations

The `pipe` command uses a ping-pong buffer design with 2 threads to maximize throughput. While one thread reads the next frame from IO, the other thread processes the current frame, then they swap roles. This design:

1. Decouples IO latency from PDQ hash computation
2. Allows each thread to keep its working data in L2/L3 caches
3. Minimizes cache thrashing between threads

This introduces potential for further optimizations by explicitly telling the OS each thread should stay on the same cache line as much as possible. with no false sharing.

This is usually not needed as OS handles it reasonably well for most practical use case throughput when there is an upstream bottleneck. If you can do better and need either a more stable throughput or need to squeeze out the last 5% possible speed (warning: you can easily make it worse), compile with `--features hpc` and use the following commands to pin the ping-pong buffer threads to specific cores.

```
> cargo run --release --features "cli hpc" -- list-cores
0
1
...

> cargo run --release --features "cli hpc" -- pipe --core0 0 --core1 1
```

You may find the `lstopo` tool from `hwloc` package useful to understand your system's topology and why one combination may be better or worse than not pinning at all.