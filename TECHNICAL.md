# Technical Design Decisions

This library is designed to be my engineering solution for the high-recall PDQ image matching in production problem. I decided to provide a PDQ variant that is highly optimized for extreme low-latency or high-throughput matching environments but decided to always use an exact thresholding solution.

TLDR: Considering all 8 dihedral transformations, <1ms per image when using a consumer-grade Vulkan GPU, or ~20ms CPU time (i.e. further parallelizable) when using an AVX512 CPU, with a statistically negligible false negative rate on top of a brute-force forensic-grade pipeline.

## The Art of Compromise

> Key Design Philosophy TLDR: We chose to "approximate" the hashing stage rather than "approximate" the matching stage because:
> - Matching has a clear ground truth - compromises directly translate to missed matches
> - Hashing is feature extraction - small variations don't necessarily mean worse results
> - Our optimizations maintain statistical compatibility while improving performance

In an ideal situation in the realm of illegal image matching, one would use an exact PDQ implementation and an exact matching solution all the time, however, current off-the-shelf PDQ pipelines frequently need up to 3.5ms to generate a hash (and many libraries do not provide easy access to dihedrally derived hashes), and a naively coded exact matching solution can easily take 40ms for matching 8 derived hashes on 10M vectors. This is way out of most production environment can accept in terms of per-image latency and CPU time (throughput).

One may think about the conventional wisdom "always use ANN for large scale similarity search" and this is true as long as one is okay with significant guaranteed false negatives. My experiment using Facebook's Faiss BinaryHNSW (a state-of-the-art ANN library) on the real NCMEC CSAM PDQ database shows 90% recall with nearing 10ms per query single-threaded using very high hyperparameters (350 ef-constructions, M=48 and an ef-search of 420), this means _10% of images that should have matched will be missed_.

Instead, I decided to "cut corners" on the hashing stage instead by tuning the algorithm to be more suitable for modern CPU and vectorization in exchange of 2% images of DISC21 image data set will hash to more than 10 bits different (the official standard for evaluating the a faithfully exact replica of the algorithm), there is an important reason for why "cutting corners" in the hashing stage is a much better deal than in the matching stage:

- In the matching stage, we have a ground truth (for any particular hash, there is a single "correct" answer), if a hypothetical ANN has 90% recall, that means in reality 10% of the images that should have matched will be missed.
- In the hashing stage, we do not have a ground truth (if one shows two images, there is no way to everybody can agree they _should_ have matched or _should not_ have matched), it is a feature extraction problem.
  
  Think about it as two students take an exam: for a particular answer key (the hash in the database), if student A gets 98% of the questions correct (a faithful PDQ implementation), and student B has 2% of the questions answered differently than student A, what is the conclusion? It's not that student B gets 96% of the questions correct, but student B get at least 96% of the questions correct, this gives us a much more lenient environment to work with. The 2 images in DISC21 that has a distance of up to 24, even if we consider them "not matchable" (which may or may not be true because the threshold is 31), they do not directly contribute to my variant being 2% less recall than if one has used a faithful PDQ implementation: student A might be wrong for these 2 images as well.


## PDQ Variant Implementation

Important: You _should NOT_ submit hashes to databases using output of any optimized (variant) kernels in this library, they are designed to be statistically compatible for matching but absolutely not for submission to a database.

Our PDQ implementation makes some intentional deviations from the reference implementation to optimize for modern hardware while maintaining practical accuracy comparable to using a faithful PDQ implementation:

1. Increased DCT dimension (127x127 vs 64x64) to:
   - Reduce compression loss from 512x512 input
   - Better utilize modern CPU architectures and vectorization, allowing for a single hash to be computed in <50us on an AVX2 CPU
   - In theory should capture more frequency domain information to make the DCT-II transform more stable against minor transformations
   - When both performance and absolute perfection are important, one can configure the variant PDQ to be used as a hash filter with an increased matching threshold, and perform a more precise comparison if the image was flagged as a "potential match".

This is a more detailed explanation of my reasoning for the above design choices:

Our definition of "accurate enough to match" was based on a worst-case FNR (false negative rate) computed using birthday paradox, assuming such as statistical model:

  - We have an unknown "ground truth" in the DCT transformation that perfectly captures the original image (the "truth" that stays consistent across minor transformations PDQ was designed to detect), and the non-existence of such "ground truth" is the reason no perceptual hash algorithm can possibly provide a numerical guarantee of "recall" through empirical testing.
  - We have an official definition of "positive" images generated using a lossy DCT, to match with our, also lossy DCT.
  - An image is a "positive" if the number of bit flips between the two lossy DCTs is less than 31 bits (per official definition).
  - We can assume a worst-case scenario that results in a birthday paradox scenario, where:
      1. Bits are all i.i.d. uniform variables (i.e. no parts of an hash are more malleable than others, in reality some parts of hash are much more likely to flip than others)
      2. All bit-flips happen to make the two lossy implementations diverge (in reality this this is also subject to probability, but we assume the worst-case)
      3. The reference implementation is perfect in that it is stable in itself against minor transformations.

  This yields `(1 / 2^((31 - $worst_distance) / 2)) * 100% / $test_set_size`, which currently computes to 0.069% upper bound FNR (24 bits) for the 100 images in DISC21 test set ([logs](fnr-test/log.txt)) (by assuming all images are as "bad" as the 2 out of 100 images that yielded this difference), and 0.00076% when using a more optimistic average distance (11 bits) to estimate this. It should also be noted that PDQ is a perceptual hash and can be malleable to imperceptible transformations (such as minor warping), thus an on-average 10-bits off implementation does not translate to 1024 times higher FNR in real-world fuzzy matching.

  - The main hypothesized source of difference than a faithful implementation is because of a changed size of input dimension for DCT2D transformation (we increased to 127x127 from the official 64x64 to make the loss from the compression from 512x512 lower and adapt to modern CPU architecture), which could potentially capture _more_ input information to compress into frequency domain (as DCT2D has ~4x more pixels to "sample" frequency domain information from) leading to a different numerical result, but is actually more stable on such malleable images with very sharp edges. If this hypothesis holds, the real-world FNR is likely orders-of-magnitude lower than the naive birthday paradox estimate above.

  - To further reduce possibility of errors when both performance and accuracy are important, it is recommended to use yume-pdq in a "hash filter"-like configuration (i.e. increasing the threshold for a "potential match" by another 8 bits to (to 39 bits), and perform a more precise comparison if the image was flagged as a "potential match").

  - One might think "24-bit" worst-case difference is a big error, however it should also be noted that apart from statistical error rates, perceptual hash have systematic errors that are likely to cause inherent Type-I and Type-II errors that are not associated with minor variations in hyperparameter tuning. These unstable bit positions are malleable in both "reference" implementations and yume-pdq, for example cropping an image by an imperceptible amount can cause [~18 bit difference](https://github.com/darwinium-com/pdqhash/blob/1a5f66f635758ee441d8884e1c15203a2ace995d/README.md#offering-similarity-resilience) using identical, reference-level officially endorsed implementations, and our "errors" are likely the same source (boundary effects), thus likely do not accumulate.

## Matching Strategy: Why not use ANN?

We deliberately chose exact linear scan over approximate nearest neighbor (ANN) solutions:

1. Performance Comparison (all are numbers for single-threaded search of 8 hashes at one time):
   - Our GPU implementation on RTX 4070: ~0.79ms for 10M vectors (~12.6G vectors/sec). AVX512 implementation: ~20ms for 10M vectors (~500M vectors/sec). All are 100% recall.
   - Facebook's Faiss BinaryHNSW: ~10ms per query when tuned for 90% recall.

2. Testability:
    - It is very difficult to definitively say one implemented a feature extraction algorithm completely correctly, because there is no ground truth. But an exact thresholding solution can easily be tested against a large batch of edge-cases.

3. Quality Guarantees:
   - Linear scan provides exact matches
   - ANN introduces guaranteed false negatives that will directly translate to a false negative on an image that should have matched.
   - At ~10M scale, brute force is both faster and more accurate

4. Implementation Benefits:
   - Simple, predictable behavior
   - Highly parallelizable at many levels (ILP, chunking, task-level parallelism) on modern hardware
   - Trivially implementable in accelerated hardware (SIMD, GPU, FPGA, etc.) with minor recurrent memory transfer overhead
   - No complex index structures needed, no memory overhead other than the ~300MB data itself.

## Matching Strategy: What about metric-tree based solutions?

We decided to not go for that route either because it is unlikely to produce a faster solution:

1. **Uniqueness of PDQ hash characteristics**: PDQ hashes end with a binary quantization at median, that means the overwhelming majority of all hashes except the most edge-case ones have a hamming weight of exactly 128. This completely eliminated any possibility of elimination of candidates purely by hamming distance.

2. **Relative High Dimensionality and Large Radii**: Metric tree based solutions such as BK-tree and KD-tree all exploit the fact that hamming distance has triangular inequality, that is, for any three points, the distance between the farthest two points is no less than the sum of the other two distances. This allows one to prune the search space significantly. However, in PDQ screening, the vast majority of hashes, and the hashes compared to the needle would be very close to 128, and coupled with a large radius (officially recommended 31 bits and some clients certainly would like to flag with a higher threshold), this means all metric tree must visit at least a "range" of 31*2 = 62 bits to ensure a exhaustive match. This is very unfriendly and likely to cause a search close to a full linear scan.

3. **Unique Batching Characteristic**: PDQ is not dihedral invariant, thus for each image all 8 hashes must be compared against the database to ensure a correct match. This naturally penalizes the use of all data-dependent graph/tree traversal based solutions, as they would be forced to run 8 independent searches while a linear scan can match all 8 at once, furthermore with the support of vectorized instructions, 8 comparisons can be done independently in the 8 64-bit lanes of an AVX512 CPU eliminating the need for any horizontal reductions (and the reason why `yume-pdq` is faster than off-the-shelf exhaustive scan solutions like Facebook(R) Faiss IndexBinaryFlat).

## Safety Considerations

Hand-written SIMD is unsafe and we take several precautions:
- No data-dependent jumps or indexing
- Debug-time assertions on pointer validity
- Keep the core kernel simple and low-dependency to ensure forward and backward-edge CFI compatibility and maximum protection.