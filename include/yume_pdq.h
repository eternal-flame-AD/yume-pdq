/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Saturday, March 22, 2025
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

extern const uint32_t YUME_PDQ_VERSION_MAJOR;
extern const uint32_t YUME_PDQ_VERSION_MINOR;
extern const uint32_t YUME_PDQ_VERSION_PATCH;
extern const char YUME_PDQ_VERSION_STRING[];
 

/* Generated with cbindgen:0.28.0 */

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>


/**
 * A callback function for visiting all dihedrals.
 *
 * The threshold, PDQF and quantized output will be available to the caller via the provided buffers ONLY before the callback returns.
 *
 * Return true to continue, false to stop.
 *
 * The function must not modify the buffers, and must copy them out before returning if they need to keep them.
 */
typedef bool (*DihedralCallback)(void *ctx, uint32_t dihedral, float threshold, float quality);

/**
 * A packed representation of a matrix for dihedral transformations.
 */
typedef struct YumePDQ_Dihedrals {
  /**
   * The packed representation of the dihedral matrix.
   *
   * Ordering is first x-to-x, then x-to-y, then y-to-x, then y-to-y. Big-endian signed 8-bit integers packed into a u32.
   */
  uint32_t packed;
} YumePDQ_Dihedrals;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * re-exported constants for the dihedrals
 */
extern const struct YumePDQ_Dihedrals YUME_PDQ_DIHEDRAL_FLIPPED;

/**
 * re-exported constants for the dihedrals
 */
extern const struct YumePDQ_Dihedrals YUME_PDQ_DIHEDRAL_FLOPPED;

/**
 * re-exported constants for the dihedrals
 */
extern const struct YumePDQ_Dihedrals YUME_PDQ_DIHEDRAL_FLOPPED_ROTATED_270;

/**
 * re-exported constants for the dihedrals
 */
extern const struct YumePDQ_Dihedrals YUME_PDQ_DIHEDRAL_NORMAL;

/**
 * re-exported constants for the dihedrals
 */
extern const struct YumePDQ_Dihedrals YUME_PDQ_DIHEDRAL_ROTATED_180;

/**
 * re-exported constants for the dihedrals
 */
extern const struct YumePDQ_Dihedrals YUME_PDQ_DIHEDRAL_ROTATED_270;

/**
 * re-exported constants for the dihedrals
 */
extern const struct YumePDQ_Dihedrals YUME_PDQ_DIHEDRAL_ROTATED_90;

/**
 * re-exported constants for the dihedrals
 */
extern const struct YumePDQ_Dihedrals YUME_PDQ_DIHEDRAL_ROTATED_90_FLOPPED;

/**
 * Compute the PDQ hash of a 512x512 single-channel image using [`kernel::smart_kernel`].
 *
 * # Safety
 *
 * - `input` is in only, must be a pointer to a 512x512 single-channel image in float32 format, row-major order.
 * - `threshold` is out only, must be a valid aligned pointer to a f32 value or NULL.
 * - `output` is out only, must be a pointer to a 2x16 array of u8 to receive the final 256-bit hash.
 * - `buf1` is in/out, must be a pointer to a 128x128 array of f32 values to receive the intermediate results of the DCT transform.
 * - `tmp` is in/out, must be a pointer to a 128x1 array of f32 values as scratch space for the DCT transform.
 * - `pdqf` is out only, must be a pointer to a 16x16 array of f32 values to receive PDQF (unquantized) hash values.
 *
 * # Returns
 *
 * The quality of the hash as a f32 value between 0.0 and 1.0. You are responsible for checking whether quality is acceptable.
 */
float yume_pdq_hash_smart_kernel(const float *input,
                                 float *threshold,
                                 uint8_t *output,
                                 float *buf1,
                                 float *tmp,
                                 float *pdqf);

/**
 * Visit the 7 alternative dihedrals of the PDQF hash.
 *
 * # Safety
 *
 * - `ctx` is transparently passed to the callback function.
 * - `threshold` must be a valid threshold value for the provided PDQF input received from [`hash_smart_kernel`].
 * - `output` is out only, must be a pointer to a 2x16 array of u8 to receive any intermediate 256-bit hash. It does not have to be initialized to any particular value.
 * - `pdqf` is in/out, must be a pointer to a 16x16 array of f32 values of the initial PDQF data, and be writable to receive derived PDQF (unquantized) hash values.
 * - `callback` must be a valid callback function that will be called for each dihedral.
 *
 * # Returns
 *
 * - `true` if all dihedrals were visited, `false` if the callback returned false for any dihedral.
 */
bool yume_pdq_visit_dihedrals_smart_kernel(void *ctx,
                                           float threshold,
                                           uint8_t *output,
                                           float *pdqf,
                                           DihedralCallback callback);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
