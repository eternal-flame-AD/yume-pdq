/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Wednesday, April 16, 2025
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

#include <cstdint>
#include <pdq/cpp/hashing/pdqhashing.h>
using namespace facebook::pdq::hashing;

extern "C"
{
    int yumepdq_official_512x512_hash_adapator(
        // facebook impl didn't add const for some reason, so it already eliminated optimization possibilities we can only follow suit
        float *image_rgba_in,
        float *tmp_512x512,
        uint16_t hash_all_dihedrals[8][16])
    {
        float buffer64x64[64][64];
        float buffer16x64[16][64];
        float output_buffer16x16[16][16];
        float buffer_aux[16][16];
        int quality;
        Hash256 hashes[8];
        pdqDihedralHash256esFromFloatLuma(
            image_rgba_in,
            tmp_512x512,
            512,
            512,
            buffer64x64,
            buffer16x64,
            output_buffer16x16,
            buffer_aux,
            &hashes[0],
            &hashes[1],
            &hashes[2],
            &hashes[3],
            &hashes[4],
            &hashes[5],
            &hashes[6],
            &hashes[7],
            quality);
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                const Hash16 hash_word = hashes[i].w[j];
                hash_all_dihedrals[i][j] = hash_word;
            }
        }
        return quality;
    }
}