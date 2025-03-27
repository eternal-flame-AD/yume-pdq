/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Thursday, March 27, 2025
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// gcc -o dummy dummy.c -L. -lyume_pdq

// Function prototype matching our Rust FFI
float yume_pdq_hash_smart_kernel(
    const float *input,
    float *threshold,
    unsigned char *output,
    float *buf1,
    float *tmp,
    float *pdqf);

int main()
{
    // Allocate all buffers
    float *input = (float *)malloc(512 * 512 * sizeof(float));
    float threshold = 0.0;
    unsigned char output[32] = {0}; // 2x16 array for hash
    float *buf1 = (float *)malloc(128 * 128 * sizeof(float));
    float *tmp = (float *)malloc(128 * sizeof(float));
    float *pdqf = (float *)malloc(16 * 16 * sizeof(float));

    // Fill input with dummy data
    for (int i = 0; i < 512 * 512; i++)
    {
        input[i] = (float)i;
    }

    // First hash
    float quality1 = yume_pdq_hash_smart_kernel(input, &threshold, output, buf1, tmp, pdqf);
    printf("First hash:\n");
    printf("Quality: %f\n", quality1);
    printf("Threshold: %f\n", threshold);
    printf("Hash: ");
    for (int i = 0; i < 32; i++)
    {
        printf("%02x", output[i]);
    }
    printf("\n\n");

    // Modify input slightly
    for (int i = 0; i < 512 * 512; i++)
    {
        input[i] = (float)(i + 1);
    }

    // Second hash
    float quality2 = yume_pdq_hash_smart_kernel(input, &threshold, output, buf1, tmp, pdqf);
    printf("Second hash:\n");
    printf("Quality: %f\n", quality2);
    printf("Threshold: %f\n", threshold);
    printf("Hash: ");
    for (int i = 0; i < 32; i++)
    {
        printf("%02x", output[i]);
    }
    printf("\n");

    // Cleanup
    free(input);
    free(buf1);
    free(pdqf);

    return 0;
}