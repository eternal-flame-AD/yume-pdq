#!/usr/bin/env python3
'''
Copyright (c) 2025 Yumechi <yume@yumechi.jp>

Created on Thursday, March 27, 2025
Author: Yumechi <yume@yumechi.jp>

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from ctypes import CDLL, c_float, c_uint8, POINTER, byref, sizeof, memmove
import numpy as np
from PIL import Image
import sys
import os
import time

image_path = sys.argv[1]

DYLIB_LOCATION = os.environ.get("LIBYUME_PDQ_PATH", "target/release/libyume_pdq.so")

class PDQHasher:
    # 
    def __init__(self, lib_path=DYLIB_LOCATION):
        # Load the shared library
        self.lib = CDLL(lib_path)
        
        # Configure the hash_smart_kernel function
        self.hash_smart_kernel = self.lib.yume_pdq_hash_smart_kernel
        self.hash_smart_kernel.restype = c_float
        self.hash_smart_kernel.argtypes = [
            POINTER(c_float),    # input
            POINTER(c_float),    # threshold
            POINTER(c_uint8),    # output
            POINTER(c_float),    # buf1
            POINTER(c_float),    # tmp
            POINTER(c_float),    # pdqf
        ]

        # allocate buffers

        self.input = (c_float * (512 * 512))()
        self.buf1 = (c_float * (128 * 128))()
        self.tmp = (c_float * (128 * 1))()
        self.pdqf = (c_float * (16 * 16))()
        self.output = (c_uint8 * 32)()

    def hash_image(self, image_path):
        start_time = time.time()

        # Load and preprocess image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((512, 512), Image.Resampling.BILINEAR)
        
        # Convert to numpy array and then to float32
        img_array = np.array(img, dtype=np.float32)

        conversion_done = time.time()
        
        threshold = c_float(0.0)

        memmove(self.input, img_array.ctypes.data_as(POINTER(c_float)), sizeof(self.input))

        # Call the hash function
        quality = self.hash_smart_kernel(
            self.input,
            byref(threshold),
            self.output,
            self.buf1,
            self.tmp,
            self.pdqf
        )

        output_cast_8 = np.frombuffer(self.output, dtype=np.uint8)

        hash_hex = output_cast_8.tobytes().hex()

        hash_done = time.time()

        return {
            'quality': quality,
            'threshold': threshold.value,
            'hash': hash_hex if quality > 0.5 else None,
            'hash_time': hash_done - conversion_done,
            'conversion_time': conversion_done - start_time,
        }

def main():
    hasher = PDQHasher()
    
    result = hasher.hash_image(image_path)
    
    print(f"Image: {image_path}")
    print(f"Quality: {result['quality']:.3f}")
    print(f"Threshold: {result['threshold']:.3f}")
    print(f"Hash: {result['hash']}")
    

    expected_hash = result['hash']

    print(f"Starting 1000 iterations")
    start_time = time.time()

    total_time_conversion = 0
    total_time_hash = 0
    for i in range(1000):
        result = hasher.hash_image(image_path)
        assert result['hash'] == expected_hash, f"Hash mismatch at iteration {i} {result['hash']} != {expected_hash} quality: {result['quality']}"
        total_time_conversion += result['conversion_time']
        total_time_hash += result['hash_time']

    end_time = time.time()
    print(f"Finished 1000 iterations")
    print(f"Average Time taken: {(end_time - start_time) / 1000 * 1_000_000:.3f} us")
    print(f"Average conversion time: {total_time_conversion / 1000 * 1_000_000:.3f} us")
    print(f"Average hash time: {total_time_hash / 1000 * 1_000_000:.3f} us")

if __name__ == "__main__":
    main()



