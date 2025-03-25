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

use yume_pdq::GenericArray;

fn main() {
    let mut input = Vec::<f32>::new();
    input.resize(512 * 512, 0.0);
    let mut output = [0u8; 32];
    let mut buf1 = GenericArray::default();
    let mut buf2 = GenericArray::default();

    println!(
        "Avx2F32Kernel: {:?}",
        yume_pdq::hash(
            &mut yume_pdq::kernel::x86::Avx2F32Kernel,
            input.as_slice().try_into().unwrap(),
            &mut output,
            &mut buf1,
            &mut buf2
        )
    );
    let mut buf1_2 = GenericArray::default();
    let mut buf2_2 = GenericArray::default();
    println!(
        "ReferenceKernel: {:?}",
        yume_pdq::hash(
            &mut yume_pdq::kernel::ReferenceKernel::<f32>::default(),
            input.as_slice().try_into().unwrap(),
            &mut output,
            &mut buf1_2,
            &mut buf2_2
        )
    );
}
