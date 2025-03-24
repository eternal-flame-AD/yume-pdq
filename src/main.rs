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
use yume_pdq::kernel::{self, Kernel};

fn main() {
    let mut data = [0.0; 4096];
    let mut output = [0.0; 256];
    println!(
        "DefaultKernel: {:?}",
        kernel::DefaultKernel.dct2d(&mut data, &mut output)
    );
    println!(
        "Avx2F32Kernel: {:?}",
        kernel::x86::Avx2F32Kernel.dct2d(&mut data, &mut output)
    );
}
