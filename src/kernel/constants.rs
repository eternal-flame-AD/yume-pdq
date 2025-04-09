/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Wednesday, April 9, 2025
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

/// Lookup table for converting RGB8 to LUMA8 using ITU-R BT.601.
pub const RGB8_TO_LUMA8_TABLE_ITU: [f32; 3] = [0.299, 0.587, 0.114];
