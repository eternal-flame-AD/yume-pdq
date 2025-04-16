/*
 * Copyright (c) 2025 Yumechi <yume@yumechi.jp>
 *
 * Created on Monday, April 14, 2025
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

use std::num::NonZero;

use generic_array::{
    ArrayLength, GenericArray,
    sequence::Flatten,
    typenum::{B0, B1, U8, U32},
};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

/// Re-export wgpu for use in the crate.
pub use wgpu;

use crate::kernel::type_traits::{EvaluateHardwareFeature, kernel_sealing::KernelSealed};

use super::PDQMatcher;

#[derive(Debug)]
/// A vector database already uploaded to the accelerator.
pub struct VulkanVectorDatabase<Dim: ArrayLength> {
    num_vectors: usize,
    data: wgpu::Buffer,
    _marker: std::marker::PhantomData<Dim>,
}

impl<Dim: ArrayLength> VulkanVectorDatabase<Dim> {
    /// Create a new vector data from a slice of vectors.
    ///
    /// Your device must have [`wgpu::Limits::max_storage_buffer_binding_size`] large enough to hold the data, you may need to manually raise it at creation time.
    pub fn new(device: &wgpu::Device, label: Option<&str>, data: &[GenericArray<u8, Dim>]) -> Self {
        let init = BufferInitDescriptor {
            label,
            contents: unsafe {
                core::slice::from_raw_parts::<u8>(
                    data.as_ptr().cast(),
                    data.len() * core::mem::size_of::<GenericArray<u8, Dim>>(),
                )
            },
            usage: wgpu::BufferUsages::STORAGE,
        };

        Self {
            num_vectors: data.len(),
            data: device.create_buffer_init(&init),
            _marker: std::marker::PhantomData,
        }
    }
}

/// A Vulkan matcher with a dynamically generated shader that does straight-forward popcnt, sum then compare.
///
/// This is extremely fast (12+G vectors per second when matching 8 needles on an RTX 4070).
pub struct VulkanMatcher<'b, NumNeedles: ArrayLength, Length: ArrayLength> {
    device: wgpu::Device,
    queue: wgpu::Queue,
    haystack: &'b VulkanVectorDatabase<Length>,
    needle_buffer: wgpu::Buffer,
    output_clear_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    output_download_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    _marker: std::marker::PhantomData<NumNeedles>,
}

impl<'b, NumNeedles: ArrayLength> VulkanMatcher<'b, NumNeedles, U32> {
    /// Create a new Vulkan kernel.
    ///
    /// Your device must have [`wgpu::Features::MAPPABLE_PRIMARY_BUFFERS`] enabled.
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        haystack: &'b VulkanVectorDatabase<U32>,
        threshold: u32,
    ) -> Self {
        let mut shader_code = format!(
            r#"
        @group(0) @binding(0)
        var<storage, read> haystack: array<vec4<u32>>;
        
        @group(0) @binding(1)
        var<uniform> needles: array<vec4<u32>, 8 * 2>;
        
        @group(0) @binding(2)
        var<storage, read_write> matches: atomic<u32>;
        
        override threshold: u32 = {threshold};
        
        @compute @workgroup_size(256, 1, 1)
        fn findThreshold(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let linear_idx = global_id.x;
        
            let haystack_length = arrayLength(&haystack) / 2;
        
            if (linear_idx >= haystack_length) {{
                return;
            }}
            var distance: vec4<u32> = vec4<u32>(0);
        "#,
        );

        for needle_idx in 0..NumNeedles::USIZE {
            use core::fmt::Write;
            write!(
                shader_code,
                r#"
                {}
                distance += countOneBits(haystack[linear_idx * 2 + 0] ^ needles[{needle_idx} * 2 + 0]);
                distance += countOneBits(haystack[linear_idx * 2 + 1] ^ needles[{needle_idx} * 2 + 1]);
        
                if (distance[0] + distance[1] + distance[2] + distance[3] <= threshold) {{
                    let needle_idx: u32 = {needle_idx} + linear_idx * {};
                    atomicMin(&matches, needle_idx);
                }}
            "#, if needle_idx == 0 {
                ""
            } else {
                "distance[0] = 0; distance[1] = 0; distance[2] = 0; distance[3] = 0;"
            }, NumNeedles::USIZE
            )
            .expect("Failed to write shader code");
        }
        shader_code.push('}');

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Thresholding Kernel"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_code)),
        });

        let needle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Needle Buffer"),
            size: ((32 * NumNeedles::USIZE) as u64).max(4096),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        let output_clear_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &[!00u8; 4],
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: 4096,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Download Buffer"),
            size: 4096,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZero::new(32 * haystack.num_vectors)
                                .unwrap()
                                .try_into()
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            NonZero::new(32 * NumNeedles::USIZE)
                                .unwrap()
                                .try_into()
                                .unwrap(),
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZero::new(4).unwrap()),
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: haystack.data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: needle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vulkan Thresholding Kernel"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("findThreshold"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: false,
            },
        });

        Self {
            device,
            queue,
            haystack,
            output_buffer,
            output_clear_buffer,
            output_download_buffer,
            bind_group,
            pipeline,
            needle_buffer,
            _marker: std::marker::PhantomData,
        }
    }
}

/// A Vulkan SPIRV compute shader accelerator requirement.
pub struct VulkanAccelerator;

impl KernelSealed for VulkanAccelerator {}

impl EvaluateHardwareFeature for VulkanAccelerator {
    type Name = &'static str;

    type EnabledStatic = B1;

    type MustCheck = B1;

    fn name() -> Self::Name {
        "Vulkan SPIRV"
    }

    fn met_runtime() -> bool {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();

        let mut features = wgpu::Features::empty();
        features.insert(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);

        let limits = wgpu::Limits::default();

        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .is_ok()
    }
}

impl<'b, NumNeedles: ArrayLength> PDQMatcher for VulkanMatcher<'b, NumNeedles, U32> {
    type BatchSize = U8;
    type RequiredHardwareFeature = VulkanAccelerator;
    type InputDimension = U32;
    type Aligner<T> = T;
    type NoFalseNegative = B1;
    type NoFalsePositive = B1;
    type FindsAllMatches = B0;
    type Ident = &'static str;

    fn ident() -> Self::Ident {
        "Vulkan Accelerated Matcher"
    }

    fn scan(
        &mut self,
        query: &GenericArray<GenericArray<u8, Self::InputDimension>, Self::BatchSize>,
    ) -> bool {
        self.needle_buffer
            .map_async(wgpu::MapMode::Write, 0..(8 * 32), |_| {});
        self.device
            .poll(wgpu::PollType::Wait)
            .expect("failed to wait for device");
        self.needle_buffer
            .get_mapped_range_mut(0..(8 * 32))
            .copy_from_slice(query.flatten().as_slice());
        self.needle_buffer.unmap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Vulkan Kernel Encoder"),
            });

        encoder.copy_buffer_to_buffer(&self.output_clear_buffer, 0, &self.output_buffer, 0, 4);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(
                self.haystack.num_vectors.div_ceil(256).try_into().unwrap(),
                1,
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.output_download_buffer, 0, 4);

        let cmd = encoder.finish();
        let wait = self.queue.submit([cmd]);

        self.output_download_buffer
            .map_async(wgpu::MapMode::Read, 0..4, |_| {});

        self.device
            .poll(wgpu::PollType::WaitForSubmissionIndex(wait))
            .expect("Failed to poll device");

        let ret = self
            .output_download_buffer
            .get_mapped_range(0..4)
            .iter()
            .any(|&x| x != !0);

        self.output_download_buffer.unmap();

        ret
    }

    fn find<R, F: FnMut(usize, usize) -> Option<R>>(
        &mut self,
        query: &GenericArray<GenericArray<u8, Self::InputDimension>, Self::BatchSize>,
        mut f: F,
    ) -> Option<R> {
        self.needle_buffer
            .map_async(wgpu::MapMode::Write, 0..(8 * 32), |_| {});
        self.device
            .poll(wgpu::PollType::Wait)
            .expect("failed to wait for device");
        self.needle_buffer
            .get_mapped_range_mut(0..(8 * 32))
            .copy_from_slice(query.flatten().as_slice());
        self.needle_buffer.unmap();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Vulkan Kernel Encoder"),
            });

        encoder.copy_buffer_to_buffer(&self.output_clear_buffer, 0, &self.output_buffer, 0, 4);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(
            self.haystack.num_vectors.div_ceil(256).try_into().unwrap(),
            1,
            1,
        );
        drop(pass);
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.output_download_buffer, 0, 4);

        let cmd = encoder.finish();
        let wait = self.queue.submit([cmd]);

        self.output_download_buffer
            .map_async(wgpu::MapMode::Read, 0..4, |_| {});

        self.device
            .poll(wgpu::PollType::WaitForSubmissionIndex(wait))
            .expect("Failed to poll device");

        let ret = {
            let data = self.output_download_buffer.get_mapped_range(0..4);
            u32::from_le_bytes([data[0], data[1], data[2], data[3]])
        };

        self.output_download_buffer.unmap();

        if ret != !0 {
            f(
                ret as usize / NumNeedles::USIZE,
                ret as usize % NumNeedles::USIZE,
            )
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use generic_array::typenum::{U32, U2048};
    use rand::prelude::*;

    use super::*;

    #[test]
    fn test_scan_vulkan() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();

        let mut features = wgpu::Features::empty();
        features.insert(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        }))
        .unwrap();

        let mut rng = rand::rng();

        // Generate random needles
        let mut needles_data = GenericArray::<GenericArray<u8, U32>, U8>::default();
        for needle in needles_data.iter_mut() {
            rng.fill_bytes(needle);
        }

        // Generate random batch data
        let mut haystack_data = GenericArray::<GenericArray<u8, U32>, U2048>::default();
        for vector in haystack_data.iter_mut() {
            rng.fill_bytes(vector);
        }

        let buffer = VulkanVectorDatabase::<U32>::new(&device, Some("test"), &haystack_data);

        let mut kernel = VulkanMatcher::<U8, U32>::new(device, queue, &buffer, 31);

        fn generate_positive_control<L: ArrayLength, R: RngCore>(
            rng: &mut R,
            reference: &GenericArray<u8, L>,
            output: &mut GenericArray<u8, L>,
            distance: u32,
        ) {
            for (i, needle) in output.iter_mut().enumerate() {
                *needle = reference[i];
            }

            for _ in 0..distance {
                let position_byte = rng.random_range(0..L::USIZE);
                let position_bit = rng.random_range(0..8);
                output[position_byte] ^= 1 << position_bit;
            }
        }

        // Compare results, initially there are no matches
        assert_eq!(kernel.scan(&needles_data), false);

        kernel.find(&needles_data, |_, i| {
            panic!("Found match at index {}", i);
            #[allow(unreachable_code)]
            Some(i)
        });

        // Insert a few matches to ensure we're testing the matching logic
        // Copy a few needles into the batch at random positions
        // Try 1000x8 times just to make sure we didn't have a blindspot or something
        for _test in 0..1000 {
            for i in 0..8 {
                let pos = rng.random_range(0..2048);
                generate_positive_control(&mut rng, &haystack_data[pos], &mut needles_data[i], 31);

                // Compare results again
                assert_eq!(kernel.scan(&needles_data), true);

                let result = kernel.find(&needles_data, |j, i| {
                    assert_eq!(pos as usize, j);
                    Some(i)
                });

                assert_eq!(result, Some(i));

                // reset the needle
                rng.fill_bytes(&mut needles_data[i]);
            }
        }

        // Compare results, make sure finally there are no matches
        assert_eq!(kernel.scan(&needles_data), false);

        kernel.find(&needles_data, |_, i| {
            panic!("Found match at index {}", i);
            #[allow(unreachable_code)]
            Some(i)
        });
    }
}
