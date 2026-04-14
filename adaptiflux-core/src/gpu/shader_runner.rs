// Copyright (C) 2026 Jocer S. <patcex@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: AGPL-3.0 OR Commercial

use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupLayout, ComputePipeline, Device, PipelineCompilationOptions,
    PipelineLayoutDescriptor, Queue, ShaderModuleDescriptor, ShaderSource,
};

pub struct ShaderRunner {
    pipeline: ComputePipeline,
    _bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl ShaderRunner {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        shader_code: &str,
        entry_point: &str,
        bind_group_layout: BindGroupLayout,
        bind_group: BindGroup,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: ShaderSource::Wgsl(shader_code.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point,
            compilation_options: PipelineCompilationOptions::default(),
        });

        Ok(Self {
            pipeline,
            _bind_group_layout: bind_group_layout,
            bind_group,
            device,
            queue,
        })
    }

    pub async fn run(&self, x: u32, y: u32, z: u32) -> Result<(), Box<dyn std::error::Error>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            compute_pass.dispatch_workgroups(x, y, z);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        Ok(())
    }
}
