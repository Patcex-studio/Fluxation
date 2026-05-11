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
// SPDX-License-Identifier: AGPL-3.0OR Commercial

//! GPU compute backend for batched Izhikevich neuron updates (PERF-003 GPU acceleration)
//!
//! Offloads spike computation for batches of neurons to GPU compute shaders,
//! providing 10-100x speedup for large batches (>1000 neurons) vs scalar CPU.

use crate::utils::types::StateValue;
use std::sync::Arc;
use tracing::debug;
use wgpu::{
    Buffer, BindGroupLayout, ComputePipeline, Device, Queue,
};

/// WGSL compute shader for batched Izhikevich updates
pub const IZHIKEVICH_COMPUTE_SHADER: &str = r#"
// Izhikevich neuron parameters and state
struct NeuronState {
    v: f32,              // Membrane potential (mV)
    u: f32,              // Recovery variable
    spike: u32,          // Spike flag this timestep (0 or 1)
    pad: u32,            // Padding for alignment
}

struct NeuronParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    dt: f32,
    threshold: f32,
    input_current: f32,
    pad: u32,           // Padding for alignment
}

@group(0) @binding(0)
var<storage, read_write> neuron_states: array<NeuronState>;

@group(0) @binding(1)
var<storage, read> neuron_params: array<NeuronParams>;

@group(0) @binding(2)
var<uniform> num_neurons: u32;

@compute @workgroup_size(256)
fn update_izhikevich(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= num_neurons {
        return;
    }

    var state = neuron_states[idx];
    let params = neuron_params[idx];

    // Izhikevich model:
    // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    // du/dt = a*(b*v - u)
    let dv = 0.04 * state.v * state.v + 5.0 * state.v + 140.0 - state.u + params.input_current;
    let du = params.a * (params.b * state.v - state.u);

    // Euler step
    state.v = state.v + params.dt * dv;
    state.u = state.u + params.dt * du;

    // Check spike threshold (default 30 mV)
    if state.v >= params.threshold {
        state.spike = 1u;
        state.v = params.c;        // Reset potential
        state.u = state.u + params.d;  // Increase recovery
    } else {
        state.spike = 0u;
    }

    neuron_states[idx] = state;
}
"#;

/// GPU compute context for Izhikevich batches
pub struct IzhikevichGpuCompute {
    device: Arc<Device>,
    queue: Arc<Queue>,
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
}

impl IzhikevichGpuCompute {
    /// Create GPU compute pipeline for Izhikevich
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("izhikevich_compute_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(IZHIKEVICH_COMPUTE_SHADER)),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("izhikevich_bind_group_layout"),
                entries: &[
                    // Binding 0: neuron_states (read-write storage buffer)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: false,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 1: neuron_params (read-only storage buffer)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: true,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Binding 2: num_neurons (uniform)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("izhikevich_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("izhikevich_compute_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "update_izhikevich",
                compilation_options: Default::default(),
            });

        Ok(Self {
            device,
            queue,
            compute_pipeline,
            bind_group_layout,
        })
    }

    /// Execute Izhikevich updates on a batch of neurons
    pub async fn compute_batch(
        &self,
        state_buffer: &Buffer,
        params_buffer: &Buffer,
        num_uniform_buffer: &Buffer,
        num_neurons: usize,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("izhikevich_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: num_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("izhikevich_encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("izhikevich_compute_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with workgroup size of 256
            let workgroup_count = (num_neurons as u32 + 255) / 256;
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        debug!("Izhikevich GPU compute dispatched for {} neurons", num_neurons);

        Ok(())
    }
}

/// Wrapper for GPU-accelerated Izhikevich batch state
#[derive(Debug, Clone)]
pub struct GpuIzhikevichBatch {
    pub v: Vec<StateValue>,
    pub u: Vec<StateValue>,
    pub spikes: Vec<bool>,
    pub neuron_count: usize,
}

impl GpuIzhikevichBatch {
    pub fn new(neuron_count: usize) -> Self {
        Self {
            v: vec![-65.0; neuron_count],
            u: vec![0.0; neuron_count],
            spikes: vec![false; neuron_count],
            neuron_count,
        }
    }

    pub fn spike_count(&self) -> usize {
        self.spikes.iter().filter(|&&s| s).count()
    }

    pub fn get_v(&self) -> &[StateValue] {
        &self.v
    }

    pub fn get_u(&self) -> &[StateValue] {
        &self.u
    }

    pub fn did_spike(&self, idx: usize) -> bool {
        self.spikes.get(idx).copied().unwrap_or(false)
    }
}
