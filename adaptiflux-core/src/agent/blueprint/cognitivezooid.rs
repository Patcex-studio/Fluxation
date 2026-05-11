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

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;

use crate::agent::blueprint::base::AgentBlueprint;
use crate::agent::synapse_manager::{SynapseManager, SynapseConfig, NormMode};
use crate::agent::state::{AgentUpdateResult, RoleType};
use crate::core::message_bus::message::Message;
use crate::core::system_config::SystemConfig;
use crate::core::topology::{TopologyChange, ZoooidTopology};
use crate::memory::types::MemoryPayload;
use crate::primitives::base::PrimitiveMessage;
use crate::primitives::spiking::{IzhikevichBatch, IzhikevichBatchParams, pack_input_currents};
use crate::primitives::spiking::izhikevich::{IzhikevichNeuron, IzhikevichParams, IzhikevichState};
use crate::utils::types::{StateValue, ZoooidId};

#[cfg(feature = "gpu")]
use crate::gpu::{GpuContext, IzhikevichGpuCompute};

#[cfg(feature = "gpu")]
async fn update_gpu_izhikevich_batch(
    state: &mut CognitivezooidState,
    primitive_inputs: &[PrimitiveMessage],
    tick_count: u64,
) -> Result<Vec<PrimitiveMessage>, Box<dyn std::error::Error + Send + Sync>> {
    use std::mem::size_of;
    use std::sync::Arc;
    use wgpu::util::DeviceExt;

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct NeuronState {
        v: f32,
        u: f32,
        spike: u32,
        pad: u32,
    }

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct NeuronParams {
        a: f32,
        b: f32,
        c: f32,
        d: f32,
        dt: f32,
        threshold: f32,
        input_current: f32,
        pad: u32,
    }

    let total_input_current: f32 = primitive_inputs
        .iter()
        .filter_map(|msg| match msg {
            PrimitiveMessage::InputCurrent(value) => Some(*value as f32),
            _ => None,
        })
        .sum();

    let num_neurons = state.neuron_count;
    let batch = state.gpu_batch.as_ref().unwrap();

    let state_data: Vec<NeuronState> = batch
        .v
        .iter()
        .zip(batch.u.iter())
        .map(|(&v, &u)| NeuronState {
            v: v as f32,
            u: u as f32,
            spike: 0,
            pad: 0,
        })
        .collect();

    let params_data: Vec<NeuronParams> = vec![NeuronParams {
        a: state.izh_params.a as f32,
        b: state.izh_params.b as f32,
        c: state.izh_params.c as f32,
        d: state.izh_params.d as f32,
        dt: state.izh_params.dt as f32,
        threshold: 30.0,
        input_current: total_input_current,
        pad: 0,
    }; num_neurons];

    let context = GpuContext::new().await?;
    let device = Arc::new(context.device);
    let queue = Arc::new(context.queue);
    let gpu_compute = IzhikevichGpuCompute::new(device.clone(), queue.clone())?;

    let state_bytes = unsafe {
        std::slice::from_raw_parts(
            state_data.as_ptr() as *const u8,
            state_data.len() * size_of::<NeuronState>(),
        )
    };
    let params_bytes = unsafe {
        std::slice::from_raw_parts(
            params_data.as_ptr() as *const u8,
            params_data.len() * size_of::<NeuronParams>(),
        )
    };

    let state_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("izhikevich_gpu_state_buffer"),
        contents: state_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("izhikevich_gpu_params_buffer"),
        contents: params_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let num_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("izhikevich_gpu_num_buffer"),
        contents: &(num_neurons as u32).to_ne_bytes(),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let result_size = (num_neurons * size_of::<NeuronState>()) as u64;
    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("izhikevich_gpu_readback_buffer"),
        size: result_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    gpu_compute
        .compute_batch(&state_buffer, &params_buffer, &num_buffer, num_neurons)
        .await?;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("izhikevich_gpu_copy_encoder"),
    });
    encoder.copy_buffer_to_buffer(&state_buffer, 0, &readback_buffer, 0, result_size);
    queue.submit(std::iter::once(encoder.finish()));

    use futures::channel::oneshot;

    let buffer_slice = readback_buffer.slice(..);
    let (sender, receiver) = oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver.await??;

    let data = buffer_slice.get_mapped_range();
    let received: &[NeuronState] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const NeuronState, num_neurons)
    };

    let batch_mut = state.gpu_batch.as_mut().unwrap();
    for (i, received_state) in received.iter().enumerate() {
        batch_mut.v[i] = received_state.v as StateValue;
        batch_mut.u[i] = received_state.u as StateValue;
        batch_mut.spikes[i] = received_state.spike != 0;
    }

    drop(data);
    readback_buffer.unmap();

    if num_neurons > 0 {
        state.izh_state.v = batch_mut.v[0];
        state.izh_state.u = batch_mut.u[0];
    }

    let mut outputs = Vec::new();
    if batch_mut.spike_count() > 0 {
        outputs.push(PrimitiveMessage::Spike {
            timestamp: tick_count,
            amplitude: 1.0,
        });
    }

    Ok(outputs)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivezooidParams {
    pub izh_params: IzhikevichParams,
    #[serde(default = "default_use_simd")]
    pub use_simd: bool,
    #[serde(default = "default_connection_request_interval")]
    pub connection_request_interval: u64,
    #[serde(default = "default_stdp_a_plus")]
    pub stdp_a_plus: f32,
    #[serde(default = "default_stdp_a_minus")]
    pub stdp_a_minus: f32,
    #[serde(default = "default_stdp_tau_plus")]
    pub stdp_tau_plus: f32,
    #[serde(default = "default_stdp_tau_minus")]
    pub stdp_tau_minus: f32,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f32,
    #[serde(default = "default_pruning_threshold")]
    pub pruning_threshold: f32,
    #[serde(default = "default_neuron_count")]
    pub neuron_count: usize,
}

fn default_connection_request_interval() -> u64 {
    10
}

fn default_use_simd() -> bool {
    true
}

fn default_neuron_count() -> usize {
    1
}

impl Default for CognitivezooidParams {
    fn default() -> Self {
        Self {
            izh_params: IzhikevichParams::default(),
            use_simd: default_use_simd(),
            connection_request_interval: default_connection_request_interval(),
            stdp_a_plus: default_stdp_a_plus(),
            stdp_a_minus: default_stdp_a_minus(),
            stdp_tau_plus: default_stdp_tau_plus(),
            stdp_tau_minus: default_stdp_tau_minus(),
            weight_decay: default_weight_decay(),
            pruning_threshold: default_pruning_threshold(),
            neuron_count: default_neuron_count(),
        }
    }
}

fn default_stdp_a_plus() -> f32 {
    0.01
}

fn default_stdp_a_minus() -> f32 {
    0.005
}

fn default_stdp_tau_plus() -> f32 {
    20.0
}

fn default_stdp_tau_minus() -> f32 {
    20.0
}

fn default_weight_decay() -> f32 {
    0.0001
}

fn default_pruning_threshold() -> f32 {
    0.001
}

#[derive(Debug, Clone)]
pub struct CognitivezooidState {
    pub izh_state: IzhikevichState,
    pub izh_params: IzhikevichParams,
    pub tick_count: u64,
    pub spike_count: u64,
    pub last_pre_spike_times: HashMap<ZoooidId, u64>, // For STDP timing
    pub synapse_manager: SynapseManager, // Centralized synapse management (replaces Vec<ZoooidId> + Vec<f32>)
    pub neuron_count: usize,
    /// ⚡ SIMD batch mode for future multi-neuron agents (PERF-003)
    /// When Some: processes N neurons in groups of 4 (3-4x speedup)
    /// When None: uses scalar Izhikevich neuron (single neuron)
    pub simd_batch: Option<IzhikevichBatch>,
    /// 🔧 GPU batch mode (PERF-003 GPU acceleration)
    /// When Some: processes N neurons on GPU compute shader (10-100x speedup for >1000 neurons)
    /// When None: uses CPU (SIMD or scalar)
    #[cfg(feature = "gpu")]
    pub gpu_batch: Option<crate::gpu::GpuIzhikevichBatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivezooidBlueprint {
    pub params: CognitivezooidParams,
}

#[async_trait]
impl AgentBlueprint for CognitivezooidBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let izh_state = <IzhikevichNeuron as crate::primitives::base::Primitive>::initialize(
            self.params.izh_params.clone(),
        );

        // Configure SynapseManager with sensible defaults for Cognitive agents
        let synapse_config = SynapseConfig {
            norm_mode: NormMode::L1, // L1 normalization for stable STDP
            max_connections: SystemConfig::global().max_degree_per_agent,
            min_weight: 0.0,
            max_weight: 1.0,
            default_weight: 0.1,
        };

        let neuron_count = self.params.neuron_count.max(1);
        let use_simd = self.params.use_simd && neuron_count >= 4;

        #[cfg(feature = "gpu")]
        let gpu_batch = if neuron_count >= 1024 {
            Some(crate::gpu::GpuIzhikevichBatch::new(neuron_count))
        } else {
            None
        };

        Ok(Box::new(CognitivezooidState {
            izh_state,
            izh_params: self.params.izh_params.clone(),
            tick_count: 0,
            spike_count: 0,
            last_pre_spike_times: HashMap::new(),
            synapse_manager: SynapseManager::new(synapse_config),
            neuron_count,
            simd_batch: if use_simd {
                Some(IzhikevichBatch::new(
                    neuron_count,
                    &IzhikevichBatchParams::from_scalar(
                        self.params.izh_params.a,
                        self.params.izh_params.b,
                        self.params.izh_params.c,
                        self.params.izh_params.d,
                        self.params.izh_params.dt,
                    ),
                ))
            } else {
                None
            },
            #[cfg(feature = "gpu")]
            gpu_batch,
        }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<(ZoooidId, Message)>,
        topology: &ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        let state = state
            .downcast_mut::<CognitivezooidState>()
            .ok_or("Invalid state type for Cognitivezooid")?;

        state.tick_count += 1;

        // Phase 1: Process incoming spikes and update synapse manager
        for (sender, msg) in &inputs {
            if let Message::SpikeEvent { .. } = msg {
                state.last_pre_spike_times.insert(*sender, state.tick_count);
                // Automatically add synapse via manager (O(1) if already exists)
                let _ = state.synapse_manager.add_synapse(*sender, 0.1);
            }
        }

        // Phase 2: Prepare inputs for Izhikevich neuron
        let primitive_inputs: Vec<PrimitiveMessage> = inputs
            .into_iter()
            .filter_map(|(_sender, msg)| match msg {
                Message::AnalogInput(value) => Some(PrimitiveMessage::InputCurrent(value)),
                _ => None,
            })
            .collect();

        // Phase 3: Update Izhikevich neuron state
        #[cfg(feature = "gpu")]
        let primitive_outputs = if let Some(_) = &mut state.gpu_batch {
            update_gpu_izhikevich_batch(state, &primitive_inputs, state.tick_count).await?
        } else if let Some(batch) = &mut state.simd_batch {
            debug_assert_eq!(batch.neuron_count, state.neuron_count);

            let input_currents: Vec<StateValue> = primitive_inputs
                .iter()
                .filter_map(|msg| match msg {
                    PrimitiveMessage::InputCurrent(value) => Some(*value),
                    _ => None,
                })
                .collect();
            let packed_inputs = pack_input_currents(&input_currents);

            let params_simd = IzhikevichBatchParams::from_scalar(
                state.izh_params.a,
                state.izh_params.b,
                state.izh_params.c,
                state.izh_params.d,
                state.izh_params.dt,
            );
            batch.update(&packed_inputs, &params_simd);

            let voltages = batch.get_v();
            let adaptations = batch.get_u();
            if !voltages.is_empty() {
                state.izh_state.v = voltages[0] as StateValue;
            }
            if !adaptations.is_empty() {
                state.izh_state.u = adaptations[0] as StateValue;
            }

            let mut outputs = Vec::new();
            if batch.spike_count() > 0 {
                outputs.push(PrimitiveMessage::Spike {
                    timestamp: state.tick_count,
                    amplitude: 1.0,
                });
            }
            outputs
        } else {
            let (new_izh_state, primitive_outputs) =
                <IzhikevichNeuron as crate::primitives::base::Primitive>::update(
                    state.izh_state.clone(),
                    &state.izh_params,
                    &primitive_inputs,
                );

            state.izh_state = new_izh_state;
            primitive_outputs
        };
        #[cfg(not(feature = "gpu"))]
        let primitive_outputs = if let Some(batch) = &mut state.simd_batch {
            debug_assert_eq!(batch.neuron_count, state.neuron_count);

            let input_currents: Vec<StateValue> = primitive_inputs
                .iter()
                .filter_map(|msg| match msg {
                    PrimitiveMessage::InputCurrent(value) => Some(*value),
                    _ => None,
                })
                .collect();
            let packed_inputs = pack_input_currents(&input_currents);

            let params_simd = IzhikevichBatchParams::from_scalar(
                state.izh_params.a,
                state.izh_params.b,
                state.izh_params.c,
                state.izh_params.d,
                state.izh_params.dt,
            );
            batch.update(&packed_inputs, &params_simd);

            let voltages = batch.get_v();
            let adaptations = batch.get_u();
            if !voltages.is_empty() {
                state.izh_state.v = voltages[0] as StateValue;
            }
            if !adaptations.is_empty() {
                state.izh_state.u = adaptations[0] as StateValue;
            }

            let mut outputs = Vec::new();
            if batch.spike_count() > 0 {
                outputs.push(PrimitiveMessage::Spike {
                    timestamp: state.tick_count,
                    amplitude: 1.0,
                });
            }
            outputs
        } else {
            let (new_izh_state, primitive_outputs) =
                <IzhikevichNeuron as crate::primitives::base::Primitive>::update(
                    state.izh_state.clone(),
                    &state.izh_params,
                    &primitive_inputs,
                );

            state.izh_state = new_izh_state;
            primitive_outputs
        };

        // Count spikes
        let has_spiked = primitive_outputs.iter().any(|prim_msg| matches!(prim_msg, PrimitiveMessage::Spike { .. }));
        if has_spiked {
            state.spike_count += 1;
        }

        // Phase 4: Apply STDP learning rule if post-synaptic spike occurred
        if has_spiked {
            for sender in state.synapse_manager.get_sources() {
                if let Some(&pre_time) = state.last_pre_spike_times.get(&sender) {
                    let delta_t = (state.tick_count as f32) - (pre_time as f32);
                    // STDP: Δw ∝ exp(-|Δt| / τ)
                    let dw = if delta_t > 0.0 {
                        // Pre-before-post: potentiation (Δt > 0)
                        self.params.stdp_a_plus * (-delta_t / self.params.stdp_tau_plus).exp()
                    } else {
                        // Post-before-pre: depression (Δt < 0)
                        -self.params.stdp_a_minus * (delta_t.abs() / self.params.stdp_tau_minus).exp()
                    };

                    // Apply weight decay and update
                    let decay_factor = 1.0 - self.params.weight_decay;
                    let actual_delta = dw * decay_factor;

                    let _ = state.synapse_manager.update_weight(sender, actual_delta, state.tick_count);

                    // Prune weak synapses
                    if let Some(weight) = state.synapse_manager.get_weight(sender) {
                        if weight.abs() < self.params.pruning_threshold {
                            let _ = state.synapse_manager.remove_synapse(sender);
                        }
                    }
                }
            }

            // Apply normalization to ensure stable weight ranges
            state.synapse_manager.normalize();
        }

        // Phase 5: Generate output spikes
        let output_messages = primitive_outputs
            .into_iter()
            .filter_map(|prim_msg| match prim_msg {
                PrimitiveMessage::Spike {
                    timestamp,
                    amplitude,
                } => Some(Message::SpikeEvent {
                    timestamp,
                    amplitude,
                }),
                _ => None,
            })
            .collect();

        // Phase 6: Periodic topology requests
        let topology_change = if state.tick_count % self.params.connection_request_interval == 0 {
            let all_nodes: Vec<_> = topology.graph.nodes().collect();
            if !all_nodes.is_empty() {
                let target_idx = (state.tick_count as usize) % all_nodes.len();
                let target = all_nodes[target_idx];
                Some(TopologyChange::RequestConnection(target))
            } else {
                None
            }
        } else {
            None
        };

        Ok(AgentUpdateResult::new(
            output_messages,
            None,
            topology_change,
            false,
        ))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Cognitive
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::topology::ZoooidTopology;
    use crate::core::message_bus::message::Message;
    use crate::utils::types::ZoooidId;

    #[tokio::test]
    async fn cognitivezooid_simd_branch_spikes() {
        let blueprint = CognitivezooidBlueprint {
            params: CognitivezooidParams {
                use_simd: true,
                neuron_count: 4,
                ..Default::default()
            },
        };

        let mut state = blueprint.initialize().await.unwrap();
        let sender = ZoooidId::new_v4();
        let inputs = vec![(sender, Message::AnalogInput(1000.0))];

        let result = blueprint
            .update(&mut state, inputs, &ZoooidTopology::new(), None)
            .await
            .unwrap();

        assert!(result
            .output_messages
            .iter()
            .any(|msg| matches!(msg, Message::SpikeEvent { .. })));
    }
}
