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
    #[serde(default = "default_stdp_causal_window")]
    pub stdp_causal_window: u64,
    #[serde(default = "default_stdp_min_correlation")]
    pub stdp_min_correlation: f32,
    #[serde(default = "default_stdp_enable_causal_gate")]
    pub stdp_enable_causal_gate: bool,
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
            stdp_causal_window: default_stdp_causal_window(),
            stdp_min_correlation: default_stdp_min_correlation(),
            stdp_enable_causal_gate: default_stdp_enable_causal_gate(),
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

fn default_stdp_causal_window() -> u64 {
    100
}

fn default_stdp_min_correlation() -> f32 {
    0.1
}

fn default_stdp_enable_causal_gate() -> bool {
    true
}

#[derive(Debug, Clone)]
pub struct CognitivezooidState {
    pub izh_state: IzhikevichState,
    pub izh_params: IzhikevichParams,
    pub tick_count: u64,
    pub spike_count: u64,
    pub last_pre_spike_times: HashMap<ZoooidId, u64>, // For STDP timing
    pub last_post_spike_time: Option<u64>, // Last post (this neuron) spike event time
    pub synapse_manager: SynapseManager, // Centralized synapse management (replaces Vec<ZoooidId> + Vec<f32>)
    pub eligibility_traces: HashMap<ZoooidId, (f32, u64)>, // (trace_value, last_update_time)
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
            last_post_spike_time: None,
            eligibility_traces: HashMap::new(),
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
        // Store the incoming spike's timestamp (event time), not local tick.
        // Also apply LTD (post-before-pre) on pre-event when a recent post exists.
        let mut pre_updates_applied = false;
        for (sender, msg) in &inputs {
            if let Message::SpikeEvent { timestamp, .. } = msg {
                let pre_time = *timestamp;
                state.last_pre_spike_times.insert(*sender, pre_time);
                // Automatically add synapse via manager (O(1) if already exists)
                let _ = state.synapse_manager.add_synapse(*sender, 0.1);

                // If this neuron has recently spiked (post-before-pre), apply LTD
                if let Some(post_time) = state.last_post_spike_time {
                    // Δt = post - pre (negative when post before pre) -> depression
                    let delta_t = (post_time as f32) - (pre_time as f32);

                    // causal gating: check window and eligibility before applying LTD
                    let mut allow_update = true;
                    if self.params.stdp_enable_causal_gate {
                        let within_window = delta_t.abs() <= (self.params.stdp_causal_window as f32);
                        let trace_val = state
                            .eligibility_traces
                            .get(sender)
                            .map(|(v, _)| *v)
                            .unwrap_or(0.0);
                        if !within_window || trace_val < self.params.stdp_min_correlation {
                            allow_update = false;
                        }
                    }

                    if allow_update {
                        let dw = if delta_t > 0.0 {
                            self.params.stdp_a_plus * (-delta_t / self.params.stdp_tau_plus).exp()
                        } else {
                            -self.params.stdp_a_minus * (delta_t.abs() / self.params.stdp_tau_minus).exp()
                        };

                        if let Some(current_w) = state.synapse_manager.get_weight(*sender) {
                            let last_updated = state
                                .synapse_manager
                                .get_entry(*sender)
                                .map(|e| e.last_updated)
                                .unwrap_or(pre_time);
                            let elapsed = if pre_time > last_updated { pre_time - last_updated } else { 0 };
                            if elapsed > 0 {
                                let decay_factor = (-(self.params.weight_decay) * (elapsed as f32)).exp();
                                let decayed = current_w * decay_factor;
                                let delta_decay = decayed - current_w;
                                let _ = state
                                    .synapse_manager
                                    .update_weight(*sender, delta_decay, pre_time);
                            }
                        }

                        if state.synapse_manager.update_weight(*sender, dw, pre_time).is_ok() {
                            pre_updates_applied = true;
                        }

                        // Prune weak synapses
                        if let Some(weight) = state.synapse_manager.get_weight(*sender) {
                            if weight.abs() < self.params.pruning_threshold {
                                let _ = state.synapse_manager.remove_synapse(*sender);
                            }
                        }
                    }
                }
                // Update eligibility trace for this pre-synaptic source.
                // Decay existing trace to current pre_time using stdp_tau_plus as time constant.
                let entry = state.eligibility_traces.entry(*sender).or_insert((0.0, pre_time));
                let (ref mut trace_val, ref mut last_ts) = *entry;
                if *last_ts < pre_time {
                    let dt = (pre_time - *last_ts) as f32;
                    let decay = (-dt / self.params.stdp_tau_plus).exp();
                    *trace_val *= decay;
                }
                *trace_val += 1.0; // increment on pre-spike
                *last_ts = pre_time;
            }
        }

        // Normalize after any pre-event updates to keep weights stable
        if pre_updates_applied {
            state.synapse_manager.normalize();
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

        // Count spikes and capture post-spike timestamp if present
        let has_spiked = primitive_outputs.iter().any(|prim_msg| matches!(prim_msg, PrimitiveMessage::Spike { .. }));
        if has_spiked {
            state.spike_count += 1;
        }

        // Determine post-spike event time (prefer primitive's timestamp; fallback to local tick)
        let post_spike_time_opt: Option<u64> = primitive_outputs.iter().find_map(|prim_msg| {
            if let PrimitiveMessage::Spike { timestamp, .. } = prim_msg {
                Some(*timestamp)
            } else {
                None
            }
        });

        // Phase 4: Apply STDP learning rule if post-synaptic spike occurred
        if has_spiked {
            let post_time = post_spike_time_opt.unwrap_or(state.tick_count);
            // record last post spike time for future pre-events
            state.last_post_spike_time = Some(post_time);

            for sender in state.synapse_manager.get_sources() {
                if let Some(&pre_time) = state.last_pre_spike_times.get(&sender) {
                    // Use event timestamps for Δt: post_time - pre_time
                    let delta_t = (post_time as f32) - (pre_time as f32);

                    // Decay eligibility trace to post_time if present
                    if let Some(ent) = state.eligibility_traces.get_mut(&sender) {
                        let (ref mut trace_val, ref mut trace_ts) = *ent;
                        if *trace_ts < post_time {
                            let dt = (post_time - *trace_ts) as f32;
                            let decay = (-dt / self.params.stdp_tau_plus).exp();
                            *trace_val *= decay;
                            *trace_ts = post_time;
                        }
                    }

                    // Causal gating: require pre/post within window and minimum eligibility
                    if self.params.stdp_enable_causal_gate {
                        let within_window = delta_t.abs() <= (self.params.stdp_causal_window as f32);
                        let trace_val = state
                            .eligibility_traces
                            .get(&sender)
                            .map(|(v, _)| *v)
                            .unwrap_or(0.0);
                        if !within_window || trace_val < self.params.stdp_min_correlation {
                            // skip update if causal condition not met
                            continue;
                        }
                    }

                    // STDP: Δw ∝ exp(-|Δt| / τ)
                    let dw = if delta_t > 0.0 {
                        // Pre-before-post: potentiation (Δt > 0)
                        self.params.stdp_a_plus * (-delta_t / self.params.stdp_tau_plus).exp()
                    } else {
                        // Post-before-pre: depression (Δt < 0)
                        -self.params.stdp_a_minus * (delta_t.abs() / self.params.stdp_tau_minus).exp()
                    };

                    // Apply continuous decay to the existing weight based on elapsed time since last update
                    if let Some(current_w) = state.synapse_manager.get_weight(sender) {
                        // retrieve last_updated from entry if available
                        let last_updated = state
                            .synapse_manager
                            .get_entry(sender)
                            .map(|e| e.last_updated)
                            .unwrap_or(post_time);
                        let elapsed = if post_time > last_updated { post_time - last_updated } else { 0 };
                        if elapsed > 0 {
                            let decay_factor = (- (self.params.weight_decay) * (elapsed as f32)).exp();
                            let decayed = current_w * decay_factor;
                            let delta_decay = decayed - current_w;
                            // apply decay delta to set decayed weight
                            let _ = state
                                .synapse_manager
                                .update_weight(sender, delta_decay, post_time);
                        }
                    }

                    // Apply the STDP delta after decay
                    let actual_delta = dw;

                    let _ = state
                        .synapse_manager
                        .update_weight(sender, actual_delta, post_time);

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

        #[tokio::test]
        async fn stdp_pre_before_post_causes_ltp() {
            let mut params = CognitivezooidParams::default();
            params.use_simd = false;
            params.neuron_count = 1;
            // allow LTD even without prior eligibility for this unit test
            params.stdp_min_correlation = 0.0;
            let blueprint = CognitivezooidBlueprint { params };

            let mut state = blueprint.initialize().await.unwrap();
            let sender = ZoooidId::new_v4();
            let sender2 = ZoooidId::new_v4();

            // disable normalization to observe raw weight changes
            {
                let st = state.downcast_mut::<CognitivezooidState>().unwrap();
                st.synapse_manager.config_mut().norm_mode = NormMode::None;
                let _ = st.synapse_manager.add_synapse(sender2, 0.1);
            }

            // send pre-spike with event timestamp (use small timestamp to be before post)
            let _ = blueprint
                .update(
                    &mut state,
                    vec![(sender, Message::SpikeEvent { timestamp: 1, amplitude: 1.0 })],
                    &ZoooidTopology::new(),
                    None,
                )
                .await
                .unwrap();

            // cause post spike via analog input
            let _ = blueprint
                .update(
                    &mut state,
                    vec![(ZoooidId::new_v4(), Message::AnalogInput(1000.0))],
                    &ZoooidTopology::new(),
                    None,
                )
                .await
                .unwrap();

            let st = state.downcast_ref::<CognitivezooidState>().unwrap();
            let updates = st.synapse_manager.get_update_count();
            assert!(updates > 0, "expected at least one synapse update (LTP) occurred");
        }

        #[tokio::test]
        async fn stdp_post_before_pre_causes_ltd() {
            let mut params = CognitivezooidParams::default();
            params.use_simd = false;
            params.neuron_count = 1;
            params.stdp_min_correlation = 0.0; // allow LTD without prior eligibility for test
            let blueprint = CognitivezooidBlueprint { params };

            let mut state = blueprint.initialize().await.unwrap();
            let sender = ZoooidId::new_v4();
            let sender2 = ZoooidId::new_v4();

            // disable normalization to observe raw weight changes
            {
                let st = state.downcast_mut::<CognitivezooidState>().unwrap();
                st.synapse_manager.config_mut().norm_mode = NormMode::None;
                let _ = st.synapse_manager.add_synapse(sender2, 0.1);
            }

            // cause post spike first
            let _ = blueprint
                .update(
                    &mut state,
                    vec![(ZoooidId::new_v4(), Message::AnalogInput(1000.0))],
                    &ZoooidTopology::new(),
                    None,
                )
                .await
                .unwrap();

            // read post_time
            let st = state.downcast_ref::<CognitivezooidState>().unwrap();
            let post_time = st.last_post_spike_time.unwrap_or(0);
            let _ = st;

            // send pre-spike with timestamp after post_time to trigger LTD
            let pre_ts = post_time + 1;
            let _ = blueprint
                .update(
                    &mut state,
                    vec![(sender, Message::SpikeEvent { timestamp: pre_ts, amplitude: 1.0 })],
                    &ZoooidTopology::new(),
                    None,
                )
                .await
                .unwrap();

            let st = state.downcast_ref::<CognitivezooidState>().unwrap();
            let updates = st.synapse_manager.get_update_count();
            assert!(updates > 0, "expected at least one synapse update (LTD) occurred");
        }

        #[tokio::test]
        async fn causal_gate_blocks_out_of_window_updates() {
            let mut params = CognitivezooidParams::default();
            params.stdp_enable_causal_gate = true;
            params.stdp_causal_window = 10; // small window
            params.stdp_min_correlation = 0.0; // allow correlation threshold low

            let blueprint = CognitivezooidBlueprint { params };
            let mut state = blueprint.initialize().await.unwrap();
            let sender = ZoooidId::new_v4();

            // cause post spike first
            let _ = blueprint
                .update(
                    &mut state,
                    vec![(ZoooidId::new_v4(), Message::AnalogInput(1000.0))],
                    &ZoooidTopology::new(),
                    None,
                )
                .await
                .unwrap();

            // read post_time
            let st = state.downcast_ref::<CognitivezooidState>().unwrap();
            let post_time = st.last_post_spike_time.unwrap_or(0);
            let _ = st;

            // send pre-spike far outside causal window
            let pre_ts = post_time + 1000;
            let _ = blueprint
                .update(
                    &mut state,
                    vec![(sender, Message::SpikeEvent { timestamp: pre_ts, amplitude: 1.0 })],
                    &ZoooidTopology::new(),
                    None,
                )
                .await
                .unwrap();

            let st = state.downcast_ref::<CognitivezooidState>().unwrap();
            let weight = st.synapse_manager.get_weight(sender).unwrap_or(0.0);
            // Expect weight to remain at default (no LTD applied)
            assert!((weight - 0.1).abs() < 1e-6, "expected causal gate to block update");
        }

        #[tokio::test]
        async fn decay_applied_for_elapsed_time() {
            let blueprint = CognitivezooidBlueprint {
                params: CognitivezooidParams { use_simd: false, neuron_count: 1, stdp_enable_causal_gate: false, ..Default::default() },
            };

            let mut state = blueprint.initialize().await.unwrap();
            let sender = ZoooidId::new_v4();

            // add synapse explicitly
            {
                let st = state.downcast_mut::<CognitivezooidState>().unwrap();
                // disable normalization to observe raw decay
                st.synapse_manager.config_mut().norm_mode = NormMode::None;
                let _ = st.synapse_manager.add_synapse(sender, 0.5);
                if let Some(entry) = st.synapse_manager.get_entry_mut(sender) {
                    entry.last_updated = 0; // set old timestamp
                }
                // ensure we have a recorded pre-spike time so post-spike STDP loop will process this sender
                st.last_pre_spike_times.insert(sender, 0);
            }

            // bump local tick_count before update to simulate large elapsed time
            {
                let st_mut = state.downcast_mut::<CognitivezooidState>().unwrap();
                st_mut.tick_count = 1_000_000;
                if let Some(entry) = st_mut.synapse_manager.get_entry(sender) {
                    // sanity check: last_updated should be old (0)
                    let last = entry.last_updated;
                    assert_eq!(last, 0, "expected entry.last_updated to be 0 before update");
                }
            }

            // cause post spike at a large time to trigger decay
            let res = blueprint
                .update(
                    &mut state,
                    vec![(ZoooidId::new_v4(), Message::AnalogInput(1000.0))],
                    &ZoooidTopology::new(),
                    None,
                )
                .await
                .unwrap();

            // ensure the neuron actually spiked (otherwise STDP won't run)
            assert!(res.output_messages.iter().any(|m| matches!(m, Message::SpikeEvent { .. })), "expected post spike to occur");

            let st = state.downcast_ref::<CognitivezooidState>().unwrap();
            let updates = st.synapse_manager.get_update_count();
            let weight = st.synapse_manager.get_weight(sender).unwrap_or(0.0);
            let last_updated = st.synapse_manager.get_entry(sender).map(|e| e.last_updated).unwrap_or(0);
            // Expect that at least one update happened and weight decayed from 0.5 towards 0 (since elapsed large)
            assert!(updates > 0, "expected synapse updates to have occurred (none seen)");
            assert!(weight < 0.5, "expected weight to decay over elapsed time; last_updated={}, weight={}", last_updated, weight);
        }
}
