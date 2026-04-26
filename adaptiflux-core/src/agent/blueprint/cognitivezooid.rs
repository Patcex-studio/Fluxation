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
use crate::core::topology::{TopologyChange, ZoooidTopology};
use crate::memory::types::MemoryPayload;
use crate::primitives::base::PrimitiveMessage;
use crate::primitives::spiking::izhikevich::{IzhikevichNeuron, IzhikevichParams, IzhikevichState};
use crate::utils::types::ZoooidId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitivezooidParams {
    pub izh_params: IzhikevichParams,
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
}

fn default_connection_request_interval() -> u64 {
    10
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
            max_connections: 50,
            min_weight: 0.0,
            max_weight: 1.0,
            default_weight: 0.1,
        };

        Ok(Box::new(CognitivezooidState {
            izh_state,
            izh_params: self.params.izh_params.clone(),
            tick_count: 0,
            spike_count: 0,
            last_pre_spike_times: HashMap::new(),
            synapse_manager: SynapseManager::new(synapse_config),
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
        let (new_izh_state, primitive_outputs) =
            <IzhikevichNeuron as crate::primitives::base::Primitive>::update(
                state.izh_state.clone(),
                &state.izh_params,
                &primitive_inputs,
            );

        state.izh_state = new_izh_state;

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
