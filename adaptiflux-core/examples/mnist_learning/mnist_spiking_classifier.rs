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
use std::sync::Arc;

use adaptiflux_core::agent::blueprint::base::AgentBlueprint;
use adaptiflux_core::agent::blueprint::pidzooid::{PIDzooidBlueprint, PIDzooidParams};
use adaptiflux_core::agent::state::{AgentUpdateResult, RoleType};
use adaptiflux_core::agent::zoooid::Zoooid;
use adaptiflux_core::core::message_bus::message::Message;
use adaptiflux_core::core::topology::{TopologyChange, ZoooidTopology};
use adaptiflux_core::learning::{FeedbackSignal, OnlineLearner};
use adaptiflux_core::memory::indexer::MetadataIndexer;
use adaptiflux_core::memory::long_term_store::TableLongTermStore;
use adaptiflux_core::memory::memory_integration::store_scalar_experience;
use adaptiflux_core::memory::types::MemoryPayload;
use adaptiflux_core::primitives::base::PrimitiveMessage;
use adaptiflux_core::primitives::control::pid::PidParams;
use adaptiflux_core::primitives::spiking::izhikevich::{
    IzhikevichNeuron, IzhikevichParams, IzhikevichState,
};
use adaptiflux_core::primitives::spiking::lif::LifParams;
use adaptiflux_core::ZoooidId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingNeuronParams {
    pub izh_params: IzhikevichParams,
    pub input_gain: f32,
    pub gain_clamp_min: f32,
    pub gain_clamp_max: f32,
    pub connection_request_interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingNeuronState {
    pub izh_state: IzhikevichState,
    pub input_gain: f32,
    pub spike_count: u64,
    pub tick_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingNeuronBlueprint {
    pub params: SpikingNeuronParams,
    pub is_classifier: bool,
}

#[async_trait]
impl AgentBlueprint for SpikingNeuronBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let izh_state =
            <IzhikevichNeuron as adaptiflux_core::primitives::base::Primitive>::initialize(
                self.params.izh_params.clone(),
            );

        Ok(Box::new(SpikingNeuronState {
            izh_state,
            input_gain: self.params.input_gain,
            spike_count: 0,
            tick_count: 0,
        }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<Message>,
        topology: &ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        let state = state
            .downcast_mut::<SpikingNeuronState>()
            .ok_or("Invalid state type for spiking neuron")?;

        state.tick_count += 1;

        let mut total_input: f32 = 0.0;
        let mut control_adjustment: f32 = 0.0;

        for message in inputs {
            match message {
                Message::SpikeEvent { amplitude, .. } => {
                    total_input += amplitude;
                }
                Message::ControlSignal(value) => {
                    control_adjustment += value;
                }
                _ => {}
            }
        }

        if control_adjustment.abs() > 1e-12 {
            state.input_gain = (state.input_gain + control_adjustment * 0.02)
                .clamp(self.params.gain_clamp_min, self.params.gain_clamp_max);
        }

        let input_current = total_input * state.input_gain;
        let primitive_inputs = vec![PrimitiveMessage::InputCurrent(input_current)];

        let (new_state, primitive_outputs) =
            <IzhikevichNeuron as adaptiflux_core::primitives::base::Primitive>::update(
                state.izh_state.clone(),
                &self.params.izh_params,
                &primitive_inputs,
            );
        state.izh_state = new_state;

        let output_messages: Vec<Message> = primitive_outputs
            .into_iter()
            .filter_map(|prim_msg| match prim_msg {
                PrimitiveMessage::Spike {
                    timestamp,
                    amplitude,
                } => {
                    if self.is_classifier {
                        state.spike_count += 1;
                    }
                    Some(Message::SpikeEvent {
                        timestamp,
                        amplitude,
                    })
                }
                _ => None,
            })
            .collect();

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

#[derive(Debug, Clone)]
pub struct MnistSpikingClassifierArchitecture {
    pub sensor_ids: Vec<ZoooidId>,
    pub hidden_ids: Vec<ZoooidId>,
    pub output_ids: Vec<ZoooidId>,
    pub pid_id: ZoooidId,
}

impl MnistSpikingClassifierArchitecture {
    pub fn all_adaptation_targets(&self) -> Vec<ZoooidId> {
        let mut ids = self.hidden_ids.clone();
        ids.extend(self.output_ids.iter());
        ids.push(self.pid_id);
        ids
    }

    pub fn reset_output_counts(
        &self,
        scheduler: &mut adaptiflux_core::core::scheduler::CoreScheduler,
    ) {
        for output_id in &self.output_ids {
            if let Some(handle) = scheduler.agents.get_mut(output_id) {
                if let Some(state) = handle.state.downcast_mut::<SpikingNeuronState>() {
                    state.spike_count = 0;
                }
            }
        }
    }

    pub fn decode_output(&self, scheduler: &adaptiflux_core::core::scheduler::CoreScheduler) -> u8 {
        let mut winner = 0;
        let mut max_spikes = 0;

        for (idx, output_id) in self.output_ids.iter().enumerate() {
            if let Some(handle) = scheduler.agents.get(output_id) {
                if let Some(state) = handle.state.downcast_ref::<SpikingNeuronState>() {
                    if state.spike_count > max_spikes {
                        max_spikes = state.spike_count;
                        winner = idx;
                    }
                }
            }
        }

        winner as u8
    }
}

pub async fn build_mnist_architecture(
    scheduler: &mut adaptiflux_core::core::scheduler::CoreScheduler,
) -> Result<MnistSpikingClassifierArchitecture, Box<dyn std::error::Error + Send + Sync>> {
    const SENSOR_SIDE: usize = 14;
    const HIDDEN_NEURONS: usize = 32;
    const OUTPUT_NEURONS: usize = 10;

    let mut sensor_ids = Vec::with_capacity(SENSOR_SIDE * SENSOR_SIDE);
    for _ in 0..SENSOR_SIDE * SENSOR_SIDE {
        let sensor_blueprint =
            adaptiflux_core::agent::blueprint::sensorzooid::SensorzooidBlueprint {
                params: adaptiflux_core::agent::blueprint::sensorzooid::SensorzooidParams {
                    lif_params: LifParams {
                        tau_m: 12.0,
                        v_rest: -70.0,
                        v_thresh: -52.0,
                        v_reset: -70.0,
                        r_m: 12.0,
                        dt: 1.0,
                    },
                    connection_request_interval: 10,
                },
            };
        let zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(sensor_blueprint)).await?;
        sensor_ids.push(zooid.id);
        scheduler.spawn_agent(zooid).await?;
    }

    let mut hidden_ids = Vec::with_capacity(HIDDEN_NEURONS);
    for _ in 0..HIDDEN_NEURONS {
        let hidden_blueprint = SpikingNeuronBlueprint {
            params: SpikingNeuronParams {
                izh_params: IzhikevichParams::default(),
                input_gain: 0.12,
                gain_clamp_min: 0.01,
                gain_clamp_max: 5.0,
                connection_request_interval: 14,
            },
            is_classifier: false,
        };
        let zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(hidden_blueprint)).await?;
        hidden_ids.push(zooid.id);
        scheduler.spawn_agent(zooid).await?;
    }

    let mut output_ids = Vec::with_capacity(OUTPUT_NEURONS);
    for _ in 0..OUTPUT_NEURONS {
        let output_blueprint = SpikingNeuronBlueprint {
            params: SpikingNeuronParams {
                izh_params: IzhikevichParams {
                    a: 0.02,
                    b: 0.2,
                    c: -65.0,
                    d: 2.0,
                    dt: 1.0,
                },
                input_gain: 0.10,
                gain_clamp_min: 0.01,
                gain_clamp_max: 6.0,
                connection_request_interval: 14,
            },
            is_classifier: true,
        };
        let zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(output_blueprint)).await?;
        output_ids.push(zooid.id);
        scheduler.spawn_agent(zooid).await?;
    }

    let pid_blueprint = PIDzooidBlueprint {
        params: PIDzooidParams {
            pid_params: PidParams {
                kp: 0.8,
                ki: 0.1,
                kd: 0.01,
                dt: 1.0,
            },
            connection_request_interval: 12,
        },
    };
    let pid_zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(pid_blueprint)).await?;
    let pid_id = pid_zooid.id;
    scheduler.spawn_agent(pid_zooid).await?;

    let mut topology = scheduler.topology.lock().await;
    for &sensor_id in &sensor_ids {
        for &hidden_id in &hidden_ids {
            topology.add_edge(sensor_id, hidden_id, Default::default());
        }
    }
    for &hidden_id in &hidden_ids {
        for &output_id in &output_ids {
            topology.add_edge(hidden_id, output_id, Default::default());
        }
    }
    for &output_id in &output_ids {
        topology.add_edge(pid_id, output_id, Default::default());
    }

    Ok(MnistSpikingClassifierArchitecture {
        sensor_ids,
        hidden_ids,
        output_ids,
        pid_id,
    })
}

pub struct SpikingGainLearner {
    pub learning_rate: f32,
}

impl OnlineLearner for SpikingGainLearner {
    fn adapt_parameters(
        &self,
        agent_id: ZoooidId,
        state: &mut Box<dyn Any + Send + Sync>,
        _role: RoleType,
        feedback: &FeedbackSignal,
    ) {
        let error = feedback
            .per_agent
            .get(&agent_id)
            .copied()
            .or(feedback.global_scalar)
            .unwrap_or(0.0)
            + feedback.memory_bias.get(&agent_id).copied().unwrap_or(0.0);

        if let Some(layer_state) = state.downcast_mut::<SpikingNeuronState>() {
            if error.abs() > 1e-9 {
                layer_state.input_gain =
                    (layer_state.input_gain - error * self.learning_rate).clamp(0.01, 8.0);
            }
        }
    }
}

pub struct OutputExperienceRecorder {
    pub output_ids: Vec<ZoooidId>,
}

impl adaptiflux_core::memory::ExperienceRecorder for OutputExperienceRecorder {
    fn record_after_step(
        &self,
        agent_id: ZoooidId,
        iteration: u64,
        inputs: &[Message],
        state: &dyn Any,
        _result: &AgentUpdateResult,
        store: &mut TableLongTermStore,
        indexer: &mut MetadataIndexer,
    ) {
        if !self.output_ids.contains(&agent_id) {
            return;
        }

        if let Some(output_state) = state.downcast_ref::<SpikingNeuronState>() {
            let embedding = vec![
                output_state.input_gain,
                output_state.spike_count as f32,
                inputs
                    .iter()
                    .filter_map(|m| match m {
                        Message::SpikeEvent { amplitude, .. } => Some(*amplitude),
                        _ => None,
                    })
                    .sum(),
            ];

            let payload = Arc::new(output_state.clone());
            store_scalar_experience(
                store,
                indexer,
                agent_id,
                iteration,
                "mnist_classification",
                Some(embedding),
                payload,
                output_state.spike_count as f32,
            );
        }
    }
}
