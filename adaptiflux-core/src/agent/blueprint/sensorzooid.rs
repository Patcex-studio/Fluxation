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

use crate::agent::blueprint::base::AgentBlueprint;
use crate::agent::state::{AgentUpdateResult, RoleType};
use crate::core::message_bus::message::Message;
use crate::core::topology::{TopologyChange, ZoooidTopology};
use crate::memory::types::MemoryPayload;
use crate::primitives::base::PrimitiveMessage;
use crate::primitives::spiking::lif::{LifNeuron, LifParams, LifState};use crate::utils::types::ZoooidId;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorzooidParams {
    pub lif_params: LifParams,
    #[serde(default = "default_connection_request_interval")]
    pub connection_request_interval: u64,
}

fn default_connection_request_interval() -> u64 {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorzooidState {
    pub lif_state: LifState,
    pub tick_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorzooidBlueprint {
    pub params: SensorzooidParams,
}

#[async_trait]
impl AgentBlueprint for SensorzooidBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let lif_state = <LifNeuron as crate::primitives::base::Primitive>::initialize(
            self.params.lif_params.clone(),
        );
        Ok(Box::new(SensorzooidState {
            lif_state,
            tick_count: 0,
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
            .downcast_mut::<SensorzooidState>()
            .ok_or("Invalid state type for Sensorzooid")?;

        // Increment tick counter
        state.tick_count += 1;

        let primitive_inputs: Vec<PrimitiveMessage> = inputs
            .into_iter()
            .filter_map(|(_sender, msg)| match msg {
                Message::AnalogInput(value) => Some(PrimitiveMessage::InputCurrent(value)),
                _ => None,
            })
            .collect();

        let (new_lif_state, primitive_outputs) =
            <LifNeuron as crate::primitives::base::Primitive>::update(
                state.lif_state.clone(),
                &self.params.lif_params,
                &primitive_inputs,
            );

        state.lif_state = new_lif_state;

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

        // Periodically request connections to other agents (topology bootstrapping)
        let topology_change = if state.tick_count % self.params.connection_request_interval == 0 {
            // Get all available nodes and pick a random one (excluding self would require knowing our ID)
            let all_nodes: Vec<_> = topology.graph.nodes().collect();
            if !all_nodes.is_empty() {
                // Use tick_count as a pseudo-random seed to select target
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
        RoleType::Sensor
    }
}
