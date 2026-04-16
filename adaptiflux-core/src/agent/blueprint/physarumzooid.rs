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
use crate::primitives::physical::physarum::{PhysarumModel, PhysarumParams, PhysarumState};
use crate::utils::types::ZoooidId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysarumzooidParams {
    pub physarum_params: PhysarumParams,
    #[serde(default = "default_connection_request_interval")]
    pub connection_request_interval: u64,
}

fn default_connection_request_interval() -> u64 {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysarumzooidState {
    pub physarum_state: PhysarumState,
    pub tick_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysarumzooidBlueprint {
    pub params: PhysarumzooidParams,
}

#[async_trait]
impl AgentBlueprint for PhysarumzooidBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let physarum_state = <PhysarumModel as crate::primitives::base::Primitive>::initialize(
            self.params.physarum_params.clone(),
        );
        Ok(Box::new(PhysarumzooidState {
            physarum_state,
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
            .downcast_mut::<PhysarumzooidState>()
            .ok_or("Invalid state type for Physarumzooid")?;

        state.tick_count += 1;

        let primitive_inputs: Vec<PrimitiveMessage> = inputs
            .into_iter()
            .filter_map(|(_sender, msg)| match msg {
                Message::PheromoneLevel(strength, type_id) => {
                    Some(PrimitiveMessage::Pheromone { strength, type_id })
                }
                _ => None,
            })
            .collect();

        let (new_physarum_state, primitive_outputs) =
            <PhysarumModel as crate::primitives::base::Primitive>::update(
                state.physarum_state.clone(),
                &self.params.physarum_params,
                &primitive_inputs,
            );

        state.physarum_state = new_physarum_state;

        let output_messages = primitive_outputs
            .into_iter()
            .filter_map(|prim_msg| match prim_msg {
                PrimitiveMessage::ControlSignal(flow_strength) => {
                    Some(Message::ConductivityUpdate(flow_strength))
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
        RoleType::Physarum
    }
}
