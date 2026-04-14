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
use crate::primitives::control::pid::{PidController, PidParams, PidState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIDzooidParams {
    pub pid_params: PidParams,
    #[serde(default = "default_connection_request_interval")]
    pub connection_request_interval: u64,
}

fn default_connection_request_interval() -> u64 {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIDzooidState {
    pub pid_state: PidState,
    /// Live gains (may be updated by online adaptation each tick).
    pub pid_params: PidParams,
    pub tick_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIDzooidBlueprint {
    pub params: PIDzooidParams,
}

#[async_trait]
impl AgentBlueprint for PIDzooidBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let pid_state = <PidController as crate::primitives::base::Primitive>::initialize(
            self.params.pid_params.clone(),
        );
        Ok(Box::new(PIDzooidState {
            pid_state,
            pid_params: self.params.pid_params.clone(),
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
            .downcast_mut::<PIDzooidState>()
            .ok_or("Invalid state type for PIDzooid")?;

        state.tick_count += 1;

        let primitive_inputs: Vec<PrimitiveMessage> = inputs
            .into_iter()
            .filter_map(|msg| match msg {
                Message::Error(value) => Some(PrimitiveMessage::Error(value)),
                _ => None,
            })
            .collect();

        let (new_pid_state, primitive_outputs) =
            <PidController as crate::primitives::base::Primitive>::update(
                state.pid_state.clone(),
                &state.pid_params,
                &primitive_inputs,
            );

        state.pid_state = new_pid_state;

        let output_messages = primitive_outputs
            .into_iter()
            .filter_map(|prim_msg| match prim_msg {
                PrimitiveMessage::ControlSignal(signal) => Some(Message::ControlSignal(signal)),
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
        RoleType::Pid
    }
}
