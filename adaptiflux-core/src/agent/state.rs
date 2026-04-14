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

use crate::core::message_bus::message::Message;
use crate::core::topology::TopologyChange;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoleType {
    Sensor,
    Pid,
    Swarm,
    Cognitive,
    Physarum,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct AgentUpdateResult {
    pub output_messages: Vec<Message>,
    pub new_role: Option<RoleType>,
    pub topology_change_request: Option<TopologyChange>,
    pub terminate: bool,
}

impl AgentUpdateResult {
    pub fn new(
        output_messages: Vec<Message>,
        new_role: Option<RoleType>,
        topology_change_request: Option<TopologyChange>,
        terminate: bool,
    ) -> Self {
        Self {
            output_messages,
            new_role,
            topology_change_request,
            terminate,
        }
    }
}
