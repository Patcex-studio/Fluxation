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

use crate::utils::types::ZoooidId;
use std::collections::HashMap;

/// Lightweight profile for a scheduled agent.
#[derive(Debug, Clone)]
pub struct AgentResourceProfile {
    pub cpu_cost: usize,
    pub gpu_cost: usize,
    pub priority: u8,
}

impl Default for AgentResourceProfile {
    fn default() -> Self {
        Self {
            cpu_cost: 1,
            gpu_cost: 0,
            priority: 1,
        }
    }
}

/// Simple policy object for estimating relative load and prioritizing tasks.
#[derive(Debug, Clone)]
pub struct ResourceManagerPolicy {
    pub max_cpu_units: usize,
    pub max_gpu_units: usize,
    pub agent_profiles: HashMap<ZoooidId, AgentResourceProfile>,
}

impl ResourceManagerPolicy {
    pub fn new(max_cpu_units: usize, max_gpu_units: usize) -> Self {
        Self {
            max_cpu_units,
            max_gpu_units,
            agent_profiles: HashMap::new(),
        }
    }

    pub fn register_agent_profile(&mut self, agent_id: ZoooidId, profile: AgentResourceProfile) {
        self.agent_profiles.insert(agent_id, profile);
    }

    pub fn get_agent_profile(&self, agent_id: &ZoooidId) -> AgentResourceProfile {
        self.agent_profiles
            .get(agent_id)
            .cloned()
            .unwrap_or_default()
    }
}
