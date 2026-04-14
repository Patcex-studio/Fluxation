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

use crate::gpu::context::GpuContext;
use crate::utils::types::ZoooidId;
use std::collections::HashSet;

pub struct GpuResourceManager {
    gpu_context: GpuContext,
    allocated_agents: HashSet<ZoooidId>,
    max_concurrent_agents: usize,
}

impl GpuResourceManager {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let gpu_context = GpuContext::new().await?;
        Ok(Self {
            gpu_context,
            allocated_agents: HashSet::new(),
            max_concurrent_agents: 1,
        })
    }

    pub fn allocate_for_agent(&mut self, agent_id: ZoooidId) -> bool {
        if self.allocated_agents.contains(&agent_id) {
            return true;
        }

        if self.allocated_agents.len() >= self.max_concurrent_agents {
            return false;
        }

        self.allocated_agents.insert(agent_id);
        true
    }

    pub fn deallocate_for_agent(&mut self, agent_id: ZoooidId) {
        self.allocated_agents.remove(&agent_id);
    }

    pub fn is_agent_on_gpu(&self, agent_id: &ZoooidId) -> bool {
        self.allocated_agents.contains(agent_id)
    }

    pub fn get_context(&self) -> &GpuContext {
        &self.gpu_context
    }
}
