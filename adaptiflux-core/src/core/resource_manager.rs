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

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub agent_id: ZoooidId,
    pub cpu_units: usize,
    pub gpu_units: usize,
}

#[derive(Debug, Clone)]
pub struct ResourceInfo {
    pub cpu_available: usize,
    pub gpu_available: usize,
}

#[derive(Debug, Clone)]
pub struct ResourceManager {
    pub cpu_pool: usize,
    pub gpu_pool: usize,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            cpu_pool: 16,
            gpu_pool: 1,
        }
    }

    pub fn get_available_resources(&self) -> ResourceInfo {
        ResourceInfo {
            cpu_available: self.cpu_pool,
            gpu_available: self.gpu_pool,
        }
    }

    pub fn allocate_resources(&mut self, agent_id: ZoooidId) -> ResourceAllocation {
        ResourceAllocation {
            agent_id,
            cpu_units: 1,
            gpu_units: 0,
        }
    }

    pub fn allocate_gpu_resources(&mut self, agent_id: ZoooidId) -> Option<ResourceAllocation> {
        if self.gpu_pool > 0 {
            self.gpu_pool -= 1;
            Some(ResourceAllocation {
                agent_id,
                cpu_units: 0,
                gpu_units: 1,
            })
        } else {
            None
        }
    }

    pub fn deallocate_resources(&mut self, allocation: ResourceAllocation) {
        self.gpu_pool += allocation.gpu_units;
    }

    pub fn deallocate_gpu_resources(&mut self, allocation: ResourceAllocation) {
        self.gpu_pool += allocation.gpu_units;
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}
