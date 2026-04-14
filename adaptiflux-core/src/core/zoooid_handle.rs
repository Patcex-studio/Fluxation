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

use std::any::Any;

use crate::agent::blueprint::AgentBlueprint;
use crate::utils::types::ZoooidId;

/// Handle to an active agent in the scheduler
pub struct ZoooidHandle {
    pub id: ZoooidId,
    pub blueprint: Box<dyn AgentBlueprint + Send + Sync>,
    pub state: Box<dyn Any + Send + Sync>,
    pub update_count: u64,
    #[cfg(feature = "gpu")]
    pub gpu_allocated: bool,
}

/// Scheduler iteration statistics
#[derive(Debug, Clone)]
pub struct SchedulerMetrics {
    pub iteration_count: u64,
    pub total_agents: usize,
    pub total_connections: usize,
    pub agents_updated: usize,
    pub agents_terminated: usize,
    pub topology_changes: usize,
    pub avg_iteration_time_ms: f64,
}

impl Default for SchedulerMetrics {
    fn default() -> Self {
        Self {
            iteration_count: 0,
            total_agents: 0,
            total_connections: 0,
            agents_updated: 0,
            agents_terminated: 0,
            topology_changes: 0,
            avg_iteration_time_ms: 0.0,
        }
    }
}
