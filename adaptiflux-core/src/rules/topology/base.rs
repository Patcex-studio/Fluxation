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

use crate::agent::blueprint::AgentBlueprint;
use crate::core::topology::{SystemMetrics, ZoooidTopology};
use crate::utils::types::{StateValue, ZoooidId};

/// Reason metadata for structural pruning (synaptic / Physarum-inspired).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PruneReason {
    LowConductivity,
    StdpDepression,
    Unused,
    Custom(String),
}

/// Actions that modify topology, edge weights, or agent lifecycle.
pub enum TopologyAction {
    AddEdge(ZoooidId, ZoooidId),
    RemoveEdge(ZoooidId, ZoooidId),
    CreateZoooid(Box<dyn AgentBlueprint + Send + Sync>),
    TerminateZoooid(ZoooidId),
    /// Remove edge with explicit plasticity reason (same graph effect as `RemoveEdge`).
    PruneEdge {
        from: ZoooidId,
        to: ZoooidId,
        reason: PruneReason,
    },
    StrengthenConnection {
        from: ZoooidId,
        to: ZoooidId,
        delta_weight: StateValue,
    },
    WeakenConnection {
        from: ZoooidId,
        to: ZoooidId,
        delta_weight: StateValue,
    },
    /// Spawn from a blueprint template; optional hint for locality-aware wiring.
    CreateAgentFromTemplate {
        template_blueprint: Box<dyn AgentBlueprint + Send + Sync>,
        target_area_hint: Option<ZoooidId>,
    },
    InitiateApoptosis {
        agent_id: ZoooidId,
        reason: String,
    },
    /// Declare a logical group (visualization / hierarchy); does not change the graph by itself.
    GroupAgents(Vec<ZoooidId>),
}

#[async_trait]
pub trait TopologyRule: Send + Sync {
    async fn evaluate(
        &self,
        topology: &ZoooidTopology,
        metrics: &SystemMetrics,
    ) -> Result<Option<TopologyAction>, Box<dyn std::error::Error + Send + Sync>>;
}
