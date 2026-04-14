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
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum ConsistencyError {
    #[error("Disconnected component found: {component_count} components")]
    DisconnectedComponent { component_count: usize },

    #[error("Isolated node: {node_id}")]
    IsolatedNode { node_id: ZoooidId },

    #[error("Cycle detected in directed topology")]
    CycleDetected,

    #[error("Metric violation: {metric_name}, expected: {expected}, actual: {actual}")]
    MetricViolation {
        metric_name: String,
        expected: String,
        actual: String,
    },

    #[error("Node count mismatch: topology has {topology_count}, agents have {agent_count}")]
    NodeCountMismatch {
        topology_count: usize,
        agent_count: usize,
    },

    #[error("Insufficient connectivity: node {node_id} has degree {degree}, minimum required: {minimum}")]
    InsufficientConnectivity {
        node_id: ZoooidId,
        degree: usize,
        minimum: usize,
    },

    #[error("Network diameter exceeds threshold: {diameter}, maximum allowed: {max_diameter}")]
    DiameterViolation {
        diameter: usize,
        max_diameter: usize,
    },

    #[error("Custom consistency check failed: {reason}")]
    Custom { reason: String },
}

pub trait ConsistencyCheck: Send + Sync {
    fn check(
        &self,
        topology: &crate::core::topology::ZoooidTopology,
        agents: &[ZoooidId],
    ) -> Result<(), ConsistencyError>;
}
