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
use std::any::Any;
use std::collections::HashMap;

use crate::agent::state::RoleType;
use crate::core::message_bus::MessageBus;
use crate::core::topology::ZoooidTopology;
use crate::rules::behavior::{BehaviorAction, BehaviorRule};
use crate::utils::types::ZoooidId;

/// Simple load balancing rule that tracks message frequency
/// If an agent processes too many messages, it signals for optimization
pub struct LoadBalancingRule {
    pub load_threshold: f64,
    pub max_incoming_degree: usize,
}

impl LoadBalancingRule {
    pub fn new(load_threshold: f64, max_degree: usize) -> Self {
        Self {
            load_threshold,
            max_incoming_degree: max_degree,
        }
    }
}

#[async_trait]
impl BehaviorRule for LoadBalancingRule {
    async fn evaluate(
        &self,
        agent_id: ZoooidId,
        _state: &dyn Any,
        topology: &ZoooidTopology,
        _bus: &dyn MessageBus,
    ) -> Result<Option<BehaviorAction>, Box<dyn std::error::Error + Send + Sync>> {
        // Check if agent has too many incoming connections (degree > threshold)
        let incoming_degree = topology.get_neighbors(agent_id).len();

        if incoming_degree > self.max_incoming_degree {
            // Suggest change to a more efficient role or load-sharing behavior
            let mut params = HashMap::new();
            params.insert("load_state".to_string(), "high".to_string());
            params.insert(
                "requested_connections".to_string(),
                self.max_incoming_degree.to_string(),
            );

            return Ok(Some(BehaviorAction::ModifyParameters(params)));
        }

        Ok(None)
    }
}

/// Role adaptation rule: changes agent role based on network density
pub struct RoleAdaptationRule {
    pub high_density_threshold: f64, // avg connectivity threshold for dense network
    pub low_density_threshold: f64,  // threshold for sparse network
}

impl RoleAdaptationRule {
    pub fn new(high_threshold: f64, low_threshold: f64) -> Self {
        Self {
            high_density_threshold: high_threshold,
            low_density_threshold: low_threshold,
        }
    }
}

#[async_trait]
impl BehaviorRule for RoleAdaptationRule {
    async fn evaluate(
        &self,
        _agent_id: ZoooidId,
        _state: &dyn Any,
        topology: &ZoooidTopology,
        _bus: &dyn MessageBus,
    ) -> Result<Option<BehaviorAction>, Box<dyn std::error::Error + Send + Sync>> {
        let node_count = topology.graph.node_count();
        if node_count == 0 {
            return Ok(None);
        }

        let edge_count = topology.graph.edge_count();
        let avg_connectivity = (2.0 * edge_count as f64) / node_count as f64;

        // In a dense network, switch to cognitive role for better coordination
        if avg_connectivity > self.high_density_threshold {
            return Ok(Some(BehaviorAction::ChangeRole(RoleType::Cognitive)));
        }

        // In a sparse network, switch to sensor role for better coverage
        if avg_connectivity < self.low_density_threshold {
            return Ok(Some(BehaviorAction::ChangeRole(RoleType::Sensor)));
        }

        Ok(None)
    }
}

/// Isolation recovery rule: attempts to reconnect isolated or poorly connected agents
pub struct IsolationRecoveryRule {
    pub min_connectivity_threshold: usize,
}

impl IsolationRecoveryRule {
    pub fn new(min_threshold: usize) -> Self {
        Self {
            min_connectivity_threshold: min_threshold,
        }
    }
}

#[async_trait]
impl BehaviorRule for IsolationRecoveryRule {
    async fn evaluate(
        &self,
        agent_id: ZoooidId,
        _state: &dyn Any,
        topology: &ZoooidTopology,
        _bus: &dyn MessageBus,
    ) -> Result<Option<BehaviorAction>, Box<dyn std::error::Error + Send + Sync>> {
        let degree = topology.get_neighbors(agent_id).len();

        if degree < self.min_connectivity_threshold {
            // Signal need for connection repair
            let mut params = HashMap::new();
            params.insert(
                "isolation_state".to_string(),
                format!("low_connectivity_{}", degree),
            );
            params.insert(
                "target_connectivity".to_string(),
                self.min_connectivity_threshold.to_string(),
            );

            return Ok(Some(BehaviorAction::ModifyParameters(params)));
        }

        Ok(None)
    }
}
