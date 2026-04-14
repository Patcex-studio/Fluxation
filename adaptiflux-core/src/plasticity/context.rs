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

//! Runtime signals for structural plasticity (activity, traffic, recency).

use std::collections::HashMap;

use crate::agent::blueprint::AgentBlueprint;
use crate::core::topology::{ConnectionProperties, ZoooidTopology};
use crate::utils::types::ZoooidId;

/// Runtime state tracking signals for structural plasticity rules
///
/// Maintains per-iteration counters and activity metrics that drive topology adaptation.
/// Updated by the scheduler during agent execution and used by plasticity rules to decide
/// on structural changes like pruning, synaptogenesis, and apoptosis.
///
/// # Activity Tracking
///
/// - **Agent Activity**: Exponential moving average of message traffic per agent
/// - **Edge Traffic**: Directed edge usage counts per iteration
/// - **Edge Recency**: Last iteration when edges were active
///
/// # Usage
///
/// The scheduler updates this state during each iteration, then plasticity rules
/// use snapshots via `PlasticityContext` to evaluate adaptation decisions.
#[derive(Debug, Clone, Default)]
pub struct PlasticityRuntimeState {
    /// Global iteration counter for the entire system
    pub global_iteration: u64,
    /// Scalar activity per agent (outputs + inputs this tick, exponential moving average)
    pub agent_activity: HashMap<ZoooidId, crate::utils::types::StateValue>,
    /// Directed edge traffic counts (from → to) for current iteration
    pub edge_traffic: HashMap<(ZoooidId, ZoooidId), u64>,
    /// Last iteration index when an edge carried traffic
    pub edge_last_used: HashMap<(ZoooidId, ZoooidId), u64>,
}

impl PlasticityRuntimeState {
    /// Create a read-only snapshot for plasticity rules and reset per-iteration counters
    ///
    /// Generates a `PlasticityContext` with current state and clears edge traffic counters
    /// for the next iteration. This ensures rules see a consistent snapshot while allowing
    /// accumulation of new traffic data.
    ///
    /// # Arguments
    ///
    /// * `topology` - Current system topology to include edge weights
    ///
    /// # Returns
    ///
    /// A `PlasticityContext` snapshot for rule evaluation
    pub fn snapshot_plasticity_context(&mut self, topology: &ZoooidTopology) -> PlasticityContext {
        let ctx = PlasticityContext::from_topology_and_runtime(topology, self);
        self.edge_traffic.clear();
        ctx
    }

    /// Record activity pulse for an agent
    ///
    /// Increments the agent's activity counter by the specified amount.
    /// Used by the scheduler to track agent engagement.
    ///
    /// # Arguments
    ///
    /// * `id` - Agent identifier
    /// * `amount` - Activity amount to add
    pub fn record_agent_pulse(&mut self, id: ZoooidId, amount: crate::utils::types::StateValue) {
        let e = self.agent_activity.entry(id).or_insert(0.0);
        *e += amount;
    }

    pub fn record_edge_use(&mut self, from: ZoooidId, to: ZoooidId) {
        *self.edge_traffic.entry((from, to)).or_insert(0) += 1;
        self.edge_last_used
            .insert((from, to), self.global_iteration);
    }

    /// Exponential moving average decay so old bursts fade.
    pub fn decay_activity(&mut self, factor: crate::utils::types::StateValue) {
        for v in self.agent_activity.values_mut() {
            *v *= factor;
        }
    }

    pub fn advance_iteration(&mut self) {
        self.global_iteration = self.global_iteration.saturating_add(1);
    }
}

/// Snapshot passed to plasticity rules (read-only view).
#[derive(Debug, Clone)]
pub struct PlasticityContext {
    pub iteration: u64,
    pub agent_activity: HashMap<ZoooidId, crate::utils::types::StateValue>,
    pub edge_traffic: HashMap<(ZoooidId, ZoooidId), u64>,
    pub edge_last_used: HashMap<(ZoooidId, ZoooidId), u64>,
    /// Copy of directed edge weights from topology at evaluation time.
    pub edge_weights: HashMap<(ZoooidId, ZoooidId), crate::utils::types::StateValue>,
}

impl PlasticityContext {
    pub fn from_topology_and_runtime(
        topology: &ZoooidTopology,
        runtime: &PlasticityRuntimeState,
    ) -> Self {
        let mut edge_weights = HashMap::new();
        let mut edge_last_used = runtime.edge_last_used.clone();
        topology.for_each_edge(|a, b, props: &ConnectionProperties| {
            edge_weights.insert((a, b), props.weight);
            edge_last_used
                .entry((a, b))
                .or_insert(runtime.global_iteration);
        });

        Self {
            iteration: runtime.global_iteration,
            agent_activity: runtime.agent_activity.clone(),
            edge_traffic: runtime.edge_traffic.clone(),
            edge_last_used,
            edge_weights,
        }
    }
}

/// Result of applying structural actions that the scheduler must execute (spawns / kills).
#[derive(Default)]
pub struct AppliedTopologyEffects {
    pub edge_operations: usize,
    /// Directed edges created in this batch (for plasticity last-use timestamps).
    pub new_edges: Vec<(ZoooidId, ZoooidId)>,
    pub spawn_requests: Vec<(Option<ZoooidId>, Box<dyn AgentBlueprint + Send + Sync>)>,
    pub terminate_requests: Vec<(ZoooidId, Option<String>)>,
    /// Cluster hints from [`crate::rules::TopologyAction::GroupAgents`].
    pub agent_groups: Vec<Vec<ZoooidId>>,
}
