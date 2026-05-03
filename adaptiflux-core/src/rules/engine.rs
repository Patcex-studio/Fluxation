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
use std::sync::Arc;

use crate::core::message_bus::MessageBus;
use crate::core::topology::{SystemMetrics, ZoooidTopology};
use crate::plasticity::{AppliedTopologyEffects, PlasticityContext};
use crate::rules::behavior::{BehaviorAction, BehaviorRule};
use crate::rules::consistency::ConsistencyCheck;
use crate::rules::topology::{TopologyAction, TopologyRule};
use crate::rules::PlasticityRule;
use crate::utils::types::ZoooidId;
use tokio::sync::RwLock;

/// Main rule engine that orchestrates behavior rules, topology rules, and consistency checks
pub struct RuleEngine {
    pub behavior_rules: Vec<Box<dyn BehaviorRule>>,
    pub topology_rules: Vec<Box<dyn TopologyRule>>,
    pub plasticity_rules: Vec<Box<dyn PlasticityRule>>,
    pub consistency_checks: Vec<Box<dyn ConsistencyCheck>>,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            behavior_rules: Vec::new(),
            topology_rules: Vec::new(),
            plasticity_rules: Vec::new(),
            consistency_checks: Vec::new(),
        }
    }

    /// Add a behavior rule to the engine
    pub fn add_behavior_rule(&mut self, rule: Box<dyn BehaviorRule>) {
        self.behavior_rules.push(rule);
    }

    /// Add a topology rule to the engine
    pub fn add_topology_rule(&mut self, rule: Box<dyn TopologyRule>) {
        self.topology_rules.push(rule);
    }

    /// Add a structural plasticity rule (evaluated after classic topology rules).
    pub fn add_plasticity_rule(&mut self, rule: Box<dyn PlasticityRule>) {
        self.plasticity_rules.push(rule);
    }

    /// Add a consistency check to the engine
    pub fn add_consistency_check(&mut self, check: Box<dyn ConsistencyCheck>) {
        self.consistency_checks.push(check);
    }

    /// Evaluate behavior rules for a single agent
    pub async fn evaluate_behavior_for_agent(
        &self,
        agent_id: ZoooidId,
        state: &dyn Any,
        topology: &ZoooidTopology,
        bus: &dyn MessageBus,
    ) -> Result<Vec<BehaviorAction>, Box<dyn std::error::Error + Send + Sync>> {
        let mut actions = Vec::new();

        for rule in &self.behavior_rules {
            if let Some(action) = rule.evaluate(agent_id, state, topology, bus).await? {
                actions.push(action);
            }
        }

        Ok(actions)
    }

    /// Evaluate behavior rules for all agents
    pub async fn run_behavior_rules(
        &self,
        agent_states: &std::collections::HashMap<ZoooidId, Box<dyn Any + Send>>,
        topology: &ZoooidTopology,
        bus: &dyn MessageBus,
    ) -> Result<
        std::collections::HashMap<ZoooidId, Vec<BehaviorAction>>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let mut result = std::collections::HashMap::new();

        for (agent_id, state) in agent_states {
            let actions = self
                .evaluate_behavior_for_agent(*agent_id, state.as_ref(), topology, bus)
                .await?;

            if !actions.is_empty() {
                result.insert(*agent_id, actions);
            }
        }

        Ok(result)
    }

    /// Run topology rules to generate topology modifications
    pub async fn run_topology_rules(
        &self,
        topology: &Arc<RwLock<ZoooidTopology>>,
        metrics: &SystemMetrics,
    ) -> Result<Vec<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        let topology = topology.read().await;
        let mut actions = Vec::new();

        for rule in &self.topology_rules {
            if let Some(action) = rule.evaluate(&topology, metrics).await? {
                actions.push(action);
            }
        }

        Ok(actions)
    }

    /// Structural plasticity pass: uses per-agent / per-edge runtime context.
    /// Respects global topology density threshold to prevent exponential growth.
    pub async fn run_plasticity_rules(
        &self,
        topology: &Arc<RwLock<ZoooidTopology>>,
        metrics: &SystemMetrics,
        ctx: &PlasticityContext,
    ) -> Result<Vec<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        let topology = topology.read().await;
        let mut actions = Vec::new();

        // Check global topology density - if too high, skip growth-oriented rules
        let current_density = topology.get_topology_density();
        let density_threshold = 0.2; // 20% density threshold
        let skip_growth = current_density >= density_threshold;

        if skip_growth {
            tracing::debug!(
                density = current_density,
                threshold = density_threshold,
                "Global density threshold exceeded; skipping synaptogenesis"
            );
        }

        for rule in &self.plasticity_rules {
            // Check if this is a growth-oriented rule and if we should skip it
            let rule_name = std::any::type_name_of_val(rule.as_ref());
            let is_growth_rule =
                rule_name.contains("Synaptogenesis") || rule_name.contains("Neurogenesis");

            if skip_growth && is_growth_rule {
                continue; // Skip growth rules when density is high
            }

            let batch = rule.evaluate(&topology, metrics, ctx).await?;
            actions.extend(batch);
        }

        Ok(actions)
    }

    /// Apply topology actions: mutates graph weights and edges; defers agent lifecycle to scheduler.
    pub async fn apply_topology_actions(
        &self,
        topology_mutex: &Arc<RwLock<ZoooidTopology>>,
        actions: Vec<TopologyAction>,
    ) -> Result<AppliedTopologyEffects, Box<dyn std::error::Error + Send + Sync>> {
        let mut topology = topology_mutex.write().await;
        let mut effects = AppliedTopologyEffects::default();

        for action in actions {
            match action {
                TopologyAction::AddEdge(source, target) => {
                    // Use try_add_edge to respect MAX_DEGREE_PER_AGENT constraints
                    if topology.try_add_edge(source, target, Default::default()) {
                        effects.new_edges.push((source, target));
                        effects.edge_operations += 1;
                    }
                    // If edge is rejected due to degree limit, it's silently dropped
                    // (normal behavior for plasticity rules)
                }
                TopologyAction::RemoveEdge(source, target) => {
                    topology.remove_edge(source, target);
                    effects.edge_operations += 1;
                }
                TopologyAction::PruneEdge { from, to, .. } => {
                    topology.remove_edge(from, to);
                    effects.edge_operations += 1;
                }
                TopologyAction::StrengthenConnection {
                    from,
                    to,
                    delta_weight,
                } => {
                    if topology.adjust_edge_weight(from, to, delta_weight) {
                        effects.edge_operations += 1;
                    }
                }
                TopologyAction::WeakenConnection {
                    from,
                    to,
                    delta_weight,
                } => {
                    if topology.adjust_edge_weight(from, to, -delta_weight) {
                        effects.edge_operations += 1;
                    }
                }
                TopologyAction::CreateZoooid(blueprint) => {
                    effects.spawn_requests.push((None, blueprint));
                }
                TopologyAction::CreateAgentFromTemplate {
                    template_blueprint,
                    target_area_hint,
                } => {
                    effects
                        .spawn_requests
                        .push((target_area_hint, template_blueprint));
                }
                TopologyAction::TerminateZoooid(id) => {
                    effects.terminate_requests.push((id, None));
                }
                TopologyAction::InitiateApoptosis { agent_id, reason } => {
                    effects.terminate_requests.push((agent_id, Some(reason)));
                }
                TopologyAction::GroupAgents(ids) => {
                    if ids.len() >= 2 {
                        effects.agent_groups.push(ids);
                    }
                }
            }
        }

        Ok(effects)
    }

    /// Run all consistency checks
    pub async fn run_consistency_checks(
        &self,
        topology: &Arc<RwLock<ZoooidTopology>>,
        agents: &[ZoooidId],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let topology = topology.read().await;

        for check in &self.consistency_checks {
            check.check(&topology, agents)?;
        }

        Ok(())
    }

    /// Complete rule cycle: behavior -> topology -> plasticity -> consistency
    pub async fn run_full_cycle(
        &self,
        agent_states: &std::collections::HashMap<ZoooidId, Box<dyn Any + Send>>,
        topology_mutex: &Arc<RwLock<ZoooidTopology>>,
        bus: &dyn MessageBus,
        agent_ids: &[ZoooidId],
        plasticity_ctx: &PlasticityContext,
    ) -> Result<
        (
            std::collections::HashMap<ZoooidId, Vec<BehaviorAction>>,
            usize,
        ),
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let topology = topology_mutex.read().await;

        // 1. Run behavior rules
        let behavior_actions = self
            .run_behavior_rules_internal(agent_states, &topology, bus)
            .await?;

        drop(topology);

        // 2. Compute metrics and run topology rules
        let topology_locked = topology_mutex.read().await;
        let metrics = SystemMetrics::from_topology(&topology_locked);
        drop(topology_locked);

        let mut topology_actions = self.run_topology_rules(topology_mutex, &metrics).await?;
        let plasticity_actions = self
            .run_plasticity_rules(topology_mutex, &metrics, plasticity_ctx)
            .await?;
        topology_actions.extend(plasticity_actions);

        // 3. Apply topology changes
        let effects = self
            .apply_topology_actions(topology_mutex, topology_actions)
            .await?;

        // 4. Run consistency checks
        self.run_consistency_checks(topology_mutex, agent_ids)
            .await?;

        Ok((behavior_actions, effects.edge_operations))
    }

    /// Internal method to run behavior rules with locked topology
    async fn run_behavior_rules_internal(
        &self,
        agent_states: &std::collections::HashMap<ZoooidId, Box<dyn Any + Send>>,
        topology: &ZoooidTopology,
        bus: &dyn MessageBus,
    ) -> Result<
        std::collections::HashMap<ZoooidId, Vec<BehaviorAction>>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let mut result = std::collections::HashMap::new();

        for (agent_id, state) in agent_states {
            let mut actions = Vec::new();

            for rule in &self.behavior_rules {
                if let Some(action) = rule
                    .evaluate(*agent_id, state.as_ref(), topology, bus)
                    .await?
                {
                    actions.push(action);
                }
            }

            if !actions.is_empty() {
                result.insert(*agent_id, actions);
            }
        }

        Ok(result)
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}
