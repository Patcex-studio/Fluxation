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

//! Structural plasticity rules: compose pruning, growth, homeostasis.

use async_trait::async_trait;

use crate::core::topology::{SystemMetrics, ZoooidTopology};
use crate::hierarchy::cluster_detection::detect_dense_groups;
use crate::plasticity::apoptosis::propose_low_activity_apoptosis;
use crate::plasticity::context::PlasticityContext;
use crate::plasticity::neurogenesis::{propose_growth_from_hotspots, BlueprintFactory};
use crate::plasticity::pruning::{
    merge_prune_actions, prune_excess_density_edges, prune_low_conductivity_edges,
    prune_low_traffic_edges, prune_unused_edges,
};
use crate::plasticity::synaptogenesis::{propose_activity_synapses, stdp_reinforce_hot_edges};
use crate::rules::TopologyAction;
use crate::utils::types::StateValue;

/// Rules that may emit **multiple** topology actions per evaluation cycle.
#[async_trait]
pub trait PlasticityRule: Send + Sync {
    async fn evaluate(
        &self,
        topology: &ZoooidTopology,
        metrics: &SystemMetrics,
        ctx: &PlasticityContext,
    ) -> Result<Vec<TopologyAction>, Box<dyn std::error::Error + Send + Sync>>;
}

/// Remove edges with low conductance (weight) or prolonged idleness.
/// Uses aggressive pruning strategies to limit topology growth.
pub struct SynapticPruningRule {
    pub min_weight: StateValue,
    pub idle_prune_after: Option<u64>,
    /// Target maximum density: if exceeded, aggressively prune weakest edges
    pub target_density: Option<f64>,
    /// Maximum edges to prune per iteration for density control
    pub max_prune_per_iter: usize,
}

#[async_trait]
impl PlasticityRule for SynapticPruningRule {
    async fn evaluate(
        &self,
        topology: &ZoooidTopology,
        _metrics: &SystemMetrics,
        ctx: &PlasticityContext,
    ) -> Result<Vec<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        let mut v = prune_low_conductivity_edges(ctx, self.min_weight);

        // Add low-traffic pruning to remove unused edges
        v.extend(prune_low_traffic_edges(ctx, 0));

        if let Some(idle) = self.idle_prune_after {
            v.extend(prune_unused_edges(ctx, idle));
        }

        // Add density-aware pruning if topology is getting too dense
        if let Some(target_density) = self.target_density {
            let density_prune =
                prune_excess_density_edges(topology, ctx, target_density, self.max_prune_per_iter);
            v.extend(density_prune);
        }

        Ok(merge_prune_actions(v))
    }
}

/// Add edges between highly active agents; optionally reinforce busy edges (STDP-like).
pub struct ActivityDependentSynaptogenesisRule {
    pub activity_threshold: StateValue,
    pub max_new_edges: usize,
    pub stdp_traffic_threshold: Option<u64>,
    pub stdp_delta: StateValue,
}

#[async_trait]
impl PlasticityRule for ActivityDependentSynaptogenesisRule {
    async fn evaluate(
        &self,
        topology: &ZoooidTopology,
        _metrics: &SystemMetrics,
        ctx: &PlasticityContext,
    ) -> Result<Vec<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        let mut out =
            propose_activity_synapses(topology, ctx, self.activity_threshold, self.max_new_edges);
        if let Some(th) = self.stdp_traffic_threshold {
            out.extend(stdp_reinforce_hot_edges(ctx, th, self.stdp_delta));
        }
        Ok(out)
    }
}

/// Remove agents with persistently low activity (requires scheduler to honor actions).
pub struct HomeostaticApoptosisRule {
    pub min_activity: StateValue,
}

#[async_trait]
impl PlasticityRule for HomeostaticApoptosisRule {
    async fn evaluate(
        &self,
        _topology: &ZoooidTopology,
        _metrics: &SystemMetrics,
        ctx: &PlasticityContext,
    ) -> Result<Vec<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(propose_low_activity_apoptosis(ctx, self.min_activity))
    }
}

/// Spawn a new agent when activity concentrates in a hotspot.
pub struct GrowthFactorNeurogenesisRule {
    pub activity_threshold: StateValue,
    pub blueprint_factory: BlueprintFactory,
}

#[async_trait]
impl PlasticityRule for GrowthFactorNeurogenesisRule {
    async fn evaluate(
        &self,
        _topology: &ZoooidTopology,
        _metrics: &SystemMetrics,
        ctx: &PlasticityContext,
    ) -> Result<Vec<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(propose_growth_from_hotspots(
            ctx,
            self.activity_threshold,
            &self.blueprint_factory,
        ))
    }
}

/// Periodically tags dense connected components as [`TopologyAction::GroupAgents`] for hierarchy layers.
pub struct ClusterGroupingPlasticityRule {
    pub min_cluster_size: usize,
    pub evaluate_every: u64,
}

#[async_trait]
impl PlasticityRule for ClusterGroupingPlasticityRule {
    async fn evaluate(
        &self,
        topology: &ZoooidTopology,
        _metrics: &SystemMetrics,
        ctx: &PlasticityContext,
    ) -> Result<Vec<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        if self.evaluate_every == 0 || !ctx.iteration.is_multiple_of(self.evaluate_every) {
            return Ok(Vec::new());
        }
        let clusters = detect_dense_groups(topology, self.min_cluster_size);
        Ok(clusters
            .into_iter()
            .map(TopologyAction::GroupAgents)
            .collect())
    }
}
