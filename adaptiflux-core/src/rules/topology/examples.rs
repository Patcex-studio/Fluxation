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

use crate::core::topology::{SystemMetrics, ZoooidTopology};
use crate::rules::{TopologyAction, TopologyRule};

/// Simple topology growth rule: connects nodes based on degree threshold
pub struct SimpleTopologyGrowthRule {
    pub max_degree: usize,
}

impl SimpleTopologyGrowthRule {
    pub fn new(max_degree: usize) -> Self {
        Self { max_degree }
    }
}

#[async_trait]
impl TopologyRule for SimpleTopologyGrowthRule {
    async fn evaluate(
        &self,
        topology: &ZoooidTopology,
        metrics: &SystemMetrics,
    ) -> Result<Option<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        if metrics.total_zoooids < 2 {
            return Ok(None);
        }

        let nodes: Vec<_> = topology.graph.nodes().collect();
        if nodes.len() < 2 {
            return Ok(None);
        }

        // Find first node with degree below threshold
        for source in &nodes {
            if topology.get_neighbors(*source).len() < self.max_degree {
                // Try to find a node it's not already connected to
                for target in &nodes {
                    if source != target
                        && !topology.graph.contains_edge(*source, *target)
                        && !topology.graph.contains_edge(*target, *source)
                    {
                        return Ok(Some(TopologyAction::AddEdge(*source, *target)));
                    }
                }
            }
        }

        Ok(None)
    }
}

/// Proximity-based connection rule: maintains local cluster connectivity
pub struct ProximityConnectionRule {
    pub max_distance_threshold: f64,
    pub min_interaction_freq: usize,
}

impl ProximityConnectionRule {
    pub fn new(max_distance: f64, min_freq: usize) -> Self {
        Self {
            max_distance_threshold: max_distance,
            min_interaction_freq: min_freq,
        }
    }
}

#[async_trait]
impl TopologyRule for ProximityConnectionRule {
    async fn evaluate(
        &self,
        _topology: &ZoooidTopology,
        _metrics: &SystemMetrics,
    ) -> Result<Option<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        // This rule would require external data about spatial proximity
        // For now, it serves as a template for proximity-based topology modifications
        // In a full implementation, proximity data would come from agent states or external service

        // Pseudocode behavior:
        // - Get spatial positions of agents from external source
        // - For agents within max_distance:
        //   - Count recent interactions (message history from bus)
        //   - If interaction_freq >= min_interaction_freq and no edge exists:
        //     - Add edge between agents

        Ok(None)
    }
}

/// Self-healing rule: removes connections to dead/unresponsive agents
pub struct SelfHealingRule {
    pub heartbeat_timeout_secs: u64,
}

impl SelfHealingRule {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            heartbeat_timeout_secs: timeout_secs,
        }
    }
}

#[async_trait]
impl TopologyRule for SelfHealingRule {
    async fn evaluate(
        &self,
        _topology: &ZoooidTopology,
        _metrics: &SystemMetrics,
    ) -> Result<Option<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        // This rule would require external monitoring of agent health
        // In a full implementation:
        // - Query external health monitor for dead agents
        // - For each dead agent: emit TerminateZoooid action
        // - Remove all edges connected to dead agents

        // Placeholder: returns None
        Ok(None)
    }
}

/// Diameter optimization rule: adds shortcuts to reduce network diameter
pub struct DiameterOptimizationRule {
    pub max_allowed_diameter: usize,
    pub connection_probability: f64, // For adding random shortcuts
}

impl DiameterOptimizationRule {
    pub fn new(max_diameter: usize, connection_prob: f64) -> Self {
        Self {
            max_allowed_diameter: max_diameter,
            connection_probability: connection_prob,
        }
    }
}

#[async_trait]
impl TopologyRule for DiameterOptimizationRule {
    async fn evaluate(
        &self,
        topology: &ZoooidTopology,
        metrics: &SystemMetrics,
    ) -> Result<Option<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        // If network diameter exceeds threshold, add random long-distance connections
        if metrics.network_diameter > self.max_allowed_diameter {
            let nodes: Vec<_> = topology.graph.nodes().collect();
            if nodes.len() >= 2 {
                // Simple heuristic: connect node to a distant (non-neighbor) node
                if let Some(&source) = nodes.first() {
                    for &target in nodes.iter().skip(nodes.len() / 2) {
                        if !topology.graph.contains_edge(source, target)
                            && !topology.graph.contains_edge(target, source)
                        {
                            return Ok(Some(TopologyAction::AddEdge(source, target)));
                        }
                    }
                }
            }
        }

        Ok(None)
    }
}

/// Clustering coefficient optimization: increases local connectivity
pub struct LocalClusteringRule {
    pub target_clustering_coefficient: f64,
}

impl LocalClusteringRule {
    pub fn new(target_cc: f64) -> Self {
        Self {
            target_clustering_coefficient: target_cc,
        }
    }
}

#[async_trait]
impl TopologyRule for LocalClusteringRule {
    async fn evaluate(
        &self,
        topology: &ZoooidTopology,
        metrics: &SystemMetrics,
    ) -> Result<Option<TopologyAction>, Box<dyn std::error::Error + Send + Sync>> {
        // If clustering coefficient is too low, increase local connectivity
        if metrics.clustering_coefficient < self.target_clustering_coefficient {
            let nodes: Vec<_> = topology.graph.nodes().collect();

            // Find a node with neighbors that aren't connected to each other
            for node in nodes {
                let neighbors = topology.get_neighbors(node);
                if neighbors.len() >= 2 {
                    // Try to connect neighbors of this node
                    for i in 0..neighbors.len() {
                        for j in (i + 1)..neighbors.len() {
                            if !topology.graph.contains_edge(neighbors[i], neighbors[j])
                                && !topology.graph.contains_edge(neighbors[j], neighbors[i])
                            {
                                return Ok(Some(TopologyAction::AddEdge(
                                    neighbors[i],
                                    neighbors[j],
                                )));
                            }
                        }
                    }
                }
            }
        }

        Ok(None)
    }
}
