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

use petgraph::graphmap::DiGraphMap;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::debug;

use crate::utils::types::{StateValue, ZoooidId};

const MAX_DEGREE_PER_AGENT: usize = 50;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConnectionProperties {
    pub weight: StateValue,
    pub bandwidth: Option<StateValue>,
    pub latency: Option<u64>,
}

impl Default for ConnectionProperties {
    fn default() -> Self {
        Self {
            weight: 1.0,
            bandwidth: None,
            latency: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZoooidTopology {
    pub graph: DiGraphMap<ZoooidId, ConnectionProperties>,
    /// Optional event bus for topology changes (for synapse synchronization)
    /// If Some, events are published when edges change; if None, no events (backward compat)
    pub event_bus: Option<Arc<crate::core::TopologyEventBus>>,
}

impl ZoooidTopology {
    pub fn new() -> Self {
        Self {
            graph: DiGraphMap::new(),
            event_bus: None,
        }
    }

    /// Creates a new topology with event bus for synapse synchronization.
    pub fn with_event_bus(event_bus: Arc<crate::core::TopologyEventBus>) -> Self {
        Self {
            graph: DiGraphMap::new(),
            event_bus: Some(event_bus),
        }
    }

    /// Sets the event bus for topology change notifications.
    pub fn set_event_bus(&mut self, event_bus: Arc<crate::core::TopologyEventBus>) {
        self.event_bus = Some(event_bus);
    }

    /// Try to add an edge; respects MAX_DEGREE_PER_AGENT constraint.
    /// Returns true if edge was added, false if degree limit would be exceeded.
    pub fn try_add_edge(
        &mut self,
        from: ZoooidId,
        to: ZoooidId,
        props: ConnectionProperties,
    ) -> bool {
        // Check if edge already exists
        if self.graph.contains_edge(from, to) {
            return false;
        }

        // Check outgoing degree from the source
        let from_outgoing = self
            .graph
            .neighbors_directed(from, Direction::Outgoing)
            .count();
        if from_outgoing >= MAX_DEGREE_PER_AGENT {
            return false;
        }

        // Check incoming degree to the target
        let to_incoming = self
            .graph
            .neighbors_directed(to, Direction::Incoming)
            .count();
        if to_incoming >= MAX_DEGREE_PER_AGENT {
            return false;
        }

        self.graph.add_edge(from, to, props.clone());

        // Publish event if event bus is available
        if let Some(bus) = &self.event_bus {
            let event = crate::core::TopologyEvent::EdgeAdded {
                from,
                to,
                initial_weight: props.weight,
            };
            let _ = bus.publish(event);
        }

        true
    }

    /// Legacy method for backward compatibility: adds edge without degree check.
    /// Use try_add_edge() for plasticity rules to respect topology limits.
    pub fn add_edge(&mut self, from: ZoooidId, to: ZoooidId, props: ConnectionProperties) {
        self.graph.add_edge(from, to, props.clone());

        // Publish event if event bus is available
        if let Some(bus) = &self.event_bus {
            let event = crate::core::TopologyEvent::EdgeAdded {
                from,
                to,
                initial_weight: props.weight,
            };
            let _ = bus.publish(event);
        }
    }

    /// Removes an edge and publishes event if event bus is configured.
    /// Returns the removed edge properties if it existed.
    pub fn remove_edge(&mut self, from: ZoooidId, to: ZoooidId) -> Option<ConnectionProperties> {
        let removed = self.graph.remove_edge(from, to);

        // Publish event if event bus is available and edge was removed
        if removed.is_some() {
            if let Some(bus) = &self.event_bus {
                let event = crate::core::TopologyEvent::EdgeRemoved { from, to };
                let _ = bus.publish(event);
                debug!("Published EdgeRemoved event: {} -> {}", from, to);
            }
        }

        removed
    }

    pub fn get_neighbors(&self, id: ZoooidId) -> Vec<ZoooidId> {
        self.graph.neighbors(id).collect()
    }

    pub fn get_outgoing_neighbors(&self, id: ZoooidId) -> Vec<ZoooidId> {
        self.graph
            .neighbors_directed(id, Direction::Outgoing)
            .collect()
    }

    pub fn get_weak_neighbors(&self, id: ZoooidId) -> Vec<ZoooidId> {
        let mut neighbors = std::collections::HashSet::new();
        for neighbor in self.graph.neighbors_directed(id, Direction::Outgoing) {
            neighbors.insert(neighbor);
        }
        for neighbor in self.graph.neighbors_directed(id, Direction::Incoming) {
            neighbors.insert(neighbor);
        }
        neighbors.into_iter().collect()
    }

    pub fn get_connection_properties(
        &self,
        from: ZoooidId,
        to: ZoooidId,
    ) -> Option<&ConnectionProperties> {
        self.graph.edge_weight(from, to)
    }

    pub fn get_random_neighbor(&self, id: ZoooidId) -> Option<ZoooidId> {
        self.get_neighbors(id).into_iter().next()
    }

    /// Get the outgoing degree (number of connections FROM this agent)
    pub fn get_outgoing_degree(&self, id: ZoooidId) -> usize {
        self.graph
            .neighbors_directed(id, Direction::Outgoing)
            .count()
    }

    /// Get the incoming degree (number of connections TO this agent)
    pub fn get_incoming_degree(&self, id: ZoooidId) -> usize {
        self.graph
            .neighbors_directed(id, Direction::Incoming)
            .count()
    }

    /// Get the total degree (both incoming and outgoing, treating edges as undirected for counting)
    pub fn get_total_degree(&self, id: ZoooidId) -> usize {
        let mut neighbors = std::collections::HashSet::new();
        for n in self.graph.neighbors_directed(id, Direction::Outgoing) {
            neighbors.insert(n);
        }
        for n in self.graph.neighbors_directed(id, Direction::Incoming) {
            neighbors.insert(n);
        }
        neighbors.len()
    }

    /// Check if an agent is near its degree capacity (>= 80% of MAX_DEGREE_PER_AGENT)
    pub fn is_near_degree_capacity(&self, id: ZoooidId) -> bool {
        let outgoing = self.get_outgoing_degree(id);
        let incoming = self.get_incoming_degree(id);
        let threshold = (MAX_DEGREE_PER_AGENT * 80) / 100; // 80% threshold
        outgoing >= threshold || incoming >= threshold
    }

    /// Get the MAX_DEGREE_PER_AGENT constant (for external use)
    pub fn max_degree_per_agent() -> usize {
        MAX_DEGREE_PER_AGENT
    }

    pub fn add_node(&mut self, id: ZoooidId) {
        self.graph.add_node(id);
    }

    pub fn remove_node(&mut self, id: ZoooidId) {
        self.graph.remove_node(id);
    }

    /// Adjust connection weight in place (clamped to a small positive range).
    pub fn adjust_edge_weight(&mut self, from: ZoooidId, to: ZoooidId, delta: StateValue) -> bool {
        if let Some(props) = self.graph.edge_weight_mut(from, to) {
            props.weight = (props.weight + delta).clamp(0.01, 1_000.0);
            return true;
        }
        false
    }

    /// Iterate all directed edges with connection properties.
    pub fn for_each_edge<F>(&self, mut f: F)
    where
        F: FnMut(ZoooidId, ZoooidId, &ConnectionProperties),
    {
        for (a, b, props) in self.graph.all_edges() {
            f(a, b, props);
        }
    }

    /// Calculate the global topology density: (total_edges) / (num_agents * MAX_DEGREE_PER_AGENT)
    pub fn get_topology_density(&self) -> f64 {
        let num_agents = self.graph.node_count() as f64;
        let total_edges = self.graph.edge_count() as f64;

        if num_agents == 0.0 {
            return 0.0;
        }

        let max_possible_edges = num_agents * MAX_DEGREE_PER_AGENT as f64;
        total_edges / max_possible_edges
    }
}

impl Default for ZoooidTopology {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub total_zoooids: usize,
    pub total_connections: usize,
    pub avg_connectivity: f64,
    pub clustering_coefficient: f64,
    pub network_diameter: usize,
    pub agent_count: usize, // For backward compatibility
}

impl SystemMetrics {
    pub fn from_topology(topology: &ZoooidTopology) -> Self {
        let node_count = topology.graph.node_count();
        let edge_count = topology.graph.edge_count();

        let avg_connectivity = if node_count > 0 {
            (2.0 * edge_count as f64) / node_count as f64
        } else {
            0.0
        };

        // Calculate clustering coefficient (simplified version)
        let clustering_coefficient = Self::calculate_clustering_coefficient(topology);

        // Calculate network diameter (simplified - max distance between nodes)
        let network_diameter = Self::calculate_network_diameter(topology);

        Self {
            total_zoooids: node_count,
            total_connections: edge_count,
            avg_connectivity,
            clustering_coefficient,
            network_diameter,
            agent_count: node_count,
        }
    }

    fn calculate_clustering_coefficient(topology: &ZoooidTopology) -> f64 {
        let nodes = topology.graph.nodes().collect::<Vec<_>>();
        if nodes.is_empty() {
            return 0.0;
        }

        let node_count = nodes.len();
        let mut total_cc = 0.0;
        for node in &nodes {
            let neighbors = topology.get_neighbors(*node);
            if neighbors.len() < 2 {
                continue;
            }

            let mut edges_between_neighbors = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    if topology.graph.contains_edge(neighbors[i], neighbors[j])
                        || topology.graph.contains_edge(neighbors[j], neighbors[i])
                    {
                        edges_between_neighbors += 1;
                    }
                }
            }

            let max_edges = neighbors.len() * (neighbors.len() - 1) / 2;
            if max_edges > 0 {
                total_cc += edges_between_neighbors as f64 / max_edges as f64;
            }
        }

        if node_count == 0 {
            0.0
        } else {
            total_cc / node_count as f64
        }
    }

    fn calculate_network_diameter(topology: &ZoooidTopology) -> usize {
        // Simplified diameter calculation - just count the depth from first node
        let nodes: Vec<_> = topology.graph.nodes().collect();
        if nodes.is_empty() {
            return 0;
        }

        let mut max_depth = 0;
        for start_node in nodes.iter().take(3) {
            // Sample first 3 nodes
            let mut visited = std::collections::HashSet::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((*start_node, 0));
            visited.insert(*start_node);

            while let Some((node, depth)) = queue.pop_front() {
                max_depth = max_depth.max(depth);

                for neighbor in topology.get_neighbors(node) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        max_depth
    }
}

#[derive(Debug, Clone)]
pub enum TopologyChange {
    RequestConnection(ZoooidId),
    RemoveConnection(ZoooidId, ZoooidId),
}
