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

use crate::core::topology::ZoooidTopology;
use crate::rules::consistency::{ConsistencyCheck, ConsistencyError};
use crate::utils::types::ZoooidId;
use std::collections::VecDeque;

/// Checks that the topology forms a connected graph (single component)
pub struct ConnectedTopologyCheck;

impl ConnectedTopologyCheck {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ConnectedTopologyCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsistencyCheck for ConnectedTopologyCheck {
    fn check(
        &self,
        topology: &ZoooidTopology,
        _agents: &[ZoooidId],
    ) -> Result<(), ConsistencyError> {
        if topology.graph.node_count() == 0 {
            return Ok(());
        }

        let nodes: Vec<_> = topology.graph.nodes().collect();
        if nodes.is_empty() {
            return Ok(());
        }

        // BFS from first node to count reachable nodes
        let mut visited = std::collections::HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(nodes[0]);
        visited.insert(nodes[0]);

        while let Some(node) = queue.pop_front() {
            for neighbor in topology.get_weak_neighbors(node) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        if visited.len() != nodes.len() {
            let components = Self::count_components(topology);
            return Err(ConsistencyError::DisconnectedComponent {
                component_count: components,
            });
        }

        Ok(())
    }
}

impl ConnectedTopologyCheck {
    fn count_components(topology: &ZoooidTopology) -> usize {
        let nodes: Vec<_> = topology.graph.nodes().collect();
        if nodes.is_empty() {
            return 0;
        }

        let mut visited = std::collections::HashSet::new();
        let mut components = 0;

        for node in nodes {
            if !visited.contains(&node) {
                components += 1;
                let mut queue = VecDeque::new();
                queue.push_back(node);
                visited.insert(node);

                while let Some(current) = queue.pop_front() {
                    for neighbor in topology.get_weak_neighbors(current) {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        components
    }
}

/// Checks that no nodes are isolated (all nodes have at least one connection)
pub struct NoIsolatedNodesCheck;

impl NoIsolatedNodesCheck {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoIsolatedNodesCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsistencyCheck for NoIsolatedNodesCheck {
    fn check(
        &self,
        topology: &ZoooidTopology,
        _agents: &[ZoooidId],
    ) -> Result<(), ConsistencyError> {
        for node in topology.graph.nodes() {
            if topology.get_weak_neighbors(node).is_empty() {
                return Err(ConsistencyError::IsolatedNode { node_id: node });
            }
        }
        Ok(())
    }
}

/// Checks that the topology node count matches the agent list
pub struct NodeCountConsistencyCheck;

impl NodeCountConsistencyCheck {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NodeCountConsistencyCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsistencyCheck for NodeCountConsistencyCheck {
    fn check(
        &self,
        topology: &ZoooidTopology,
        agents: &[ZoooidId],
    ) -> Result<(), ConsistencyError> {
        let topology_count = topology.graph.node_count();
        let agent_count = agents.len();

        if topology_count != agent_count {
            return Err(ConsistencyError::NodeCountMismatch {
                topology_count,
                agent_count,
            });
        }

        Ok(())
    }
}

/// Checks minimum connectivity requirement (average degree >= threshold)
pub struct MinConnectivityCheck {
    pub min_avg_connectivity: f64,
}

impl MinConnectivityCheck {
    pub fn new(min_connectivity: f64) -> Self {
        Self {
            min_avg_connectivity: min_connectivity,
        }
    }
}

impl ConsistencyCheck for MinConnectivityCheck {
    fn check(
        &self,
        topology: &ZoooidTopology,
        _agents: &[ZoooidId],
    ) -> Result<(), ConsistencyError> {
        let node_count = topology.graph.node_count();
        if node_count == 0 {
            return Ok(());
        }

        let edge_count = topology.graph.edge_count();
        let avg_connectivity = (2.0 * edge_count as f64) / node_count as f64;

        if avg_connectivity < self.min_avg_connectivity {
            return Err(ConsistencyError::MetricViolation {
                metric_name: "average_connectivity".to_string(),
                expected: self.min_avg_connectivity.to_string(),
                actual: avg_connectivity.to_string(),
            });
        }

        Ok(())
    }
}

/// Checks that all nodes meet minimum degree requirement
pub struct MinimumDegreeCheck {
    pub min_degree: usize,
}

impl MinimumDegreeCheck {
    pub fn new(min_degree: usize) -> Self {
        Self { min_degree }
    }
}

impl ConsistencyCheck for MinimumDegreeCheck {
    fn check(
        &self,
        topology: &ZoooidTopology,
        _agents: &[ZoooidId],
    ) -> Result<(), ConsistencyError> {
        for node in topology.graph.nodes() {
            let degree = topology.get_weak_neighbors(node).len();
            if degree < self.min_degree {
                return Err(ConsistencyError::InsufficientConnectivity {
                    node_id: node,
                    degree,
                    minimum: self.min_degree,
                });
            }
        }

        Ok(())
    }
}

/// Checks network diameter doesn't exceed threshold
pub struct MaxDiameterCheck {
    pub max_diameter: usize,
}

impl MaxDiameterCheck {
    pub fn new(max_diameter: usize) -> Self {
        Self { max_diameter }
    }
}

impl ConsistencyCheck for MaxDiameterCheck {
    fn check(
        &self,
        topology: &ZoooidTopology,
        _agents: &[ZoooidId],
    ) -> Result<(), ConsistencyError> {
        let diameter = Self::calculate_diameter(topology);

        if diameter > self.max_diameter {
            return Err(ConsistencyError::DiameterViolation {
                diameter,
                max_diameter: self.max_diameter,
            });
        }

        Ok(())
    }
}

impl MaxDiameterCheck {
    fn calculate_diameter(topology: &ZoooidTopology) -> usize {
        let nodes: Vec<_> = topology.graph.nodes().collect();
        if nodes.len() <= 1 {
            return 0;
        }

        let mut max_distance = 0;

        // Check distance from first node to all others
        for start in nodes.iter().take(1) {
            let mut visited = std::collections::HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back((*start, 0));
            visited.insert(*start);

            while let Some((node, dist)) = queue.pop_front() {
                max_distance = max_distance.max(dist);

                for neighbor in topology.get_neighbors(node) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back((neighbor, dist + 1));
                    }
                }
            }
        }

        max_distance
    }
}
