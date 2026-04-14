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

//! Connectivity-based grouping over [`crate::core::topology::ZoooidTopology`].

use std::collections::{HashMap, HashSet, VecDeque};

use crate::core::topology::ZoooidTopology;
use crate::utils::types::ZoooidId;

/// Undirected connected components with at least `min_size` nodes.
pub fn detect_dense_groups(topology: &ZoooidTopology, min_size: usize) -> Vec<Vec<ZoooidId>> {
    let nodes: Vec<ZoooidId> = topology.graph.nodes().collect();
    if nodes.is_empty() {
        return Vec::new();
    }

    let mut adj: HashMap<ZoooidId, Vec<ZoooidId>> = HashMap::new();
    for &n in &nodes {
        adj.entry(n).or_default();
    }
    topology.for_each_edge(|a, b, _| {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    });

    let mut visited: HashSet<ZoooidId> = HashSet::new();
    let mut clusters = Vec::new();

    for &start in &nodes {
        if visited.contains(&start) {
            continue;
        }
        let mut q = VecDeque::new();
        q.push_back(start);
        visited.insert(start);
        let mut comp = Vec::new();
        while let Some(u) = q.pop_front() {
            comp.push(u);
            if let Some(neigh) = adj.get(&u) {
                for &v in neigh {
                    if visited.insert(v) {
                        q.push_back(v);
                    }
                }
            }
        }
        if comp.len() >= min_size {
            clusters.push(comp);
        }
    }

    clusters
}
