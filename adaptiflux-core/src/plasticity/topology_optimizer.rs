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

//! Experimental global hints: bridges, spanning shortcuts (graph-level).

use std::collections::{HashSet, VecDeque};

use crate::core::topology::ZoooidTopology;
use crate::rules::TopologyAction;
use crate::utils::types::ZoooidId;

/// Suggest one edge that would reduce BFS depth between two far-apart nodes (very small sample).
pub fn propose_diameter_bridge(
    topology: &ZoooidTopology,
    max_samples: usize,
) -> Option<TopologyAction> {
    let nodes: Vec<ZoooidId> = topology.graph.nodes().collect();
    if nodes.len() < 2 {
        return None;
    }

    let mut best_pair: Option<(ZoooidId, ZoooidId, usize)> = None;

    for (idx, &start) in nodes.iter().take(max_samples).enumerate() {
        let mut dist = std::collections::HashMap::new();
        let mut q = VecDeque::new();
        dist.insert(start, 0usize);
        q.push_back(start);

        while let Some(u) = q.pop_front() {
            let d = *dist.get(&u).unwrap_or(&0);
            for v in topology.get_neighbors(u) {
                if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(v) {
                    e.insert(d + 1);
                    q.push_back(v);
                }
            }
        }

        for &target in nodes.iter().skip(idx + 1) {
            if topology.graph.contains_edge(start, target)
                || topology.graph.contains_edge(target, start)
            {
                continue;
            }
            if let Some(&dt) = dist.get(&target) {
                if dt > 2 {
                    let improve = (start, target, dt);
                    match best_pair {
                        None => best_pair = Some(improve),
                        Some((_, _, best_dt)) if dt > best_dt => best_pair = Some(improve),
                        _ => {}
                    }
                }
            }
        }
    }

    best_pair.map(|(a, b, _)| TopologyAction::AddEdge(a, b))
}

/// Build a set of edges that form a greedy spanning tree (undirected view) to limit redundancy.
pub fn spanning_tree_edge_set(topology: &ZoooidTopology) -> HashSet<(ZoooidId, ZoooidId)> {
    let nodes: Vec<ZoooidId> = topology.graph.nodes().collect();
    let mut tree = HashSet::new();
    if nodes.is_empty() {
        return tree;
    }

    let mut visited = HashSet::new();
    let mut stack = vec![nodes[0]];
    visited.insert(nodes[0]);

    while let Some(u) = stack.pop() {
        for v in topology.get_neighbors(u) {
            if visited.insert(v) {
                let key = if u < v { (u, v) } else { (v, u) };
                tree.insert(key);
                stack.push(v);
            }
        }
    }

    tree
}
