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

//! Activity-dependent creation of new synapses (edges).

use std::collections::HashSet;

use crate::core::topology::ZoooidTopology;
use crate::rules::TopologyAction;
use crate::utils::types::{StateValue, ZoooidId};

use super::context::PlasticityContext;

/// Connect highly active agents that are not yet linked (directed A→B once).
/// Respects MAX_DEGREE_PER_AGENT limit on both source and target agents.
pub fn propose_activity_synapses(
    topology: &ZoooidTopology,
    ctx: &PlasticityContext,
    activity_threshold: crate::utils::types::StateValue,
    max_new_edges: usize,
) -> Vec<TopologyAction> {
    let nodes: Vec<ZoooidId> = topology.graph.nodes().collect();
    let mut active: Vec<ZoooidId> = nodes
        .iter()
        .copied()
        .filter(|id| ctx.agent_activity.get(id).copied().unwrap_or(0.0) >= activity_threshold)
        .collect();

    active.sort_by(|a, b| {
        let va = ctx.agent_activity.get(a).copied().unwrap_or(0.0);
        let vb = ctx.agent_activity.get(b).copied().unwrap_or(0.0);
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = Vec::new();
    let mut seen = HashSet::new();

    'outer: for i in 0..active.len() {
        let a = active[i];

        // Skip agents that are already near their degree capacity
        if topology.is_near_degree_capacity(a) {
            continue;
        }

        for &b in active.iter().skip(i + 1) {
            // Skip if edge already exists in either direction
            if topology.graph.contains_edge(a, b) || topology.graph.contains_edge(b, a) {
                continue;
            }

            let key = (a, b);
            if !seen.insert(key) {
                continue;
            }

            // Check if target is near capacity
            if topology.is_near_degree_capacity(b) {
                continue;
            }

            // Prefer direction higher activity → lower as a simple bias
            let from = a;
            let to = b;

            // Double-check degrees are within limits before proposing
            if topology.get_outgoing_degree(from) < ZoooidTopology::max_degree_per_agent()
                && topology.get_incoming_degree(to) < ZoooidTopology::max_degree_per_agent()
            {
                out.push(TopologyAction::AddEdge(from, to));
            }

            if out.len() >= max_new_edges {
                break 'outer;
            }
        }
    }

    out
}

/// STDP-style strengthening: co-active endpoints of a busy edge get weight bumps (as actions).
pub fn stdp_reinforce_hot_edges(
    ctx: &PlasticityContext,
    traffic_threshold: u64,
    delta_weight: StateValue,
) -> Vec<TopologyAction> {
    let mut out = Vec::new();
    for (&(from, to), &count) in &ctx.edge_traffic {
        if count >= traffic_threshold {
            out.push(TopologyAction::StrengthenConnection {
                from,
                to,
                delta_weight,
            });
        }
    }
    out
}
