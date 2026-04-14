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

//! Prune weak or unused connections (synaptic / conductance metaphor).

use crate::core::topology::ZoooidTopology;
use crate::rules::{PruneReason, TopologyAction};
use crate::utils::types::StateValue;
use crate::utils::types::ZoooidId;

use super::context::PlasticityContext;

/// Propose `PruneEdge` for edges whose weight is below `min_weight`.
pub fn prune_low_conductivity_edges(
    ctx: &PlasticityContext,
    min_weight: StateValue,
) -> Vec<TopologyAction> {
    let mut out = Vec::new();
    for (&(from, to), &w) in &ctx.edge_weights {
        if w < min_weight {
            out.push(TopologyAction::PruneEdge {
                from,
                to,
                reason: PruneReason::LowConductivity,
            });
        }
    }
    out
}

/// Prune edges with no traffic for at least `idle_iterations` (if last-used is known).
pub fn prune_unused_edges(ctx: &PlasticityContext, idle_iterations: u64) -> Vec<TopologyAction> {
    let mut out = Vec::new();
    let now = ctx.iteration;
    for (&edge, &last) in &ctx.edge_last_used {
        if now.saturating_sub(last) >= idle_iterations
            && ctx.edge_traffic.get(&edge).copied().unwrap_or(0) == 0
        {
            let (from, to) = edge;
            out.push(TopologyAction::PruneEdge {
                from,
                to,
                reason: PruneReason::Unused,
            });
        }
    }
    out
}

/// Aggregate pruning: removes edges with very low traffic
/// (complements low-conductivity pruning for edges that might have ok weight but no usage).
pub fn prune_low_traffic_edges(ctx: &PlasticityContext, min_traffic: u64) -> Vec<TopologyAction> {
    let mut out = Vec::new();
    for (&edge, &traffic) in &ctx.edge_traffic {
        if traffic <= min_traffic {
            let (from, to) = edge;
            // Only prune if edge isn't very new (at least 10 iterations old)
            let age = edge_idle_iterations(ctx, from, to);
            if age > 10 {
                out.push(TopologyAction::PruneEdge {
                    from,
                    to,
                    reason: PruneReason::LowConductivity,
                });
            }
        }
    }
    out
}

/// Density-aware pruning: if topology density is above threshold,
/// aggressively prune the weakest edges to bring it down.
pub fn prune_excess_density_edges(
    topology: &ZoooidTopology,
    ctx: &PlasticityContext,
    density_threshold: f64,
    max_prune_per_iteration: usize,
) -> Vec<TopologyAction> {
    let density = topology.get_topology_density();

    if density <= density_threshold {
        return Vec::new();
    }

    // If density is high, prune weakest edges
    let mut edges_with_strength: Vec<_> = ctx
        .edge_weights
        .iter()
        .map(|(&edge, &weight)| (edge, weight))
        .collect();

    // Sort by weight (ascending) - weakest first
    edges_with_strength.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut out = Vec::new();
    for &((from, to), _weight) in edges_with_strength.iter().take(max_prune_per_iteration) {
        out.push(TopologyAction::PruneEdge {
            from,
            to,
            reason: PruneReason::LowConductivity,
        });
    }
    out
}

/// Combine pruning heuristics; avoids duplicate (from, to) pairs.
pub fn merge_prune_actions(actions: Vec<TopologyAction>) -> Vec<TopologyAction> {
    let mut seen = std::collections::HashSet::new();
    let mut merged = Vec::new();
    for a in actions {
        if let TopologyAction::PruneEdge { from, to, .. } = &a {
            if !seen.insert((*from, *to)) {
                continue;
            }
        }
        merged.push(a);
    }
    merged
}

/// Heuristic "age" of an edge for sorting / batch limits.
pub fn edge_idle_iterations(ctx: &PlasticityContext, from: ZoooidId, to: ZoooidId) -> u64 {
    let now = ctx.iteration;
    let last = ctx.edge_last_used.get(&(from, to)).copied().unwrap_or(0);
    now.saturating_sub(last)
}
