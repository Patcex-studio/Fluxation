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

//! Agent removal driven by homeostasis or resource limits.

use crate::rules::TopologyAction;
use crate::utils::types::ZoooidId;

use super::context::PlasticityContext;

/// Request apoptosis for agents below `min_activity` (optional `grace_iterations` not enforced here).
pub fn propose_low_activity_apoptosis(
    ctx: &PlasticityContext,
    min_activity: crate::utils::types::StateValue,
) -> Vec<TopologyAction> {
    let mut out = Vec::new();
    for (&id, &act) in &ctx.agent_activity {
        if act < min_activity {
            out.push(TopologyAction::InitiateApoptosis {
                agent_id: id,
                reason: "homeostatic_low_activity".to_string(),
            });
        }
    }
    out
}

/// Same as above but only for ids in `candidates` (e.g. leaf nodes).
pub fn propose_apoptosis_for_set(
    ctx: &PlasticityContext,
    candidates: &[ZoooidId],
    min_activity: crate::utils::types::StateValue,
) -> Vec<TopologyAction> {
    let set: std::collections::HashSet<ZoooidId> = candidates.iter().copied().collect();
    let mut out = Vec::new();
    for id in candidates {
        let act = ctx.agent_activity.get(id).copied().unwrap_or(0.0);
        if set.contains(id) && act < min_activity {
            out.push(TopologyAction::InitiateApoptosis {
                agent_id: *id,
                reason: "restricted_homeostasis".to_string(),
            });
        }
    }
    out
}
