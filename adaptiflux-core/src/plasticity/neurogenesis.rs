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

//! Spawning new agents from templates (experimental).

use std::sync::Arc;

use crate::agent::blueprint::AgentBlueprint;
use crate::rules::TopologyAction;
use crate::utils::types::{StateValue, ZoooidId};

use super::context::PlasticityContext;

pub type BlueprintFactory = Arc<dyn Fn() -> Box<dyn AgentBlueprint + Send + Sync> + Send + Sync>;

/// If any agent exceeds `activity_threshold`, emit one spawn near the hottest id.
pub fn propose_growth_from_hotspots(
    ctx: &PlasticityContext,
    activity_threshold: StateValue,
    factory: &BlueprintFactory,
) -> Vec<TopologyAction> {
    let mut best: Option<(ZoooidId, StateValue)> = None;
    for (&id, &act) in &ctx.agent_activity {
        if act >= activity_threshold {
            match best {
                None => best = Some((id, act)),
                Some((_, best_act)) if act > best_act => best = Some((id, act)),
                _ => {}
            }
        }
    }

    let Some((hot_id, _)) = best else {
        return vec![];
    };

    vec![TopologyAction::CreateAgentFromTemplate {
        template_blueprint: factory(),
        target_area_hint: Some(hot_id),
    }]
}
