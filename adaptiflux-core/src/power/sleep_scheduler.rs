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

use crate::utils::types::ZoooidId;
use std::collections::HashMap;
use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Registry for agents that should be kept idle until activity resumes.
#[derive(Debug)]
pub struct SleepScheduler {
    pub inactivity_threshold: Duration,
    last_activity: HashMap<ZoooidId, Instant>,
}

impl SleepScheduler {
    pub fn new(inactivity_threshold: Duration) -> Self {
        Self {
            inactivity_threshold,
            last_activity: HashMap::new(),
        }
    }

    pub fn record_activity(&mut self, agent_id: ZoooidId) {
        self.last_activity.insert(agent_id, Instant::now());
    }

    pub fn record_idle(&mut self, agent_id: ZoooidId) {
        self.last_activity
            .entry(agent_id)
            .or_insert_with(Instant::now);
    }

    pub fn should_sleep(&self, agent_id: ZoooidId) -> bool {
        self.last_activity
            .get(&agent_id)
            .map(|last| last.elapsed() >= self.inactivity_threshold)
            .unwrap_or(false)
    }

    pub fn prune_missing_agents(&mut self, active_agent_ids: impl IntoIterator<Item = ZoooidId>) {
        let keep: HashSet<_> = active_agent_ids.into_iter().collect();
        self.last_activity.retain(|id, _| keep.contains(id));
    }
}
