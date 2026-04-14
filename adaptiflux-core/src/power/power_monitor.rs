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

use std::time::{Duration, Instant};

/// Lightweight metrics for power and activity estimation.
#[derive(Debug, Clone, Default)]
pub struct PowerMonitor {
    pub last_update: Option<Instant>,
    pub active_cycles: usize,
    pub idle_cycles: usize,
}

#[derive(Debug, Clone)]
pub struct PowerMetrics {
    pub active_cycles: usize,
    pub idle_cycles: usize,
    pub age: Option<Duration>,
}

impl PowerMonitor {
    pub fn record_cycle(&mut self, active: bool) {
        self.last_update = Some(Instant::now());
        if active {
            self.active_cycles += 1;
        } else {
            self.idle_cycles += 1;
        }
    }

    pub fn sample(&self) -> PowerMetrics {
        let age = self.last_update.map(|instant| instant.elapsed());
        PowerMetrics {
            active_cycles: self.active_cycles,
            idle_cycles: self.idle_cycles,
            age,
        }
    }
}
