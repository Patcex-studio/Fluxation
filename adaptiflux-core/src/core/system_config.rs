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

use std::sync::OnceLock;

use tracing::warn;

pub const ENV_MAX_DEGREE_PER_AGENT: &str = "ADAPTIFLUX_MAX_DEGREE_PER_AGENT";
pub const ENV_SCHEDULER_CYCLE_MS: &str = "ADAPTIFLUX_SCHEDULER_CYCLE_MS";
pub const ENV_MAX_CONCURRENT_AGENT_UPDATES: &str = "ADAPTIFLUX_MAX_CONCURRENT_AGENT_UPDATES";

const DEFAULT_MAX_DEGREE_PER_AGENT: usize = 50;
const DEFAULT_SCHEDULER_CYCLE_MS: u64 = 100;
const DEFAULT_MAX_CONCURRENT_AGENT_UPDATES: usize = 0; // 0 means auto-detect via num_cpus

const MAX_ALLOWED_DEGREE_PER_AGENT: usize = 10_000;
const MAX_ALLOWED_SCHEDULER_CYCLE_MS: u64 = 60_000;
const MAX_ALLOWED_CONCURRENT_UPDATES: usize = 8_192;

#[derive(Debug, Clone)]
pub struct SystemConfig {
    pub max_degree_per_agent: usize,
    pub scheduler_cycle_ms: u64,
    pub max_concurrent_agent_updates: usize,
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            max_degree_per_agent: DEFAULT_MAX_DEGREE_PER_AGENT,
            scheduler_cycle_ms: DEFAULT_SCHEDULER_CYCLE_MS,
            max_concurrent_agent_updates: std::cmp::max(
                1,
                if DEFAULT_MAX_CONCURRENT_AGENT_UPDATES == 0 {
                    num_cpus::get()
                } else {
                    DEFAULT_MAX_CONCURRENT_AGENT_UPDATES
                },
            ),
        }
    }
}

impl SystemConfig {
    pub fn from_env() -> Self {
        let defaults = Self::default();

        let max_degree_per_agent = parse_env_usize(
            ENV_MAX_DEGREE_PER_AGENT,
            defaults.max_degree_per_agent,
            1,
            MAX_ALLOWED_DEGREE_PER_AGENT,
        );

        let scheduler_cycle_ms = parse_env_u64(
            ENV_SCHEDULER_CYCLE_MS,
            defaults.scheduler_cycle_ms,
            1,
            MAX_ALLOWED_SCHEDULER_CYCLE_MS,
        );

        let max_concurrent_agent_updates = parse_env_usize(
            ENV_MAX_CONCURRENT_AGENT_UPDATES,
            defaults.max_concurrent_agent_updates,
            1,
            MAX_ALLOWED_CONCURRENT_UPDATES,
        );

        Self {
            max_degree_per_agent,
            scheduler_cycle_ms,
            max_concurrent_agent_updates,
        }
    }

    pub fn global() -> &'static Self {
        static INSTANCE: OnceLock<SystemConfig> = OnceLock::new();
        INSTANCE.get_or_init(Self::from_env)
    }
}

fn parse_env_usize(key: &str, default: usize, min: usize, max: usize) -> usize {
    match std::env::var(key) {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(value) if value >= min && value <= max => value,
            Ok(value) => {
                warn!(
                    "Environment variable {}={} is outside allowed range [{}..={}], using default {}",
                    key, value, min, max, default
                );
                default
            }
            Err(_) => {
                warn!(
                    "Environment variable {}='{}' is not a valid usize, using default {}",
                    key, raw, default
                );
                default
            }
        },
        Err(_) => default,
    }
}

fn parse_env_u64(key: &str, default: u64, min: u64, max: u64) -> u64 {
    match std::env::var(key) {
        Ok(raw) => match raw.parse::<u64>() {
            Ok(value) if value >= min && value <= max => value,
            Ok(value) => {
                warn!(
                    "Environment variable {}={} is outside allowed range [{}..={}], using default {}",
                    key, value, min, max, default
                );
                default
            }
            Err(_) => {
                warn!(
                    "Environment variable {}='{}' is not a valid u64, using default {}",
                    key, raw, default
                );
                default
            }
        },
        Err(_) => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        let cfg = SystemConfig::default();
        assert!(cfg.max_degree_per_agent >= 1);
        assert!(cfg.scheduler_cycle_ms >= 1);
        assert!(cfg.max_concurrent_agent_updates >= 1);
    }

    #[test]
    fn from_env_uses_defaults_for_invalid_values() {
        std::env::set_var(ENV_MAX_DEGREE_PER_AGENT, "0");
        std::env::set_var(ENV_SCHEDULER_CYCLE_MS, "999999");
        std::env::set_var(ENV_MAX_CONCURRENT_AGENT_UPDATES, "invalid");

        let defaults = SystemConfig::default();
        let cfg = SystemConfig::from_env();

        assert_eq!(cfg.max_degree_per_agent, defaults.max_degree_per_agent);
        assert_eq!(cfg.scheduler_cycle_ms, defaults.scheduler_cycle_ms);
        assert_eq!(
            cfg.max_concurrent_agent_updates,
            defaults.max_concurrent_agent_updates
        );

        std::env::remove_var(ENV_MAX_DEGREE_PER_AGENT);
        std::env::remove_var(ENV_SCHEDULER_CYCLE_MS);
        std::env::remove_var(ENV_MAX_CONCURRENT_AGENT_UPDATES);
    }
}
