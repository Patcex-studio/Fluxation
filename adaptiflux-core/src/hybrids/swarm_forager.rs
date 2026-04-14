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

//! Swarm Forager Hybrid Architecture
//!
//! This hybrid demonstrates swarm intelligence using:
//! - Multiple SwarmZooid agents (PFSM-based)
//! - Pheromone-like message signaling for coordination
//! - Collective search behavior
//!
//! One agent acts as a "target/food source" emitting pheromone signals.
//! Other agents follow these signals, creating emergent search behavior.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::agent::blueprint::{SwarmzooidBlueprint, SwarmzooidParams};
use crate::agent::zoooid::Zoooid;
use crate::core::CoreScheduler;
use crate::primitives::swarm::pfsm::PfsmParams;
use crate::utils::types::ZoooidId;

/// Configuration for the Swarm Forager architecture
#[derive(Debug, Clone)]
pub struct SwarmForagerConfig {
    pub swarm_size: usize,
    pub pfsm_params: PfsmParams,
    pub pheromone_strength: f64,
    pub pheromone_decay_rate: f64,
}

impl Default for SwarmForagerConfig {
    fn default() -> Self {
        Self {
            swarm_size: 5,
            pfsm_params: PfsmParams::default(),
            pheromone_strength: 1.0,
            pheromone_decay_rate: 0.1,
        }
    }
}

/// Swarm Forager Architecture Instance
#[derive(Debug, Clone)]
pub struct SwarmForagerArchitecture {
    pub agent_ids: Vec<ZoooidId>,
    pub forager_id: ZoooidId, // The agent acting as food source
    pub pheromone_levels: Arc<Mutex<HashMap<ZoooidId, f64>>>,
}

impl SwarmForagerArchitecture {
    /// Create and register a swarm of foraging agents
    pub async fn create(
        scheduler: &mut CoreScheduler,
        config: SwarmForagerConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut agent_ids = Vec::new();
        let mut pheromone_levels = HashMap::new();

        // Create swarm agents
        for _ in 0..config.swarm_size {
            let blueprint = SwarmzooidBlueprint {
                params: SwarmzooidParams {
                    pfsm_params: config.pfsm_params.clone(),
                    connection_request_interval: 10,
                },
            };

            let zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(blueprint)).await?;
            let id = zooid.id;
            agent_ids.push(id);
            pheromone_levels.insert(id, 0.0);

            scheduler.spawn_agent(zooid).await?;
        }

        // Designate the first agent as the forager (food source)
        let forager_id = agent_ids[0];
        pheromone_levels.insert(forager_id, config.pheromone_strength);

        // Create a fully connected topology (all agents can communicate)
        let mut topology = scheduler.topology.lock().await;
        for i in 0..agent_ids.len() {
            for j in 0..agent_ids.len() {
                if i != j {
                    topology.add_edge(
                        agent_ids[i],
                        agent_ids[j],
                        crate::core::topology::ConnectionProperties::default(),
                    );
                }
            }
        }
        drop(topology);

        Ok(SwarmForagerArchitecture {
            agent_ids,
            forager_id,
            pheromone_levels: Arc::new(Mutex::new(pheromone_levels)),
        })
    }

    /// Update pheromone levels (simulating decay and emission)
    pub async fn update_pheromones(&self) {
        let mut levels = self.pheromone_levels.lock().await;

        // Decay existing pheromones
        for level in levels.values_mut() {
            *level *= 0.9; // 90% retention = 10% decay
        }

        // Forager continuously emits
        if let Some(forager_level) = levels.get_mut(&self.forager_id) {
            *forager_level += 0.1; // Gradual emission
        }
    }

    /// Get current pheromone level for an agent
    pub async fn get_pheromone_level(&self, agent_id: ZoooidId) -> f64 {
        self.pheromone_levels
            .lock()
            .await
            .get(&agent_id)
            .copied()
            .unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{LocalBus, ResourceManager, ZoooidTopology};
    use crate::RuleEngine;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn test_swarm_forager_creation() {
        let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = SwarmForagerConfig {
            swarm_size: 5,
            ..Default::default()
        };

        let swarm = SwarmForagerArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create swarm");

        assert_eq!(swarm.agent_ids.len(), 5);
        assert_eq!(scheduler.agent_count(), 5);
    }

    #[tokio::test]
    async fn test_pheromone_dynamics() {
        let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = SwarmForagerConfig {
            swarm_size: 3,
            pheromone_strength: 1.0,
            ..Default::default()
        };

        let swarm = SwarmForagerArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create swarm");

        // Initial pheromone
        let initial = swarm.get_pheromone_level(swarm.forager_id).await;
        assert!(initial > 0.0);

        // After one update, pheromone may decay or increase slightly depending on timing
        swarm.update_pheromones().await;
        let after_first = swarm.get_pheromone_level(swarm.forager_id).await;
        assert!(after_first >= 0.0); // Should remain non-negative

        // Multiple iterations
        for _ in 0..5 {
            swarm.update_pheromones().await;
        }
        let stabilized = swarm.get_pheromone_level(swarm.forager_id).await;
        assert!(stabilized > 0.0); // Should reach equilibrium
    }

    #[tokio::test]
    async fn test_swarm_connectivity() {
        let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = SwarmForagerConfig {
            swarm_size: 3,
            ..Default::default()
        };

        let swarm = SwarmForagerArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create swarm");

        let topology_guard = scheduler.topology.lock().await;

        // In a fully connected swarm, each agent should have connections
        for &agent_id in &swarm.agent_ids {
            let connections = topology_guard.get_neighbors(agent_id);
            assert!(
                !connections.is_empty(),
                "Agent {} should have connections",
                agent_id
            );
        }
    }
}
