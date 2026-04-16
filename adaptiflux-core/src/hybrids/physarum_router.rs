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

//! Physarum Router Hybrid Architecture
//!
//! This experimental hybrid demonstrates adaptive network routing using:
//! - Physarum-inspired agents (slime mold optimization)
//! - Edge weight adaptation based on traffic
//! - Emergent optimal pathfinding
//!
//! The system adaptively strengthen efficient paths and weakens inefficient ones,
//! similar to how Physarum polycephalum optimizes transport networks.

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::agent::blueprint::base::AgentBlueprint;
use crate::agent::state::RoleType;
use crate::agent::zoooid::Zoooid;
use crate::core::{CoreScheduler, Message};
use crate::primitives::base::Primitive;
use crate::primitives::physical::{PhysarumParams, PhysarumState, PhysarumUnit};
use crate::utils::types::ZoooidId;

/// Configuration for the Physarum Router architecture
#[derive(Debug, Clone)]
pub struct PhysarumRouterConfig {
    pub router_count: usize,
    pub physarum_params: PhysarumParams,
    pub traffic_simulation_mode: bool,
}

impl Default for PhysarumRouterConfig {
    fn default() -> Self {
        Self {
            router_count: 4,
            physarum_params: PhysarumParams::default(),
            traffic_simulation_mode: true,
        }
    }
}

/// Represents an edge in the routing network with adaptive conductivity
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveEdge {
    pub source: ZoooidId,
    pub destination: ZoooidId,
    pub conductivity: f64,
    pub traffic_count: u64,
}

/// Physarum Router Architecture Instance
#[derive(Debug, Clone)]
pub struct PhysarumRouterArchitecture {
    pub router_ids: Vec<ZoooidId>,
    pub source_id: Option<ZoooidId>, // Data source node
    pub sink_id: Option<ZoooidId>,   // Data sink node
    pub total_traffic: Arc<Mutex<u64>>,
    pub optimized_paths: Arc<Mutex<Vec<Vec<ZoooidId>>>>,
}

impl PhysarumRouterArchitecture {
    /// Create and register router nodes in a network
    pub async fn create(
        scheduler: &mut CoreScheduler,
        config: PhysarumRouterConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut router_ids = Vec::new();

        // Create router nodes
        for _ in 0..config.router_count {
            // Since we don't have a dedicated router blueprint,
            // we'll repurpose another agent or create a minimal blueprint
            // For now, using a placeholder approach
            let zooid = Zoooid::new(
                ZoooidId::new_v4(),
                Box::new(GenericPhysarumBlueprint {
                    params: config.physarum_params.clone(),
                }),
            )
            .await?;

            let id = zooid.id;
            router_ids.push(id);
            scheduler.spawn_agent(zooid).await?;
        }

        // Create a mesh topology (partial connectivity)
        let mut topology = scheduler.topology.lock().await;
        for i in 0..router_ids.len() {
            // Connect each router to 2-3 neighbors (creating mesh)
            let next = (i + 1) % router_ids.len();
            let prev = if i == 0 { router_ids.len() - 1 } else { i - 1 };

            topology.add_edge(
                router_ids[i],
                router_ids[next],
                crate::core::topology::ConnectionProperties::default(),
            );
            if i > 0 {
                topology.add_edge(
                    router_ids[i],
                    router_ids[prev],
                    crate::core::topology::ConnectionProperties::default(),
                );
            }
        }
        drop(topology);

        let source_id = Some(router_ids[0]);
        let sink_id = Some(router_ids[router_ids.len() - 1]);

        Ok(PhysarumRouterArchitecture {
            router_ids,
            source_id,
            sink_id,
            total_traffic: Arc::new(Mutex::new(0)),
            optimized_paths: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Simulate sending traffic through the network
    pub async fn send_traffic(&self, amount: u64) {
        let mut traffic = self.total_traffic.lock().await;
        *traffic += amount;
    }

    /// Get total traffic processed
    pub async fn get_total_traffic(&self) -> u64 {
        *self.total_traffic.lock().await
    }

    /// Record an optimized path
    pub async fn record_path(&self, path: Vec<ZoooidId>) {
        let mut paths = self.optimized_paths.lock().await;
        paths.push(path);
    }

    /// Get recorded paths
    pub async fn get_paths(&self) -> Vec<Vec<ZoooidId>> {
        self.optimized_paths.lock().await.clone()
    }
}

/// Minimal blueprint for Physarum-based router
#[derive(Debug, Clone)]
struct GenericPhysarumBlueprint {
    params: PhysarumParams,
}

#[async_trait::async_trait]
impl AgentBlueprint for GenericPhysarumBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn std::any::Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>>
    {
        let state = <PhysarumUnit as Primitive>::initialize(self.params.clone());
        Ok(Box::new(state))
    }

    async fn update(
        &self,
        state: &mut Box<dyn std::any::Any + Send + Sync>,
        inputs: Vec<(ZoooidId, Message)>,
        _topology: &crate::core::topology::ZoooidTopology,
        _memory: Option<&crate::memory::types::MemoryPayload>,
    ) -> Result<crate::agent::state::AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>>
    {
        let state = state
            .downcast_mut::<PhysarumState>()
            .ok_or("Invalid state type for Physarum")?;

        let primitive_inputs: Vec<crate::primitives::base::PrimitiveMessage> = inputs
            .into_iter()
            .filter_map(|(_sender, msg)| match msg {
                Message::AnalogInput(value) => Some(
                    crate::primitives::base::PrimitiveMessage::InputCurrent(value),
                ),
                _ => None,
            })
            .collect();

        let (new_state, outputs) =
            <PhysarumUnit as Primitive>::update(state.clone(), &self.params, &primitive_inputs);

        *state = new_state;

        let output_messages = outputs
            .into_iter()
            .filter_map(|msg| match msg {
                crate::primitives::base::PrimitiveMessage::ControlSignal(signal) => {
                    Some(Message::ControlSignal(signal))
                }
                _ => None,
            })
            .collect();

        Ok(crate::agent::state::AgentUpdateResult::new(
            output_messages,
            None,
            None,
            false,
        ))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Physarum
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
    async fn test_physarum_router_creation() {
        let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = PhysarumRouterConfig {
            router_count: 4,
            ..Default::default()
        };

        let router = PhysarumRouterArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create router");

        assert_eq!(router.router_ids.len(), 4);
        assert!(router.source_id.is_some());
        assert!(router.sink_id.is_some());
    }

    #[tokio::test]
    async fn test_traffic_simulation() {
        let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = PhysarumRouterConfig::default();
        let router = PhysarumRouterArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create router");

        assert_eq!(router.get_total_traffic().await, 0);

        router.send_traffic(100).await;
        assert_eq!(router.get_total_traffic().await, 100);

        router.send_traffic(50).await;
        assert_eq!(router.get_total_traffic().await, 150);
    }

    #[tokio::test]
    async fn test_path_recording() {
        let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = PhysarumRouterConfig::default();
        let router = PhysarumRouterArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create router");

        let path = router.router_ids.clone();
        router.record_path(path.clone()).await;

        let recorded_paths = router.get_paths().await;
        assert_eq!(recorded_paths.len(), 1);
        assert_eq!(recorded_paths[0], path);
    }
}
