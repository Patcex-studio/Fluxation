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

use adaptiflux_core::core::topology::ZoooidTopology;
/// Baseline Scenario - Static Network without Adaptive Components
///
/// This example demonstrates:
/// - Static network with fixed routing and PID controllers
/// - No plasticity, learning, or cognitive components
/// - Packet simulation with failures to show baseline behavior
use adaptiflux_core::memory::types::MemoryPayload;
use adaptiflux_core::AsyncOptimizationConfig;
use adaptiflux_core::power::sleep_scheduler::SleepScheduler;
use adaptiflux_core::rules::consistency::{ConnectedTopologyCheck, MinConnectivityCheck};
use adaptiflux_core::{
    AgentBlueprint, AgentUpdateResult, CoreScheduler, LocalBus, Message, ResourceManager, RoleType,
    RuleEngine, Zoooid,
};
use serde_json;
use uuid::Uuid;
use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration, Instant};
use tracing::{debug, info};

/// Packet structure for simulation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Packet {
    id: u64,
    source: usize,
    destination: usize,
    hops: u8,
    created_at_ms: u64,
    max_hops: u8,
}

type NodeId = usize;

/// Node state for each network node
#[derive(Debug, Clone)]
struct NodeState {
    load: f64,
    active_packets: u32,
}

/// Static blueprint with fixed routing
struct StaticNetworkNodeBlueprint {
    node_id: NodeId,
    routing_table: Vec<NodeId>, // Fixed routing
}

#[derive(Debug, Clone)]
struct StaticNetworkNodeState {
    node_state: NodeState,
}


#[async_trait]
impl AgentBlueprint for StaticNetworkNodeBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(StaticNetworkNodeState {
            node_state: NodeState {
                load: 0.0,
                active_packets: 0,
            },
        }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<(adaptiflux_core::ZoooidId, Message)>,
        _topology: &adaptiflux_core::ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(node_state) = state.downcast_mut::<StaticNetworkNodeState>() {
            let mut output_messages = Vec::new();

            // Process packet messages
            for (_sender, input) in inputs {
                match input {
                    Message::Text(text) => {
                        if let Ok(packet) = serde_json::from_str::<Packet>(&text) {
                            node_state.node_state.active_packets += 1;
                            node_state.node_state.load += 1.0;

                            // Sensor detects load
                            let spike = if node_state.node_state.load > 2.0 { 1.0 } else { 0.0 };
                            if spike > 0.0 {
                                // Fixed routing - use routing table
                                if packet.hops < packet.max_hops
                                    && packet.destination != self.node_id
                                {
                                    let _next_hop = self.routing_table
                                        [packet.destination % self.routing_table.len()];
                                    let updated_packet = Packet {
                                        hops: packet.hops + 1,
                                        ..packet
                                    };
                                    output_messages.push(Message::Text(
                                        serde_json::to_string(&updated_packet).unwrap(),
                                    ));
                                } else {
                                    // Packet delivered or expired
                                    node_state.node_state.active_packets -= 1;
                                    node_state.node_state.load -= 1.0;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Decay load
            node_state.node_state.load *= 0.95;

            Ok(AgentUpdateResult::new(output_messages, None, None, false))
        } else {
            Err("Invalid state type".into())
        }
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Custom("Baseline".to_string())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    info!("Starting Baseline Scenario");

    let num_nodes = 50;
    let mut scheduler = create_baseline_scheduler(num_nodes).await?;
    let packet_interval = interval(Duration::from_millis(100));

    tokio::pin!(packet_interval);

    let start_time = Instant::now();
    let mut packet_id = 0;

    loop {
        tokio::select! {
            _ = packet_interval.tick() => {
                let source = (packet_id as usize) % num_nodes;
                let destination = ((packet_id as usize) * 7 + 1) % num_nodes;
                if source != destination {
                    let packet = Packet {
                        id: packet_id,
                        source,
                        destination,
                        hops: 0,
                        created_at_ms: start_time.elapsed().as_millis() as u64,
                        max_hops: 10,
                    };
                    packet_id += 1;

                    let source_id = Uuid::from_u128(source as u128);
                    let target_id = Uuid::from_u128(destination as u128);
                    let message = Message::Text(serde_json::to_string(&packet).unwrap());
                    scheduler.message_bus.send(source_id, target_id, message).await?;
                }
            }
            _ = tokio::time::sleep(Duration::from_secs(1)) => {
                scheduler.run_one_iteration().await?;
                debug!("Baseline scheduler tick completed");
            }
        }

        let elapsed = start_time.elapsed();
        if elapsed > Duration::from_secs(30) && elapsed < Duration::from_secs(31) {
            info!("Baseline Event: Single node failure at T=30s");
            scheduler.agents.remove(&Uuid::from_u128(42));
        }

        if elapsed > Duration::from_secs(90) && elapsed < Duration::from_secs(91) {
            info!("Baseline Event: Repeat failure at T=90s");
        }

        if elapsed > Duration::from_secs(150) && elapsed < Duration::from_secs(151) {
            info!("Baseline Event: Mass failure at T=150s");
            let to_remove: Vec<_> = scheduler
                .agents
                .keys()
                .take((num_nodes * 3) / 10)
                .cloned()
                .collect();
            for id in to_remove {
                scheduler.agents.remove(&id);
            }
        }

        if elapsed > Duration::from_secs(300) {
            break;
        }
    }

    info!("Baseline Scenario completed");
    Ok(())
}

async fn create_baseline_scheduler(
    num_nodes: usize,
) -> Result<CoreScheduler, Box<dyn std::error::Error + Send + Sync>> {
    let topology = Arc::new(RwLock::new(ZoooidTopology::new()));
    let mut rule_engine = RuleEngine::new();
    rule_engine.add_consistency_check(Box::new(ConnectedTopologyCheck));
    rule_engine.add_consistency_check(Box::new(MinConnectivityCheck::new(1.0)));

    let resource_manager = ResourceManager::new();
    let message_bus = Arc::new(LocalBus::new());

    let mut scheduler = CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

    scheduler.async_optimization = Some(AsyncOptimizationConfig::new(4));
    scheduler.sleep_scheduler = Some(SleepScheduler::new(Duration::from_secs(30)));

    // Create agents with fixed routing
    for i in 0..num_nodes {
        let routing_table = (0..num_nodes).collect::<Vec<_>>();
        let blueprint = StaticNetworkNodeBlueprint {
            node_id: i,
            routing_table,
        };
        let zoooid = Zoooid::new(Uuid::from_u128(i as u128), Box::new(blueprint)).await?;
        scheduler.spawn_agent(zoooid).await?;
    }

    Ok(scheduler)
}
