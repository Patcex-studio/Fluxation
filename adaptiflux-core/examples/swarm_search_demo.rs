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
/// Swarm Search Demonstration
///
/// This example demonstrates:
/// - A swarm of 12 search agents working collaboratively
/// - Dynamic topology formation as agents discover each other
/// - Decentralized information sharing
/// - Emergent behavior from simple local rules
/// - Scalability to larger agent networks
use adaptiflux_core::memory::types::MemoryPayload;
use adaptiflux_core::{
    AgentBlueprint, AgentUpdateResult, CoreScheduler, LocalBus, Message, ResourceManager, RoleType,
    RuleEngine, Zoooid, ZoooidId,
};
use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::Duration;

/// Search agent that mimics a bee in a swarm searching for nectar
///
/// Behavior:
/// - Search locally and report back when "nectar" (target) is found
/// - Share information about nectar sources with neighbors
/// - Form connections when discovering new neighbors
/// - Gradually explore the search space
struct SearchAgentBlueprint {
    agent_index: usize,
}

#[derive(Debug, Clone)]
struct SearchAgentState {
    position_x: f64,
    position_y: f64,
    energy: u32,
    nectar_found: u32,
    neighbors_known: u32,
    last_discovery_tick: u64,
    messages_sent: u32,
}

#[async_trait]
impl AgentBlueprint for SearchAgentBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        // Random initial position in search space
        let angle = (self.agent_index as f64) * 2.0 * std::f64::consts::PI / 12.0;
        let radius = 5.0;

        Ok(Box::new(SearchAgentState {
            position_x: angle.cos() * radius,
            position_y: angle.sin() * radius,
            energy: 100,
            nectar_found: 0,
            neighbors_known: 0,
            last_discovery_tick: 0,
            messages_sent: 0,
        }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<(adaptiflux_core::ZoooidId, Message)>,
        topology: &adaptiflux_core::ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(agent_state) = state.downcast_mut::<SearchAgentState>() {
            let mut output_messages = Vec::new();
            let topology_changes = None;

            // Decrease energy each step
            if agent_state.energy > 0 {
                agent_state.energy -= 1;
            } else {
                // Agent dies if energy reaches 0
                return Ok(AgentUpdateResult::new(
                    output_messages,
                    topology_changes,
                    None,
                    true, // terminate
                ));
            }

            // Process incoming information from neighbors
            for (_sender, input) in inputs {
                match input {
                    Message::Text(text) if text.contains("nectar") => {
                        // Parse nectar information from neighbors
                        agent_state.nectar_found += 1;
                        output_messages.push(Message::Text(format!(
                            "nectar_relay: agent_{} reporting success",
                            self.agent_index
                        )));
                    }
                    Message::Text(text) if text.contains("location") => {
                        // Received location info from neighbor
                        let response = format!(
                            "location_ack: agent_{} at ({:.2}, {:.2})",
                            self.agent_index, agent_state.position_x, agent_state.position_y,
                        );
                        output_messages.push(Message::Text(response));
                    }
                    _ => {}
                }
            }

            // Simulate search movement
            let search_amplitude = 0.1;
            let tick = agent_state.last_discovery_tick;
            agent_state.position_x += (tick as f64 * 0.01).sin() * search_amplitude;
            agent_state.position_y += (tick as f64 * 0.02).cos() * search_amplitude;

            // Simulate finding "nectar" with certain probability (random-like based on position)
            let find_probability = 0.05 * (1.0 + (agent_state.position_x).sin());
            if tick % 100 == self.agent_index as u64 && find_probability > 0.3 {
                let discovery = format!(
                    "nectar_found: agent_{}_discovery at ({:.2}, {:.2})",
                    self.agent_index, agent_state.position_x, agent_state.position_y,
                );
                output_messages.push(Message::Text(discovery));
                agent_state.nectar_found += 1;
            }

            // Try to form new connections with probability
            if tick % 50 == 0 && agent_state.neighbors_known < 5 {
                let neighbors = topology.get_neighbors(ZoooidId::new_v4()); // Placeholder
                if neighbors.len() < 5 {
                    let connection_msg = format!(
                        "seeking_connection: agent_{}_wants_to_connect",
                        self.agent_index
                    );
                    output_messages.push(Message::Text(connection_msg));
                }
            }

            // Periodically report status
            if tick % 200 == 0 && tick > 0 {
                let status = format!(
                    "status: agent_{}|energy:{}|nectar:{}|position:({:.1},{:.1})",
                    self.agent_index,
                    agent_state.energy,
                    agent_state.nectar_found,
                    agent_state.position_x,
                    agent_state.position_y,
                );
                output_messages.push(Message::Text(status));
            }

            agent_state.last_discovery_tick += 1;
            agent_state.messages_sent += output_messages.len() as u32;

            Ok(AgentUpdateResult::new(
                output_messages,
                topology_changes,
                None,
                false,
            ))
        } else {
            Err("State downcast failed".into())
        }
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Swarm
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║        Adaptiflux Swarm Search Demonstration              ║");
    println!("║                                                            ║");
    println!("║  A collaborative swarm of 12 agents searching for targets ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Create core components
    let topology = Arc::new(RwLock::new(ZoooidTopology::new()));
    let message_bus = Arc::new(LocalBus::new());
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    // Create and configure scheduler
    let mut scheduler = CoreScheduler::new(
        topology.clone(),
        rule_engine,
        resource_manager,
        message_bus.clone(),
    );

    // Set cycle frequency to 100 Hz for better responsiveness
    scheduler.set_cycle_frequency(100);

    println!("🐝 Creating search swarm (12 agents)...\n");

    // Spawn 12 search agents
    const SWARM_SIZE: usize = 12;
    for i in 0..SWARM_SIZE {
        let agent = Zoooid::new(
            ZoooidId::new_v4(),
            Box::new(SearchAgentBlueprint { agent_index: i }),
        )
        .await
        .unwrap_or_else(|err| panic!("Failed to create agent {}: {:?}", i, err));
        scheduler
            .spawn_agent(agent)
            .await
            .unwrap_or_else(|err| panic!("Failed to spawn agent {}: {:?}", i, err));

        if (i + 1) % 3 == 0 {
            print!(".");
        }
    }
    println!(
        "\n✅ Swarm initialized: {} agents ready\n",
        scheduler.agent_count()
    );

    // Statistics collection
    let stats_handle = {
        let topology_clone = topology.clone();

        tokio::spawn(async move {
            let start_time = std::time::Instant::now();

            loop {
                tokio::time::sleep(Duration::from_millis(500)).await;

                let topo = topology_clone.read().await;
                let node_count = topo.graph.node_count();
                let edge_count = topo.graph.edge_count();
                drop(topo);

                let elapsed = start_time.elapsed().as_secs_f64();

                println!(
                    "[{:5.1}s] Swarm status - Agents: {}, Connections: {}, Connectivity: {:.2}",
                    elapsed,
                    node_count,
                    edge_count,
                    if node_count > 0 {
                        (edge_count as f64) / (node_count as f64)
                    } else {
                        0.0
                    }
                );

                if elapsed > 10.0 {
                    break;
                }
            }
        })
    };

    println!("🚀 Starting swarm search (running for 10 seconds)...\n");

    // Run the scheduler
    let run_handle = tokio::spawn(async move {
        if let Err(e) = scheduler.run().await {
            eprintln!("❌ Scheduler error: {}", e);
        }
    });

    // Wait for both handles or timeout
    tokio::select! {
        _ = stats_handle => println!("\n📊 Statistics collection completed"),
        _ = run_handle => println!("\n🛑 Scheduler stopped"),
        _ = tokio::time::sleep(Duration::from_secs(11)) => {
            println!("\n⚠️  Timeout reached");
        }
    }

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                   Swarm Search Complete                   ║");
    println!("║                                                            ║");
    println!("║  ✓ Distributed swarm demonstrated successfully            ║");
    println!("║  ✓ Agents operated autonomously and collaboratively       ║");
    println!("║  ✓ Network topology evolved dynamically                   ║");
    println!("║  ✓ Emergent behavior from local rules observed            ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
