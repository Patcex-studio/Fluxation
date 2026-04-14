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

//! Baseline Scenario - Control test without adaptive components
//!
//! This example demonstrates the same network scenario but without:
//! - Cognitive agents
//! - Physarum agents  
//! - Plasticity engine
//! - Online adaptation
//! - Memory and attention mechanisms

use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::primitives::{LifNeuron, Pfsm, PidController};
use adaptiflux_core::rules::consistency::{ConnectedTopologyCheck, MinConnectivityCheck};
use adaptiflux_core::{
    AgentBlueprint, AgentUpdateResult, CoreScheduler, LocalBus, Message, ResourceManager, RoleType,
    RuleEngine, Zoooid,
};
use adaptiflux_core::{AsyncOptimizationConfig, SparseExecutionHook};
use async_trait::async_trait;
use rand::distr::Uniform;
use rand::prelude::*;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{interval, Duration, Instant};
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use clap::Parser;

/// Command line arguments
#[derive(Parser, Debug)]
#[clap(name = "Evolving Flow Baseline")]
#[clap(about = "Baseline scenario without adaptive components", long_about = None)]
struct Args {
    /// Number of nodes in the network
    #[clap(long, default_value = "50")]
    nodes: usize,

    /// Duration of scenario in seconds
    #[clap(long, default_value = "300")]
    duration: u64,
}

/// Packet structure for simulation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Packet {
    id: u64,
    source: NodeId,
    destination: NodeId,
    hops: u8,
    created_at_ms: u64,
    max_hops: u8,
    data: Vec<u8>,
}

type NodeId = usize;

/// Node state for baseline network node
#[derive(Debug, Clone)]
struct BaselineNodeState {
    load: f64,
    active_packets: u32,
    neighbors: Vec<NodeId>,
    fixed_routes: HashMap<NodeId, NodeId>, // Static routing table
}

/// Baseline network blueprint (no adaptive components)
struct BaselineNodeBlueprint {
    node_id: NodeId,
    total_nodes: usize,
}

#[derive(Debug, Clone)]
struct BaselineNodeInternalState {
    sensor: LifNeuron,
    pid: PidController,
    swarm: Pfsm,
    node_state: BaselineNodeState,
    packet_queue: Vec<Packet>,
}

#[async_trait]
impl AgentBlueprint for BaselineNodeBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::rng();

        // Create static routing table
        let mut fixed_routes = HashMap::new();
        for dest in 0..self.total_nodes {
            if dest != self.node_id {
                // Simple static routing: always route to first neighbor
                let neighbors = self.generate_neighbors(&mut rng);
                fixed_routes.insert(dest, neighbors[0]);
            }
        }

        Ok(Box::new(BaselineNodeInternalState {
            sensor: LifNeuron::new(0.5, 0.0, 0.1),
            pid: PidController::new(0.1, 0.0, 0.0), // No integral or derivative
            swarm: Pfsm::new(vec!["idle", "forward"], 0.0), // No state transitions
            node_state: BaselineNodeState {
                load: 0.0,
                active_packets: 0,
                neighbors: self.generate_neighbors(&mut rng),
                fixed_routes,
            },
            packet_queue: Vec::new(),
        }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<Message>,
        _topology: &adaptiflux_core::ZoooidTopology,
        _memory: Option<&adaptiflux_core::memory::types::MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(node_state) = state.downcast_mut::<BaselineNodeInternalState>() {
            let mut output_messages = Vec::new();

            // Process incoming messages
            for input in inputs {
                match input {
                    Message::Text(text) => {
                        if let Ok(packet) = serde_json::from_str::<Packet>(&text) {
                            self.handle_packet(node_state, packet, &mut output_messages).await?;
                        }
                    }
                    _ => {}
                }
            }

            // Simple fixed processing (no adaptation)
            if node_state.node_state.load > 0.0 {
                // Process queued packets with fixed routing
                self.process_packet_queue(node_state, &mut output_messages).await?;
            }

            // No load decay (static system)
            // No feedback signals (no learning)
            // No topology changes (static network)

            Ok(AgentUpdateResult::new(output_messages, None, None, false))
        } else {
            Err("Invalid state type".into())
        }
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Custom("BaselineNode".to_string())
    }
}

impl BaselineNodeBlueprint {
    fn generate_neighbors(&self, rng: &mut impl Rng) -> Vec<NodeId> {
        let num_neighbors = 3; // Fixed number of neighbors
        let mut candidates: Vec<NodeId> = (0..self.total_nodes)
            .filter(|&id| id != self.node_id)
            .collect();
        candidates.shuffle(rng);
        candidates.into_iter().take(num_neighbors).collect()
    }

    async fn handle_packet(
        &self,
        node_state: &mut BaselineNodeInternalState,
        packet: Packet,
        output_messages: &mut Vec<Message>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        node_state.node_state.active_packets += 1;
        node_state.node_state.load += 1.0;

        // Check if packet reached destination
        if packet.destination == self.node_id {
            debug!("Packet {} reached destination at node {}", packet.id, self.node_id);
            node_state.node_state.active_packets -= 1;
            node_state.node_state.load -= 1.0;
            return Ok(());
        }

        // Check max hops
        if packet.hops >= packet.max_hops {
            warn!("Packet {} exceeded max hops at node {}", packet.id, self.node_id);
            node_state.node_state.active_packets -= 1;
            node_state.node_state.load -= 1.0;
            return Ok(());
        }

        // Queue packet for processing
        node_state.packet_queue.push(packet);
        Ok(())
    }

    async fn process_packet_queue(
        &self,
        node_state: &mut BaselineNodeInternalState,
        output_messages: &mut Vec<Message>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let packets_to_process = std::mem::take(&mut node_state.packet_queue);
        
        for packet in packets_to_process {
            // Use fixed routing table (no adaptation)
            let next_hop = if let Some(&hop) = node_state.node_state.fixed_routes.get(&packet.destination) {
                hop
            } else {
                // Fallback to first neighbor
                node_state.node_state.neighbors[0]
            };

            // Update packet
            let updated_packet = Packet {
                hops: packet.hops + 1,
                ..packet
            };

            // Send to next hop (no load consideration)
            let target_id = Uuid::from_u128(next_hop as u128);
            let message = Message::Text(serde_json::to_string(&updated_packet).unwrap());
            output_messages.push(message);

            // Update metrics
            node_state.node_state.active_packets -= 1;
            node_state.node_state.load -= 1.0;
        }
        Ok(())
    }
}

/// Statistics for baseline scenario
#[derive(Debug)]
struct BaselineStats {
    packets_generated: u64,
    packets_lost: u64,
    iteration_times: Vec<Duration>,
    network_collapsed: bool,
    collapse_time: Option<Duration>,
}

impl BaselineStats {
    fn new() -> Self {
        Self {
            packets_generated: 0,
            packets_lost: 0,
            iteration_times: Vec::new(),
            network_collapsed: false,
            collapse_time: None,
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Parse command line arguments
    let args = Args::parse();

    // Create logs directory if it doesn't exist
    std::fs::create_dir_all("logs/evolving_flow")?;

    // Initialize tracing with file output
    let file_appender = tracing_appender::rolling::daily("logs/evolving_flow", "baseline.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    
    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("=== Starting Baseline Scenario ===");
    info!("Network: {} nodes WITHOUT adaptive components", args.nodes);
    info!("Duration: {} seconds", args.duration);

    let num_nodes = args.nodes;
    let scenario_duration = Duration::from_secs(args.duration);
    let mut scheduler = create_baseline_scheduler(num_nodes).await?;
    let packet_interval = interval(Duration::from_millis(50));

    tokio::pin!(packet_interval);

    let start_time = Instant::now();
    let mut packet_id = 0;
    let mut stats = BaselineStats::new();
    let mut failed_nodes = HashSet::new();

    info!("[T=0s] Baseline: Flow starting with static routing.");

    loop {
        tokio::select! {
            _ = packet_interval.tick() => {
                // Generate new packet
                let source = rand::rng().sample(Uniform::new(0, num_nodes).unwrap());
                let destination = rand::rng().sample(Uniform::new(0, num_nodes).unwrap());
                if source != destination && !failed_nodes.contains(&source) {
                    let packet = Packet {
                        id: packet_id,
                        source,
                        destination,
                        hops: 0,
                        created_at_ms: start_time.elapsed().as_millis() as u64,
                        max_hops: 15,
                        data: vec![0; 128],
                    };
                    packet_id += 1;
                    stats.packets_generated += 1;

                    let target_id = Uuid::from_u128(source as u128);
                    let message = Message::Text(serde_json::to_string(&packet).unwrap());
                    scheduler.message_bus.send(Uuid::nil(), target_id, message).await?;
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                // Scheduler tick
                let iteration_start = Instant::now();
                scheduler.run_one_iteration().await?;
                let iteration_time = iteration_start.elapsed();
                stats.iteration_times.push(iteration_time);
            }
        }

        // Check for scenario events
        let elapsed = start_time.elapsed();
        
        // Single node failure at T=30s
        if elapsed > Duration::from_secs(30) && elapsed < Duration::from_secs(31) {
            info!("[T=30.0s] Single node failure (Node 42)");
            simulate_baseline_failure(&mut scheduler, 42, &mut failed_nodes).await?;
        }
        
        // Mass failure at T=150s
        if elapsed > Duration::from_secs(150) && elapsed < Duration::from_secs(151) {
            info!("[T=150.0s] Mass failure (30% nodes)");
            
            let nodes_to_fail = (num_nodes * 3) / 10;
            for i in 0..nodes_to_fail {
                if i != 42 {
                    simulate_baseline_failure(&mut scheduler, i, &mut failed_nodes).await?;
                }
            }
            
            // Check if network collapses
            let active_ratio = scheduler.agents.len() as f32 / num_nodes as f32;
            if active_ratio < 0.5 && !stats.network_collapsed {
                stats.network_collapsed = true;
                stats.collapse_time = Some(elapsed);
                error!("[Baseline] Network collapsed at T={:?}. Active nodes: {}/{}", 
                      elapsed, scheduler.agents.len(), num_nodes);
                break;
            }
        }

        // End after configured duration or if network collapsed
        if elapsed > scenario_duration || stats.network_collapsed {
            info!("=== Baseline Scenario completed ===");
            print_baseline_stats(&stats, &scheduler);
            break;
        }
    }

    Ok(())
}

async fn create_baseline_scheduler(
    num_nodes: usize,
) -> Result<CoreScheduler, Box<dyn std::error::Error + Send + Sync>> {
    let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
    let mut rule_engine = RuleEngine::new();
    
    // Only basic consistency checks (no adaptive rules)
    rule_engine.add_consistency_check(Box::new(ConnectedTopologyCheck::new()));
    rule_engine.add_consistency_check(Box::new(MinConnectivityCheck::new(1.0)));

    let resource_manager = ResourceManager::new();
    let message_bus = Arc::new(LocalBus::new());

    let mut scheduler = CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

    // Minimal hooks (no adaptation)
    scheduler.async_optimization = Some(AsyncOptimizationConfig::new(4));
    scheduler.sparse_execution = Some(SparseExecutionHook::new(Duration::from_millis(100)));

    // Create and spawn baseline agents
    for i in 0..num_nodes {
        let blueprint = BaselineNodeBlueprint {
            node_id: i,
            total_nodes: num_nodes,
        };
        let zoooid = Zoooid::new(Uuid::from_u128(i as u128), Box::new(blueprint)).await?;
        scheduler.spawn_agent(zoooid).await?;
    }

    Ok(scheduler)
}

async fn simulate_baseline_failure(
    scheduler: &mut CoreScheduler,
    node_id: NodeId,
    failed_nodes: &mut HashSet<NodeId>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let agent_id = Uuid::from_u128(node_id as u128);
    
    // Remove agent from scheduler
    scheduler.agents.remove(&agent_id);
    failed_nodes.insert(node_id);
    
    // No notification to neighbors (static routing doesn't adapt)
    warn!("Node {} failed. Static routing continues unchanged.", node_id);
    
    Ok(())
}

fn print_baseline_stats(stats: &BaselineStats, scheduler: &CoreScheduler) {
    info!("=== BASELINE FINAL STATISTICS ===");
    info!("Packets generated: {}", stats.packets_generated);
    info!("Active agents: {}", scheduler.agents.len());
    
    if stats.network_collapsed {
        error!("✗ NETWORK COLLAPSED at T={:?}", stats.collapse_time.unwrap());
        error!("✗ Packet loss: 100% (network unreachable)");
    } else {
        info!("✓ Network remained operational");
        
        if !stats.iteration_times.is_empty() {
            let total_time: Duration = stats.iteration_times.iter().sum();
            let avg_time = total_time / stats.iteration_times.len() as u32;
            info!("Scheduler iteration time - Avg: {:?}", avg_time);
        }
    }
    
    // Compare with adaptive scenario expectations
    info!("=== EXPECTED vs ADAPTIVE ===");
    info!("Expected: Network collapses under mass failure");
    info!("Adaptive: Network should recover and continue");
    info!("Comparison: Baseline shows what happens WITHOUT adaptation");
}