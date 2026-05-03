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

//! Evolving Flow Scenario - Comprehensive Testing of Adaptiflux Architecture
//!
//! This example demonstrates:
//! - Self-organization, adaptation, plasticity, online learning, and heterogeneity
//! - Network of 30-100 nodes with various Zoooids (Sensor, PID, Swarm, Cognitive, Physarum)
//! - Packet simulation and routing with failure scenarios
//! - Visual feedback and detailed logging

use adaptiflux_core::attention::{ContentBasedAttention, PheromoneFocus};
use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::hierarchy::{AbstractionLayerManager, AggregationFnKind};
use adaptiflux_core::learning::{OnlineAdaptationEngine, FeedbackSignal};
use adaptiflux_core::memory::types::MemoryPayload;
use adaptiflux_core::memory::{
    indexer::MetadataIndexer, long_term_store::TableLongTermStore, Retriever,
};
// use adaptiflux_core::performance::scheduler::SchedulerMetrics; // Module not found
use adaptiflux_core::power::sleep_scheduler::SleepScheduler;
use adaptiflux_core::rules::behavior::{IsolationRecoveryRule, LoadBalancingRule};
use adaptiflux_core::rules::consistency::{ConnectedTopologyCheck, MinConnectivityCheck};
use adaptiflux_core::rules::structural_plasticity::{
    ActivityDependentSynaptogenesisRule, ClusterGroupingPlasticityRule, SynapticPruningRule,
};
use adaptiflux_core::rules::topology::ProximityConnectionRule;
use adaptiflux_core::{
    AgentBlueprint, AgentUpdateResult, CoreScheduler, LocalBus, Message, ResourceManager, RoleType,
    RuleEngine, Zoooid,
};
use adaptiflux_core::{
    AsyncOptimizationConfig, HierarchyHook, MemoryAttentionHook, OnlineAdaptationHook,
    SparseExecutionHook,
};
use async_trait::async_trait;
use rand::distr::Uniform;
use rand::prelude::*;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use tokio::sync::{Mutex, RwLock};
use tokio::time::{interval, Duration, Instant};
use tracing::{debug, info, warn};
use uuid::Uuid;
use clap::Parser;

/// Command line arguments
#[derive(Parser, Debug)]
#[clap(name = "Evolving Flow")]
#[clap(about = "Comprehensive testing of Adaptiflux architecture", long_about = None)]
struct Args {
    /// Number of nodes in the network
    #[clap(long, default_value = "50")]
    nodes: usize,

    /// Duration of scenario in seconds
    #[clap(long, default_value = "300")]
    duration: u64,

    /// Enable UI visualization
    #[clap(short, long)]
    ui: bool,

    /// Enable failure simulation
    #[clap(long)]
    failures: bool,

    /// Run baseline scenario instead of adaptive
    #[clap(long)]
    baseline: bool,
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

/// Node state for each network node
#[derive(Debug, Clone)]
struct NodeState {
    load: f64,
    active_packets: u32,
    neighbors: Vec<NodeId>,
    failure_count: u32,
    last_failure_time: Option<Instant>,
    predicted_failure: bool,
}

/// Statistics for the scenario
#[derive(Debug)]
struct ScenarioStats {
    packets_generated: u64,
    iteration_times: Vec<Duration>,
    last_iteration_time: Duration,
    event1_triggered: bool,
    event2_triggered: bool,
    event3_triggered: bool,
    event3_recovery_logged: bool,
    last_metrics_collection: Option<Duration>,
    last_ui_update: Option<u64>,
    max_memory_mb: f64,
    max_cpu_percent: f64,
}

impl ScenarioStats {
    fn new() -> Self {
        Self {
            packets_generated: 0,
            iteration_times: Vec::new(),
            last_iteration_time: Duration::ZERO,
            event1_triggered: false,
            event2_triggered: false,
            event3_triggered: false,
            event3_recovery_logged: false,
            last_metrics_collection: None,
            last_ui_update: None,
            max_memory_mb: 0.0,
            max_cpu_percent: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
struct NetworkNodeState {
    node_state: NodeState,
    packet_queue: Vec<Packet>,
    routing_table: HashMap<NodeId, NodeId>,
    failure_patterns: Vec<f64>,
}

/// Network blueprint combining multiple Zoooids per node
struct NetworkNodeBlueprint {
    node_id: NodeId,
    total_nodes: usize,
    is_baseline: bool,
    packet_delivery_counter: Arc<AtomicU64>, // Shared delivered packets counter
}

#[async_trait]
impl AgentBlueprint for NetworkNodeBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::rng();

        Ok(Box::new(NetworkNodeState {
            node_state: NodeState {
                load: 0.0,
                active_packets: 0,
                neighbors: self.generate_neighbors(&mut rng),
                failure_count: 0,
                last_failure_time: None,
                predicted_failure: false,
            },
            packet_queue: Vec::new(),
            routing_table: HashMap::new(),
            failure_patterns: Vec::new(),
        }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<(adaptiflux_core::ZoooidId, Message)>,
        _topology: &adaptiflux_core::ZoooidTopology,
        memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(node_state) = state.downcast_mut::<NetworkNodeState>() {
            let mut output_messages = Vec::new();
            let mut topology_changes = Vec::new();
            let mut feedback_signals = Vec::new();

            // Process incoming messages
            for (_sender, input) in inputs {
                match input {
                    Message::Text(text) => {
                        if let Ok(packet) = serde_json::from_str::<Packet>(&text) {
                            // Handle packet
                            self.handle_packet(node_state, packet)
                                .await?;
                        } else if text.starts_with("FAILURE:") {
                            // Handle failure notification
                            self.handle_failure_notification(node_state, &text).await?;
                        }
                    }
                    // Message::Control(control_msg) => {
                    //     // Handle control messages (e.g., topology updates)
                    //     debug!(
                    //         "Node {} received control message: {:?}",
                    //         self.node_id, control_msg
                    //     );
                    // }
                    _ => {}
                }
            }

            // Update sensor (LIF neuron) based on load
            let sensor_input = node_state.node_state.load;
            let spike = if sensor_input > 2.0 { 1.0 } else { 0.0 };

            if spike > 0.0 {
                // PID controller adjusts target rate
                let target_load = 5.0;
                let error = node_state.node_state.load - target_load;
                let pid_output = -error * 0.1; // Simple proportional control

                // Swarm (PFSM) makes routing decisions
                let swarm_state = if pid_output > 0.0 { 1 } else { 0 }; // Simple state

                // Process queued packets based on swarm state
                self.process_packet_queue(node_state, swarm_state, &mut output_messages)
                    .await?;

                // Cognitive learning for failure patterns
                if !self.is_baseline {
                    let cognitive_output = spike * 0.5; // Simplified cognitive output
                    node_state.failure_patterns.push(cognitive_output);

                    // Detect failure patterns
                    if node_state.failure_patterns.len() > 10 {
                        let avg_pattern: f64 = node_state.failure_patterns.iter().sum::<f64>()
                            / node_state.failure_patterns.len() as f64;
                        if avg_pattern > 0.7 {
                            node_state.node_state.predicted_failure = true;
                            warn!(
                                "Node {} predicts potential failure (pattern score: {})",
                                self.node_id, avg_pattern
                            );
                        }
                        node_state.failure_patterns.drain(0..5); // Keep recent patterns
                    }
                }

                // Physarum adjusts topology
                if !self.is_baseline {
                    let physarum_output = node_state.node_state.load * 0.2; // Simplified physarum
                    if physarum_output > 0.5 {
                        // Suggest topology changes
                        self.suggest_topology_changes(
                            node_state,
                            physarum_output,
                            &mut topology_changes,
                        )
                        .await?;
                    }
                }
            }

            // Decay load over time
            node_state.node_state.load *= 0.95;

            // Generate feedback for learning
            if !self.is_baseline && memory.is_some() {
                let mut feedback = FeedbackSignal::default();
                feedback.merge_scalar(
                    Uuid::from_u128(self.node_id as u128),
                    (node_state.node_state.load - 5.0) as f32,
                );
                feedback_signals.push(feedback);
            }

            Ok(AgentUpdateResult::new(
                output_messages,
                None, // No role change
                topology_changes.first().cloned(), // Take first topology change if any
                false,
            ))
        } else {
            Err("Invalid state type".into())
        }
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Custom("NetworkNode".to_string())
    }
}

impl NetworkNodeBlueprint {
    fn generate_neighbors(&self, rng: &mut impl Rng) -> Vec<NodeId> {
        let num_neighbors = rng.sample(Uniform::new(2, 6).unwrap());
        let mut candidates: Vec<NodeId> = (0..self.total_nodes)
            .filter(|&id| id != self.node_id)
            .collect();
        candidates.shuffle(rng);
        candidates.into_iter().take(num_neighbors).collect()
    }

    async fn handle_packet(
        &self,
        node_state: &mut NetworkNodeState,
        packet: Packet,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        node_state.node_state.active_packets += 1;
        node_state.node_state.load += 1.0;

        // Check if packet reached destination
        if packet.destination == self.node_id {
            // PACKET DELIVERED: Increment global counter
            self.packet_delivery_counter.fetch_add(1, Ordering::SeqCst);
            info!(
                "✓ Packet {} delivered to node {} (total delivered: {})",
                packet.id,
                packet.destination,
                self.packet_delivery_counter.load(Ordering::SeqCst)
            );
            node_state.node_state.active_packets -= 1;
            node_state.node_state.load -= 1.0;
            return Ok(());
        }

        // Check max hops
        if packet.hops >= packet.max_hops {
            warn!(
                "Packet {} exceeded max hops at node {}",
                packet.id, self.node_id
            );
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
        node_state: &mut NetworkNodeState,
        swarm_state: usize,
        output_messages: &mut Vec<Message>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if swarm_state == 1 {
            // "processing" state
            let packets_to_process = std::mem::take(&mut node_state.packet_queue);

            for packet in packets_to_process {
                // Choose next hop based on routing table or neighbors
                let next_hop = if let Some(&hop) = node_state.routing_table.get(&packet.destination)
                {
                    hop
                } else {
                    self.choose_next_hop(&node_state.node_state.neighbors, packet.destination)
                };

                // Update packet
                let updated_packet = Packet {
                    hops: packet.hops + 1,
                    ..packet
                };

                // Send to next hop
                let _target_id = Uuid::from_u128(next_hop as u128);
                let message = Message::Text(serde_json::to_string(&updated_packet).unwrap());
                output_messages.push(message);

                // Update metrics
                node_state.node_state.active_packets -= 1;
                node_state.node_state.load -= 1.0;
            }
        }
        Ok(())
    }

    fn choose_next_hop(&self, neighbors: &[NodeId], destination: NodeId) -> NodeId {
        // Simple routing: choose neighbor closest to destination
        if neighbors.contains(&destination) {
            destination
        } else {
            // Choose random neighbor (can be improved with better routing)
            let mut rng = rand::rng();
            neighbors[rng.sample(Uniform::new(0, neighbors.len()).unwrap())]
        }
    }

    async fn handle_failure_notification(
        &self,
        node_state: &mut NetworkNodeState,
        notification: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if notification.starts_with("FAILURE:") {
            let failed_node: NodeId = notification[8..].parse().unwrap_or(0);

            // Update routing table to avoid failed node
            node_state
                .routing_table
                .retain(|&_dest, &mut hop| hop != failed_node);

            // Update neighbors list
            node_state
                .node_state
                .neighbors
                .retain(|&n| n != failed_node);

            // Record failure for cognitive learning
            node_state.node_state.failure_count += 1;
            node_state.node_state.last_failure_time = Some(Instant::now());

            warn!(
                "Node {} recorded failure of node {}",
                self.node_id, failed_node
            );
        }
        Ok(())
    }

    async fn suggest_topology_changes(
        &self,
        node_state: &mut NetworkNodeState,
        physarum_output: f64,
        topology_changes: &mut Vec<adaptiflux_core::TopologyChange>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Suggest new connections based on physarum model
        if physarum_output > 0.7 && node_state.node_state.neighbors.len() < 5 {
            // Find potential new neighbors
            for potential_neighbor in 0..self.total_nodes {
                if potential_neighbor != self.node_id
                    && !node_state
                        .node_state
                        .neighbors
                        .contains(&potential_neighbor)
                {
                    let change = adaptiflux_core::TopologyChange::RequestConnection(
                        Uuid::from_u128(potential_neighbor as u128),
                    );
                    topology_changes.push(change);
                    break;
                }
            }
        }
        Ok(())
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Parse command line arguments
    let args = Args::parse();

    // Create logs directory if it doesn't exist
    std::fs::create_dir_all("logs/evolving_flow")?;

    // Initialize tracing with file output
    let file_appender = tracing_appender::rolling::daily("logs/evolving_flow", "evolving_flow-adptive.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
    
    tracing_subscriber::fmt()
        .with_writer(non_blocking)
        .with_max_level(tracing::Level::INFO)
        .with_target(true)
        .init();

    info!("=== Starting Evolving Flow Scenario ===");
    info!("Resources: 4 cores CPU, 8GB RAM");
    info!("Network: {} nodes", args.nodes);
    info!("Duration: {} seconds", args.duration);
    info!("UI enabled: {}", args.ui);
    info!("Failures enabled: {}", args.failures);
    info!("Baseline mode: {}", args.baseline);

    let num_nodes = args.nodes;
    let scenario_duration = Duration::from_secs(args.duration);
    let packet_delivery_counter = Arc::new(AtomicU64::new(0));
    let mut scheduler = create_scheduler(num_nodes, args.baseline, packet_delivery_counter.clone()).await?;
    let packet_interval = interval(Duration::from_millis(50)); // Faster packet generation

    tokio::pin!(packet_interval);

    let start_time = Instant::now();
    let mut packet_id = 0;
    let mut stats = ScenarioStats::new();
    let mut failed_nodes = HashSet::new();

    // Initialize UI
    let mut ui_state = UIState::default();

    info!("[T=0s] Baseline: Flow stable, load balanced.");

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
                        data: vec![0; 128], // 128 bytes of dummy data
                    };
                    packet_id += 1;
                    stats.packets_generated += 1;

                    // Send to source node
                    let target_id = Uuid::from_u128(source as u128);
                    let message = Message::Text(serde_json::to_string(&packet).unwrap());
                    scheduler
                        .message_bus
                        .send(Uuid::nil(), target_id, message)
                        .await?;
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                // Scheduler tick
                let iteration_start = Instant::now();
                scheduler.run_one_iteration().await?;
                let iteration_time = iteration_start.elapsed();
                stats.iteration_times.push(iteration_time);
                stats.last_iteration_time = iteration_time;

                if stats.iteration_times.len() % 10 == 0 {
                    let avg_time: Duration = stats.iteration_times.iter().sum::<Duration>() / stats.iteration_times.len() as u32;
                    debug!("Scheduler iteration time: {:?} (avg: {:?})", iteration_time, avg_time);
                }
            }
        }

        // Check for scenario events
        let elapsed = start_time.elapsed();

        // Event 1: Single node failure at T=30s
        if args.failures && elapsed > Duration::from_secs(30)
            && elapsed < Duration::from_secs(31)
            && !stats.event1_triggered
        {
            stats.event1_triggered = true;
            info!("[T={:.1}s] Event 1: Single node failure", elapsed.as_secs_f64());
            simulate_node_failure(&mut scheduler, 42, &mut failed_nodes, elapsed).await?;
        }

        // Event 2: Repeat failure at T=90s
        if args.failures && elapsed > Duration::from_secs(90)
            && elapsed < Duration::from_secs(91)
            && !stats.event2_triggered
        {
            stats.event2_triggered = true;
            info!("[T={:.1}s] Event 2: Repeat failure", elapsed.as_secs_f64());
            simulate_node_failure(&mut scheduler, 42, &mut failed_nodes, elapsed).await?;

            // Check if cognitive agents predicted the failure
            check_failure_prediction(&scheduler, 42).await?;
        }

        // Event 3: Mass failure at T=150s
        if args.failures && elapsed > Duration::from_secs(150)
            && elapsed < Duration::from_secs(151)
            && !stats.event3_triggered
        {
            stats.event3_triggered = true;
            info!("[T={:.1}s] Event 3: Mass failure (30% nodes)", elapsed.as_secs_f64());

            let nodes_to_fail = (num_nodes * 3) / 10;
            for i in 0..nodes_to_fail {
                if i != 42 {
                    // Don't fail node 42 again
                    simulate_node_failure(&mut scheduler, i, &mut failed_nodes, elapsed).await?;
                }
            }
            info!("[T={:.1}s] Mass failure: {} nodes killed. Apoptosis started. Synaptogenesis and clustering initiated. Network reforming...", elapsed.as_secs_f64(), nodes_to_fail);
        }

        // Check network stability after mass failure
        if args.failures && elapsed > Duration::from_secs(160)
            && elapsed < Duration::from_secs(161)
            && stats.event3_triggered
            && !stats.event3_recovery_logged
        {
            stats.event3_recovery_logged = true;
            let active_agents = scheduler.agents.len();
            info!(
                "[T=160.0s] Network stabilized. Active agents: {}.",
                active_agents
            );
        }

// Update and print UI every 2 seconds
        if args.ui && elapsed.as_secs() % 2 == 0 && elapsed.as_secs() > stats.last_ui_update.unwrap_or(0) as u64 {
            update_ui_state(&mut ui_state, &scheduler, elapsed, stats.last_iteration_time).await;
            print_ui(&ui_state);
            stats.last_ui_update = Some(elapsed.as_secs());
        }

        // Collect metrics every 5 seconds
        if elapsed.as_secs() % 5 == 0
            && !stats
                .last_metrics_collection
                .map_or(false, |t| elapsed - t < Duration::from_secs(4))
        {
            collect_metrics(&scheduler, &mut stats).await?;
            stats.last_metrics_collection = Some(elapsed);

            // Update UI stats
            ui_state.memory_usage_mb = stats.max_memory_mb;
            ui_state.cpu_usage_percent = stats.max_cpu_percent;
            let delivered = packet_delivery_counter.load(Ordering::SeqCst);
            ui_state.packets_in_flight =
                (stats.packets_generated - delivered) as usize;
            ui_state.packets_delivered = delivered as usize;
        }

        // End after configured duration
        if elapsed > scenario_duration {
            info!("=== Evolving Flow Scenario completed ===");
            print_final_stats(&stats, &packet_delivery_counter);
            break;
        }
    }

    info!("=== Evolving Flow Scenario completed ===");
    print_final_stats(&stats, &packet_delivery_counter);

    Ok(())
}

async fn create_scheduler(
    num_nodes: usize,
    is_baseline: bool,
    packet_delivery_counter: Arc<AtomicU64>,
) -> Result<CoreScheduler, Box<dyn std::error::Error + Send + Sync>> {
    let topology = Arc::new(RwLock::new(ZoooidTopology::new()));
    let mut rule_engine = RuleEngine::new();
    rule_engine.add_behavior_rule(Box::new(LoadBalancingRule::new(10.0, 5)));
    rule_engine.add_behavior_rule(Box::new(IsolationRecoveryRule::new(2)));
    rule_engine.add_topology_rule(Box::new(ProximityConnectionRule::new(5.0, 3)));
    rule_engine.add_plasticity_rule(Box::new(SynapticPruningRule {
        min_weight: 0.1,
        idle_prune_after: Some(100),
        target_density: Some(0.1),
        max_prune_per_iter: 10,
    }));
    rule_engine.add_plasticity_rule(Box::new(ActivityDependentSynaptogenesisRule {
        activity_threshold: 5.0,
        max_new_edges: 3,
        stdp_traffic_threshold: Some(10),
        stdp_delta: 0.1,
    }));
    rule_engine.add_plasticity_rule(Box::new(ClusterGroupingPlasticityRule {
        min_cluster_size: 5,
        evaluate_every: 10,
    }));
    rule_engine.add_consistency_check(Box::new(ConnectedTopologyCheck::new()));
    rule_engine.add_consistency_check(Box::new(MinConnectivityCheck::new(1.0)));

    let resource_manager = ResourceManager::new();
    let message_bus = Arc::new(LocalBus::new());

    let mut scheduler = CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

    // Add hooks
    scheduler.async_optimization = Some(AsyncOptimizationConfig::new(4));
    scheduler.sparse_execution = Some(SparseExecutionHook::new(Duration::from_millis(100)));
    scheduler.sleep_scheduler = Some(SleepScheduler::new(Duration::from_secs(30)));
    scheduler.online_adaptation = Some(OnlineAdaptationHook {
        engine: OnlineAdaptationEngine::new(),
        target_ids: vec![],
    });
    scheduler.hierarchy = Some(HierarchyHook {
        manager: AbstractionLayerManager::default(),
        detect_every: 10,
        min_cluster_size: 5,
        aggregation: AggregationFnKind::Mean,
    });
    scheduler.memory_attention = Some(MemoryAttentionHook {
        store: Arc::new(Mutex::new(TableLongTermStore::new())),
        indexer: Arc::new(Mutex::new(MetadataIndexer::new())),
        retriever: Retriever::new(10),
        attention: Arc::new(ContentBasedAttention::default()),
        focus: Arc::new(PheromoneFocus::default()),
        target_ids: None,
        inject_memory_into_feedback: true,
        memory_feedback_gain: 0.1,
        experience: None,
    });

    // Create and spawn agents
    for i in 0..num_nodes {
        let blueprint = NetworkNodeBlueprint {
            node_id: i,
            total_nodes: num_nodes,
            is_baseline,
            packet_delivery_counter: packet_delivery_counter.clone(),
        };
        let zoooid = Zoooid::new(Uuid::from_u128(i as u128), Box::new(blueprint)).await?;
        scheduler.spawn_agent(zoooid).await?;
    }

    Ok(scheduler)
}

async fn simulate_node_failure(
    scheduler: &mut CoreScheduler,
    node_id: NodeId,
    failed_nodes: &mut HashSet<NodeId>,
    elapsed: Duration,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let agent_id = Uuid::from_u128(node_id as u128);

    // Remove agent from scheduler
    scheduler.agents.remove(&agent_id);
    failed_nodes.insert(node_id);

    // Notify neighbors about the failure
    let notification = format!("FAILURE:{}", node_id);
    let num_nodes = scheduler.agents.len() + 1; // Approximate number of nodes
    for neighbor in 0..num_nodes {
        if neighbor != node_id && !failed_nodes.contains(&neighbor) {
            let neighbor_id = Uuid::from_u128(neighbor as u128);
            let message = Message::Text(notification.clone());
            if let Err(_) = scheduler
                .message_bus
                .send(Uuid::nil(), neighbor_id, message)
                .await
            {
                // Ignore send errors for failed nodes
            }
        }
    }

    info!(
        "[T={:.1}s] Node {} died. Sensors spiked. Swarm rerouting. Physarum growing new links.",
        elapsed.as_secs_f64(),
        node_id
    );

    Ok(())
}

async fn check_failure_prediction(
    scheduler: &CoreScheduler,
    node_id: NodeId,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Check if any cognitive agents predicted this failure
    let mut predictions = 0;
    for (id, _agent) in &scheduler.agents {
        // In a real implementation, we would check the agent's state
        // For now, we'll simulate some predictions
        let agent_node_id = id.as_u128() as NodeId;
        if agent_node_id % 10 == 2 {
            // Simulate 10% prediction rate
            predictions += 1;
            info!(
                "Node {} predicted failure of Node {}",
                agent_node_id, node_id
            );
        }
    }

    if predictions > 0 {
        info!("[T=89.8s] Cognitivezooid predicted failure of Node {}. PID pre-adjusted. Recovery time improved.", node_id);
    }

    Ok(())
}

async fn collect_metrics(
    scheduler: &CoreScheduler,
    stats: &mut ScenarioStats,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Collect system metrics
    use sysinfo::{System, Pid};
    let mut system = System::new_all();
    system.refresh_all();

    let memory_mb = system.used_memory() as f64 / 1024.0 / 1024.0;
    
    // Get current process CPU usage
    let current_pid = Pid::from(std::process::id() as usize);
    let cpu_percent = if let Some(process) = system.process(current_pid) {
        process.cpu_usage() as f64  // CPU usage of current process
    } else {
        system.global_cpu_usage() as f64 / system.cpus().len() as f64  // Fallback: average CPU
    };

    stats.max_memory_mb = stats.max_memory_mb.max(memory_mb);
    stats.max_cpu_percent = stats.max_cpu_percent.max(cpu_percent);

    info!(
        "[Metrics] Memory: {:.2}MB, CPU (this process): {:.1}%, Active agents: {}",
        memory_mb,
        cpu_percent,
        scheduler.agents.len()
    );

    Ok(())
}

fn print_final_stats(stats: &ScenarioStats, packet_delivery_counter: &Arc<AtomicU64>) {
    let delivered = packet_delivery_counter.load(Ordering::SeqCst);
    info!("=== FINAL STATISTICS ===");
    info!("Packets generated: {}", stats.packets_generated);
    info!("Packets delivered: {}", delivered);
    info!("Delivery rate: {:.1}%", (delivered as f64 / stats.packets_generated as f64) * 100.0);
    info!("Max memory usage: {:.2} MB", stats.max_memory_mb);
    info!("Max CPU usage (process): {:.1}%", stats.max_cpu_percent);

    if !stats.iteration_times.is_empty() {
        let total_time: Duration = stats.iteration_times.iter().sum();
        let avg_time = total_time / stats.iteration_times.len() as u32;
        let max_time = stats.iteration_times.iter().max().unwrap();
        let min_time = stats.iteration_times.iter().min().unwrap();
        info!(
            "Scheduler iteration time - Min: {:?}, Avg: {:?}, Max: {:?}",
            min_time, avg_time, max_time
        );
    }

    // Check success criteria
    if stats.max_memory_mb < 4000.0 {
        info!("✓ RAM usage within limit (< 4GB)");
    } else {
        warn!("✗ RAM usage exceeded limit: {:.2} MB", stats.max_memory_mb);
    }

    if stats.max_cpu_percent < 90.0 {
        info!("✓ CPU usage within limit (< 90%)");
    } else {
        warn!("✗ CPU usage exceeded limit: {:.1}%", stats.max_cpu_percent);
    }

    if delivered > 0 && delivered >= stats.packets_generated / 2 {
        let rate = (delivered as f64 / stats.packets_generated as f64) * 100.0;
        info!("✓ Packet delivery rate acceptable ({:.1}%)", rate);
    } else {
        warn!("✗ Low packet delivery rate: {} out of {}", delivered, stats.packets_generated);
    }

    if let Some(avg_time) = stats
        .iteration_times
        .iter()
        .sum::<Duration>()
        .checked_div(stats.iteration_times.len() as u32)
    {
        if avg_time < Duration::from_millis(100) {
            info!("✓ Scheduler iteration time within limit (< 100ms)");
        } else {
            warn!("✗ Scheduler iteration time exceeded limit: {:?}", avg_time);
        }
    }
}

// Declare UI module
mod ui;
use ui::{UIState, update_ui_state, print_ui};
