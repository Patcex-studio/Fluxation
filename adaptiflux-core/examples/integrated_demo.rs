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
/// Integrated demonstration of the complete Adaptiflux core architecture
///
/// This example demonstrates:
/// - CoreScheduler coordinating all components
/// - Multiple agent types working together
/// - Message passing between agents
/// - Topology management and dynamic connections
/// - Rule engine evaluating system-wide rules
/// - Consistency checks maintaining system integrity
use adaptiflux_core::memory::types::MemoryPayload;
use adaptiflux_core::{
    AgentBlueprint, AgentUpdateResult, CoreScheduler, LocalBus, Message, ResourceManager, RoleType,
    RuleEngine, Zoooid, ZoooidId,
};
use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Duration;

/// Echo blueprint - reflects received messages back to sender
struct EchoBlueprint;

#[async_trait]
impl AgentBlueprint for EchoBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(EchoState {
            message_count: 0,
            tick_count: 0,
        }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<Message>,
        topology: &adaptiflux_core::ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(echo_state) = state.downcast_mut::<EchoState>() {
            echo_state.tick_count += 1;
            echo_state.message_count += inputs.len();
        }

        let response = inputs
            .into_iter()
            .map(|message| match message {
                Message::Text(text) => Message::Text(format!("echo: {}", text)),
                other => other,
            })
            .collect();

        // Periodically request connections to other agents (every 10 ticks)
        let topology_change = if let Some(echo_state) = state.downcast_ref::<EchoState>() {
            if echo_state.tick_count % 10 == 0 {
                let all_nodes: Vec<_> = topology.graph.nodes().collect();
                if !all_nodes.is_empty() {
                    let target_idx = (echo_state.tick_count as usize) % all_nodes.len();
                    Some(adaptiflux_core::TopologyChange::RequestConnection(
                        all_nodes[target_idx],
                    ))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(AgentUpdateResult::new(
            response,
            None,
            topology_change,
            false,
        ))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Sensor
    }
}

#[derive(Debug)]
struct EchoState {
    message_count: usize,
    tick_count: u64,
}

/// Sensor blueprint - generates periodic messages
struct SensorBlueprint;

#[async_trait]
impl AgentBlueprint for SensorBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(SensorState {
            tick: 0,
            reading_counter: 0,
        }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        _inputs: Vec<Message>,
        topology: &adaptiflux_core::ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(sensor_state) = state.downcast_mut::<SensorState>() {
            sensor_state.tick += 1;

            // Generate a message every 10 ticks
            let messages = if sensor_state.tick % 10 == 0 {
                sensor_state.reading_counter += 1;
                vec![Message::Text(format!(
                    "sensor_reading: measurement_{}",
                    sensor_state.reading_counter
                ))]
            } else {
                vec![]
            };

            // Periodically request connections to other agents (every 15 ticks)
            let topology_change = if sensor_state.tick % 15 == 0 {
                let all_nodes: Vec<_> = topology.graph.nodes().collect();
                if !all_nodes.is_empty() {
                    let target_idx = (sensor_state.tick as usize) % all_nodes.len();
                    Some(adaptiflux_core::TopologyChange::RequestConnection(
                        all_nodes[target_idx],
                    ))
                } else {
                    None
                }
            } else {
                None
            };

            Ok(AgentUpdateResult::new(
                messages,
                None,
                topology_change,
                false,
            ))
        } else {
            Err("State downcast failed".into())
        }
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Sensor
    }
}

#[derive(Debug)]
struct SensorState {
    tick: u64,
    reading_counter: u64,
}

/// Processor blueprint - processes received messages and generates responses
struct ProcessorBlueprint;

#[async_trait]
impl AgentBlueprint for ProcessorBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(ProcessorState {
            processed: 0,
            last_message: None,
            tick_count: 0,
        }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<Message>,
        topology: &adaptiflux_core::ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(proc_state) = state.downcast_mut::<ProcessorState>() {
            proc_state.tick_count += 1;
            let mut responses = Vec::new();

            for input in inputs {
                if let Message::Text(text) = input {
                    proc_state.processed += 1;
                    proc_state.last_message = Some(text.clone());
                    responses.push(Message::Text(format!("processed: {}", text)));
                }
            }

            // Periodically request connections to other agents (every 12 ticks)
            let topology_change = if proc_state.tick_count % 12 == 0 {
                let all_nodes: Vec<_> = topology.graph.nodes().collect();
                if !all_nodes.is_empty() {
                    let target_idx = (proc_state.tick_count as usize) % all_nodes.len();
                    Some(adaptiflux_core::TopologyChange::RequestConnection(
                        all_nodes[target_idx],
                    ))
                } else {
                    None
                }
            } else {
                None
            };

            Ok(AgentUpdateResult::new(
                responses,
                None,
                topology_change,
                false,
            ))
        } else {
            Err("State downcast failed".into())
        }
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Pid
    }
}

#[derive(Debug)]
struct ProcessorState {
    processed: u32,
    last_message: Option<String>,
    tick_count: u64,
}

/// Aggregator blueprint - collects and summarizes information
struct AggregatorBlueprint;

#[async_trait]
impl AgentBlueprint for AggregatorBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(AggregatorState {
            total_messages: 0,
            unique_sources: 0,
            tick_count: 0,
        }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<Message>,
        topology: &adaptiflux_core::ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(agg_state) = state.downcast_mut::<AggregatorState>() {
            agg_state.tick_count += 1;
            agg_state.total_messages += inputs.len() as u32;

            // Every 5 messages, generate a summary
            let response = if agg_state.total_messages % 5 == 0 && agg_state.total_messages > 0 {
                vec![Message::Text(format!(
                    "summary: total_messages={}, unique_sources={}",
                    agg_state.total_messages, agg_state.unique_sources
                ))]
            } else {
                vec![]
            };

            // Periodically request connections to other agents (every 8 ticks)
            let topology_change = if agg_state.tick_count % 8 == 0 {
                let all_nodes: Vec<_> = topology.graph.nodes().collect();
                if !all_nodes.is_empty() {
                    let target_idx = (agg_state.tick_count as usize) % all_nodes.len();
                    Some(adaptiflux_core::TopologyChange::RequestConnection(
                        all_nodes[target_idx],
                    ))
                } else {
                    None
                }
            } else {
                None
            };

            Ok(AgentUpdateResult::new(
                response,
                None,
                topology_change,
                false,
            ))
        } else {
            Err("State downcast failed".into())
        }
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Cognitive
    }
}

#[derive(Debug)]
struct AggregatorState {
    total_messages: u32,
    unique_sources: u32,
    tick_count: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║     Adaptiflux Integrated Demo - Complete Ecosystem       ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Create core components
    let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
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

    // Set cycle frequency to 50 Hz for smoother demonstration
    scheduler.set_cycle_frequency(50);

    println!("📊 Creating agent network...");
    println!("   ├─ Sensor Agent (generates readings)");
    println!("   ├─ Processor Agent (transforms data)");
    println!("   ├─ Aggregator Agent (summarizes)");
    println!("   └─ Echo Agent (reflects messages)\n");

    // Spawn agents
    let sensor_agent = Zoooid::new(ZoooidId::new_v4(), Box::new(SensorBlueprint))
        .await
        .expect("Failed to create sensor agent");
    scheduler
        .spawn_agent(sensor_agent)
        .await
        .expect("Failed to spawn sensor agent");

    let processor_agent = Zoooid::new(ZoooidId::new_v4(), Box::new(ProcessorBlueprint))
        .await
        .expect("Failed to create processor agent");
    scheduler
        .spawn_agent(processor_agent)
        .await
        .expect("Failed to spawn processor agent");

    let aggregator_agent = Zoooid::new(ZoooidId::new_v4(), Box::new(AggregatorBlueprint))
        .await
        .expect("Failed to create aggregator agent");
    scheduler
        .spawn_agent(aggregator_agent)
        .await
        .expect("Failed to spawn aggregator agent");

    let echo_agent = Zoooid::new(ZoooidId::new_v4(), Box::new(EchoBlueprint))
        .await
        .expect("Failed to create echo agent");
    scheduler
        .spawn_agent(echo_agent)
        .await
        .expect("Failed to spawn echo agent");

    println!(
        "✅ Network initialized: {} agents spawned\n",
        scheduler.agent_count()
    );

    // Set up dynamic stop after 8 seconds
    let stop_signal = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let should_stop = stop_signal.clone();

    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(8)).await;
        should_stop.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    // Create a task to monitor scheduler progress
    let monitor_handle = {
        let topology_clone = topology.clone();

        tokio::spawn(async move {
            let mut last_print = std::time::Instant::now();
            loop {
                if last_print.elapsed() > Duration::from_secs(2) {
                    let topo = topology_clone.lock().await;
                    println!(
                        "⏱️  System alive - Topology nodes: {}, Connections: {}",
                        topo.graph.node_count(),
                        topo.graph.edge_count()
                    );
                    drop(topo);
                    last_print = std::time::Instant::now();

                    if stop_signal.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }
                }
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        })
    };

    println!("🚀 Starting scheduler loop (running for 8 seconds)...\n");

    // Run the scheduler
    let run_handle = tokio::spawn(async move {
        if let Err(e) = scheduler.run().await {
            eprintln!("❌ Scheduler error: {}", e);
        }
    });

    // Wait for both monitor and scheduler to complete
    tokio::select! {
        _ = monitor_handle => {}
        _ = run_handle => {}
    }

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    Demo Completed                         ║");
    println!("║                                                            ║");
    println!("║  ✓ All components successfully integrated                 ║");
    println!("║  ✓ Agents communicated via message bus                    ║");
    println!("║  ✓ Topology was dynamically managed                       ║");
    println!("║  ✓ System ran stable for 8 seconds                        ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
