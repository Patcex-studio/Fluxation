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

use adaptiflux_core::{
    agent::blueprint::base::AgentBlueprint,
    AgentUpdateResult,
    CoreScheduler,
    LocalBus,
    Message,
    ResourceManager,
    RoleType,
    RuleEngine,
    Zoooid,
    ZoooidId,
    ZoooidTopology,
};
use async_trait::async_trait;
use std::any::Any;
use std::time::Instant;

#[derive(Debug)]
struct BenchState {
    counter: u64,
    spike_count: u64,
}

struct BenchCognitiveAgent;

#[async_trait]
impl AgentBlueprint for BenchCognitiveAgent {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(BenchState { counter: 0, spike_count: 0 }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<(ZoooidId, Message)>,
        _topology: &ZoooidTopology,
        _memory: Option<&adaptiflux_core::MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(bench_state) = state.downcast_mut::<BenchState>() {
            bench_state.counter += 1;
            for (_sender, msg) in &inputs {
                if matches!(msg, Message::SpikeEvent { .. }) {
                    bench_state.spike_count += 1;
                }
            }
        }

        let output_messages = if inputs.len() > 2 {
            vec![Message::SpikeEvent { timestamp: 0, amplitude: 1.0 }]
        } else {
            vec![]
        };

        Ok(AgentUpdateResult::new(output_messages, None, None, false))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Cognitive
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("=== adaptiflux-core perf_regression runner ===");

    benchmark_topology_contention().await;
    benchmark_metrics_cache().await;
    benchmark_simd_neuron_performance().await;
    benchmark_scheduler_end_to_end().await;
}

async fn setup_connected_scheduler(num_agents: usize, connection_density: f32) -> CoreScheduler {
    let topology = std::sync::Arc::new(tokio::sync::RwLock::new(ZoooidTopology::new()));
    let message_bus = std::sync::Arc::new(LocalBus::new());
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    let mut scheduler = CoreScheduler::new(topology.clone(), rule_engine, resource_manager, message_bus);
    let mut agent_ids = Vec::with_capacity(num_agents);

    for _ in 0..num_agents {
        let agent = Zoooid::new(ZoooidId::new_v4(), Box::new(BenchCognitiveAgent))
            .await
            .unwrap();
        agent_ids.push(agent.id);
        scheduler.spawn_agent(agent).await.unwrap();
    }

    let connections_per_agent = ((num_agents as f32) * connection_density) as usize;
    let mut topo = topology.write().await;
    for (i, &agent_id) in agent_ids.iter().enumerate() {
        for j in 1..=connections_per_agent.min(num_agents - 1) {
            let target_idx = (i + j) % agent_ids.len();
            let _ = topo.try_add_edge(agent_id, agent_ids[target_idx], Default::default());
        }
    }

    scheduler
}

async fn benchmark_topology_contention() {
    let num_agents = 239;
    let iterations = 20;
    println!("\n[PERF-001] Topology contention benchmark: {} agents, {} iterations", num_agents, iterations);

    let mut scheduler = setup_connected_scheduler(num_agents, 0.1).await;
    let start = Instant::now();
    for _ in 0..iterations {
        scheduler.run_one_iteration().await.unwrap();
    }
    let elapsed = start.elapsed();
    println!("Total time: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    println!("Average per iteration: {:.2} ms", elapsed.as_secs_f64() * 1000.0 / iterations as f64);
}

async fn benchmark_metrics_cache() {
    let num_agents = 239;
    println!("\n[PERF-002] Metrics cache benchmark: {} agents", num_agents);

    let scheduler = setup_connected_scheduler(num_agents, 0.05).await;
    let topology = scheduler.topology.read().await;

    // Dirty cache measurement
    let start_dirty = Instant::now();
    let metrics_dirty = adaptiflux_core::SystemMetrics::from_topology(&topology);
    let dirty_elapsed = start_dirty.elapsed();

    // Clean cache measurement
    let start_clean = Instant::now();
    let _metrics_clean = adaptiflux_core::SystemMetrics::from_topology(&topology);
    let clean_elapsed = start_clean.elapsed();

    drop(topology);
    println!("Dirty computation: {:.2} ms", dirty_elapsed.as_secs_f64() * 1000.0);
    println!("Clean cached read: {:.4} ms", clean_elapsed.as_secs_f64() * 1000.0);
    println!("Metrics sample: total_zoooids={}, total_connections={}", metrics_dirty.total_zoooids, metrics_dirty.total_connections);
}

async fn benchmark_simd_neuron_performance() {
    use adaptiflux_core::primitives::spiking::izhikevich::simd::{pack_input_currents, IzhikevichBatch, IzhikevichBatchParams};

    println!("\n[PERF-003] SIMD neuron benchmark");
    let params = IzhikevichBatchParams::from_scalar(0.02, 0.2, -65.0, 8.0, 0.1);
    for &neuron_count in &[4, 8, 16, 32] {
        let inputs = pack_input_currents(&vec![10.0; neuron_count]);
        let mut batch = IzhikevichBatch::new(neuron_count, &params);
        let iterations = 1000;
        let start = Instant::now();
        for _ in 0..iterations {
            batch.update(&inputs, &params);
        }
        let elapsed = start.elapsed();
        println!("Neurons={} iterations={} total={:.2} ms avg={:.4} ms", neuron_count, iterations, elapsed.as_secs_f64() * 1000.0, elapsed.as_secs_f64() * 1000.0 / iterations as f64);
    }
}

async fn benchmark_scheduler_end_to_end() {
    println!("\n[END-TO-END] Scheduler iteration benchmark");
    for &num_agents in &[50, 100, 239] {
        let iterations = 20;
        let mut scheduler = setup_connected_scheduler(num_agents, 0.1).await;
        let start = Instant::now();
        for _ in 0..iterations {
            scheduler.run_one_iteration().await.unwrap();
        }
        let elapsed = start.elapsed();
        println!("Agents={} iterations={} total={:.2} ms avg={:.2} ms", num_agents, iterations, elapsed.as_secs_f64() * 1000.0, elapsed.as_secs_f64() * 1000.0 / iterations as f64);
    }
}
