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

//! Performance regression benchmarks for PERF-001, PERF-002, PERF-003 optimizations
//! 
//! Metrics:
//! - PERF-001 (Mutex → RwLock): Reduced contention on topology reads
//! - PERF-002 (Metrics caching): O(N²) → O(1) on clean cache
//! - PERF-003 (SIMD neurons): 3-4x speedup on batch updates

use adaptiflux_core::{
    AgentBlueprint, AgentUpdateResult, CoreScheduler, LocalBus, Message, ResourceManager, RoleType,
    RuleEngine, Zoooid, ZoooidId, ZoooidTopology,
};
use async_trait::async_trait;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, BatchSize, Criterion};
use std::any::Any;
use std::sync::Arc;
use std::time::Instant;
use tokio::runtime::Runtime;

/// Simple cognitive agent for benchmarking
struct BenchCognitiveAgent;

#[async_trait]
impl AgentBlueprint for BenchCognitiveAgent {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(BenchState { 
            counter: 0,
            spike_count: 0,
        }))
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
            
            // Simulate neuron activity
            for (_sender, msg) in &inputs {
                if matches!(msg, Message::SpikeEvent { .. }) {
                    bench_state.spike_count += 1;
                }
            }
        }

        // Simulate spike events
        let output_messages = if inputs.len() > 2 {
            vec![Message::SpikeEvent { 
                timestamp: 0,
                amplitude: 1.0 
            }]
        } else {
            vec![]
        };

        Ok(AgentUpdateResult::new(output_messages, None, None, false))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Cognitive
    }
}

#[derive(Debug)]
struct BenchState {
    counter: u64,
    spike_count: u64,
}

async fn setup_connected_scheduler(num_agents: usize, connection_density: f32) -> CoreScheduler {
    let topology_arc = std::sync::Arc::new(tokio::sync::RwLock::new(
        ZoooidTopology::new(),
    ));
    let message_bus = std::sync::Arc::new(LocalBus::new());
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    let mut scheduler = CoreScheduler::new(
        topology_arc.clone(),
        rule_engine,
        resource_manager,
        message_bus.clone(),
    );

    // Spawn agents
    let mut agent_ids = Vec::new();
    for _i in 0..num_agents {
        let agent = Zoooid::new(ZoooidId::new_v4(), Box::new(BenchCognitiveAgent))
            .await
            .unwrap();
        let id = agent.id;
        scheduler.spawn_agent(agent).await.unwrap();
        agent_ids.push(id);
    }

    // Add connections based on density
    {
        let mut topo = topology_arc.write().await;
        let connections_per_agent = ((num_agents as f32) * connection_density) as usize;
        for (i, &agent_id) in agent_ids.iter().enumerate() {
            for j in 1..=connections_per_agent.min(num_agents - 1) {
                let target_idx = (i + j) % agent_ids.len();
                let _ = topo.try_add_edge(
                    agent_id,
                    agent_ids[target_idx],
                    Default::default(),
                );
            }
        }
    }

    scheduler
}

/// Benchmark PERF-001: RwLock contention reduction
/// Measures topology read throughput under concurrent agent updates
fn bench_topology_read_contention(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("perf_001_rwlock_contention");
    group.sample_size(10);

    for num_agents in [50, 100, 239].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_agents),
            num_agents,
            |b, &num_agents| {
                b.iter_batched(
                    || rt.block_on(setup_connected_scheduler(num_agents, 0.1)),
                    |mut scheduler| {
                        rt.block_on(async {
                            scheduler.run_one_iteration().await.unwrap();
                        });
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark PERF-002: Metrics cache effectiveness
/// Measures topology metrics calculation with cached values versus fresh recompute
fn bench_metrics_cache_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("perf_002_metrics_cache");
    group.sample_size(20);

    for num_agents in [50, 100, 239].iter() {
        // Prepare scheduler and warm cache once for clean-cache measurement
        let mut scheduler = rt.block_on(setup_connected_scheduler(*num_agents, 0.05));
        let topology = rt.block_on(scheduler.topology.read());
        let _ = adaptiflux_core::SystemMetrics::from_topology(&topology);
        drop(topology);

        group.bench_with_input(
            BenchmarkId::new("clean_cache", num_agents),
            num_agents,
            |b, _| {
                b.iter(|| {
                    let topology = rt.block_on(scheduler.topology.read());
                    let metrics = adaptiflux_core::SystemMetrics::from_topology(&topology);
                    black_box(metrics);
                })
            },
        );

        // Invalidate cache to measure dirty recomputation
        rt.block_on(async {
            let mut topo = scheduler.topology.write().await;
            topo.add_node(ZoooidId::new_v4());
        });

        group.bench_with_input(
            BenchmarkId::new("dirty_cache", num_agents),
            num_agents,
            |b, _| {
                b.iter(|| {
                    let topology = rt.block_on(scheduler.topology.read());
                    let metrics = adaptiflux_core::SystemMetrics::from_topology(&topology);
                    black_box(metrics);
                })
            },
        );
    }
    group.finish();
}

/// Benchmark PERF-003: SIMD neuron batch processing
/// Measures performance of batched Izhikevich updates with f32x4 vectors
fn bench_simd_neuron_performance(c: &mut Criterion) {
    use adaptiflux_core::primitives::spiking::izhikevich::simd::{IzhikevichBatch, pack_input_currents};

    let mut group = c.benchmark_group("perf_003_simd_neurons");
    group.sample_size(20);

    for neuron_count in [4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_update", neuron_count),
            neuron_count,
            |b, &neuron_count| {
                b.iter_batched(
                    || {
                        let batch = IzhikevichBatch::new(neuron_count, 0.02, 0.2, -65.0, 8.0);
                        let inputs = pack_input_currents(&vec![10.0; neuron_count]);
                        (batch, inputs)
                    },
                    |(mut batch, inputs)| {
                        for _ in 0..100 {
                            batch.update(&inputs, 0.1);
                        }
                        black_box(batch)
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark end-to-end scheduler iteration time
/// Shows combined effect of all optimizations
fn bench_scheduler_end_to_end(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("scheduler_e2e_iterations");
    group.sample_size(10);

    for num_agents in [50, 100, 239].iter() {
        group.bench_with_input(
            BenchmarkId::new("iterations", num_agents),
            num_agents,
            |b, &num_agents| {
                b.iter_batched(
                    || rt.block_on(setup_connected_scheduler(num_agents, 0.1)),
                    |mut scheduler| {
                        rt.block_on(async {
                            for _ in 0..20 {
                                scheduler.run_one_iteration().await.unwrap();
                            }
                        });
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_topology_read_contention,
    bench_metrics_cache_efficiency,
    bench_simd_neuron_performance,
    bench_scheduler_end_to_end
);
criterion_main!(benches);
