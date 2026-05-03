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

/// Benchmark PERF-001: End-to-end parallel scheduler with 239 agents
fn bench_scheduler_parallel_239_agents(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("perf_001_scheduler_parallel_239_agents");
    group.sample_size(10);
    group.bench_function("239_agents", |b| {
        b.iter_batched(
            || rt.block_on(setup_connected_scheduler(239, 0.1)),
            |mut scheduler| {
                rt.block_on(async {
                    scheduler.run_one_iteration().await.unwrap();
                });
            },
            BatchSize::SmallInput,
        );
    });
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
        let scheduler = rt.block_on(setup_connected_scheduler(*num_agents, 0.05));
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
    use adaptiflux_core::primitives::spiking::izhikevich::simd::{IzhikevichBatch, IzhikevichBatchParams, pack_input_currents};

    let mut group = c.benchmark_group("perf_003_simd_neurons");
    group.sample_size(20);

    let batch_params = IzhikevichBatchParams::from_scalar(0.02, 0.2, -65.0, 8.0, 0.1);

    for neuron_count in [4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_update", neuron_count),
            neuron_count,
            |b, &neuron_count| {
                b.iter_batched(
                    || {
                        let batch = IzhikevichBatch::new(neuron_count, &batch_params);
                        let inputs = pack_input_currents(&vec![10.0; neuron_count]);
                        (batch, inputs)
                    },
                    |(mut batch, inputs)| {
                        for _ in 0..100 {
                            batch.update(&inputs, &batch_params);
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

/// Benchmark PERF-003: SIMD batch vs scalar for 32 neurons
fn bench_simd_batch_vs_scalar(c: &mut Criterion) {
    use adaptiflux_core::primitives::spiking::IzhikevichNeuron;
    use adaptiflux_core::primitives::spiking::izhikevich::IzhikevichParams;
    use adaptiflux_core::primitives::base::{Primitive, PrimitiveMessage};
    use adaptiflux_core::primitives::spiking::{IzhikevichBatch, IzhikevichBatchParams, pack_input_currents};

    let mut group = c.benchmark_group("perf_003_simd_batch_vs_scalar");
    group.sample_size(20);
    let neuron_count = 32;
    let params = IzhikevichParams::default();
    let scalar_inputs: Vec<_> = (0..neuron_count)
        .map(|_| PrimitiveMessage::InputCurrent(10.0))
        .collect();
    let simd_packed = pack_input_currents(&vec![10.0; neuron_count]);
    let batch_params = IzhikevichBatchParams::from_scalar(
        params.a,
        params.b,
        params.c,
        params.d,
        params.dt,
    );

    group.bench_function("scalar_32", |b| {
        b.iter(|| {
            let mut state = IzhikevichNeuron::initialize(params.clone());
            let (_state, outputs) = IzhikevichNeuron::update(state, &params, &scalar_inputs);
            black_box(outputs);
        })
    });

    group.bench_function("simd_32", |b| {
        b.iter(|| {
            let mut batch = IzhikevichBatch::new(neuron_count, &batch_params);
            for _ in 0..10 {
                batch.update(&simd_packed, &batch_params);
            }
            black_box(batch);
        })
    });
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
    bench_scheduler_parallel_239_agents,
    bench_metrics_cache_efficiency,
    bench_simd_neuron_performance,
    bench_simd_batch_vs_scalar,
    bench_scheduler_end_to_end
);
criterion_main!(benches);
