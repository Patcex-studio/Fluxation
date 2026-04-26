use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use adaptiflux_core::agent::synapse_manager::{SynapseManager, SynapseConfig, NormMode};
use uuid::Uuid;

fn synthetic_id(n: u32) -> Uuid {
    Uuid::new_v5(&Uuid::NAMESPACE_DNS, &n.to_le_bytes())
}

/// Benchmark: O(1) weight lookup with 50 synapses
fn bench_get_weight_o1(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_lookup");
    
    for num_synapses in [10, 25, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_synapses),
            num_synapses,
            |b, &n| {
                let mut manager = SynapseManager::new(SynapseConfig {
                    max_connections: 200,
                    ..Default::default()
                });

                // Populate with n synapses
                for i in 0..n {
                    let _ = manager.add_synapse(synthetic_id(i as u32), 0.1);
                }

                b.iter(|| {
                    // Look up weight for middle synapse
                    let mid = n / 2;
                    black_box(manager.get_weight(synthetic_id(mid as u32)))
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Adding synapses (O(1) amortized)
fn bench_add_synapse(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_add");

    for num_adds in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_adds),
            num_adds,
            |b, &n| {
                let mut config = SynapseConfig::default();
                config.max_connections = n * 2;

                b.iter(|| {
                    let mut manager = SynapseManager::new(config.clone());
                    for i in 0..n {
                        let _ = black_box(manager.add_synapse(synthetic_id(i as u32), 0.1));
                    }
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: Removing synapses (includes swap-remove overhead)
fn bench_remove_synapse(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_remove");

    for num_ops in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_ops),
            num_ops,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut manager = SynapseManager::new(SynapseConfig {
                            max_connections: n * 2,
                            ..Default::default()
                        });
                        for i in 0..n {
                            let _ = manager.add_synapse(synthetic_id(i as u32), 0.1);
                        }
                        manager
                    },
                    |mut manager| {
                        // Remove all synapses sequentially
                        for i in 0..n {
                            black_box(manager.remove_synapse(synthetic_id(i as u32)));
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark: Weight normalization (L1)
fn bench_normalize_l1(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_normalize");

    for num_synapses in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_synapses),
            num_synapses,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut config = SynapseConfig::default();
                        config.norm_mode = NormMode::L1;
                        config.max_connections = n * 2;
                        let mut manager = SynapseManager::new(config);
                        
                        for i in 0..n {
                            let weight = (i as f32) / (n as f32);
                            let _ = manager.add_synapse(synthetic_id(i as u32), weight);
                        }
                        manager
                    },
                    |mut manager| {
                        black_box(manager.normalize());
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

/// Benchmark: Topology event processing
fn bench_topology_events(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_events");

    for num_synapses in [10, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("edge_added", num_synapses),
            num_synapses,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut config = SynapseConfig::default();
                        config.max_connections = n * 2;
                        SynapseManager::new(config)
                    },
                    |mut manager| {
                        for i in 0..n {
                            let event = adaptiflux_core::core::TopologyEvent::EdgeAdded {
                                from: synthetic_id(i as u32),
                                to: synthetic_id(999),
                                initial_weight: 0.1,
                            };
                            black_box(manager.on_topology_event(&event));
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    for num_synapses in [10, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("edge_removed", num_synapses),
            num_synapses,
            |b, &n| {
                b.iter_batched(
                    || {
                        let mut config = SynapseConfig::default();
                        config.max_connections = n * 2;
                        let mut manager = SynapseManager::new(config);
                        for i in 0..n {
                            let _ = manager.add_synapse(synthetic_id(i as u32), 0.1);
                        }
                        manager
                    },
                    |mut manager| {
                        for i in 0..n {
                            let event = adaptiflux_core::core::TopologyEvent::EdgeRemoved {
                                from: synthetic_id(i as u32),
                                to: synthetic_id(999),
                            };
                            black_box(manager.on_topology_event(&event));
                        }
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory overhead comparison
/// This is a simple measurement, not a traditional benchmark
fn bench_memory_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("synapse_memory");
    group.measurement_time(std::time::Duration::from_secs(1));

    group.bench_function("synapse_manager_50_connections", |b| {
        b.iter(|| {
            let mut manager = SynapseManager::new(SynapseConfig::default());
            for i in 0..50 {
                let _ = black_box(manager.add_synapse(synthetic_id(i), 0.5));
            }
            let size = std::mem::size_of_val(&manager);
            black_box(size)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_get_weight_o1,
    bench_add_synapse,
    bench_remove_synapse,
    bench_normalize_l1,
    bench_topology_events,
    bench_memory_size,
);

criterion_main!(benches);
