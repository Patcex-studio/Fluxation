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

use adaptiflux_core::agent::blueprint::cognitivezooid::CognitivezooidBlueprint;
use adaptiflux_core::agent::blueprint::pidzooid::PIDzooidBlueprint;
use adaptiflux_core::agent::blueprint::sensorzooid::SensorzooidBlueprint;
use adaptiflux_core::agent::zoooid::Zoooid;
use adaptiflux_core::core::message_bus::LocalBus;
use adaptiflux_core::core::resource_manager::ResourceManager;
use adaptiflux_core::core::scheduler::{CoreScheduler, OnlineAdaptationHook};
use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::learning::online_adaptation::{
    GradientDescentLearner, OnlineAdaptationEngine,
};
use adaptiflux_core::primitives::control::pid::PidParams;
use adaptiflux_core::primitives::spiking::izhikevich::IzhikevichParams;
use adaptiflux_core::primitives::spiking::lif::LifParams;
use adaptiflux_core::rules::RuleEngine;
use adaptiflux_core::utils::types::new_zoooid_id;
use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::Arc;

fn bench_mnist_integration(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    c.bench_function("mnist_network_iteration", |b| {
        b.iter(|| {
            rt.block_on(async {
                let topology = Arc::new(tokio::sync::RwLock::new(ZoooidTopology::new()));
                let rule_engine = RuleEngine::new();
                let resource_manager = ResourceManager {
                    cpu_pool: 4,
                    gpu_pool: 0,
                };
                let message_bus = Arc::new(LocalBus::new());

                let mut scheduler =
                    CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

                let mut input_ids = Vec::new();
                let mut hidden_ids = Vec::new();
                let mut output_ids = Vec::new();

                for _ in 0..10 {
                    let id = new_zoooid_id();
                    input_ids.push(id);
                    let sensor_params =
                        adaptiflux_core::agent::blueprint::sensorzooid::SensorzooidParams {
                            lif_params: LifParams::default(),
                            connection_request_interval: 10,
                        };
                    let blueprint = SensorzooidBlueprint {
                        params: sensor_params,
                    };
                    let agent = Zoooid::new(id, Box::new(blueprint)).await.unwrap();
                    scheduler.spawn_agent(agent).await.unwrap();
                }

                for _ in 0..5 {
                    let id = new_zoooid_id();
                    hidden_ids.push(id);
                    let cognitive_params =
                        adaptiflux_core::agent::blueprint::cognitivezooid::CognitivezooidParams {
                            izh_params: IzhikevichParams::default(),
                            connection_request_interval: 10,
                            stdp_a_plus: 0.01,
                            stdp_a_minus: 0.005,
                            stdp_tau_plus: 20.0,
                            stdp_tau_minus: 20.0,
                            weight_decay: 0.0001,
                            pruning_threshold: 0.001,
                            neuron_count: 1,
                        };
                    let blueprint = CognitivezooidBlueprint {
                        params: cognitive_params,
                    };
                    let agent = Zoooid::new(id, Box::new(blueprint)).await.unwrap();
                    scheduler.spawn_agent(agent).await.unwrap();
                }

                for _ in 0..2 {
                    let id = new_zoooid_id();
                    output_ids.push(id);
                    let pid_params = PidParams::default();
                    let zoo_params = adaptiflux_core::agent::blueprint::pidzooid::PIDzooidParams {
                        pid_params,
                        connection_request_interval: 10,
                    };
                    let blueprint = PIDzooidBlueprint { params: zoo_params };
                    let agent = Zoooid::new(id, Box::new(blueprint)).await.unwrap();
                    scheduler.spawn_agent(agent).await.unwrap();
                }

                for &src in &input_ids {
                    for &dst in &hidden_ids {
                        scheduler
                            .topology
                            .write()
                            .await
                            .add_edge(src, dst, Default::default());
                    }
                }
                for &src in &hidden_ids {
                    for &dst in &output_ids {
                        scheduler
                            .topology
                            .write()
                            .await
                            .add_edge(src, dst, Default::default());
                    }
                }

                let mut adaptation_engine = OnlineAdaptationEngine::new();
                let learner = Arc::new(GradientDescentLearner::default());
                adaptation_engine.set_default_learner(learner);

                let target_ids = input_ids
                    .iter()
                    .chain(hidden_ids.iter())
                    .chain(output_ids.iter())
                    .cloned()
                    .collect();
                let adaptation_hook = OnlineAdaptationHook {
                    engine: adaptation_engine,
                    target_ids,
                };
                scheduler.online_adaptation = Some(adaptation_hook);

                scheduler.run_one_iteration().await.unwrap();
            })
        });
    });
}

criterion_group!(benches, bench_mnist_integration);
criterion_main!(benches);
