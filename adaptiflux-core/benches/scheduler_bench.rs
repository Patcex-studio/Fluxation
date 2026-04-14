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
    AgentBlueprint, AgentUpdateResult, CoreScheduler, LocalBus, Message, ResourceManager, RoleType,
    RuleEngine, Zoooid, ZoooidId,
};
use async_trait::async_trait;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::any::Any;

/// Simple test agent for benchmarking
struct BenchAgent;

#[async_trait]
impl AgentBlueprint for BenchAgent {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(BenchState { counter: 0 }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<Message>,
        _topology: &adaptiflux_core::ZoooidTopology,
        _memory: Option<&adaptiflux_core::MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(bench_state) = state.downcast_mut::<BenchState>() {
            bench_state.counter += 1;
        }

        let output_messages = inputs
            .into_iter()
            .map(|msg| match msg {
                Message::Text(text) => Message::Text(format!("echo: {}", text)),
                other => other,
            })
            .collect();

        Ok(AgentUpdateResult::new(output_messages, None, None, false))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Sensor
    }
}

#[derive(Debug)]
struct BenchState {
    counter: u64,
}

async fn setup_scheduler(num_agents: usize) -> CoreScheduler {
    let topology = std::sync::Arc::new(tokio::sync::Mutex::new(
        adaptiflux_core::ZoooidTopology::new(),
    ));
    let message_bus = std::sync::Arc::new(LocalBus::new());
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    let mut scheduler = CoreScheduler::new(
        topology.clone(),
        rule_engine,
        resource_manager,
        message_bus.clone(),
    );

    // Spawn agents
    for _i in 0..num_agents {
        let agent = Zoooid::new(ZoooidId::new_v4(), Box::new(BenchAgent))
            .await
            .unwrap();
        scheduler.spawn_agent(agent).await.unwrap();
    }

    scheduler
}

fn bench_scheduler_iteration(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("scheduler_iteration");

    for num_agents in [10, 50, 100, 239].iter() {
        group.bench_function(format!("{}_agents", num_agents), |b| {
            let mut scheduler = rt.block_on(setup_scheduler(*num_agents));
            b.iter(|| {
                rt.block_on(async {
                    scheduler.run_one_iteration().await.unwrap();
                    black_box(());
                });
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_scheduler_iteration);
criterion_main!(benches);
