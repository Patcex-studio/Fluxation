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

//! Online PID tuning using `custom_optim` via `CustomOptimizerLearner`.
//!
//! Build with:
//!   cargo run --example custom_optim_pid --features custom_optim

use adaptiflux_core::agent::blueprint::AgentBlueprint;
use adaptiflux_core::agent::state::{AgentUpdateResult, RoleType};
use adaptiflux_core::core::message_bus::message::Message;
use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::memory::types::MemoryPayload;
use async_trait::async_trait;
use std::any::Any;

#[cfg(feature = "custom_optim")]
use custom_optim::{BackendType, OptimizerConfig, OptimizerStrategyType};

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PlantBlueprint {
    target: f32,
}

#[async_trait]
impl AgentBlueprint for PlantBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(0.0_f32))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<Message>,
        _topology: &ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        let pos = state.downcast_mut::<f32>().ok_or("plant state")?;
        let u: f32 = inputs
            .iter()
            .filter_map(|m| match m {
                Message::ControlSignal(x) => Some(*x),
                _ => None,
            })
            .sum();
        *pos += u * 0.08;
        *pos = pos.clamp(-3.0, 3.0);
        let err = self.target - *pos;
        Ok(AgentUpdateResult::new(
            vec![Message::Error(err)],
            None,
            None,
            false,
        ))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Custom("plant".into())
    }
}

#[cfg(not(feature = "custom_optim"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    eprintln!("Enable the custom_optim feature to run this example.");
    Ok(())
}

#[cfg(feature = "custom_optim")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let plant_id = ZoooidId::from_u128(1);
    let pid_id = ZoooidId::from_u128(2);

    let bus = Arc::new(LocalBus::new());
    let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    let mut scheduler =
        CoreScheduler::new(topology.clone(), rule_engine, resource_manager, bus.clone());
    scheduler.set_cycle_frequency(20);

    let mut engine = OnlineAdaptationEngine::new();
    let config = OptimizerConfig {
        strategy: OptimizerStrategyType::Hybrid,
        backend: if cfg!(feature = "gpu") {
            BackendType::CUDA
        } else {
            BackendType::CPU
        },
        population_size: Some(16),
        learning_rate: Some(0.05),
        generations: Some(4),
        iterations: Some(25),
    };
    let learner = adaptiflux_core::CustomOptimizerLearner::new(config)?;
    engine.set_default_learner(Arc::new(learner));

    scheduler.online_adaptation = Some(OnlineAdaptationHook {
        engine,
        target_ids: vec![pid_id],
    });

    let plant = Zoooid::new(plant_id, Box::new(PlantBlueprint { target: 1.0 })).await?;
    let pid_agent = Zoooid::new(
        pid_id,
        Box::new(PIDzooidBlueprint {
            params: PIDzooidParams {
                pid_params: PidParams {
                    kp: 0.2,
                    ki: 0.02,
                    kd: 0.002,
                    dt: 0.1,
                },
                connection_request_interval: u64::MAX,
            },
        }),
    )
    .await?;

    scheduler.spawn_agent(plant).await?;
    scheduler.spawn_agent(pid_agent).await?;

    {
        let mut g = topology.lock().await;
        g.add_edge(plant_id, pid_id, Default::default());
        g.add_edge(pid_id, plant_id, Default::default());
    }

    println!("Running online PID tuning with custom_optim...");
    for i in 0..20 {
        scheduler.run_one_iteration().await?;
        if i % 5 == 0 {
            let kp = scheduler
                .agents
                .get(&pid_id)
                .and_then(|h| {
                    h.state
                        .downcast_ref::<adaptiflux_core::agent::blueprint::pidzooid::PIDzooidState>(
                        )
                })
                .map(|s| s.pid_params.kp)
                .unwrap_or(0.0);
            let pos = scheduler
                .agents
                .get(&plant_id)
                .and_then(|h| h.state.downcast_ref::<f64>())
                .copied()
                .unwrap_or(0.0);
            println!("step {:>2}: plant_pos {:.4}, Kp {:.4}", i, pos, kp);
        }
    }

    let final_params = scheduler
        .agents
        .get(&pid_id)
        .and_then(|h| {
            h.state
                .downcast_ref::<adaptiflux_core::agent::blueprint::pidzooid::PIDzooidState>()
        })
        .map(|s| (s.pid_params.kp, s.pid_params.ki, s.pid_params.kd))
        .unwrap_or((0.0, 0.0, 0.0));
    println!(
        "Final PID gains: Kp={:.4} Ki={:.4} Kd={:.4}",
        final_params.0, final_params.1, final_params.2
    );
    Ok(())
}
