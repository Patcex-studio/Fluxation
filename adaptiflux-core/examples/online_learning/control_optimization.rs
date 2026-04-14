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

//! PID plant under online SGD tuning of `Kp`, `Ki`, `Kd` via [`OnlineAdaptationHook`].
//!
//! A deterministic agent order (`plant` id `<` `pid` id) lets the plant post `Message::Error`
//! each tick before the controller consumes it on the following iteration.

use std::sync::Arc;

use adaptiflux_core::agent::blueprint::pidzooid::{PIDzooidBlueprint, PIDzooidParams, PIDzooidState};
use adaptiflux_core::agent::blueprint::AgentBlueprint;
use adaptiflux_core::agent::state::{AgentUpdateResult, RoleType};
use adaptiflux_core::core::message_bus::message::Message;
use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::primitives::control::pid::PidParams;
use adaptiflux_core::memory::types::MemoryPayload;
use adaptiflux_core::{
    CoreScheduler, GradientDescentLearner, LocalBus, OnlineAdaptationEngine, OnlineAdaptationHook,
    ResourceManager, RuleEngine, Zoooid, ZoooidId,
};

#[cfg(feature = "adaptiflux_optim")]
use adaptiflux_core::AdaptiveOptimizerLearner;
use async_trait::async_trait;
use std::any::Any;
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
struct PlantBlueprint {
    target: f64,
}

#[async_trait]
impl AgentBlueprint for PlantBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(0.0_f64))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<Message>,
        _topology: &ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        let pos = state
            .downcast_mut::<f64>()
            .ok_or("plant state")?;
        let u: f64 = inputs
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

    let mut scheduler = CoreScheduler::new(
        topology.clone(),
        rule_engine,
        resource_manager,
        bus.clone(),
    );
    scheduler.set_cycle_frequency(20);

    let mut engine = OnlineAdaptationEngine::new();
    #[cfg(feature = "adaptiflux_optim")]
    engine.set_default_learner(Arc::new(AdaptiveOptimizerLearner::new(0.012, 4)));
    #[cfg(not(feature = "adaptiflux_optim"))]
    engine.set_default_learner(Arc::new(GradientDescentLearner {
        learning_rate: 0.012,
    }));
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

    println!("Online PID tuning — plant {:?} → controller {:?}", plant_id, pid_id);
    for i in 0..5 {
        scheduler.run_one_iteration().await?;
        if i == 0 {
            continue;
        }
        let kp = scheduler
            .agents
            .get(&pid_id)
            .and_then(|h| h.state.downcast_ref::<PIDzooidState>())
            .map(|s| s.pid_params.kp)
            .unwrap_or(0.0);
        let pos = scheduler
            .agents
            .get(&plant_id)
            .and_then(|h| h.state.downcast_ref::<f64>())
            .copied()
            .unwrap_or(0.0);
        println!("  step {:>2}: plant_pos ≈ {:.4}, Kp ≈ {:.4}", i, pos, kp);
    }

    for _ in 5..120 {
        scheduler.run_one_iteration().await?;
    }

    let kp = scheduler
        .agents
        .get(&pid_id)
        .and_then(|h| h.state.downcast_ref::<PIDzooidState>())
        .map(|s| (s.pid_params.kp, s.pid_params.ki, s.pid_params.kd))
        .unwrap_or((0.0, 0.0, 0.0));
    let pos = scheduler
        .agents
        .get(&plant_id)
        .and_then(|h| h.state.downcast_ref::<f64>())
        .copied()
        .unwrap_or(0.0);
    println!(
        "After 120 steps: plant_pos ≈ {:.4}, tuned gains Kp={:.4} Ki={:.4} Kd={:.4}",
        pos, kp.0, kp.1, kp.2
    );

    Ok(())
}
