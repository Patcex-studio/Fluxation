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
use adaptiflux_core::memory::types::MemoryPayload;
use adaptiflux_core::{
    core::message_bus::message::Message, core::message_bus::LocalBus, AgentBlueprint,
    AgentUpdateResult, CoreScheduler, ResourceManager, RoleType, RuleEngine, Zoooid, ZoooidId,
};
use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;
use tokio::sync::Mutex;

struct PidAgent;

#[async_trait]
impl AgentBlueprint for PidAgent {
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
        let measurement = inputs
            .iter()
            .filter_map(|message| match message {
                Message::ControlSignal(value) => Some(*value),
                _ => None,
            })
            .sum::<f32>();

        let error = 1.0_f32 - measurement;
        if let Some(current) = state.downcast_mut::<f32>() {
            *current = measurement;
        }

        let output = Message::ControlSignal(error);
        Ok(AgentUpdateResult::new(vec![output], None, None, false))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Pid
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let message_bus = Arc::new(LocalBus::new());
    let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    let mut scheduler = CoreScheduler::new(
        topology.clone(),
        rule_engine,
        resource_manager,
        message_bus.clone(),
    );
    let agent = Zoooid::new(ZoooidId::new_v4(), Box::new(PidAgent)).await?;

    scheduler.spawn_agent(agent).await?;
    println!("PID control loop example created.");
    Ok(())
}
