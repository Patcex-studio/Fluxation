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

use adaptiflux_core::core::message_bus::message::Message;
use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::memory::types::MemoryPayload;
use adaptiflux_core::{
    AgentBlueprint, AgentUpdateResult, CoreScheduler, LocalBus, ResourceManager, RoleType,
    RuleEngine, Zoooid, ZoooidId,
};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::any::Any;

struct EchoBlueprint;

#[async_trait]
impl AgentBlueprint for EchoBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(()))
    }

    async fn update(
        &self,
        _state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<(adaptiflux_core::ZoooidId, Message)>,
        _topology: &ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        let response = inputs
            .into_iter()
            .map(|(_sender, message)| match message {
                Message::Text(text) => Message::Text(format!("echo: {}", text)),
                other => other,
            })
            .collect();

        Ok(AgentUpdateResult::new(response, None, None, false))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Sensor
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let message_bus = Arc::new(LocalBus::new());
    let topology = Arc::new(RwLock::new(ZoooidTopology::new()));
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    let mut scheduler = CoreScheduler::new(
        topology.clone(),
        rule_engine,
        resource_manager,
        message_bus.clone(),
    );

    let agent = Zoooid::new(ZoooidId::new_v4(), Box::new(EchoBlueprint)).await?;
    scheduler.spawn_agent(agent).await?;

    println!("Adaptiflux core initialized with one echo zood.");
    Ok(())
}
