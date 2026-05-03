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
use std::any::Any;
use tokio::time::Instant;

/// Simple test agent for timing
struct TimingAgent;

#[async_trait]
impl AgentBlueprint for TimingAgent {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(TimingState { counter: 0 }))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<(adaptiflux_core::ZoooidId, Message)>,
        _topology: &adaptiflux_core::ZoooidTopology,
        _memory: Option<&adaptiflux_core::MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(timing_state) = state.downcast_mut::<TimingState>() {
            timing_state.counter += 1;
        }

        let output_messages = inputs
            .into_iter()
            .map(|(_sender, msg)| match msg {
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
struct TimingState {
    counter: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Scheduler Timing Test");

    // Test with different agent counts
    for &num_agents in &[10, 50, 100, 239] {
        println!("\n=== Testing with {} agents ===", num_agents);

        // Create scheduler for this run
        let mut scheduler = CoreScheduler::new(
            std::sync::Arc::new(tokio::sync::RwLock::new(
                adaptiflux_core::ZoooidTopology::new(),
            )),
            RuleEngine::new(),
            ResourceManager::new(),
            std::sync::Arc::new(LocalBus::new()),
        );

        for _ in 0..num_agents {
            let agent = Zoooid::new(ZoooidId::new_v4(), Box::new(TimingAgent))
                .await
                .unwrap();
            scheduler.spawn_agent(agent).await.unwrap();
        }

        // Warm up
        for _ in 0..5 {
            scheduler.run_one_iteration().await.unwrap();
        }

        // Time 100 iterations
        let start = Instant::now();
        for _ in 0..100 {
            scheduler.run_one_iteration().await.unwrap();
        }
        let elapsed = start.elapsed();

        let avg_per_iter = elapsed / 100;
        println!(
            "Total time for 100 iterations: {:.2}s",
            elapsed.as_secs_f64()
        );
        println!(
            "Average time per iteration: {:.2}ms",
            avg_per_iter.as_millis()
        );
        println!(
            "Iterations per second: {:.1}",
            1000.0 / avg_per_iter.as_millis() as f64
        );
    }

    Ok(())
}
