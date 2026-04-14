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

//! Self-healing connectivity: remove a node; activity-dependent synaptogenesis restores links.
//!
//! Run: `cargo run --example self_healing_network -p adaptiflux-core`

use adaptiflux_core::memory::types::MemoryPayload;
use adaptiflux_core::{
    ActivityDependentSynaptogenesisRule, AgentBlueprint, AgentUpdateResult, CoreScheduler, LocalBus,
    Message, ResourceManager, RoleType, RuleEngine, TopologyChange, Zoooid, ZoooidId, ZoooidTopology,
};
use async_trait::async_trait;
use std::any::Any;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone)]
struct ChattyAgent {
    label: &'static str,
}

#[derive(Default)]
struct ChattyState {
    tick: u64,
}

#[async_trait]
impl AgentBlueprint for ChattyAgent {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(ChattyState::default()))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        _inputs: Vec<Message>,
        topology: &ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        let st = state.downcast_mut::<ChattyState>().unwrap();
        st.tick += 1;
        let mut out = Vec::new();
        if st.tick % 2 == 0 {
            out.push(Message::Text(format!("{} ping {}", self.label, st.tick)));
        }
        let change = if st.tick % 3 == 0 {
            let nodes: Vec<ZoooidId> = topology.graph.nodes().collect();
            if !nodes.is_empty() {
                let t = nodes[(st.tick as usize) % nodes.len()];
                Some(TopologyChange::RequestConnection(t))
            } else {
                None
            }
        } else {
            None
        };
        Ok(AgentUpdateResult::new(out, None, change, false))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Sensor
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
    let bus = Arc::new(LocalBus::new());
    let mut engine = RuleEngine::new();
    engine.add_plasticity_rule(Box::new(ActivityDependentSynaptogenesisRule {
        activity_threshold: 0.5,
        max_new_edges: 4,
        stdp_traffic_threshold: None,
        stdp_delta: 0.05,
    }));

    let mut scheduler = CoreScheduler::new(
        topology.clone(),
        engine,
        ResourceManager::new(),
        bus,
    );
    scheduler.set_cycle_frequency(200);

    let mut ids = Vec::new();
    for i in 0..4 {
        let id = ZoooidId::new_v4();
        ids.push(id);
        let z = Zoooid::new(
            id,
            Box::new(ChattyAgent {
                label: match i {
                    0 => "a",
                    1 => "b",
                    2 => "c",
                    _ => "d",
                },
            }),
        )
        .await?;
        scheduler.spawn_agent(z).await?;
    }

    {
        let mut t = topology.lock().await;
        for i in 0..4 {
            let a = ids[i];
            let b = ids[(i + 1) % 4];
            t.add_edge(a, b, Default::default());
        }
    }

    let edges_before = scheduler.topology.lock().await.graph.edge_count();

    scheduler.run_for_iterations(40).await?;

    let victim = ids[2];
    scheduler.remove_agent_from_system(victim).await;

    scheduler.run_for_iterations(60).await?;

    let edges_after = scheduler.topology.lock().await.graph.edge_count();
    let agents_after = scheduler.agent_count();

    println!("self_healing_network demo");
    println!("  edges before failure / healing window: {}", edges_before);
    println!("  edges after plasticity iterations: {}", edges_after);
    println!("  agents after removal: {}", agents_after);
    Ok(())
}
