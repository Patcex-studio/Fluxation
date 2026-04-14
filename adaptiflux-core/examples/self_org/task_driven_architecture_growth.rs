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

//! Experimental: seed network + growth rule that spawns agents near activity hotspots.
//!
//! Run: `cargo run --example task_driven_architecture_growth -p adaptiflux-core`

use adaptiflux_core::agent::blueprint::swarmzooid::{SwarmzooidBlueprint, SwarmzooidParams};
use adaptiflux_core::plasticity::BlueprintFactory;
use adaptiflux_core::primitives::swarm::pfsm::PfsmParams;
use adaptiflux_core::{
    AgentBlueprint, CoreScheduler, GrowthFactorNeurogenesisRule, LocalBus, ResourceManager,
    RuleEngine, Zoooid, ZoooidId, ZoooidTopology,
};
use std::sync::Arc;
use tokio::sync::Mutex;

fn swarm_factory() -> Box<dyn AgentBlueprint + Send + Sync> {
    Box::new(SwarmzooidBlueprint {
        params: SwarmzooidParams {
            pfsm_params: PfsmParams::default(),
            connection_request_interval: 5,
        },
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt::init();

    let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
    let bus = Arc::new(LocalBus::new());
    let mut engine = RuleEngine::new();
    let factory: BlueprintFactory = Arc::new(|| swarm_factory());
    engine.add_plasticity_rule(Box::new(GrowthFactorNeurogenesisRule {
        activity_threshold: 2.0,
        blueprint_factory: factory,
    }));

    let mut scheduler = CoreScheduler::new(
        topology.clone(),
        engine,
        ResourceManager::new(),
        bus,
    );
    scheduler.set_cycle_frequency(200);

    for _ in 0..2 {
        let id = ZoooidId::new_v4();
        let z = Zoooid::new(id, swarm_factory()).await?;
        scheduler.spawn_agent(z).await?;
    }

    let ids: Vec<ZoooidId> = scheduler.agents.keys().copied().collect();
    if ids.len() == 2 {
        let mut t = topology.lock().await;
        t.add_edge(ids[0], ids[1], Default::default());
    }

    let start_agents = scheduler.agent_count();

    scheduler.run_for_iterations(120).await?;

    let grown = scheduler.agent_count();

    println!("task_driven_architecture_growth (experimental)");
    println!("  agents at start: {}", start_agents);
    println!("  agents after growth window: {}", grown);
    Ok(())
}
