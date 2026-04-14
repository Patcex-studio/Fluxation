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

//! Swarm agents wire themselves into components; [`ClusterGroupingPlasticityRule`] and the
//! scheduler [`HierarchyHook`] maintain [`AbstractionLayerManager`] groups for coarse control.

use adaptiflux_core::agent::blueprint::swarmzooid::{SwarmzooidBlueprint, SwarmzooidParams};
use adaptiflux_core::{
    AbstractionLayerManager, AggregationFnKind, ClusterGroupingPlasticityRule, CoreScheduler,
    HierarchyHook, LocalBus, ResourceManager, RuleEngine, Zoooid, ZoooidId, ZoooidTopology,
};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let bus = Arc::new(LocalBus::new());
    let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
    let mut rule_engine = RuleEngine::new();
    rule_engine.add_plasticity_rule(Box::new(ClusterGroupingPlasticityRule {
        min_cluster_size: 3,
        evaluate_every: 8,
    }));

    let resource_manager = ResourceManager::new();
    let mut scheduler = CoreScheduler::new(
        topology.clone(),
        rule_engine,
        resource_manager,
        bus,
    );
    scheduler.set_cycle_frequency(50);

    scheduler.hierarchy = Some(HierarchyHook {
        manager: AbstractionLayerManager::default(),
        detect_every: 12,
        min_cluster_size: 3,
        aggregation: AggregationFnKind::Mean,
    });

    let template = SwarmzooidParams {
        pfsm_params: Default::default(),
        connection_request_interval: 3,
    };

    let mut ids = Vec::new();
    for i in 0..18 {
        let id = ZoooidId::from_u128(0x7000 + i as u128);
        ids.push(id);
        let agent = Zoooid::new(
            id,
            Box::new(SwarmzooidBlueprint {
                params: template.clone(),
            }),
        )
        .await?;
        scheduler.spawn_agent(agent).await?;
    }

    println!("Spawned {} swarm agents; running topology + clustering…", ids.len());
    scheduler.run_for_iterations(64).await?;

    let groups = scheduler
        .hierarchy
        .as_ref()
        .map(|h| h.manager.group_count())
        .unwrap_or(0);
    println!("Abstraction groups tracked: {}", groups);
    if let Some(h) = &scheduler.hierarchy {
        for (i, g) in h.manager.groups().iter().take(5).enumerate() {
            println!("  group {}: {} members", i, g.members.len());
        }
    }

    Ok(())
}
