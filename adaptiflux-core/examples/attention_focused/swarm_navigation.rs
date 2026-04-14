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

//! Swarm agent uses memory + hard attention to process only the strongest relevant pheromone cue.
//!
//! Run: `cargo run --example swarm_navigation -p adaptiflux-core`

use std::any::Any;
use std::sync::Arc;

use adaptiflux_core::agent::blueprint::swarmzooid::{SwarmzooidBlueprint, SwarmzooidParams};
use adaptiflux_core::agent::blueprint::AgentBlueprint;
use adaptiflux_core::agent::state::{AgentUpdateResult, RoleType};
use adaptiflux_core::attention::{HardAttentionSelector, PheromoneFocus};
use adaptiflux_core::core::message_bus::message::Message;
use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::memory::types::MemoryPayload;
use adaptiflux_core::memory::{
    store_scalar_experience, ExperienceRecorder, MetadataIndexer, TableLongTermStore,
};
use adaptiflux_core::primitives::swarm::pfsm::PfsmParams;
use adaptiflux_core::{
    CoreScheduler, LocalBus, MemoryAttentionHook, ResourceManager, Retriever, RuleEngine, Zoooid,
    ZoooidId,
};
use async_trait::async_trait;
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
struct PheromoneField {
    phase: u64,
}

#[async_trait]
impl AgentBlueprint for PheromoneField {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(Box::new(0u64))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        _inputs: Vec<Message>,
        _topology: &ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        let tick = state.downcast_mut::<u64>().ok_or("beacon state")?;
        *tick += 1;
        let p = *tick + self.phase;
        let a = 0.2 + 0.6 * ((p % 7) as f64 / 6.0);
        let b = 0.2 + 0.6 * (((p + 3) % 7) as f64 / 6.0);
        let c = 0.2 + 0.6 * (((p + 5) % 7) as f64 / 6.0);
        Ok(AgentUpdateResult::new(
            vec![
                Message::PheromoneLevel(a, 0),
                Message::PheromoneLevel(b, 1),
                Message::PheromoneLevel(c, 2),
            ],
            None,
            None,
            false,
        ))
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Custom("pheromone_field".into())
    }
}

#[derive(Debug, Clone)]
struct PheromoneContext;

struct PheromoneExperienceSink {
    nav_id: ZoooidId,
}

fn pheromone_embedding(inputs: &[Message]) -> Vec<f32> {
    let mut v: Vec<f32> = inputs
        .iter()
        .filter_map(|m| {
            if let Message::PheromoneLevel(s, _) = m {
                Some(*s as f32)
            } else {
                None
            }
        })
        .collect();
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    while v.len() < 4 {
        v.push(0.0);
    }
    v.truncate(4);
    v
}

impl ExperienceRecorder for PheromoneExperienceSink {
    fn record_after_step(
        &self,
        agent_id: ZoooidId,
        iteration: u64,
        inputs: &[Message],
        _state: &dyn Any,
        _result: &AgentUpdateResult,
        store: &mut TableLongTermStore,
        indexer: &mut MetadataIndexer,
    ) {
        if agent_id != self.nav_id {
            return;
        }
        let emb = pheromone_embedding(inputs);
        let payload = Arc::new(PheromoneContext);
        store_scalar_experience(
            store,
            indexer,
            agent_id,
            iteration,
            "pheromone_context",
            Some(emb),
            payload,
            0.0,
        );
    }
}

/// When memory is non-empty, forwards only the strongest pheromone reading to the inner swarm primitive.
struct AttentionSwarm {
    inner: SwarmzooidBlueprint,
}

#[async_trait]
impl AgentBlueprint for AttentionSwarm {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        self.inner.initialize().await
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<Message>,
        topology: &ZoooidTopology,
        memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
        let inputs = if memory.map(|m| !m.is_empty()).unwrap_or(false) {
            let mut best: Option<Message> = None;
            let mut best_s = f64::NEG_INFINITY;
            for m in &inputs {
                if let Message::PheromoneLevel(s, _) = m {
                    if *s > best_s {
                        best_s = *s;
                        best = Some(m.clone());
                    }
                }
            }
            best.into_iter().collect::<Vec<_>>()
        } else {
            inputs
        };
        self.inner.update(state, inputs, topology, None).await
    }

    fn blueprint_type(&self) -> RoleType {
        RoleType::Swarm
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    let nav_id = ZoooidId::from_u128(10);
    let field_a = ZoooidId::from_u128(11);
    let field_b = ZoooidId::from_u128(12);

    let store = Arc::new(Mutex::new(TableLongTermStore::new()));
    let indexer = Arc::new(Mutex::new(MetadataIndexer::new()));

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
    scheduler.set_cycle_frequency(30);

    scheduler.memory_attention = Some(MemoryAttentionHook {
        store: Arc::clone(&store),
        indexer: Arc::clone(&indexer),
        retriever: Retriever::new(5),
        attention: Arc::new(HardAttentionSelector),
        focus: Arc::new(PheromoneFocus { top_signals: 3 }),
        target_ids: Some(vec![nav_id]),
        inject_memory_into_feedback: false,
        memory_feedback_gain: 0.0,
        experience: Some(Arc::new(PheromoneExperienceSink { nav_id })),
    });

    let nav = Zoooid::new(
        nav_id,
        Box::new(AttentionSwarm {
            inner: SwarmzooidBlueprint {
                params: SwarmzooidParams {
                    pfsm_params: PfsmParams::default(),
                    connection_request_interval: u64::MAX,
                },
            },
        }),
    )
    .await?;

    let f1 = Zoooid::new(
        field_a,
        Box::new(PheromoneField { phase: 0 }),
    )
    .await?;
    let f2 = Zoooid::new(
        field_b,
        Box::new(PheromoneField { phase: 2 }),
    )
    .await?;

    scheduler.spawn_agent(nav).await?;
    scheduler.spawn_agent(f1).await?;
    scheduler.spawn_agent(f2).await?;

    {
        let mut g = topology.lock().await;
        for fid in [field_a, field_b] {
            g.add_edge(fid, nav_id, Default::default());
            g.add_edge(nav_id, fid, Default::default());
        }
    }

    println!("Swarm navigator {:?} + pheromone sources {:?}, {:?}", nav_id, field_a, field_b);
    for _ in 0..40 {
        scheduler.run_one_iteration().await?;
    }

    let mem_n = store.lock().await.len();
    println!("Stored pheromone-context experiences: {}", mem_n);

    Ok(())
}
