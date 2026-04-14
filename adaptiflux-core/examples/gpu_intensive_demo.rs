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

#[cfg(feature = "gpu")]
use adaptiflux_core::agent::blueprint::AgentBlueprint;
#[cfg(feature = "gpu")]
use adaptiflux_core::agent::zoooid::Zoooid;
#[cfg(feature = "gpu")]
use adaptiflux_core::core::message_bus::{LocalBus, Message, MessageBus};
#[cfg(feature = "gpu")]
use adaptiflux_core::core::resource_manager::ResourceManager;
#[cfg(feature = "gpu")]
use adaptiflux_core::core::scheduler::CoreScheduler;
#[cfg(feature = "gpu")]
use adaptiflux_core::core::topology::ZoooidTopology;
#[cfg(feature = "gpu")]
use adaptiflux_core::gpu::primitive_wrappers::{
    BatchIzhikevichParams, BatchIzhikevichPrimitive, BatchIzhikevichState,
};
#[cfg(feature = "gpu")]
use adaptiflux_core::gpu::resource_manager::GpuResourceManager;
#[cfg(feature = "gpu")]
use adaptiflux_core::memory::types::MemoryPayload;
#[cfg(feature = "gpu")]
use adaptiflux_core::primitives::base::PrimitiveMessage;
#[cfg(feature = "gpu")]
use adaptiflux_core::rules::RuleEngine;
#[cfg(feature = "gpu")]
use adaptiflux_core::utils::types::new_zoooid_id;
#[cfg(feature = "gpu")]
use std::any::Any;
#[cfg(feature = "gpu")]
use std::sync::Arc;
#[cfg(feature = "gpu")]
use tokio::sync::Mutex;

#[cfg(feature = "gpu")]
struct GpuIzhikevichBlueprint {
    params: BatchIzhikevichParams,
    _gpu_manager: Arc<Mutex<GpuResourceManager>>,
}

#[cfg(feature = "gpu")]
impl GpuIzhikevichBlueprint {
    pub fn new(params: BatchIzhikevichParams, gpu_manager: Arc<Mutex<GpuResourceManager>>) -> Self {
        Self {
            params,
            _gpu_manager: gpu_manager,
        }
    }
}

#[cfg(feature = "gpu")]
#[async_trait::async_trait]
impl AgentBlueprint for GpuIzhikevichBlueprint {
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let state =
            <BatchIzhikevichPrimitive as adaptiflux_core::primitives::base::Primitive>::initialize(
                self.params.clone(),
            );
        Ok(Box::new(state))
    }

    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<Message>,
        _topology: &ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<
        adaptiflux_core::agent::state::AgentUpdateResult,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let primitive_inputs: Vec<PrimitiveMessage> = inputs
            .into_iter()
            .filter_map(|message| match message {
                Message::AnalogInput(value) => Some(PrimitiveMessage::InputCurrent(value)),
                _ => None,
            })
            .collect();

        let concrete_state = state
            .downcast_mut::<BatchIzhikevichState>()
            .ok_or("Failed to downcast GPU primitive state")?;

        let (new_state, primitive_outputs) =
            <BatchIzhikevichPrimitive as adaptiflux_core::primitives::base::Primitive>::update(
                concrete_state.clone(),
                &self.params,
                &primitive_inputs,
            );

        *concrete_state = new_state;

        let output_messages = primitive_outputs
            .into_iter()
            .filter_map(|primitive_message| match primitive_message {
                PrimitiveMessage::Spike {
                    timestamp,
                    amplitude,
                } => Some(Message::SpikeEvent {
                    timestamp,
                    amplitude,
                }),
                _ => None,
            })
            .collect();

        Ok(adaptiflux_core::agent::state::AgentUpdateResult::new(
            output_messages,
            None,
            None,
            false,
        ))
    }

    fn blueprint_type(&self) -> adaptiflux_core::agent::state::RoleType {
        adaptiflux_core::agent::state::RoleType::Cognitive
    }

    fn supports_gpu(&self) -> bool {
        true
    }
}

#[cfg(feature = "gpu")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu_manager = Arc::new(Mutex::new(GpuResourceManager::new().await?));
    let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
    let message_bus: Arc<dyn MessageBus + Send + Sync> = Arc::new(LocalBus::new());
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    let scheduler = Arc::new(Mutex::new(CoreScheduler::new_with_gpu(
        topology.clone(),
        rule_engine,
        resource_manager,
        message_bus.clone(),
        Some(gpu_manager.clone()),
    )));

    let params = BatchIzhikevichParams {
        a: vec![0.02; 1024],
        b: vec![0.2; 1024],
        c: vec![-65.0; 1024],
        d: vec![2.0; 1024],
        dt: 0.5,
    };

    let gpu_blueprint = GpuIzhikevichBlueprint::new(params, gpu_manager.clone());
    let zooid = Zoooid::new(new_zoooid_id(), Box::new(gpu_blueprint)).await?;

    {
        let mut scheduler_lock = scheduler.lock().await;
        scheduler_lock.spawn_agent(zooid).await?;
    }

    let scheduler_runner = scheduler.clone();
    let scheduler_task = tokio::spawn(async move {
        let mut scheduler = scheduler_runner.lock().await;
        scheduler.run().await
    });

    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    {
        let scheduler_handle = scheduler.lock().await;
        scheduler_handle.stop();
    }

    scheduler_task.await??;
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("The GPU example requires cargo features 'gpu'. Run with: cargo run --example gpu_intensive_demo --features gpu");
}
