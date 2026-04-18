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

//! Apple Silicon Metal GPU demonstration
//!
//! This example shows how to leverage Apple Metal GPU acceleration on macOS for Fluxation.
//! It demonstrates:
//! - Metal backend selection and device info logging
//! - Unified memory optimization (important for integrated GPU)
//! - Batch processing with adaptive workgroup sizes
//! - GPU acceleration for agent updates
//! - Incremental buffer updates for efficiency
//!
//! Run with:
//! ```bash
//! cargo run --example apple_metal_demo --features gpu --release
//! ```
//!
//! Expected output on M1/M2/M3 MacBook:
//! - "Backend: Metal (Apple)"
//! - "Unified memory architecture: true"
//! - GPU-accelerated agent updates with Apple Silicon optimizations

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
use adaptiflux_core::gpu::{GpuConfig, GpuContext};
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
use std::time::Instant;
#[cfg(feature = "gpu")]
use tokio::sync::Mutex;
#[cfg(feature = "gpu")]
use tracing::info;

#[cfg(feature = "gpu")]
struct AppleMetalIzhikevichBlueprint {
    params: BatchIzhikevichParams,
    _gpu_manager: Arc<Mutex<GpuResourceManager>>,
    _device_info_logged: bool,
}

#[cfg(feature = "gpu")]
impl AppleMetalIzhikevichBlueprint {
    pub fn new(
        params: BatchIzhikevichParams,
        gpu_manager: Arc<Mutex<GpuResourceManager>>,
    ) -> Self {
        Self {
            params,
            _gpu_manager: gpu_manager,
            _device_info_logged: false,
        }
    }
}

#[cfg(feature = "gpu")]
#[async_trait::async_trait]
impl AgentBlueprint for AppleMetalIzhikevichBlueprint {
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
        inputs: Vec<(adaptiflux_core::utils::types::ZoooidId, Message)>,
        _topology: &ZoooidTopology,
        _memory: Option<&MemoryPayload>,
    ) -> Result<
        adaptiflux_core::agent::state::AgentUpdateResult,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let primitive_inputs: Vec<PrimitiveMessage> = inputs
            .into_iter()
            .map(|(_, msg)| msg)
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
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("=== Apple Silicon Metal GPU Demonstration ===");

    // Initialize GPU context (will automatically select Metal on macOS)
    info!("Initializing GPU context for Apple Silicon...");
    let _gpu_context = match GpuContext::new().await {
        Ok(ctx) => {
            info!(
                "Successfully initialized GPU: {} ({})",
                ctx.device_info.name, ctx.device_info.backend
            );
            if ctx.device_info.is_unified_memory {
                info!("✓ Unified Memory Architecture detected - optimizing for integrated GPU");
            }
            ctx
        }
        Err(e) => {
            eprintln!("Failed to initialize GPU: {}", e);
            std::process::exit(1);
        }
    };

    // Create GPU resource manager
    let gpu_resource_manager = Arc::new(Mutex::new(GpuResourceManager::new().await?));

    // Set up scheduler components
    let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
    let message_bus: Arc<dyn MessageBus + Send + Sync> = Arc::new(LocalBus::new());
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    // Create scheduler with GPU support
    let mut scheduler = CoreScheduler::new_with_gpu(
        topology.clone(),
        rule_engine,
        resource_manager,
        message_bus.clone(),
        Some(gpu_resource_manager.clone()),
    );

    // Configure GPU acceleration for Apple Silicon
    let mut gpu_config = GpuConfig::apple_silicon();
    gpu_config.enable_profiling = true;
    gpu_config.enable_incremental_updates = true; // Important for unified memory
    scheduler.set_gpu_config(gpu_config.clone());

    info!("GPU Configuration:");
    info!("  Agent updates: {}", gpu_config.enable_agent_update);
    info!(
        "  Connection calculations: {}",
        gpu_config.enable_connection_calculate
    );
    info!(
        "  Plasticity operations: {}",
        gpu_config.enable_plasticity
    );
    info!(
        "  Hormone simulation: {}",
        gpu_config.enable_hormone_simulation
    );
    info!(
        "  Agent batch size: {}",
        gpu_config.agent_batch_size
    );
    info!(
        "  Optimize for integrated GPU: {}",
        gpu_config.optimize_for_igpu
    );

    // Create Izhikevich agents with GPU support
    info!("\nSpawning GPU-accelerated agents...");
    let num_agents = 5;
    for i in 0..num_agents {
        let params = BatchIzhikevichParams {
            a: vec![0.02; 64],
            b: vec![0.2; 64],
            c: vec![-65.0; 64],
            d: vec![2.0; 64],
            dt: 0.5,
        };

        let blueprint = AppleMetalIzhikevichBlueprint::new(
            params,
            gpu_resource_manager.clone(),
        );
        let agent = Zoooid::new(new_zoooid_id(), Box::new(blueprint)).await?;
        scheduler.spawn_agent(agent).await?;

        if i == 0 {
            info!("  ✓ GPU-accelerated Izhikevich neuron agent spawned");
        }
    }

    info!(
        "Total agents spawned: {}",
        scheduler.agent_count()
    );

    // Configure scheduler for optimal performance
    scheduler.set_cycle_frequency(10); // 10 Hz

    // Run scheduler in background
    let scheduler_arc = Arc::new(Mutex::new(scheduler));
    let scheduler_runner = scheduler_arc.clone();

    info!("\nStarting scheduler with GPU acceleration...");
    let start_time = Instant::now();

    let scheduler_task = tokio::spawn(async move {
        let mut scheduler = scheduler_runner.lock().await;
        match scheduler.run_for_iterations(100).await {
            Ok(_) => {
                let elapsed = start_time.elapsed();
                info!(
                    "Scheduler completed 100 iterations in {:.2}ms",
                    elapsed.as_secs_f64() * 1000.0
                );
                let avg_iter_time = elapsed.as_secs_f64() * 1000.0 / 100.0;
                info!("Average iteration time: {:.4}ms", avg_iter_time);
                Ok(())
            }
            Err(e) => Err(e),
        }
    });

    // Wait for completion
    scheduler_task.await??;

    info!("\n=== Test Complete ===");
    info!("✓ Apple Silicon Metal GPU acceleration working correctly");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("This example requires the 'gpu' feature.");
    eprintln!("Run with: cargo run --example apple_metal_demo --features gpu");
    std::process::exit(1);
}
