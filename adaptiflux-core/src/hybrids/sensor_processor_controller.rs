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

//! Sensor-Processor-Controller Hybrid Architecture
//!
//! This hybrid combines:
//! - SensorZooid (LIF neuron): Detects input signals
//! - CognitiveZooid (Izhikevich neuron): Processes/interprets the signal
//! - PIDZooid (PID controller): Generates control output
//!
//! Flow: AnalogInput → Sensor (converts to spike) → Cognitive (generates response) → PID (generates control signal)

use crate::agent::blueprint::{
    CognitivezooidBlueprint, CognitivezooidParams, PIDzooidBlueprint, PIDzooidParams,
    SensorzooidBlueprint, SensorzooidParams,
};
use crate::agent::zoooid::Zoooid;
use crate::core::CoreScheduler;
use crate::primitives::control::pid::PidParams;
use crate::primitives::spiking::izhikevich::IzhikevichParams;
use crate::primitives::spiking::lif::LifParams;
use crate::utils::types::ZoooidId;

/// Configuration for the Sensor-Processor-Controller architecture
#[derive(Debug, Clone, Default)]
pub struct SensorProcessorControllerConfig {
    pub sensor_lif_params: LifParams,
    pub cognitive_izh_params: IzhikevichParams,
    pub controller_pid_params: PidParams,
}

/// Sensor-Processor-Controller Architecture Instance
#[derive(Debug, Clone)]
pub struct SensorProcessorControllerArchitecture {
    pub sensor_id: ZoooidId,
    pub processor_id: ZoooidId,
    pub controller_id: ZoooidId,
}

impl SensorProcessorControllerArchitecture {
    /// Create and register all three agents in the scheduler
    ///
    /// Returns IDs of (sensor, processor, controller)
    pub async fn create(
        scheduler: &mut CoreScheduler,
        config: SensorProcessorControllerConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Create Sensor (LIF)
        let sensor_blueprint = SensorzooidBlueprint {
            params: SensorzooidParams {
                lif_params: config.sensor_lif_params,
                connection_request_interval: 10,
            },
        };

        let sensor_zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(sensor_blueprint)).await?;
        let sensor_id = sensor_zooid.id;

        // Create Processor (Izhikevich)
        let processor_blueprint = CognitivezooidBlueprint {
            params: CognitivezooidParams {
                izh_params: config.cognitive_izh_params,
                connection_request_interval: 10,
                stdp_a_plus: 0.01,
                stdp_a_minus: 0.005,
                stdp_tau_plus: 20.0,
                stdp_tau_minus: 20.0,
                weight_decay: 0.0001,
                pruning_threshold: 0.001,
                enable_simd_batch: false,
            },
        };

        let processor_zooid =
            Zoooid::new(ZoooidId::new_v4(), Box::new(processor_blueprint)).await?;
        let processor_id = processor_zooid.id;

        // Create Controller (PID)
        let controller_blueprint = PIDzooidBlueprint {
            params: PIDzooidParams {
                pid_params: config.controller_pid_params,
                connection_request_interval: 10,
            },
        };

        let controller_zooid =
            Zoooid::new(ZoooidId::new_v4(), Box::new(controller_blueprint)).await?;
        let controller_id = controller_zooid.id;

        // Spawn all agents
        scheduler.spawn_agent(sensor_zooid).await?;
        scheduler.spawn_agent(processor_zooid).await?;
        scheduler.spawn_agent(controller_zooid).await?;

        // Establish connections: Sensor -> Processor -> Controller
        let mut topology = scheduler.topology.write().await;
        topology.add_edge(
            sensor_id,
            processor_id,
            crate::core::topology::ConnectionProperties::default(),
        );
        topology.add_edge(
            processor_id,
            controller_id,
            crate::core::topology::ConnectionProperties::default(),
        );
        drop(topology);

        Ok(SensorProcessorControllerArchitecture {
            sensor_id,
            processor_id,
            controller_id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{LocalBus, ResourceManager, ZoooidTopology};
    use crate::RuleEngine;
    use std::sync::Arc;
    use tokio::sync::{Mutex, RwLock};

    #[tokio::test]
    async fn test_sensor_processor_controller_creation() {
        let topology = Arc::new(RwLock::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = SensorProcessorControllerConfig::default();
        let arch = SensorProcessorControllerArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create architecture");

        assert!(scheduler.agent_count() == 3);
        assert!(arch.sensor_id != arch.processor_id);
        assert!(arch.processor_id != arch.controller_id);
    }

    #[tokio::test]
    async fn test_sensor_processor_controller_message_flow() {
        let topology = Arc::new(RwLock::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = SensorProcessorControllerConfig::default();
        let arch = SensorProcessorControllerArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create architecture");

        // Verify topology connections
        let topology_guard = scheduler.topology.read().await;
        let connections = topology_guard.get_neighbors(arch.sensor_id);
        assert!(
            !connections.is_empty(),
            "Sensor should have outgoing connections"
        );

        let processor_connections = topology_guard.get_neighbors(arch.processor_id);
        assert!(
            !processor_connections.is_empty(),
            "Processor should have outgoing connections"
        );
    }
}
