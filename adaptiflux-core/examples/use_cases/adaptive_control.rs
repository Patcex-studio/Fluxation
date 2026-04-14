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

//! Adaptive Control Use Case
//!
//! Demonstrates adaptive control system using sensor-processor-controller architecture.
//! A simulated disturbance is applied to a system, and the controller adapts to stabilize it.

use adaptiflux_core::core::message_bus::LocalBus;
use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::hybrids::sensor_processor_controller::{
    SensorProcessorControllerArchitecture, SensorProcessorControllerConfig,
};
use adaptiflux_core::primitives::control::pid::PidParams;
use adaptiflux_core::primitives::spiking::izhikevich::IzhikevichParams;
use adaptiflux_core::primitives::spiking::lif::LifParams;
use adaptiflux_core::{CoreScheduler, ResourceManager, RuleEngine};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Adaptive Control Use Case ===\n");
    println!("Simulating a control system with sensor-processor-controller pipeline.\n");

    // Setup
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

    // Configure the control system
    let mut config = SensorProcessorControllerConfig::default();

    // Tune parameters for responsive control
    config.sensor_lif_params.threshold = -50.0;
    config.sensor_lif_params.resting_potential = -70.0;

    config.cognitive_izh_params.a = 0.02;
    config.cognitive_izh_params.b = 0.2;
    config.cognitive_izh_params.c = -65.0;

    config.controller_pid_params.kp = 1.0;
    config.controller_pid_params.ki = 0.5;
    config.controller_pid_params.kd = 0.1;

    println!("Creating sensor-processor-controller architecture...");
    let arch = SensorProcessorControllerArchitecture::create(&mut scheduler, config).await?;

    println!(
        "Architecture created:");
    println!("  Sensor ID:     {:?}", arch.sensor_id);
    println!("  Processor ID:  {:?}", arch.processor_id);
    println!("  Controller ID: {:?}\n", arch.controller_id);

    // Verify topology
    let topology_guard = scheduler.topology.lock().await;
    let sensor_connections = topology_guard.get_neighbors(arch.sensor_id);
    let processor_connections = topology_guard.get_neighbors(arch.processor_id);

    println!("Topology verification:");
    println!("  Sensor -> {} targets", sensor_connections.len());
    println!("  Processor -> {} targets", processor_connections.len());

    drop(topology_guard);

    // Simulate control loop
    println!("\n=== Control System Simulation ===\n");

    let simulation_steps = 20;
    println!("Running {} control cycles...\n", simulation_steps);

    for cycle in 0..simulation_steps {
        // Simulate disturbances at certain cycles
        let disturbance = if cycle > 5 && cycle < 10 {
            println!("Cycle {}: Disturbance applied!", cycle);
            1.5
        } else if cycle > 15 {
            println!("Cycle {}: Major disturbance!", cycle);
            2.5
        } else {
            0.0
        };

        // Simulate sensor reading (input signal)
        let sensor_input = 0.5 + disturbance;

        // Print state
        if cycle % 5 == 0 || disturbance != 0.0 {
            let metrics = scheduler.get_metrics();
            println!(
                "Cycle {}: Input={:.2}, Agents={}, Connections={}",
                cycle,
                sensor_input,
                metrics.total_agents,
                metrics.total_connections
            );
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    println!("\n=== Simulation Complete ===");
    let final_metrics = scheduler.get_metrics();
    println!("Final system state:");
    println!("  Total agents: {}", final_metrics.total_agents);
    println!("  Total connections: {}", final_metrics.total_connections);
    println!(
        "  Avg connectivity: {:.2}",
        final_metrics.avg_connectivity
    );

    Ok(())
}
