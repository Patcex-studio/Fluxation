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

//! Distributed Search Use Case
//!
//! Demonstrates swarm intelligence searching for a target/resource.
//! Uses the `SwarmForagerArchitecture` with realistic timing and metrics.

use adaptiflux_core::core::message_bus::LocalBus;
use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::hybrids::swarm_forager::{SwarmForagerArchitecture, SwarmForagerConfig};
use adaptiflux_core::{CoreScheduler, ResourceManager, RuleEngine};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Distributed Search Use Case ===\n");
    println!("Simulating a swarm of agents searching for a target using pheromone coordination.\n");

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

    // Configure swarm
    let mut config = SwarmForagerConfig::default();
    config.swarm_size = 10;
    config.pheromone_strength = 2.0;
    config.pheromone_decay_rate = 0.15;

    println!("Creating swarm with {} foragers...", config.swarm_size);
    let swarm = SwarmForagerArchitecture::create(&mut scheduler, config).await?;

    println!(
        "Swarm created with {} agents. Agent IDs: {:?}\n",
        swarm.agent_ids.len(),
        &swarm.agent_ids[0..swarm.agent_ids.len().min(3)]
    );

    // Simulate search over 50 iterations
    println!("Starting search simulation (50 iterations)...\n");
    let start_time = Instant::now();
    let iterations = 50;

    for iteration in 0..iterations {
        // Update pheromone levels (simulating agent exploration)
        swarm.update_pheromones().await;

        if iteration % 10 == 0 {
            let forager_pheromone = swarm.get_pheromone_level(swarm.forager_id).await;
            let metrics = scheduler.get_metrics();

            println!(
                "Iteration {}: Forager pheromone level: {:.3}, Total agents: {}",
                iteration, forager_pheromone, metrics.total_agents
            );
        }

        // Small delay to simulate agent processing
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    let elapsed = start_time.elapsed();

    println!(
        "\nSearch simulation completed in {:.2}s",
        elapsed.as_secs_f64()
    );

    // Final metrics
    let final_pheromone = swarm.get_pheromone_level(swarm.forager_id).await;
    let metrics = scheduler.get_metrics();

    println!("\n=== Final Results ===");
    println!("Swarm size: {}", swarm.agent_ids.len());
    println!("Final forager pheromone level: {:.3}", final_pheromone);
    println!("Total agents in system: {}", metrics.total_agents);
    println!("Total network connections: {}", metrics.total_connections);

    Ok(())
}
