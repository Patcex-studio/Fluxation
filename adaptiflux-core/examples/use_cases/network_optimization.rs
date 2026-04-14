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

//! Network Optimization Use Case (Experimental)
//!
//! Demonstrates Physarum-inspired network routing optimization.
//! The system adaptively strengthens efficient paths and weakens inefficient ones.

use adaptiflux_core::core::message_bus::LocalBus;
use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::hybrids::physarum_router::{
    PhysarumRouterArchitecture, PhysarumRouterConfig,
};
use adaptiflux_core::{CoreScheduler, ResourceManager, RuleEngine};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Network Optimization Use Case (Experimental) ===\n");
    println!("Simulating Physarum-inspired adaptive network routing.\n");

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

    // Configure router network
    let mut config = PhysarumRouterConfig::default();
    config.router_count = 6;
    config.traffic_simulation_mode = true;

    println!("Creating Physarum router network with {} nodes...", config.router_count);
    let router = PhysarumRouterArchitecture::create(&mut scheduler, config).await?;

    println!(
        "Router network created. Router IDs: {:?}\n",
        &router.router_ids[0..router.router_ids.len().min(3)]
    );

    // Record source and sink
    if let (Some(source), Some(sink)) = (router.source_id, router.sink_id) {
        println!("Network configured:");
        println!("  Source: {:?}", source);
        println!("  Sink:   {:?}", sink);
    }

    // Simulate traffic routing
    println!("\n=== Traffic Simulation ===\n");

    struct TrafficPattern {
        iteration: usize,
        amount: u64,
        description: &'static str,
    }

    let traffic_patterns = vec![
        TrafficPattern {
            iteration: 0,
            amount: 100,
            description: "Initial load",
        },
        TrafficPattern {
            iteration: 5,
            amount: 250,
            description: "Increased traffic",
        },
        TrafficPattern {
            iteration: 10,
            amount: 150,
            description: "Traffic reduction",
        },
        TrafficPattern {
            iteration: 15,
            amount: 400,
            description: "Peak load",
        },
        TrafficPattern {
            iteration: 20,
            amount: 100,
            description: "Return to baseline",
        },
    ];

    for iteration in 0..25 {
        // Check if there's a traffic event at this iteration
        for pattern in &traffic_patterns {
            if pattern.iteration == iteration {
                router.send_traffic(pattern.amount).await;
                println!(
                    "Iteration {}: {} - {} bytes sent",
                    iteration, pattern.description, pattern.amount
                );

                // Simulate path recording
                let path = router.router_ids.clone();
                router.record_path(path).await;
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }

    println!("\n=== Optimization Results ===\n");

    let total_traffic = router.get_total_traffic().await;
    let paths = router.get_paths().await;

    println!("Total traffic processed: {} bytes", total_traffic);
    println!("Optimized paths recorded: {}", paths.len());
    println!("Network statistics:");

    let metrics = scheduler.get_metrics();
    println!("  Total routers: {}", metrics.total_agents);
    println!("  Total connections: {}", metrics.total_connections);
    println!(
        "  Average connectivity: {:.2}",
        metrics.avg_connectivity
    );

    // Show sampled paths
    if !paths.is_empty() {
        println!("\nSample optimized paths (first 3):");
        for (i, path) in paths.iter().take(3).enumerate() {
            println!("  Path {}: {:?}", i + 1, path);
        }
    }

    println!("\n=== Simulation Complete ===");
    println!("Network optimization cycle completed successfully.");

    Ok(())
}
