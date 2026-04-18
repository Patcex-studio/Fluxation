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

//! Example: Using the new unified accelerator architecture.
//!
//! This example demonstrates how to:
//! 1. Create an accelerator configuration with fallback chain
//! 2. Use the factory to create backends
//! 3. Create a shader runner for compute operations
//! 4. Use the accelerator pool for parallel operations

use adaptiflux_core::accelerator::{
    AcceleratorConfig, AcceleratorFactory, AcceleratorPool, LoadBalancingStrategy,
    ShaderRunner,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // ==================== Example 1: CPU-only setup ====================
    println!("\n=== Example 1: CPU-only Accelerator ===");
    let cpu_config = AcceleratorConfig::cpu_only();
    match AcceleratorFactory::create_from_config(&cpu_config) {
        Ok(cpu_backend) => {
            cpu_backend.initialize().await.ok();
            println!("CPU Backend Info: {}", cpu_backend.info().name);
            cpu_backend.shutdown().await.ok();
        }
        Err(e) => eprintln!("Error creating CPU backend: {}", e),
    }

    // ==================== Example 2: Platform-optimized configuration ====================
    println!("\n=== Example 2: Platform-optimized Configuration ===");
    let platform_config = AcceleratorConfig::auto_detect();
    println!(
        "Detected platform, preferred: {}",
        platform_config.preferred_type
    );
    println!("Fallback chain: {:?}", platform_config.fallback_chain);

    // ==================== Example 3: Apple Silicon with GPU and CPU fallback ====================
    println!("\n=== Example 3: Apple Silicon optimized ===");
    let apple_config = AcceleratorConfig::apple_silicon_optimized();
    match AcceleratorFactory::create_from_config(&apple_config) {
        Ok(apple_backend) => {
            apple_backend.initialize().await.ok();
            println!(
                "Apple Silicon Backend: {}",
                apple_backend.info().name
            );
            println!("Batch sizes: {}", apple_config.batch_sizes.agent_update);
            apple_backend.shutdown().await.ok();
        }
        Err(e) => eprintln!("Error with Apple Silicon backend: {}", e),
    }

    // ==================== Example 4: Using ShaderRunner ====================
    println!("\n=== Example 4: ShaderRunner for compute operations ===");
    match AcceleratorFactory::create_from_config(&AcceleratorConfig::cpu_only()) {
        Ok(backend) => {
            backend.initialize().await.ok();

            let runner = ShaderRunner::new(backend.clone());
            println!("Shader Runner Backend: {}", runner.backend_info());

            // Simulate agent data
            let agent_data = vec![1.0f32; 100]
                .into_iter()
                .map(|v| v.to_le_bytes())
                .flatten()
                .collect::<Vec<_>>();
            
            // Execute agent update shader
            match runner.run_agent_update(&agent_data).await {
                Ok(results) => println!("Agent update completed, got {} bytes", results.len()),
                Err(e) => println!("Agent update note: {}", e),
            }

            backend.shutdown().await.ok();
        }
        Err(e) => eprintln!("Error with ShaderRunner setup: {}", e),
    }

    // ==================== Example 5: Accelerator Pool ====================
    println!("\n=== Example 5: Accelerator Pool ===");
    let cpu1 = Arc::new(adaptiflux_core::accelerator::CpuBackend::new()) as Arc<dyn adaptiflux_core::accelerator::AcceleratorBackend>;
    let cpu2 = Arc::new(adaptiflux_core::accelerator::CpuBackend::new()) as Arc<dyn adaptiflux_core::accelerator::AcceleratorBackend>;

    let pool = AcceleratorPool::new(
        vec![cpu1, cpu2],
        LoadBalancingStrategy::RoundRobin,
    );

    println!("Pool created with {} backends", pool.len());
    pool.initialize_all().await.ok();

    println!("Backend info:");
    for info in pool.backend_info_all() {
        println!("  - {}", info);
    }

    pool.shutdown_all().await.ok();

    // ==================== Example 6: Customized configuration ====================
    println!("\n=== Example 6: Custom configuration ===");
    let mut custom_config = AcceleratorConfig::cpu_only();
    custom_config.batch_sizes.agent_update = 1024;
    custom_config.batch_sizes.connection_calculate = 2048;
    custom_config.enable_profiling = true;

    println!("Custom config:");
    println!("  Agent batch: {}", custom_config.batch_sizes.agent_update);
    println!("  Connection batch: {}", custom_config.batch_sizes.connection_calculate);
    println!("  Profiling: {}", custom_config.enable_profiling);

    Ok(())
}
