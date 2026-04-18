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

//! Unified accelerator backend system.
//!
//! This module provides a flexible, extensible architecture for connecting various types of
//! accelerators (CPU, GPU Metal, Vulkan, DX12, CUDA, WASM SIMD) to the Adaptiflux core,
//! with minimal coupling and maximum interchangeability.
//!
//! # Architecture
//!
//! - **`backend`**: Trait definition `AcceleratorBackend` for all backends
//! - **`config`**: Configuration system with platform-specific presets
//! - **`cpu_backend`**: CPU-only fallback implementation
//! - **`gpu_backend`**: GPU implementation using wgpu (Metal, Vulkan, DX12)
//! - **`pool`**: Multi-backend pool for parallel computation
//! - **`factory`**: Factory pattern for creating backends from config
//! - **`shader_runner`**: High-level shader execution interface
//!
//! # Example
//!
//! ```ignore
//! use adaptiflux_core::accelerator::{AcceleratorFactory, AcceleratorConfig, ShaderRunner};
//!
//! // Create configuration
//! let config = AcceleratorConfig::apple_silicon_optimized();
//!
//! // Create backend from config (with fallback chain)
//! let backend = AcceleratorFactory::create_from_config(&config)?;
//!
//! // Create shader runner
//! let runner = ShaderRunner::new(backend);
//!
//! // Use it
//! runner.run_agent_update(&agent_data).await?;
//! ```

pub mod backend;
pub mod config;
pub mod cpu_backend;
pub mod factory;
pub mod gpu_backend;
pub mod pool;
pub mod shader_runner;

// Re-export key types
pub use backend::{AcceleratorBackend, AcceleratorType, BackendInfo};
pub use config::{AcceleratorConfig, BatchSizes};
pub use cpu_backend::CpuBackend;
pub use factory::AcceleratorFactory;
pub use gpu_backend::gpu_wgpu::GpuBackendWgpu;
pub use pool::{AcceleratorPool, LoadBalancingStrategy};
pub use shader_runner::{
    AgentUpdateArgs, ConnectionCalculateArgs, HormoneSimulationArgs, PlasticityArgs,
    ShaderRunner,
};