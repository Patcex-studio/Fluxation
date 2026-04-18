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

use serde::{Deserialize, Serialize};

/// GPU acceleration configuration for CoreScheduler
///
/// Controls which computational phases run on GPU, buffer management strategy,
/// and fallback behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU-accelerated agent updates
    pub enable_agent_update: bool,

    /// Enable GPU-accelerated connection strength calculations
    pub enable_connection_calculate: bool,

    /// Enable GPU-accelerated plasticity/pruning
    pub enable_plasticity: bool,

    /// Enable GPU-accelerated hormone/neuromodulator simulation
    pub enable_hormone_simulation: bool,

    /// Maximum number of agents per GPU batch
    /// Larger batches are more efficient but use more memory
    pub agent_batch_size: u32,

    /// Maximum number of connections per GPU batch
    pub connection_batch_size: u32,

    /// Enable fallback to CPU if GPU operation fails
    pub enable_cpu_fallback: bool,

    /// Enable incremental buffer updates (only changed regions)
    pub enable_incremental_updates: bool,

    /// Sync GPU results back to CPU every N iterations (0 = no sync)
    pub sync_interval_iterations: u64,

    /// Log GPU timing and statistics
    pub enable_profiling: bool,

    /// Optimize for Integrated GPU (reduces memory transfers)
    pub optimize_for_igpu: bool,

    /// Power preference for GPU selection
    pub prefer_high_performance: bool,
}

impl GpuConfig {
    /// Default configuration optimized for Apple Silicon
    pub fn apple_silicon() -> Self {
        Self {
            enable_agent_update: true,
            enable_connection_calculate: true,
            enable_plasticity: true,
            enable_hormone_simulation: true,
            agent_batch_size: 512,      // Conservative for integrated GPU
            connection_batch_size: 1024,
            enable_cpu_fallback: true,
            enable_incremental_updates: true,  // Important for unified memory
            sync_interval_iterations: 10,      // Sync every 10 iterations
            enable_profiling: true,
            optimize_for_igpu: true,   // Optimize for M1/M2/M3
            prefer_high_performance: true,
        }
    }

    /// Configuration for discrete GPU (NVIDIA, AMD)
    pub fn discrete_gpu() -> Self {
        Self {
            enable_agent_update: true,
            enable_connection_calculate: true,
            enable_plasticity: true,
            enable_hormone_simulation: true,
            agent_batch_size: 2048,     // Larger batches for discrete GPU
            connection_batch_size: 4096,
            enable_cpu_fallback: true,
            enable_incremental_updates: false, // Less important for discrete GPU
            sync_interval_iterations: 100,     // Less frequent sync
            enable_profiling: true,
            optimize_for_igpu: false,
            prefer_high_performance: true,
        }
    }

    /// CPU-only fallback configuration
    pub fn cpu_only() -> Self {
        Self {
            enable_agent_update: false,
            enable_connection_calculate: false,
            enable_plasticity: false,
            enable_hormone_simulation: false,
            agent_batch_size: u32::MAX,
            connection_batch_size: u32::MAX,
            enable_cpu_fallback: true,
            enable_incremental_updates: false,
            sync_interval_iterations: 0,
            enable_profiling: false,
            optimize_for_igpu: false,
            prefer_high_performance: true,
        }
    }

    /// Check if any GPU operations are enabled
    pub fn is_gpu_enabled(&self) -> bool {
        self.enable_agent_update
            || self.enable_connection_calculate
            || self.enable_plasticity
            || self.enable_hormone_simulation
    }

    /// Validate config parameters and reject clearly invalid GPU settings
    pub fn validate(&self) -> Result<(), String> {
        if self.agent_batch_size == 0 {
            return Err("GpuConfig.agent_batch_size must be greater than zero".into());
        }
        if self.connection_batch_size == 0 {
            return Err("GpuConfig.connection_batch_size must be greater than zero".into());
        }
        if self.sync_interval_iterations == 0 && self.is_gpu_enabled() && !self.enable_cpu_fallback {
            return Err("GpuConfig.sync_interval_iterations cannot be zero when GPU is enabled and fallback is disabled".into());
        }
        Ok(())
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        return Self::apple_silicon();

        #[cfg(not(target_os = "macos"))]
        return Self::discrete_gpu();
    }
}
