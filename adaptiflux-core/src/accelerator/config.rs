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

use crate::accelerator::backend::AcceleratorType;
use serde::{Deserialize, Serialize};

/// Batch size configuration for different computation phases.
///
/// Larger batches → better GPU utilization but more memory.
/// Smaller batches → lower latency but less efficient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizes {
    /// Batch size for agent state updates
    pub agent_update: usize,
    /// Batch size for connection strength calculations
    pub connection_calculate: usize,
    /// Batch size for plasticity (pruning/synaptogenesis)
    pub plasticity: usize,
    /// Batch size for hormone/neuromodulator simulation
    pub hormone_simulation: usize,
}

impl Default for BatchSizes {
    fn default() -> Self {
        Self {
            agent_update: 512,
            connection_calculate: 1024,
            plasticity: 256,
            hormone_simulation: 512,
        }
    }
}

impl BatchSizes {
    /// Batch sizes optimized for Apple Silicon (integrated GPU with unified memory)
    pub fn apple_silicon() -> Self {
        Self {
            agent_update: 512,      // Conservative, unified memory
            connection_calculate: 1024,
            plasticity: 256,
            hormone_simulation: 512,
        }
    }

    /// Batch sizes optimized for discrete GPUs (NVIDIA, AMD with dedicated VRAM)
    pub fn discrete_gpu() -> Self {
        Self {
            agent_update: 2048,     // Larger batches for dedicated VRAM
            connection_calculate: 4096,
            plasticity: 1024,
            hormone_simulation: 2048,
        }
    }

    /// Batch sizes optimized for CPU-only execution
    pub fn cpu_only() -> Self {
        Self {
            agent_update: 128,      // Smaller, CPU can handle fewer in parallel
            connection_calculate: 256,
            plasticity: 64,
            hormone_simulation: 128,
        }
    }
}

/// Unified configuration for accelerator backends.
///
/// Allows precise control over:
/// - Which accelerator to use (with fallback chain)
/// - Batch sizes for each computation phase
/// - GPU-specific optimizations (incremental updates, profiling)
/// - Resource and performance tradeoffs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorConfig {
    /// Preferred accelerator type
    pub preferred_type: AcceleratorType,

    /// Fallback chain: if preferred type fails, try these in order.
    /// If empty, falls back to CPU.
    pub fallback_chain: Vec<AcceleratorType>,

    /// Batch size configuration for different phases
    pub batch_sizes: BatchSizes,

    /// Enable CPU fallback if accelerator operation fails
    pub enable_cpu_fallback: bool,

    /// Enable incremental buffer updates (only changed regions)
    /// Useful for integrated GPUs with unified memory (Apple Silicon)
    pub enable_incremental_updates: bool,

    /// Synchronize GPU results back to CPU every N iterations (0 = no sync)
    pub sync_interval_iterations: u64,

    /// Enable detailed timing and statistics logging
    pub enable_profiling: bool,

    /// Enable memory usage optimization for integrated GPUs
    pub optimize_for_igpu: bool,

    /// Prefer high-performance GPU mode (vs power-saving)
    pub prefer_high_performance: bool,
}

impl Default for AcceleratorConfig {
    fn default() -> Self {
        Self {
            preferred_type: AcceleratorType::Cpu,
            fallback_chain: vec![],
            batch_sizes: BatchSizes::default(),
            enable_cpu_fallback: true,
            enable_incremental_updates: false,
            sync_interval_iterations: 0,
            enable_profiling: false,
            optimize_for_igpu: false,
            prefer_high_performance: false,
        }
    }
}

impl AcceleratorConfig {
    /// Configuration for Apple Silicon with GPU acceleration
    pub fn apple_silicon_optimized() -> Self {
        Self {
            preferred_type: AcceleratorType::GpuMetal,
            fallback_chain: vec![AcceleratorType::Cpu],
            batch_sizes: BatchSizes::apple_silicon(),
            enable_cpu_fallback: true,
            enable_incremental_updates: true,  // Critical for unified memory
            sync_interval_iterations: 10,
            enable_profiling: true,
            optimize_for_igpu: true,
            prefer_high_performance: true,
        }
    }

    /// Configuration for discrete NVIDIA GPU with CUDA
    pub fn nvidia_cuda_optimized() -> Self {
        Self {
            preferred_type: AcceleratorType::Cuda,
            fallback_chain: vec![AcceleratorType::GpuVulkan, AcceleratorType::Cpu],
            batch_sizes: BatchSizes::discrete_gpu(),
            enable_cpu_fallback: true,
            enable_incremental_updates: false,
            sync_interval_iterations: 100,  // Less frequent sync for dedicated VRAM
            enable_profiling: true,
            optimize_for_igpu: false,
            prefer_high_performance: true,
        }
    }

    /// Configuration for discrete GPU with Vulkan (Linux/Windows)
    pub fn vulkan_optimized() -> Self {
        Self {
            preferred_type: AcceleratorType::GpuVulkan,
            fallback_chain: vec![AcceleratorType::Cpu],
            batch_sizes: BatchSizes::discrete_gpu(),
            enable_cpu_fallback: true,
            enable_incremental_updates: false,
            sync_interval_iterations: 100,
            enable_profiling: true,
            optimize_for_igpu: false,
            prefer_high_performance: true,
        }
    }

    /// Configuration for WindowsDirectX 12 GPU
    pub fn dx12_optimized() -> Self {
        Self {
            preferred_type: AcceleratorType::GpuDx12,
            fallback_chain: vec![AcceleratorType::GpuVulkan, AcceleratorType::Cpu],
            batch_sizes: BatchSizes::discrete_gpu(),
            enable_cpu_fallback: true,
            enable_incremental_updates: false,
            sync_interval_iterations: 100,
            enable_profiling: true,
            optimize_for_igpu: false,
            prefer_high_performance: true,
        }
    }

    /// CPU-only configuration (maximum compatibility)
    pub fn cpu_only() -> Self {
        Self {
            preferred_type: AcceleratorType::Cpu,
            fallback_chain: vec![],
            batch_sizes: BatchSizes::cpu_only(),
            enable_cpu_fallback: true,
            enable_incremental_updates: false,
            sync_interval_iterations: 0,
            enable_profiling: false,
            optimize_for_igpu: false,
            prefer_high_performance: false,
        }
    }

    /// Configuration for WebAssembly with SIMD
    pub fn wasm_simd() -> Self {
        Self {
            preferred_type: AcceleratorType::WasmSimd,
            fallback_chain: vec![AcceleratorType::Cpu],
            batch_sizes: BatchSizes::cpu_only(),
            enable_cpu_fallback: true,
            enable_incremental_updates: false,
            sync_interval_iterations: 0,
            enable_profiling: false,
            optimize_for_igpu: false,
            prefer_high_performance: false,
        }
    }

    /// Auto-detect best configuration based on platform
    pub fn auto_detect() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::apple_silicon_optimized()
        }
        #[cfg(all(target_os = "windows", not(target_os = "macos")))]
        {
            Self::dx12_optimized()
        }
        #[cfg(all(target_os = "linux", not(target_os = "macos")))]
        {
            Self::vulkan_optimized()
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::wasm_simd()
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux", target_arch = "wasm32")))]
        {
            Self::cpu_only()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_sizes_defaults() {
        let bs = BatchSizes::default();
        assert_eq!(bs.agent_update, 512);
        assert_eq!(bs.connection_calculate, 1024);
    }

    #[test]
    fn test_accelerator_config_defaults() {
        let cfg = AcceleratorConfig::default();
        assert_eq!(cfg.preferred_type, AcceleratorType::Cpu);
        assert!(cfg.enable_cpu_fallback);
    }

    #[test]
    fn test_apple_silicon_config() {
        let cfg = AcceleratorConfig::apple_silicon_optimized();
        assert_eq!(cfg.preferred_type, AcceleratorType::GpuMetal);
        assert!(cfg.optimize_for_igpu);
    }
}
// Copyright (C) 2026 Jocer S. <patcex@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
