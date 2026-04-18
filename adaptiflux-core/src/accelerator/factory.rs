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

use crate::accelerator::backend::{AcceleratorBackend, AcceleratorType};
use crate::accelerator::config::AcceleratorConfig;
use crate::accelerator::cpu_backend::CpuBackend;
#[cfg(feature = "gpu")]
use crate::accelerator::gpu_backend::gpu_wgpu::GpuBackendWgpu;
use std::sync::Arc;
use tracing::{info, warn};

/// Factory for creating accelerator backends based on configuration
pub struct AcceleratorFactory;

impl AcceleratorFactory {
    /// Create a backend for a specific accelerator type
    ///
    /// Returns None if the backend type is not available (e.g., GPU not compiled in)
    pub fn create_backend(
        backend_type: AcceleratorType,
    ) -> Option<Arc<dyn AcceleratorBackend>> {
        info!("Creating backend for {}", backend_type);

        match backend_type {
            AcceleratorType::Cpu => {
                Some(Arc::new(CpuBackend::new()) as Arc<dyn AcceleratorBackend>)
            }
            AcceleratorType::GpuMetal | AcceleratorType::GpuVulkan | AcceleratorType::GpuDx12 => {
                #[cfg(feature = "gpu")]
                {
                    Some(Arc::new(GpuBackendWgpu::new()) as Arc<dyn AcceleratorBackend>)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    warn!("GPU requested but gpu feature not enabled");
                    None
                }
            }
            AcceleratorType::Cuda => {
                #[cfg(feature = "gpu_cuda")]
                {
                    warn!("CUDA backend not yet fully implemented");
                    None // TODO: Implement CUDA backend
                }
                #[cfg(not(feature = "gpu_cuda"))]
                {
                    warn!("CUDA requested but gpu_cuda feature not enabled");
                    None
                }
            }
            AcceleratorType::WasmSimd => {
                warn!("WebAssembly SIMD backend not yet implemented");
                None
            }
        }
    }

    /// Create a backend following the configuration's preferences and fallback chain
    ///
    /// Attempts to create the preferred backend first, then tries fallbacks in order.
    /// If all fail, returns a CPU backend as final fallback (or error if CPU creation fails).
    pub fn create_from_config(
        config: &AcceleratorConfig,
    ) -> Result<Arc<dyn AcceleratorBackend>, Box<dyn std::error::Error + Send + Sync>> {
        // Try preferred type first
        if let Some(backend) = Self::create_backend(config.preferred_type) {
            info!("Successfully created preferred backend: {}", config.preferred_type);
            return Ok(backend);
        }

        warn!(
            "Failed to create preferred backend: {}, trying fallbacks",
            config.preferred_type
        );

        // Try fallback chain
        for fallback_type in &config.fallback_chain {
            if let Some(backend) = Self::create_backend(*fallback_type) {
                info!("Successfully created fallback backend: {}", fallback_type);
                return Ok(backend);
            }
            warn!("Failed to create fallback backend: {}", fallback_type);
        }

        // Final fallback: CPU (always available)
        info!("All backends failed, falling back to CPU");
        Ok(Arc::new(CpuBackend::new()) as Arc<dyn AcceleratorBackend>)
    }

    /// Create multiple backends from the fallback chain (including preferred)
    ///
    /// Returns all successfully created backends in order
    pub fn create_all_available(
        config: &AcceleratorConfig,
    ) -> Vec<Arc<dyn AcceleratorBackend>> {
        let mut backends = vec![];

        // Add preferred if available
        if let Some(backend) = Self::create_backend(config.preferred_type) {
            backends.push(backend);
        }

        // Add fallbacks if available
        for fallback_type in &config.fallback_chain {
            if let Some(backend) = Self::create_backend(*fallback_type) {
                backends.push(backend);
            }
        }

        // Always ensure CPU is available
        if backends.is_empty() {
            backends.push(Arc::new(CpuBackend::new()) as Arc<dyn AcceleratorBackend>);
        }

        info!("Created {} available backends", backends.len());
        backends
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_cpu_backend() {
        let backend = AcceleratorFactory::create_backend(AcceleratorType::Cpu);
        assert!(backend.is_some());
    }

    #[test]
    fn test_create_from_cpu_config() {
        let config = AcceleratorConfig::cpu_only();
        let backend = AcceleratorFactory::create_from_config(&config);
        assert!(backend.is_ok());
    }

    #[test]
    fn test_create_all_available() {
        let config = AcceleratorConfig::cpu_only();
        let backends = AcceleratorFactory::create_all_available(&config);
        assert!(!backends.is_empty());
        assert!(backends[0].info().backend_type == AcceleratorType::Cpu);
    }

    #[test]
    fn test_create_with_fallback() {
        let config = AcceleratorConfig::apple_silicon_optimized();
        // Ensure we can create even if GPU not available
        let backend = AcceleratorFactory::create_from_config(&config);
        // Should at least fallback to CPU
        assert!(backend.is_ok());
    }
}
