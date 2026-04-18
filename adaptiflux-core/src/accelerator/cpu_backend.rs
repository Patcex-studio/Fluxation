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

use crate::accelerator::backend::{AcceleratorBackend, AcceleratorType, BackendInfo};
use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Mutex;
use tracing::info;

/// CPU-based accelerator backend.
///
/// Provides a fallback implementation that runs on CPU.
/// Stores buffers in a HashMap for simplicity.
pub struct CpuBackend {
    /// Named buffers stored in system memory
    buffers: Mutex<HashMap<String, Vec<u8>>>,
    /// Whether the backend has been initialized
    initialized: Mutex<bool>,
}

impl CpuBackend {
    /// Create a new CPU backend instance
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
            initialized: Mutex::new(false),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AcceleratorBackend for CpuBackend {
    async fn initialize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut init = self.initialized.lock().unwrap();
        if !*init {
            info!("Initializing CPU backend");
            *init = true;
        }
        Ok(())
    }

    async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut init = self.initialized.lock().unwrap();
        if *init {
            info!("Shutting down CPU backend");
            *init = false;
            // Clear all buffers
            self.buffers.lock().unwrap().clear();
        }
        Ok(())
    }

    async fn execute_compute(
        &self,
        shader_name: &str,
        _args: &dyn Any,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // CPU backend is a pass-through for compute execution
        // In a real implementation, this would run CPU-based kernels
        // For now, we just log and accept
        info!("CPU backend executing: {}", shader_name);
        Ok(())
    }

    async fn upload_data(
        &self,
        data: &[u8],
        buffer_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(buffer_id.to_string(), data.to_vec());
        Ok(())
    }

    async fn download_data(
        &self,
        buffer_id: &str,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        let buffers = self.buffers.lock().unwrap();
        buffers
            .get(buffer_id)
            .cloned()
            .ok_or_else(|| format!("Buffer not found: {}", buffer_id).into())
    }

    fn info(&self) -> BackendInfo {
        let num_cpus = num_cpus::get() as u32;
        // Estimate total system memory (rough approximation)
        let total_memory = num_cpus as u64 * 1024 * 1024 * 512; // ~512MB per core estimate

        BackendInfo {
            name: format!("CPU ({} cores)", num_cpus),
            backend_type: AcceleratorType::Cpu,
            memory_bytes: total_memory,
            compute_units: Some(num_cpus),
            is_unified_memory: true,  // CPU memory is unified
            version: "1.0".to_string(),
        }
    }

    fn is_available(&self) -> bool {
        *self.initialized.lock().unwrap()
    }

    async fn synchronize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // CPU operations are synchronous, so nothing to do
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cpu_backend_init() {
        let backend = CpuBackend::new();
        assert!(!backend.is_available());
        backend.initialize().await.unwrap();
        assert!(backend.is_available());
    }

    #[tokio::test]
    async fn test_cpu_backend_buffers() {
        let backend = CpuBackend::new();
        backend.initialize().await.unwrap();

        let data = vec![1, 2, 3, 4, 5];
        backend.upload_data(&data, "test_buffer").await.unwrap();

        let retrieved = backend.download_data("test_buffer").await.unwrap();
        assert_eq!(data, retrieved);
    }

    #[tokio::test]
    async fn test_cpu_backend_info() {
        let backend = CpuBackend::new();
        let info = backend.info();
        assert_eq!(info.backend_type, AcceleratorType::Cpu);
        assert!(info.compute_units.is_some());
        assert!(info.is_unified_memory);
    }

    #[tokio::test]
    async fn test_cpu_backend_shutdown() {
        let backend = CpuBackend::new();
        backend.initialize().await.unwrap();

        let data = vec![1, 2, 3];
        backend.upload_data(&data, "test_buffer").await.unwrap();

        backend.shutdown().await.unwrap();
        assert!(!backend.is_available());

        // After shutdown, buffers should be cleared
        let result = backend.download_data("test_buffer").await;
        assert!(result.is_err());
    }
}
