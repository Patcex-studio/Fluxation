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

use crate::accelerator::backend::AcceleratorBackend;
use std::any::Any;
use std::sync::Arc;
use tracing::info;

/// Load balancing strategy for distributing work across accelerators
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Round-robin: distribute work sequentially
    RoundRobin,
    /// Always use the first available backend
    Sequential,
    /// No specific strategy (user selects)
    Manual,
}

/// Pool of accelerators for parallel computation.
///
/// Allows simultaneous use of multiple backends (e.g., CPU + GPU)
/// to distribute computational load and maximize throughput.
pub struct AcceleratorPool {
    /// List of available backends
    backends: Vec<Arc<dyn AcceleratorBackend>>,
    /// Current index for round-robin distribution
    current_index: std::sync::Mutex<usize>,
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,
}

impl AcceleratorPool {
    /// Create a new accelerator pool with the given backends
    pub fn new(
        backends: Vec<Arc<dyn AcceleratorBackend>>,
        strategy: LoadBalancingStrategy,
    ) -> Self {
        info!("Creating accelerator pool with {} backends", backends.len());
        Self {
            backends,
            current_index: std::sync::Mutex::new(0),
            strategy,
        }
    }

    /// Initialize all backends in the pool
    pub async fn initialize_all(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Initializing all backends in pool");
        for (idx, backend) in self.backends.iter().enumerate() {
            match backend.initialize().await {
                Ok(_) => info!("Backend {} initialized successfully", idx),
                Err(e) => {
                    tracing::warn!("Failed to initialize backend {}: {}", idx, e);
                }
            }
        }
        Ok(())
    }

    /// Shutdown all backends in the pool
    pub async fn shutdown_all(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Shutting down all backends in pool");
        for (idx, backend) in self.backends.iter().enumerate() {
            if let Err(e) = backend.shutdown().await {
                tracing::warn!("Error shutting down backend {}: {}", idx, e);
            }
        }
        Ok(())
    }

    /// Get the next backend according to the load balancing strategy
    fn get_next_backend(&self) -> Option<Arc<dyn AcceleratorBackend>> {
        if self.backends.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let mut idx = self.current_index.lock().unwrap();
                let backend = self.backends[*idx % self.backends.len()].clone();
                *idx += 1;
                Some(backend)
            }
            LoadBalancingStrategy::Sequential => {
                // Always use the first backend
                self.backends.first().cloned()
            }
            LoadBalancingStrategy::Manual => {
                // Default to first backend if no explicit selection
                self.backends.first().cloned()
            }
        }
    }

    /// Execute a compute operation on the next available backend (round-robin)
    pub async fn execute_on_next(
        &self,
        shader_name: &str,
        args: &dyn Any,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let backend = self
            .get_next_backend()
            .ok_or("No backends available in pool")?;

        info!("Executing {} on next backend", shader_name);
        backend.execute_compute(shader_name, args).await
    }

    /// Execute a compute operation on a specific backend (by index)
    pub async fn execute_on_backend(
        &self,
        backend_idx: usize,
        shader_name: &str,
        args: &dyn Any,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let backend = self
            .backends
            .get(backend_idx)
            .ok_or(format!("Backend index {} out of range", backend_idx))?;

        info!(
            "Executing {} on backend {}",
            shader_name, backend_idx
        );
        backend.execute_compute(shader_name, args).await
    }

    /// Execute a compute operation on all backends in parallel
    ///
    /// Note: This requires futures to be properly awaited with join_all or similar.
    /// Each item in args_list corresponds to one backend.
    pub async fn execute_parallel(
        &self,
        shader_name: &str,
        args_list: Vec<Box<dyn Any + Send + Sync>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if args_list.len() != self.backends.len() {
            return Err(format!(
                "Argument list length ({}) must match backend count ({})",
                args_list.len(),
                self.backends.len()
            )
            .into());
        }

        info!("Executing {} in parallel on {} backends", shader_name, self.backends.len());

        let mut handles = vec![];
        for (_idx, (backend, args)) in self.backends.iter().zip(args_list.into_iter()).enumerate() {
            let backend = backend.clone();
            let shader_name = shader_name.to_string();

            // Spawn a task for this backend
            let handle = tokio::spawn(async move {
                backend.execute_compute(&shader_name, args.as_ref()).await
            });
            handles.push(handle);
        }

        // Wait for all to complete and collect errors
        let mut errors = vec![];
        for (idx, handle) in handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(())) => info!("Backend {} completed", idx),
                Ok(Err(e)) => {
                    let err_msg = format!("Backend {} failed: {}", idx, e);
                    tracing::warn!("{}", err_msg);
                    errors.push(err_msg);
                }
                Err(e) => {
                    let err_msg = format!("Backend {} task failed: {}", idx, e);
                    tracing::warn!("{}", err_msg);
                    errors.push(err_msg);
                }
            }
        }

        if !errors.is_empty() {
            return Err(format!("Parallel execution errors: {}", errors.join("; ")).into());
        }

        Ok(())
    }

    /// Get the number of backends in the pool
    pub fn len(&self) -> usize {
        self.backends.len()
    }

    /// Check if the pool is empty
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }

    /// Get information about all backends
    pub fn backend_info_all(&self) -> Vec<String> {
        self.backends
            .iter()
            .map(|b| format!("{} ({})", b.info().name, b.info().backend_type))
            .collect()
    }

    /// Get a specific backend by index
    pub fn get_backend(&self, idx: usize) -> Option<Arc<dyn AcceleratorBackend>> {
        self.backends.get(idx).cloned()
    }

    /// Synchronize all backends (wait for pending operations)
    pub async fn synchronize_all(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Synchronizing all backends");
        for (idx, backend) in self.backends.iter().enumerate() {
            if let Err(e) = backend.synchronize().await {
                tracing::warn!("Error synchronizing backend {}: {}", idx, e);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::cpu_backend::CpuBackend;

    #[tokio::test]
    async fn test_pool_creation() {
        let cpu = Arc::new(CpuBackend::new()) as Arc<dyn AcceleratorBackend>;
        let pool = AcceleratorPool::new(vec![cpu], LoadBalancingStrategy::Sequential);
        assert_eq!(pool.len(), 1);
    }

    #[tokio::test]
    async fn test_pool_round_robin() {
        let cpu1 = Arc::new(CpuBackend::new()) as Arc<dyn AcceleratorBackend>;
        let cpu2 = Arc::new(CpuBackend::new()) as Arc<dyn AcceleratorBackend>;

        let pool = AcceleratorPool::new(vec![cpu1, cpu2], LoadBalancingStrategy::RoundRobin);
        assert_eq!(pool.len(), 2);

        let b1 = pool.get_next_backend();
        let b2 = pool.get_next_backend();
        assert!(b1.is_some());
        assert!(b2.is_some());
    }

    #[tokio::test]
    async fn test_pool_empty() {
        let pool: AcceleratorPool = AcceleratorPool::new(vec![], LoadBalancingStrategy::Sequential);
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[tokio::test]
    async fn test_pool_backend_info() {
        let cpu = Arc::new(CpuBackend::new()) as Arc<dyn AcceleratorBackend>;
        let pool = AcceleratorPool::new(vec![cpu], LoadBalancingStrategy::Sequential);

        let infos = pool.backend_info_all();
        assert_eq!(infos.len(), 1);
        assert!(infos[0].contains("CPU"));
    }
}