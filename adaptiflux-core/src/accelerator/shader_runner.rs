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

/// Shader execution arguments: agent data update
#[derive(Clone)]
pub struct AgentUpdateArgs {
    pub agent_data: Arc<Vec<u8>>,
}

/// Shader execution arguments: connection calculations
#[derive(Clone)]
pub struct ConnectionCalculateArgs {
    pub connection_data: Arc<Vec<u8>>,
}

/// Shader execution arguments: plasticity (pruning/synaptogenesis)
#[derive(Clone)]
pub struct PlasticityArgs {
    pub neuron_data: Arc<Vec<u8>>,
    pub connection_data: Arc<Vec<u8>>,
}

/// Shader execution arguments: hormone/neuromodulator simulation
#[derive(Clone)]
pub struct HormoneSimulationArgs {
    pub hormone_state: Arc<Vec<u8>>,
    pub agent_activity: Arc<Vec<u8>>,
}

/// Universal shader runner that works with any AcceleratorBackend.
///
/// Provides high-level shader execution interface decoupled from specific backend implementations.
/// Each method handles uploading data, executing the compute kernel, and downloading results.
pub struct ShaderRunner {
    /// The underlying accelerator backend
    backend: Arc<dyn AcceleratorBackend>,
}

impl ShaderRunner {
    /// Create a new shader runner with the given backend
    pub fn new(backend: Arc<dyn AcceleratorBackend>) -> Self {
        info!("Creating shader runner with backend: {}", backend.info().name);
        Self { backend }
    }

    /// Get reference to the underlying backend
    pub fn backend(&self) -> &Arc<dyn AcceleratorBackend> {
        &self.backend
    }

    /// Execute agent update shader
    ///
    /// Uploads agent data, runs the update kernel, and downloads results
    pub async fn run_agent_update(
        &self,
        agent_data: &[u8],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        info!("Running agent update shader");

        let args = AgentUpdateArgs {
            agent_data: Arc::new(agent_data.to_vec()),
        };

        self.backend
            .execute_compute("agent_update", &args as &dyn Any)
            .await?;

        self.backend.download_data("updated_agents").await
    }

    /// Execute connection strength calculation shader
    pub async fn run_connection_calculate(
        &self,
        connection_data: &[u8],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        info!("Running connection calculate shader");

        let args = ConnectionCalculateArgs {
            connection_data: Arc::new(connection_data.to_vec()),
        };

        self.backend
            .execute_compute("connection_calculate", &args as &dyn Any)
            .await?;

        self.backend.download_data("updated_connections").await
    }

    /// Execute plasticity/pruning shader
    pub async fn run_plasticity_pruning(
        &self,
        neuron_data: &[u8],
        connection_data: &[u8],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        info!("Running plasticity pruning shader");

        let args = PlasticityArgs {
            neuron_data: Arc::new(neuron_data.to_vec()),
            connection_data: Arc::new(connection_data.to_vec()),
        };

        self.backend
            .execute_compute("plasticity_pruning", &args as &dyn Any)
            .await?;

        self.backend.download_data("pruned_connections").await
    }

    /// Execute synaptogenesis shader
    pub async fn run_plasticity_synaptogenesis(
        &self,
        neuron_data: &[u8],
        connection_data: &[u8],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        info!("Running plasticity synaptogenesis shader");

        let args = PlasticityArgs {
            neuron_data: Arc::new(neuron_data.to_vec()),
            connection_data: Arc::new(connection_data.to_vec()),
        };

        self.backend
            .execute_compute("plasticity_synaptogenesis", &args as &dyn Any)
            .await?;

        self.backend.download_data("new_connections").await
    }

    /// Execute hormone/neuromodulator simulation shader
    pub async fn run_hormone_simulation(
        &self,
        hormone_state: &[u8],
        agent_activity: &[u8],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        info!("Running hormone simulation shader");

        let args = HormoneSimulationArgs {
            hormone_state: Arc::new(hormone_state.to_vec()),
            agent_activity: Arc::new(agent_activity.to_vec()),
        };

        self.backend
            .execute_compute("hormone_simulation", &args as &dyn Any)
            .await?;

        self.backend.download_data("updated_hormones").await
    }

    /// Generic compute execution for custom shaders
    pub async fn execute_compute(
        &self,
        shader_name: &str,
        args: &dyn Any,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Executing custom shader: {}", shader_name);
        self.backend.execute_compute(shader_name, args).await
    }

    /// Upload data to the accelerator
    pub async fn upload_data(
        &self,
        data: &[u8],
        buffer_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Uploading {} bytes to buffer: {}", data.len(), buffer_id);
        self.backend.upload_data(data, buffer_id).await
    }

    /// Download data from the accelerator
    pub async fn download_data(
        &self,
        buffer_id: &str,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
        info!("Downloading buffer: {}", buffer_id);
        self.backend.download_data(buffer_id).await
    }

    /// Synchronize with the accelerator (wait for pending operations)
    pub async fn synchronize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Synchronizing accelerator");
        self.backend.synchronize().await
    }

    /// Get backend information
    pub fn backend_info(&self) -> String {
        let info = self.backend.info();
        format!(
            "{} - {} (Compute Units: {:?}, Memory: {} MB)",
            info.name,
            info.backend_type,
            info.compute_units,
            info.memory_bytes / (1024 * 1024)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::cpu_backend::CpuBackend;

    #[tokio::test]
    async fn test_shader_runner_creation() {
        let backend = Arc::new(CpuBackend::new()) as Arc<dyn AcceleratorBackend>;
        let runner = ShaderRunner::new(backend.clone());

        assert_eq!(runner.backend().info().backend_type, crate::accelerator::backend::AcceleratorType::Cpu);
    }

    #[tokio::test]
    async fn test_agent_update_args() {
        let args = AgentUpdateArgs {
            agent_data: Arc::new(vec![1, 2, 3]),
        };
        let _any = &args as &dyn std::any::Any;
        assert!(args.agent_data.len() > 0);
    }
}