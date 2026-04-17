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

use crate::backend::{cpu, cuda};
use crate::strategy::OptimizerStrategy;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub strategy: OptimizerStrategyType,
    pub backend: BackendType,
    pub population_size: Option<usize>,
    pub learning_rate: Option<f32>,
    pub generations: Option<usize>,
    pub iterations: Option<usize>,
    pub block_size: Option<u32>,
}

pub struct Optimizer {
    strategy: Box<dyn OptimizerStrategy>,
}

impl Optimizer {
    pub fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        let strategy: Box<dyn OptimizerStrategy> = match (config.strategy, config.backend) {
            (OptimizerStrategyType::Genetic, BackendType::CPU) => {
                Box::new(cpu::GeneticAlgorithm::new(config.clone())?)
            }
            (OptimizerStrategyType::Genetic, BackendType::CUDA) => {
                Box::new(cuda::CudaGeneticAlgorithm::new(config.clone())?)
            }
            (OptimizerStrategyType::SGD, BackendType::CPU) => {
                Box::new(cpu::SGD::new(config.clone())?)
            }
            (OptimizerStrategyType::SGD, BackendType::CUDA) => {
                Box::new(cuda::CudaSGD::new(config.clone())?)
            }
            (OptimizerStrategyType::Hybrid, BackendType::CPU) => Box::new(
                crate::strategies::hybrid::HybridStrategy::new(config.clone(), BackendType::CPU)?,
            ),
            (OptimizerStrategyType::Hybrid, BackendType::CUDA) => Box::new(
                crate::strategies::hybrid::HybridStrategy::new(config.clone(), BackendType::CUDA)?,
            ),
        };
        Ok(Self { strategy })
    }

    pub fn optimize(
        &mut self,
        params: &mut [f32],
        loss_fn: &dyn Fn(&[f32]) -> f32,
    ) -> Result<(), OptimizerError> {
        self.strategy.optimize(params, loss_fn)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizerStrategyType {
    Genetic,
    SGD,
    Hybrid,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BackendType {
    CPU,
    CUDA,
}

#[derive(thiserror::Error, Debug)]
pub enum OptimizerError {
    #[error("CUDA initialization failed")]
    CudaInitializationFailed,
    #[error("CUDA error: {0}")]
    CudaError(String),
    #[error("Kernel launch failed")]
    KernelLaunchFailed,
    #[error("Invalid optimizer configuration")]
    InvalidConfig,
    #[error("Unsupported backend or strategy")]
    Unsupported,
}
