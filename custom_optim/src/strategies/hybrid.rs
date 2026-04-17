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

use crate::{
    backend::{cpu, cuda},
    optimizer::{BackendType, OptimizerConfig, OptimizerError},
    strategy::OptimizerStrategy,
};

pub struct HybridStrategy {
    pub ga: Box<dyn OptimizerStrategy>,
    pub sgd: Box<dyn OptimizerStrategy>,
}

impl HybridStrategy {
    pub fn new(config: OptimizerConfig, backend: BackendType) -> Result<Self, OptimizerError> {
        let ga_backend = backend.clone();
        let sgd_backend = backend;

        let ga: Box<dyn OptimizerStrategy> = match ga_backend {
            BackendType::CPU => Box::new(cpu::GeneticAlgorithm::new(config.clone())?),
            BackendType::CUDA => Box::new(cuda::CudaGeneticAlgorithm::new(config.clone())?),
        };
        let sgd: Box<dyn OptimizerStrategy> = match sgd_backend {
            BackendType::CPU => Box::new(cpu::SGD::new(config.clone())?),
            BackendType::CUDA => Box::new(cuda::CudaSGD::new(config.clone())?),
        };
        Ok(Self { ga, sgd })
    }
}

impl OptimizerStrategy for HybridStrategy {
    fn optimize(
        &mut self,
        params: &mut [f32],
        loss_fn: &dyn Fn(&[f32]) -> f32,
    ) -> Result<(), OptimizerError> {
        self.ga.optimize(params, loss_fn)?;
        self.sgd.optimize(params, loss_fn)
    }
}
