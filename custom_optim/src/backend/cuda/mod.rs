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

#[cfg(feature = "cuda")]
pub mod cuda_strategy;
#[cfg(feature = "cuda")]
pub mod kernels;

#[cfg(feature = "cuda")]
pub use cuda_strategy::{CudaGeneticAlgorithm, CudaSGD};

#[cfg(not(feature = "cuda"))]
pub mod cuda_strategy {
    use crate::{
        optimizer::{OptimizerConfig, OptimizerError},
        strategy::OptimizerStrategy,
    };

    pub struct CudaSGD;
    pub struct CudaGeneticAlgorithm;

    impl CudaSGD {
        pub fn new(_config: OptimizerConfig) -> Result<Self, OptimizerError> {
            Err(OptimizerError::Unsupported)
        }
    }

    impl CudaGeneticAlgorithm {
        pub fn new(_config: OptimizerConfig) -> Result<Self, OptimizerError> {
            Err(OptimizerError::Unsupported)
        }
    }

    impl OptimizerStrategy for CudaSGD {
        fn optimize(
            &mut self,
            _params: &mut [f32],
            _loss_fn: &dyn Fn(&[f32]) -> f32,
        ) -> Result<(), OptimizerError> {
            Err(OptimizerError::Unsupported)
        }
    }

    impl OptimizerStrategy for CudaGeneticAlgorithm {
        fn optimize(
            &mut self,
            _params: &mut [f32],
            _loss_fn: &dyn Fn(&[f32]) -> f32,
        ) -> Result<(), OptimizerError> {
            Err(OptimizerError::Unsupported)
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub mod kernels {}

#[cfg(not(feature = "cuda"))]
pub use cuda_strategy::{CudaGeneticAlgorithm, CudaSGD};
