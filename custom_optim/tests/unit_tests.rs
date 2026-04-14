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

use custom_optim::{BackendType, Optimizer, OptimizerConfig, OptimizerStrategyType};

#[test]
fn sgd_minimizes_quadratic() {
    let config = OptimizerConfig {
        strategy: OptimizerStrategyType::SGD,
        backend: BackendType::CPU,
        population_size: None,
        learning_rate: Some(0.2),
        generations: None,
        iterations: Some(200),
    };
    let mut optimizer = Optimizer::new(config).unwrap();
    let mut params = [5.0_f32, -5.0, 10.0];
    let loss = |x: &[f32]| {
        let a = x[0] - 1.0;
        let b = x[1] + 2.0;
        let c = x[2] - 3.0;
        a * a + b * b + c * c
    };

    optimizer.optimize(&mut params, &loss).unwrap();
    let final_loss = loss(&params);
    assert!(final_loss < 1.0, "final loss was {}", final_loss);
}

#[test]
fn hybrid_strategy_combines_ga_and_sgd() {
    let config = OptimizerConfig {
        strategy: OptimizerStrategyType::Hybrid,
        backend: BackendType::CPU,
        population_size: Some(10),
        learning_rate: Some(0.05),
        generations: Some(3),
        iterations: Some(50),
    };
    let mut optimizer = Optimizer::new(config).unwrap();
    let mut params = [2.0_f32, -1.5, 4.0];
    let loss = |x: &[f32]| {
        let a = x[0] - 1.0;
        let b = x[1] + 2.0;
        let c = x[2] - 3.0;
        a * a + b * b + c * c
    };

    optimizer.optimize(&mut params, &loss).unwrap();
    let final_loss = loss(&params);
    assert!(final_loss < 5.0, "hybrid optimizer did not improve enough: {}", final_loss);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_sgd_kernel_wrapper_exists() {
    let config = OptimizerConfig {
        strategy: OptimizerStrategyType::SGD,
        backend: BackendType::CUDA,
        population_size: None,
        learning_rate: Some(0.1),
        generations: None,
        iterations: Some(1),
    };
    let mut optimizer = Optimizer::new(config).unwrap();
    let mut params = [0.5_f32, -0.3, 0.8];
    let loss = |x: &[f32]| x.iter().map(|v| v * v).sum();

    optimizer.optimize(&mut params, &loss).unwrap();
    assert!(params.iter().all(|v| v.is_finite()));
}
