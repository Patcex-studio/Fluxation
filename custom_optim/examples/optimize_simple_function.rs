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

fn main() {
    let config = OptimizerConfig {
        strategy: OptimizerStrategyType::SGD,
        backend: BackendType::CPU,
        population_size: None,
        learning_rate: Some(0.05),
        generations: None,
        iterations: Some(50),
        block_size: None,
    };

    let mut optimizer = Optimizer::new(config).expect("Failed to create optimizer");
    let mut params = [2.0_f32, -1.0, 3.0];
    let loss = |x: &[f32]| {
        let a = x[0] - 1.0;
        let b = x[1] + 2.0;
        let c = x[2] - 3.0;
        a * a + b * b + c * c
    };

    optimizer
        .optimize(&mut params, &loss)
        .expect("Optimization failed");
    println!("Optimized params: {:?}", params);
}
