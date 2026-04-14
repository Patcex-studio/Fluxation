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

use crate::optimizer::Optimizer;
use crate::simd_kernels::vectorized_sgd_step;

#[derive(Debug, Clone)]
pub struct SGDConfig {
    pub lr: f32,
}

#[derive(Debug, Clone)]
pub struct SGD {
    config: SGDConfig,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            config: SGDConfig { lr },
        }
    }
}

impl Optimizer for SGD {
    fn init(&mut self, _params: &mut [f32]) {
        // No internal state required for plain SGD.
    }

    fn step(&mut self, params: &mut [f32], gradients: &[f32]) {
        assert_eq!(params.len(), gradients.len(), "SGD requires equal param and grad lengths");
        vectorized_sgd_step(params, gradients, self.config.lr);
    }

    fn zero_grad(&mut self, gradients: &mut [f32]) {
        gradients.fill(0.0);
    }
}
