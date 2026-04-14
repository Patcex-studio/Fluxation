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
use crate::simd_kernels::vectorized_adam_step;

#[derive(Debug, Clone)]
pub struct AdamConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

#[derive(Debug, Clone)]
pub struct AdamState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub t: usize,
}

#[derive(Debug, Clone)]
pub struct Adam {
    pub config: AdamConfig,
    pub state: AdamState,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            config: AdamConfig {
                lr,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            state: AdamState {
                m: Vec::new(),
                v: Vec::new(),
                t: 0,
            },
        }
    }
}

impl Optimizer for Adam {
    fn init(&mut self, params: &mut [f32]) {
        let len = params.len();
        self.state.m.clear();
        self.state.v.clear();
        self.state.m.resize(len, 0.0);
        self.state.v.resize(len, 0.0);
        self.state.t = 0;
    }

    fn step(&mut self, params: &mut [f32], gradients: &[f32]) {
        assert_eq!(params.len(), gradients.len(), "Adam requires equal param and grad lengths");
        self.state.t += 1;
        let t = self.state.t as f32;
        let bias_correction1 = 1.0 - self.config.beta1.powf(t);
        let bias_correction2 = 1.0 - self.config.beta2.powf(t);

        vectorized_adam_step(
            params,
            &mut self.state.m,
            &mut self.state.v,
            gradients,
            self.config.lr,
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon,
            bias_correction1,
            bias_correction2,
        );
    }

    fn zero_grad(&mut self, _gradients: &mut [f32]) {
        // Online updates generally recompute gradients each step.
    }
}
