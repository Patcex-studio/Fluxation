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

#[derive(Debug, Clone)]
pub struct RMSPropConfig {
    pub lr: f32,
    pub beta: f32,
    pub epsilon: f32,
}

#[derive(Debug, Clone)]
pub struct RMSProp {
    config: RMSPropConfig,
    cache: Vec<f32>,
}

impl RMSProp {
    pub fn new(lr: f32) -> Self {
        Self {
            config: RMSPropConfig {
                lr,
                beta: 0.9,
                epsilon: 1e-8,
            },
            cache: Vec::new(),
        }
    }
}

impl Optimizer for RMSProp {
    fn init(&mut self, params: &mut [f32]) {
        self.cache.clear();
        self.cache.resize(params.len(), 0.0);
    }

    fn step(&mut self, params: &mut [f32], gradients: &[f32]) {
        assert_eq!(params.len(), gradients.len(), "RMSProp requires equal param and grad lengths");
        for i in 0..params.len() {
            self.cache[i] = self.config.beta * self.cache[i]
                + (1.0 - self.config.beta) * gradients[i] * gradients[i];
            params[i] -= self.config.lr * gradients[i]
                / (self.cache[i].sqrt() + self.config.epsilon);
        }
    }

    fn zero_grad(&mut self, _gradients: &mut [f32]) {
        // No gradient accumulator state to reset.
    }
}
