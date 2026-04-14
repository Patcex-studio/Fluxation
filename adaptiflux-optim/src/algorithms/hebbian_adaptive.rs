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
pub struct HebbianAdaptiveConfig {
    pub lr: f32,
    pub normalization: f32,
}

#[derive(Debug, Clone)]
pub struct HebbianAdaptive {
    config: HebbianAdaptiveConfig,
}

impl HebbianAdaptive {
    pub fn new(lr: f32) -> Self {
        Self {
            config: HebbianAdaptiveConfig {
                lr,
                normalization: 1.0,
            },
        }
    }
}

impl Optimizer for HebbianAdaptive {
    fn init(&mut self, _params: &mut [f32]) {
        // No persistent state required for this rule.
    }

    fn step(&mut self, params: &mut [f32], gradients: &[f32]) {
        assert_eq!(params.len(), gradients.len(), "HebbianAdaptive requires equal param and grad lengths");
        let norm = self
            .config
            .normalization
            .max(1e-6)
            + gradients.iter().map(|g| g.abs()).sum::<f32>();

        for i in 0..params.len() {
            params[i] += self.config.lr * gradients[i] / norm;
        }
    }

    fn zero_grad(&mut self, _gradients: &mut [f32]) {
        // No gradient buffer is preserved.
    }
}
