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
    pub a_plus: f32, // LTP amplitude
    pub a_minus: f32, // LTD amplitude
    pub tau_plus: f32, // LTP time constant
    pub tau_minus: f32, // LTD time constant
    pub weight_decay: f32, // Decay rate
    pub pruning_threshold: f32, // Pruning threshold
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
                a_plus: 0.01,
                a_minus: 0.005,
                tau_plus: 20.0,
                tau_minus: 20.0,
                weight_decay: 0.0001,
                pruning_threshold: 0.001,
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

        for i in 0..params.len() {
            // gradients[i] is delta_t (post - pre), positive for LTP, negative for LTD
            let delta_t = gradients[i];
            let dw = if delta_t > 0.0 {
                self.config.a_plus * (-delta_t / self.config.tau_plus).exp()
            } else {
                -self.config.a_minus * (delta_t.abs() / self.config.tau_minus).exp()
            };

            // Apply update
            params[i] += self.config.lr * dw;

            // Weight decay
            params[i] *= 1.0 - self.config.weight_decay;

            // Pruning
            if params[i].abs() < self.config.pruning_threshold {
                params[i] = 0.0;
            }
        }
    }

    fn zero_grad(&mut self, _gradients: &mut [f32]) {
        // No gradient buffer is preserved.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_ltp() {
        let mut opt = HebbianAdaptive::new(1.0);
        let mut params = vec![0.1];
        let gradients = vec![10.0]; // delta_t = 10, LTP
        opt.step(&mut params, &gradients);
        assert!(params[0] > 0.1);
    }

    #[test]
    fn test_stdp_ltd() {
        let mut opt = HebbianAdaptive::new(1.0);
        let mut params = vec![0.1];
        let gradients = vec![-10.0]; // delta_t = -10, LTD
        opt.step(&mut params, &gradients);
        assert!(params[0] < 0.1);
    }

    #[test]
    fn test_weight_decay() {
        let mut opt = HebbianAdaptive::new(1.0);
        let mut params = vec![0.1];
        let gradients = vec![0.0]; // no update
        opt.step(&mut params, &gradients);
        assert!(params[0] < 0.1);
    }

    #[test]
    fn test_pruning() {
        let mut opt = HebbianAdaptive::new(1.0);
        let mut params = vec![0.00005];
        let gradients = vec![1000.0]; // large delta_t, dw ≈ 0
        opt.step(&mut params, &gradients);
        assert_eq!(params[0], 0.0);
    }
}
