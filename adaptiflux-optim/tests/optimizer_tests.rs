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

use adaptiflux_optim::{Adam, GradientAccumulator, Optimizer, SGD};

#[test]
fn sgd_step_updates_parameters() {
    let mut params = vec![1.0_f32, 2.0, 3.0, 4.0];
    let grads = vec![0.5_f32, 0.5, 0.5, 0.5];
    let mut optimizer = SGD::new(0.1);
    optimizer.init(&mut params);
    optimizer.step(&mut params, &grads);
    assert_eq!(params, vec![0.95, 1.95, 2.95, 3.95]);
}

#[test]
fn adam_step_reduces_quadratic_error() {
    let mut params = vec![2.0_f32, 2.0, 2.0, 2.0];
    let grads = vec![4.0_f32, 4.0, 4.0, 4.0];
    let mut optimizer = Adam::new(0.1);
    optimizer.init(&mut params);
    optimizer.step(&mut params, &grads);
    for p in params {
        assert!(p < 2.0);
    }
}

#[test]
fn gradient_accumulator_averages_and_flushes() {
    let mut accumulator = GradientAccumulator::new(3, 2);
    accumulator.accumulate_batch(&[1.0, 2.0, 3.0]);
    assert!(accumulator.flush().is_none());
    accumulator.accumulate_batch(&[3.0, 4.0, 5.0]);
    let avg = accumulator.flush().expect("should flush after threshold");
    assert_eq!(avg, &[2.0, 3.0, 4.0]);
}
