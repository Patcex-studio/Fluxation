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

//! Small optimization primitives for online adaptation (SGD-style steps, evolutionary jitter).

use crate::utils::types::{Param, StateValue};

/// One SGD step on a scalar parameter.
#[inline]
pub fn sgd_step(param: Param, gradient: Param, learning_rate: Param) -> Param {
    param - learning_rate * gradient
}

/// Clamp PID gains to a stable range.
pub fn clamp_pid_gains(kp: Param, ki: Param, kd: Param) -> (Param, Param, Param) {
    (
        kp.clamp(0.08, 500.0),
        ki.clamp(0.0, 50.0),
        kd.clamp(0.0, 50.0),
    )
}

/// Heuristic SGD on PID gains using the scalar tracking error as a surrogate loss driver.
pub fn pid_gains_sgd_step(
    kp: Param,
    ki: Param,
    kd: Param,
    error: StateValue,
    lr: Param,
) -> (Param, Param, Param) {
    // Mild gradient proxy: push gains so |error| tends to shrink without exploding.
    let g = error * (1.0 + error.abs()).recip();
    let new_kp = sgd_step(kp, g * 0.06, lr);
    let new_ki = sgd_step(ki, g * 0.02, lr);
    let new_kd = sgd_step(kd, g * 0.015, lr);
    clamp_pid_gains(new_kp, new_ki, new_kd)
}

/// Deterministic pseudo-random in [-1, 1] from a seed (xorshift64*).
pub fn deterministic_noise(seed: u64) -> Param {
    let mut x = seed.max(1);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    let x = x.wrapping_mul(2685821657736338717);
    (x as Param / u64::MAX as Param) * 2.0 - 1.0
}

/// Local evolutionary-style update: jitter current value, bias step using error sign.
pub fn evolutionary_scalar_update(
    current: Param,
    error: StateValue,
    seed: u64,
    sigma: Param,
) -> Param {
    let n = deterministic_noise(seed);
    let bias = if error.abs() < 1e-6 {
        0.0
    } else {
        -error.signum() * sigma * 0.25
    };
    (current + n * sigma + bias).clamp(0.001, 500.0)
}

/// Mutate `tau_m`-like time constant toward lower error (used for LIF-style demos).
pub fn evolutionary_tau_update(tau: Param, error: StateValue, seed: u64, sigma: Param) -> Param {
    let n = deterministic_noise(seed.rotate_left(3));
    let pull = if error > 0.0 {
        -sigma * 0.15
    } else {
        sigma * 0.15
    };
    (tau + n * sigma * 0.5 + pull).clamp(0.5, 200.0)
}
