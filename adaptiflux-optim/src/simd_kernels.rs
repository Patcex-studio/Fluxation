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

use wide::f32x4;

const VECTOR_WIDTH: usize = 4;

pub fn vectorized_sgd_step(params: &mut [f32], grads: &[f32], lr: f32) {
    let len = params.len();
    let step = VECTOR_WIDTH;
    let limit = len - (len % step);
    let lr_vec = f32x4::splat(lr);

    for i in (0..limit).step_by(step) {
        let p = f32x4::new([
            params[i],
            params[i + 1],
            params[i + 2],
            params[i + 3],
        ]);
        let g = f32x4::new([
            grads[i],
            grads[i + 1],
            grads[i + 2],
            grads[i + 3],
        ]);
        let result = p - g * lr_vec;
        let arr = result.to_array();
        params[i..i + step].copy_from_slice(&arr);
    }

    for i in limit..len {
        params[i] -= grads[i] * lr;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn vectorized_adam_step(
    params: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grads: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    bias_correction1: f32,
    bias_correction2: f32,
) {
    let len = params.len();
    let step = VECTOR_WIDTH;
    let limit = len - (len % step);

    let beta1_vec = f32x4::splat(beta1);
    let beta2_vec = f32x4::splat(beta2);
    let one_minus_beta1 = f32x4::splat(1.0 - beta1);
    let one_minus_beta2 = f32x4::splat(1.0 - beta2);
    let lr_vec = f32x4::splat(lr);
    let epsilon_vec = f32x4::splat(epsilon);
    let bias1_vec = f32x4::splat(bias_correction1);
    let bias2_vec = f32x4::splat(bias_correction2);

    for i in (0..limit).step_by(step) {
        let p = f32x4::new([
            params[i],
            params[i + 1],
            params[i + 2],
            params[i + 3],
        ]);
        let g = f32x4::new([
            grads[i],
            grads[i + 1],
            grads[i + 2],
            grads[i + 3],
        ]);
        let m_prev = f32x4::new([m[i], m[i + 1], m[i + 2], m[i + 3]]);
        let v_prev = f32x4::new([v[i], v[i + 1], v[i + 2], v[i + 3]]);

        let m_new = beta1_vec * m_prev + one_minus_beta1 * g;
        let v_new = beta2_vec * v_prev + one_minus_beta2 * g * g;
        let m_hat = m_new / bias1_vec;
        let v_hat = v_new / bias2_vec;
        let update = lr_vec * m_hat / (v_hat.sqrt() + epsilon_vec);

        let result = p - update;
        let p_arr = result.to_array();
        let m_arr = m_new.to_array();
        let v_arr = v_new.to_array();

        params[i..i + step].copy_from_slice(&p_arr);
        m[i..i + step].copy_from_slice(&m_arr);
        v[i..i + step].copy_from_slice(&v_arr);
    }

    for i in limit..len {
        let g = grads[i];
        let m_new = beta1 * m[i] + (1.0 - beta1) * g;
        let v_new = beta2 * v[i] + (1.0 - beta2) * g * g;
        let m_hat = m_new / bias_correction1;
        let v_hat = v_new / bias_correction2;
        params[i] -= lr * m_hat / (v_hat.sqrt() + epsilon);
        m[i] = m_new;
        v[i] = v_new;
    }
}
