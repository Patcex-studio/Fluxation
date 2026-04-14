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

pub fn finite_difference_gradient<F>(params: &[f32], loss_fn: F, epsilon: f32) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    let mut gradient = Vec::with_capacity(params.len());
    let base_loss = loss_fn(params);
    for i in 0..params.len() {
        let mut perturbed = params.to_vec();
        perturbed[i] += epsilon;
        let loss = loss_fn(&perturbed);
        gradient.push((loss - base_loss) / epsilon);
    }
    gradient
}

pub fn serialize_params(params: &[f32]) -> String {
    serde_json::to_string(params).unwrap_or_default()
}

pub fn deserialize_params(serialized: &str) -> Vec<f32> {
    serde_json::from_str(serialized).unwrap_or_default()
}
