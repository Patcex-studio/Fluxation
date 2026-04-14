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

//! Combine scalar signals from grouped agents.

pub type Value = f64;

pub fn aggregate_mean(inputs: &[Value]) -> Value {
    if inputs.is_empty() {
        return 0.0;
    }
    inputs.iter().sum::<Value>() / inputs.len() as Value
}

pub fn aggregate_max(inputs: &[Value]) -> Value {
    inputs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

pub fn aggregate_sum(inputs: &[Value]) -> Value {
    inputs.iter().sum()
}

/// Majority vote on sign; magnitude is the mean of |x|.
pub fn aggregate_vote(inputs: &[Value]) -> Value {
    if inputs.is_empty() {
        return 0.0;
    }
    let mut pos = 0usize;
    let mut neg = 0usize;
    let mut sum_abs = 0.0_f64;
    for x in inputs {
        sum_abs += x.abs();
        if *x > 0.0 {
            pos += 1;
        } else if *x < 0.0 {
            neg += 1;
        }
    }
    let mag = sum_abs / inputs.len() as f64;
    if pos > neg {
        mag
    } else if neg > pos {
        -mag
    } else {
        0.0
    }
}
