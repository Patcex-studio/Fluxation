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

use crate::primitives::base::{Primitive, PrimitiveMessage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidParams {
    pub kp: crate::utils::types::Param,
    pub ki: crate::utils::types::Param,
    pub kd: crate::utils::types::Param,
    pub dt: crate::utils::types::Param,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PidState {
    pub integral: crate::utils::types::StateValue,
    pub previous_error: crate::utils::types::StateValue,
}

#[derive(Debug, Clone)]
pub struct PidController;

impl PidController {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for PidController {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PidParams {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.1,
            kd: 0.01,
            dt: 0.1,
        }
    }
}

impl Primitive for PidController {
    type State = PidState;
    type Params = PidParams;

    fn initialize(_params: Self::Params) -> Self::State {
        PidState {
            integral: 0.0,
            previous_error: 0.0,
        }
    }

    fn update(
        mut state: Self::State,
        params: &Self::Params,
        input: &[PrimitiveMessage],
    ) -> (Self::State, Vec<PrimitiveMessage>) {
        let mut outputs = Vec::new();

        if let Some(error) = input.iter().find_map(|msg| {
            if let PrimitiveMessage::Error(value) = msg {
                Some(*value)
            } else {
                None
            }
        }) {
            state.integral += error * params.dt;
            let derivative = (error - state.previous_error) / params.dt;
            let control = params.kp * error + params.ki * state.integral + params.kd * derivative;
            state.previous_error = error;
            outputs.push(PrimitiveMessage::ControlSignal(control));
        }

        (state, outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_proportional_output() {
        let params = PidParams {
            kp: 2.0,
            ki: 0.0,
            kd: 0.0,
            dt: 1.0,
        };
        let state = PidController::initialize(params.clone());
        let input = vec![PrimitiveMessage::Error(1.5)];

        let (new_state, outputs) = PidController::update(state, &params, &input);

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], PrimitiveMessage::ControlSignal(3.0));
        assert_eq!(new_state.previous_error, 1.5);
    }
}
