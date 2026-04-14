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
pub struct LifParams {
    pub tau_m: crate::utils::types::Param,
    pub v_rest: crate::utils::types::Param,
    pub v_thresh: crate::utils::types::Param,
    pub v_reset: crate::utils::types::Param,
    pub r_m: crate::utils::types::Param,
    pub dt: crate::utils::types::Param,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifState {
    pub membrane_potential: crate::utils::types::StateValue,
    pub last_spike_time: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct LifNeuron;

impl LifNeuron {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for LifNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LifParams {
    fn default() -> Self {
        Self {
            tau_m: 10.0,
            v_rest: -70.0,
            v_thresh: -55.0,
            v_reset: -70.0,
            r_m: 10.0,
            dt: 0.1,
        }
    }
}

impl Primitive for LifNeuron {
    type State = LifState;
    type Params = LifParams;

    fn initialize(params: Self::Params) -> Self::State {
        LifState {
            membrane_potential: params.v_rest,
            last_spike_time: None,
        }
    }

    fn update(
        mut state: Self::State,
        params: &Self::Params,
        input: &[PrimitiveMessage],
    ) -> (Self::State, Vec<PrimitiveMessage>) {
        let mut outputs = Vec::new();

        let total_input_current: crate::utils::types::StateValue = input
            .iter()
            .filter_map(|msg| {
                if let PrimitiveMessage::InputCurrent(value) = msg {
                    Some(*value)
                } else {
                    None
                }
            })
            .sum();

        state.membrane_potential += (-(state.membrane_potential - params.v_rest)
            + params.r_m * total_input_current)
            / params.tau_m
            * params.dt;

        if state.membrane_potential >= params.v_thresh {
            outputs.push(PrimitiveMessage::Spike {
                timestamp: 0,
                amplitude: 1.0,
            });
            state.membrane_potential = params.v_reset;
            state.last_spike_time = Some(0);
        }

        (state, outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lif_spike_on_large_current() {
        let params = LifParams::default();
        let state = LifNeuron::initialize(params.clone());
        let input = vec![PrimitiveMessage::InputCurrent(1600.0)];

        let (new_state, outputs) = LifNeuron::update(state, &params, &input);

        assert_eq!(outputs.len(), 1);
        assert_eq!(new_state.membrane_potential, params.v_reset);
    }

    #[test]
    fn lif_no_spike_on_small_current() {
        let params = LifParams::default();
        let state = LifNeuron::initialize(params.clone());
        let input = vec![PrimitiveMessage::InputCurrent(0.1)];

        let (new_state, outputs) = LifNeuron::update(state, &params, &input);

        assert_eq!(outputs.len(), 0);
        assert!(new_state.membrane_potential > params.v_rest);
        assert!(new_state.membrane_potential < params.v_thresh);
    }
}
