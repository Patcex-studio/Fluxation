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
pub struct IzhikevichParams {
    pub a: crate::utils::types::Param,
    pub b: crate::utils::types::Param,
    pub c: crate::utils::types::Param,
    pub d: crate::utils::types::Param,
    pub dt: crate::utils::types::Param,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IzhikevichState {
    pub v: crate::utils::types::StateValue,
    pub u: crate::utils::types::StateValue,
}

#[derive(Debug, Clone)]
pub struct IzhikevichNeuron;

impl IzhikevichNeuron {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for IzhikevichNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for IzhikevichParams {
    fn default() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            dt: 0.1,
        }
    }
}

impl Primitive for IzhikevichNeuron {
    type State = IzhikevichState;
    type Params = IzhikevichParams;

    fn initialize(_params: Self::Params) -> Self::State {
        IzhikevichState { v: -65.0, u: -13.0 }
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

        if state.v >= 30.0 {
            outputs.push(PrimitiveMessage::Spike {
                timestamp: 0,
                amplitude: 1.0,
            });
            state.v = params.c;
            state.u += params.d;
            return (state, outputs);
        }

        state.v += 0.04 * state.v * state.v + 5.0 * state.v + 140.0 - state.u + total_input_current;
        state.u += params.a * (params.b * state.v - state.u) * params.dt;

        if state.v >= 30.0 {
            outputs.push(PrimitiveMessage::Spike {
                timestamp: 0,
                amplitude: 1.0,
            });
            state.v = params.c;
            state.u += params.d;
        }

        (state, outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn izhikevich_spike_after_high_current() {
        let params = IzhikevichParams::default();
        let state = IzhikevichNeuron::initialize(params.clone());
        let input = vec![PrimitiveMessage::InputCurrent(1000.0)];

        let (_new_state, outputs) = IzhikevichNeuron::update(state, &params, &input);

        assert!(!outputs.is_empty());
        assert!(matches!(outputs[0], PrimitiveMessage::Spike { .. }));
    }
}
