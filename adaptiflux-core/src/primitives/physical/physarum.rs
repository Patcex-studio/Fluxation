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
use crate::utils::types::{Param, StateValue};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysarumParams {
    pub diffusion_coeff: Param,
    pub flow_sensitivity: Param,
    pub signal_decay: Param,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysarumState {
    pub density: StateValue,
    pub flow_strength: StateValue,
    pub signal_strength: StateValue,
}

#[derive(Debug, Clone)]
pub struct PhysarumModel;

impl PhysarumModel {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for PhysarumModel {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PhysarumParams {
    fn default() -> Self {
        Self {
            diffusion_coeff: 0.1,
            flow_sensitivity: 0.5,
            signal_decay: 0.05,
        }
    }
}

impl Primitive for PhysarumModel {
    type State = PhysarumState;
    type Params = PhysarumParams;

    fn initialize(_params: Self::Params) -> Self::State {
        PhysarumState {
            density: 0.1,
            flow_strength: 0.0,
            signal_strength: 0.0,
        }
    }

    fn update(
        mut state: Self::State,
        params: &Self::Params,
        input: &[PrimitiveMessage],
    ) -> (Self::State, Vec<PrimitiveMessage>) {
        let mut outputs = Vec::new();
        let incoming_signal: StateValue = input
            .iter()
            .filter_map(|msg| {
                if let PrimitiveMessage::Pheromone {
                    strength,
                    type_id: 0,
                } = msg
                {
                    Some(*strength)
                } else {
                    None
                }
            })
            .sum();

        state.density += incoming_signal * params.diffusion_coeff;
        state.density *= 1.0 - params.signal_decay;
        state.signal_strength += incoming_signal * params.flow_sensitivity;
        state.signal_strength *= 1.0 - params.signal_decay;
        state.flow_strength = state.density * state.signal_strength;

        outputs.push(PrimitiveMessage::ControlSignal(state.flow_strength));

        (state, outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physarum_generates_flow_signal() {
        let params = PhysarumParams::default();
        let state = PhysarumModel::initialize(params.clone());
        let input = vec![PrimitiveMessage::Pheromone {
            strength: 1.0,
            type_id: 0,
        }];

        let (new_state, outputs) = PhysarumModel::update(state, &params, &input);

        assert!(!outputs.is_empty());
        if let PrimitiveMessage::ControlSignal(flow) = outputs[0] {
            assert!(flow > 0.0);
        } else {
            panic!("Expected ControlSignal: got {:?}", outputs[0]);
        }
        assert!(new_state.density > 0.1);
    }
}
