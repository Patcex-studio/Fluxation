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
use crate::utils::types::StateValue;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PfsmStateEnum {
    Search,
    Approach,
    Avoid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PfsmParams {
    pub pheromone_threshold: StateValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PfsmState {
    pub current_state: PfsmStateEnum,
}

#[derive(Debug, Clone)]
pub struct Pfsm;

impl Pfsm {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for Pfsm {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PfsmParams {
    fn default() -> Self {
        Self {
            pheromone_threshold: 0.5,
        }
    }
}

impl Primitive for Pfsm {
    type State = PfsmState;
    type Params = PfsmParams;

    fn initialize(_params: Self::Params) -> Self::State {
        PfsmState {
            current_state: PfsmStateEnum::Search,
        }
    }

    fn update(
        mut state: Self::State,
        params: &Self::Params,
        input: &[PrimitiveMessage],
    ) -> (Self::State, Vec<PrimitiveMessage>) {
        let mut outputs = Vec::new();
        let mut found_pheromone = false;
        let mut found_obstacle = false;

        for msg in input {
            match msg {
                PrimitiveMessage::Pheromone { strength, .. }
                    if *strength >= params.pheromone_threshold =>
                {
                    found_pheromone = true;
                }
                PrimitiveMessage::ControlSignal(value) if *value < 0.0 => {
                    found_obstacle = true;
                }
                _ => {}
            }
        }

        state.current_state = if found_obstacle {
            PfsmStateEnum::Avoid
        } else if found_pheromone {
            PfsmStateEnum::Approach
        } else {
            PfsmStateEnum::Search
        };

        let signal = match state.current_state {
            PfsmStateEnum::Search => 0.0,
            PfsmStateEnum::Approach => 1.0,
            PfsmStateEnum::Avoid => -1.0,
        };

        outputs.push(PrimitiveMessage::ControlSignal(signal));

        (state, outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pfsm_moves_to_approach_on_pheromone() {
        let params = PfsmParams::default();
        let state = Pfsm::initialize(params.clone());
        let input = vec![PrimitiveMessage::Pheromone {
            strength: 1.0,
            type_id: 0,
        }];

        let (new_state, outputs) = Pfsm::update(state, &params, &input);

        assert!(matches!(new_state.current_state, PfsmStateEnum::Approach));
        assert_eq!(outputs, vec![PrimitiveMessage::ControlSignal(1.0)]);
    }
}
