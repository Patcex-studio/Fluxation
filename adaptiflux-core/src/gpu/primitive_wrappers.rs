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
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchIzhikevichState {
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    pub spikes: Vec<bool>,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchIzhikevichParams {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub d: Vec<f32>,
    pub dt: f32,
}

pub struct BatchIzhikevichPrimitive;

impl BatchIzhikevichPrimitive {
    pub fn describe() {
        debug!("BatchIzhikevichPrimitive is configured for GPU execution.");
    }
}

impl Primitive for BatchIzhikevichPrimitive {
    type State = BatchIzhikevichState;
    type Params = BatchIzhikevichParams;

    fn initialize(params: Self::Params) -> Self::State {
        let count = params.a.len();
        BatchIzhikevichState {
            v: vec![-65.0; count],
            u: vec![0.0; count],
            spikes: vec![false; count],
            count,
        }
    }

    fn update(
        state: Self::State,
        _params: &Self::Params,
        _input: &[PrimitiveMessage],
    ) -> (Self::State, Vec<PrimitiveMessage>) {
        debug!("BatchIzhikevichPrimitive::update called; GPU kernel would execute here.");

        let outputs = state
            .spikes
            .iter()
            .filter_map(|did_spike| {
                if *did_spike {
                    Some(PrimitiveMessage::Spike {
                        timestamp: 0,
                        amplitude: 1.0,
                    })
                } else {
                    None
                }
            })
            .collect();

        (state, outputs)
    }
}
