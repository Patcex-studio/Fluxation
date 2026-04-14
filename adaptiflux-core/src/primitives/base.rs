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

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PrimitiveMessage {
    InputCurrent(crate::utils::types::StateValue),
    Spike {
        timestamp: u64,
        amplitude: crate::utils::types::StateValue,
    },
    Error(crate::utils::types::StateValue),
    ControlSignal(crate::utils::types::StateValue),
    Pheromone {
        strength: crate::utils::types::StateValue,
        type_id: u8,
    },
    SensorData {
        value: crate::utils::types::StateValue,
    },
    TaskRequest {
        task_id: u64,
    },
    RoleChangeAck {
        role: String,
    },
}

pub trait Primitive {
    type State: Send + Sync + 'static;
    type Params: Clone + Send + Sync + 'static;

    fn initialize(params: Self::Params) -> Self::State;

    fn update(
        state: Self::State,
        params: &Self::Params,
        input: &[PrimitiveMessage],
    ) -> (Self::State, Vec<PrimitiveMessage>);

    fn output(_state: &Self::State, _params: &Self::Params) -> Vec<PrimitiveMessage> {
        vec![]
    }
}
