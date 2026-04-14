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

use std::any::Any;

use crate::agent::blueprint::AgentBlueprint;
use crate::utils::types::ZoooidId;

pub struct Zoooid {
    pub id: ZoooidId,
    pub blueprint: Box<dyn AgentBlueprint + Send + Sync>,
    pub state: Box<dyn Any + Send + Sync>,
}

impl Zoooid {
    pub async fn new(
        id: ZoooidId,
        blueprint: Box<dyn AgentBlueprint + Send + Sync>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let state = blueprint.initialize().await?;
        Ok(Self {
            id,
            blueprint,
            state,
        })
    }
}
