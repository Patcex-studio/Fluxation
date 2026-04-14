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

use async_trait::async_trait;
use std::any::Any;

use crate::agent::state::RoleType;
use crate::core::message_bus::MessageBus;
use crate::core::topology::ZoooidTopology;
use crate::utils::types::ZoooidId;

#[derive(Debug)]
pub enum BehaviorAction {
    ChangeRole(RoleType),
    ModifyParameters(std::collections::HashMap<String, String>),
}

#[async_trait]
pub trait BehaviorRule: Send + Sync {
    async fn evaluate(
        &self,
        agent_id: ZoooidId,
        state: &dyn Any,
        topology: &ZoooidTopology,
        bus: &dyn MessageBus,
    ) -> Result<Option<BehaviorAction>, Box<dyn std::error::Error + Send + Sync>>;
}
