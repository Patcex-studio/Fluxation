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

//! Represent a set of zoooids as one coordination unit with an aggregation policy.

use crate::core::message_bus::message::Message;
use crate::core::message_bus::{MessageBus, SendError};
use crate::hierarchy::aggregation_functions::{
    aggregate_max, aggregate_mean, aggregate_sum, Value,
};
use crate::utils::types::ZoooidId;

/// How to fold member scalars into a single supervisory signal.
#[derive(Debug, Clone, Copy)]
pub enum AggregationFnKind {
    Mean,
    Max,
    Sum,
}

impl AggregationFnKind {
    pub fn apply(&self, inputs: &[Value]) -> Value {
        match self {
            Self::Mean => aggregate_mean(inputs),
            Self::Max => aggregate_max(inputs),
            Self::Sum => aggregate_sum(inputs),
        }
    }
}

/// One logical super-node over concrete zoooids.
#[derive(Debug, Clone)]
pub struct AgentGroupAbstraction {
    pub members: Vec<ZoooidId>,
    pub aggregation: AggregationFnKind,
}

impl AgentGroupAbstraction {
    pub fn new(members: Vec<ZoooidId>, aggregation: AggregationFnKind) -> Self {
        Self {
            members,
            aggregation,
        }
    }

    /// Fan-out the same message from `hub` to every member (must be registered on the bus).
    pub async fn dispatch_to_members(
        &self,
        bus: &dyn MessageBus,
        hub: ZoooidId,
        msg: Message,
    ) -> Result<(), SendError> {
        for &m in &self.members {
            if m == hub {
                continue;
            }
            bus.send(hub, m, msg.clone()).await?;
        }
        Ok(())
    }

    /// Combine scalar readings (e.g. pheromone strengths) using the configured aggregator.
    pub fn aggregate(&self, values: &[Value]) -> Value {
        self.aggregation.apply(values)
    }
}

/// Keeps a small set of cluster abstractions for rules / demos.
#[derive(Debug, Default, Clone)]
pub struct AbstractionLayerManager {
    groups: Vec<AgentGroupAbstraction>,
}

impl AbstractionLayerManager {
    pub fn clear(&mut self) {
        self.groups.clear();
    }

    pub fn upsert_group(&mut self, members: Vec<ZoooidId>, aggregation: AggregationFnKind) {
        if members.len() < 2 {
            return;
        }
        self.groups
            .retain(|g| !g.members.iter().any(|m| members.contains(m)));
        self.groups
            .push(AgentGroupAbstraction::new(members, aggregation));
    }

    pub fn sync_from_clusters(
        &mut self,
        clusters: Vec<Vec<ZoooidId>>,
        aggregation: AggregationFnKind,
    ) {
        for c in clusters {
            self.upsert_group(c, aggregation);
        }
    }

    pub fn groups(&self) -> &[AgentGroupAbstraction] {
        &self.groups
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
}
