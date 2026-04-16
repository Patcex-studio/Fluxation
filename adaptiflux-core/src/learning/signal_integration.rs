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

//! Build learning feedback from message traffic and explicit scalars.

use std::collections::HashMap;

use crate::core::message_bus::message::Message;
use crate::core::message_bus::{MessageBus, RecvError};
use crate::utils::types::ZoooidId;

/// Scalar feedback packaged for [`crate::learning::online_adaptation::OnlineAdaptationEngine`].
#[derive(Debug, Clone, Default)]
pub struct FeedbackSignal {
    /// Per-agent error or reward channel (e.g. last `Message::Error` seen for that id).
    pub per_agent: HashMap<ZoooidId, crate::utils::types::StateValue>,
    /// Optional global term (e.g. fleet-average error).
    pub global_scalar: Option<crate::utils::types::StateValue>,
    /// Extra bias from long-term memory / attention (merged by [`crate::core::scheduler::CoreScheduler`]).
    pub memory_bias: HashMap<ZoooidId, crate::utils::types::StateValue>,
}

impl FeedbackSignal {
    pub fn error_for(&self, id: ZoooidId) -> Option<crate::utils::types::StateValue> {
        self.per_agent.get(&id).copied()
    }

    pub fn merge_scalar(&mut self, id: ZoooidId, value: crate::utils::types::StateValue) {
        self.per_agent.insert(id, value);
    }

    pub fn set_global(&mut self, v: crate::utils::types::StateValue) {
        self.global_scalar = Some(v);
    }

    pub fn merge_memory_bias(&mut self, id: ZoooidId, v: crate::utils::types::StateValue) {
        self.memory_bias.insert(id, v);
    }
}

/// Drain pending `Message::Error` values from each agent inbox (non-blocking).
pub async fn collect_error_feedback_from_bus(
    bus: &dyn MessageBus,
    agent_ids: &[ZoooidId],
) -> Result<FeedbackSignal, RecvError> {
    let mut out = FeedbackSignal::default();
    for &id in agent_ids {
        let msgs = bus.receive(id).await?;
        let mut last_err = None;
        for (_sender, msg) in msgs {
            if let Message::Error(e) = msg {
                last_err = Some(e);
            }
        }
        if let Some(e) = last_err {
            out.per_agent.insert(id, e);
        }
    }
    Ok(out)
}

/// After an agent step, fold outputs that encode error (e.g. monitor zooid emitted `Message::Error`).
pub fn merge_from_posted_errors(
    posted: &[(ZoooidId, crate::utils::types::StateValue)],
    into: &mut FeedbackSignal,
) {
    for &(id, e) in posted {
        into.per_agent.insert(id, e);
    }
}
