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

//! Map runtime context to a memory [`Query`].

use crate::agent::state::RoleType;
use crate::core::message_bus::message::Message;
use crate::memory::types::Query;
use crate::utils::types::ZoooidId;

pub trait FocusScheduler: Send + Sync {
    fn focus_query(
        &self,
        agent_id: ZoooidId,
        iteration: u64,
        inputs: &[Message],
        role: RoleType,
    ) -> Query;

    /// Query-side embedding for [`crate::attention::attention_mechanism::AttentionMechanism`]
    /// (should match the situation vector implied by [`Self::focus_query`]).
    fn query_vector(
        &self,
        agent_id: ZoooidId,
        iteration: u64,
        inputs: &[Message],
        role: RoleType,
    ) -> Vec<f32>;
}

/// Uses last `Message::Error` (if any) with `observation` to build a hybrid similarity query.
pub struct ErrorSimilarityFocus {
    pub tag: Option<String>,
    pub min_similarity: f32,
    pub observation_fn: fn(&[Message]) -> crate::utils::types::StateValue,
}

impl Default for ErrorSimilarityFocus {
    fn default() -> Self {
        Self {
            tag: None,
            min_similarity: 0.0,
            observation_fn: |_| 0.0,
        }
    }
}

impl ErrorSimilarityFocus {
    fn situation_embedding(&self, inputs: &[Message]) -> Vec<f32> {
        let mut err = 0.0_f32;
        for m in inputs.iter().rev() {
            if let Message::Error(e) = m {
                err = *e;
                break;
            }
        }
        let obs = (self.observation_fn)(inputs);
        vec![err, obs, 0.0]
    }
}

impl FocusScheduler for ErrorSimilarityFocus {
    fn focus_query(
        &self,
        agent_id: ZoooidId,
        _iteration: u64,
        inputs: &[Message],
        _role: RoleType,
    ) -> Query {
        let emb = self.situation_embedding(inputs);
        Query::Hybrid {
            agent: Some(agent_id),
            tag: self.tag.clone(),
            embedding: Some(emb),
        }
    }

    fn query_vector(
        &self,
        _agent_id: ZoooidId,
        _iteration: u64,
        inputs: &[Message],
        _role: RoleType,
    ) -> Vec<f32> {
        self.situation_embedding(inputs)
    }
}

/// Pheromone / neighbor-focused query for swarm-style agents: embed strongest signal strengths.
pub struct PheromoneFocus {
    pub top_signals: usize,
}

impl Default for PheromoneFocus {
    fn default() -> Self {
        Self { top_signals: 3 }
    }
}

impl PheromoneFocus {
    fn pheromone_embedding(&self, inputs: &[Message]) -> Vec<f32> {
        let mut strengths: Vec<f32> = inputs
            .iter()
            .filter_map(|m| {
                if let Message::PheromoneLevel(s, _) = m {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        strengths.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        strengths.truncate(self.top_signals.max(1));
        while strengths.len() < 4 {
            strengths.push(0.0);
        }
        strengths
    }
}

impl FocusScheduler for PheromoneFocus {
    fn focus_query(
        &self,
        agent_id: ZoooidId,
        _iteration: u64,
        inputs: &[Message],
        role: RoleType,
    ) -> Query {
        let _ = role;
        let strengths = self.pheromone_embedding(inputs);
        Query::Hybrid {
            agent: Some(agent_id),
            tag: Some("pheromone_context".into()),
            embedding: Some(strengths),
        }
    }

    fn query_vector(
        &self,
        _agent_id: ZoooidId,
        _iteration: u64,
        inputs: &[Message],
        _role: RoleType,
    ) -> Vec<f32> {
        self.pheromone_embedding(inputs)
    }
}
