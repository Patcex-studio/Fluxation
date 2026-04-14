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

//! Shared types for long-term memory (keys, metadata, queries, scheduler-facing views).

use std::sync::Arc;

use uuid::Uuid;

use crate::utils::types::ZoooidId;

/// Stable identifier for a stored memory item.
pub type MemoryKey = Uuid;

/// Metadata attached at store time for indexing and retrieval.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    pub agent_id: Option<ZoooidId>,
    pub iteration: Option<u64>,
    pub tags: Vec<String>,
    /// Optional embedding for similarity / attention (e.g. situation vector).
    pub embedding: Option<Vec<f32>>,
}

/// Retrieval request: by id, filters, or similarity.
#[derive(Debug, Clone)]
pub enum Query {
    ByKey(MemoryKey),
    ByAgent(ZoooidId),
    ByTag(String),
    BySimilarity {
        embedding: Vec<f32>,
        min_score: f32,
    },
    Hybrid {
        agent: Option<ZoooidId>,
        tag: Option<String>,
        embedding: Option<Vec<f32>>,
    },
}

/// Key with an unnormalized relevance score (higher is better).
#[derive(Debug, Clone)]
pub struct KeyAndScore {
    pub key: MemoryKey,
    pub score: f32,
}

/// Scalar or vector hint for blueprints that do not downcast stored payloads.
#[derive(Debug, Clone)]
pub enum MemorySummary {
    Scalar(f32),
    Vector(Vec<f32>),
    Empty,
}

/// One weighted memory item passed into [`crate::agent::blueprint::AgentBlueprint::update`].
#[derive(Clone)]
pub struct MemoryEntryPayload {
    pub key: MemoryKey,
    pub weight: f32,
    pub data: Arc<dyn std::any::Any + Send + Sync>,
    pub summary: MemorySummary,
}

/// Bundle produced by the scheduler after retrieve + attention.
#[derive(Clone, Default)]
pub struct MemoryPayload {
    pub entries: Vec<MemoryEntryPayload>,
}

impl MemoryPayload {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Dominant scalar summary if any entry carries `MemorySummary::Scalar`.
    pub fn weighted_scalar_hint(&self) -> Option<f32> {
        let mut num = 0.0_f32;
        let mut den = 0.0_f32;
        for e in &self.entries {
            if let MemorySummary::Scalar(v) = e.summary {
                let w = e.weight;
                num += v * w;
                den += w;
            }
        }
        if den > 1e-12 {
            Some(num / den)
        } else {
            None
        }
    }
}
