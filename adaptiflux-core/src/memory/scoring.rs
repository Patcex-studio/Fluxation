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

//! Similarity and query–metadata scoring helpers.

use crate::memory::types::{Metadata, Query};

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let d = na.sqrt() * nb.sqrt();
    if d < 1e-12 {
        0.0
    } else {
        (dot / d).clamp(0.0, 1.0)
    }
}

/// Score how well `metadata` matches `query` in \[0, 1\] (heuristic).
pub fn score_query_against_meta(query: &Query, metadata: &Metadata) -> f32 {
    match query {
        Query::ByKey(_) => 1.0,
        Query::ByAgent(id) => metadata
            .agent_id
            .map(|a| if a == *id { 1.0 } else { 0.0 })
            .unwrap_or(0.0),
        Query::ByTag(tag) => {
            if metadata.tags.iter().any(|t| t == tag) {
                1.0
            } else {
                0.0
            }
        }
        Query::BySimilarity {
            embedding,
            min_score,
        } => metadata
            .embedding
            .as_ref()
            .map(|e| cosine_similarity(embedding, e))
            .filter(|&s| s >= *min_score)
            .unwrap_or(0.0),
        Query::Hybrid {
            agent,
            tag,
            embedding,
        } => {
            if let Some(id) = agent {
                let m = metadata
                    .agent_id
                    .map(|a| if a == *id { 1.0 } else { 0.0 })
                    .unwrap_or(0.0);
                if m < 0.5 {
                    return 0.0;
                }
            }
            if let Some(t) = tag {
                if !metadata.tags.iter().any(|x| x == t) {
                    return 0.0;
                }
            }
            if let Some(emb) = embedding {
                let sim = metadata
                    .embedding
                    .as_ref()
                    .map(|e| cosine_similarity(emb, e))
                    .unwrap_or(0.0);
                return sim.max(1e-6);
            }
            1.0
        }
    }
}
