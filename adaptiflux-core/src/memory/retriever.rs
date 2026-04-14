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

//! Compose indexer candidates with store metadata / similarity scoring.

use std::collections::HashSet;

use crate::memory::indexer::MetadataIndexer;
use crate::memory::long_term_store::LongTermStore;
use crate::memory::scoring::score_query_against_meta;
use crate::memory::types::{KeyAndScore, Query};

/// Retrieves ranked keys using an indexer for pruning and a store for metadata scoring.
pub struct Retriever {
    pub top_k: usize,
}

impl Default for Retriever {
    fn default() -> Self {
        Self { top_k: 8 }
    }
}

impl Retriever {
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }

    pub fn retrieve(
        &self,
        store: &dyn LongTermStore,
        indexer: &MetadataIndexer,
        query: &Query,
    ) -> Vec<KeyAndScore> {
        let candidates: Vec<_> = match query {
            Query::Hybrid {
                agent,
                tag,
                embedding: _,
            } => {
                let keys = indexer.hybrid_filter_keys(*agent, tag.as_deref());
                if keys.is_empty() && (agent.is_some() || tag.is_some()) {
                    vec![]
                } else if keys.is_empty() {
                    indexer.candidate_keys(query)
                } else {
                    keys
                }
            }
            _ => indexer.candidate_keys(query),
        };

        let mut seen = HashSet::new();
        let mut scored: Vec<KeyAndScore> = Vec::new();

        for key in candidates {
            if !seen.insert(key) {
                continue;
            }
            let Some(meta) = store.retrieve_metadata(&key) else {
                continue;
            };
            let score = score_query_against_meta(query, &meta);
            if score > 0.0 {
                scored.push(KeyAndScore { key, score });
            }
        }

        if scored.is_empty() {
            scored = store.search(query);
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if self.top_k > 0 && scored.len() > self.top_k {
            scored.truncate(self.top_k);
        }
        scored
    }
}
