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

//! Index structures for narrowing candidate keys before scoring.

use std::collections::{HashMap, HashSet};

use crate::memory::types::{MemoryKey, Metadata, Query};
use crate::utils::types::ZoooidId;

/// Maintains inverted indices over metadata for faster candidate selection.
#[derive(Debug, Default)]
pub struct MetadataIndexer {
    by_agent: HashMap<ZoooidId, Vec<MemoryKey>>,
    by_tag: HashMap<String, Vec<MemoryKey>>,
    all_keys: Vec<MemoryKey>,
}

impl MetadataIndexer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, key: MemoryKey, metadata: &Metadata) {
        if !self.all_keys.contains(&key) {
            self.all_keys.push(key);
        }
        if let Some(a) = metadata.agent_id {
            self.by_agent.entry(a).or_default().push(key);
        }
        for t in &metadata.tags {
            self.by_tag.entry(t.clone()).or_default().push(key);
        }
    }

    /// Candidate keys to retrieve and score (may contain duplicates — retriever dedupes).
    pub fn candidate_keys(&self, query: &Query) -> Vec<MemoryKey> {
        match query {
            Query::ByKey(k) => vec![*k],
            Query::ByAgent(id) => self.by_agent.get(id).cloned().unwrap_or_default(),
            Query::ByTag(tag) => self.by_tag.get(tag).cloned().unwrap_or_default(),
            Query::BySimilarity { .. } | Query::Hybrid { .. } => self.all_keys.clone(),
        }
    }

    /// Keys matching structural filters for a hybrid query (before embedding score).
    pub fn hybrid_filter_keys(&self, agent: Option<ZoooidId>, tag: Option<&str>) -> Vec<MemoryKey> {
        match (agent, tag) {
            (Some(a), Some(t)) => {
                let sa: HashSet<MemoryKey> = self
                    .by_agent
                    .get(&a)
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                self.by_tag
                    .get(t)
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|k| sa.contains(k))
                    .collect()
            }
            (Some(a), None) => self.by_agent.get(&a).cloned().unwrap_or_default(),
            (None, Some(t)) => self.by_tag.get(t).cloned().unwrap_or_default(),
            (None, None) => self.all_keys.clone(),
        }
    }
}
