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

//! Long-term storage backends: tabular and graph-linked facts.

use std::collections::HashMap;
use std::sync::Arc;

use petgraph::graph::UnGraph;
use uuid::Uuid;

use crate::memory::indexer::MetadataIndexer;
use crate::memory::scoring::score_query_against_meta;
use crate::memory::types::{KeyAndScore, MemoryKey, Metadata, Query};

/// Core storage trait: write, read by key, and scored search.
pub trait LongTermStore: Send {
    fn store(
        &mut self,
        state: Arc<dyn std::any::Any + Send + Sync>,
        metadata: Metadata,
    ) -> MemoryKey;

    fn retrieve(&self, key: &MemoryKey) -> Option<Arc<dyn std::any::Any + Send + Sync>>;

    fn retrieve_metadata(&self, key: &MemoryKey) -> Option<Metadata>;

    /// Full scan or backend-specific search; used when no indexer is available.
    fn search(&self, query: &Query) -> Vec<KeyAndScore>;
}

/// In-memory row store with optional embeddings for similarity search.
#[derive(Debug, Default)]
pub struct TableLongTermStore {
    rows: HashMap<MemoryKey, (Arc<dyn std::any::Any + Send + Sync>, Metadata)>,
}

impl TableLongTermStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

impl LongTermStore for TableLongTermStore {
    fn store(
        &mut self,
        state: Arc<dyn std::any::Any + Send + Sync>,
        metadata: Metadata,
    ) -> MemoryKey {
        let key = Uuid::new_v4();
        self.rows.insert(key, (state, metadata));
        key
    }

    fn retrieve(&self, key: &MemoryKey) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.rows.get(key).map(|(v, _)| Arc::clone(v))
    }

    fn retrieve_metadata(&self, key: &MemoryKey) -> Option<Metadata> {
        self.rows.get(key).map(|(_, m)| m.clone())
    }

    fn search(&self, query: &Query) -> Vec<KeyAndScore> {
        let mut out = Vec::new();
        for (key, (_, meta)) in &self.rows {
            let s = score_query_against_meta(query, meta);
            if s > 0.0 {
                out.push(KeyAndScore {
                    key: *key,
                    score: s,
                });
            }
        }
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}

/// Graph of related memory keys; payloads live in an inner [`TableLongTermStore`].
#[derive(Debug)]
pub struct GraphLongTermStore {
    pub table: TableLongTermStore,
    graph: UnGraph<MemoryKey, ()>,
    key_to_node: HashMap<MemoryKey, petgraph::graph::NodeIndex>,
}

impl Default for GraphLongTermStore {
    fn default() -> Self {
        Self {
            table: TableLongTermStore::new(),
            graph: UnGraph::new_undirected(),
            key_to_node: HashMap::new(),
        }
    }
}

impl GraphLongTermStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_node(&mut self, key: MemoryKey) -> petgraph::graph::NodeIndex {
        if let Some(&ix) = self.key_to_node.get(&key) {
            return ix;
        }
        let ix = self.graph.add_node(key);
        self.key_to_node.insert(key, ix);
        ix
    }

    /// Link two stored keys (both should already exist in `table`).
    pub fn link(&mut self, a: MemoryKey, b: MemoryKey) {
        let na = self.ensure_node(a);
        let nb = self.ensure_node(b);
        if na != nb && !self.graph.contains_edge(na, nb) {
            self.graph.add_edge(na, nb, ());
        }
    }

    /// Seed scores from `table.search`, then add neighbors with damped score.
    pub fn search_with_expansion(&self, query: &Query, neighbor_bonus: f32) -> Vec<KeyAndScore> {
        let mut scores: HashMap<MemoryKey, f32> = HashMap::new();
        for ks in self.table.search(query) {
            scores.insert(ks.key, ks.score);
        }
        let snapshot: Vec<(MemoryKey, f32)> = scores.iter().map(|(&k, &s)| (k, s)).collect();
        for (k, s) in snapshot {
            let Some(&n) = self.key_to_node.get(&k) else {
                continue;
            };
            for neigh in self.graph.neighbors(n) {
                let nk = self.graph[neigh];
                let ns = s * neighbor_bonus;
                scores
                    .entry(nk)
                    .and_modify(|e| *e = e.max(ns))
                    .or_insert(ns);
            }
        }
        let mut out: Vec<KeyAndScore> = scores
            .into_iter()
            .map(|(key, score)| KeyAndScore { key, score })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}

impl LongTermStore for GraphLongTermStore {
    fn store(
        &mut self,
        state: Arc<dyn std::any::Any + Send + Sync>,
        metadata: Metadata,
    ) -> MemoryKey {
        let key = self.table.store(state, metadata);
        self.ensure_node(key);
        key
    }

    fn retrieve(&self, key: &MemoryKey) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        self.table.retrieve(key)
    }

    fn retrieve_metadata(&self, key: &MemoryKey) -> Option<Metadata> {
        self.table.retrieve_metadata(key)
    }

    fn search(&self, query: &Query) -> Vec<KeyAndScore> {
        self.search_with_expansion(query, 0.35)
    }
}

/// Register all keys currently in `store` into `indexer` (e.g. after loading a checkpoint).
pub fn reindex_table(store: &TableLongTermStore, indexer: &mut MetadataIndexer) {
    for (key, (_, meta)) in &store.rows {
        indexer.register(*key, meta);
    }
}
