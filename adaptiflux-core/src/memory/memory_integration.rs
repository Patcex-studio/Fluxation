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

//! Helpers to wire memory retrieve/store into the scheduler and blueprints.

use std::sync::Arc;

use crate::agent::state::AgentUpdateResult;
use crate::core::message_bus::message::Message;
use crate::memory::indexer::MetadataIndexer;
use crate::memory::long_term_store::{LongTermStore, TableLongTermStore};
use crate::memory::retriever::Retriever;
use crate::memory::scoring::cosine_similarity;
use crate::memory::types::{
    MemoryEntryPayload, MemoryKey, MemoryPayload, MemorySummary, Metadata, Query,
};
use crate::utils::types::{StateValue, ZoooidId};

/// Build [`MemoryPayload`] from ranked keys, retrieval scores, and attention weights.
pub fn build_weighted_payload(
    store: &dyn LongTermStore,
    keys_and_retrieval: &[(MemoryKey, f32)],
    attention_weights: &[f32],
) -> MemoryPayload {
    let mut entries = Vec::new();
    let n = keys_and_retrieval.len().min(attention_weights.len());
    for i in 0..n {
        let (key, rscore) = keys_and_retrieval[i];
        let w = attention_weights[i] * rscore;
        let Some(data) = store.retrieve(&key) else {
            continue;
        };
        entries.push(MemoryEntryPayload {
            key,
            weight: w,
            data,
            summary: MemorySummary::Scalar(w),
        });
    }
    MemoryPayload { entries }
}

/// Store a compact experience record after an agent step (optional tag e.g. `"pid_tune"`).
#[allow(clippy::too_many_arguments)]
pub fn store_scalar_experience(
    store: &mut dyn LongTermStore,
    indexer: &mut MetadataIndexer,
    agent_id: ZoooidId,
    iteration: u64,
    tag: &str,
    embedding: Option<Vec<f32>>,
    payload: Arc<dyn std::any::Any + Send + Sync>,
    scalar_summary: StateValue,
) -> MemoryKey {
    let emb = embedding.or_else(|| Some(vec![scalar_summary, 0.0, 0.0]));
    let meta = Metadata {
        agent_id: Some(agent_id),
        iteration: Some(iteration),
        tags: vec![tag.to_string()],
        embedding: emb,
    };
    let key = store.store(payload, meta.clone());
    indexer.register(key, &meta);
    key
}

/// Convenience: embedding from a few scalars (situation fingerprint).
pub fn simple_situation_embedding(
    error: StateValue,
    observation: StateValue,
    extra: StateValue,
) -> Vec<f32> {
    vec![error, observation, extra]
}

/// Default query for “similar situations for this agent”.
pub fn agent_similarity_query(agent_id: ZoooidId, embedding: Vec<f32>, _min_score: f32) -> Query {
    Query::Hybrid {
        agent: Some(agent_id),
        tag: None,
        embedding: Some(embedding),
    }
}

/// Run retriever + optional attention weights (uniform if `attention` is empty).
pub fn retrieve_and_weight(
    store: &dyn LongTermStore,
    indexer: &MetadataIndexer,
    retriever: &Retriever,
    query: &Query,
    attention_weights: &[f32],
) -> MemoryPayload {
    let ranked = retriever.retrieve(store, indexer, query);
    let keys_scores: Vec<(MemoryKey, f32)> = ranked.iter().map(|ks| (ks.key, ks.score)).collect();
    let mut weights = attention_weights.to_vec();
    if weights.len() != keys_scores.len() {
        let u = if keys_scores.is_empty() {
            0.0
        } else {
            1.0 / keys_scores.len() as f32
        };
        weights = vec![u; keys_scores.len()];
    }
    build_weighted_payload(store, &keys_scores, &weights)
}

/// Derive content-based attention weights from query embedding vs stored metadata embeddings.
pub fn content_attention_weights(
    store: &dyn LongTermStore,
    keys: &[MemoryKey],
    query_emb: &[f32],
) -> Vec<f32> {
    let mut raw: Vec<f32> = keys
        .iter()
        .filter_map(|k| store.retrieve_metadata(k))
        .map(|m| {
            m.embedding
                .as_ref()
                .map(|e| cosine_similarity(query_emb, e))
                .unwrap_or(0.0)
        })
        .collect();
    if raw.is_empty() {
        return vec![];
    }
    let sum: f32 = raw.iter().sum();
    if sum < 1e-12 {
        let u = 1.0 / raw.len() as f32;
        raw.fill(u);
    } else {
        for x in &mut raw {
            *x /= sum;
        }
    }
    raw
}

/// Serialize salient scalars from [`AgentUpdateResult`] + last error from inputs for memory tagging.
pub fn summarize_step_inputs(inputs: &[Message]) -> Option<StateValue> {
    for m in inputs.iter().rev() {
        if let Message::Error(e) = m {
            return Some(*e);
        }
    }
    None
}

/// Optional hook to persist experiences after [`crate::core::scheduler::CoreScheduler`] runs an agent.
pub trait ExperienceRecorder: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn record_after_step(
        &self,
        agent_id: ZoooidId,
        iteration: u64,
        inputs: &[(ZoooidId, Message)],
        state: &dyn std::any::Any,
        result: &AgentUpdateResult,
        store: &mut TableLongTermStore,
        indexer: &mut MetadataIndexer,
    );
}
