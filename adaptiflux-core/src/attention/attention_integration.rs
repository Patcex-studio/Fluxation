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

//! Glue between retrieval scores, metadata embeddings, and [`crate::attention::attention_mechanism`].

use crate::attention::attention_mechanism::AttentionMechanism;
use crate::attention::attention_mechanism::{AttentionKey, AttentionValue};
use crate::memory::long_term_store::LongTermStore;
use crate::memory::memory_integration::build_weighted_payload;
use crate::memory::types::{KeyAndScore, MemoryPayload};

/// Build attention keys/values from store metadata for ranked retrieval hits.
pub fn keys_values_from_hits(
    store: &dyn LongTermStore,
    hits: &[KeyAndScore],
) -> (Vec<AttentionKey>, Vec<AttentionValue>) {
    let mut keys = Vec::new();
    let mut vals = Vec::new();
    for ks in hits {
        let emb = store
            .retrieve_metadata(&ks.key)
            .and_then(|m| m.embedding)
            .unwrap_or_else(|| vec![ks.score, 0.0, 0.0]);
        keys.push(AttentionKey {
            id: ks.key,
            vector: emb.clone(),
        });
        vals.push(AttentionValue {
            id: ks.key,
            vector: emb,
        });
    }
    (keys, vals)
}

/// Full pipeline: ranked keys → attention weights → [`MemoryPayload`].
pub fn apply_attention_to_hits(
    store: &dyn LongTermStore,
    mechanism: &dyn AttentionMechanism,
    query_embedding: &[f32],
    hits: &[KeyAndScore],
) -> MemoryPayload {
    if hits.is_empty() {
        return MemoryPayload::default();
    }
    let (keys, vals) = keys_values_from_hits(store, hits);
    let w = mechanism.compute_weights(query_embedding, &keys, &vals);
    let pairs: Vec<_> = hits.iter().map(|ks| (ks.key, ks.score)).collect();
    build_weighted_payload(store, &pairs, &w)
}
