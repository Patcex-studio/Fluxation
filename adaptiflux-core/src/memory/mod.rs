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

pub mod indexer;
pub mod long_term_store;
pub mod memory_integration;
pub mod retriever;
pub mod scoring;
pub mod types;

pub use indexer::MetadataIndexer;
pub use long_term_store::{reindex_table, GraphLongTermStore, LongTermStore, TableLongTermStore};
pub use memory_integration::{
    agent_similarity_query, build_weighted_payload, content_attention_weights, retrieve_and_weight,
    simple_situation_embedding, store_scalar_experience, summarize_step_inputs, ExperienceRecorder,
};
pub use retriever::Retriever;
pub use scoring::{cosine_similarity, score_query_against_meta};
pub use types::{
    KeyAndScore, MemoryEntryPayload, MemoryKey, MemoryPayload, MemorySummary, Metadata, Query,
};
