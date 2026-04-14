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

//! Attention interfaces and reference implementations.

use crate::memory::scoring::cosine_similarity;
use crate::memory::types::MemoryKey;

/// Key side for attention (embedding space).
#[derive(Debug, Clone)]
pub struct AttentionKey {
    pub id: MemoryKey,
    pub vector: Vec<f32>,
}

/// Value side (often same embedding as key for content-based attention).
#[derive(Debug, Clone)]
pub struct AttentionValue {
    pub id: MemoryKey,
    pub vector: Vec<f32>,
}

pub trait AttentionMechanism: Send + Sync {
    fn compute_weights(
        &self,
        query: &[f32],
        keys: &[AttentionKey],
        values: &[AttentionValue],
    ) -> Vec<f32>;
}

/// Scaled dot-product style scores via cosine similarity, then softmax normalize.
pub struct DotProductAttention {
    pub temperature: f32,
}

impl Default for DotProductAttention {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

impl AttentionMechanism for DotProductAttention {
    fn compute_weights(
        &self,
        query: &[f32],
        keys: &[AttentionKey],
        values: &[AttentionValue],
    ) -> Vec<f32> {
        let n = keys.len().min(values.len());
        if n == 0 {
            return vec![];
        }
        let t = self.temperature.max(1e-6);
        let mut logits: Vec<f32> = (0..n)
            .map(|i| cosine_similarity(query, &keys[i].vector) / t)
            .collect();
        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0_f32;
        for x in &mut logits {
            *x = (*x - max_l).exp();
            exp_sum += *x;
        }
        if exp_sum < 1e-12 {
            let u = 1.0 / n as f32;
            return vec![u; n];
        }
        for x in &mut logits {
            *x /= exp_sum;
        }
        logits
    }
}

/// Same as dot-product but uses `key` vectors only (ignores value vectors except length check).
#[derive(Default)]
pub struct ContentBasedAttention {
    pub inner: DotProductAttention,
}

impl AttentionMechanism for ContentBasedAttention {
    fn compute_weights(
        &self,
        query: &[f32],
        keys: &[AttentionKey],
        values: &[AttentionValue],
    ) -> Vec<f32> {
        self.inner.compute_weights(query, keys, values)
    }
}

/// One-hot on the argmax cosine match.
pub struct HardAttentionSelector;

impl AttentionMechanism for HardAttentionSelector {
    fn compute_weights(
        &self,
        query: &[f32],
        keys: &[AttentionKey],
        values: &[AttentionValue],
    ) -> Vec<f32> {
        let n = keys.len().min(values.len());
        if n == 0 {
            return vec![];
        }
        let mut best = 0usize;
        let mut best_s = -1.0_f32;
        for (i, key) in keys.iter().enumerate().take(n) {
            let s = cosine_similarity(query, &key.vector);
            if s > best_s {
                best_s = s;
                best = i;
            }
        }
        let mut w = vec![0.0_f32; n];
        w[best] = 1.0;
        w
    }
}
