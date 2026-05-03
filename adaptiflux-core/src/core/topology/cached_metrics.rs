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

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Cached metrics with lazy invalidation.
/// O(1) access when clean, O(N²) recalculation when dirty.
#[derive(Debug, Clone)]
pub struct CachedMetrics {
    inner: Arc<CachedMetricsInner>,
}

#[derive(Debug)]
struct CachedMetricsInner {
    clustering_coefficient: Mutex<Option<f64>>,
    network_diameter: Mutex<Option<usize>>,
    avg_connectivity: Mutex<Option<f64>>,
    dirty: AtomicBool,
}

impl CachedMetrics {
    /// Create new cache (starts dirty).
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CachedMetricsInner {
                clustering_coefficient: Mutex::new(None),
                network_diameter: Mutex::new(None),
                avg_connectivity: Mutex::new(None),
                dirty: AtomicBool::new(true),
            }),
        }
    }

    /// Mark cache as needing recomputation.
    pub fn invalidate(&self) {
        self.inner.dirty.store(true, Ordering::Relaxed);
    }

    /// Check if cache is dirty (needs recomputation).
    pub fn is_dirty(&self) -> bool {
        self.inner.dirty.load(Ordering::Relaxed)
    }

    /// Mark cache as clean after recomputation.
    pub fn mark_clean(&self) {
        self.inner.dirty.store(false, Ordering::Relaxed);
    }

    /// Set clustering coefficient and mark clean.
    pub fn set_clustering_coefficient(&self, value: f64) {
        *self.inner.clustering_coefficient.lock().unwrap() = Some(value);
    }

    /// Set network diameter and mark clean.
    pub fn set_network_diameter(&self, value: usize) {
        *self.inner.network_diameter.lock().unwrap() = Some(value);
    }

    /// Set average connectivity and mark clean.
    pub fn set_avg_connectivity(&self, value: f64) {
        *self.inner.avg_connectivity.lock().unwrap() = Some(value);
    }

    /// Get clustering coefficient (None if not computed).
    pub fn get_clustering_coefficient(&self) -> Option<f64> {
        *self.inner.clustering_coefficient.lock().unwrap()
    }

    /// Get network diameter (None if not computed).
    pub fn get_network_diameter(&self) -> Option<usize> {
        *self.inner.network_diameter.lock().unwrap()
    }

    /// Get average connectivity (None if not computed).
    pub fn get_avg_connectivity(&self) -> Option<f64> {
        *self.inner.avg_connectivity.lock().unwrap()
    }

    /// Clear all cache values and mark dirty.
    pub fn clear(&self) {
        *self.inner.clustering_coefficient.lock().unwrap() = None;
        *self.inner.network_diameter.lock().unwrap() = None;
        *self.inner.avg_connectivity.lock().unwrap() = None;
        self.inner.dirty.store(true, Ordering::Relaxed);
    }
}

impl Default for CachedMetrics {
    fn default() -> Self {
        Self::new()
    }
}
