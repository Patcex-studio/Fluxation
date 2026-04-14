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

use std::future::Future;

use tracing;

/// Configuration for batched asynchronous agent execution.
#[derive(Debug, Clone)]
pub struct AsyncOptimizationConfig {
    pub max_concurrent_updates: usize,
}

impl AsyncOptimizationConfig {
    pub fn new(max_concurrent_updates: usize) -> Self {
        Self {
            max_concurrent_updates: max_concurrent_updates.max(1),
        }
    }

    /// Execute an iterator of futures in bounded batches, reducing task scheduling overhead.
    pub async fn run_batched<T, Fut>(&self, tasks: Vec<Fut>) -> Vec<T>
    where
        Fut: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let mut results = Vec::with_capacity(tasks.len());
        let mut pending = tasks.into_iter();

        while let Some(task) = pending.next() {
            let mut handles = Vec::new();
            handles.push(tokio::task::spawn(task));
            for _ in 1..self.max_concurrent_updates {
                if let Some(task) = pending.next() {
                    handles.push(tokio::task::spawn(task));
                } else {
                    break;
                }
            }

            for handle in handles {
                match handle.await {
                    Ok(output) => results.push(output),
                    Err(err) => {
                        tracing::warn!("AsyncOptimizationConfig task panic: {}", err);
                        // Skip failed task
                    }
                }
            }
        }

        results
    }
}
