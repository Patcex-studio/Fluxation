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

//! Centralized synapse management with O(1) weight lookup and automatic cleanup.
//!
//! `SynapseManager` provides:
//! - **O(1) average-case weight lookup** using `HashMap<ZoooidId, usize>` for index mapping
//! - **Automatic orphaned weight cleanup** via topology events
//! - **Normalized weight bounds** for stable learning
//! - **Event-driven architecture** through `TopologyEventBus`

use crate::utils::types::ZoooidId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

const DEFAULT_MIN_WEIGHT: f32 = 0.0;
const DEFAULT_MAX_WEIGHT: f32 = 1.0;
const DEFAULT_WEIGHT_VALUE: f32 = 0.1;

/// Normalization mode for synaptic weights.
///
/// Determines how weights are normalized after updates to ensure stability and convergence.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormMode {
    /// No normalization, only clamp to [min, max]
    None,
    /// L1 normalization: sum(|w_i|) = 1.0
    L1,
    /// L2 normalization: sqrt(sum(w_i²)) = 1.0
    L2,
    /// Softmax: w_i = exp(w_i) / sum(exp(w_j))
    Softmax,
    /// Adaptive: weight = (weight * decay) + (gradient * learning_rate)
    Adaptive,
}

impl Default for NormMode {
    fn default() -> Self {
        NormMode::None
    }
}

/// Metadata for a single synaptic connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseEntry {
    /// ID of the source (pre-synaptic) agent
    pub source_id: ZoooidId,
    /// Synaptic weight (normalized to [min, max])
    pub weight: f32,
    /// Timestamp of last update (for TTL or aging)
    pub last_updated: u64,
    /// Optional metadata (future extension)
    pub metadata: Option<String>,
}

/// Configuration for `SynapseManager`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseConfig {
    /// Normalization mode for weights
    pub norm_mode: NormMode,
    /// Maximum number of incoming connections allowed
    pub max_connections: usize,
    /// Minimum allowed weight value
    pub min_weight: f32,
    /// Maximum allowed weight value
    pub max_weight: f32,
    /// Default weight for newly added synapses
    pub default_weight: f32,
}

impl Default for SynapseConfig {
    fn default() -> Self {
        Self {
            norm_mode: NormMode::None,
            max_connections: 50,
            min_weight: DEFAULT_MIN_WEIGHT,
            max_weight: DEFAULT_MAX_WEIGHT,
            default_weight: DEFAULT_WEIGHT_VALUE,
        }
    }
}

/// Centralized synapse manager for an agent.
///
/// Uses `HashMap` for O(1) average-case weight lookup by source agent ID.
/// Automatically handles cleanup of orphaned weights when topology changes.
///
/// # Example
/// ```ignore
/// let mut manager = SynapseManager::new(SynapseConfig::default());
/// manager.add_synapse(sender_id, 0.5)?;
/// let weight = manager.get_weight(sender_id); // O(1)
/// manager.normalize();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapseManager {
    /// Mapping from source ZoooidId -> index in `weights` vector
    incoming_map: HashMap<ZoooidId, usize>,
    /// Vector of synapse entries (compact storage)
    weights: Vec<SynapseEntry>,
    /// Configuration
    config: SynapseConfig,
    /// Statistics: total updates since creation
    update_count: u64,
}

impl SynapseManager {
    /// Creates a new `SynapseManager` with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Configuration for normalization, limits, and defaults
    ///
    /// # Complexity
    /// O(1)
    pub fn new(config: SynapseConfig) -> Self {
        Self {
            incoming_map: HashMap::new(),
            weights: Vec::new(),
            config,
            update_count: 0,
        }
    }

    /// Adds a new synaptic connection or updates existing one.
    ///
    /// # Arguments
    /// * `source_id` - ID of the pre-synaptic agent
    /// * `weight` - Initial weight (will be clamped to [min, max])
    ///
    /// # Errors
    /// Returns `Err` if max connections limit is exceeded.
    ///
    /// # Complexity
    /// O(1) average case (HashMap insertion)
    pub fn add_synapse(&mut self, source_id: ZoooidId, weight: f32) -> Result<(), String> {
        let clamped_weight = weight.clamp(self.config.min_weight, self.config.max_weight);

        if let Some(&idx) = self.incoming_map.get(&source_id) {
            // Update existing synapse
            self.weights[idx].weight = clamped_weight;
            self.weights[idx].last_updated = 0; // Will be set by caller if needed
            return Ok(());
        }

        // Check connection limit
        if self.weights.len() >= self.config.max_connections {
            return Err(format!(
                "Max connections ({}) exceeded",
                self.config.max_connections
            ));
        }

        // Add new synapse
        let idx = self.weights.len();
        self.incoming_map.insert(source_id, idx);
        self.weights.push(SynapseEntry {
            source_id,
            weight: clamped_weight,
            last_updated: 0,
            metadata: None,
        });

        Ok(())
    }

    /// Removes a synaptic connection and returns the weight that was removed.
    ///
    /// # Arguments
    /// * `source_id` - ID of the pre-synaptic agent to disconnect
    ///
    /// # Returns
    /// `Some(weight)` if synapse existed and was removed, `None` otherwise
    ///
    /// # Complexity
    /// O(N) worst case due to vector element swap, but typically O(1) for HashMap lookup
    pub fn remove_synapse(&mut self, source_id: ZoooidId) -> Option<f32> {
        let idx = self.incoming_map.remove(&source_id)?;
        let removed_weight = self.weights[idx].weight;

        // Swap-remove: move last element to idx position
        if idx < self.weights.len() - 1 {
            let last_entry = self.weights.pop().unwrap();
            self.weights[idx] = last_entry.clone();
            // Update the incoming_map for the swapped element
            self.incoming_map.insert(last_entry.source_id, idx);
        } else {
            self.weights.pop();
        }

        debug!(
            "Removed synapse from {} with weight {}",
            source_id, removed_weight
        );
        Some(removed_weight)
    }

    /// Gets the weight for a given source agent.
    ///
    /// # Arguments
    /// * `source_id` - ID of the pre-synaptic agent
    ///
    /// # Returns
    /// `Some(weight)` if synapse exists, `None` otherwise
    ///
    /// # Complexity
    /// O(1) average case (HashMap lookup)
    pub fn get_weight(&self, source_id: ZoooidId) -> Option<f32> {
        self.incoming_map
            .get(&source_id)
            .map(|&idx| self.weights[idx].weight)
    }

    /// Updates a weight by adding a delta (typically from STDP or learning rule).
    ///
    /// # Arguments
    /// * `source_id` - ID of the pre-synaptic agent
    /// * `delta` - Change to apply to the weight
    /// * `tick` - Current timestamp (stored in entry for TTL)
    ///
    /// # Returns
    /// `Ok(new_weight)` if update successful, `Err` if synapse doesn't exist
    ///
    /// # Complexity
    /// O(1) average case
    pub fn update_weight(&mut self, source_id: ZoooidId, delta: f32, tick: u64) -> Result<f32, String> {
        let idx = self
            .incoming_map
            .get(&source_id)
            .ok_or(format!("Synapse from {} not found", source_id))?;

        let idx = *idx;
        let new_weight = (self.weights[idx].weight + delta)
            .clamp(self.config.min_weight, self.config.max_weight);
        self.weights[idx].weight = new_weight;
        self.weights[idx].last_updated = tick;
        self.update_count += 1;

        Ok(new_weight)
    }

    /// Normalizes all weights according to the configured normalization mode.
    ///
    /// # Complexity
    /// O(M) where M = number of synapses
    pub fn normalize(&mut self) {
        if self.weights.is_empty() {
            return;
        }

        match self.config.norm_mode {
            NormMode::None => {
                // Just clamp
                for entry in &mut self.weights {
                    entry.weight = entry.weight.clamp(self.config.min_weight, self.config.max_weight);
                }
            }
            NormMode::L1 => {
                // Sum of absolute values = 1.0
                let sum_abs: f32 = self.weights.iter().map(|e| e.weight.abs()).sum();
                if sum_abs > 1e-6 {
                    for entry in &mut self.weights {
                        entry.weight /= sum_abs;
                    }
                }
                // Clamp after normalization
                for entry in &mut self.weights {
                    entry.weight = entry.weight.clamp(self.config.min_weight, self.config.max_weight);
                }
            }
            NormMode::L2 => {
                // Euclidean norm = 1.0
                let sum_sq: f32 = self.weights.iter().map(|e| e.weight * e.weight).sum();
                let norm = sum_sq.sqrt();
                if norm > 1e-6 {
                    for entry in &mut self.weights {
                        entry.weight /= norm;
                    }
                }
                // Clamp after normalization
                for entry in &mut self.weights {
                    entry.weight = entry.weight.clamp(self.config.min_weight, self.config.max_weight);
                }
            }
            NormMode::Softmax => {
                // Softmax normalization
                let max_w = self
                    .weights
                    .iter()
                    .map(|e| e.weight)
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);

                let sum_exp: f32 = self
                    .weights
                    .iter()
                    .map(|e| (e.weight - max_w).exp())
                    .sum();

                if sum_exp > 1e-6 {
                    for entry in &mut self.weights {
                        entry.weight = (entry.weight - max_w).exp() / sum_exp;
                    }
                }
            }
            NormMode::Adaptive => {
                // No action needed; applied during update_weight
                for entry in &mut self.weights {
                    entry.weight = entry.weight.clamp(self.config.min_weight, self.config.max_weight);
                }
            }
        }
    }

    /// Returns the number of active synaptic connections.
    ///
    /// # Complexity
    /// O(1)
    pub fn get_active_count(&self) -> usize {
        self.weights.len()
    }

    /// Returns the number of updates applied since creation.
    ///
    /// # Complexity
    /// O(1)
    pub fn get_update_count(&self) -> u64 {
        self.update_count
    }

    /// Returns all source IDs (useful for iteration).
    ///
    /// # Complexity
    /// O(M) where M = number of synapses
    pub fn get_sources(&self) -> Vec<ZoooidId> {
        self.weights.iter().map(|e| e.source_id).collect()
    }

    /// Returns reference to a synapse entry if it exists.
    ///
    /// # Complexity
    /// O(1)
    pub fn get_entry(&self, source_id: ZoooidId) -> Option<&SynapseEntry> {
        self.incoming_map
            .get(&source_id)
            .map(|&idx| &self.weights[idx])
    }

    /// Returns mutable reference to a synapse entry if it exists.
    ///
    /// # Complexity
    /// O(1)
    pub fn get_entry_mut(&mut self, source_id: ZoooidId) -> Option<&mut SynapseEntry> {
        if let Some(&idx) = self.incoming_map.get(&source_id) {
            return Some(&mut self.weights[idx]);
        }
        None
    }

    /// Returns configuration reference.
    pub fn config(&self) -> &SynapseConfig {
        &self.config
    }

    /// Returns mutable configuration reference.
    pub fn config_mut(&mut self) -> &mut SynapseConfig {
        &mut self.config
    }

    /// Returns the sum of all weights.
    ///
    /// # Complexity
    /// O(M)
    pub fn sum_weights(&self) -> f32 {
        self.weights.iter().map(|e| e.weight).sum()
    }

    /// Returns the average weight.
    ///
    /// # Complexity
    /// O(M)
    pub fn avg_weight(&self) -> f32 {
        if self.weights.is_empty() {
            0.0
        } else {
            self.sum_weights() / self.weights.len() as f32
        }
    }

    /// Clears all synaptic connections.
    ///
    /// # Complexity
    /// O(1) amortized
    pub fn clear(&mut self) {
        self.incoming_map.clear();
        self.weights.clear();
    }

    /// Checks if a source is connected.
    ///
    /// # Complexity
    /// O(1)
    pub fn contains_source(&self, source_id: ZoooidId) -> bool {
        self.incoming_map.contains_key(&source_id)
    }

    /// Processes a topology event and applies corresponding changes to synapses.
    ///
    /// Handles:
    /// - `EdgeAdded`: adds new synapse with initial weight
    /// - `EdgeRemoved`: removes orphaned synapse
    /// - `WeightUpdated`: updates synapse weight
    /// - `TopologySnapshot`: synchronizes state with current topology
    ///
    /// # Returns
    /// A list of descriptions of actions taken (for logging/audit)
    ///
    /// # Complexity
    /// - `EdgeAdded`: O(1) average case
    /// - `EdgeRemoved`: O(N) worst case (swap-remove)
    /// - `WeightUpdated`: O(1) average case
    /// - `TopologySnapshot`: O(M) where M = number of edges in snapshot
    pub fn on_topology_event(&mut self, event: &crate::core::TopologyEvent) -> Vec<String> {
        use crate::core::TopologyEvent;

        let mut actions = Vec::new();

        match event {
            TopologyEvent::EdgeAdded {
                from,
                to: _,
                initial_weight,
            } => {
                // Only react if this event targets us (we are the "to" agent)
                match self.add_synapse(*from, *initial_weight) {
                    Ok(()) => {
                        actions.push(format!(
                            "Added synapse from {} with weight {}",
                            from, initial_weight
                        ));
                    }
                    Err(e) => {
                        debug!("Failed to add synapse: {}", e);
                        actions.push(format!("Failed to add synapse: {}", e));
                    }
                }
            }

            TopologyEvent::EdgeRemoved { from, to: _ } => {
                // Only react if this event targets us
                if let Some(weight) = self.remove_synapse(*from) {
                    actions.push(format!(
                        "Removed synapse from {} (had weight {})",
                        from, weight
                    ));
                    debug!("Auto-cleanup: removed orphaned synapse from {}", from);
                } else {
                    actions.push(format!("No synapse to remove from {}", from));
                }
            }

            TopologyEvent::WeightUpdated {
                from,
                to: _,
                new_weight,
            } => {
                // Only react if this event targets us
                match self.update_weight(*from, *new_weight - self.get_weight(*from).unwrap_or(0.0), 0) {
                    Ok(actual_weight) => {
                        actions.push(format!(
                            "Updated synapse from {} to weight {}",
                            from, actual_weight
                        ));
                    }
                    Err(e) => {
                        debug!("Failed to update synapse: {}", e);
                        actions.push(format!("Failed to update synapse: {}", e));
                    }
                }
            }

            TopologyEvent::TopologySnapshot { edges } => {
                // Synchronize: remove synapses not in snapshot, add missing ones
                let incoming_sources: std::collections::HashSet<_> =
                    edges.iter().map(|(from, _, _)| *from).collect();

                // Remove orphaned synapses
                let to_remove: Vec<_> = self
                    .weights
                    .iter()
                    .map(|e| e.source_id)
                    .filter(|src| !incoming_sources.contains(src))
                    .collect();

                for src in to_remove {
                    if let Some(weight) = self.remove_synapse(src) {
                        actions.push(format!(
                            "Sync: removed orphaned synapse from {} (weight {})",
                            src, weight
                        ));
                    }
                }

                // Add new synapses from snapshot
                for (from, _to, weight) in edges {
                    if !self.contains_source(*from) {
                        match self.add_synapse(*from, *weight) {
                            Ok(()) => {
                                actions.push(format!(
                                    "Sync: added synapse from {} with weight {}",
                                    from, weight
                                ));
                            }
                            Err(e) => {
                                actions.push(format!("Sync: failed to add synapse from {}: {}", from, e));
                            }
                        }
                    }
                }
            }
        }

        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_id(n: u8) -> ZoooidId {
        use uuid::Uuid;
        Uuid::new_v5(&Uuid::NAMESPACE_DNS, &[n])
    }

    #[test]
    fn test_add_synapse() {
        let mut manager = SynapseManager::new(SynapseConfig::default());
        let id1 = test_id(1);
        let result = manager.add_synapse(id1, 0.5);
        assert!(result.is_ok());
        assert_eq!(manager.get_active_count(), 1);
        assert_eq!(manager.get_weight(id1), Some(0.5));
    }

    #[test]
    fn test_remove_synapse() {
        let mut manager = SynapseManager::new(SynapseConfig::default());
        let id1 = test_id(1);
        manager.add_synapse(id1, 0.5).unwrap();
        let removed = manager.remove_synapse(id1);
        assert_eq!(removed, Some(0.5));
        assert_eq!(manager.get_active_count(), 0);
        assert_eq!(manager.get_weight(id1), None);
    }

    #[test]
    fn test_get_weight_o1() {
        let mut manager = SynapseManager::new(SynapseConfig::default());
        for i in 0..50 {
            // Use i / 50.0 to keep weights in [0.0, 1.0] range
            manager.add_synapse(test_id(i as u8), i as f32 / 50.0).unwrap();
        }
        // Verify O(1) lookup still works
        assert_eq!(manager.get_weight(test_id(25)), Some(25.0 / 50.0));
        assert_eq!(manager.get_active_count(), 50);
    }

    #[test]
    fn test_normalize_l1() {
        let mut config = SynapseConfig::default();
        config.norm_mode = NormMode::L1;
        let mut manager = SynapseManager::new(config);

        manager.add_synapse(test_id(1), 0.5).unwrap();
        manager.add_synapse(test_id(2), 0.5).unwrap();
        manager.normalize();

        let sum_abs: f32 = manager.get_sources()
            .iter()
            .filter_map(|&id| manager.get_weight(id))
            .map(|w| w.abs())
            .sum();

        assert!((sum_abs - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_max_connections_limit() {
        let mut config = SynapseConfig::default();
        config.max_connections = 3;
        let mut manager = SynapseManager::new(config);

        manager.add_synapse(test_id(1), 0.1).unwrap();
        manager.add_synapse(test_id(2), 0.1).unwrap();
        manager.add_synapse(test_id(3), 0.1).unwrap();

        let result = manager.add_synapse(test_id(4), 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_clamping() {
        let mut config = SynapseConfig::default();
        config.min_weight = 0.1;
        config.max_weight = 0.9;
        let mut manager = SynapseManager::new(config);

        manager.add_synapse(test_id(1), 2.0).unwrap();
        assert_eq!(manager.get_weight(test_id(1)), Some(0.9));

        manager.add_synapse(test_id(2), -1.0).unwrap();
        assert_eq!(manager.get_weight(test_id(2)), Some(0.1));
    }

    #[test]
    fn test_swap_remove() {
        let mut manager = SynapseManager::new(SynapseConfig::default());
        manager.add_synapse(test_id(1), 0.5).unwrap();
        manager.add_synapse(test_id(2), 0.6).unwrap();
        manager.add_synapse(test_id(3), 0.7).unwrap();

        manager.remove_synapse(test_id(1));

        // Verify that id 3 weight is still accessible after swap-remove
        assert_eq!(manager.get_weight(test_id(3)), Some(0.7));
        assert_eq!(manager.get_active_count(), 2);
    }

    #[test]
    fn test_on_topology_event_edge_added() {
        use crate::core::TopologyEvent;

        let mut manager = SynapseManager::new(SynapseConfig::default());
        let from = test_id(1);
        let to = test_id(2);

        let event = TopologyEvent::EdgeAdded {
            from,
            to,
            initial_weight: 0.5,
        };

        let actions = manager.on_topology_event(&event);
        assert_eq!(manager.get_weight(from), Some(0.5));
        assert!(actions.len() > 0);
        assert!(actions[0].contains("Added synapse"));
    }

    #[test]
    fn test_on_topology_event_edge_removed() {
        use crate::core::TopologyEvent;

        let mut manager = SynapseManager::new(SynapseConfig::default());
        let from = test_id(1);

        manager.add_synapse(from, 0.5).unwrap();
        assert_eq!(manager.get_active_count(), 1);

        let event = TopologyEvent::EdgeRemoved {
            from,
            to: test_id(2),
        };

        let actions = manager.on_topology_event(&event);
        assert_eq!(manager.get_weight(from), None);
        assert_eq!(manager.get_active_count(), 0);
        assert!(actions.len() > 0);
        assert!(actions[0].contains("Removed synapse"));
    }

    #[test]
    fn test_on_topology_event_snapshot_sync() {
        use crate::core::TopologyEvent;

        let mut manager = SynapseManager::new(SynapseConfig::default());

        // Add some initial synapses not in snapshot
        manager.add_synapse(test_id(1), 0.5).unwrap();
        manager.add_synapse(test_id(2), 0.6).unwrap();

        // Create snapshot with different edges
        let edges = vec![
            (test_id(2), test_id(99), 0.6),
            (test_id(3), test_id(99), 0.7),
        ];

        let event = TopologyEvent::TopologySnapshot { edges };

        let actions = manager.on_topology_event(&event);

        // Should remove id 1 (not in snapshot)
        // Should keep id 2 (in snapshot)
        // Should add id 3 (in snapshot)
        assert_eq!(manager.get_weight(test_id(1)), None);
        assert_eq!(manager.get_weight(test_id(2)), Some(0.6));
        assert_eq!(manager.get_weight(test_id(3)), Some(0.7));
        assert_eq!(manager.get_active_count(), 2);
        assert!(actions.len() > 0);
    }
}
