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

//! Cognitive Memory Hybrid Architecture
//!
//! This hybrid demonstrates short-term memory capabilities using:
//! - A network of CognitiveZooid agents (Izhikevich neurons)
//! - Recurrent connections creating feedback loops
//! - Persistent oscillations encoding memory
//!
//! A brief stimulus creates a pattern of activity that persists and gradually decays,
//! demonstrating working memory principles.

use std::sync::Arc;
use tokio::sync::Mutex;

use crate::agent::blueprint::{CognitivezooidBlueprint, CognitivezooidParams};
use crate::agent::zoooid::Zoooid;
use crate::core::CoreScheduler;
use crate::primitives::spiking::izhikevich::IzhikevichParams;
use crate::utils::types::ZoooidId;

/// Activity snapshot of a neuron
#[derive(Debug, Clone, Copy)]
pub struct NeuronActivity {
    pub neuron_id: ZoooidId,
    pub spike_count: u64,
    pub last_spike_time: Option<u64>,
    pub membrane_potential: f64,
}

/// Configuration for the Cognitive Memory architecture
#[derive(Debug, Clone)]
pub struct CognitiveMemoryConfig {
    pub network_size: usize, // Number of neurons
    pub izh_params: IzhikevichParams,
    pub connection_density: f64, // 0.0 to 1.0, fraction of possible connections
    pub memory_trace_depth: usize, // How long to keep activity history
}

impl Default for CognitiveMemoryConfig {
    fn default() -> Self {
        Self {
            network_size: 5,
            izh_params: IzhikevichParams::default(),
            connection_density: 0.6,
            memory_trace_depth: 100,
        }
    }
}

/// Activity history entry
#[derive(Debug, Clone)]
pub struct ActivityTrace {
    pub timestamp: u64,
    pub activities: Vec<NeuronActivity>,
}

/// Cognitive Memory Architecture Instance
#[derive(Debug, Clone)]
pub struct CognitiveMemoryArchitecture {
    pub neuron_ids: Vec<ZoooidId>,
    pub activity_history: Arc<Mutex<Vec<ActivityTrace>>>,
    pub memory_trace_depth: usize,
    pub timestamp: Arc<Mutex<u64>>,
}

impl CognitiveMemoryArchitecture {
    /// Create and register a network of cognitive agents (neurons)
    pub async fn create(
        scheduler: &mut CoreScheduler,
        config: CognitiveMemoryConfig,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut neuron_ids = Vec::new();

        // Create neuron agents
        for _ in 0..config.network_size {
            let blueprint = CognitivezooidBlueprint {
                params: CognitivezooidParams {
                    izh_params: config.izh_params.clone(),
                    connection_request_interval: 10,
                    stdp_a_plus: 0.01,
                    stdp_a_minus: 0.005,
                    stdp_tau_plus: 20.0,
                    stdp_tau_minus: 20.0,
                    weight_decay: 0.0001,
                    pruning_threshold: 0.001,
                    enable_simd_batch: false,
                },
            };

            let zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(blueprint)).await?;
            let id = zooid.id;
            neuron_ids.push(id);

            scheduler.spawn_agent(zooid).await?;
        }

        // Create recurrent connections based on connection_density
        let mut topology = scheduler.topology.write().await;
        for i in 0..neuron_ids.len() {
            for j in 0..neuron_ids.len() {
                if i != j {
                    // Probabilistic connection based on density
                    let should_connect = ((i * neuron_ids.len() + j) as f64
                        / (neuron_ids.len() * neuron_ids.len()) as f64)
                        < config.connection_density;

                    if should_connect {
                        topology.add_edge(
                            neuron_ids[i],
                            neuron_ids[j],
                            crate::core::topology::ConnectionProperties::default(),
                        );
                    }
                }
            }
        }
        drop(topology);

        Ok(CognitiveMemoryArchitecture {
            neuron_ids,
            activity_history: Arc::new(Mutex::new(Vec::new())),
            memory_trace_depth: config.memory_trace_depth,
            timestamp: Arc::new(Mutex::new(0)),
        })
    }

    /// Record neural activity snapshot
    pub async fn record_activity(&self, activities: Vec<NeuronActivity>) {
        let mut history = self.activity_history.lock().await;
        let mut ts = self.timestamp.lock().await;

        let trace = ActivityTrace {
            timestamp: *ts,
            activities,
        };

        history.push(trace);

        // Maintain memory depth limit
        if history.len() > self.memory_trace_depth {
            history.remove(0);
        }

        *ts += 1;
    }

    /// Get recent activity history
    pub async fn get_activity_history(&self, last_n: usize) -> Vec<ActivityTrace> {
        let history = self.activity_history.lock().await;
        let start = if history.len() > last_n {
            history.len() - last_n
        } else {
            0
        };
        history[start..].to_vec()
    }

    /// Calculate network activity level (0.0 to 1.0)
    pub async fn get_network_activity_level(&self) -> f64 {
        let history = self.activity_history.lock().await;
        if history.is_empty() {
            return 0.0;
        }

        let recent = &history[history.len().saturating_sub(10)..];
        let total_spikes: u64 = recent
            .iter()
            .flat_map(|trace| trace.activities.iter())
            .map(|activity| activity.spike_count)
            .sum();

        let max_possible_spikes = (recent.len() * self.neuron_ids.len()) as u64 * 10; // Rough estimate
        if max_possible_spikes == 0 {
            0.0
        } else {
            (total_spikes as f64 / max_possible_spikes as f64).min(1.0)
        }
    }

    /// Get network connectivity statistics
    pub async fn get_connectivity_stats(&self, scheduler: &CoreScheduler) -> (usize, usize) {
        let topology = scheduler.topology.read().await;
        let mut total_connections = 0;

        for &neuron_id in &self.neuron_ids {
            let connections = topology.get_neighbors(neuron_id);
            total_connections += connections.len();
        }

        let avg_connections = total_connections / self.neuron_ids.len().max(1);
        (total_connections, avg_connections)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{LocalBus, ResourceManager, ZoooidTopology};
    use crate::RuleEngine;
    use std::sync::Arc;
    use tokio::sync::{Mutex, RwLock};

    #[tokio::test]
    async fn test_cognitive_memory_creation() {
        let topology = Arc::new(RwLock::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = CognitiveMemoryConfig {
            network_size: 5,
            ..Default::default()
        };

        let memory = CognitiveMemoryArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create cognitive memory");

        assert_eq!(memory.neuron_ids.len(), 5);
        assert_eq!(scheduler.agent_count(), 5);
    }

    #[tokio::test]
    async fn test_activity_recording() {
        let topology = Arc::new(RwLock::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = CognitiveMemoryConfig::default();
        let memory = CognitiveMemoryArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create cognitive memory");

        let activities = memory
            .neuron_ids
            .iter()
            .map(|&id| NeuronActivity {
                neuron_id: id,
                spike_count: 1,
                last_spike_time: Some(0),
                membrane_potential: -65.0,
            })
            .collect();

        memory.record_activity(activities).await;

        let history = memory.get_activity_history(10).await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].activities.len(), memory.neuron_ids.len());
    }

    #[tokio::test]
    async fn test_activity_level_calculation() {
        let topology = Arc::new(RwLock::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = CognitiveMemoryConfig::default();
        let memory = CognitiveMemoryArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create cognitive memory");

        // Initial activity should be 0
        let initial = memory.get_network_activity_level().await;
        assert_eq!(initial, 0.0);

        // Record some activity
        for _ in 0..5 {
            let activities = std::iter::repeat_with(|| {
                vec![NeuronActivity {
                    neuron_id: memory.neuron_ids[0],
                    spike_count: 1,
                    last_spike_time: Some(0),
                    membrane_potential: -65.0,
                }]
            })
            .take(1)
            .flatten()
            .collect();

            memory.record_activity(activities).await;
        }

        let activity = memory.get_network_activity_level().await;
        assert!((0.0..=1.0).contains(&activity));
    }

    #[tokio::test]
    async fn test_network_recurrency() {
        let topology = Arc::new(RwLock::new(ZoooidTopology::new()));
        let rule_engine = RuleEngine::new();
        let resource_manager = ResourceManager::new();
        let message_bus = Arc::new(LocalBus::new());

        let mut scheduler =
            CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

        let config = CognitiveMemoryConfig {
            network_size: 5,
            connection_density: 0.5,
            ..Default::default()
        };

        let memory = CognitiveMemoryArchitecture::create(&mut scheduler, config)
            .await
            .expect("Failed to create cognitive memory");

        let (total_connections, _) = memory.get_connectivity_stats(&scheduler).await;
        assert!(total_connections > 0);
    }
}
