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

use std::cmp::max;
use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use adaptiflux_core::attention::{DotProductAttention, ErrorSimilarityFocus};
use adaptiflux_core::core::message_bus::Message;
use adaptiflux_core::learning::GradientDescentLearner;
use adaptiflux_core::memory::long_term_store::TableLongTermStore;
use adaptiflux_core::CoreScheduler;
use adaptiflux_core::LocalBus;
use adaptiflux_core::MemoryAttentionHook;
use adaptiflux_core::MetadataIndexer;
use adaptiflux_core::OnlineAdaptationEngine;
use adaptiflux_core::OnlineAdaptationHook;
use adaptiflux_core::PowerMonitor;
use adaptiflux_core::ResourceManager;
use adaptiflux_core::Retriever;
use adaptiflux_core::RuleEngine;
use adaptiflux_core::ZoooidId;
use adaptiflux_core::ZoooidTopology;
use sysinfo::{Pid, ProcessesToUpdate, System};

use crate::data_loader::{encode_image_to_sensors, MnistDataset};
use crate::mnist_spiking_classifier::{
    build_mnist_architecture, MnistSpikingClassifierArchitecture, OutputExperienceRecorder,
    SpikingGainLearner,
};

pub async fn run_training(
    dataset: &MnistDataset,
    max_duration: Duration,
) -> Result<(CoreScheduler, MnistSpikingClassifierArchitecture), Box<dyn Error + Send + Sync>> {
    let bus = Arc::new(LocalBus::new());
    let topology = Arc::new(tokio::sync::Mutex::new(ZoooidTopology::new()));
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    let mut scheduler = CoreScheduler::new(topology, rule_engine, resource_manager, bus.clone());
    scheduler.set_cycle_frequency(40);
    scheduler.enable_power_monitor(PowerMonitor::default());

    let architecture = build_mnist_architecture(&mut scheduler).await?;

    let mut adaptation_engine = OnlineAdaptationEngine::new();
    adaptation_engine.set_default_learner(Arc::new(GradientDescentLearner {
        learning_rate: 0.008,
    }));
    for target in architecture.all_adaptation_targets() {
        adaptation_engine.register(
            target,
            Arc::new(SpikingGainLearner {
                learning_rate: 0.02,
            }),
        );
    }
    scheduler.online_adaptation = Some(OnlineAdaptationHook {
        engine: adaptation_engine,
        target_ids: architecture.all_adaptation_targets(),
    });

    let memory_store = Arc::new(tokio::sync::Mutex::new(TableLongTermStore::new()));
    let memory_indexer = Arc::new(tokio::sync::Mutex::new(MetadataIndexer::new()));
    let memory_attention = MemoryAttentionHook {
        store: memory_store.clone(),
        indexer: memory_indexer.clone(),
        retriever: Retriever::new(8),
        attention: Arc::new(DotProductAttention::default()),
        focus: Arc::new(ErrorSimilarityFocus {
            tag: Some("mnist_spiking".into()),
            min_similarity: 0.0,
            observation_fn: observation_from_inputs,
        }),
        target_ids: Some(architecture.output_ids.clone()),
        inject_memory_into_feedback: true,
        memory_feedback_gain: 0.06,
        experience: Some(Arc::new(OutputExperienceRecorder {
            output_ids: architecture.output_ids.clone(),
        })),
    };
    scheduler.memory_attention = Some(memory_attention);

    let mut system = System::new_all();
    let pid = Pid::from(std::process::id() as usize);
    let mut max_rss_mb = 0.0;
    let mut correct = 0;
    let mut samples = 0;
    let start = Instant::now();
    let mut epoch = 0;

    while start.elapsed() < max_duration {
        epoch += 1;
        tracing::info!(epoch, "starting epoch");

        for (image, label) in dataset.train_images.iter().zip(dataset.train_labels.iter()) {
            if start.elapsed() >= max_duration {
                break;
            }

            architecture.reset_output_counts(&mut scheduler);
            encode_image_to_sensors(&scheduler.message_bus, &architecture.sensor_ids, image)
                .await?;
            scheduler.run_for_iterations(3).await?;

            let prediction = architecture.decode_output(&scheduler);
            let error_value = if prediction == *label { 0.0 } else { 1.0 };
            if prediction == *label {
                correct += 1;
            }
            samples += 1;

            let monitor_id = ZoooidId::new_v4();
            for target in architecture.all_adaptation_targets() {
                scheduler
                    .message_bus
                    .send(monitor_id, target, Message::Error(error_value))
                    .await
                    .map_err(|e| format!("Failed to post error feedback: {:?}", e))?;
            }

            scheduler.run_one_iteration().await?;

            if samples % 100 == 0 {
                system.refresh_processes(ProcessesToUpdate::All, true);
                let rss_mb = if let Some(process) = system.process(pid) {
                    let value = process.memory() as f64 / 1024.0 / 1024.0;
                    max_rss_mb = max(max_rss_mb as usize, value as usize) as f64;
                    value
                } else {
                    0.0
                };
                let accuracy = correct as f64 / samples as f64;
                let active_agents = scheduler.agent_count();
                tracing::info!(
                    epoch,
                    step = samples,
                    error = error_value,
                    ram_mb = rss_mb,
                    active_agents,
                    accuracy_window = accuracy,
                    "📊 Checkpoint"
                );
                if rss_mb > 3800.0 {
                    tracing::warn!(
                        "⚠️ RAM > 3.8GB, skipping sparse cleanup because the method is unavailable"
                    );
                }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    tracing::info!(
        epoch,
        samples,
        duration_s = elapsed,
        final_accuracy = correct as f64 / samples as f64,
        peak_memory_mb = max_rss_mb,
        "completed timed training"
    );

    Ok((scheduler, architecture))
}

fn observation_from_inputs(inputs: &[(ZoooidId, Message)]) -> f32 {
    inputs
        .iter()
        .filter_map(|(_sender, msg)| match msg {
            Message::SpikeEvent { amplitude, .. } => Some(*amplitude),
            _ => None,
        })
        .sum()
}
