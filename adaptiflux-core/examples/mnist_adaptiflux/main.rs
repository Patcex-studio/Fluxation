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

use adaptiflux_core::agent::blueprint::cognitivezooid::CognitivezooidBlueprint;
use adaptiflux_core::agent::blueprint::sensorzooid::SensorzooidBlueprint;
use adaptiflux_core::agent::zoooid::Zoooid;
use adaptiflux_core::core::message_bus::LocalBus;
use adaptiflux_core::core::resource_manager::ResourceManager;
use adaptiflux_core::core::scheduler::{CoreScheduler, OnlineAdaptationHook};
use adaptiflux_core::core::topology::ZoooidTopology;
use adaptiflux_core::learning::online_adaptation::{
    GradientDescentLearner, OnlineAdaptationEngine,
};
use adaptiflux_core::performance::async_optimization::AsyncOptimizationConfig;
use adaptiflux_core::performance::sparse_execution::SparseExecutionHook;
use adaptiflux_core::primitives::spiking::izhikevich::IzhikevichParams;
use adaptiflux_core::primitives::spiking::lif::LifParams;
use adaptiflux_core::rules::RuleEngine;
use adaptiflux_core::utils::types::new_zoooid_id;
use mnist::{Mnist, MnistBuilder};
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, Level};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("Starting MNIST Adaptiflux Demo");

    // Load MNIST data
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(2_000)
        .validation_set_length(500)
        .test_set_length(500)
        .finalize();

    let train_images: Vec<Vec<u8>> = trn_img.chunks(784).map(|chunk| chunk.to_vec()).collect();
    let train_labels: Vec<u8> = trn_lbl;
    let test_images: Vec<Vec<u8>> = tst_img.chunks(784).map(|chunk| chunk.to_vec()).collect();
    let test_labels: Vec<u8> = tst_lbl;

    // Create scheduler
    let topology = Arc::new(tokio::sync::Mutex::new(ZoooidTopology::new()));
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager {
        cpu_pool: 4,
        gpu_pool: 0,
    };
    let message_bus = Arc::new(LocalBus::new());

    let mut scheduler = CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);

    // Enable optimizations
    scheduler.enable_async_optimization(AsyncOptimizationConfig::new(4));
    scheduler.enable_sparse_execution(SparseExecutionHook::new(Duration::from_millis(100)));

    // Create hybrid network
    let (input_ids, output_ids) = build_mnist_network(&mut scheduler).await?;

    // Enable online adaptation with async optimization
    let mut adaptation_engine = OnlineAdaptationEngine::new();
    #[cfg(feature = "adaptiflux_optim")]
    adaptation_engine.enable_async_optimization(0.01, 10, 50); // lr, batch_size, interval

    let learner = Arc::new(GradientDescentLearner::default());
    adaptation_engine.set_default_learner(learner);

    let adaptation_hook = OnlineAdaptationHook {
        engine: adaptation_engine,
        target_ids: output_ids.clone(),
    };
    scheduler.online_adaptation = Some(adaptation_hook);

    #[cfg(feature = "adaptiflux_optim")]
    scheduler.enable_async_adaptation(0.01, 10, 50);
    // Training loop
    let epochs = 2;
    let batch_size = 16;

    for epoch in 0..epochs {
        info!("Epoch {}", epoch);
        let mut total_loss = 0.0;
        let mut correct = 0;

        for batch_start in (0..train_images.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_images.len());
            let batch_images = &train_images[batch_start..batch_end];
            let batch_labels = &train_labels[batch_start..batch_end];

            for (img, &label) in batch_images.iter().zip(batch_labels.iter()) {
                feed_image_to_input(&mut scheduler, &input_ids, img).await?;

                scheduler.run_for_iterations(10).await?;

                let prediction = get_output_prediction(&mut scheduler, &output_ids).await?;
                let loss = compute_loss(prediction, label);
                total_loss += loss;

                if prediction == label as usize {
                    correct += 1;
                }

                send_feedback(&mut scheduler, &output_ids, prediction, label).await?;
            }
        }

        let accuracy = correct as f32 / train_images.len() as f32;
        info!(
            "Epoch {}: Loss {:.4}, Accuracy {:.4}",
            epoch,
            total_loss / train_images.len() as f32,
            accuracy
        );
    }

    let eval_samples = test_images.len().min(50);
    let mut eval_correct = 0;
    for (img, &label) in test_images
        .iter()
        .zip(test_labels.iter())
        .take(eval_samples)
    {
        feed_image_to_input(&mut scheduler, &input_ids, img).await?;
        scheduler.run_for_iterations(10).await?;
        let prediction = get_output_prediction(&mut scheduler, &output_ids).await?;
        if prediction == label as usize {
            eval_correct += 1;
        }
    }

    info!(
        "Validation accuracy on {} samples: {:.4}",
        eval_samples,
        eval_correct as f32 / eval_samples as f32
    );

    info!("Training complete");
    Ok(())
}

async fn build_mnist_network(
    scheduler: &mut CoreScheduler,
) -> Result<
    (
        Vec<adaptiflux_core::utils::types::ZoooidId>,
        Vec<adaptiflux_core::utils::types::ZoooidId>,
    ),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let mut input_ids = Vec::new();
    let mut hidden_ids = Vec::new();
    let mut output_ids = Vec::new();

    for _ in 0..784 {
        let id = new_zoooid_id();
        input_ids.push(id);
        let blueprint = SensorzooidBlueprint {
            params: adaptiflux_core::agent::blueprint::sensorzooid::SensorzooidParams {
                lif_params: LifParams {
                    tau_m: 20.0,
                    v_rest: -65.0,
                    v_thresh: -50.0,
                    v_reset: -65.0,
                    r_m: 1.0,
                    dt: 0.1,
                },
                connection_request_interval: 10,
            },
        };
        let agent = Zoooid::new(id, Box::new(blueprint)).await?;
        scheduler.spawn_agent(agent).await?;
    }

    for _ in 0..64 {
        let id = new_zoooid_id();
        hidden_ids.push(id);
        let blueprint = CognitivezooidBlueprint {
            params: adaptiflux_core::agent::blueprint::cognitivezooid::CognitivezooidParams {
                izh_params: IzhikevichParams {
                    a: 0.02,
                    b: 0.2,
                    c: -65.0,
                    d: 8.0,
                    dt: 0.1,
                },
                connection_request_interval: 10,
            },
        };
        let agent = Zoooid::new(id, Box::new(blueprint)).await?;
        scheduler.spawn_agent(agent).await?;
    }

    for _ in 0..10 {
        let id = new_zoooid_id();
        output_ids.push(id);
        let blueprint = CognitivezooidBlueprint {
            params: adaptiflux_core::agent::blueprint::cognitivezooid::CognitivezooidParams {
                izh_params: IzhikevichParams {
                    a: 0.02,
                    b: 0.2,
                    c: -65.0,
                    d: 8.0,
                    dt: 0.1,
                },
                connection_request_interval: 10,
            },
        };
        let agent = Zoooid::new(id, Box::new(blueprint)).await?;
        scheduler.spawn_agent(agent).await?;
    }

    for &src in &input_ids {
        for &dst in &hidden_ids {
            scheduler
                .topology
                .lock()
                .await
                .add_edge(src, dst, Default::default());
        }
    }

    for &src in &hidden_ids {
        for &dst in &output_ids {
            scheduler
                .topology
                .lock()
                .await
                .add_edge(src, dst, Default::default());
        }
    }

    Ok((input_ids, output_ids))
}

async fn feed_image_to_input(
    scheduler: &mut CoreScheduler,
    input_ids: &[adaptiflux_core::utils::types::ZoooidId],
    image: &[u8],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    for (i, &agent_id) in input_ids.iter().enumerate() {
        if let Some(&pixel) = image.get(i) {
            let current = pixel as f32 / 255.0 * 10.0;
            let message =
                adaptiflux_core::core::message_bus::message::Message::AnalogInput(current);
            scheduler
                .message_bus
                .send(agent_id, agent_id, message)
                .await?;
        }
    }
    Ok(())
}

async fn get_output_prediction(
    scheduler: &mut CoreScheduler,
    output_ids: &[adaptiflux_core::utils::types::ZoooidId],
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let mut max_spikes = 0;
    let mut prediction = 0;

    for (idx, &agent_id) in output_ids.iter().enumerate() {
        if let Some(handle) = scheduler.agents.get(&agent_id) {
            if let Some(cog) = handle.state.downcast_ref::<adaptiflux_core::agent::blueprint::cognitivezooid::CognitivezooidState>() {
                let spikes = cog.spike_count as usize;
                if spikes > max_spikes {
                    max_spikes = spikes;
                    prediction = idx;
                }
            }
        }
    }

    // Reset output spike counters for the next sample
    for &agent_id in output_ids {
        if let Some(handle) = scheduler.agents.get_mut(&agent_id) {
            if let Some(cog) = handle.state.downcast_mut::<adaptiflux_core::agent::blueprint::cognitivezooid::CognitivezooidState>() {
                cog.spike_count = 0;
            }
        }
    }

    Ok(prediction)
}

async fn send_feedback(
    scheduler: &mut CoreScheduler,
    output_ids: &[adaptiflux_core::utils::types::ZoooidId],
    prediction: usize,
    label: u8,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let error = if prediction == label as usize {
        0.0
    } else {
        1.0
    };
    let msg = adaptiflux_core::core::message_bus::message::Message::Error(error);

    for &agent_id in output_ids {
        scheduler
            .message_bus
            .send(agent_id, agent_id, msg.clone())
            .await?;
    }
    Ok(())
}

fn compute_loss(prediction: usize, label: u8) -> f32 {
    if prediction == label as usize {
        0.0
    } else {
        1.0
    }
}
