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

use std::error::Error;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use adaptiflux_core::attention::{DotProductAttention, ErrorSimilarityFocus};
use adaptiflux_core::core::message_bus::Message;
use adaptiflux_core::learning::GradientDescentLearner;
use adaptiflux_core::rules::behavior::{IsolationRecoveryRule, LoadBalancingRule};
use adaptiflux_core::rules::consistency::{ConnectedTopologyCheck, MinConnectivityCheck};
use adaptiflux_core::rules::structural_plasticity::{
    ActivityDependentSynaptogenesisRule, ClusterGroupingPlasticityRule, SynapticPruningRule,
};
use adaptiflux_core::rules::topology::ProximityConnectionRule;
use adaptiflux_core::AbstractionLayerManager;
use adaptiflux_core::AggregationFnKind;
use adaptiflux_core::AsyncOptimizationConfig;
use adaptiflux_core::CoreScheduler;
use adaptiflux_core::HierarchyHook;
use adaptiflux_core::LocalBus;
use adaptiflux_core::MemoryAttentionHook;
use adaptiflux_core::MetadataIndexer;
use adaptiflux_core::OnlineAdaptationEngine;
use adaptiflux_core::OnlineAdaptationHook;
use adaptiflux_core::PowerMonitor;
use adaptiflux_core::ResourceManager;
use adaptiflux_core::RuleEngine;
use adaptiflux_core::SleepScheduler;
use adaptiflux_core::SparseExecutionHook;
use adaptiflux_core::ZoooidId;
use adaptiflux_core::ZoooidTopology;
use serde::Serialize;
use sysinfo::{Pid, ProcessesToUpdate, System};
use tracing::Level;

mod dataset {
    use adaptiflux_core::core::message_bus::{Message, MessageBus};
    use adaptiflux_core::utils::types::ZoooidId;
    use image::DynamicImage;
    use image::ImageError;
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::RowAccessor;
    use rayon::prelude::*;
    use std::error::Error;
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    pub struct MnistDataset {
        pub train_images: Vec<Vec<f32>>,
        pub train_labels: Vec<u8>,
        pub test_images: Vec<Vec<f32>>,
        pub test_labels: Vec<u8>,
    }

    pub async fn load_mnist(
        base_path: &str,
        max_train: usize,
        max_test: usize,
    ) -> Result<MnistDataset, Box<dyn Error + Send + Sync>> {
        let train_path = Path::new(base_path).join("train-00000-of-00001(1).parquet");
        let test_path = Path::new(base_path).join("test-00000-of-00001(1).parquet");

        let train_images = load_images_from_parquet(&train_path, max_train)?;
        let train_labels = load_labels_from_parquet(&train_path, max_train)?;
        let test_images = load_images_from_parquet(&test_path, max_test)?;
        let test_labels = load_labels_from_parquet(&test_path, max_test)?;

        Ok(MnistDataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    fn load_images_from_parquet(
        path: &Path,
        max_count: usize,
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>> {
        let file = File::open(path)?;
        let reader = SerializedFileReader::new(file)?;
        let iter = reader.get_row_iter(None)?;

        let rows: Vec<_> = iter.take(max_count).collect();
        let images: Vec<Vec<f32>> = rows
            .into_par_iter()
            .map(|row| -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
                let image_group = row.get_group(0)?;
                let bytes = image_group.get_bytes(0)?;
                let img = image::load_from_memory(bytes.data())?;
                let pixels = extract_pixels(&img)?;
                Ok(normalize_and_downsample(&pixels))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(images)
    }

    fn load_labels_from_parquet(
        path: &Path,
        max_count: usize,
    ) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
        let file = File::open(path)?;
        let reader = SerializedFileReader::new(file)?;
        let iter = reader.get_row_iter(None)?;

        let rows: Vec<_> = iter.take(max_count).collect();
        let labels: Vec<u8> = rows
            .into_par_iter()
            .map(|row| -> Result<u8, Box<dyn Error + Send + Sync>> {
                let label: i64 = row.get_long(1)?;
                Ok(label as u8)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(labels)
    }

    fn extract_pixels(img: &DynamicImage) -> Result<Vec<u8>, ImageError> {
        let gray_img = img.to_luma8();
        Ok(gray_img.into_raw())
    }

    fn normalize_and_downsample(image: &[u8]) -> Vec<f32> {
        const OUT_SIZE: usize = 14;
        const IN_SIZE: usize = 28;
        let mut result = Vec::with_capacity(OUT_SIZE * OUT_SIZE);

        for block_row in 0..OUT_SIZE {
            for block_col in 0..OUT_SIZE {
                let mut sum = 0_u32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let row = block_row * 2 + dy;
                        let col = block_col * 2 + dx;
                        let idx = row * IN_SIZE + col;
                        sum += image[idx] as u32;
                    }
                }
                let avg = sum as f32 / 4.0 / 255.0;
                result.push(avg.clamp(0.0, 1.0));
            }
        }

        result
    }

    pub async fn encode_image_to_sensors(
        bus: &Arc<dyn MessageBus + Send + Sync>,
        sensor_ids: &[ZoooidId],
        image: &[f32],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        const MAX_CURRENT: f32 = 32.0;
        let sender = ZoooidId::new_v4();

        for (&agent_id, &pixel) in sensor_ids.iter().zip(image.iter()) {
            let current = pixel * MAX_CURRENT;
            bus.send(sender, agent_id, Message::AnalogInput(current))
                .await
                .map_err(|e| format!("Failed to push sensor current: {:?}", e))?;
        }

        Ok(())
    }
}

mod mnist_architecture {
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};
    use std::any::Any;
    use std::sync::Arc;

    use adaptiflux_core::agent::blueprint::base::AgentBlueprint;
    use adaptiflux_core::agent::blueprint::pidzooid::{PIDzooidBlueprint, PIDzooidParams};
    use adaptiflux_core::agent::state::{AgentUpdateResult, RoleType};
    use adaptiflux_core::agent::zoooid::Zoooid;
    use adaptiflux_core::core::message_bus::message::Message;
    use adaptiflux_core::core::topology::{TopologyChange, ZoooidTopology};
    use adaptiflux_core::memory::types::MemoryPayload;
    use adaptiflux_core::primitives::base::PrimitiveMessage;
    use adaptiflux_core::primitives::control::pid::PidParams;
    use adaptiflux_core::primitives::spiking::izhikevich::{
        IzhikevichNeuron, IzhikevichParams, IzhikevichState,
    };
    use adaptiflux_core::primitives::spiking::lif::LifParams;
    use adaptiflux_core::ZoooidId;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SpikingNeuronParams {
        pub izh_params: IzhikevichParams,
        pub input_gain: f32,
        pub gain_clamp_min: f32,
        pub gain_clamp_max: f32,
        pub connection_request_interval: u64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SpikingNeuronState {
        pub izh_state: IzhikevichState,
        pub input_gain: f32,
        pub spike_count: u64,
        pub tick_count: u64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SpikingNeuronBlueprint {
        pub params: SpikingNeuronParams,
        pub is_classifier: bool,
    }

    #[async_trait]
    impl AgentBlueprint for SpikingNeuronBlueprint {
        async fn initialize(
            &self,
        ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
            let izh_state =
                <IzhikevichNeuron as adaptiflux_core::primitives::base::Primitive>::initialize(
                    self.params.izh_params.clone(),
                );

            Ok(Box::new(SpikingNeuronState {
                izh_state,
                input_gain: self.params.input_gain,
                spike_count: 0,
                tick_count: 0,
            }))
        }

        async fn update(
            &self,
            state: &mut Box<dyn Any + Send + Sync>,
            inputs: Vec<(ZoooidId, Message)>,
            topology: &ZoooidTopology,
            _memory: Option<&MemoryPayload>,
        ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
            let state = state
                .downcast_mut::<SpikingNeuronState>()
                .ok_or("Invalid state type for spiking neuron")?;

            state.tick_count += 1;
            let mut total_input: f32 = 0.0;
            let mut control_adjustment: f32 = 0.0;

            for (sender, message) in inputs {
                match message {
                    Message::SpikeEvent { amplitude, .. } => {
                        total_input += amplitude;
                    }
                    Message::ControlSignal(value) => {
                        control_adjustment += value;
                    }
                    _ => {}
                }
            }

            if control_adjustment.abs() > 1e-12 {
                state.input_gain = (state.input_gain + control_adjustment * 0.02)
                    .clamp(self.params.gain_clamp_min, self.params.gain_clamp_max);
            }

            let input_current = total_input * state.input_gain;
            let primitive_inputs = vec![PrimitiveMessage::InputCurrent(input_current)];

            let (new_state, primitive_outputs) =
                <IzhikevichNeuron as adaptiflux_core::primitives::base::Primitive>::update(
                    state.izh_state.clone(),
                    &self.params.izh_params,
                    &primitive_inputs,
                );
            state.izh_state = new_state;

            let output_messages: Vec<Message> = primitive_outputs
                .into_iter()
                .filter_map(|prim_msg| match prim_msg {
                    PrimitiveMessage::Spike {
                        timestamp,
                        amplitude,
                    } => {
                        if self.is_classifier {
                            state.spike_count += 1;
                        }
                        Some(Message::SpikeEvent {
                            timestamp,
                            amplitude,
                        })
                    }
                    _ => None,
                })
                .collect();

            let topology_change = if state.tick_count % self.params.connection_request_interval == 0
            {
                let all_nodes: Vec<_> = topology.graph.nodes().collect();
                if !all_nodes.is_empty() {
                    let target_idx = (state.tick_count as usize) % all_nodes.len();
                    let target = all_nodes[target_idx];
                    Some(TopologyChange::RequestConnection(target))
                } else {
                    None
                }
            } else {
                None
            };

            Ok(AgentUpdateResult::new(
                output_messages,
                None,
                topology_change,
                false,
            ))
        }

        fn blueprint_type(&self) -> RoleType {
            RoleType::Cognitive
        }
    }

    pub struct MnistSpikingClassifierArchitecture {
        pub sensor_ids: Vec<ZoooidId>,
        pub hidden_ids: Vec<ZoooidId>,
        pub output_ids: Vec<ZoooidId>,
        pub pid_id: ZoooidId,
    }

    impl MnistSpikingClassifierArchitecture {
        pub fn all_adaptation_targets(&self) -> Vec<ZoooidId> {
            let mut ids = self.hidden_ids.clone();
            ids.extend(self.output_ids.iter());
            ids.push(self.pid_id);
            ids
        }

        pub fn reset_output_counts(
            &self,
            scheduler: &mut adaptiflux_core::core::scheduler::CoreScheduler,
        ) {
            for output_id in &self.output_ids {
                if let Some(handle) = scheduler.agents.get_mut(output_id) {
                    if let Some(state) = handle.state.downcast_mut::<SpikingNeuronState>() {
                        state.spike_count = 0;
                    }
                }
            }
        }

        pub fn decode_output(
            &self,
            scheduler: &adaptiflux_core::core::scheduler::CoreScheduler,
        ) -> u8 {
            let mut winner = 0;
            let mut max_spikes = 0;

            for (idx, output_id) in self.output_ids.iter().enumerate() {
                if let Some(handle) = scheduler.agents.get(output_id) {
                    if let Some(state) = handle.state.downcast_ref::<SpikingNeuronState>() {
                        if state.spike_count > max_spikes {
                            max_spikes = state.spike_count;
                            winner = idx;
                        }
                    }
                }
            }

            winner as u8
        }
    }

    pub async fn build_mnist_architecture(
        scheduler: &mut adaptiflux_core::core::scheduler::CoreScheduler,
    ) -> Result<MnistSpikingClassifierArchitecture, Box<dyn std::error::Error + Send + Sync>> {
        const SENSOR_SIDE: usize = 14;
        const HIDDEN_NEURONS: usize = 32;
        const OUTPUT_NEURONS: usize = 10;

        let mut sensor_ids = Vec::with_capacity(SENSOR_SIDE * SENSOR_SIDE);
        for _ in 0..SENSOR_SIDE * SENSOR_SIDE {
            let sensor_blueprint =
                adaptiflux_core::agent::blueprint::sensorzooid::SensorzooidBlueprint {
                    params: adaptiflux_core::agent::blueprint::sensorzooid::SensorzooidParams {
                        lif_params: LifParams {
                            tau_m: 12.0,
                            v_rest: -70.0,
                            v_thresh: -52.0,
                            v_reset: -70.0,
                            r_m: 12.0,
                            dt: 1.0,
                        },
                        connection_request_interval: 10,
                    },
                };
            let zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(sensor_blueprint)).await?;
            sensor_ids.push(zooid.id);
            scheduler.spawn_agent(zooid).await?;
        }

        let mut hidden_ids = Vec::with_capacity(HIDDEN_NEURONS);
        for _ in 0..HIDDEN_NEURONS {
            let hidden_blueprint = SpikingNeuronBlueprint {
                params: SpikingNeuronParams {
                    izh_params: IzhikevichParams::default(),
                    input_gain: 0.12,
                    gain_clamp_min: 0.01,
                    gain_clamp_max: 5.0,
                    connection_request_interval: 14,
                },
                is_classifier: false,
            };
            let zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(hidden_blueprint)).await?;
            hidden_ids.push(zooid.id);
            scheduler.spawn_agent(zooid).await?;
        }

        let mut output_ids = Vec::with_capacity(OUTPUT_NEURONS);
        for _ in 0..OUTPUT_NEURONS {
            let output_blueprint = SpikingNeuronBlueprint {
                params: SpikingNeuronParams {
                    izh_params: IzhikevichParams {
                        a: 0.02,
                        b: 0.2,
                        c: -65.0,
                        d: 2.0,
                        dt: 1.0,
                    },
                    input_gain: 0.10,
                    gain_clamp_min: 0.01,
                    gain_clamp_max: 6.0,
                    connection_request_interval: 14,
                },
                is_classifier: true,
            };
            let zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(output_blueprint)).await?;
            output_ids.push(zooid.id);
            scheduler.spawn_agent(zooid).await?;
        }

        let pid_blueprint = PIDzooidBlueprint {
            params: PIDzooidParams {
                pid_params: PidParams {
                    kp: 0.8,
                    ki: 0.1,
                    kd: 0.01,
                    dt: 1.0,
                },
                connection_request_interval: 12,
            },
        };
        let pid_zooid = Zoooid::new(ZoooidId::new_v4(), Box::new(pid_blueprint)).await?;
        let pid_id = pid_zooid.id;
        scheduler.spawn_agent(pid_zooid).await?;

        let mut topology = scheduler.topology.lock().await;
        for &sensor_id in &sensor_ids {
            for &hidden_id in &hidden_ids {
                topology.add_edge(sensor_id, hidden_id, Default::default());
            }
        }
        for &hidden_id in &hidden_ids {
            for &output_id in &output_ids {
                topology.add_edge(hidden_id, output_id, Default::default());
            }
        }
        for &output_id in &output_ids {
            topology.add_edge(pid_id, output_id, Default::default());
        }

        Ok(MnistSpikingClassifierArchitecture {
            sensor_ids,
            hidden_ids,
            output_ids,
            pid_id,
        })
    }

    pub struct SpikingGainLearner {
        pub learning_rate: f32,
    }

    impl adaptiflux_core::learning::OnlineLearner for SpikingGainLearner {
        fn adapt_parameters(
            &self,
            agent_id: ZoooidId,
            state: &mut Box<dyn Any + Send + Sync>,
            _role: RoleType,
            feedback: &adaptiflux_core::learning::signal_integration::FeedbackSignal,
        ) {
            let error = feedback
                .per_agent
                .get(&agent_id)
                .copied()
                .or(feedback.global_scalar)
                .unwrap_or(0.0)
                + feedback.memory_bias.get(&agent_id).copied().unwrap_or(0.0);

            if let Some(layer_state) = state.downcast_mut::<SpikingNeuronState>() {
                if error.abs() > 1e-9 {
                    layer_state.input_gain =
                        (layer_state.input_gain - error * self.learning_rate).clamp(0.01, 8.0);
                }
            }
        }
    }

    pub struct OutputExperienceRecorder {
        pub output_ids: Vec<ZoooidId>,
    }

    impl adaptiflux_core::memory::ExperienceRecorder for OutputExperienceRecorder {
        fn record_after_step(
            &self,
            agent_id: ZoooidId,
            iteration: u64,
            inputs: &[(ZoooidId, Message)],
            state: &dyn Any,
            _result: &AgentUpdateResult,
            store: &mut adaptiflux_core::memory::long_term_store::TableLongTermStore,
            indexer: &mut adaptiflux_core::memory::indexer::MetadataIndexer,
        ) {
            if !self.output_ids.contains(&agent_id) {
                return;
            }

            if let Some(output_state) = state.downcast_ref::<SpikingNeuronState>() {
                let embedding = vec![
                    output_state.input_gain,
                    output_state.spike_count as f32,
                    inputs
                        .iter()
                        .filter_map(|(_sender, msg)| match msg {
                            Message::SpikeEvent { amplitude, .. } => Some(*amplitude),
                            _ => None,
                        })
                        .sum(),
                ];

                let payload = Arc::new(output_state.clone());
                adaptiflux_core::memory::memory_integration::store_scalar_experience(
                    store,
                    indexer,
                    agent_id,
                    iteration,
                    "mnist_classification",
                    Some(embedding),
                    payload,
                    output_state.spike_count as f32,
                );
            }
        }
    }
}

mod evaluation {
    use crate::dataset::MnistDataset;
    use crate::mnist_architecture::MnistSpikingClassifierArchitecture;
    use adaptiflux_core::core::scheduler::CoreScheduler;
    use std::error::Error;

    pub async fn evaluate_mnist(
        scheduler: &mut CoreScheduler,
        architecture: &MnistSpikingClassifierArchitecture,
        dataset: &MnistDataset,
    ) -> Result<f64, Box<dyn Error + Send + Sync>> {
        let mut correct = 0;
        for (image, label) in dataset.test_images.iter().zip(dataset.test_labels.iter()) {
            architecture.reset_output_counts(scheduler);
            crate::dataset::encode_image_to_sensors(
                &scheduler.message_bus,
                &architecture.sensor_ids,
                image,
            )
            .await?;
            scheduler.run_for_iterations(3).await?;
            let prediction = architecture.decode_output(scheduler);
            if prediction == *label {
                correct += 1;
            }
        }
        Ok(correct as f64 / dataset.test_images.len() as f64)
    }
}

#[derive(Debug, Serialize)]
struct ExperimentCheckpoint {
    timestamp_ms: u128,
    epoch: usize,
    step: usize,
    elapsed_s: f64,
    accuracy: f64,
    error_value: f32,
    ram_mb: f64,
    cpu_usage: f32,
    active_agents: usize,
    total_agents: usize,
    total_connections: usize,
    topology_changes: usize,
    avg_iteration_time_ms: f64,
    cluster_groups: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .init();

    let dataset = dataset::load_mnist("../mnist-data", 60000, 10000).await?;

    let log_path = Path::new("logs/mnist_full_experiment.log");
    if let Some(parent) = log_path.parent() {
        create_dir_all(parent)?;
    }
    let mut log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;

    tracing::info!("Starting MNIST full experiment");

    let (mut scheduler, architecture) =
        run_training(&dataset, Duration::from_secs(7200), &mut log_file).await?;

    tracing::info!("Training completed, beginning evaluation");
    let accuracy = evaluation::evaluate_mnist(&mut scheduler, &architecture, &dataset).await?;
    tracing::info!("Final test accuracy = {:.2}%", accuracy * 100.0);

    serde_json::to_writer(
        &mut log_file,
        &serde_json::json!({
            "event": "final_evaluation",
            "accuracy": accuracy,
            "timestamp_ms": Instant::now().elapsed().as_millis(),
        }),
    )?;
    log_file.write_all(b"\n")?;

    println!("Final test accuracy: {:.2}%", accuracy * 100.0);
    Ok(())
}

async fn run_training(
    dataset: &dataset::MnistDataset,
    max_duration: Duration,
    log_file: &mut File,
) -> Result<
    (
        CoreScheduler,
        mnist_architecture::MnistSpikingClassifierArchitecture,
    ),
    Box<dyn Error + Send + Sync>,
> {
    let bus = Arc::new(LocalBus::new());
    let topology = Arc::new(tokio::sync::Mutex::new(ZoooidTopology::new()));
    let mut rule_engine = RuleEngine::new();
    rule_engine.add_behavior_rule(Box::new(LoadBalancingRule::new(0.8, 12)));
    rule_engine.add_behavior_rule(Box::new(IsolationRecoveryRule::new(1)));
    rule_engine.add_topology_rule(Box::new(ProximityConnectionRule::new(6.0, 3)));
    rule_engine.add_plasticity_rule(Box::new(SynapticPruningRule {
        min_weight: 0.05,
        idle_prune_after: Some(200),
        target_density: Some(0.15), // Target 15% density; prune aggressively if exceeded
        max_prune_per_iter: 10,     // Prune up to 10 edges per iteration when density is high
    }));
    rule_engine.add_plasticity_rule(Box::new(ActivityDependentSynaptogenesisRule {
        activity_threshold: 0.02,
        max_new_edges: 5,
        stdp_traffic_threshold: Some(40),
        stdp_delta: 0.02,
    }));
    rule_engine.add_plasticity_rule(Box::new(ClusterGroupingPlasticityRule {
        min_cluster_size: 4,
        evaluate_every: 400,
    }));
    rule_engine.add_consistency_check(Box::new(ConnectedTopologyCheck));
    rule_engine.add_consistency_check(Box::new(MinConnectivityCheck::new(0.1)));

    let resource_manager = ResourceManager::new();
    let mut scheduler = CoreScheduler::new(topology, rule_engine, resource_manager, bus.clone());
    scheduler.set_cycle_frequency(40);
    scheduler.enable_async_optimization(AsyncOptimizationConfig::new(num_cpus::get()));
    scheduler.enable_sparse_execution(SparseExecutionHook::new(Duration::from_millis(20)));
    scheduler.enable_sleep_scheduler(SleepScheduler::new(Duration::from_secs(10)));
    scheduler.enable_power_monitor(PowerMonitor::default());
    scheduler.hierarchy = Some(HierarchyHook {
        manager: AbstractionLayerManager::default(),
        detect_every: 500,
        min_cluster_size: 3,
        aggregation: AggregationFnKind::Mean,
    });

    let architecture = mnist_architecture::build_mnist_architecture(&mut scheduler).await?;

    let mut adaptation_engine = OnlineAdaptationEngine::new();
    adaptation_engine.set_default_learner(Arc::new(GradientDescentLearner {
        learning_rate: 0.008,
    }));
    for target in architecture.all_adaptation_targets() {
        adaptation_engine.register(
            target,
            Arc::new(mnist_architecture::SpikingGainLearner {
                learning_rate: 0.02,
            }),
        );
    }
    // NOTE: enable_async_optimization() requires the 'adaptiflux_optim' feature
    // Uncomment when building with: cargo run --example mnist_full_experiment --features adaptiflux_optim
    // adaptation_engine.enable_async_optimization(0.004, 8, 10);
    scheduler.online_adaptation = Some(OnlineAdaptationHook {
        engine: adaptation_engine,
        target_ids: architecture.all_adaptation_targets(),
    });

    let memory_store = Arc::new(tokio::sync::Mutex::new(
        adaptiflux_core::TableLongTermStore::new(),
    ));
    let memory_indexer = Arc::new(tokio::sync::Mutex::new(MetadataIndexer::new()));
    scheduler.memory_attention = Some(MemoryAttentionHook {
        store: memory_store.clone(),
        indexer: memory_indexer.clone(),
        retriever: adaptiflux_core::Retriever::new(8),
        attention: Arc::new(DotProductAttention::default()),
        focus: Arc::new(ErrorSimilarityFocus {
            tag: Some("mnist_full_experiment".into()),
            min_similarity: 0.0,
            observation_fn: observation_from_inputs,
        }),
        target_ids: Some(architecture.output_ids.clone()),
        inject_memory_into_feedback: true,
        memory_feedback_gain: 0.06,
        experience: Some(Arc::new(mnist_architecture::OutputExperienceRecorder {
            output_ids: architecture.output_ids.clone(),
        })),
    });

    let mut system = System::new_all();
    let pid = Pid::from(std::process::id() as usize);
    let mut max_rss_mb: f64 = 0.0;
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

            samples += 1;
            architecture.reset_output_counts(&mut scheduler);
            dataset::encode_image_to_sensors(
                &scheduler.message_bus,
                &architecture.sensor_ids,
                image,
            )
            .await?;
            scheduler.run_for_iterations(5).await?;
            let prediction = architecture.decode_output(&scheduler);
            let error_value = if prediction == *label { 0.0 } else { 1.0 };
            if prediction == *label {
                correct += 1;
            }

            let monitor_id = ZoooidId::new_v4();
            for target in architecture.all_adaptation_targets() {
                scheduler
                    .message_bus
                    .send(monitor_id, target, Message::Error(error_value))
                    .await
                    .map_err(|e| format!("Failed to post error feedback: {:?}", e))?;
            }

            scheduler.run_one_iteration().await?;

            let elapsed = start.elapsed().as_secs_f64();
            if samples % 100 == 0 {
                system.refresh_processes(ProcessesToUpdate::All, true);
                let (rss_mb, cpu_usage) = if let Some(process) = system.process(pid) {
                    (
                        process.memory() as f64 / 1024.0 / 1024.0,
                        process.cpu_usage(),
                    )
                } else {
                    (0.0, 0.0)
                };
                max_rss_mb = max_rss_mb.max(rss_mb);
                let accuracy = correct as f64 / samples as f64;
                let metrics = scheduler.get_metrics();
                let cluster_groups = scheduler
                    .hierarchy
                    .as_ref()
                    .map(|h| h.manager.group_count())
                    .unwrap_or(0);

                let checkpoint = ExperimentCheckpoint {
                    timestamp_ms: start.elapsed().as_millis(),
                    epoch,
                    step: samples,
                    elapsed_s: elapsed,
                    accuracy,
                    error_value,
                    ram_mb: rss_mb,
                    cpu_usage,
                    active_agents: scheduler.agent_count(),
                    total_agents: metrics.total_agents,
                    total_connections: metrics.total_connections,
                    topology_changes: metrics.topology_changes,
                    avg_iteration_time_ms: metrics.avg_iteration_time_ms,
                    cluster_groups,
                };
                serde_json::to_writer(&mut *log_file, &checkpoint)?;
                log_file.write_all(b"\n")?;
                tracing::info!(
                    epoch,
                    step = samples,
                    accuracy = accuracy,
                    ram_mb = rss_mb,
                    cpu_usage = cpu_usage,
                    active_agents = scheduler.agent_count(),
                    "checkpoint"
                );
                if rss_mb > 3800.0 {
                    tracing::warn!(ram_mb = rss_mb, "RAM usage exceeds 3.8 GB");
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

    serde_json::to_writer(
        &mut *log_file,
        &serde_json::json!({
            "event": "training_complete",
            "samples": samples,
            "duration_s": elapsed,
            "accuracy": correct as f64 / samples as f64,
            "peak_memory_mb": max_rss_mb,
        }),
    )?;
    log_file.write_all(b"\n")?;

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
