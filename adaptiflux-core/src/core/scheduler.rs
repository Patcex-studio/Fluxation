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

use crate::agent::state::AgentUpdateResult;
use crate::performance::async_optimization::AsyncOptimizationConfig;
use crate::performance::sparse_execution::SparseExecutionHook;
use crate::power::power_monitor::PowerMonitor;
use crate::power::sleep_scheduler::SleepScheduler;
use num_cpus;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, trace, warn};

use crate::agent::zoooid::Zoooid;
use crate::attention::apply_attention_to_hits;
use crate::attention::AttentionMechanism;
use crate::attention::FocusScheduler;
use crate::core::message_bus::message::Message;
use crate::core::message_bus::MessageBus;
use crate::core::resource_manager::ResourceManager;
use crate::core::topology::{ConnectionProperties, SystemMetrics, ZoooidTopology};
use crate::core::zoooid_handle::{SchedulerMetrics, ZoooidHandle};
use crate::hierarchy::{detect_dense_groups, AbstractionLayerManager, AggregationFnKind};
use crate::learning::{collect_error_feedback_from_bus, FeedbackSignal, OnlineAdaptationEngine};
use crate::memory::indexer::MetadataIndexer;
use crate::memory::long_term_store::TableLongTermStore;
use crate::memory::memory_integration::ExperienceRecorder;
use crate::memory::types::MemoryPayload;
use crate::memory::Retriever;
use crate::plasticity::{AppliedTopologyEffects, PlasticityRuntimeState};
use crate::rules::RuleEngine;
use crate::utils::types::{new_zoooid_id, StateValue, ZoooidId};

#[cfg(feature = "gpu")]
use crate::gpu::resource_manager::GpuResourceManager;

/// Optional online learning pass (parameter adaptation after agent updates).
pub struct OnlineAdaptationHook {
    pub engine: OnlineAdaptationEngine,
    pub target_ids: Vec<ZoooidId>,
}

/// Periodic cluster detection and abstraction bookkeeping inside the scheduler loop.
pub struct HierarchyHook {
    pub manager: AbstractionLayerManager,
    pub detect_every: u64,
    pub min_cluster_size: usize,
    pub aggregation: AggregationFnKind,
}

/// Long-term memory retrieval + attention before each agent update, optional experience logging.
pub struct MemoryAttentionHook {
    pub store: Arc<Mutex<TableLongTermStore>>,
    pub indexer: Arc<Mutex<MetadataIndexer>>,
    pub retriever: Retriever,
    pub attention: Arc<dyn AttentionMechanism + Send + Sync>,
    pub focus: Arc<dyn FocusScheduler + Send + Sync>,
    /// `None` → memory pass runs for every agent.
    pub target_ids: Option<Vec<ZoooidId>>,
    /// Scale and merge [`MemoryPayload::weighted_scalar_hint`] into [`FeedbackSignal::memory_bias`].
    pub inject_memory_into_feedback: bool,
    pub memory_feedback_gain: StateValue,
    /// Optional persistence after each agent step.
    pub experience: Option<Arc<dyn ExperienceRecorder + Send + Sync>>,
}

/// Core scheduler managing all agents, topology, rules, and consistency checks
///
/// The `CoreScheduler` is the central orchestrator of the Adaptiflux system. It manages the lifecycle
/// of agents (zoooids), coordinates message passing, applies plasticity rules for topology adaptation,
/// and integrates various hooks for learning, hierarchy, and performance optimization.
///
/// # Architecture
///
/// The scheduler operates in a continuous loop, executing agent updates, collecting feedback,
/// applying plasticity rules, and managing resources. It supports optional features like GPU acceleration,
/// asynchronous optimization, sparse execution, and power monitoring.
///
/// # Key Components
///
/// - **Agents**: Managed through `ZoooidHandle`s in a hash map.
/// - **Topology**: Dynamic graph of agent connections, protected by a mutex.
/// - **Rule Engine**: Applies plasticity rules for self-organization.
/// - **Resource Manager**: Handles computational resource allocation.
/// - **Message Bus**: Facilitates communication between agents.
/// - **Hooks**: Optional components for online adaptation, hierarchy, and memory attention.
///
/// # Examples
///
/// ```rust,no_run
/// use adaptiflux_core::{CoreScheduler, ZoooidTopology, RuleEngine, ResourceManager, LocalBus};
/// use std::sync::Arc;
/// use tokio::sync::Mutex;
///
/// // Create components
/// let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
/// let rule_engine = RuleEngine::new();
/// let resource_manager = ResourceManager::new();
/// let message_bus = Arc::new(LocalBus::new());
///
/// // Create scheduler
/// let mut scheduler = CoreScheduler::new(
///     topology,
///     rule_engine,
///     resource_manager,
///     message_bus,
/// );
///
/// // Add agents and run
/// // ... add agents ...
/// scheduler.run().unwrap();
/// ```
pub struct CoreScheduler {
    /// Map of agent IDs to their handles for lifecycle management
    pub agents: HashMap<ZoooidId, ZoooidHandle>,
    /// Dynamic topology graph of agent connections, protected by mutex for thread safety
    pub topology: Arc<Mutex<ZoooidTopology>>,
    /// Engine for applying plasticity rules to adapt the topology
    pub rule_engine: RuleEngine,
    /// Manager for computational resource allocation and monitoring
    pub resource_manager: ResourceManager,
    /// Message bus for inter-agent communication
    pub message_bus: Arc<dyn MessageBus + Send + Sync>,

    // Scheduler control
    /// Atomic flag to signal scheduler shutdown
    should_stop: Arc<AtomicBool>,
    /// Duration between scheduler cycles
    cycle_duration_ms: Duration,

    /// GPU resource manager for GPU-accelerated operations (requires "gpu" feature)
    #[cfg(feature = "gpu")]
    pub gpu_resource_manager: Option<Arc<Mutex<GpuResourceManager>>>,

    /// Configuration for asynchronous optimization tasks
    pub async_optimization: Option<AsyncOptimizationConfig>,
    /// Hook for sparse execution optimization
    pub sparse_execution: Option<SparseExecutionHook>,
    /// Scheduler for power management and sleep cycles
    pub sleep_scheduler: Option<SleepScheduler>,
    /// Monitor for power consumption metrics
    pub power_monitor: Option<PowerMonitor>,

    // Metrics
    /// Performance and operational metrics
    metrics: SchedulerMetrics,

    /// Runtime state for tracking activity and traffic signals used by plasticity rules
    pub plasticity_state: PlasticityRuntimeState,

    /// Accumulated wall-clock time for calculating average iteration duration
    time_accum_ms: f64,

    /// Optional hook for online parameter adaptation after agent updates
    /// Optional hook for online parameter adaptation after agent updates
    pub online_adaptation: Option<OnlineAdaptationHook>,
    /// Optional hook for hierarchical abstraction and cluster detection
    pub hierarchy: Option<HierarchyHook>,
    /// Optional hook for memory attention and retrieval integration
    pub memory_attention: Option<MemoryAttentionHook>,
}

impl CoreScheduler {
    /// Create a new CoreScheduler instance
    ///
    /// Initializes the scheduler with the core components required for operation.
    /// Optional hooks and features can be configured after creation.
    ///
    /// # Arguments
    ///
    /// * `topology` - Shared topology graph for managing agent connections
    /// * `rule_engine` - Engine for applying plasticity rules
    /// * `resource_manager` - Manager for computational resources
    /// * `message_bus` - Bus for inter-agent message passing
    ///
    /// # Returns
    ///
    /// A new `CoreScheduler` instance with default configuration
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use adaptiflux_core::{CoreScheduler, ZoooidTopology, RuleEngine, ResourceManager, LocalBus};
    /// use std::sync::Arc;
    /// use tokio::sync::Mutex;
    ///
    /// let topology = Arc::new(Mutex::new(ZoooidTopology::new()));
    /// let rule_engine = RuleEngine::new();
    /// let resource_manager = ResourceManager::new();
    /// let message_bus = Arc::new(LocalBus::new());
    ///
    /// let scheduler = CoreScheduler::new(topology, rule_engine, resource_manager, message_bus);
    /// ```
    pub fn new(
        topology: Arc<Mutex<ZoooidTopology>>,
        rule_engine: RuleEngine,
        resource_manager: ResourceManager,
        message_bus: Arc<dyn MessageBus + Send + Sync>,
    ) -> Self {
        Self {
            agents: HashMap::new(),
            topology,
            rule_engine,
            resource_manager,
            message_bus,
            should_stop: Arc::new(AtomicBool::new(false)),
            cycle_duration_ms: Duration::from_millis(100), // 10 Hz by default
            metrics: SchedulerMetrics::default(),
            plasticity_state: PlasticityRuntimeState::default(),
            time_accum_ms: 0.0,
            online_adaptation: None,
            hierarchy: None,
            memory_attention: None,
            async_optimization: None,
            sparse_execution: None,
            sleep_scheduler: None,
            power_monitor: None,
            #[cfg(feature = "gpu")]
            gpu_resource_manager: None,
        }
    }

    #[cfg(feature = "gpu")]
    pub fn new_with_gpu(
        topology: Arc<Mutex<ZoooidTopology>>,
        rule_engine: RuleEngine,
        resource_manager: ResourceManager,
        message_bus: Arc<dyn MessageBus + Send + Sync>,
        gpu_resource_manager: Option<Arc<Mutex<GpuResourceManager>>>,
    ) -> Self {
        Self {
            agents: HashMap::new(),
            topology,
            rule_engine,
            resource_manager,
            message_bus,
            should_stop: Arc::new(AtomicBool::new(false)),
            cycle_duration_ms: Duration::from_millis(100),
            metrics: SchedulerMetrics::default(),
            plasticity_state: PlasticityRuntimeState::default(),
            time_accum_ms: 0.0,
            online_adaptation: None,
            hierarchy: None,
            memory_attention: None,
            async_optimization: None,
            sparse_execution: None,
            sleep_scheduler: None,
            power_monitor: None,
            gpu_resource_manager,
        }
    }

    /// Set the cycle frequency (how often the scheduler runs iterations)
    pub fn set_cycle_frequency(&mut self, frequency_hz: u32) {
        if frequency_hz > 0 {
            self.cycle_duration_ms = Duration::from_millis(1000 / frequency_hz as u64);
            info!("CoreScheduler cycle frequency set to {} Hz", frequency_hz);
        }
    }

    /// Enable bounded asynchronous agent execution.
    pub fn enable_async_optimization(&mut self, config: AsyncOptimizationConfig) {
        self.async_optimization = Some(config);
    }

    /// Enable event-driven waiting for the scheduler.
    pub fn enable_sparse_execution(&mut self, hook: SparseExecutionHook) {
        self.sparse_execution = Some(hook);
    }

    /// Attach the sleep scheduler for low-activity agents.
    pub fn enable_sleep_scheduler(&mut self, scheduler: SleepScheduler) {
        self.sleep_scheduler = Some(scheduler);
    }

    /// Attach an optional power monitor.
    pub fn enable_power_monitor(&mut self, monitor: PowerMonitor) {
        self.power_monitor = Some(monitor);
    }

    /// Enable asynchronous optimization integration
    #[cfg(feature = "adaptiflux_optim")]
    pub fn enable_async_adaptation(&mut self, lr: f32, batch_size: usize, interval_steps: usize) {
        if let Some(ref mut hook) = self.online_adaptation {
            hook.engine
                .enable_async_optimization(lr, batch_size, interval_steps);
            // Set up channels if needed, but since engine has them, perhaps not
        }
    }

    /// Spawn a new agent into the system
    ///
    /// Registers the agent with the message bus, adds it to the topology graph,
    /// and allocates GPU resources if available and supported.
    ///
    /// # Arguments
    ///
    /// * `agent` - The `Zoooid` instance to spawn, containing blueprint and initial state
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on successful spawning, or an error if registration fails
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use adaptiflux_core::{CoreScheduler, Zoooid, ZoooidId, AgentBlueprint};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// # let mut scheduler = // ... create scheduler
    /// # let blueprint: Box<dyn AgentBlueprint + Send + Sync> = // ... create blueprint
    /// let agent = Zoooid::new(ZoooidId::new_v4(), blueprint).await?;
    /// scheduler.spawn_agent(agent).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn spawn_agent(
        &mut self,
        agent: Zoooid,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let agent_id = agent.id;
        self.message_bus.register_agent(agent_id).await?;
        self.topology.lock().await.add_node(agent_id);

        #[cfg(feature = "gpu")]
        let gpu_allocated = if agent.blueprint.supports_gpu() {
            if let Some(manager) = &self.gpu_resource_manager {
                let mut manager = manager.lock().await;
                if manager.allocate_for_agent(agent_id) {
                    info!("Allocated GPU resources for agent {}", agent_id);
                    true
                } else {
                    warn!("GPU resources unavailable for agent {}", agent_id);
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        let handle = ZoooidHandle {
            id: agent_id,
            blueprint: agent.blueprint,
            state: agent.state,
            update_count: 0,
            #[cfg(feature = "gpu")]
            gpu_allocated,
        };

        self.agents.insert(handle.id, handle);
        info!(
            "Spawned agent {}, total agents: {}",
            agent_id,
            self.agents.len()
        );

        Ok(())
    }

    /// Remove an agent and its topology node (e.g. simulated failure / external kill).
    pub async fn remove_agent_from_system(&mut self, id: ZoooidId) {
        if self.agents.remove(&id).is_some() {
            self.topology.lock().await.remove_node(id);
            info!("remove_agent_from_system: dropped {}", id);
        }
    }

    /// Get a clone of the current scheduler metrics
    pub fn get_metrics(&self) -> SchedulerMetrics {
        self.metrics.clone()
    }

    /// Get the number of active agents
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Signal the scheduler to stop gracefully
    pub fn stop(&self) {
        self.should_stop.store(true, Ordering::SeqCst);
        info!("CoreScheduler stop signal sent");
    }

    /// Clone the stop flag so another task can call [`AtomicBool::store`] without locking the scheduler.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.should_stop)
    }

    async fn apply_lifecycle_effects(
        &mut self,
        effects: AppliedTopologyEffects,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let mut extra_edge_ops = 0usize;
        for (area_hint, blueprint) in effects.spawn_requests {
            let id = new_zoooid_id();
            let agent = Zoooid::new(id, blueprint).await?;
            self.spawn_agent(agent).await?;
            if let Some(near) = area_hint {
                let mut topo = self.topology.lock().await;
                topo.add_edge(near, id, ConnectionProperties::default());
                extra_edge_ops += 1;
                drop(topo);
                self.plasticity_state
                    .edge_last_used
                    .insert((near, id), self.plasticity_state.global_iteration);
            }
        }
        for (id, reason) in effects.terminate_requests {
            if self.agents.remove(&id).is_some() {
                self.topology.lock().await.remove_node(id);
                match &reason {
                    Some(r) => info!("Topology lifecycle: removed {} ({})", id, r),
                    None => info!("Topology lifecycle: removed {}", id),
                }
            }
        }
        Ok(extra_edge_ops)
    }

    fn merge_hierarchy_groups(&mut self, groups: Vec<Vec<ZoooidId>>) {
        let Some(h) = &mut self.hierarchy else {
            return;
        };
        for g in groups {
            h.manager.upsert_group(g, h.aggregation);
        }
    }

    async fn apply_topology_effects_bundle(
        &mut self,
        mut effects: AppliedTopologyEffects,
        topology_changes: &mut usize,
        context: &'static str,
    ) {
        *topology_changes += effects.edge_operations;
        for (a, b) in &effects.new_edges {
            self.plasticity_state
                .edge_last_used
                .insert((*a, *b), self.plasticity_state.global_iteration);
        }
        let groups = std::mem::take(&mut effects.agent_groups);
        self.merge_hierarchy_groups(groups);
        match self.apply_lifecycle_effects(effects).await {
            Ok(extra) => *topology_changes += extra,
            Err(e) => error!("Lifecycle after {}: {}", context, e),
        }
    }

    async fn iteration_step(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let iter_start = Instant::now();
        let iteration = self.plasticity_state.global_iteration;

        let metrics_phase_start = Instant::now();
        // Phase 1: Metrics (before rules)
        let metrics = {
            let topology_guard = self.topology.lock().await;
            SystemMetrics::from_topology(&topology_guard)
        };
        let metrics_phase_duration_ms = metrics_phase_start.elapsed().as_secs_f64() * 1000.0;

        debug!(
            "Topology metrics - Agents: {}, Connections: {}",
            metrics.total_zoooids, metrics.total_connections
        );

        let mut topology_changes = 0usize;
        let topology_rules_phase_start = Instant::now();

        // Phase 2a: Classical topology rules → apply → lifecycle
        match self
            .rule_engine
            .run_topology_rules(&self.topology, &metrics)
            .await
        {
            Ok(topo_actions) => {
                match self
                    .rule_engine
                    .apply_topology_actions(&self.topology, topo_actions)
                    .await
                {
                    Ok(effects) => {
                        self.apply_topology_effects_bundle(
                            effects,
                            &mut topology_changes,
                            "topology rules",
                        )
                        .await;
                    }
                    Err(e) => error!("apply_topology_actions (topology rules): {}", e),
                }
            }
            Err(e) => error!("Topology rule execution failed: {}", e),
        }
        let topology_rules_phase_duration_ms =
            topology_rules_phase_start.elapsed().as_secs_f64() * 1000.0;

        // Phase 2b: Structural plasticity (signals from previous iteration)
        let metrics_plasticity = {
            let g = self.topology.lock().await;
            SystemMetrics::from_topology(&g)
        };
        let plasticity_ctx = {
            let topo = self.topology.lock().await;
            self.plasticity_state.snapshot_plasticity_context(&topo)
        };

        let plasticity_rules_phase_start = Instant::now();
        match self
            .rule_engine
            .run_plasticity_rules(&self.topology, &metrics_plasticity, &plasticity_ctx)
            .await
        {
            Ok(p_actions) => {
                match self
                    .rule_engine
                    .apply_topology_actions(&self.topology, p_actions)
                    .await
                {
                    Ok(effects) => {
                        self.apply_topology_effects_bundle(
                            effects,
                            &mut topology_changes,
                            "plasticity rules",
                        )
                        .await;
                    }
                    Err(e) => error!("apply_topology_actions (plasticity): {}", e),
                }
            }
            Err(e) => error!("Plasticity rule execution failed: {}", e),
        }
        let plasticity_rules_phase_duration_ms =
            plasticity_rules_phase_start.elapsed().as_secs_f64() * 1000.0;

        // Phase 3: Update all agents (deterministic order by id for reproducible message timing)
        let mut agents_updated = 0;
        let mut terminated_agents = Vec::new();
        let mut inline_error_feedback = FeedbackSignal::default();

        let mut sorted_ids: Vec<ZoooidId> = self.agents.keys().cloned().collect();
        sorted_ids.sort();

        let mut memory_hints: HashMap<ZoooidId, crate::utils::types::StateValue> = HashMap::new();

        struct PendingAgentUpdate {
            agent_id: ZoooidId,
            handle: ZoooidHandle,
            inputs: Vec<(ZoooidId, Message)>,
            memory_inputs_snapshot: Vec<(ZoooidId, Message)>,
            mem_for_update: Option<MemoryPayload>,
        }

        struct AgentUpdateOutcome {
            handle: ZoooidHandle,
            result: Result<AgentUpdateResult, String>,
            agent_id: ZoooidId,
            memory_inputs_snapshot: Vec<(ZoooidId, Message)>,
            inputs: Vec<(ZoooidId, Message)>,
        }

        let mut pending_updates = Vec::new();
        for agent_id in sorted_ids {
            if let Some(handle) = self.agents.remove(&agent_id) {
                let inputs = self.message_bus.receive(agent_id).await.unwrap_or_default();
                for (_sender, msg) in &inputs {
                    if let Message::Error(e) = msg {
                        inline_error_feedback.merge_scalar(agent_id, *e);
                    }
                }

                if let Some(scheduler) = &mut self.sleep_scheduler {
                    if scheduler.should_sleep(agent_id) && inputs.is_empty() {
                        scheduler.record_idle(agent_id);
                        self.agents.insert(agent_id, handle);
                        continue;
                    }
                }

                let memory_inputs_snapshot = if self.memory_attention.is_some() {
                    inputs.clone()
                } else {
                    Vec::new()
                };

                let mut mem_for_update: Option<MemoryPayload> = None;
                if let Some(hook) = &self.memory_attention {
                    let apply = hook
                        .target_ids
                        .as_ref()
                        .map(|ids| ids.contains(&agent_id))
                        .unwrap_or(true);
                    if apply {
                        let q = hook.focus.focus_query(
                            agent_id,
                            iteration,
                            &inputs,
                            handle.blueprint.blueprint_type().clone(),
                        );
                        let qvec = hook.focus.query_vector(
                            agent_id,
                            iteration,
                            &inputs,
                            handle.blueprint.blueprint_type(),
                        );
                        let hits = {
                            let store = hook.store.lock().await;
                            let indexer = hook.indexer.lock().await;
                            #[allow(clippy::explicit_auto_deref)]
                            hook.retriever.retrieve(&*store, &*indexer, &q)
                        };
                        let mem = {
                            let store = hook.store.lock().await;
                            #[allow(clippy::explicit_auto_deref)]
                            apply_attention_to_hits(&*store, hook.attention.as_ref(), &qvec, &hits)
                        };
                        if !mem.is_empty() {
                            mem_for_update = Some(mem);
                        }

                        let hint = mem_for_update
                            .as_ref()
                            .and_then(MemoryPayload::weighted_scalar_hint)
                            .unwrap_or(0.0)
                            as crate::utils::types::StateValue;
                        memory_hints.insert(agent_id, hint);
                    }
                }

                pending_updates.push(PendingAgentUpdate {
                    agent_id,
                    handle,
                    inputs,
                    memory_inputs_snapshot,
                    mem_for_update,
                });
            }
        }

        let topology_snapshot = {
            let topology_guard = self.topology.lock().await;
            topology_guard.clone()
        };

        let config = self
            .async_optimization
            .clone()
            .unwrap_or_else(|| AsyncOptimizationConfig::new(num_cpus::get()));

        let update_phase_start = std::time::Instant::now();
        let tasks: Vec<_> = pending_updates
            .into_iter()
            .map(|pending| {
                let topology_snapshot = topology_snapshot.clone();
                async move {
                    let PendingAgentUpdate {
                        agent_id,
                        mut handle,
                        inputs,
                        memory_inputs_snapshot,
                        mem_for_update,
                    } = pending;

                    let inputs_for_update = inputs.clone();
                    let update_start = Instant::now();
                    let result = match handle
                        .blueprint
                        .update(
                            &mut handle.state,
                            inputs_for_update.clone(),
                            &topology_snapshot,
                            mem_for_update.as_ref(),
                        )
                        .await
                    {
                        Ok(res) => Ok(res),
                        Err(e) => Err(e.to_string()),
                    };
                    let update_duration_ms = update_start.elapsed().as_secs_f64() * 1000.0;
                    if iteration.is_multiple_of(100) {
                        let output_count = result
                            .as_ref()
                            .map(|r| r.output_messages.len())
                            .unwrap_or(0);
                        trace!(
                            agent_id = ?agent_id,
                            iteration,
                            duration_ms = update_duration_ms,
                            input_count = inputs_for_update.len(),
                            output_count,
                            "Agent update duration"
                        );
                    }

                    AgentUpdateOutcome {
                        handle,
                        result,
                        agent_id,
                        memory_inputs_snapshot,
                        inputs,
                    }
                }
            })
            .collect();

        let outcomes = config.run_batched(tasks).await;
        let update_phase_duration_ms = update_phase_start.elapsed().as_secs_f64() * 1000.0;
        debug!(
            "Iteration {} agent update phase duration: {:.2}ms",
            iteration, update_phase_duration_ms
        );
        for outcome in outcomes {
            let agent_id = outcome.agent_id;
            let handle = outcome.handle;
            match outcome.result {
                Ok(result) => {
                    if let Some(hook) = &self.memory_attention {
                        if let Some(rec) = &hook.experience {
                            let mut store = hook.store.lock().await;
                            let mut indexer = hook.indexer.lock().await;
                            rec.record_after_step(
                                agent_id,
                                iteration,
                                &outcome.memory_inputs_snapshot,
                                handle.state.as_ref(),
                                &result,
                                &mut store,
                                &mut indexer,
                            );
                        }
                    }

                    self.plasticity_state.record_agent_pulse(
                        agent_id,
                        result.output_messages.len() as crate::utils::types::StateValue,
                    );
                    if !result.output_messages.is_empty() {
                        let neighbors = self.topology.lock().await.get_neighbors(agent_id);
                        if !neighbors.is_empty() {
                            debug!(
                                "Agent {} sending {} messages to {} neighbors",
                                agent_id,
                                result.output_messages.len(),
                                neighbors.len()
                            );
                            for target in neighbors {
                                for message in &result.output_messages {
                                    self.plasticity_state.record_edge_use(agent_id, target);
                                    if let Err(e) = self
                                        .message_bus
                                        .send(agent_id, target, message.clone())
                                        .await
                                    {
                                        warn!(
                                            "Failed to send message from {} to {}: {}",
                                            agent_id, target, e
                                        );
                                    }
                                }
                            }
                        } else {
                            debug!(
                                "Agent {} has {} output messages but no neighbors connected",
                                agent_id,
                                result.output_messages.len()
                            );
                        }
                    }

                    if let Some(change) = result.topology_change_request {
                        let mut topology = self.topology.lock().await;
                        match change {
                            crate::core::topology::TopologyChange::RequestConnection(target) => {
                                topology.add_edge(agent_id, target, Default::default());
                                topology_changes += 1;
                                self.plasticity_state.edge_last_used.insert(
                                    (agent_id, target),
                                    self.plasticity_state.global_iteration,
                                );
                                debug!("Added connection: {} -> {}", agent_id, target);
                            }
                            crate::core::topology::TopologyChange::RemoveConnection(
                                source,
                                target,
                            ) => {
                                topology.remove_edge(source, target);
                                topology_changes += 1;
                                debug!("Removed connection: {} -> {}", source, target);
                            }
                        }
                    }

                    if result.terminate {
                        terminated_agents.push(agent_id);
                    }

                    if let Some(scheduler) = &mut self.sleep_scheduler {
                        if !outcome.inputs.is_empty() || !result.output_messages.is_empty() {
                            scheduler.record_activity(agent_id);
                        }
                    }

                    self.agents.insert(agent_id, handle);
                    if let Some(handle) = self.agents.get_mut(&agent_id) {
                        handle.update_count += 1;
                    }
                    agents_updated += 1;
                }
                Err(err) => {
                    error!("Agent {} update failed: {}", agent_id, err);
                    self.agents.insert(agent_id, handle);
                }
            }
        }

        let learning_phase_start = Instant::now();
        // Phase 4: Learning feedback (errors consumed this tick + any leftover inbox) + adaptation
        if let Some(hook) = &mut self.online_adaptation {
            let ids = hook.target_ids.clone();
            let mut fb = inline_error_feedback;
            if let Ok(extra) =
                collect_error_feedback_from_bus(self.message_bus.as_ref(), &ids).await
            {
                if let Some(g) = extra.global_scalar {
                    fb.set_global(g);
                }
                for (k, v) in extra.per_agent {
                    fb.per_agent.entry(k).or_insert(v);
                }
            }
            if let Some(mh) = &self.memory_attention {
                if mh.inject_memory_into_feedback {
                    for (&id, &hint) in &memory_hints {
                        fb.merge_memory_bias(id, hint * mh.memory_feedback_gain);
                    }
                }
            }

            // Synchronous adaptation
            hook.engine
                .run_for_targets_on_handles(&ids, &fb, &mut self.agents);

            // Asynchronous adaptation: send gradients
            #[cfg(feature = "adaptiflux_optim")]
            for &id in &ids {
                if let Some(handle) = self.agents.get(&id) {
                    let role = handle.blueprint.blueprint_type();
                    if matches!(role, crate::agent::state::RoleType::Pid) {
                        if let Some(pid) = handle.state.downcast_ref::<PIDzooidState>() {
                            let err = fb
                                .per_agent
                                .get(&id)
                                .copied()
                                .unwrap_or(fb.global_scalar.unwrap_or(0.0));
                            let params = [pid.pid_params.kp, pid.pid_params.ki, pid.pid_params.kd];
                            let sum = params[0] + params[1] + params[2];
                            let gradient = 2.0 * (sum - err);
                            let gradients = vec![gradient, gradient, gradient];
                            let current_params = params.to_vec();
                            hook.engine
                                .request_parameter_update(
                                    id,
                                    "pid_gains".to_string(),
                                    gradients,
                                    current_params,
                                )
                                .await;
                        }
                    } else if matches!(role, crate::agent::state::RoleType::Cognitive) {
                        if let Some(cog) = handle.state.downcast_ref::<CognitivezooidState>() {
                            let err = fb
                                .per_agent
                                .get(&id)
                                .copied()
                                .unwrap_or(fb.global_scalar.unwrap_or(0.0));
                            let params = [
                                cog.izh_params.a,
                                cog.izh_params.b,
                                cog.izh_params.c,
                                cog.izh_params.d,
                            ];
                            let sum = params[0] + params[1] + params[2] + params[3];
                            let gradient = 2.0 * (sum - err);
                            let gradients = vec![gradient, gradient, gradient, gradient];
                            let current_params = params.to_vec();
                            hook.engine
                                .request_parameter_update(
                                    id,
                                    "izhikevich_params".to_string(),
                                    gradients,
                                    current_params,
                                )
                                .await;
                        }
                    }
                    // Add more roles as needed
                }
            }

            // Apply updated parameters
            #[cfg(feature = "adaptiflux_optim")]
            hook.engine.apply_updated_parameters(&mut self.agents);
        }
        let learning_phase_duration_ms = learning_phase_start.elapsed().as_secs_f64() * 1000.0;

        let hierarchy_phase_start = Instant::now();
        // Phase 5: Optional native cluster pass (complements `TopologyAction::GroupAgents`)
        if let Some(h) = &mut self.hierarchy {
            if h.detect_every > 0 && iteration > 0 && iteration.is_multiple_of(h.detect_every) {
                let topo = self.topology.lock().await;
                let clusters = detect_dense_groups(&topo, h.min_cluster_size);
                drop(topo);
                h.manager.sync_from_clusters(clusters, h.aggregation);
            }
        }
        let hierarchy_phase_duration_ms = hierarchy_phase_start.elapsed().as_secs_f64() * 1000.0;

        let consistency_phase_start = Instant::now();
        // Phase 6: Consistency checks
        let agent_ids: Vec<ZoooidId> = self.agents.keys().cloned().collect();
        if let Err(e) = self
            .rule_engine
            .run_consistency_checks(&self.topology, &agent_ids)
            .await
        {
            warn!("Consistency check failed: {}", e);
        }
        let consistency_phase_duration_ms =
            consistency_phase_start.elapsed().as_secs_f64() * 1000.0;

        let lifecycle_phase_start = Instant::now();
        // Phase 7: Clean up terminated agents
        let mut agents_terminated = 0;
        if !terminated_agents.is_empty() {
            let mut topology = self.topology.lock().await;
            for agent_id in terminated_agents {
                self.agents.remove(&agent_id);
                topology.remove_node(agent_id);
                agents_terminated += 1;
                info!("Terminated and removed agent {}", agent_id);
            }
        }
        let lifecycle_phase_duration_ms = lifecycle_phase_start.elapsed().as_secs_f64() * 1000.0;

        self.plasticity_state.decay_activity(0.98);
        self.plasticity_state.advance_iteration();

        // Update metrics
        let iter_duration = iter_start.elapsed().as_secs_f64() * 1000.0;
        self.time_accum_ms += iter_duration;

        self.metrics.iteration_count = iteration;
        self.metrics.total_agents = self.agents.len();
        self.metrics.total_connections = self.topology.lock().await.graph.edge_count();
        self.metrics.agents_updated = agents_updated;
        self.metrics.agents_terminated = agents_terminated;
        self.metrics.topology_changes = topology_changes;

        self.metrics.avg_iteration_time_ms = self.time_accum_ms / (iteration as f64 + 1.0);

        debug!(
            "Iteration {} phase breakdown - metrics: {:.2}ms, topology: {:.2}ms, plasticity: {:.2}ms, agent_update: {:.2}ms, learning: {:.2}ms, hierarchy: {:.2}ms, consistency: {:.2}ms, lifecycle: {:.2}ms, total: {:.2}ms",
            iteration,
            metrics_phase_duration_ms,
            topology_rules_phase_duration_ms,
            plasticity_rules_phase_duration_ms,
            update_phase_duration_ms,
            learning_phase_duration_ms,
            hierarchy_phase_duration_ms,
            consistency_phase_duration_ms,
            lifecycle_phase_duration_ms,
            iter_duration,
        );

        if iteration.is_multiple_of(100) && iteration > 0 {
            info!(
                "Scheduler stats - Iteration: {}, Agents: {}, Avg time: {:.2}ms",
                iteration, self.metrics.total_agents, self.metrics.avg_iteration_time_ms
            );
        }

        if let Some(monitor) = &mut self.power_monitor {
            monitor.record_cycle(agents_updated > 0);
        }

        Ok(())
    }

    async fn wait_for_next_cycle(&self) {
        let notify = self.message_bus.notifier();
        if let Some(sparse) = &self.sparse_execution {
            sparse.wait_for_next_cycle(notify).await;
            return;
        }

        if let Some(external) = notify {
            tokio::select! {
                _ = sleep(self.cycle_duration_ms) => {},
                _ = external.notified() => {},
            }
        } else {
            sleep(self.cycle_duration_ms).await;
        }
    }

    /// Single scheduler iteration (topology rules, plasticity, agents, learning, consistency).
    pub async fn run_one_iteration(
        &mut self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.iteration_step().await
    }

    /// Run a fixed number of iterations (for scripted demos, tests, and phased experiments).
    pub async fn run_for_iterations(
        &mut self,
        count: u64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for _ in 0..count {
            self.iteration_step().await?;
        }
        Ok(())
    }

    /// Main scheduler loop - coordinates all system components
    ///
    /// Runs the continuous scheduler loop that executes agent updates, applies plasticity rules,
    /// manages resources, and handles topology adaptation. The loop continues until `stop()` is called
    /// or an error occurs.
    ///
    /// The loop performs the following phases per iteration:
    /// 1. Collect system metrics
    /// 2. Apply topology rules for plasticity
    /// 3. Update agents and collect feedback
    /// 4. Apply online adaptation if enabled
    /// 5. Update hierarchy abstractions
    /// 6. Apply memory attention mechanisms
    /// 7. Handle power management and sleep scheduling
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` when the scheduler stops normally, or an error if a critical failure occurs
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use adaptiflux_core::CoreScheduler;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    /// # let mut scheduler = // ... create and configure scheduler
    /// scheduler.run().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!("Starting CoreScheduler main loop...");
        info!("Initial agent count: {}", self.agents.len());

        while !self.should_stop.load(Ordering::SeqCst) {
            self.iteration_step().await?;
            self.wait_for_next_cycle().await;
        }

        let total = self.plasticity_state.global_iteration;
        info!("CoreScheduler stopped after {} iterations", total);
        info!(
            "Final metrics - Total agents: {}, Total iterations: {}, Avg time: {:.2}ms",
            self.agents.len(),
            total,
            self.metrics.avg_iteration_time_ms
        );

        Ok(())
    }
}
