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

//! Online parameter adaptation driven by [`super::signal_integration::FeedbackSignal`].

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(any(feature = "custom_optim", feature = "adaptiflux_optim"))]
use std::sync::Mutex;

use crate::agent::blueprint::pidzooid::PIDzooidState;
use crate::agent::state::RoleType;
use crate::learning::parameter_tuning::{
    evolutionary_scalar_update, evolutionary_tau_update, pid_gains_sgd_step,
};
use crate::learning::signal_integration::FeedbackSignal;
use crate::utils::types::ZoooidId;

#[cfg(feature = "adaptiflux_optim")]
use adaptiflux_optim::{Adam, GradientAccumulator, Optimizer};
#[cfg(feature = "custom_optim")]
use custom_optim::{Optimizer, OptimizerConfig};

#[cfg(feature = "adaptiflux_optim")]
use tokio::sync::mpsc::{self, Receiver, Sender};
#[cfg(feature = "adaptiflux_optim")]
use tokio::task::JoinHandle;
/// Messages for asynchronous optimization
#[derive(Debug, Clone)]
pub struct AgentGradients {
    pub agent_id: ZoooidId,
    pub params_id: String, // e.g., "pid_gains" or "izhikevich_params"
    pub gradients: Vec<f32>,
    pub current_params: Vec<f32>, // Current parameter values for optimizer initialization
}

#[derive(Debug, Clone)]
pub struct UpdatedAgentParams {
    pub agent_id: ZoooidId,
    pub params_id: String,
    pub params: Vec<f32>,
}
/// Adapts primitive parameters held inside agent state using a feedback signal.
pub trait OnlineLearner: Send + Sync {
    fn adapt_parameters(
        &self,
        agent_id: ZoooidId,
        state: &mut Box<dyn Any + Send + Sync>,
        role: RoleType,
        feedback: &FeedbackSignal,
    );
}

/// Registry of per-agent learners with an optional default.
#[derive(Default)]
pub struct OnlineAdaptationEngine {
    learners: HashMap<ZoooidId, Arc<dyn OnlineLearner + Send + Sync>>,
    default_learner: Option<Arc<dyn OnlineLearner + Send + Sync>>,

    // Asynchronous optimization fields
    #[cfg(feature = "adaptiflux_optim")]
    grad_tx: Option<Sender<AgentGradients>>,
    #[cfg(feature = "adaptiflux_optim")]
    updated_params_rx: Option<Receiver<UpdatedAgentParams>>,
    #[cfg(feature = "adaptiflux_optim")]
    optim_task_handle: Option<JoinHandle<()>>,
}

impl OnlineAdaptationEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_default_learner(&mut self, learner: Arc<dyn OnlineLearner + Send + Sync>) {
        self.default_learner = Some(learner);
    }

    pub fn register(&mut self, agent_id: ZoooidId, learner: Arc<dyn OnlineLearner + Send + Sync>) {
        self.learners.insert(agent_id, learner);
    }

    /// Enable asynchronous optimization with adaptiflux-optim
    #[cfg(feature = "adaptiflux_optim")]
    pub fn enable_async_optimization(&mut self, lr: f32, batch_size: usize, interval_steps: usize) {
        let (grad_tx, grad_rx) = mpsc::channel(100);
        let (updated_params_tx, updated_params_rx) = mpsc::channel(100);

        self.grad_tx = Some(grad_tx);
        self.updated_params_rx = Some(updated_params_rx);

        self.optim_task_handle =
            Some(self.spawn_optim_loop(grad_rx, updated_params_tx, lr, batch_size, interval_steps));
    }

    /// Spawn the optimization task
    #[cfg(feature = "adaptiflux_optim")]
    fn spawn_optim_loop(
        &self,
        mut grad_rx: Receiver<AgentGradients>,
        updated_params_tx: Sender<UpdatedAgentParams>,
        lr: f32,
        batch_size: usize,
        interval_steps: usize,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            let mut param_buffers: HashMap<(ZoooidId, String), Vec<Vec<f32>>> = HashMap::new();
            let mut optimizers: HashMap<(ZoooidId, String), Adam> = HashMap::new();
            let mut step_counters: HashMap<(ZoooidId, String), usize> = HashMap::new();

            while let Some(grads) = grad_rx.recv().await {
                let key = (grads.agent_id, grads.params_id.clone());
                let buffer = param_buffers.entry(key.clone()).or_insert_with(Vec::new);
                buffer.push(grads.gradients);

                let counter = step_counters.entry(key.clone()).or_insert(0);
                *counter += 1;

                if buffer.len() >= batch_size || *counter % interval_steps == 0 {
                    // Compute averaged gradients
                    let num_grads = buffer.len() as f32;
                    let mut avg_grads = vec![0.0; buffer[0].len()];
                    for g in buffer.iter() {
                        for (i, &val) in g.iter().enumerate() {
                            avg_grads[i] += val;
                        }
                    }
                    for g in &mut avg_grads {
                        *g /= num_grads;
                    }

                    // Get or create optimizer with current params
                    let optimizer = optimizers.entry(key.clone()).or_insert_with(|| {
                        let mut opt = Adam::new(lr);
                        let mut params_copy = grads.current_params.clone();
                        opt.init(&mut params_copy); // Use current params for initialization
                        opt
                    });

                    let mut params = grads.current_params.clone(); // Start from current params
                    optimizer.step(&mut params, &avg_grads);

                    // Send updated params
                    let updated = UpdatedAgentParams {
                        agent_id: grads.agent_id,
                        params_id: grads.params_id,
                        params,
                    };
                    if updated_params_tx.send(updated).await.is_err() {
                        break; // Receiver dropped
                    }

                    buffer.clear();
                }
            }
        })
    }

    /// Request parameter update by sending gradients
    #[cfg(feature = "adaptiflux_optim")]
    pub async fn request_parameter_update(
        &self,
        agent_id: ZoooidId,
        params_id: String,
        gradients: Vec<f32>,
        current_params: Vec<f32>,
    ) {
        if let Some(tx) = &self.grad_tx {
            let grads = AgentGradients {
                agent_id,
                params_id,
                gradients,
                current_params,
            };
            let _ = tx.send(grads).await;
        }
    }

    /// Apply updated parameters if available
    #[cfg(feature = "adaptiflux_optim")]
    pub fn apply_updated_parameters(
        &mut self,
        agents: &mut HashMap<ZoooidId, crate::core::zoooid_handle::ZoooidHandle>,
    ) {
        if let Some(rx) = &mut self.updated_params_rx {
            while let Ok(updated) = rx.try_recv() {
                if let Some(handle) = agents.get_mut(&updated.agent_id) {
                    // Apply params based on params_id
                    // This is simplified; need to map params_id to actual state fields
                    if updated.params_id == "pid_gains" {
                        if let Some(pid) = handle.state.downcast_mut::<PIDzooidState>() {
                            if updated.params.len() >= 3 {
                                pid.pid_params.kp = updated.params[0];
                                pid.pid_params.ki = updated.params[1];
                                pid.pid_params.kd = updated.params[2];
                            }
                        }
                    } else if updated.params_id == "izhikevich_params" {
                        use crate::agent::blueprint::cognitivezooid::CognitivezooidState;
                        if let Some(cog) = handle.state.downcast_mut::<CognitivezooidState>() {
                            if updated.params.len() >= 4 {
                                cog.izh_params.a = updated.params[0];
                                cog.izh_params.b = updated.params[1];
                                cog.izh_params.c = updated.params[2];
                                cog.izh_params.d = updated.params[3];
                            }
                        }
                    }
                    // Add more cases for other params
                }
            }
        }
    }

    /// Run adaptation for registered targets using live [`crate::core::scheduler::ZoooidHandle`] state.
    pub fn run_for_targets_on_handles(
        &self,
        targets: &[ZoooidId],
        feedback: &FeedbackSignal,
        agents: &mut HashMap<ZoooidId, crate::core::zoooid_handle::ZoooidHandle>,
    ) {
        for &id in targets {
            let Some(handle) = agents.get_mut(&id) else {
                continue;
            };
            let role = handle.blueprint.blueprint_type();
            if let Some(l) = self.learners.get(&id) {
                l.adapt_parameters(id, &mut handle.state, role, feedback);
            } else if let Some(l) = &self.default_learner {
                l.adapt_parameters(id, &mut handle.state, role, feedback);
            }
        }
    }
}

use crate::utils::types::Param;

/// SGD-style tuning of PID gains stored in [`PIDzooidState`].
pub struct GradientDescentLearner {
    pub learning_rate: Param,
}

impl Default for GradientDescentLearner {
    fn default() -> Self {
        Self {
            learning_rate: 0.02,
        }
    }
}

impl OnlineLearner for GradientDescentLearner {
    fn adapt_parameters(
        &self,
        agent_id: ZoooidId,
        state: &mut Box<dyn Any + Send + Sync>,
        role: RoleType,
        feedback: &FeedbackSignal,
    ) {
        if !matches!(role, RoleType::Pid) {
            return;
        }
        let Some(pid) = state.downcast_mut::<PIDzooidState>() else {
            return;
        };
        let mem = feedback.memory_bias.get(&agent_id).copied().unwrap_or(0.0);
        let err = match (
            feedback
                .per_agent
                .get(&agent_id)
                .or(feedback.global_scalar.as_ref())
                .copied(),
            mem,
        ) {
            (Some(e), m) => e + m,
            (None, m) if m.abs() >= 1e-6 => m,
            _ => return,
        };
        let (kp, ki, kd) = pid_gains_sgd_step(
            pid.pid_params.kp,
            pid.pid_params.ki,
            pid.pid_params.kd,
            err,
            self.learning_rate,
        );
        pid.pid_params.kp = kp;
        pid.pid_params.ki = ki;
        pid.pid_params.kd = kd;
    }
}

/// Evolutionary jitter on PID gains or a single `tau_m` field in [`EvolutionaryTuneState`].
pub struct EvolutionaryOptimizerLearner {
    pub sigma: Param,
    pub tick: std::sync::atomic::AtomicU64,
}

impl EvolutionaryOptimizerLearner {
    pub fn new(sigma: Param) -> Self {
        Self {
            sigma,
            tick: std::sync::atomic::AtomicU64::new(0),
        }
    }

    fn next_seed(&self, agent_id: ZoooidId) -> u64 {
        let t = self.tick.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let low = (agent_id.as_u128() as u64) ^ ((agent_id.as_u128() >> 64) as u64);
        t ^ low.rotate_left(17)
    }
}

impl Default for EvolutionaryOptimizerLearner {
    fn default() -> Self {
        Self::new(0.08)
    }
}

/// Minimal attachable state for non-PID evolutionary demos (e.g. LIF time constant).
#[derive(Debug, Clone)]
pub struct EvolutionaryTuneState {
    pub tau_m: Param,
}

impl OnlineLearner for EvolutionaryOptimizerLearner {
    fn adapt_parameters(
        &self,
        agent_id: ZoooidId,
        state: &mut Box<dyn Any + Send + Sync>,
        role: RoleType,
        feedback: &FeedbackSignal,
    ) {
        let mem = feedback.memory_bias.get(&agent_id).copied().unwrap_or(0.0);
        let err = match (
            feedback
                .per_agent
                .get(&agent_id)
                .or(feedback.global_scalar.as_ref())
                .copied(),
            mem,
        ) {
            (Some(e), m) => e + m,
            (None, m) if m.abs() >= 1e-15 => m,
            _ => return,
        };
        let seed = self.next_seed(agent_id);

        if matches!(role, RoleType::Pid) {
            if let Some(pid) = state.downcast_mut::<PIDzooidState>() {
                pid.pid_params.kp =
                    evolutionary_scalar_update(pid.pid_params.kp, err, seed, self.sigma);
                pid.pid_params.ki = evolutionary_scalar_update(
                    pid.pid_params.ki,
                    err,
                    seed.rotate_left(1),
                    self.sigma * 0.5,
                );
                pid.pid_params.kd = evolutionary_scalar_update(
                    pid.pid_params.kd,
                    err,
                    seed.rotate_left(2),
                    self.sigma * 0.5,
                );
            }
            return;
        }

        if let Some(tune) = state.downcast_mut::<EvolutionaryTuneState>() {
            tune.tau_m = evolutionary_tau_update(tune.tau_m, err, seed, self.sigma);
        }
    }
}

#[cfg(feature = "custom_optim")]
pub struct CustomOptimizerLearner {
    optimizer: Mutex<Optimizer>,
}

#[cfg(feature = "custom_optim")]
impl CustomOptimizerLearner {
    pub fn new(config: OptimizerConfig) -> Result<Self, custom_optim::OptimizerError> {
        Ok(Self {
            optimizer: Mutex::new(Optimizer::new(config)?),
        })
    }
}

#[cfg(feature = "custom_optim")]
impl OnlineLearner for CustomOptimizerLearner {
    fn adapt_parameters(
        &self,
        agent_id: ZoooidId,
        state: &mut Box<dyn Any + Send + Sync>,
        role: RoleType,
        feedback: &FeedbackSignal,
    ) {
        if !matches!(role, RoleType::Pid) {
            return;
        }
        let Some(pid) = state.downcast_mut::<PIDzooidState>() else {
            return;
        };

        let mem = feedback.memory_bias.get(&agent_id).copied().unwrap_or(0.0);
        let err = match (
            feedback
                .per_agent
                .get(&agent_id)
                .or(feedback.global_scalar.as_ref())
                .copied(),
            mem,
        ) {
            (Some(e), m) => e + m,
            (None, m) if m.abs() >= 1e-6 => m,
            _ => return,
        };

        let mut params = [pid.pid_params.kp, pid.pid_params.ki, pid.pid_params.kd];
        let mut optimizer = match self.optimizer.lock() {
            Ok(guard) => guard,
            Err(_) => {
                tracing::error!("Optimizer mutex poisoned in CustomOptimizerLearner");
                return; // Выходим без паники
            }
        };
        let loss_fn = |current: &[f32]| {
            let kp = current[0];
            let ki = current[1];
            let kd = current[2];
            let candidate = kp + ki + kd;
            let target = err;
            (candidate - target) * (candidate - target)
        };
        if optimizer.optimize(&mut params, &loss_fn).is_ok() {
            pid.pid_params.kp = params[0];
            pid.pid_params.ki = params[1];
            pid.pid_params.kd = params[2];
        }
    }
}

#[cfg(feature = "adaptiflux_optim")]
pub struct AdaptiveOptimizerLearner {
    optimizers: Mutex<HashMap<ZoooidId, Adam>>,
    accumulators: Mutex<HashMap<ZoooidId, GradientAccumulator>>,
    lr: f32,
    batch_size: usize,
}

#[cfg(feature = "adaptiflux_optim")]
impl AdaptiveOptimizerLearner {
    pub fn new(lr: f32, batch_size: usize) -> Self {
        Self {
            optimizers: Mutex::new(HashMap::new()),
            accumulators: Mutex::new(HashMap::new()),
            lr,
            batch_size: batch_size.max(1),
        }
    }
}

#[cfg(feature = "adaptiflux_optim")]
impl OnlineLearner for AdaptiveOptimizerLearner {
    fn adapt_parameters(
        &self,
        agent_id: ZoooidId,
        state: &mut Box<dyn Any + Send + Sync>,
        role: RoleType,
        feedback: &FeedbackSignal,
    ) {
        if !matches!(role, RoleType::Pid) {
            return;
        }
        let Some(pid) = state.downcast_mut::<PIDzooidState>() else {
            return;
        };

        let mem = feedback.memory_bias.get(&agent_id).copied().unwrap_or(0.0);
        let err = match (
            feedback
                .per_agent
                .get(&agent_id)
                .or(feedback.global_scalar.as_ref())
                .copied(),
            mem,
        ) {
            (Some(e), m) => e + m,
            (None, m) if m.abs() >= 1e-15 => m,
            _ => return,
        };

        let mut params = [pid.pid_params.kp, pid.pid_params.ki, pid.pid_params.kd];
        let sum = params[0] + params[1] + params[2];
        let gradient = 2.0 * (sum - err);
        let grads = [gradient, gradient, gradient];

        let mut average_gradients: Option<Vec<f32>> = None;
        {
            let mut accumulators = match self.accumulators.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    tracing::error!("Accumulator mutex poisoned in AsyncAdamLearner");
                    return; // Выходим без паники
                }
            };
            let accumulator = accumulators
                .entry(agent_id)
                .or_insert_with(|| GradientAccumulator::new(params.len(), self.batch_size));
            accumulator.accumulate_batch(&grads);
            if let Some(avg) = accumulator.flush() {
                average_gradients = Some(avg.to_vec());
            }
        }

        let Some(avg_grads) = average_gradients else {
            return;
        };

        let mut optimizers = match self.optimizers.lock() {
            Ok(guard) => guard,
            Err(_) => {
                tracing::error!("Optimizer mutex poisoned in AsyncAdamLearner");
                return; // Выходим без паники
            }
        };
        let optimizer = optimizers.entry(agent_id).or_insert_with(|| {
            let mut opt = Adam::new(self.lr);
            opt.init(&mut [0.0f32; 3]);
            opt
        });
        optimizer.step(&mut params, &avg_grads);

        pid.pid_params.kp = params[0];
        pid.pid_params.ki = params[1];
        pid.pid_params.kd = params[2];
    }
}
