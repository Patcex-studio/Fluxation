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

//! GPU compute shaders for Fluxation agent operations
//!
//! This module contains WGSL (WebGPU Shading Language) kernels for:
//! - Agent state updates (LIF, Izhikevich, PID)
//! - Connection strength calculations
//! - Plasticity rules (pruning, synaptogenesis)
//! - Hormone/neuromodulator simulations

/// Agent state data structure (shared across shaders)
/// Each agent has voltage (v), recovery (u), input current (I), spike flag
pub const AGENT_UPDATE_SHADER: &str = r#"
struct Agent {
    v: f32,           // Membrane potential
    u: f32,           // Recovery variable (for Izhikevich)
    current_in: f32,  // Input current this step
    spike: u32,       // Spike flag (0 or 1)
}

struct AgentParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    dt: f32,
    threshold: f32,
}

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var<storage, read> params: array<AgentParams>;
@group(0) @binding(2) var<uniform> num_agents: u32;

@compute @workgroup_size(256)
fn update_agents(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= num_agents) {
        return;
    }

    var agent = agents[idx];
    let param = params[idx];
    
    // Izhikevich neuron model update
    let dv = 0.04 * agent.v * agent.v + 5.0 * agent.v + 140.0 - agent.u + agent.current_in;
    let du = param.a * (param.b * agent.v - agent.u);
    
    agent.v = agent.v + param.dt * dv;
    agent.u = agent.u + param.dt * du;
    
    // Spike detection and reset
    if (agent.v >= param.threshold) {
        agent.spike = 1u;
        agent.v = param.c;
        agent.u = agent.u + param.d;
    } else {
        agent.spike = 0u;
    }
    
    // Clear input for next step
    agent.current_in = 0.0;
    
    agents[idx] = agent;
}
"#;

/// Calculate connection strengths based on pre/post-synaptic activity
pub const CONNECTION_CALCULATE_SHADER: &str = r#"
struct Agent {
    v: f32,
    u: f32,
    current_in: f32,
    spike: u32,
}

struct Edge {
    strength: f32,        // Synaptic weight
    pre_activity: f32,    // Presynaptic activity trace
    post_activity: f32,   // Postsynaptic activity trace
}

struct ConnectionParams {
    learning_rate: f32,
    decay_rate: f32,
    max_weight: f32,
}

@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<storage, read_write> edges: array<Edge>;
@group(0) @binding(2) var<uniform> params: ConnectionParams;
@group(0) @binding(3) var<uniform> num_edges: u32;
@group(0) @binding(4) var<storage, read> edge_sources: array<u32>;    // Source agent indices
@group(0) @binding(5) var<storage, read> edge_targets: array<u32>;    // Target agent indices

@compute @workgroup_size(256)
fn calculate_connections(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= num_edges) {
        return;
    }

    let source_idx = edge_sources[idx];
    let target_idx = edge_targets[idx];
    
    let source_agent = agents[source_idx];
    let target_agent = agents[target_idx];
    
    var edge = edges[idx];
    
    // Update activity traces with STDP (Spike-Timing-Dependent Plasticity)
    edge.pre_activity = edge.pre_activity * params.decay_rate + f32(source_agent.spike);
    edge.post_activity = edge.post_activity * params.decay_rate + f32(target_agent.spike);
    
    // STDP: strengthen if pre fires before post
    let corr = edge.pre_activity * edge.post_activity;
    edge.strength = edge.strength + params.learning_rate * corr;
    
    // Clip weight to valid range [0, max_weight]
    edge.strength = max(0.0, min(edge.strength, params.max_weight));
    
    edges[idx] = edge;
}
"#;

/// Pruning shader: mark weak connections for removal
pub const PLASTICITY_PRUNING_SHADER: &str = r#"
struct Edge {
    strength: f32,
    pre_activity: f32,
    post_activity: f32,
}

struct PruningParams {
    pruning_threshold: f32,   // Connections below this are pruned
    activity_threshold: f32,  // Edges with low activity are candidates
}

@group(0) @binding(0) var<storage, read_write> edges: array<Edge>;
@group(0) @binding(1) var<storage, read_write> prune_flags: array<u32>;  // 1 = prune, 0 = keep
@group(0) @binding(2) var<uniform> params: PruningParams;
@group(0) @binding(3) var<uniform> num_edges: u32;

@compute @workgroup_size(256)
fn prune_connections(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= num_edges) {
        return;
    }

    let edge = edges[idx];
    
    // Prune if weight is low and activity is low
    let total_activity = edge.pre_activity + edge.post_activity;
    if (edge.strength < params.pruning_threshold && total_activity < params.activity_threshold) {
        prune_flags[idx] = 1u;
    } else {
        prune_flags[idx] = 0u;
    }
}
"#;

/// Synaptogenesis shader: create new connections based on activity
pub const PLASTICITY_SYNAPTOGENESIS_SHADER: &str = r#"
struct Agent {
    v: f32,
    u: f32,
    current_in: f32,
    spike: u32,
}

struct SynaptogenesisParams {
    growth_threshold: f32,     // Correlated activity above this triggers synaptogenesis
    initial_weight: f32,       // Initial weight for new connections
    max_synapses_per_agent: u32,
}

@group(0) @binding(0) var<storage, read> agents: array<Agent>;
@group(0) @binding(1) var<storage, read_write> synapse_create_count: atomic<u32>;
@group(0) @binding(2) var<uniform> params: SynaptogenesisParams;
@group(0) @binding(3) var<uniform> num_agents: u32;

@compute @workgroup_size(256)
fn create_synapses(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= num_agents) {
        return;
    }

    let agent = agents[idx];
    
    // Simple heuristic: agents with high voltage + recent spike are candidates for new connections
    if (agent.v > -40.0 && agent.spike == 1u) {
        let _ = atomicAdd(&synapse_create_count, 1u);
    }
}
"#;

/// Hormone/neuromodulator simulation shader
/// Global signals affecting network behavior (dopamine, cortisol, etc.)
pub const HORMONE_SIMULATION_SHADER: &str = r#"
struct HormoneParams {
    baseline_dopamine: f32,
    baseline_cortisol: f32,
    stress_sensitivity: f32,
    reward_sensitivity: f32,
}

struct HormoneState {
    dopamine: f32,
    cortisol: f32,
    adrenaline: f32,
    decay_rate: f32,
}

@group(0) @binding(0) var<storage, read_write> hormone_state: HormoneState;
@group(0) @binding(1) var<uniform> params: HormoneParams;
@group(0) @binding(2) var<uniform> network_error: f32;       // Stress signal (error magnitude)
@group(0) @binding(3) var<uniform> network_reward: f32;      // Reward signal

@compute @workgroup_size(1)
fn update_hormones() {
    var state = hormone_state;
    
    // Stress increases cortisol
    state.cortisol = state.cortisol * state.decay_rate + 
                     params.baseline_cortisol + 
                     params.stress_sensitivity * network_error;
    
    // Reward increases dopamine
    state.dopamine = state.dopamine * state.decay_rate + 
                     params.baseline_dopamine + 
                     params.reward_sensitivity * network_reward;
    
    // Adrenaline responds to cortisol
    state.adrenaline = state.adrenaline * state.decay_rate + state.cortisol * 0.5;
    
    // Clamp to reasonable ranges [0, 10]
    state.dopamine = max(0.0, min(state.dopamine, 10.0));
    state.cortisol = max(0.0, min(state.cortisol, 10.0));
    state.adrenaline = max(0.0, min(state.adrenaline, 10.0));
    
    hormone_state = state;
}
"#;

/// LIF (Leaky Integrate-and-Fire) neuron shader - simpler than Izhikevich
pub const LIF_UPDATE_SHADER: &str = r#"
struct LifAgent {
    v: f32,           // Membrane potential
    threshold: f32,   // Spike threshold
    current_in: f32,  // Input current
    spike: u32,       // Spike flag
}

struct LifParams {
    leak_rate: f32,   // Exponential decay rate
    reset_v: f32,     // Reset potential after spike
    dt: f32,
}

@group(0) @binding(0) var<storage, read_write> agents: array<LifAgent>;
@group(0) @binding(1) var<storage, read> params: array<LifParams>;
@group(0) @binding(2) var<uniform> num_agents: u32;

@compute @workgroup_size(256)
fn update_lif_neurons(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= num_agents) {
        return;
    }

    var agent = agents[idx];
    let param = params[idx];
    
    // Leaky integration: dv/dt = -leak*(v-0) + I
    agent.v = agent.v * param.leak_rate + agent.current_in * param.dt;
    
    // Spike detection and reset
    if (agent.v >= agent.threshold) {
        agent.spike = 1u;
        agent.v = param.reset_v;
    } else {
        agent.spike = 0u;
    }
    
    agent.current_in = 0.0;
    agents[idx] = agent;
}
"#;
