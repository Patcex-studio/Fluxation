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

//! # Adaptiflux Core
//!
//! `adaptiflux-core` is the foundational Rust library for the Adaptiflux distributed self-organizing swarm framework.
//! It provides a comprehensive set of modular systems designed to enable adaptive, scalable, and intelligent multi-agent architectures.
//!
//! ## Overview
//!
//! The library implements hybrid computational models that combine neural-inspired plasticity, evolutionary optimization,
//! attention mechanisms, and hierarchical organization. Core features include:
//!
//! - **Self-Organization**: Dynamic topology adaptation through plasticity rules and neurogenesis.
//! - **Learning and Adaptation**: Online learning engines with gradient descent and evolutionary optimization.
//! - **Attention and Memory**: Content-based attention mechanisms and long-term memory storage with retrieval.
//! - **Performance Optimization**: Resource management, sparse execution, and power monitoring.
//! - **Hierarchical Scaling**: Abstraction layers for managing large-scale agent groups.
//!
//! ## Architecture
//!
//! The library is organized into several key modules:
//!
//! - [`core`]: Core scheduling, message passing, and topology management.
//! - [`agent`]: Agent blueprints, roles, and lifecycle management.
//! - [`primitives`]: Basic computational primitives for swarm operations.
//! - [`rules`]: Plasticity rules for topology adaptation (synaptogenesis, pruning, apoptosis, neurogenesis).
//! - [`plasticity`]: Runtime state and context for applying plasticity rules.
//! - [`learning`]: Online adaptation engines and evolutionary learners.
//! - [`memory`]: Long-term storage, retrieval, and attention-weighted memory operations.
//! - [`attention`]: Attention mechanisms for focusing computational resources.
//! - [`hierarchy`]: Hierarchical abstraction and group management.
//! - [`performance`]: Resource management and optimization hooks.
//! - [`power`]: Power monitoring and sleep scheduling.
//! - [`utils`]: Utility types and helper functions.
//! - [`gpu`] (optional): GPU-accelerated operations.
//!
//! ## Key Components
//!
//! ### Core Scheduler
//!
//! The [`CoreScheduler`] is the central orchestrator that manages agent execution, message routing,
//! and applies plasticity rules. It integrates hooks for hierarchy, memory, and online adaptation.
//!
//! ### Agents and Primitives
//!
//! Agents are defined by [`AgentBlueprint`] traits, allowing custom implementations. Primitives provide
//! basic operations like message processing and state updates.
//!
//! ### Plasticity and Rules
//!
//! Plasticity enables the system to adapt its topology dynamically. Rules such as [`SynapticPruningRule`],
//! [`ActivityDependentSynaptogenesisRule`], and [`HomeostaticApoptosisRule`] govern these adaptations.
//!
//! ### Learning
//!
//! The [`OnlineAdaptationEngine`] supports continuous learning through feedback signals and optimization.
//! Evolutionary learners like [`EvolutionaryOptimizerLearner`] provide population-based adaptation.
//!
//! ### Memory and Attention
//!
//! Memory systems store experiences with embeddings and support similarity-based retrieval.
//! Attention mechanisms like [`ContentBasedAttention`] focus processing on relevant information.
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use adaptiflux_core::{CoreScheduler, AgentBlueprint, ZoooidHandle};
//!
//! // Create a scheduler
//! let mut scheduler = CoreScheduler::new();
//!
//! // Define an agent blueprint
//! struct MyAgent;
//! impl AgentBlueprint for MyAgent {
//!     // Implement required methods...
//! }
//!
//! // Add agents to the scheduler
//! let handle = scheduler.add_agent(Box::new(MyAgent)).unwrap();
//!
//! // Run the scheduler
//! scheduler.run().unwrap();
//! ```
//!
//! For more detailed examples, see the `examples/` directory.
//!
//! ## Features
//!
//! - `gpu`: Enables GPU-accelerated operations.
//! - `adaptiflux_optim`: Integrates with the adaptiflux-optim optimization backend.
//! - `custom_optim`: Enables custom CUDA-based optimization.
//!
//! ## Dependencies
//!
//! The library depends on several crates for concurrency, data structures, and optimization.
//! See `Cargo.toml` for the full list.

pub mod agent;
pub mod attention;
pub mod core;
pub mod hierarchy;
pub mod hybrids;
pub mod learning;
pub mod memory;
pub mod performance;
pub mod plasticity;
pub mod power;
pub mod primitives;
pub mod rules;
pub mod utils;

#[cfg(feature = "gpu")]
pub mod gpu;

pub use crate::agent::{AgentBlueprint, AgentUpdateResult, RoleType, Zoooid};
pub use crate::core::message_bus::{LocalBus, Message, MessageBus};
pub use crate::core::topology::{
    ConnectionProperties, SystemMetrics, TopologyChange, ZoooidTopology,
};
pub use crate::core::{
    CoreScheduler, HierarchyHook, MemoryAttentionHook, OnlineAdaptationHook, ResourceManager,
    SchedulerMetrics, ZoooidHandle,
};
pub use crate::hierarchy::{
    detect_dense_groups, AbstractionLayerManager, AgentGroupAbstraction, AggregationFnKind,
};
pub use crate::learning::{
    collect_error_feedback_from_bus, merge_from_posted_errors, EvolutionaryOptimizerLearner,
    EvolutionaryTuneState, FeedbackSignal, GradientDescentLearner, OnlineAdaptationEngine,
    OnlineLearner,
};
pub use crate::plasticity::{
    AppliedTopologyEffects, BlueprintFactory, PlasticityContext, PlasticityRuntimeState,
};

pub use crate::attention::{
    apply_attention_to_hits, keys_values_from_hits, AttentionKey, AttentionMechanism,
    AttentionValue, ContentBasedAttention, DotProductAttention, ErrorSimilarityFocus,
    FocusScheduler, HardAttentionSelector, PheromoneFocus,
};
#[cfg(feature = "adaptiflux_optim")]
pub use crate::learning::AdaptiveOptimizerLearner;
#[cfg(feature = "custom_optim")]
pub use crate::learning::CustomOptimizerLearner;
pub use crate::memory::{
    agent_similarity_query, build_weighted_payload, content_attention_weights, cosine_similarity,
    reindex_table, retrieve_and_weight, simple_situation_embedding, store_scalar_experience,
    summarize_step_inputs, ExperienceRecorder, GraphLongTermStore, KeyAndScore, LongTermStore,
    MemoryEntryPayload, MemoryKey, MemoryPayload, MemorySummary, Metadata, MetadataIndexer, Query,
    Retriever, TableLongTermStore,
};
pub use crate::performance::async_optimization::AsyncOptimizationConfig;
pub use crate::performance::resource_manager::{AgentResourceProfile, ResourceManagerPolicy};
pub use crate::performance::sparse_execution::SparseExecutionHook;
pub use crate::power::power_monitor::{PowerMetrics, PowerMonitor};
pub use crate::power::sleep_scheduler::SleepScheduler;
pub use crate::rules::{
    ActivityDependentSynaptogenesisRule, ClusterGroupingPlasticityRule,
    GrowthFactorNeurogenesisRule, HomeostaticApoptosisRule, PlasticityRule, PruneReason,
    RuleEngine, SynapticPruningRule, TopologyAction, TopologyRule,
};
pub use crate::utils::types::ZoooidId;
