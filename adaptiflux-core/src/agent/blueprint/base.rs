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

use async_trait::async_trait;
use std::any::Any;

use crate::agent::state::{AgentUpdateResult, RoleType};
use crate::core::message_bus::message::Message;
use crate::core::topology::ZoooidTopology;
use crate::memory::types::MemoryPayload;
use crate::utils::types::ZoooidId;

/// Core trait for defining agent blueprints in the Adaptiflux system
///
/// An `AgentBlueprint` defines the behavior and lifecycle of an agent (zoooid) in the swarm.
/// Implementations provide initialization, update logic, and metadata about the agent's capabilities.
///
/// # Lifecycle
///
/// 1. **Initialization**: `initialize()` is called once when the agent is spawned
/// 2. **Update Loop**: `update()` is called repeatedly during scheduler iterations
/// 3. **Termination**: Handled by the scheduler when topology rules dictate
///
/// # Type Safety
///
/// Agent state is stored as `Box<dyn Any + Send + Sync>` to allow flexible implementations
/// while maintaining thread safety. Downcast to concrete types as needed.
///
/// # Examples
///
/// ```rust,no_run
/// use async_trait::async_trait;
/// use adaptiflux_core::{AgentBlueprint, AgentUpdateResult, RoleType, Message, ZoooidTopology};
/// use std::any::Any;
///
/// struct MyAgent;
///
/// #[async_trait]
/// impl AgentBlueprint for MyAgent {
///     async fn initialize(&self) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
///         Ok(Box::new(MyState { counter: 0 }))
///     }
///
///     async fn update(
///         &self,
///         state: &mut Box<dyn Any + Send + Sync>,
///         inputs: Vec<Message>,
///         _topology: &ZoooidTopology,
///         _memory: Option<&adaptiflux_core::MemoryPayload>,
///     ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>> {
///         // Update logic here
///         Ok(AgentUpdateResult::NoChange)
///     }
///
///     fn blueprint_type(&self) -> RoleType {
///         RoleType::Worker
///     }
/// }
///
/// #[derive(Debug)]
/// struct MyState {
///     counter: u32,
/// }
/// ```
#[async_trait]
pub trait AgentBlueprint: Send + Sync {
    /// Initialize the agent's state
    ///
    /// Called once when the agent is first spawned into the system.
    /// Use this to set up initial state, allocate resources, or perform setup operations.
    ///
    /// # Returns
    ///
    /// Returns the initial state wrapped in a `Box<dyn Any + Send + Sync>`, or an error if initialization fails
    async fn initialize(
        &self,
    ) -> Result<Box<dyn Any + Send + Sync>, Box<dyn std::error::Error + Send + Sync>>;

    /// Update the agent's state based on inputs and context
    ///
    /// Called on every scheduler iteration. Process incoming messages, update internal state,
    /// and return any outgoing messages or topology changes.
    ///
    /// # Arguments
    ///
    /// * `state` - Mutable reference to the agent's current state
    /// * `inputs` - Vector of incoming messages from other agents, with sender IDs
    /// * `topology` - Current system topology for spatial/contextual awareness
    /// * `memory` - Optional memory payload for attention/memory integration
    ///
    /// # Returns
    ///
    /// Returns an `AgentUpdateResult` containing output messages, topology changes, and update flags
    async fn update(
        &self,
        state: &mut Box<dyn Any + Send + Sync>,
        inputs: Vec<(ZoooidId, Message)>,
        topology: &ZoooidTopology,
        memory: Option<&MemoryPayload>,
    ) -> Result<AgentUpdateResult, Box<dyn std::error::Error + Send + Sync>>;

    /// Get the blueprint's role type for system categorization
    ///
    /// Used by the scheduler for agent classification and rule application.
    ///
    /// # Returns
    ///
    /// The `RoleType` enum value representing this agent's functional role
    fn blueprint_type(&self) -> RoleType;

    /// Check if this blueprint supports GPU acceleration
    ///
    /// Defaults to `false`. Override to `true` if the agent can utilize GPU resources.
    ///
    /// # Returns
    ///
    /// `true` if GPU acceleration is supported, `false` otherwise
    fn supports_gpu(&self) -> bool {
        false
    }
}
