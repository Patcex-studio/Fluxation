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

//! Topology event system for coordinating synapse updates across agents and topology.
//!
//! `TopologyEventBus` provides an event-driven architecture for:
//! - Notifying agents when edges are added/removed/updated
//! - Automatic synchronization of `SynapseManager` state
//! - "At-least-once" delivery semantics via `tokio::sync::broadcast`

use crate::utils::types::ZoooidId;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::debug;

/// Events emitted by the topology system.
///
/// Agents with `SynapseManager` subscribe to these events to maintain consistency
/// between the topology graph and synaptic weight state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyEvent {
    /// New edge added: from -> to with initial weight
    EdgeAdded {
        from: ZoooidId,
        to: ZoooidId,
        initial_weight: f32,
    },

    /// Edge removed: from -> to
    EdgeRemoved { from: ZoooidId, to: ZoooidId },

    /// Weight updated on existing edge
    WeightUpdated {
        from: ZoooidId,
        to: ZoooidId,
        new_weight: f32,
    },

    /// Snapshot of all current edges (for synchronization/recovery)
    TopologySnapshot {
        edges: Vec<(ZoooidId, ZoooidId, f32)>,
    },
}

impl TopologyEvent {
    /// Returns true if this event targets the given agent as the destination.
    pub fn targets(&self, agent_id: ZoooidId) -> bool {
        match self {
            TopologyEvent::EdgeAdded { to, .. } => *to == agent_id,
            TopologyEvent::EdgeRemoved { to, .. } => *to == agent_id,
            TopologyEvent::WeightUpdated { to, .. } => *to == agent_id,
            TopologyEvent::TopologySnapshot { .. } => true, // Snapshot is for synchronization
        }
    }

    /// Returns the source agent ID if this is a point-to-point event.
    pub fn source(&self) -> Option<ZoooidId> {
        match self {
            TopologyEvent::EdgeAdded { from, .. } => Some(*from),
            TopologyEvent::EdgeRemoved { from, .. } => Some(*from),
            TopologyEvent::WeightUpdated { from, .. } => Some(*from),
            TopologyEvent::TopologySnapshot { .. } => None,
        }
    }
}

/// Broadcast channel for topology events.
///
/// Provides "at-least-once" delivery with configurable buffer size.
/// Subscribers that fall behind (buffer wraps) will lose events but can re-sync
/// by requesting a `TopologySnapshot`.
#[derive(Clone)]
pub struct TopologyEventBus {
    tx: Arc<broadcast::Sender<TopologyEvent>>,
    buffer_size: usize,
}

impl TopologyEventBus {
    /// Creates a new `TopologyEventBus` with specified buffer size.
    ///
    /// # Arguments
    /// * `buffer_size` - Max events in broadcast buffer (default ~1024)
    ///
    /// # Panics
    /// If `buffer_size == 0`
    pub fn new(buffer_size: usize) -> Self {
        if buffer_size == 0 {
            panic!("TopologyEventBus buffer_size must be > 0");
        }
        let (tx, _) = broadcast::channel(buffer_size);
        Self {
            tx: Arc::new(tx),
            buffer_size,
        }
    }

    /// Creates a new subscriber to topology events.
    ///
    /// Subscribers receive events asynchronously via `recv()`.
    /// If the buffer overflows, subscribers must handle `broadcast::error::RecvError::Lagged`.
    pub fn subscribe(&self) -> broadcast::Receiver<TopologyEvent> {
        self.tx.subscribe()
    }

    /// Publishes an event to all subscribers.
    ///
    /// Returns the number of subscribers that received the event.
    /// Returns an error if all receivers have been dropped (no subscribers).
    pub fn publish(&self, event: TopologyEvent) -> Result<usize, broadcast::error::SendError<TopologyEvent>> {
        debug!("Publishing topology event: {:?}", event);
        self.tx.send(event)
    }

    /// Returns the number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.tx.receiver_count()
    }

    /// Returns the buffer size of this event bus.
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
}

impl std::fmt::Debug for TopologyEventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopologyEventBus")
            .field("buffer_size", &self.buffer_size)
            .field("subscriber_count", &self.subscriber_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_id(n: u8) -> ZoooidId {
        use uuid::Uuid;
        Uuid::new_v5(&Uuid::NAMESPACE_DNS, &[n])
    }

    #[tokio::test]
    async fn test_event_bus_publish_subscribe() {
        let bus = TopologyEventBus::new(16);
        let mut rx = bus.subscribe();

        let from = test_id(1);
        let to = test_id(2);

        let event = TopologyEvent::EdgeAdded {
            from,
            to,
            initial_weight: 0.5,
        };

        let count = bus.publish(event.clone()).expect("publish failed");
        assert_eq!(count, 1); // One subscriber

        let received = rx.recv().await.expect("recv failed");
        assert!(matches!(received, TopologyEvent::EdgeAdded { .. }));
    }

    #[tokio::test]
    async fn test_event_targets() {
        let from = test_id(1);
        let to = test_id(2);

        let event = TopologyEvent::EdgeAdded {
            from,
            to,
            initial_weight: 0.5,
        };

        assert!(event.targets(to));
        assert!(!event.targets(from));
        assert!(!event.targets(test_id(3)));
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let bus = TopologyEventBus::new(16);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        let event = TopologyEvent::EdgeRemoved {
            from: test_id(1),
            to: test_id(2),
        };

        let count = bus.publish(event.clone()).expect("publish failed");
        assert_eq!(count, 2);

        let received1 = rx1.recv().await.expect("rx1 failed");
        let received2 = rx2.recv().await.expect("rx2 failed");

        assert!(matches!(received1, TopologyEvent::EdgeRemoved { .. }));
        assert!(matches!(received2, TopologyEvent::EdgeRemoved { .. }));
    }

    #[tokio::test]
    async fn test_snapshot_event() {
        let bus = TopologyEventBus::new(16);
        let mut rx = bus.subscribe();

        let edges = vec![
            (test_id(1), test_id(2), 0.5),
            (test_id(2), test_id(3), 0.7),
        ];

        let event = TopologyEvent::TopologySnapshot {
            edges: edges.clone(),
        };

        bus.publish(event).expect("publish failed");
        let received = rx.recv().await.expect("recv failed");

        match received {
            TopologyEvent::TopologySnapshot { edges: received_edges } => {
                assert_eq!(received_edges, edges);
            }
            _ => panic!("expected TopologySnapshot"),
        }
    }
}
