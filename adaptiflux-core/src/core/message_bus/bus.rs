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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::{self, Receiver, Sender};
use tokio::sync::{Mutex, Notify};
use tracing::trace;

use crate::core::message_bus::message::Message;
use crate::utils::types::ZoooidId;
use async_trait::async_trait;

#[derive(Debug)]
pub struct SendError;

#[derive(Debug)]
pub struct RecvError;

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SendError")
    }
}

impl std::error::Error for SendError {}

impl std::fmt::Display for RecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RecvError")
    }
}

impl std::error::Error for RecvError {}

#[async_trait]
pub trait MessageBus: Send + Sync {
    async fn register_agent(&self, id: ZoooidId) -> Result<(), SendError>;
    async fn send(&self, from: ZoooidId, to: ZoooidId, message: Message) -> Result<(), SendError>;
    async fn broadcast(
        &self,
        from: ZoooidId,
        targets: &[ZoooidId],
        message: Message,
    ) -> Result<(), SendError>;
    async fn receive(&self, id: ZoooidId) -> Result<Vec<(ZoooidId, Message)>, RecvError>;

    fn notifier(&self) -> Option<Arc<Notify>>;
}

#[derive(Clone)]
pub struct LocalBus {
    senders: Arc<Mutex<HashMap<ZoooidId, Sender<(ZoooidId, Message)>>>>,
    receivers: Arc<Mutex<HashMap<ZoooidId, Receiver<(ZoooidId, Message)>>>>,
    notify: Arc<Notify>,
}

impl LocalBus {
    pub fn new() -> Self {
        Self {
            senders: Arc::new(Mutex::new(HashMap::new())),
            receivers: Arc::new(Mutex::new(HashMap::new())),
            notify: Arc::new(Notify::new()),
        }
    }

    async fn create_channel(&self, id: ZoooidId) {
        let mut senders = self.senders.lock().await;
        let mut receivers = self.receivers.lock().await;

        if let std::collections::hash_map::Entry::Vacant(e) = senders.entry(id) {
            let (tx, rx) = mpsc::channel(128);
            e.insert(tx);
            receivers.insert(id, rx);
        }
    }
}

impl Default for LocalBus {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MessageBus for LocalBus {
    async fn register_agent(&self, id: ZoooidId) -> Result<(), SendError> {
        self.create_channel(id).await;
        Ok(())
    }

    async fn send(&self, from: ZoooidId, to: ZoooidId, message: Message) -> Result<(), SendError> {
        let sender_opt = {
            let senders = self.senders.lock().await;
            senders.get(&to).cloned()
        };

        if let Some(sender) = sender_opt {
            let send_start = Instant::now();
            let result = sender.send((from, message)).await.map_err(|_| SendError);
            let duration_ms = send_start.elapsed().as_secs_f64() * 1000.0;
            trace!(to = ?to, duration_ms, "LocalBus send completed");
            if result.is_ok() {
                self.notify.notify_one();
            }
            result
        } else {
            trace!(to = ?to, "LocalBus send failed: receiver not registered");
            Err(SendError)
        }
    }

    async fn broadcast(
        &self,
        from: ZoooidId,
        targets: &[ZoooidId],
        message: Message,
    ) -> Result<(), SendError> {
        // Fast path: empty target list
        if targets.is_empty() {
            return Ok(());
        }

        // Collect senders for all targets in one lock cycle
        let senders = self.senders.lock().await;
        let mut targets_to_send = Vec::with_capacity(targets.len());
        for target in targets {
            if let Some(sender) = senders.get(target) {
                targets_to_send.push(sender.clone());
            }
        }
        drop(senders);

        trace!(target_count = targets_to_send.len(), "LocalBus broadcast");

        // Send to all targets sequentially (already optimized by holding mutex only once)
        for sender in targets_to_send {
            sender.send((from, message.clone())).await.map_err(|_| SendError)?;
        }

        self.notify.notify_one();
        Ok(())
    }

    async fn receive(&self, id: ZoooidId) -> Result<Vec<(ZoooidId, Message)>, RecvError> {
        let mut receivers = self.receivers.lock().await;
        if let Some(receiver) = receivers.get_mut(&id) {
            let mut messages = Vec::new();
            let initial_len = receiver.len();
            while let Ok(msg) = receiver.try_recv() {
                messages.push(msg);
            }
            trace!(id = ?id, initial_queue_len = initial_len, received = messages.len(), "LocalBus receive");
            return Ok(messages);
        }

        Err(RecvError)
    }

    fn notifier(&self) -> Option<Arc<Notify>> {
        Some(self.notify.clone())
    }
}
