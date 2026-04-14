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

use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::{sleep, Duration};

/// Event-driven execution helper that waits for either a scheduler tick or an external notification.
#[derive(Clone)]
pub struct SparseExecutionHook {
    pub wake_notify: Arc<Notify>,
    pub poll_interval: Duration,
}

impl SparseExecutionHook {
    pub fn new(poll_interval: Duration) -> Self {
        Self {
            wake_notify: Arc::new(Notify::new()),
            poll_interval,
        }
    }

    pub fn notifier(&self) -> Arc<Notify> {
        self.wake_notify.clone()
    }

    pub async fn wait_for_next_cycle(&self, external_notify: Option<Arc<Notify>>) {
        if let Some(external) = external_notify {
            tokio::select! {
                _ = sleep(self.poll_interval) => {},
                _ = external.notified() => {},
                _ = self.wake_notify.notified() => {},
            }
        } else {
            tokio::select! {
                _ = sleep(self.poll_interval) => {},
                _ = self.wake_notify.notified() => {},
            }
        }
    }

    pub fn wake(&self) {
        self.wake_notify.notify_one();
    }
}
