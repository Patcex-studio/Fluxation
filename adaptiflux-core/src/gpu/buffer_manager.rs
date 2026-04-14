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
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

pub struct BufferManager {
    buffers: HashMap<String, Buffer>,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl BufferManager {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            buffers: HashMap::new(),
            device,
            queue,
        }
    }

    pub fn create_buffer(&mut self, name: &str, size: u64, usage: BufferUsages) -> &Buffer {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some(name),
            size,
            usage,
            mapped_at_creation: false,
        });
        self.buffers.entry(name.to_string()).or_insert(buffer)
    }

    pub fn get_buffer(&self, name: &str) -> Option<&Buffer> {
        self.buffers.get(name)
    }

    pub fn write_buffer(&self, name: &str, offset: u64, data: &[u8]) {
        if let Some(buffer) = self.buffers.get(name) {
            self.queue.write_buffer(buffer, offset, data);
        }
    }

    pub fn read_buffer(&self, name: &str) -> Option<&Buffer> {
        self.buffers.get(name)
    }

    pub fn remove_buffer(&mut self, name: &str) -> Option<Buffer> {
        self.buffers.remove(name)
    }
}

impl Drop for BufferManager {
    fn drop(&mut self) {
        // Все буферы автоматически освобождаются при удалении BufferManager.
        // wgpu управляет временем жизни буферов через RAII,
        // поэтому здесь достаточно оставить пустой деструктор.
    }
}
