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

/// Metadata for tracking buffer state and synchronization
#[derive(Debug, Clone)]
pub struct BufferMetadata {
    pub usage: BufferUsages,
    pub size: u64,
    pub is_dirty: bool,
    pub last_update: std::time::Instant,
}

pub struct BufferManager {
    buffers: HashMap<String, Buffer>,
    metadata: HashMap<String, BufferMetadata>,
    device: Arc<Device>,
    queue: Arc<Queue>,
}

impl BufferManager {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        Self {
            buffers: HashMap::new(),
            metadata: HashMap::new(),
            device,
            queue,
        }
    }

    /// Create a buffer with specified usage flags
    pub fn create_buffer(
        &mut self,
        name: &str,
        size: u64,
        usage: BufferUsages,
    ) -> Result<(), String> {
        let max_size = self.device.limits().max_buffer_size;
        if size > max_size {
            return Err(format!(
                "Buffer '{}' size {} exceeds device max buffer size {}",
                name, size, max_size
            ));
        }

        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some(name),
            size,
            usage,
            mapped_at_creation: false,
        });

        self.buffers.insert(name.to_string(), buffer);
        self.metadata.insert(
            name.to_string(),
            BufferMetadata {
                usage,
                size,
                is_dirty: false,
                last_update: std::time::Instant::now(),
            },
        );

        Ok(())
    }

    /// Create a storage buffer (read/write)
    pub fn create_storage_buffer(&mut self, name: &str, size: u64) -> Result<(), String> {
        self.create_buffer(
            name,
            size,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        )
    }

    /// Create a uniform buffer (read-only)
    pub fn create_uniform_buffer(&mut self, name: &str, size: u64) -> Result<(), String> {
        self.create_buffer(
            name,
            size,
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        )
    }

    /// Create a copy source buffer (for readback)
    pub fn create_copy_buffer(&mut self, name: &str, size: u64) -> Result<(), String> {
        self.create_buffer(name, size, BufferUsages::COPY_SRC | BufferUsages::COPY_DST)
    }

    /// Get a reference to a buffer
    pub fn get_buffer(&self, name: &str) -> Option<&Buffer> {
        self.buffers.get(name)
    }

    /// Write data to a buffer (full or partial)
    pub fn write_buffer(&mut self, name: &str, offset: u64, data: &[u8]) -> Result<(), String> {
        if let Some(buffer) = self.buffers.get(name) {
            self.queue.write_buffer(buffer, offset, data);

            // Mark as dirty for tracking
            if let Some(meta) = self.metadata.get_mut(name) {
                meta.is_dirty = true;
                meta.last_update = std::time::Instant::now();
            }

            Ok(())
        } else {
            Err(format!("Buffer '{}' not found", name))
        }
    }

    /// Perform incremental buffer update - marks changed region for optimization
    pub fn write_partial_buffer(
        &mut self,
        name: &str,
        offset: u64,
        data: &[u8],
    ) -> Result<(), String> {
        self.write_buffer(name, offset, data)
    }

    /// Read buffer content (requires mapping on CPU)
    pub fn get_buffer_mut(&mut self, name: &str) -> Option<&mut Buffer> {
        self.buffers.get_mut(name)
    }

    /// Remove a buffer
    pub fn remove_buffer(&mut self, name: &str) -> Option<Buffer> {
        self.metadata.remove(name);
        self.buffers.remove(name)
    }

    /// Check if buffer is marked as dirty
    pub fn is_dirty(&self, name: &str) -> bool {
        self.metadata
            .get(name)
            .map(|m| m.is_dirty)
            .unwrap_or(false)
    }

    /// Clear dirty flag after synchronization
    pub fn clear_dirty(&mut self, name: &str) {
        if let Some(meta) = self.metadata.get_mut(name) {
            meta.is_dirty = false;
        }
    }

    /// Get buffer metadata
    pub fn get_metadata(&self, name: &str) -> Option<&BufferMetadata> {
        self.metadata.get(name)
    }

    /// Clear all buffers
    pub fn clear_all(&mut self) {
        self.buffers.clear();
        self.metadata.clear();
    }
}

impl Drop for BufferManager {
    fn drop(&mut self) {
        // All buffers are automatically freed when dropped due to RAII
        // wgpu handles cleanup
        self.clear_all();
    }
}
