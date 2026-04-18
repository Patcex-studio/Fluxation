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
use serde::{Serialize, Deserialize};

/// Information about the accelerator backend's capabilities and properties
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Human-readable name of the backend (e.g. "Apple Metal GPU M1")
    pub name: String,
    /// Type of accelerator (CPU, GPU Metal, Vulkan, CUDA, WASM SIMD, etc.)
    pub backend_type: AcceleratorType,
    /// Total available memory in bytes (or estimate for CPU)
    pub memory_bytes: u64,
    /// Number of compute units (cores, work units, etc.)
    /// For GPU: number of compute cores or work units
    /// For CPU: number of threads typically
    pub compute_units: Option<u32>,
    /// Whether this is a unified memory architecture (e.g. Apple Silicon)
    pub is_unified_memory: bool,
    /// Backend version or driver info
    pub version: String,
}

/// Supported accelerator types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcceleratorType {
    /// CPU-only computation (fallback, always available)
    Cpu,
    /// GPU with Apple Metal backend (Apple Silicon / macOS)
    GpuMetal,
    /// GPU with Vulkan backend (Linux / Windows)
    GpuVulkan,
    /// GPU with DirectX 12 backend (Windows)
    GpuDx12,
    /// NVIDIA CUDA compute
    Cuda,
    /// WebAssembly SIMD
    WasmSimd,
}

impl std::fmt::Display for AcceleratorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::GpuMetal => write!(f, "GPU (Metal)"),
            Self::GpuVulkan => write!(f, "GPU (Vulkan)"),
            Self::GpuDx12 => write!(f, "GPU (DirectX 12)"),
            Self::Cuda => write!(f, "GPU (CUDA)"),
            Self::WasmSimd => write!(f, "WASM SIMD"),
        }
    }
}

/// Universal trait for accelerator backends.
///
/// Provides abstraction over different acceleration hardware types (CPU, GPU Metal, Vulkan, CUDA, etc.).
/// Each concrete backend implementation must implement this trait to be compatible with CoreScheduler
/// and other components in the Adaptiflux framework.
///
/// # Design Principles
///
/// - **Decoupling**: Core logic doesn't depend on specific backend implementations
/// - **Swappability**: Backends can be switched at runtime or compile time
/// - **Fallback Safety**: Each backend should support graceful degradation
/// - **Async-first**: GPU operations are inherently asynchronous
///
/// # Example
///
/// ```ignore
/// // Create a GPU backend
/// let backend = Arc::new(GpuBackendWgpu::new() as Box<dyn AcceleratorBackend>);
///
/// // Execute computation
/// backend.initialize().await?;
/// backend.execute_compute("agent_update", &args).await?;
/// let results = backend.download_data("results").await?;
/// ```
#[async_trait]
pub trait AcceleratorBackend: Send + Sync {
    /// Initialize the accelerator backend.
    ///
    /// This is called once at startup to set up resources, allocate memory,
    /// compile shaders, etc. Subsequent calls should be idempotent (no-op if already initialized).
    ///
    /// # Returns
    ///
    /// - `Ok(())` on success
    /// - `Err` if initialization fails (device unavailable, compilation errors, etc.)
    async fn initialize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Shutdown the accelerator and release resources.
    ///
    /// Should be called when the accelerator is no longer needed or before switching backends.
    /// Idempotent: safe to call multiple times.
    async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Execute a compute shader or kernel with the given name and arguments.
    ///
    /// The shader must be registered with the backend (via shaders module or configuration).
    /// Arguments are passed as `&dyn Any` for maximum flexibility; concrete implementations
    /// can downcast to their expected argument types.
    ///
    /// # Arguments
    ///
    /// - `shader_name`: Name/identifier of the compute shader to run
    /// - `args`: Opaque arguments struct (specific to each shader)
    ///
    /// # Returns
    ///
    /// - `Ok(())` if computation succeeded
    /// - `Err` if shader not found, arguments invalid, or execution failed
    async fn execute_compute(
        &self,
        shader_name: &str,
        args: &dyn Any,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Upload data to the accelerator (e.g., to GPU memory).
    ///
    /// Implementations may use different strategies:
    /// - GPU backends: Copy data to VRAM
    /// - CPU backends: Store in memory or return immediately
    /// - WASM: Copy to WebGL texture or buffer
    ///
    /// # Arguments
    ///
    /// - `data`: Raw bytes to upload
    /// - `buffer_id`: Named buffer identifier (for retrieval later)
    ///
    /// # Returns
    ///
    /// - `Ok(())` if upload succeeded
    /// - `Err` if buffer not found, memory exhausted, or operation failed
    async fn upload_data(
        &self,
        data: &[u8],
        buffer_id: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Download data from the accelerator.
    ///
    /// Retrieves previously uploaded or computed data from the accelerator's memory.
    ///
    /// # Arguments
    ///
    /// - `buffer_id`: Name of buffer to download
    ///
    /// # Returns
    ///
    /// - `Ok(data)` with the buffer contents
    /// - `Err` if buffer not found or download failed
    async fn download_data(
        &self,
        buffer_id: &str,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>>;

    /// Get information about this accelerator backend.
    ///
    /// Returns static capabilities, memory info, compute unit count, etc.
    /// Can be called without initialization.
    fn info(&self) -> BackendInfo;

    /// Check if the accelerator is currently available and initialized.
    ///
    /// Used to determine if this backend can be used for computation.
    fn is_available(&self) -> bool;

    /// Wait for all pending operations to complete.
    ///
    /// This is important for GPU backends where operations are queued asynchronously.
    /// CPU backends can return immediately.
    async fn synchronize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accelerator_type_display() {
        assert_eq!(AcceleratorType::Cpu.to_string(), "CPU");
        assert_eq!(AcceleratorType::GpuMetal.to_string(), "GPU (Metal)");
        assert_eq!(AcceleratorType::Cuda.to_string(), "GPU (CUDA)");
    }
}