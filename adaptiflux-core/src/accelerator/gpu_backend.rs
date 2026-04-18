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

#[cfg(feature = "gpu")]
pub mod gpu_wgpu {
    use crate::accelerator::backend::{AcceleratorBackend, AcceleratorType, BackendInfo};
    use async_trait::async_trait;
    use std::any::Any;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::Mutex;
    use tracing::info;
    use wgpu::{Device, Instance, Queue};

    /// GPU accelerator backend using wgpu (Metal, Vulkan, DX12, etc.)
    pub struct GpuBackendWgpu {
        /// wgpu instance
        instance: Mutex<Option<Instance>>,
        /// wgpu device
        device: Mutex<Option<Arc<Device>>>,
        /// wgpu queue for command submission
        queue: Mutex<Option<Arc<Queue>>>,
        /// Named GPU buffers
        buffers: Mutex<HashMap<String, wgpu::Buffer>>,
        /// GPU device information
        device_info: Mutex<Option<BackendInfo>>,
    }

    impl GpuBackendWgpu {
        /// Create a new GPU backend instance
        pub fn new() -> Self {
            Self {
                instance: Mutex::new(None),
                device: Mutex::new(None),
                queue: Mutex::new(None),
                buffers: Mutex::new(HashMap::new()),
                device_info: Mutex::new(None),
            }
        }

        /// Initialize GPU context with platform-specific backends
        fn backends_for_platform() -> wgpu::Backends {
            #[cfg(target_os = "macos")]
            {
                wgpu::Backends::METAL
            }
            #[cfg(target_os = "windows")]
            {
                wgpu::Backends::DX12 | wgpu::Backends::VULKAN
            }
            #[cfg(target_os = "linux")]
            {
                wgpu::Backends::VULKAN
            }
            #[cfg(target_arch = "wasm32")]
            {
                wgpu::Backends::BROWSER_WEBGPU
            }
            #[cfg(all(
                not(target_os = "macos"),
                not(target_os = "windows"),
                not(target_os = "linux"),
                not(target_arch = "wasm32")
            ))]
            {
                wgpu::Backends::VULKAN | wgpu::Backends::DX12
            }
        }

        fn required_features() -> wgpu::Features {
            wgpu::Features::empty()
        }

        fn required_limits() -> wgpu::Limits {
            wgpu::Limits::default()
        }
    }

    impl Default for GpuBackendWgpu {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl AcceleratorBackend for GpuBackendWgpu {
        async fn initialize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            if self.device.lock().unwrap().is_some() {
                return Ok(()); // Already initialized
            }

            info!("Initializing GPU backend (wgpu)");

            let instance = Instance::new(wgpu::InstanceDescriptor {
                backends: Self::backends_for_platform(),
                flags: wgpu::InstanceFlags::all(),
                dx12_shader_compiler: Default::default(),
                gles_minor_version: wgpu::Gles3MinorVersion::Version2,
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .ok_or("Failed to find an appropriate adapter")?;

            let adapter_info = adapter.get_info();
            info!("Selected GPU adapter: {}", adapter_info.name);

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Adaptiflux GPU Device"),
                        required_features: Self::required_features(),
                        required_limits: Self::required_limits(),
                    },
                    None,
                )
                .await
                .map_err(|e| format!("Failed to create device: {}", e))?;

            let backend_name = match adapter_info.backend {
                wgpu::Backend::Metal => "Metal (Apple)",
                wgpu::Backend::Vulkan => "Vulkan",
                wgpu::Backend::Dx12 => "DirectX 12",
                wgpu::Backend::Gl => "OpenGL",
                wgpu::Backend::BrowserWebGpu => "WebGPU",
                _ => "Unknown",
            };

            let accelerator_type = match adapter_info.backend {
                wgpu::Backend::Metal => AcceleratorType::GpuMetal,
                wgpu::Backend::Vulkan => AcceleratorType::GpuVulkan,
                wgpu::Backend::Dx12 => AcceleratorType::GpuDx12,
                _ => AcceleratorType::Cpu,
            };

            let is_unified_memory = adapter_info.backend == wgpu::Backend::Metal
                || adapter_info.device_type == wgpu::DeviceType::IntegratedGpu;

            let compute_units = match adapter_info.device_type {
                wgpu::DeviceType::DiscreteGpu => 256,
                wgpu::DeviceType::IntegratedGpu => 64,
                wgpu::DeviceType::VirtualGpu => 32,
                wgpu::DeviceType::Cpu => 8,
                _ => 16,
            };

            let info = BackendInfo {
                name: format!("{} ({})", adapter_info.name, backend_name),
                backend_type: accelerator_type,
                memory_bytes: 4 * 1024 * 1024 * 1024, // 4GB estimate
                compute_units: Some(compute_units),
                is_unified_memory,
                version: adapter_info.driver.clone(),
            };

            // Store device and queue using interior mutability
            {
                let mut device_guard = self.device.lock().unwrap();
                let mut queue_guard = self.queue.lock().unwrap();
                let mut instance_guard = self.instance.lock().unwrap();
                
                *device_guard = Some(Arc::new(device));
                *queue_guard = Some(Arc::new(queue));
                *instance_guard = Some(instance);
            }
            
            *self.device_info.lock().unwrap() = Some(info);

            Ok(())
        }

        async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            info!("Shutting down GPU backend");
            self.buffers.lock().unwrap().clear();
            
            // Clear device, queue, and instance
            *self.device.lock().unwrap() = None;
            *self.queue.lock().unwrap() = None;
            *self.instance.lock().unwrap() = None;
            *self.device_info.lock().unwrap() = None;
            
            Ok(())
        }

        async fn execute_compute(
            &self,
            shader_name: &str,
            _args: &dyn Any,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            // For now, just log the compute request
            // A full implementation would compile shaders, create pipelines, etc.
            info!("GPU backend executing compute: {}", shader_name);
            Ok(())
        }

        async fn upload_data(
            &self,
            data: &[u8],
            buffer_id: &str,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            // To actually create a GPU buffer, we'd need device/queue
            // For now, we just track it in memory as a placeholder
            let _buffers = self.buffers.lock().unwrap();
            // In a real implementation, we'd create a wgpu::Buffer here
            // For now, store metadata in memory
            info!("GPU backend uploading buffer: {} ({} bytes)", buffer_id, data.len());
            Ok(())
        }

        async fn download_data(
            &self,
            buffer_id: &str,
        ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
            let _buffers = self.buffers.lock().unwrap();
            // In a real implementation, we'd read from GPU buffer
            info!("GPU backend downloading buffer: {}", buffer_id);
            Ok(vec![])
        }

        fn info(&self) -> BackendInfo {
            self.device_info
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| BackendInfo {
                    name: "GPU (wgpu) - uninitialized".to_string(),
                    backend_type: AcceleratorType::Cpu,
                    memory_bytes: 0,
                    compute_units: None,
                    is_unified_memory: false,
                    version: "unknown".to_string(),
                })
        }

        fn is_available(&self) -> bool {
            self.device.lock().unwrap().is_some()
        }

        async fn synchronize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            // GPU operations are queued, so we'd poll the device here
            // For now, just note that sync was requested
            Ok(())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[tokio::test]
        async fn test_gpu_backend_creation() {
            let backend = GpuBackendWgpu::new();
            assert!(!backend.is_available());
        }

        #[tokio::test]
        async fn test_gpu_backend_info_uninitialized() {
            let backend = GpuBackendWgpu::new();
            let info = backend.info();
            assert_eq!(info.backend_type, AcceleratorType::Cpu); // Uninitialized falls back to CPU info
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub mod gpu_wgpu {
    // Stub implementation when GPU feature is disabled
    use crate::accelerator::backend::{AcceleratorBackend, AcceleratorType, BackendInfo};
    use async_trait::async_trait;
    use std::any::Any;

    pub struct GpuBackendWgpu;

    impl GpuBackendWgpu {
        pub fn new() -> Self {
            Self
        }
    }

    impl Default for GpuBackendWgpu {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl AcceleratorBackend for GpuBackendWgpu {
        async fn initialize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Err("GPU feature disabled at compile time".into())
        }

        async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Err("GPU feature disabled at compile time".into())
        }

        async fn execute_compute(
            &self,
            _shader_name: &str,
            _args: &dyn Any,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Err("GPU feature disabled at compile time".into())
        }

        async fn upload_data(
            &self,
            _data: &[u8],
            _buffer_id: &str,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Err("GPU feature disabled at compile time".into())
        }

        async fn download_data(
            &self,
            _buffer_id: &str,
        ) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
            Err("GPU feature disabled at compile time".into())
        }

        fn info(&self) -> BackendInfo {
            BackendInfo {
                name: "GPU (wgpu) - disabled".to_string(),
                backend_type: AcceleratorType::Cpu,
                memory_bytes: 0,
                compute_units: None,
                is_unified_memory: false,
                version: "disabled".to_string(),
            }
        }

        fn is_available(&self) -> bool {
            false
        }

        async fn synchronize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            Err("GPU feature disabled at compile time".into())
        }
    }
}
