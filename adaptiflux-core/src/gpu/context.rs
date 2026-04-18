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

use tracing::info;
use wgpu::{Adapter, AdapterInfo, Device, Instance, Queue, RequestDeviceError};

/// Information about GPU device capabilities and backend
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Name of the GPU device
    pub name: String,
    /// Backend type (Metal, Vulkan, DX12, etc.)
    pub backend: String,
    /// Device driver version
    pub driver: String,
    /// Maximum total working group size
    pub max_workgroup_size: u32,
    /// Maximum number of compute units / cores
    pub compute_units: u32,
    /// Estimated device memory (if available)
    pub memory_estimate_mb: u64,
    /// Is unified memory architecture (Apple Silicon)
    pub is_unified_memory: bool,
}

impl GpuDeviceInfo {
    /// Create device info from wgpu adapter info and limits
    pub fn from_adapter(info: &AdapterInfo, limits: &wgpu::Limits) -> Self {
        let backend = match info.backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DirectX 12",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "Unknown",
        };

        // Detect unified memory on Apple platforms and integrated GPUs
        let is_unified_memory = info.backend == wgpu::Backend::Metal
            || info.device_type == wgpu::DeviceType::IntegratedGpu;

        let compute_units = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => 256,
            wgpu::DeviceType::IntegratedGpu => 64,
            wgpu::DeviceType::VirtualGpu => 32,
            wgpu::DeviceType::Cpu => 8,
            _ => 16,
        };

        Self {
            name: info.name.clone(),
            backend: backend.to_string(),
            driver: info.driver.clone(),
            max_workgroup_size: limits.max_compute_workgroup_size_x,
            compute_units,
            memory_estimate_mb: 4096, // Conservative; will vary by device
            is_unified_memory,
        }
    }
}

pub struct GpuContext {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub device_info: GpuDeviceInfo,
}

impl GpuContext {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::new_with_preference(wgpu::PowerPreference::HighPerformance).await
    }

    /// Create GPU context with explicit backend selection and power preference.
    pub async fn new_with_backends(
        backends: wgpu::Backends,
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends,
            flags: wgpu::InstanceFlags::all(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Version2,
        });

        // Request adapter with explicit backend selection
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find an appropriate adapter")?;

        Self::build_from_adapter(instance, adapter).await
    }

    /// Create GPU context with platform-preferred backends.
    pub async fn new_with_preference(
        power_preference: wgpu::PowerPreference,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::new_with_backends(Self::backends_for_platform(), power_preference).await
    }

    async fn build_from_adapter(
        instance: Instance,
        adapter: Adapter,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let adapter_info = adapter.get_info();
        info!("Selected GPU adapter: {}", adapter_info.name);
        info!(
            "Backend: {}",
            match adapter_info.backend {
                wgpu::Backend::Metal => "Metal (Apple)",
                wgpu::Backend::Vulkan => "Vulkan",
                wgpu::Backend::Dx12 => "DirectX 12",
                wgpu::Backend::Gl => "OpenGL",
                wgpu::Backend::BrowserWebGpu => "WebGPU",
                _ => "Unknown",
            }
        );

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
            .map_err(|e: RequestDeviceError| format!("Failed to create device: {}", e))?;

        let limits = device.limits();
        let device_info = GpuDeviceInfo::from_adapter(&adapter_info, &limits);

        info!("GPU Context initialized successfully");
        info!("Device: {}", device_info.name);
        info!("Backend: {}", device_info.backend);
        info!("Driver: {}", device_info.driver);
        info!("Max workgroup size: {}", device_info.max_workgroup_size);
        info!(
            "Unified memory architecture: {}",
            device_info.is_unified_memory
        );

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            device_info,
        })
    }

    /// Get appropriate backends for the current platform
    #[cfg(target_os = "macos")]
    fn backends_for_platform() -> wgpu::Backends {
        // On macOS, prioritize Metal
        wgpu::Backends::METAL
    }

    #[cfg(target_os = "windows")]
    fn backends_for_platform() -> wgpu::Backends {
        wgpu::Backends::DX12 | wgpu::Backends::VULKAN
    }

    #[cfg(target_os = "linux")]
    fn backends_for_platform() -> wgpu::Backends {
        wgpu::Backends::VULKAN
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
    fn backends_for_platform() -> wgpu::Backends {
        wgpu::Backends::all()
    }

    /// Get required GPU features for compute operations
    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }

    /// Get device limits optimized for compute
    fn required_limits() -> wgpu::Limits {
        let mut limits = wgpu::Limits::default();
        // Increase max compute workgroups for batch processing
        limits.max_compute_invocations_per_workgroup = 1024;
        limits
    }
}
