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
// SPDX-License-Identifier: AGPL-3.0 OR Commercial

#![cfg(all(test, feature = "gpu"))]

use super::*;
use crate::core::message_bus::bus::LocalBus;
use crate::core::resource_manager::ResourceManager;
use crate::core::scheduler::CoreScheduler;
use crate::core::topology::ZoooidTopology;
use crate::rules::RuleEngine;
use std::convert::TryInto;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use tokio::sync::Mutex;
use wgpu::util::DeviceExt;

static GPU_TEST_LOCK: OnceLock<StdMutex<()>> = OnceLock::new();

fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_TEST_LOCK.get_or_init(|| StdMutex::new(())).lock().unwrap()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn u32_vec_from_bytes(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn f32_vec_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

async fn read_buffer_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error + Send + Sync>> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-u32"),
        size: (len * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, (len * 4) as u64);
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let result = u32_vec_from_bytes(&data);
    drop(data);
    readback.unmap();
    Ok(result)
}

async fn read_buffer_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback-f32"),
        size: (len * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("readback-encoder"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, (len * 4) as u64);
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = slice.get_mapped_range();
    let result = f32_vec_from_bytes(&data);
    drop(data);
    readback.unmap();
    Ok(result)
}

async fn create_gpu_context()
    -> Result<GpuContext, Box<dyn std::error::Error + Send + Sync>> {
    Ok(GpuContext::new().await?)
}

#[tokio::test]
async fn can_initialize_on_available_backends() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let backends = [
        wgpu::Backends::VULKAN,
        wgpu::Backends::DX12,
        wgpu::Backends::METAL,
        wgpu::Backends::GL,
        wgpu::Backends::BROWSER_WEBGPU,
    ];

    let mut initialized = false;
    for backend in backends {
        if let Ok(ctx) = GpuContext::new_with_backends(backend, wgpu::PowerPreference::HighPerformance).await {
            initialized = true;
            assert!(!ctx.device_info.name.is_empty());
            assert!(!ctx.device_info.backend.is_empty());
        }
    }

    assert!(initialized, "No supported wgpu backend was available on this host");
    Ok(())
}

#[tokio::test]
async fn reports_correct_device_info() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    assert!(!ctx.device_info.name.is_empty());
    assert!(!ctx.device_info.backend.is_empty());
    assert!(ctx.device_info.max_workgroup_size > 0);
    assert!(ctx.device_info.compute_units > 0);
    assert!(ctx.device_info.memory_estimate_mb > 0);
    Ok(())
}

#[tokio::test]
async fn detects_unified_memory_if_available() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let info = ctx.adapter.get_info();
    let expected = info.backend == wgpu::Backend::Metal
        || info.device_type == wgpu::DeviceType::IntegratedGpu;
    assert_eq!(ctx.device_info.is_unified_memory, expected);
    Ok(())
}

#[tokio::test]
async fn creates_buffer_with_correct_usage_and_size() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let mut manager = BufferManager::new(
        Arc::new(ctx.device),
        Arc::new(ctx.queue),
    );

    manager.create_storage_buffer("storage", 128)?;
    let meta = manager.get_metadata("storage").unwrap();
    assert_eq!(meta.size, 128);
    assert!(meta.usage.contains(wgpu::BufferUsages::STORAGE));
    assert!(!meta.is_dirty);
    Ok(())
}

#[tokio::test]
async fn marks_buffer_as_dirty_on_write() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let mut manager = BufferManager::new(
        Arc::new(ctx.device),
        Arc::new(ctx.queue),
    );

    manager.create_copy_buffer("copy", 64)?;
    assert!(!manager.is_dirty("copy"));
    manager.write_buffer("copy", 0, &[0u8, 1, 2, 3])?;
    assert!(manager.is_dirty("copy"));
    Ok(())
}

#[tokio::test]
async fn handles_multiple_buffer_types() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let mut manager = BufferManager::new(
        Arc::new(ctx.device),
        Arc::new(ctx.queue),
    );

    manager.create_storage_buffer("storage", 64)?;
    manager.create_uniform_buffer("uniform", 32)?;
    manager.create_copy_buffer("copy", 32)?;

    let storage_meta = manager.get_metadata("storage").unwrap();
    let uniform_meta = manager.get_metadata("uniform").unwrap();
    let copy_meta = manager.get_metadata("copy").unwrap();

    assert!(storage_meta.usage.contains(wgpu::BufferUsages::STORAGE));
    assert!(uniform_meta.usage.contains(wgpu::BufferUsages::UNIFORM));
    assert!(copy_meta.usage.contains(wgpu::BufferUsages::COPY_SRC));
    assert!(copy_meta.usage.contains(wgpu::BufferUsages::COPY_DST));
    Ok(())
}

#[tokio::test]
async fn manages_buffer_lifecycle() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let mut manager = BufferManager::new(
        Arc::new(ctx.device),
        Arc::new(ctx.queue),
    );

    manager.create_copy_buffer("temp", 16)?;
    assert!(manager.get_buffer("temp").is_some());
    assert!(manager.get_metadata("temp").is_some());
    manager.remove_buffer("temp");
    assert!(manager.get_buffer("temp").is_none());
    assert!(manager.get_metadata("temp").is_none());
    Ok(())
}

#[test]
fn loads_presets_correctly() {
    let apple = GpuConfig::apple_silicon();
    assert!(apple.enable_agent_update);
    assert_eq!(apple.agent_batch_size, 512);

    let discrete = GpuConfig::discrete_gpu();
    assert!(discrete.enable_agent_update);
    assert_eq!(discrete.agent_batch_size, 2048);

    let cpu_only = GpuConfig::cpu_only();
    assert!(!cpu_only.enable_agent_update);
    assert_eq!(cpu_only.agent_batch_size, u32::MAX);
}

#[test]
fn validates_config_parameters() {
    let good = GpuConfig::apple_silicon();
    assert!(good.validate().is_ok());

    let mut bad_agent = GpuConfig::apple_silicon();
    bad_agent.agent_batch_size = 0;
    assert!(bad_agent.validate().is_err());

    let mut bad_connection = GpuConfig::apple_silicon();
    bad_connection.connection_batch_size = 0;
    assert!(bad_connection.validate().is_err());
}

#[tokio::test]
async fn gpu_execute_addition_works() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let a_data = [1u32, 2, 3];
    let b_data = [4u32, 5, 6];
    let expected = [5u32, 7, 9];

    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("a-buffer"),
        contents: &u32_bytes(&a_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("b-buffer"),
        contents: &u32_bytes(&b_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out-buffer"),
        size: (a_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("len-buffer"),
        contents: &u32_bytes(&[a_data.len() as u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let shader = r#"
        @group(0) @binding(0) var<storage, read> a: array<u32>;
        @group(0) @binding(1) var<storage, read> b: array<u32>;
        @group(0) @binding(2) var<storage, read_write> out: array<u32>;
        @group(0) @binding(3) var<uniform> len: u32;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx >= len) { return; }
            out[idx] = a[idx] + b[idx];
        }
    "#;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("add-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("add-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: len_buf.as_entire_binding(),
            },
        ],
    });

    let runner = ShaderRunner::new(
        device.clone(),
        queue.clone(),
        shader,
        "main",
        bind_group_layout,
        bind_group,
    )?;
    runner.run(1, 1, 1).await?;

    let output = read_buffer_u32(&device, &queue, &out_buf, a_data.len()).await?;
    assert_eq!(output, expected);
    Ok(())
}

#[tokio::test]
async fn gpu_execute_copy_works() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let source_data = [7u32, 8, 9, 10];
    let source_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("source-buffer"),
        contents: &u32_bytes(&source_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let target_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("target-buffer"),
        size: (source_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = r#"
        @group(0) @binding(0) var<storage, read> src: array<u32>;
        @group(0) @binding(1) var<storage, read_write> dst: array<u32>;
        @group(0) @binding(2) var<uniform> len: u32;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (idx >= len) { return; }
            dst[idx] = src[idx];
        }
    "#;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("copy-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
        ],
    });

    let len_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("copy-len"),
        contents: &u32_bytes(&[source_data.len() as u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("copy-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: len_buf.as_entire_binding(),
            },
        ],
    });

    let runner = ShaderRunner::new(
        device.clone(),
        queue.clone(),
        shader,
        "main",
        bind_group_layout,
        bind_group,
    )?;
    runner.run(1, 1, 1).await?;

    let output = read_buffer_u32(&device, &queue, &target_buf, source_data.len()).await?;
    assert_eq!(output, source_data);
    Ok(())
}

#[tokio::test]
async fn gpu_execute_compare_works() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let payload = [11u32, 11u32];
    let source_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compare-buffer"),
        contents: &u32_bytes(&payload),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let result_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compare-result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = r#"
        @group(0) @binding(0) var<storage, read> values: array<u32>;
        @group(0) @binding(1) var<storage, read_write> result: array<u32>;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            if (values[0] == values[1]) {
                result[0] = 1u;
            } else {
                result[0] = 0u;
            }
        }
    "#;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compare-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compare-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buf.as_entire_binding(),
            },
        ],
    });

    let runner = ShaderRunner::new(
        device.clone(),
        queue.clone(),
        shader,
        "main",
        bind_group_layout,
        bind_group,
    )?;
    runner.run(1, 1, 1).await?;

    let result = read_buffer_u32(&device, &queue, &result_buf, 1).await?;
    assert_eq!(result[0], 1);
    Ok(())
}

#[tokio::test]
async fn gpu_agent_update_basic() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let initial = [1.0f32, 2.0, 3.0];
    let expected = [2.0f32, 3.0, 4.0];

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("agent-buffer"),
        contents: &f32_bytes(&initial),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });

    let shader = r#"
        @group(0) @binding(0) var<storage, read_write> values: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            values[global_id.x] = values[global_id.x] + 1.0;
        }
    "#;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("agent-update-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("agent-update-bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let runner = ShaderRunner::new(
        device.clone(),
        queue.clone(),
        shader,
        "main",
        bind_group_layout,
        bind_group,
    )?;
    runner.run(1, 1, 1).await?;

    let result = read_buffer_f32(&device, &queue, &buffer, initial.len()).await?;
    for (a, b) in expected.iter().zip(result.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    Ok(())
}

#[tokio::test]
async fn gpu_connection_calc_basic() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let inputs = [2.0f32, 4.0];
    let expected = [8.0f32];

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("connection-input"),
        contents: &f32_bytes(&inputs),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("connection-output"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = r#"
        @group(0) @binding(0) var<storage, read> values: array<f32>;
        @group(0) @binding(1) var<storage, read_write> result: array<f32>;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            result[0] = values[0] * values[1];
        }
    "#;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("connection-calc-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("connection-calc-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let runner = ShaderRunner::new(
        device.clone(),
        queue.clone(),
        shader,
        "main",
        bind_group_layout,
        bind_group,
    )?;
    runner.run(1, 1, 1).await?;

    let result = read_buffer_f32(&device, &queue, &output_buf, expected.len()).await?;
    assert!((result[0] - expected[0]).abs() < 1e-6);
    Ok(())
}

#[tokio::test]
async fn gpu_vs_cpu_agent_update_consistent() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let input = [1.0f32, 2.5, -0.5];
    let expected: Vec<f32> = input.iter().map(|v| v + 1.0).collect();

    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cpu-gpu-agent"),
        contents: &f32_bytes(&input),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });

    let shader = r#"
        @group(0) @binding(0) var<storage, read_write> values: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            values[global_id.x] = values[global_id.x] + 1.0;
        }
    "#;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cpu-gpu-agent-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cpu-gpu-agent-bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let runner = ShaderRunner::new(
        device.clone(),
        queue.clone(),
        shader,
        "main",
        bind_group_layout,
        bind_group,
    )?;
    runner.run(1, 1, 1).await?;

    let result = read_buffer_f32(&device, &queue, &buffer, input.len()).await?;
    for (actual, expected_value) in result.iter().zip(expected.iter()) {
        assert!((actual - expected_value).abs() < 1e-6);
    }
    Ok(())
}

#[tokio::test]
async fn gpu_vs_cpu_plasticity_consistent() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let strengths = [0.1f32, 0.7, 0.3, 0.9];
    let threshold = 0.5;
    let expected: Vec<u32> = strengths
        .iter()
        .map(|&weight| if weight < threshold { 1 } else { 0 })
        .collect();

    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let edge_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("edge-strengths"),
        contents: &f32_bytes(&strengths),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let flag_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prune-flags"),
        size: (strengths.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = r#"
        @group(0) @binding(0) var<storage, read> strengths: array<f32>;
        @group(0) @binding(1) var<storage, read_write> flags: array<u32>;
        @group(0) @binding(2) var<uniform> threshold: f32;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            if (strengths[idx] < threshold) {
                flags[idx] = 1u;
            } else {
                flags[idx] = 0u;
            }
        }
    "#;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("plasticity-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
        ],
    });

    let threshold_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("threshold-buffer"),
        contents: &f32_bytes(&[threshold]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("plasticity-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: edge_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: flag_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: threshold_buf.as_entire_binding(),
            },
        ],
    });

    let runner = ShaderRunner::new(
        device.clone(),
        queue.clone(),
        shader,
        "main",
        bind_group_layout,
        bind_group,
    )?;
    runner.run(1, 1, 1).await?;

    let result = read_buffer_u32(&device, &queue, &flag_buf, strengths.len()).await?;
    assert_eq!(result, expected);
    Ok(())
}

#[tokio::test]
async fn stress_large_buffer_allocation() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let max_size = ctx.device.limits().max_buffer_size;
    let mut manager = BufferManager::new(
        Arc::new(ctx.device),
        Arc::new(ctx.queue),
    );

    let sizes = [100u64 * 1024 * 1024, std::cmp::min(max_size, 200u64 * 1024 * 1024), max_size + 1];
    for size in sizes {
        let name = format!("large-{}-bytes", size);
        let result = manager.create_copy_buffer(&name, size);
        if size > max_size {
            assert!(result.is_err(), "Expected over-limit buffer to be rejected");
        } else {
            result?;
            assert_eq!(manager.get_metadata(&name).unwrap().size, size);
            manager.remove_buffer(&name);
        }
    }

    Ok(())
}

#[tokio::test]
async fn stress_many_dispatch_calls() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let data = [1u32];
    let src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("stress-src"),
        contents: &u32_bytes(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stress-dst"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = r#"
        @group(0) @binding(0) var<storage, read> src: array<u32>;
        @group(0) @binding(1) var<storage, read_write> dst: array<u32>;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            dst[0] = src[0];
        }
    "#;
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("stress-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stress-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst.as_entire_binding(),
            },
        ],
    });

    let runner = ShaderRunner::new(
        device.clone(),
        queue.clone(),
        shader,
        "main",
        bind_group_layout,
        bind_group,
    )?;

    for _ in 0..1000 {
        runner.run(1, 1, 1).await?;
    }

    let result = read_buffer_u32(&device, &queue, &dst, 1).await?;
    assert_eq!(result[0], data[0]);
    Ok(())
}

#[tokio::test]
async fn stress_concurrent_compute_passes() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("concurrent-src"),
        contents: &u32_bytes(&[42u32]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("concurrent-dst"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("concurrent-shader"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
                @group(0) @binding(0) var<storage, read> src: array<u32>;
                @group(0) @binding(1) var<storage, read_write> dst: array<u32>;
                @compute @workgroup_size(1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    dst[0] = src[0];
                }
            "#
            .into(),
        ),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("concurrent-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("concurrent-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: dst.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("concurrent-pipeline-layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("concurrent-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("concurrent-encoder"),
    });
    for _ in 0..4 {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("concurrent-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::Maintain::Wait);

    let result = read_buffer_u32(&device, &queue, &dst, 1).await?;
    assert_eq!(result[0], 42);
    Ok(())
}

#[tokio::test]
async fn stress_long_running_simulation() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("long-running"),
        contents: &f32_bytes(&[0.0f32; 1]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    });
    let shader = r#"
        @group(0) @binding(0) var<storage, read_write> value: array<f32>;
        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            value[0] = value[0] + 1.0;
        }
    "#;
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("long-running-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("long-running-bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let runner = ShaderRunner::new(
        device.clone(),
        queue.clone(),
        shader,
        "main",
        bind_group_layout,
        bind_group,
    )?;

    for _ in 0..200 {
        runner.run(1, 1, 1).await?;
    }

    let result = read_buffer_f32(&device, &queue, &buffer, 1).await?;
    assert!((result[0] - 200.0).abs() < 1e-6);
    Ok(())
}

#[tokio::test]
async fn gpu_handles_zero_agents() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("zero-agents"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = r#"
        @group(0) @binding(0) var<storage, read_write> value: array<u32>;
        @group(0) @binding(1) var<uniform> count: u32;

        @compute @workgroup_size(1)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            if (global_id.x >= count) {
                return;
            }
            value[0] = 0u;
        }
    "#;

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("zero-agents-bgl"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            },
        ],
    });

    let count_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("zero-count"),
        contents: &u32_bytes(&[0u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("zero-agents-bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: count_buf.as_entire_binding(),
            },
        ],
    });

    let runner = ShaderRunner::new(
        device.clone(),
        queue.clone(),
        shader,
        "main",
        bind_group_layout,
        bind_group,
    )?;
    runner.run(1, 1, 1).await?;
    Ok(())
}

#[test]
fn gpu_handles_invalid_config() {
    let mut config = GpuConfig::apple_silicon();
    config.agent_batch_size = 0;
    assert!(config.validate().is_err());

    let mut config = GpuConfig::apple_silicon();
    config.connection_batch_size = 0;
    assert!(config.validate().is_err());
}

#[tokio::test]
async fn scheduler_initializes_gpu_resources() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let gpu_manager = Arc::new(Mutex::new(GpuResourceManager::new().await?));
    let scheduler = CoreScheduler::new_with_gpu(
        Arc::new(RwLock::new(ZoooidTopology::new())),
        RuleEngine::new(),
        ResourceManager::new(),
        Arc::new(LocalBus::new()),
        Some(gpu_manager),
    );

    assert!(scheduler.gpu_resource_manager.is_some());
    assert!(scheduler.get_gpu_config().is_gpu_enabled());
    Ok(())
}

#[test]
fn scheduler_falls_back_to_cpu_when_gpu_disabled() {
    let mut scheduler = CoreScheduler::new_with_gpu(
        Arc::new(RwLock::new(ZoooidTopology::new())),
        RuleEngine::new(),
        ResourceManager::new(),
        Arc::new(LocalBus::new()),
        None,
    );
    scheduler.set_gpu_config(GpuConfig::cpu_only());
    assert!(!scheduler.get_gpu_config().is_gpu_enabled());
}

#[tokio::test]
async fn gpu_fallback_on_invalid_shader_reported() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _gpu_guard = gpu_test_lock();
    let ctx = create_gpu_context().await?;
    let device = Arc::new(ctx.device);
    let queue = Arc::new(ctx.queue);

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("invalid-shader"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = "@compute @workgroup_size(1) fn main() { invalid }";
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("invalid-shader-bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("invalid-shader-bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ShaderRunner::new(
            device.clone(),
            queue.clone(),
            shader,
            "main",
            bind_group_layout,
            bind_group,
        )
    }));

    assert!(result.is_err() || result.ok().unwrap().is_err());
    Ok(())
}
