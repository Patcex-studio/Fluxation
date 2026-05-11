# Apple Silicon Metal GPU Optimization Guide

## Overview

This guide documents the current Apple Metal GPU acceleration support in `adaptiflux-core`.
It is based on the actual implementation of `GpuContext`, `GpuConfig`, and `GpuResourceManager` in the codebase.

## Architecture

### Key Components

1. **GpuContext**
   - Initializes `wgpu` with Metal backend prioritization on macOS.
   - Uses `GpuContext::new_with_preference(wgpu::PowerPreference::HighPerformance)`.
   - On macOS selects `wgpu::Backends::METAL`.
   - On Linux selects `wgpu::Backends::VULKAN`.
   - On Windows selects `wgpu::Backends::DX12 | wgpu::Backends::VULKAN`.
   - Logs device capabilities and estimates unified memory support.

2. **GpuResourceManager**
   - Wraps `GpuContext` and exposes GPU allocation for agents.
   - Tracks which agents are allocated to GPU.
   - Limits concurrent GPU usage via `max_concurrent_agents` (default 1).
   - Provides `allocate_for_agent`, `deallocate_for_agent`, and `is_agent_on_gpu`.

3. **GpuConfig**
   - Controls which GPU phases are enabled.
   - Configures batch sizes, sync intervals, profiling, and fallback behavior.
   - Provides profiles for Apple silicon, discrete GPUs, and CPU-only operation.

4. **Shader modules**
   - The GPU module exports shader names used by the backend:
     - `AGENT_UPDATE_SHADER`
     - `CONNECTION_CALCULATE_SHADER`
     - `PLASTICITY_PRUNING_SHADER`
     - `PLASTICITY_SYNAPTOGENESIS_SHADER`
     - `HORMONE_SIMULATION_SHADER`
     - `LIF_UPDATE_SHADER`

## Supported GPU Configuration

### Apple Silicon Profile

`GpuConfig::apple_silicon()` returns the current macOS-oriented profile:

- `enable_agent_update = true`
- `enable_connection_calculate = true`
- `enable_plasticity = true`
- `enable_hormone_simulation = true`
- `agent_batch_size = 512`
- `connection_batch_size = 1024`
- `enable_cpu_fallback = true`
- `enable_incremental_updates = true`
- `sync_interval_iterations = 10`
- `enable_profiling = true`
- `optimize_for_igpu = true`
- `prefer_high_performance = true`

### Discrete GPU Profile

`GpuConfig::discrete_gpu()` returns:

- `agent_batch_size = 2048`
- `connection_batch_size = 4096`
- `enable_incremental_updates = false`
- `sync_interval_iterations = 100`
- `optimize_for_igpu = false`

### CPU-Only Profile

`GpuConfig::cpu_only()` disables all GPU phases and keeps fallback semantics safe.

### Validation Rules

`GpuConfig::validate()` checks:

- `agent_batch_size > 0`
- `connection_batch_size > 0`
- if GPU is enabled and `enable_cpu_fallback == false`, then `sync_interval_iterations` must be non-zero.

## GPU Backend Initialization

### Creating a GPU context

```rust
let context = GpuContext::new_with_preference(
    wgpu::PowerPreference::HighPerformance,
).await?;
```

`GpuContext::new()` is an alias for the same high-performance preference.

### Platform backend selection

- macOS: `wgpu::Backends::METAL`
- Linux: `wgpu::Backends::VULKAN`
- Windows: `wgpu::Backends::DX12 | wgpu::Backends::VULKAN`

### Notes

- The implementation currently does not expose a public `force_fallback_adapter` option.
- If adapter creation fails, verify platform GPU support, macOS version, and `wgpu` compatibility.

## Using GPU Acceleration in CoreScheduler

### Example: create scheduler with GPU support

```rust
use std::sync::Arc;
use tokio::sync::Mutex;
use adaptiflux_core::gpu::{GpuConfig, GpuResourceManager};
use adaptiflux_core::core::CoreScheduler;

let gpu_manager = Arc::new(Mutex::new(GpuResourceManager::new().await?));
let mut scheduler = CoreScheduler::new_with_gpu(
    topology,
    rule_engine,
    resource_manager,
    message_bus,
    Some(gpu_manager),
);

scheduler.set_gpu_config(GpuConfig::apple_silicon());
```

### GPU-capable agent blueprints

```rust
#[async_trait]
impl AgentBlueprint for MyGpuAgent {
    // ... required methods ...

    fn supports_gpu(&self) -> bool {
        true
    }
}
```

The scheduler allocates GPU resources only for agents where `supports_gpu()` returns `true`.

## Performance Tuning

### Batch sizes

Current code uses these empirical defaults:

| Network Size | Recommended `agent_batch_size` | Notes |
|---|---|---|
| < 500 agents | 256 | GPU overhead can dominate |
| 500-2000 | 512 | Apple silicon default |
| 2000-5000 | 1024 | Better balance of throughput |
| 5000+ | 2048 | Use with larger GPUs |

### Sync interval

- `sync_interval_iterations = 10` is used by the Apple silicon profile.
- `sync_interval_iterations = 100` is used by the discrete GPU profile.
- `0` is only safe when GPU is disabled or CPU fallback remains enabled.

### Profiling

`enable_profiling` is enabled in both built-in GPU profiles.

### Integrated GPU characteristics

Apple GPUs typically offer:
- shared memory/cache with the CPU
- high bandwidth to system RAM
- power-efficient compute

The current implementation optimizes for these characteristics using smaller batch sizes and incremental updates.

## WGSL and Shader Notes

The code exports WGSL shader constants in `adaptiflux_core::gpu`.
The exact shader definitions are stored in `src/gpu/shaders`.

## Troubleshooting

### Initialization failures

If `GpuContext::new()` fails with `Failed to find an appropriate adapter`, verify:

- macOS 11+ and Metal support
- the `gpu` feature is enabled in the crate
- the system GPU drivers are current

### Memory pressure

If GPU memory is exhausted, reduce `agent_batch_size` and `connection_batch_size`.

### Shader compilation errors

Validate WGSL binding layout and data structure alignment between CPU and GPU.

## Implementation Reality

- `GpuContext` uses `wgpu::Features::empty()` and a custom `wgpu::Limits` with `max_compute_invocations_per_workgroup = 1024`.
- `GpuResourceManager` stores a `HashSet<ZoooidId>` and currently allows only one GPU allocation by default.
- On macOS, `GpuConfig::default()` resolves to `GpuConfig::apple_silicon()`.
- `GpuConfig` supports Apple silicon, discrete GPU, and CPU-only builder profiles.
- `enable_cpu_fallback` protects against GPU failures.

## References

- [wgpu Documentation](https://docs.rs/wgpu/)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [Adaptiflux Core API](../../README.md)

## License

This documentation and associated code are licensed under AGPL-3.0 or Commercial License.
See LICENSE files for details.
