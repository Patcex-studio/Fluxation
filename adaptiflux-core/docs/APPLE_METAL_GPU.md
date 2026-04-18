# Apple Silicon Metal GPU Optimization Guide

## Overview

This guide explains how to leverage Apple Metal GPU acceleration for Fluxation on macOS, particularly on Apple Silicon (M1/M2/M3) integrated GPUs.

## Architecture

### Key Components

1. **GpuContext** - Initializes wgpu with Metal backend prioritization on macOS
   - Automatically selects Metal backend on macOS
   - Falls back to Vulkan on Linux, DirectX 12 on Windows
   - Logs device capabilities (unified memory, compute units, etc.)

2. **GpuResourceManager** - Allocates GPU resources to agents
   - Tracks which agents are GPU-accelerated
   - Manages maximum concurrent GPU operations
   - Integrates with CoreScheduler

3. **BufferManager** - Manages GPU buffers efficiently
   - Supports STORAGE, UNIFORM, and COPY buffers
   - Tracks dirty flags for incremental updates
   - Optimized for unified memory architectures

4. **ShaderRunner** - Executes WGSL compute shaders
   - Manages compute pipelines
   - Dispatches workgroups
   - Synchronizes GPU/CPU

5. **GpuConfig** - Configuration for GPU acceleration
   - Enables/disables specific GPU operations
   - Controls batch sizes and sync intervals
   - Provides pre-configured profiles

### WGSL Shaders

The following compute shaders are provided:

- **AGENT_UPDATE_SHADER** - Izhikevich neuron model updates
  - Input: agent state buffer, parameters
  - Output: updated agent states with spike detection
  - Workgroup size: 256 threads (optimized for Apple GPU)

- **CONNECTION_CALCULATE_SHADER** - STDP plasticity calculations
  - Input: agents, edges, STDP parameters
  - Output: updated synaptic weights
  - Implements Spike-Timing-Dependent Plasticity

- **PLASTICITY_PRUNING_SHADER** - Weak connection removal
  - Input: edge strengths, activity traces
  - Output: prune flags for edge removal
  - Supports structural plasticity

- **PLASTICITY_SYNAPTOGENESIS_SHADER** - New connection formation
  - Input: agent activity signals
  - Output: synapse creation count
  - Drives network growth

- **HORMONE_SIMULATION_SHADER** - Neuromodulator dynamics
  - Input: network error/reward signals
  - Output: dopamine, cortisol, adrenaline levels
  - Global state updates (single workgroup)

- **LIF_UPDATE_SHADER** - Simplified Leaky Integrate-and-Fire model
  - Alternative to Izhikevich
  - Lower compute requirement
  - Suitable for large networks

## Apple Silicon Optimizations

### Unified Memory Architecture

Apple Silicon's unified memory is a key advantage:

- **Single memory pool** for CPU and GPU
- **No explicit transfers** - automatic coherency
- **Lower latency** for CPU/GPU synchronization
- **Reduced memory overhead** - no duplication

**Optimization**: Use `GpuConfig::apple_silicon()` which enables:
- Incremental buffer updates (only changed regions)
- Lower sync intervals (10 iterations vs 100)
- Smaller batch sizes (512 agents vs 2048 for discrete GPU)

### Integrated GPU Characteristics

M1/M2/M3 GPUs have:
- **Shared L4 cache** with CPU
- **Up to 8 GPU cores** (variable by chip)
- **High memory bandwidth** to system RAM
- **Power efficient** - great for laptops

**Optimization**: 
- Minimize memory transfers
- Use workgroup size of 256 (safe for all Apple GPUs)
- Favor frequent small kernels over rare large ones
- Leverage cache locality

### Power Preference

Always use `PowerPreference::HighPerformance` for Fluxation:
```rust
let context = GpuContext::new_with_preference(
    wgpu::PowerPreference::HighPerformance
).await?;
```

This ensures we use GPU cores, not GPU throttling.

## Usage Examples

### Basic GPU-Accelerated Scheduler

```rust
use adaptiflux_core::gpu::{GpuConfig, GpuResourceManager};
use adaptiflux_core::core::CoreScheduler;

// Initialize GPU for Apple Silicon
let gpu_manager = Arc::new(Mutex::new(
    GpuResourceManager::new().await?
));

// Create scheduler with GPU
let mut scheduler = CoreScheduler::new_with_gpu(
    topology, rule_engine, resource_manager, message_bus,
    Some(gpu_manager),
);

// Configure for Apple Silicon
scheduler.set_gpu_config(GpuConfig::apple_silicon());

// Run with GPU acceleration
scheduler.run().await?;
```

### Custom GPU Configuration

```rust
let mut config = GpuConfig::apple_silicon();

// Disable plasticity GPU acceleration (keep on CPU)
config.enable_plasticity = false;

// Increase batch size for faster computation
config.agent_batch_size = 1024;

// Enable detailed profiling
config.enable_profiling = true;

scheduler.set_gpu_config(config);
```

### GPU Agent Blueprint

```rust
#[async_trait]
impl AgentBlueprint for MyGpuAgent {
    // ... required methods ...

    fn supports_gpu(&self) -> bool {
        true  // Signal that this agent can run on GPU
    }
}
```

## Performance Tuning

### Batch Sizes

Adjust `agent_batch_size` based on network size:

| Network Size | Recommended Batch | Device Type |
|---|---|---|
| < 500 agents | 256 | Any Apple GPU |
| 500-2000 | 512 | M1 CPU-bounded |
| 2000-5000 | 1024 | M2/M3 |
| 5000+ | 2048 | Discrete GPU |

### Sync Intervals

`sync_interval_iterations` controls how often GPU results are synced back to CPU:

- **Apple Silicon**: 10-20 (fast unified memory)
- **Discrete GPU**: 100+ (high transfer cost)
- **0**: Never sync back (GPU-only mode)

### Workgroup Sizes

WGSL shaders use adaptive workgroup sizing:

```wgsl
@compute @workgroup_size(256)
fn kernel() {
    // Safe for all Apple GPUs
    // 256 threads per workgroup
}
```

For very high parallelism:
```wgsl
@compute @workgroup_size(512)  // M2+
```

### Memory Layout

Arrange buffers for cache efficiency:

```rust
// Good: agents and params are read sequentially
let _ = agents[idx];
let _ = params[idx];

// Bad: strided memory access (cache miss)
let _ = agents[idx * 2];
```

## Troubleshooting

### GPU Initialization Fails

```
Failed to find an appropriate adapter
```

**Cause**: Metal framework not initialized or unavailable

**Solution**:
1. Ensure macOS 11.0 or newer
2. Check that discrete GPU drivers are up to date
3. Set `force_fallback_adapter: true` in desperation

### Out of Memory

```
Validation error: not enough space in staging belt
```

**Solution**:
1. Reduce `agent_batch_size`
2. Reduce number of agents per GPU
3. Use `enable_incremental_updates: false` to save memory

### Shader Compilation Errors

```
Shader compilation failed: ...
```

**Solution**:
1. Ensure WGSL syntax is correct
2. Check buffer bindings (group and binding numbers)
3. Verify type sizes match CPU struct layouts

### Performance Degradation

Slower than CPU on small networks (< 256 agents)?

**Reason**: GPU overhead dominates computation time

**Solution**:
1. Increase network size to 1000+ agents
2. Use `GpuConfig::cpu_only()` for small networks
3. Profile with `enable_profiling: true`

## Architecture Decisions

### Why wgpu?

- **Cross-platform**: Metal, Vulkan, DX12, OpenGL in single code
- **Safe**: Memory safety, no undefined behavior
- **Modern**: Compute shaders, bindless resources
- **Active**: Maintained, following GPU standards

### Why WGSL?

- **Portable**: Compiles to Metal, SPIR-V, etc.
- **Safe**: TypeScript-like safety
- **Modern**: Aligned with WebGPU standard
- **Readable**: Clear compute semantics

### Why Unified Memory for Apple?

- **Natural fit**: Apple Silicon architecture is unified
- **Efficient**: Automatic coherency
- **Simple**: No explicit sync code
- **Performance**: L4 cache utilization

## Future Enhancements

1. **Adaptive workgroup sizing** - Query device limits at runtime
2. **Shader specialization** - Different kernels for different agent types
3. **Async compute** - Overlapping compute with memory transfers
4. **Profiling integration** - MTLCounterSampleBuffer support on Metal
5. **Distributed GPU** - Multiple GPUs across machines
6. **Quantization** - FP16/INT8 for memory efficiency
7. **Sparsity** - Sparse matrix operations for sparse networks

## References

- [wgpu Documentation](https://docs.rs/wgpu/)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [Adaptiflux Core API](../../README.md)

## License

This documentation and associated code are licensed under AGPL-3.0 or Commercial License.
See LICENSE files for details.
