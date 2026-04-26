# SynapseManager: Migration Guide

## Overview

This guide helps developers migrate from the old Vec-based synaptic weight storage to the new **`SynapseManager`** system.

### Key Benefits of Migration

- **O(1) Average-Case Lookup**: Instead of O(N) linear search through `incoming_senders` Vec
- **Automatic Cleanup**: Orphaned weights are automatically removed when edges are deleted
- **Normalized Weight Bounds**: Stable learning through automatic weight normalization (L1, L2, Softmax)
- **Event-Driven Architecture**: Topology changes automatically synchronize agent state
- **Better Memory Management**: No dangling references after network rewiring

---

## Before: Old Implementation

```rust
// OLD WAY - CognitivezooidState
pub struct CognitivezooidState {
    pub izh_state: IzhikevichState,
    pub izh_params: IzhikevichParams,
    pub tick_count: u64,
    pub spike_count: u64,
    pub last_pre_spike_times: HashMap<ZoooidId, u64>,
    
    // ❌ OLD: Two parallel Vecs that must be kept in sync
    pub incoming_senders: Vec<ZoooidId>,      // O(N) lookup
    pub synaptic_weights: Vec<f32>,           // Parallel array
}
```

### Problems with Old Approach

```rust
// ❌ O(N) lookup for every spike
if !state.incoming_senders.contains(sender) {  // Linear scan!
    state.incoming_senders.push(*sender);
    state.synaptic_weights.push(0.1);
}

// ❌ Manual index management
for (i, &sender) in state.incoming_senders.iter().enumerate() {
    state.synaptic_weights[i] += dw;  // Easy to get indices wrong
}

// ❌ No automatic cleanup when topology changes
// If edge A→B is removed, B's synapse from A is never cleaned up
```

---

## After: New Implementation

```rust
// NEW WAY - CognitivezooidState with SynapseManager
pub struct CognitivezooidState {
    pub izh_state: IzhikevichState,
    pub izh_params: IzhikevichParams,
    pub tick_count: u64,
    pub spike_count: u64,
    pub last_pre_spike_times: HashMap<ZoooidId, u64>,
    
    // ✅ NEW: Single SynapseManager with O(1) lookup
    pub synapse_manager: SynapseManager,  // HashMap-based, O(1) average
}
```

### Benefits of New Approach

```rust
// ✅ O(1) lookup
let _ = state.synapse_manager.add_synapse(*sender, 0.1);  // Safe, idempotent

// ✅ Safe index management through API
let weight = state.synapse_manager.get_weight(sender)?;  // O(1)
let _ = state.synapse_manager.update_weight(sender, delta, tick)?;

// ✅ Automatic cleanup via events
// When topology.remove_edge(A, B) is called:
// 1. Event published: TopologyEvent::EdgeRemoved { A, B }
// 2. Agent B's synapse_manager processes event
// 3. Synapse from A is automatically removed
```

---

## Migration Steps

### Step 1: Update CognitivezooidState

```rust
// BEFORE
#[derive(Debug, Clone)]
pub struct CognitivezooidState {
    pub izh_state: IzhikevichState,
    pub izh_params: IzhikevichParams,
    pub tick_count: u64,
    pub spike_count: u64,
    pub last_pre_spike_times: HashMap<ZoooidId, u64>,
    pub incoming_senders: Vec<ZoooidId>,
    pub synaptic_weights: Vec<f32>,
}

// AFTER
use crate::agent::synapse_manager::{SynapseManager, SynapseConfig, NormMode};

#[derive(Debug, Clone)]
pub struct CognitivezooidState {
    pub izh_state: IzhikevichState,
    pub izh_params: IzhikevichParams,
    pub tick_count: u64,
    pub spike_count: u64,
    pub last_pre_spike_times: HashMap<ZoooidId, u64>,
    pub synapse_manager: SynapseManager,  // ← New field
}
```

### Step 2: Update Initialization

```rust
// BEFORE
async fn initialize(&self) -> Result<Box<dyn Any + Send + Sync>, ...> {
    Ok(Box::new(CognitivezooidState {
        izh_state: ...,
        izh_params: ...,
        tick_count: 0,
        spike_count: 0,
        last_pre_spike_times: HashMap::new(),
        incoming_senders: Vec::new(),      // ❌ Old
        synaptic_weights: Vec::new(),      // ❌ Old
    }))
}

// AFTER
async fn initialize(&self) -> Result<Box<dyn Any + Send + Sync>, ...> {
    let synapse_config = SynapseConfig {
        norm_mode: NormMode::L1,         // ✅ Automatic weight normalization
        max_connections: 50,              // ✅ Protection from memory explosion
        min_weight: 0.0,
        max_weight: 1.0,
        default_weight: 0.1,
    };

    Ok(Box::new(CognitivezooidState {
        izh_state: ...,
        izh_params: ...,
        tick_count: 0,
        spike_count: 0,
        last_pre_spike_times: HashMap::new(),
        synapse_manager: SynapseManager::new(synapse_config),  // ✅ New
    }))
}
```

### Step 3: Update Spike Processing

```rust
// BEFORE
for (sender, msg) in &inputs {
    if let Message::SpikeEvent { .. } = msg {
        state.last_pre_spike_times.insert(*sender, state.tick_count);
        if !state.incoming_senders.contains(sender) {  // ❌ O(N)!
            state.incoming_senders.push(*sender);      // ❌ Index sync problem
            state.synaptic_weights.push(0.1);
        }
    }
}

// AFTER
for (sender, msg) in &inputs {
    if let Message::SpikeEvent { .. } = msg {
        state.last_pre_spike_times.insert(*sender, state.tick_count);
        let _ = state.synapse_manager.add_synapse(*sender, 0.1);  // ✅ O(1), safe
    }
}
```

### Step 4: Update STDP Learning

```rust
// BEFORE
if has_spiked {
    for (i, &sender) in state.incoming_senders.iter().enumerate() {
        if let Some(&pre_time) = state.last_pre_spike_times.get(&sender) {
            let delta_t = (state.tick_count as f32) - (pre_time as f32);
            let dw = ...;
            state.synaptic_weights[i] += dw;           // ❌ Index-based access
            state.synaptic_weights[i] *= 1.0 - self.params.weight_decay;
            if state.synaptic_weights[i].abs() < self.params.pruning_threshold {
                state.synaptic_weights[i] = 0.0;       // ❌ Manual pruning
            }
        }
    }
}

// AFTER
if has_spiked {
    for sender in state.synapse_manager.get_sources() {  // ✅ Iterate through manager
        if let Some(&pre_time) = state.last_pre_spike_times.get(&sender) {
            let delta_t = (state.tick_count as f32) - (pre_time as f32);
            let dw = ...;
            
            let decay_factor = 1.0 - self.params.weight_decay;
            let actual_delta = dw * decay_factor;
            
            let _ = state.synapse_manager.update_weight(sender, actual_delta, state.tick_count);  // ✅ Safe API
            
            // ✅ Automatic pruning
            if let Some(weight) = state.synapse_manager.get_weight(sender) {
                if weight.abs() < self.params.pruning_threshold {
                    let _ = state.synapse_manager.remove_synapse(sender);
                }
            }
        }
    }
    
    // ✅ Automatic normalization (stable learning)
    state.synapse_manager.normalize();
}
```

---

## API Reference: SynapseManager

### Creating a Manager

```rust
use crate::agent::synapse_manager::{SynapseManager, SynapseConfig, NormMode};

// Default configuration
let manager = SynapseManager::new(SynapseConfig::default());

// Custom configuration
let config = SynapseConfig {
    norm_mode: NormMode::L2,              // L1, L2, Softmax, Adaptive, None
    max_connections: 100,                  // Limit incoming connections
    min_weight: -1.0,                      // Custom range
    max_weight: 1.0,
    default_weight: 0.0,
};
let manager = SynapseManager::new(config);
```

### Basic Operations

```rust
// Add synapse (O(1) average)
manager.add_synapse(source_id, 0.5)?;

// Get weight (O(1) average)
if let Some(weight) = manager.get_weight(source_id) {
    println!("Weight: {}", weight);
}

// Update weight (O(1) average)
manager.update_weight(source_id, delta, current_tick)?;

// Remove synapse (O(N) worst case due to swap-remove)
if let Some(weight) = manager.remove_synapse(source_id) {
    println!("Removed weight: {}", weight);
}

// Normalize all weights
manager.normalize();
```

### Batch Operations

```rust
// Get all source IDs
let sources = manager.get_sources();  // O(M) where M = num synapses

// Get statistics
let count = manager.get_active_count();      // O(1)
let sum = manager.sum_weights();             // O(M)
let avg = manager.avg_weight();              // O(M)

// Clear all
manager.clear();                             // O(1) amortized
```

---

## Normalization Modes

| Mode | Formula | When to Use |
|------|---------|------------|
| **None** | `w = clamp(w, min, max)` | Debugging, custom learning rules |
| **L1** | `w /= sum(\|w_i\|)` | Stable STDP, ensures bounded spikes |
| **L2** | `w /= sqrt(sum(w_i²))` | Euclidean regularization |
| **Softmax** | `w = exp(w) / sum(exp(w_j))` | Probabilistic interpretation, attention |
| **Adaptive** | `w = decay*w + lr*grad` | Applied during update_weight, not in normalize() |

---

## Topology Event Integration

### For Topology Modifications

```rust
use crate::core::{TopologyEvent, TopologyEventBus};
use std::sync::Arc;

// Create event bus
let event_bus = Arc::new(TopologyEventBus::new(1024));

// Attach to topology
let mut topology = ZoooidTopology::new();
topology.set_event_bus(event_bus);

// Now when edges are added/removed, events are automatically published
topology.add_edge(from, to, props);           // ✅ Publishes EdgeAdded event
topology.remove_edge(from, to);              // ✅ Publishes EdgeRemoved event
```

### Processing Events in Agent

```rust
// In your agent's update() loop:
let event = TopologyEvent::EdgeRemoved { from, to };
let actions = state.synapse_manager.on_topology_event(&event);
// ✅ Synapse from `from` is automatically removed!
```

---

## Common Patterns

### Pattern 1: Bulk Import from Old System

```rust
// Converting old Vec-based state to new system
fn migrate_state_old_to_new(old_state: &OldCognitivezooidState) -> CognitivezooidState {
    let mut synapse_config = SynapseConfig::default();
    synapse_config.max_connections = old_state.incoming_senders.len();
    
    let mut manager = SynapseManager::new(synapse_config);
    
    // Load old weights into manager
    for (sender, weight) in old_state.incoming_senders.iter()
        .zip(old_state.synaptic_weights.iter()) {
        let _ = manager.add_synapse(*sender, *weight);
    }
    
    CognitivezooidState {
        izh_state: old_state.izh_state.clone(),
        izh_params: old_state.izh_params.clone(),
        tick_count: old_state.tick_count,
        spike_count: old_state.spike_count,
        last_pre_spike_times: old_state.last_pre_spike_times.clone(),
        synapse_manager: manager,
    }
}
```

### Pattern 2: Synaptic Plasticity Rule

```rust
fn apply_hebbian_rule(
    manager: &mut SynapseManager,
    sender: ZoooidId,
    post_activity: f32, 
    pre_activity: f32,
    learning_rate: f32,
) -> Result<(), String> {
    let delta = learning_rate * post_activity * pre_activity;
    manager.update_weight(sender, delta, 0)?;
    Ok(())
}
```

### Pattern 3: Monitoring Weight Statistics

```rust
fn log_synapse_health(manager: &SynapseManager) {
    println!("Active synapses: {}", manager.get_active_count());
    println!("Average weight: {:.4}", manager.avg_weight());
    println!("Total weight sum: {:.4}", manager.sum_weights());
    
    if manager.get_active_count() > 0 {
        println!("Weight range: {:.4} - {:.4}", 
            manager.config().min_weight,
            manager.config().max_weight);
    }
}
```

---

## Testing Your Migration

### Unit Tests

```rust
#[test]
fn test_migrated_agent_spike_response() {
    // Create new agent with SynapseManager
    let blueprint = CognitivezooidBlueprint { 
        params: CognitivezooidParams::default() 
    };
    
    let mut state = blueprint.initialize().await.unwrap();
    
    // Simulate spike input
    let inputs = vec![(test_id(1), Message::SpikeEvent { timestamp: 0, amplitude: 1.0 })];
    
    // Verify synapse was auto-added
    let result = blueprint.update(&mut state, inputs, &topology, None).await;
    assert!(result.is_ok());
}
```

### Performance Validation

```bash
# Run benchmarks to verify O(1) lookup performance
cargo bench --bench synapse_lookup_bench -- --measurement-time 5

# Profile memory usage
valgrind --tool=massif --massif-out-file=massif.out \
    target/release/your_app

# Check for memory leaks
valgrind --leak-check=full \
    target/release/your_app
```

---

## Debugging Tips

### Issue: Weights Not Updating

**Symptom**: Synapses added but weights don't change

**Solution**:
1. Check event bus is connected: `assert!(topology.event_bus.is_some())`
2. Verify `normalize()` is called after updates
3. Check weight bounds: `println!("{:?}", manager.config())`

### Issue: Memory Growing Unbounded

**Symptom**: RAM usage grows with time

**Solution**:
1. Set reasonable `max_connections` limit
2. Verify pruning threshold is not too small
3. Check `get_active_count()` doesn't exceed limits
4. Profile with `cargo flamegraph` or `valgrind`

### Issue: Events Not Propagating

**Symptom**: Edge removed but synapse stays

**Solution**:
1. Verify `set_event_bus()` called on topology
2. Check subscribers are listening: `event_bus.subscriber_count()`
3. Verify `on_topology_event()` called in agent
4. Add debug logging: `tracing::debug!(...)`

---

## Backward Compatibility

- **Old code**: Still works if you don't update state structure
- **Gradual migration**: Create new agents with `SynapseManager`, keep old ones as-is
- **Interoperability**: Mix old and new agents in same topology (with caveats)

---

## Further Reading

- [SynapseManager API Docs](../src/agent/synapse_manager.rs)
- [TopologyEventBus Architecture](../src/core/topology_events.rs)
- [Cognitivezooid Integration](../src/agent/blueprint/cognitivezooid.rs)
- [Synapse Lifecycle Diagram](./synapse_lifecycle.md)

---

## Questions?

For issues or clarifications:
1. Check existing tests: `src/agent/synapse_manager.rs`
2. Run benchmarks: `cargo bench --bench synapse_lookup_bench`
3. Review examples: `examples/` directory
