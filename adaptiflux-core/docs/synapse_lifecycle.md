# Synapse Lifecycle Architecture

## Overview

`SynapseManager` centralizes incoming synapse state for an agent and keeps it consistent with topology events.
It uses a compact `Vec<SynapseEntry>` plus an `incoming_map: HashMap<ZoooidId, usize>` for fast lookup and efficient cleanup.

## Event-driven synapse state

`TopologyEventBus` broadcasts topology events to subscribers.
`SynapseManager::on_topology_event` handles the current event set and updates local synapse state.

### Topology events supported

- `TopologyEvent::EdgeAdded { from, to, initial_weight }`
- `TopologyEvent::EdgeRemoved { from, to }`
- `TopologyEvent::WeightUpdated { from, to, new_weight }`
- `TopologyEvent::TopologySnapshot { edges }`

Snapshots use the edge format `Vec<(ZoooidId, ZoooidId, f32)>`.

## SynapseManager internals

`SynapseManager` stores:

- `incoming_map: HashMap<ZoooidId, usize>` — maps source IDs to indices in the vector
- `weights: Vec<SynapseEntry>` — compact list of synapses
- `config: SynapseConfig` — bounds, normalization mode, and max connections
- `update_count: u64`

### Synapse entry

Each `SynapseEntry` contains:

- `source_id`
- `weight`
- `last_updated`
- `metadata`

### Connection limits

`SynapseConfig::default()` sets `max_connections = 50`.
`add_synapse` returns `Err` if this limit is exceeded.

## Add, update, remove

### `add_synapse`

- Clamps weight to `[min_weight, max_weight]`
- Updates existing synapse if the source already exists
- Adds a new entry otherwise
- Enforces `max_connections`

### `remove_synapse`

- Removes the synapse by source ID
- Uses swap-remove to keep the vector compact
- Updates `incoming_map` for the swapped entry

### `update_weight`

- Applies a delta to an existing weight
- Clamps the result to `[min_weight, max_weight]`
- Stores the provided `tick` in `last_updated`
- Increments `update_count`

### `get_weight`

- Returns the current weight for a source ID
- O(1) average case via `HashMap` lookup

## Normalization modes

`NormMode` controls normalization:

- `None` — clamp only
- `L1` — `sum(|w_i|) == 1.0`
- `L2` — `sqrt(sum(w_i^2)) == 1.0`
- `Softmax` — `exp(w_i) / sum(exp(w_j))`
- `Adaptive` — no extra normalization, clamp only

`SynapseManager::normalize()` applies the configured mode.

## Topology event handling

`on_topology_event` handles event targets:

- `EdgeAdded` → `add_synapse(from, initial_weight)`
- `EdgeRemoved` → `remove_synapse(from)`
- `WeightUpdated` → adjust the stored weight based on the delta
- `TopologySnapshot` → remove stale synapses and add missing ones

The manager returns descriptions of actions taken for logging.

## TopologyEventBus semantics

`TopologyEventBus` is a Tokio broadcast channel with a configurable buffer.
Subscribers receive `TopologyEvent` messages and may encounter `broadcast::RecvError::Lagged` if they fall behind.
If lag occurs, the subscriber can resynchronize using a `TopologySnapshot` event.

### Event publication

- `TopologyEventBus::new(buffer_size)` creates the bus
- `TopologyEventBus::subscribe()` returns a receiver
- `TopologyEventBus::publish(event)` sends the event to all active subscribers

## Practical lifecycle

1. Topology adds an edge.
2. It publishes `TopologyEvent::EdgeAdded`.
3. Agent subscribers receive the event.
4. `SynapseManager::on_topology_event` updates local synapses.
5. On edge removal, `EdgeRemoved` triggers swap-remove cleanup.
6. On snapshot, manager synchronizes local state to the topology.

## Runtime behavior

- `get_sources()` returns all pre-synaptic source IDs.
- `sum_weights()` and `avg_weight()` provide aggregate synapse metrics.
- `clear()` removes all synapses.

## Implementation notes

- `SynapseManager` avoids dangling references by keeping topology state consistent with events.
- `remove_synapse` uses swap-remove and updates the index map.
- `TopologySnapshot` synchronization is incremental: stale entries are removed, missing edges are added.
- The manager does not itself compute STDP rules; it only stores and normalizes weights.

## Testing guarantees

The current code includes tests for:

- adding and removing synapses
- weight lookup and swap-remove integrity
- L1 normalization
- connection limit enforcement
- topology event handling for edge addition, removal, and snapshots

## Reality vs older design

The implementation is not a speculative architecture document. It reflects the current code:

- `TopologyEventBus` is based on Tokio broadcast.
- `SynapseManager` uses `HashMap<ZoooidId, usize>` and `Vec<SynapseEntry>`.
- The lifecycle is driven by `TopologyEvent` events, not by implicit manual bookkeeping.
- `TopologySnapshot` events carry full edge lists for resynchronization.
