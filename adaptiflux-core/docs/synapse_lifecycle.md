# Synapse Lifecycle Architecture

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SYNAPSE LIFECYCLE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

TIME ──────────────────────────────────────────────────────────────────────────>

 1. CREATION PHASE
    ─────────────────────────────────────────────────────────────────────────

    Topology               CoreScheduler             CognitivezooidAgent
    ─────────              ─────────────             ─────────────────
        │                      │                           │
        ├─ add_edge(A→B) ─────>│                           │
        │                      │                           │
        ├─ Publish:EdgeAdded   │  ┌─────────────────────>  │
        │                      │  │                        │
        │                      │  └─── on_topology_event() │
        │                      │                           │
        │                      │  synapse_manager         │
        │                      │  .add_synapse(A, 0.1)    │
        │                      │                           │
        │                      │  ✅ Synapse Created      │
        │                      │  incoming_map[A] = 0     │
        │                      │  weights[0] = 0.1        │

 2. ACTIVE PHASE (During Network Operation)
    ─────────────────────────────────────────────────────────────────────────

    Agent A              TopologyEventBus           Agent B (Cognitivezooid)
    ────────             ────────────────           ──────────────────────
        │                     │                           │
        ├─ Spike ──────────>  │                           │
        │  (SpikeEvent)       │                           │
        │                     ├──────────────────────>   │
        │                     │ on_topology_event()      │
        │                     │ (if EdgeAdded)           │
        │                     │                           │
        │                     │                    ┌──── │
        │                     │                    │      │
        │                     │     Update Synapse Manager:
        │                     │                    │
        │                    "▼"                   "▼"
        │            SynapseManager                │
        │            ─────────────────             │
        │            incoming_map[A] = 0  ✅ O(1) lookup
        │            weights[0].weight = 0.1
        │
        │         Get Weight (O(1) average):
        │         ────────────────────────
        │         weight = manager.get_weight(A)
        │         │
        │         └──> HashMap lookup ──> Return 0.1
        │
        │         Update Weight (STDP):
        │         ──────────────────
        │         manager.update_weight(A, ΔW, tick)
        │         │
        │         ├──> ΔW = STDP_rule(Δt)
        │         ├──> w_new = w_old + ΔW
        │         ├──> Clamp to [min, max]
        │         └──> Store in weights[0]
        │
        │         Normalize Weights:
        │         ──────────────────
        │         manager.normalize() {
        │           match norm_mode {
        │             L1      => w /= sum(|w_i|)
        │             L2      => w /= sqrt(sum(w_i²))
        │             Softmax => w = exp(w) / sum(exp(w_j))
        │             None    => w = clamp(w, min, max)
        │           }
        │         }

 3. TERMINATION PHASE (Edge Removal)
    ─────────────────────────────────────────────────────────────────────────

    Topology               TopologyEventBus          Agent B
    ─────────              ────────────────          ───────
        │                      │                       │
        ├─ remove_edge(A→B) ───│                       │
        │                      │                       │
        ├─ Publish:            │                       │
        │  EdgeRemoved(A,B)    │                       │
        │                      │                       │
        │                      ├──────────────────>    │
        │                      │ on_topology_event()   │
        │                      │ (EdgeRemoved)         │
        │                      │                       │
        │                      │  synapse_manager     │
        │                      │  .remove_synapse(A)  │
        │                      │                       │
        │                      │  ┌─────────────────  │
        │                      │  │ Swap-Remove:     │
        │                      │  │ ─────────────     │
        │                      │  │ idx = map[A] = 0  │
        │                      │  │ map.remove(A)     │
        │                      │  │ If 0 < len-1:     │
        │                      │  │   last = weights  │
        │                      │  │   weights[0] = last
        │                      │  │   map[last.id] = 0
        │                      │  │ weights.pop()     │
        │                      │  └──────────────────  │
        │                      │                       │
        │                      │  ✅ Synapse Removed  │
        │                      │  ❌ No memory leak    │

 4. ERROR HANDLING PHASE
    ─────────────────────────────────────────────────────────────────────────

    Error Condition                    Mitigation
    ───────────────                    ──────────
    Max connections exceeded   ──>  Return Err, silently fail add_synapse
                                    Old: crash with panic
                                    New: graceful degradation

    Weight tracking mismatch  ──>   TopologyEventBus keeps sync:
                                    Automatic remove on EdgeRemoved
                                    No manual index tracking needed

    Subscriber lag in events  ──>   Broadcast buffer (>1024 events)
                                    If overflow: subscriber gets Lagged
                                    Recovery: request TopologySnapshot

    Orphaned weights          ──>   Automatic cleanup on topology change
                                    No dangling references
                                    Memory is reclaimed


DETAILED SEQUENCE: STDP Update Cycle
════════════════════════════════════════════════════════════════════════════

 t₀=100ms: Spike from A
     │
     ├─> Recording: last_pre_spike_times[A] = 100
     │
     └─> state.synapse_manager.add_synapse(A, 0.1)
           │
           ├─ HashMap insert: incoming_map[A] = 0
           │
           └─ Vector push: weights = [SynapseEntry {
                  source_id: A,
                  weight: 0.1,
                  last_updated: 0,
                  metadata: None
              }]

 t₁=105ms: Post-synaptic spike in B
     │
     ├─> has_spiked = true
     │
     └─> For each source in manager.get_sources():
           │ (e.g., A)
           │
           ├─> delta_t = (105 - 100) = +5ms  [pre BEFORE post]
           │
           ├─> dw = stdp_a_plus * exp(-5 / tau_plus)
           │   ≈ 0.01 * exp(-5 / 20)
           │   ≈ 0.01 * 0.779
           │   ≈ 0.0078
           │
           ├─> actual_delta = 0.0078 * (1 - 0.0001)  [weight decay]
           │                 ≈ 0.00779
           │
           └─> manager.update_weight(A, 0.00779, 105)
                 │
                 ├─> new_weight = 0.1 + 0.00779 = 0.10779
                 │
                 ├─> Clamp to [0.0, 1.0]: 0.10779
                 │
                 └─> Store: weights[0].weight = 0.10779

 t₁ (cont.): Normalization
     │
     └─> manager.normalize()  [norm_mode = L1]
           │
           ├─> sum_abs = |0.10779| = 0.10779
           │
           └─> new_weight = 0.10779 / 0.10779 = 1.0
                 └─> (Normalized to L1 unit ball)


EVENT BUS ARCHITECTURE
══════════════════════════════════════════════════════════════════════════════

 TopologyEventBus (Global)
 ────────────────
     │
     ├─ Sender<TopologyEvent>     ◄─ Published events
     │                              (add_edge, remove_edge, etc.)
     │
     ├─ Broadcast Channel (1024 slots)
     │  │
     │  ├─ Slot 0: EdgeAdded(A→B, w=0.5)
     │  ├─ Slot 1: EdgeAdded(C→B, w=0.3)
     │  ├─ Slot 2: EdgeRemoved(A→B)
     │  ├─ Slot 3: TopologySnapshot(edges=[...])
     │  └─ ...
     │
     └─ Subscribers (per agent)
        │
        ├─ Agent B.rx: Receiver<TopologyEvent>  ◄─ Listens to all events
        │                                         Processes only those
        │                                         where targets(self)
        │
        └─ Every update():
            │
            └─> Check for events:
                ├─ EdgeAdded(from, to, w)?
                │  └─ If to == self: add_synapse(from, w)
                │
                ├─ EdgeRemoved(from, to)?
                │  └─ If to == self: remove_synapse(from)
                │
                └─ TopologySnapshot(edges)?
                   └─ Sync all synapses to match snapshot


MEMORY MANAGEMENT
═════════════════════════════════════════════════════════════════════════════

Before (Vec-based):
   Memory Layout for 50 synapses:
   ────────────────────────────────
   incoming_senders: Vec<ZoooidId>  (50 × 16 bytes)  = 800 bytes
   synaptic_weights: Vec<f32>       (50 × 4 bytes)   = 200 bytes
   ─────────────────────────────────────────────────────────────
   Total: ~1000 bytes + Vec allocation overhead
   
   ❌ Problem: When topology changes, no automatic cleanup
   ❌ Dangling references can accumulate over 24h+ runtime

After (SynapseManager):
   Memory Layout for 50 synapses:
   ────────────────────────────────
   incoming_map: HashMap<Uuid, usize>     ≈ 800 bytes + hash overhead
   weights: Vec<SynapseEntry>             ≈ 2500 bytes (50 × 50 bytes)
                                             per entry has source_id (16) +
                                             weight (4) + timestamp (8) + metadata
   ─────────────────────────────────────────────────────────────
   Total: ≈ 3300 bytes (slightly more but O(1) lookup)
   
   ✅ Automatic cleanup via events
   ✅ No dangling references
   ✅ Deterministic memory usage


ERROR RECOVERY
════════════════════════════════════════════════════════════════════════════════

Scenario: Event bus buffer overflow (>1024 events published)

Before (manual sync):
   ❌ Agent doesn't know topology changed
   ❌ Continues using stale weights
   ❌ Learning becomes incorrect
   ❌ Memory leak potential

After (event-driven):
   1. Agent's RX gets broadcast::error::RecvError::Lagged
   2. Agent logs warning
   3. Requests TopologySnapshot event
   4. Server publishes current full topology
   5. Agent syncs: removes orphaned, adds missing
   6. Continues with fresh state
   
   ✅ Automatic recovery
   ✅ No data corruption
   ✅ Bounded memory
