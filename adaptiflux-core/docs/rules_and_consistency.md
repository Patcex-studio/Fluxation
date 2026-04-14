<!--
Copyright (C) 2026 Jocer S. <patcex@proton.me>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

SPDX-License-Identifier: AGPL-3.0 OR Commercial
-->

# Rules Engine and Consistency Checks Implementation

## Overview

The **Rules Engine** and **Consistency Checks** system implements dynamic behavior adaptation and topology self-organization in Adaptiflux. These components embody the principles of **self-organization**, **adaptation**, and **stability** described in the architectural process.

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                    RuleEngine                            │
├─────────────────────────────────────────────────────────┤
│  • BehaviorRules     → Agent-level adaptations           │
│  • TopologyRules     → Network-level transformations     │
│  • ConsistencyChecks → System integrity validation       │
└─────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
    │  Behavior   │  │  Topology    │  │ Consistency     │
    │  Rules      │  │  Rules       │  │ Checks          │
    ├─────────────┤  ├──────────────┤  ├─────────────────┤
    │ • LoadBal.  │  │ • Growth     │  │ • Connected     │
    │ • RoleAdapt │  │ • Proximity  │  │ • NoIsolated    │
    │ • Isolation │  │ • SelfHealing│  │ • MinConnectiv. │
    │   Recovery  │  │ • Diameter   │  │ • MinDegree     │
    │             │  │ • Clustering │  │ • MaxDiameter   │
    └─────────────┘  └──────────────┘  └─────────────────┘
```

## Key Features

### 1. Enhanced System Metrics

`SystemMetrics` now provides comprehensive network analysis:

- **total_zoooids**: Total agent count
- **total_connections**: Total edge count  
- **avg_connectivity**: Average degree (normalized)
- **clustering_coefficient**: Local cluster density (0-1)
- **network_diameter**: Maximum distance between nodes
- **agent_count**: For backward compatibility

### 2. Behavior Rules

Behavior rules enable agents to adapt their role and parameters based on local conditions:

#### Available Rules:

**LoadBalancingRule**
- Monitors incoming degree
- Signals high-load state when degree exceeds threshold
- Enables load distribution strategies

**RoleAdaptationRule**
- Adapts agent role based on network density
- High density → Cognitive role (coordination)
- Low density → Sensor role (coverage)

**IsolationRecoveryRule**
- Detects poorly connected agents
- Signals recovery actions when connectivity is below threshold
- Enables network self-repair

### 3. Topology Rules

Topology rules manage network structure modifications:

#### Available Rules:

**SimpleTopologyGrowthRule**
- Progressive network expansion
- Respects maximum degree constraints
- Ensures scalable growth

**ProximityConnectionRule** *(Template)*
- Connects agents based on spatial proximity
- Requires interaction frequency data
- Enables locality-aware topologies

**SelfHealingRule** *(Template)*
- Removes connections to unresponsive agents
- Requires external health monitoring
- Enables fault tolerance

**DiameterOptimizationRule**
- Adds shortcuts when network diameter exceeds threshold
- Reduces message hop count
- Improves communication efficiency

**LocalClusteringRule**
- Increases clustering coefficient
- Strengthens local communities
- Improves local information flow

### 4. Consistency Checks

Consistency checks validate system integrity:

#### Available Checks:

**ConnectedTopologyCheck**
- Ensures single connected component
- Detects network fragmentation
- Error: DisconnectedComponent with component count

**NoIsolatedNodesCheck**
- Verifies all nodes have at least one connection
- Prevents orphaned agents
- Error: IsolatedNode with node ID

**NodeCountConsistencyCheck**
- Synchronizes topology nodes with agent list
- Detects zombie topology entries
- Error: NodeCountMismatch with counts

**MinConnectivityCheck**
- Enforces minimum average degree
- Ensures sufficient network density
- Error: MetricViolation with expected/actual values

**MinimumDegreeCheck**
- Guarantees minimum connections per node
- Prevents isolated clusters
- Error: InsufficientConnectivity with node info

**MaxDiameterCheck**
- Enforces maximum network diameter
- Ensures bounded message propagation
- Error: DiameterViolation with diameter info

## Integration with CoreScheduler

The complete rule cycle runs synchronously:

```
1. Evaluate behavior rules for all agents
   ↓ (generates BehaviorActions)
   
2. Calculate system metrics from current topology
   ↓
   
3. Evaluate topology rules
   ↓ (generates TopologyActions)
   
4. Apply topology modifications
   ↓
   
5. Run consistency checks
   ↓ (validates system state)
   
6. Report errors or success
```

## Error Handling

### ConsistencyError Enum

```rust
pub enum ConsistencyError {
    DisconnectedComponent { component_count: usize },
    IsolatedNode { node_id: ZoooidId },
    CycleDetected,
    MetricViolation { metric_name, expected, actual },
    NodeCountMismatch { topology_count, agent_count },
    InsufficientConnectivity { node_id, degree, minimum },
    DiameterViolation { diameter, max_diameter },
    Custom { reason: String },
}
```

All errors implement `std::error::Error` for ergonomic handling.

## Usage Examples

### Creating a Rule Engine

```rust
let mut engine = RuleEngine::new();

// Add behavior rules
engine.add_behavior_rule(Box::new(
    LoadBalancingRule::new(0.8, 10)
));
engine.add_behavior_rule(Box::new(
    RoleAdaptationRule::new(2.5, 0.8)
));

// Add topology rules
engine.add_topology_rule(Box::new(
    SimpleTopologyGrowthRule::new(5)
));
engine.add_topology_rule(Box::new(
    DiameterOptimizationRule::new(6, 0.1)
));

// Add consistency checks
engine.add_consistency_check(Box::new(
    ConnectedTopologyCheck::new()
));
engine.add_consistency_check(Box::new(
    MinConnectivityCheck::new(1.0)
));
```

### Running the Complete Cycle

```rust
let (behavior_actions, topology_changes) = engine.run_full_cycle(
    &agent_states,
    &topology_mutex,
    &message_bus,
    &agent_ids,
).await?;

println!("Applied {} topology changes", topology_changes);
for (agent_id, actions) in behavior_actions {
    println!("Agent {} needs: {:?}", agent_id, actions);
}
```

## Design Principles

### 1. **Composability**
- Rules and checks are independent
- Can be mixed and matched
- Enable custom domain-specific logic

### 2. **Scalability**
- Asynchronous evaluation
- Lazy computation of metrics
- Efficient graph algorithms

### 3. **Safety**
- Type-safe error handling
- Strong consistency guarantees
- Prevents invalid state transitions

### 4. **Extensibility**
- Trait-based architecture
- Template rules for common patterns
- Custom error variants

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Connectivity check | O(V+E) | BFS single component |
| Clustering coefficient | O(V·K²) | K = avg degree |
| Network diameter | O(V·(V+E)) | Sampled computation |
| Topology growth | O(V) | Finding valid edge |
| Apply topology action | O(1) | Direct graph update |

Where V = vertices, E = edges

## Testing

Comprehensive test suite in `tests/rules_tests.rs`:

- ✅ System metrics calculation
- ✅ All consistency checks
- ✅ Topology rule evaluation
- ✅ RuleEngine integration
- ✅ Error variants and handling
- ✅ Network topology scenarios

Run with:
```bash
cargo test --test rules_tests
```

## Future Enhancements

1. **Rule Composition**: Combine rules with logical operators (AND, OR)
2. **Rule Scheduling**: Periodic vs. event-driven execution
3. **Rule Statistics**: Track rule application frequency and effectiveness
4. **Dynamic Rule Loading**: Runtime rule deployment
5. **Constraint Satisfaction**: Goal-seeking topology adaptation
6. **Reinforcement Learning**: Data-driven rule parameter tuning

## Integration Points

### With CoreScheduler
- Receives agent states and topology
- Returns actions for application
- Validates final state consistency

### With AgentBlueprints
- Uses agent states for evaluation
- Generates role change actions
- Enables strategy switching

### With MessageBus
- Available for message pattern analysis
- Enables interaction-based rules
- Supports communication metrics

### With Resource Manager
- Can consider resource constraints
- Enables resource-aware adaptation
- Supports capacity-based decisions

## References

- Architectural Process Document
- "От клеточного автомата..." (Self-organization theory)
- Network Science principles (clustering, diameter)
- Distributed Systems consistency models
