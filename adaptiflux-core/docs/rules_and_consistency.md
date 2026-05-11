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

The `RuleEngine` orchestrates behavior rules, topology rules, plasticity rules, and consistency checks.
It operates on the current topology and agent state, applies topology modifications, and validates system integrity.

## Architecture

### RuleEngine components

- `behavior_rules: Vec<Box<dyn BehaviorRule>>`
- `topology_rules: Vec<Box<dyn TopologyRule>>`
- `plasticity_rules: Vec<Box<dyn PlasticityRule>>`
- `consistency_checks: Vec<Box<dyn ConsistencyCheck>>`

### Execution flow

1. Evaluate behavior rules for each agent.
2. Compute `SystemMetrics` from the current topology.
3. Run topology rules against the metrics.
4. Run plasticity rules against topology state and context.
5. Apply topology actions to the graph.
6. Run consistency checks against the updated topology.

## System Metrics

`SystemMetrics` is built from the topology and includes:

- `total_zoooids`: the number of agents/nodes in the topology
- `total_connections`: the number of edges in the topology
- `avg_connectivity`: average degree computed as `2 * edge_count / node_count`
- `clustering_coefficient`: simplified average clustering coefficient
- `network_diameter`: approximate maximum depth from sampled nodes
- `agent_count`: alias for `total_zoooids`

## Behavior Rules

Behavior rules inspect an agent's state, topology, and messages and may produce actions.
The current implementation does not define a fixed set of built-in rules in `rules/behavior/`; custom rules are registered through `engine.add_behavior_rule(...)`.

## Topology Rules

Topology rules generate `TopologyAction` values that modify the graph.
Built-in topology action variants include:

- `TopologyAction::AddEdge(source, target)`
- `TopologyAction::RemoveEdge(source, target)`
- `TopologyAction::PruneEdge { from, to, .. }`
- `TopologyAction::StrengthenConnection { from, to, delta_weight }`
- `TopologyAction::WeakenConnection { from, to, delta_weight }`
- `TopologyAction::CreateZoooid(blueprint)`
- `TopologyAction::CreateAgentFromTemplate { template_blueprint, target_area_hint }`
- `TopologyAction::TerminateZoooid(id)`
- `TopologyAction::InitiateApoptosis { agent_id, reason }`
- `TopologyAction::GroupAgents(ids)`

`RuleEngine::apply_topology_actions` applies these actions against the locked topology and produces `AppliedTopologyEffects`.

## Plasticity Rules

Plasticity rules are evaluated after topology rules and may be skipped for growth-oriented rules when topology density is high.
The engine currently identifies growth rules by type name containing `Synaptogenesis` or `Neurogenesis`.

## Consistency Checks

Consistency checks are implemented via the `ConsistencyCheck` trait.
Each check inspects the topology and agent list and returns `Result<(), ConsistencyError>`.

### Available checks

- `ConnectedTopologyCheck`
- `NoIsolatedNodesCheck`
- `NodeCountConsistencyCheck`
- `MinConnectivityCheck`
- `MinimumDegreeCheck`
- `MaxDiameterCheck`

### Notes

- A `CycleDetected` error variant exists in `ConsistencyError`, but there is no built-in check for it in `rules/consistency/checks.rs`.
- `MaxDiameterCheck` uses a simplified diameter estimate over a small sampled subset of nodes for performance.

## Consistency Errors

```rust
pub enum ConsistencyError {
    DisconnectedComponent { component_count: usize },
    IsolatedNode { node_id: ZoooidId },
    CycleDetected,
    MetricViolation { metric_name: String, expected: String, actual: String },
    NodeCountMismatch { topology_count: usize, agent_count: usize },
    InsufficientConnectivity { node_id: ZoooidId, degree: usize, minimum: usize },
    DiameterViolation { diameter: usize, max_diameter: usize },
    Custom { reason: String },
}
```

## Example usage

### Creating a Rule Engine

```rust
let mut engine = RuleEngine::new();

// Add rules
engine.add_behavior_rule(Box::new(MyBehaviorRule::new()));
engine.add_topology_rule(Box::new(MyTopologyRule::new()));
engine.add_plasticity_rule(Box::new(MyPlasticityRule::new()));
engine.add_consistency_check(Box::new(ConnectedTopologyCheck::new()));
```

### Running the complete cycle

```rust
let metrics = SystemMetrics::from_topology(&topology);

let mut topology_actions = engine.run_topology_rules(&topology_mutex, &metrics).await?;
let plasticity_actions = engine
    .run_plasticity_rules(&topology_mutex, &metrics, &plasticity_ctx)
    .await?;

topology_actions.extend(plasticity_actions);

let effects = engine
    .apply_topology_actions(&topology_mutex, topology_actions)
    .await?;

engine
    .run_consistency_checks(&topology_mutex, &agent_ids)
    .await?;
```

## Full cycle via `run_full_cycle`

The `RuleEngine::run_full_cycle` method performs:

1. `run_behavior_rules` for all agents
2. `run_topology_rules`
3. `run_plasticity_rules`
4. `apply_topology_actions`
5. `run_consistency_checks`

It returns a map of behavior actions and the number of topology edge operations applied.

## Implementation Reality

- `RuleEngine` stores separate vectors for behavior, topology, plasticity, and consistency rule collections.
- Consistency checks run after topology changes and before the scheduler continues.
- `NodeCountConsistencyCheck` compares `topology.graph.node_count()` with the number of agent IDs.
- `MinConnectivityCheck` computes average degree as `(2 * edge_count) / node_count`.
- `MaximumDegreeCheck` is currently absent; the available check is `MinimumDegreeCheck`.

## Important details

- `RuleEngine::run_topology_rules` clones the topology before evaluation.
- `RuleEngine::run_plasticity_rules` may skip growth-oriented plasticity when density exceeds `0.2`.
- `RuleEngine::apply_topology_actions` can silently drop edges rejected by topology constraints.
- `run_consistency_checks` reads the latest topology snapshot and checks with the current agent list.

