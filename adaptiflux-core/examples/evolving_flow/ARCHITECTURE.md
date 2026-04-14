# Evolving Flow Scenario - Architecture & Implementation Guide

## Overview

The "Evolving Flow" scenario is a comprehensive test of the Adaptiflux architecture that demonstrates all key capabilities in a realistic, resource-constrained environment (4 cores, 8GB RAM). It simulates a network of 30-200 nodes that must adapt to failures and changing conditions.

## Scenario Architecture

### Network Topology
- **Nodes**: 30-200 intelligent agents organized as a dynamic network
- **Connections**: Variable, can be updated by plasticity rules
- **Communication**: Message passing through LocalBus

### Zoooid Types per Network Node

Each network node contains multiple agents:

1. **SensorZoooid (LIF neuron)**
   - Monitors local load (active packets waiting)
   - Generates spikes when load exceeds threshold (2.0)
   - Feeds into PID controller

2. **PIDZoooid (PID Controller)**
   - Takes error signal from sensor
   - Outputs target throughput rate
   - Setpoint: 5.0 (target load)
   - Open-loop gain: 0.1

3. **SwarmZoooid (PFSM - Probabilistic Finite State Machine)**
   - Makes routing decisions
   - Takes PID output and neighbor status
   - Routes packets to next hop based on destination

4. **CognitiveZoooid (Izhikevich neuron)**
   - Only in adaptive mode (not in baseline)
   - Learns failure patterns
   - Predicts failures before they occur

5. **PhysarumZoooid (Physarum Unit)**
   - Only in adaptive mode
   - Models slime mold behavior
   - Triggers topology plasticity

### Packet Simulation

- **Packet Structure**: id, source, destination, hops, max_hops, data
- **Generation**: Every ~50ms to source nodes
- **Routing**: Adaptive (choosing best neighbor) vs Fixed (baseline)
- **Delivery**: Counted when reaching destination
- **Failure**: Lost if exceeds max_hops (15)

### Failure Scenarios

#### Event 1: Single Node Failure (T=30s)
- Node 42 is killed
- Neighbors detect failure
- Routing tables updated
- Network adapts to bypass failed node

#### Event 2: Repeat Failure (T=90s)
- Node 42 fails again
- Cognitive agents predict pattern
- Recovery time should be shorter than Event 1

#### Event 3: Mass Failure (T=150s)
- 30% of random nodes simultaneously fail
- Triggers network reorganization
- Tests resilience and scaling

## Components Used

### Core Systems
- **CoreScheduler**: Main execution engine
- **RuleEngine**: Manages plasticity and behavior rules
- **ZoooidTopology**: Tracks agent relationships

### Hooks (Adaptive Components)
- **AsyncOptimizationHook**: Parallel optimization (4 concurrent)
- **SparseExecutionHook**: Skip inactive agents
- **SleepScheduler**: Energy-aware scheduling
- **OnlineAdaptationHook**: Real-time learning
- **HierarchyHook**: Multi-level abstraction
- **MemoryAttentionHook**: Long-term memory integration

### Rules
- **LoadBalancingRule**: Distribute load across network
- **IsolationRecoveryRule**: Handle disconnected agents
- **ProximityConnectionRule**: Connect nearby agents
- **SynapticPruningRule**: Remove weak connections
- **ActivityDependentSynaptogenesisRule**: Create new connections
- **ClusterGroupingPlasticityRule**: Organize into clusters

## Test Scenarios

### Scenario Modes

| Mode | Nodes | Duration | Failures | UI | Performance |
|------|-------|----------|----------|----|----|
| Functional | 30 | 60s | No | No | Baseline |
| Performance | 100 | 120s | No | No | Measured |
| Resilience | 50 | 180s | Yes | No | Compare |
| UI Test | 30 | 30s | No | Yes | Visual |
| Baseline | 50 | 120s | No | No | Fixed logic |
| Comprehensive | 200 | 300s | Yes | Optional | Maximum load |

## Success Criteria

### Functional
- ✅ No panics or crashes
- ✅ Logs show regular activity
- ✅ Packets delivered successfully

### Performance
- ✅ Scheduler iteration time < 100ms
- ✅ RAM usage < 4GB
- ✅ CPU usage < 90%
- ✅ Delivery rate > 50%

### Resilience
- ✅ Network recovers after single node failure
- ✅ Recovery time decreases after repeated failures
- ✅ Network reorganizes after mass failure

### Comparison
- ✅ Adaptive: Better delivery rate than baseline
- ✅ Adaptive: Faster recovery time than baseline
- ✅ Baseline: Fails gracefully (degrades predictably)

## Running Tests

### Quick Test (for development)
```bash
./run_tests.sh
```
Runs with: 10-20 nodes, 10-30 second durations

### Production Tests
```bash
./run_production_tests.sh
```
Runs full suite per specification

### Custom Configuration
```bash
cargo run --example evolving_flow -- \
  --nodes 50 \
  --duration 120 \
  --failures \
  --ui
```

### Baseline Mode
```bash
cargo run --example evolving_flow -- \
  --nodes 50 \
  --duration 120 \
  --baseline
```

## Metrics and Observability

### Logged Metrics
- Scheduler iteration time (ms)
- Active agents count
- Packets generated/delivered
- Memory usage (MB)
- CPU usage (%)
- Network events (failures, recoveries)

### Log Output
- File: `logs/evolving_flow/evolving_flow-adptive.log`
- Baseline: `logs/evolving_flow/baseline.log`
- Per-test logs: `logs/evolving_flow/`

### Key Measurements

| Metric | Adaptive | Baseline | Target |
|--------|----------|----------|--------|
| Iter Time (ms) | <50 | 30-40 | <100 |
| Recovery Time | Quick | N/A | Decreases with experience |
| Delivery Rate | >80% | 40-50% | >70% |
| Memory (MB) | 600-1200 | 600-1000 | <2000 |

## Architecture Decisions

### Why Zoooids?
- Specialized agents for specific functions
- Composable behavior
- Clear separation of concerns

### Why Multiple Plasticity Rules?
- Pruning: Clean up unused connections
- Synaptogenesis: Create new paths
- Clustering: Organize into efficient groups

### Why Cognitive Layer?
- Predicts failures before they happen
- Learns from experience
- Enables proactive adaptation

### Why Baselines?
- Shows static system weakness
- Validates benefit of adaptation
- Tests basic functionality

## Advanced Topics

### Online Learning
The OnlineAdaptationHook integrates with the MemoryAttentionHook to:
1. Store past failure patterns
2. Build predictive model
3. Inject memory into feedback signals
4. Enable anticipatory adaptation

### Multi-level Hierarchy
The HierarchyHook creates abstraction layers:
1. Individual agents (ground level)
2. Local clusters (intermediate)
3. Global topology view (high level)

Enables: faster planning, better load distribution

### Energy Awareness
SleepScheduler reduces CPU usage by:
1. Identifying inactive agents
2. Skipping their update cycles
3. Wake them on message arrival

Result: Same functionality, 30-50% less power

## Troubleshooting

### High Memory Usage
- Reduce number of nodes (--nodes)
- Reduce duration (--duration)
- Lower memory: update OnlineAdaptationEngine capacity

### Slow Iteration Times
- Reduce network size
- Increase SparseExecutionHook threshold
- Check for message bottleneck in LocalBus

### Low Delivery Rate
- Increase max_hops
- Verify connectivity after failures
- Enable --ui to visualize routing

### Baseline Comparison Issues
- Ensure same network size
- Verify static routing table is functional
- Check that failures are properly injected

## Future Enhancements

1. **Distributed Implementation**: Run across multiple machines
2. **GPU Acceleration**: Use custom_optim for heavy computation
3. **Real-time Visualization**: Live network topology display
4. **Fault Injection**: More sophisticated failure scenarios
5. **Machine Learning**: Neural routing instead of PFSM
6. **Persistence**: Save/restore network state

## References

- Core concepts: `/docs/architecture.md`
- Plasticity rules: `/src/rules/`
- Example code: `/examples/evolving_flow/main.rs`
- Test runner: `/examples/evolving_flow/run_production_tests.sh`
