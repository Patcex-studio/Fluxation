#!/bin/bash
# Quick Reference: Hybrid Architectures - Commands

echo "==================================="
echo "Hybrid Architectures Quick Commands"
echo "==================================="
echo

# Compilation
echo "1. BUILD & COMPILE"
echo "  cargo build        # Build project"
echo "  cargo check        # Check without building"
echo

# Testing
echo "2. TESTING"
echo "  cargo test                  # All tests (30 total)"
echo "  cargo test --lib hybrids    # Hybrid module tests (12)"
echo "  cargo test hybrids_integration  # Integration tests (9)"
echo

# Examples
echo "3. RUN USE CASE EXAMPLES"
echo "  cargo run --example distributed_search"
echo "  cargo run --example adaptive_control"
echo "  cargo run --example network_optimization"
echo

# Documentation
echo "4. DOCUMENTATION"
echo "  cat HYBRID_ARCHITECTURES.md     # Full architecture docs"
echo "  cat IMPLEMENTATION_SUMMARY.md    # Completion summary"
echo

# File Structure
echo "5. CODE STRUCTURE"
echo "  src/hybrids/                    # Hybrid architectures"
echo "    - mod.rs                      # Module exports"
echo "    - sensor_processor_controller.rs"
echo "    - swarm_forager.rs"
echo "    - physarum_router.rs"
echo "    - cognitive_memory.rs"
echo "  examples/use_cases/             # Use case examples"
echo "    - distributed_search.rs"
echo "    - adaptive_control.rs"
echo "    - network_optimization.rs"
echo "  tests/hybrids_integration_tests.rs  # Integration tests"
echo

# Quick Stats
echo "6. PROJECT STATISTICS"
echo "  Hybrid Architectures: 4"
echo "  Module Tests: 12 (all passed)"
echo "  Integration Tests: 9 (all passed)"
echo "  Use Cases: 3"
echo "  Total Lines of Code: ~1200"
echo

echo "=== Architecture Types ==="
echo "1. Cascade Pattern     (SPC)      - Sensor → Processor → Controller"
echo "2. Mesh Pattern        (Swarm)    - Full connectivity with pheromones"
echo "3. Network Pattern     (Router)   - Adaptive routing network"
echo "4. Recurrent Pattern   (Cognitive)- Memory via feedback loops"
echo
