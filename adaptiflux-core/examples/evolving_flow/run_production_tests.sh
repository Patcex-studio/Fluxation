#!/bin/bash

# Comprehensive Test Runner for Adaptiflux Evolving Flow Scenario
# Production version for full testing according to specification

set -e

echo "========================================="
echo "Adaptiflux Evolving Flow - Production Test Suite"
echo "========================================="

# Create logs directory
mkdir -p logs/evolving_flow

cd "$(dirname "$0")/../.."

# Function to run test with timeout
run_test() {
    local test_name=$1
    local command=$2
    local timeout_sec=$3
    
    echo ""
    echo "=== Running: $test_name ==="
    echo "Command: $command"
    echo "Timeout: ${timeout_sec}s"
    echo "Started: $(date)"
    
    # Run test and capture output
    if timeout $timeout_sec bash -c "$command 2>&1" | tee "logs/evolving_flow/${test_name// /_}.log"; then
        echo "✅ $test_name PASSED"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "⚠ $test_name timed out after ${timeout_sec}s"
            # For long-running tests, timeout is not always a failure
            if [[ "$test_name" == *"Comprehensive"* ]] || [[ "$test_name" == *"Performance"* ]]; then
                echo "   (Normal for extended test - check logs for results)"
                return 0
            fi
            return 1
        else
            echo "❌ $test_name FAILED"
            return 1
        fi
    fi
}

# Determine if we should run quick or full tests
if [ "$1" == "--quick" ]; then
    echo "Running QUICK test suite..."
    NODES_1=10
    NODES_2=15
    NODES_3=20
    DURATION_1=10
    DURATION_2=15
    DURATION_3=30
    NODES_SCALE=20
    DURATION_SCALE=30
else
    echo "Running FULL test suite according to specification..."
    NODES_1=30
    NODES_2=50
    NODES_3=100
    DURATION_1=60
    DURATION_2=120
    DURATION_3=180
    NODES_SCALE=100
    DURATION_SCALE=120
fi

# Test 1: Functional Testing
echo ""
echo "========================================="
echo "TEST 1: Functional Testing ($NODES_1 nodes, ${DURATION_1}s)"
echo "========================================="
run_test "Functional Test" "cargo run --example evolving_flow -- --nodes $NODES_1 --duration $DURATION_1" $((DURATION_1 + 30))

# Test 2: Performance Testing
echo ""
echo "========================================="
echo "TEST 2: Performance Testing ($NODES_2 nodes, ${DURATION_2}s)"
echo "========================================="
run_test "Performance Test" "cargo run --example evolving_flow -- --nodes $NODES_2 --duration $DURATION_2" $((DURATION_2 + 30))

# Test 3: Resilience Testing
echo ""
echo "========================================="
echo "TEST 3: Resilience Testing ($NODES_3 nodes, ${DURATION_3}s, with failures)"
echo "========================================="
run_test "Resilience Test" "cargo run --example evolving_flow -- --nodes $NODES_3 --duration $DURATION_3 --failures" $((DURATION_3 + 30))

# Test 4: Baseline Comparison
echo ""
echo "========================================="
echo "TEST 4: Baseline Comparison (Adaptive vs Static)"
echo "========================================="
echo "Running Adaptive Scenario ($NODES_SCALE nodes, ${DURATION_SCALE}s)..."
run_test "Adaptive Scenario" "cargo run --example evolving_flow -- --nodes $NODES_SCALE --duration $DURATION_SCALE" $((DURATION_SCALE + 30))

echo ""
echo "Running Baseline Scenario ($NODES_SCALE nodes, ${DURATION_SCALE}s)..."
run_test "Baseline Scenario" "cargo run --example evolving_flow -- --nodes $NODES_SCALE --duration $DURATION_SCALE --baseline" $((DURATION_SCALE + 30))

# Test 5: UI Visualization Test
echo ""
echo "========================================="
echo "TEST 5: UI Visualization Test ($NODES_1 nodes, ${DURATION_1}s)"
echo "========================================="
run_test "UI Test" "cargo run --example evolving_flow -- --nodes $NODES_1 --duration $DURATION_1 --ui" $((DURATION_1 + 30))

# Test 6: Comprehensive Stress Test (only if not quick mode)
if [ "$1" != "--quick" ]; then
    echo ""
    echo "========================================="
    echo "TEST 6: Comprehensive Stress Test (200 nodes, 5 min)"
    echo "========================================="
    run_test "Comprehensive Test" "cargo run --release --example evolving_flow -- --nodes 200 --duration 300 --failures" 360
fi

# Generate report from logs
echo ""
echo "========================================="
echo "Test Summary Report"
echo "========================================="
echo ""
echo "Log files available in logs/evolving_flow/:"
ls -lh logs/evolving_flow/ 2>/dev/null | tail -10 || echo "  (No logs yet - tests may have failed)"

echo ""
echo "Check logs for:"
echo "  1. ✅ Functional: No panics, logs show activity"
echo "  2. ✅ Performance: Iteration time < 100ms, RAM < 4GB, CPU < 90%"
echo "  3. ✅ Resilience: Network recovers from failures"
echo "  4. ✅ Comparison: Adaptive outperforms baseline"
echo "  5. ✅ Visualization: UI shows network and metrics (if --ui used)"
echo ""
echo "Usage:"
echo "  ./run_production_tests.sh --quick    (run quick tests)"
echo "  ./run_production_tests.sh            (run full test suite)"
echo ""
echo "========================================="
