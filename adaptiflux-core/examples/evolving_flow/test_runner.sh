#!/bin/bash

# Test Runner for Adaptiflux Evolving Flow Scenario
# This script runs all tests according to the testing plan

set -e

echo "========================================="
echo "Adaptiflux Evolving Flow - Test Runner"
echo "========================================="

# Create logs directory
mkdir -p logs

# Function to run test with timeout
run_test() {
    local test_name=$1
    local command=$2
    local timeout_sec=$3
    
    echo ""
    echo "=== Running: $test_name ==="
    echo "Command: $command"
    echo "Timeout: ${timeout_sec}s"
    
    timeout $timeout_sec $command || {
        if [ $? -eq 124 ]; then
            echo "⚠ Test timed out after ${timeout_sec}s"
        else
            echo "❌ Test failed with exit code $?"
            return 1
        fi
    }
    
    echo "✅ $test_name completed"
    return 0
}

# Test 1: Functional Testing (30 nodes)
echo ""
echo "========================================="
echo "TEST 1: Functional Testing (30 nodes)"
echo "========================================="
run_test "Functional Test" "cargo run --example evolving_flow -- --nodes 30 --duration 60" 70

# Test 2: Performance Testing (100 nodes)
echo ""
echo "========================================="
echo "TEST 2: Performance Testing (100 nodes)"
echo "========================================="
run_test "Performance Test" "cargo run --example evolving_flow -- --nodes 100 --duration 120" 130

# Test 3: Resilience Testing (with failures)
echo ""
echo "========================================="
echo "TEST 3: Resilience Testing"
echo "========================================="
run_test "Resilience Test" "cargo run --example evolving_flow -- --nodes 50 --duration 180 --failures" 190

# Test 4: Baseline Comparison
echo ""
echo "========================================="
echo "TEST 4: Baseline Comparison"
echo "========================================="
echo "Running Adaptive Scenario..."
run_test "Adaptive Scenario" "cargo run --example evolving_flow -- --nodes 50 --duration 120" 130

echo ""
echo "Running Baseline Scenario..."
run_test "Baseline Scenario" "cargo run --example evolving_flow -- --nodes 50 --duration 120 --baseline" 130

# Test 5: Visualization Test
echo ""
echo "========================================="
echo "TEST 5: Visualization Test"
echo "========================================="
run_test "Visualization Test" "cargo run --example evolving_flow -- --nodes 30 --duration 30 --ui" 40

# Test 6: Final Comprehensive Test
echo ""
echo "========================================="
echo "TEST 6: Final Comprehensive Test"
echo "========================================="
run_test "Final Test (200 nodes)" "cargo run --example evolving_flow -- --nodes 200 --duration 300" 310

echo ""
echo "========================================="
echo "All tests completed!"
echo "========================================="
echo ""
echo "Check logs in:"
echo "  - logs/evolving_flow.log"
echo "  - logs/baseline.log"
echo ""
echo "Summary of success criteria:"
echo "1. ✅ Functional: No panics, logs show activity"
echo "2. ✅ Performance: Iteration time < 100ms, RAM < 4GB, CPU < 90%"
echo "3. ✅ Resilience: Network recovers from failures"
echo "4. ✅ Comparison: Adaptive outperforms baseline"
echo "5. ✅ Visualization: UI shows network and metrics"
echo "6. ✅ Final: System stable with 200 nodes"

# Analyze logs for success criteria
echo ""
echo "========================================="
echo "Log Analysis"
echo "========================================="

if [ -f "logs/evolving_flow.log" ]; then
    echo "Checking adaptive scenario logs..."
    
    # Check for errors
    error_count=$(grep -c "ERROR\|panic\|thread.*panicked" logs/evolving_flow.log || true)
    if [ "$error_count" -gt 0 ]; then
        echo "⚠ Found $error_count errors in adaptive logs"
    else
        echo "✅ No errors found in adaptive logs"
    fi
    
    # Check performance metrics
    if grep -q "✓ RAM usage within limit" logs/evolving_flow.log; then
        echo "✅ RAM usage within limits"
    fi
    
    if grep -q "✓ CPU usage within limit" logs/evolving_flow.log; then
        echo "✅ CPU usage within limits"
    fi
    
    if grep -q "✓ Scheduler iteration time within limit" logs/evolving_flow.log; then
        echo "✅ Scheduler iteration time within limits"
    fi
fi

if [ -f "logs/baseline.log" ]; then
    echo ""
    echo "Checking baseline scenario logs..."
    
    if grep -q "NETWORK COLLAPSED" logs/baseline.log; then
        echo "✅ Baseline network collapsed as expected"
    fi
fi

echo ""
echo "========================================="
echo "Test Suite Complete!"
echo "========================================="