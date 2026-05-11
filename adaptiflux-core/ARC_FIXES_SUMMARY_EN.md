# Adaptiflux-Core Architecture Fix Summary

This document summarizes the recent architecture changes implemented in `adaptiflux-core`.

## Fixes Implemented

### 1. Edge API consistency
- Added `ZoooidTopology::try_add_edge()` as the public safe API.
- Deprecated `ZoooidTopology::add_edge()` and changed it to delegate to `try_add_edge()`.
- Introduced `add_edge_unchecked()` for internal topology operations.
- Updated all internal `add_edge()` call sites to `try_add_edge()` where applicable.
- Added regression tests to ensure the degree limit is enforced.

### 2. Agent update thread safety
- Documented `AgentBlueprint::update()` concurrency guarantees.
- Updated `CoreScheduler` to support sequential execution when `max_concurrent_updates == 1`.
- Added a regression test proving sequential update execution.

### 3. Centralized runtime configuration
- Added `SystemConfig` for runtime configuration of:
  - `max_degree_per_agent`
  - `scheduler_cycle_ms`
  - `max_concurrent_agent_updates`
- Added environment variables:
  - `ADAPTIFLUX_MAX_DEGREE_PER_AGENT`
  - `ADAPTIFLUX_SCHEDULER_CYCLE_MS`
  - `ADAPTIFLUX_MAX_CONCURRENT_AGENT_UPDATES`
- Replaced hardcoded constants with config values in scheduler and topology.

## Notes
- All library tests pass successfully after the changes.
- This summary is intended for developers and reviewers of the current fix.
