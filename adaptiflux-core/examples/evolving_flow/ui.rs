// Copyright (C) 2026 Jocer S. <patcex@proton.me>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: AGPL-3.0 OR Commercial

//! Simple UI for Evolving Flow scenario
//!
//! This module provides basic terminal output without external dependencies

use adaptiflux_core::CoreScheduler;
use std::time::Duration;

/// UI state for visualization
pub struct UIState {
    pub elapsed_time: Duration,
    pub active_nodes: usize,
    pub total_nodes: usize,
    pub packets_in_flight: usize,
    pub packets_delivered: usize,
    pub iteration_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub current_event: String,
}

impl Default for UIState {
    fn default() -> Self {
        Self {
            elapsed_time: Duration::default(),
            active_nodes: 0,
            total_nodes: 50,
            packets_in_flight: 0,
            packets_delivered: 0,
            iteration_time_ms: 0.0,
            memory_usage_mb: 0.0,
            cpu_usage_percent: 0.0,
            current_event: "Initializing...".to_string(),
        }
    }
}

/// Print UI to terminal
pub fn print_ui(state: &UIState) {
    println!("\n{}", "=".repeat(60));
    println!("Adaptiflux - Evolving Flow Scenario (T={:.1}s)", state.elapsed_time.as_secs_f32());
    println!("{}", "=".repeat(60));
    
    println!("\n📊 NETWORK STATUS:");
    println!("  Active Nodes: {}/{}", state.active_nodes, state.total_nodes);
    println!("  Packets: {} in flight, {} delivered", state.packets_in_flight, state.packets_delivered);
    
    println!("\n⚡ PERFORMANCE:");
    println!("  Iteration Time: {:.2} ms", state.iteration_time_ms);
    println!("  Memory Usage: {:.1} MB", state.memory_usage_mb);
    println!("  CPU Usage: {:.1}%", state.cpu_usage_percent);
    
    println!("\n📈 LOAD INDICATORS:");
    let load_bar = "█".repeat((state.packets_in_flight as f32 / 10.0).min(20.0) as usize);
    println!("  [{}]", load_bar);
    
    println!("\n🔔 CURRENT EVENT:");
    println!("  {}", state.current_event);
    println!("{}", "=".repeat(60));
}

/// Update UI state from scheduler
pub async fn update_ui_state(
    state: &mut UIState,
    scheduler: &CoreScheduler,
    elapsed: Duration,
    iteration_time: Duration,
) {
    // Update elapsed time
    state.elapsed_time = elapsed;
    
    // Update stats
    state.active_nodes = scheduler.agents.len();
    state.iteration_time_ms = iteration_time.as_secs_f64() * 1000.0;
    
    // Update current event based on time
    if elapsed < Duration::from_secs(30) {
        state.current_event = "Phase 1: Stable operation, load balanced".to_string();
    } else if elapsed < Duration::from_secs(90) {
        state.current_event = "Phase 2: Single node failure, adaptation in progress".to_string();
    } else if elapsed < Duration::from_secs(150) {
        state.current_event = "Phase 3: Repeated failure, cognitive prediction active".to_string();
    } else {
        state.current_event = "Phase 4: Mass failure, network reorganization".to_string();
    }
}

