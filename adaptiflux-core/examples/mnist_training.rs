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

use adaptiflux_core::*;
use std::error::Error;
use std::sync::Arc;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let message_bus = Arc::new(LocalBus::new());
    let topology = Arc::new(tokio::sync::RwLock::new(ZoooidTopology::new()));
    let rule_engine = RuleEngine::new();
    let resource_manager = ResourceManager::new();

    let _scheduler = CoreScheduler::new(
        topology.clone(),
        rule_engine,
        resource_manager,
        message_bus.clone(),
    );

    info!("🚀 MNIST training example stub initialized.");
    Ok(())
}

// TODO: Implement these functions
#[allow(dead_code)]
fn hybrid_mnist_blueprint(
    _input_size: usize,
    _hidden_size: usize,
    _output_size: usize,
) -> Box<dyn AgentBlueprint> {
    // Placeholder implementation for example - not fully implemented
    panic!("MNIST training example is a placeholder and not fully implemented")
}

#[allow(dead_code)]
fn load_mnist_data() -> Result<MnistDataset, Box<dyn Error>> {
    // Placeholder implementation for example - not fully implemented
    panic!("MNIST data loading is a placeholder and not fully implemented")
}

#[allow(dead_code)]
fn encode_to_currents(_image: &[u8]) -> Vec<f32> {
    // Placeholder implementation for example - not fully implemented
    panic!("Image encoding is a placeholder and not fully implemented")
}

#[allow(dead_code)]
fn compute_error(_pred: Vec<f32>, _label: u8) -> f32 {
    // Placeholder implementation for example - not fully implemented
    panic!("Error computation is a placeholder and not fully implemented")
}

#[allow(dead_code)]
struct MnistDataset {
    train_images: Vec<Vec<u8>>,
    train_labels: Vec<u8>,
}
