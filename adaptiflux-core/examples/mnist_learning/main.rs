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

mod data_loader;
mod evaluation;
mod mnist_spiking_classifier;
mod training_loop;

use std::error::Error;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    tracing::info!("starting MNIST learning run");

    let dataset = data_loader::load_mnist("../mnist-data", 60000, 10000).await?;
    let (mut scheduler, architecture) =
        training_loop::run_training(&dataset, Duration::from_secs(3600)).await?;

    tracing::info!("starting evaluation after training");
    let accuracy = evaluation::evaluate_mnist(&mut scheduler, &architecture, &dataset).await?;
    tracing::info!(
        "final evaluation complete, accuracy = {:.2}%",
        accuracy * 100.0
    );
    println!("Final evaluation accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}
