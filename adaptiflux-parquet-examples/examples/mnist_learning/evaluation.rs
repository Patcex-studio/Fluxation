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

use std::error::Error;

use adaptiflux_core::core::scheduler::CoreScheduler;

use crate::data_loader::MnistDataset;
use crate::mnist_spiking_classifier::MnistSpikingClassifierArchitecture;

pub async fn evaluate_mnist(
    scheduler: &mut CoreScheduler,
    architecture: &MnistSpikingClassifierArchitecture,
    dataset: &MnistDataset,
) -> Result<f64, Box<dyn Error + Send + Sync>> {
    let mut correct = 0;

    for (image, label) in dataset.test_images.iter().zip(dataset.test_labels.iter()) {
        architecture.reset_output_counts(scheduler);
        crate::data_loader::encode_image_to_sensors(
            &scheduler.message_bus,
            &architecture.sensor_ids,
            image,
        )
        .await?;
        scheduler.run_for_iterations(3).await?;
        let prediction = architecture.decode_output(scheduler);
        if prediction == *label {
            correct += 1;
        }
    }

    Ok(correct as f64 / dataset.test_images.len() as f64)
}
