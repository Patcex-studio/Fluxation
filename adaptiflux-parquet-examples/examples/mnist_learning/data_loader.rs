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

use adaptiflux_core::core::message_bus::{Message, MessageBus};
use adaptiflux_core::utils::types::ZoooidId;
use image::DynamicImage;
use image::ImageError;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub struct MnistDataset {
    pub train_images: Vec<Vec<f32>>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<Vec<f32>>,
    pub test_labels: Vec<u8>,
}

pub async fn load_mnist(
    base_path: &str,
    max_train: usize,
    max_test: usize,
) -> Result<MnistDataset, Box<dyn Error + Send + Sync>> {
    let train_path = Path::new(base_path).join("train-00000-of-00001(1).parquet");
    let test_path = Path::new(base_path).join("test-00000-of-00001(1).parquet");

    let train_images = load_images_from_parquet(&train_path, max_train)?;
    let train_labels = load_labels_from_parquet(&train_path, max_train)?;
    let test_images = load_images_from_parquet(&test_path, max_test)?;
    let test_labels = load_labels_from_parquet(&test_path, max_test)?;

    Ok(MnistDataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
    })
}

fn load_images_from_parquet(
    path: &Path,
    max_count: usize,
) -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let iter = reader.get_row_iter(None)?;

    let rows: Vec<_> = iter.take(max_count).collect();

    let images: Vec<Vec<f32>> = rows
        .into_par_iter()
        .map(|row| -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
            let image_group = row.get_group(0)?;
            let bytes = image_group.get_bytes(0)?;
            let img = image::load_from_memory(bytes.data())?;
            let pixels = extract_pixels(&img)?;
            Ok(normalize_and_downsample(&pixels))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(images)
}

fn load_labels_from_parquet(
    path: &Path,
    max_count: usize,
) -> Result<Vec<u8>, Box<dyn Error + Send + Sync>> {
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let iter = reader.get_row_iter(None)?;

    let rows: Vec<_> = iter.take(max_count).collect();

    let labels: Vec<u8> = rows
        .into_par_iter()
        .map(|row| -> Result<u8, Box<dyn Error + Send + Sync>> {
            let label: i64 = row.get_long(1)?;
            Ok(label as u8)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(labels)
}

fn extract_pixels(img: &DynamicImage) -> Result<Vec<u8>, ImageError> {
    let gray_img = img.to_luma8();
    Ok(gray_img.into_raw())
}

fn normalize_and_downsample(image: &[u8]) -> Vec<f32> {
    const OUT_SIZE: usize = 14;
    const IN_SIZE: usize = 28;
    let mut result = Vec::with_capacity(OUT_SIZE * OUT_SIZE);

    for block_row in 0..OUT_SIZE {
        for block_col in 0..OUT_SIZE {
            let mut sum = 0_u32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let row = block_row * 2 + dy;
                    let col = block_col * 2 + dx;
                    let idx = row * IN_SIZE + col;
                    sum += image[idx] as u32;
                }
            }
            let avg = sum as f32 / 4.0 / 255.0;
            result.push(avg.clamp(0.0, 1.0));
        }
    }

    result
}

pub async fn encode_image_to_sensors(
    bus: &Arc<dyn MessageBus + Send + Sync>,
    sensor_ids: &[ZoooidId],
    image: &[f32],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    const MAX_CURRENT: f32 = 32.0;
    let sender = ZoooidId::new_v4();

    for (&agent_id, &pixel) in sensor_ids.iter().zip(image.iter()) {
        let current = pixel * MAX_CURRENT;
        bus.send(sender, agent_id, Message::AnalogInput(current))
            .await
            .map_err(|e| format!("Failed to push sensor current: {:?}", e))?;
    }

    Ok(())
}
