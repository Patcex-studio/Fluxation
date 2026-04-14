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

#[derive(Debug, Clone)]
pub struct GradientAccumulator {
    buffer: Vec<f32>,
    count: usize,
    threshold: usize,
}

impl GradientAccumulator {
    pub fn new(length: usize, threshold: usize) -> Self {
        Self {
            buffer: vec![0.0; length],
            count: 0,
            threshold: threshold.max(1),
        }
    }

    pub fn accumulate_batch(&mut self, grad: &[f32]) {
        assert_eq!(grad.len(), self.buffer.len(), "gradient length mismatch");
        self.count += 1;
        for (dst, src) in self.buffer.iter_mut().zip(grad.iter()) {
            *dst += *src;
        }
    }

    pub fn flush(&mut self) -> Option<&[f32]> {
        if self.count >= self.threshold {
            let divisor = self.count as f32;
            for value in &mut self.buffer {
                *value /= divisor;
            }
            self.count = 0;
            Some(&self.buffer)
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.count = 0;
    }
}
