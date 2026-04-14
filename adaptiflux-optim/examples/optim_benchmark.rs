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

use adaptiflux_optim::{Adam, Optimizer};

fn main() {
    let mut params = vec![1.0f32; 10_000];
    let grads = vec![0.1f32; 10_000];
    let mut optimizer = Adam::new(0.001);

    optimizer.init(&mut params);

    let start = std::time::Instant::now();
    for _ in 0..1_000 {
        optimizer.step(&mut params, &grads);
    }
    let duration = start.elapsed();

    println!("1000 steps for 10k params: {:?}", duration);
    println!("first 4 params: {:?}", &params[..4]);
}
