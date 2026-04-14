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

#[cfg(feature = "adaptiflux_optim")]
use adaptiflux_optim::{Adam, Optimizer};
#[cfg(feature = "adaptiflux_optim")]
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(feature = "adaptiflux_optim")]
fn bench_adam_10k(c: &mut Criterion) {
    let mut params = vec![0.5f32; 10_000];
    let grads = vec![0.1f32; 10_000];
    let mut optim = Adam::new(0.001);
    optim.init(&mut params);

    c.bench_function("adam_10k_params", |b| {
        b.iter(|| optim.step(&mut params, &grads))
    });
}

#[cfg(feature = "adaptiflux_optim")]
criterion_group!(benches, bench_adam_10k);
#[cfg(feature = "adaptiflux_optim")]
criterion_main!(benches);

#[cfg(not(feature = "adaptiflux_optim"))]
fn main() {
    println!("Benchmark requires 'adaptiflux_optim' feature");
}
