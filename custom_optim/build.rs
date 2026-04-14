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

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:rerun-if-changed=src/backend/cuda/kernels.cu");
        cc::Build::new()
            .cuda(true)
            .file("src/backend/cuda/kernels.cu")
            .compile("cuda_kernels");

        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
    }
}
