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

#[cfg(feature = "cuda")]
extern "C" {
    pub fn cuda_sgd_step_kernel_wrapper(
        params_ptr: *mut f32,
        grad_ptr: *const f32,
        lr: f32,
        n: u32,
    ) -> i32;

    pub fn cuda_compute_squared_norms_kernel_wrapper(
        params_ptr: *const f32,
        losses_ptr: *mut f32,
        num_params: u32,
        pop_size: u32,
    ) -> i32;
}
