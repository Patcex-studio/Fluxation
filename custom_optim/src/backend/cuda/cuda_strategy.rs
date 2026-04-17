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

use crate::backend::cuda::kernels;
use crate::{
    optimizer::{OptimizerConfig, OptimizerError},
    strategy::OptimizerStrategy,
    utils::finite_difference_gradient,
};
use libc::c_char;
use rand::distr::{Distribution, Uniform};
use std::{ffi::CStr, ptr::NonNull};

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaOptimizerState {
    raw: NonNull<libc::c_void>,
    len: usize,
    block_size: u32,
}

#[cfg(feature = "cuda")]
impl CudaOptimizerState {
    pub fn new(params: &mut [f32], block_size: u32) -> Result<Self, String> {
        let mut raw_state: *mut libc::c_void = std::ptr::null_mut();
        let status = unsafe {
            kernels::cuda_init_cuda_optimizer_state(
                params.as_ptr(),
                params.len() as u32,
                block_size,
                &mut raw_state,
            )
        };
        if status != 0 {
            return Err(Self::format_cuda_error(status));
        }

        let raw = NonNull::new(raw_state)
            .ok_or_else(|| "Failed to initialize CUDA optimizer state".to_string())?;

        Ok(Self {
            raw,
            len: params.len(),
            block_size,
        })
    }

    pub fn step(&mut self, params: &mut [f32], gradient: &[f32], lr: f32) -> Result<(), String> {
        if params.len() != self.len || gradient.len() != self.len {
            return Err("CUDA optimizer state length mismatch".into());
        }

        let status = unsafe {
            kernels::cuda_sgd_step_with_state_wrapper(
                self.raw.as_ptr(),
                params.as_mut_ptr(),
                gradient.as_ptr(),
                lr,
                1,
            )
        };
        if status != 0 {
            return Err(Self::format_cuda_error(status));
        }
        Ok(())
    }

    fn format_cuda_error(code: i32) -> String {
        unsafe {
            let message_ptr = kernels::cuda_error_message_wrapper(code);
            if message_ptr.is_null() {
                return format!("CUDA error code {}", code);
            }
            CStr::from_ptr(message_ptr).to_string_lossy().into_owned()
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaOptimizerState {
    fn drop(&mut self) {
        unsafe {
            kernels::cuda_free_optimizer_state(self.raw.as_ptr());
        }
    }
}

pub struct CudaSGD {
    pub learning_rate: f32,
    pub iterations: usize,
    pub block_size: u32,
}

impl CudaSGD {
    pub fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        let learning_rate = config.learning_rate.unwrap_or(0.01);
        let iterations = config.iterations.unwrap_or(50);
        let block_size = config.block_size.unwrap_or(0);
        Ok(Self {
            learning_rate,
            iterations,
            block_size,
        })
    }

    fn initialize(&self) -> Result<(), OptimizerError> {
        Ok(())
    }
}

impl OptimizerStrategy for CudaSGD {
    fn optimize(
        &mut self,
        params: &mut [f32],
        loss_fn: &dyn Fn(&[f32]) -> f32,
    ) -> Result<(), OptimizerError> {
        self.initialize()?;
        let mut state =
            CudaOptimizerState::new(params, self.block_size).map_err(OptimizerError::CudaError)?;

        for _ in 0..self.iterations {
            let gradient = finite_difference_gradient(params, loss_fn, 1e-4);
            let result = state.step(params, &gradient, self.learning_rate);
            if let Err(error_message) = result {
                return Err(OptimizerError::CudaError(error_message));
            }
        }
        Ok(())
    }
}

pub struct CudaGeneticAlgorithm {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
}

impl CudaGeneticAlgorithm {
    pub fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        let population_size = config
            .population_size
            .ok_or(OptimizerError::InvalidConfig)?;
        let generations = config.generations.ok_or(OptimizerError::InvalidConfig)?;
        Ok(Self {
            population_size,
            generations,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
        })
    }

    fn mutate(&self, individual: &mut [f32]) {
        let mut rng = rand::rng();
        let uniform = Uniform::new(0.0_f32, 1.0_f32).unwrap();
        for x in individual.iter_mut() {
            if uniform.sample(&mut rng) < self.mutation_rate {
                *x += (uniform.sample(&mut rng) - 0.5) * 0.1;
            }
        }
    }

    fn crossover(&self, parent_a: &[f32], parent_b: &[f32], child: &mut [f32]) {
        let split = parent_a.len() / 2;
        for i in 0..parent_a.len() {
            child[i] = if i < split { parent_a[i] } else { parent_b[i] };
        }
    }
}

impl OptimizerStrategy for CudaGeneticAlgorithm {
    fn optimize(
        &mut self,
        params: &mut [f32],
        loss_fn: &dyn Fn(&[f32]) -> f32,
    ) -> Result<(), OptimizerError> {
        let dim = params.len();
        if dim == 0 {
            return Ok(());
        }

        let mut rng = rand::rng();
        let uniform = Uniform::new(0.0_f32, 1.0_f32).unwrap();
        let mut population: Vec<Vec<f32>> = (0..self.population_size)
            .map(|_| {
                params
                    .iter()
                    .map(|v| *v + (uniform.sample(&mut rng) - 0.5) * 0.1)
                    .collect()
            })
            .collect();

        let mut losses = vec![0.0_f32; self.population_size];
        let mut flat = vec![0.0_f32; self.population_size * dim];
        for _ in 0..self.generations {
            for (i, individual) in population.iter().enumerate() {
                let start = i * dim;
                flat[start..start + dim].copy_from_slice(individual);
            }

            let result = unsafe {
                kernels::cuda_compute_squared_norms_kernel_wrapper(
                    flat.as_ptr(),
                    losses.as_mut_ptr(),
                    dim as u32,
                    self.population_size as u32,
                    0,
                )
            };
            if result != 0 {
                return Err(OptimizerError::CudaError(
                    CudaOptimizerState::format_cuda_error(result),
                ));
            }

            let mut scored: Vec<(f32, Vec<f32>)> = population
                .iter()
                .map(|individual| (loss_fn(individual), individual.clone()))
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            population.clear();
            population.extend(scored.iter().take(2).map(|(_, ind)| ind.clone()));

            let mut rng = rand::rng();
            let between = Uniform::new(0, scored.len()).unwrap();
            while population.len() < self.population_size {
                let parent_a = &scored[between.sample(&mut rng)].1;
                let parent_b = &scored[between.sample(&mut rng)].1;
                let mut child = vec![0.0_f32; dim];
                self.crossover(parent_a, parent_b, &mut child);
                self.mutate(&mut child);
                population.push(child);
            }
        }

        if let Some(best) = population.iter().min_by(|a, b| {
            loss_fn(a)
                .partial_cmp(&loss_fn(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            params.copy_from_slice(best);
        }
        Ok(())
    }
}
