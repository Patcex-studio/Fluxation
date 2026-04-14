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

use crate::{optimizer::OptimizerConfig, strategy::OptimizerStrategy, OptimizerError};
use crate::utils::finite_difference_gradient;

pub struct SGD {
    pub learning_rate: f32,
    pub iterations: usize,
}

impl SGD {
    pub fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        let learning_rate = config.learning_rate.unwrap_or(0.01);
        let iterations = config.iterations.unwrap_or(50);
        Ok(Self {
            learning_rate,
            iterations,
        })
    }
}

impl OptimizerStrategy for SGD {
    fn optimize(&mut self, params: &mut [f32], loss_fn: &dyn Fn(&[f32]) -> f32) -> Result<(), OptimizerError> {
        for _ in 0..self.iterations {
            let gradient = finite_difference_gradient(params, loss_fn, 1e-3);
            for (p, g) in params.iter_mut().zip(gradient.iter()) {
                *p -= self.learning_rate * *g;
            }
        }
        Ok(())
    }
}

pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
}

impl GeneticAlgorithm {
    pub fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        let population_size = config.population_size.ok_or(OptimizerError::InvalidConfig)?;
        let generations = config.generations.ok_or(OptimizerError::InvalidConfig)?;
        Ok(Self {
            population_size,
            generations,
            mutation_rate: 0.15,
            crossover_rate: 0.7,
        })
    }

    fn mutate(&self, individual: &mut [f32]) {
        for x in individual.iter_mut() {
            if rand::random::<f32>() < self.mutation_rate {
                *x += (rand::random::<f32>() - 0.5) * 0.1;
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

impl OptimizerStrategy for GeneticAlgorithm {
    fn optimize(&mut self, params: &mut [f32], loss_fn: &dyn Fn(&[f32]) -> f32) -> Result<(), OptimizerError> {
        let dim = params.len();
        let mut population: Vec<Vec<f32>> = (0..self.population_size)
            .map(|_| params.iter().map(|v| *v + (rand::random::<f32>() - 0.5) * 0.1).collect())
            .collect();

        for _ in 0..self.generations {
            let mut scored: Vec<(f32, Vec<f32>)> = population
                .iter()
                .map(|individual| (loss_fn(individual), individual.clone()))
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            population.clear();
            let elites = scored.iter().take(2).map(|(_, ind)| ind.clone()).collect::<Vec<_>>();
            population.extend(elites.clone());

            while population.len() < self.population_size {
                let parent_a = &scored[rand::random::<usize>() % scored.len()].1;
                let parent_b = &scored[rand::random::<usize>() % scored.len()].1;
                let mut child = vec![0.0; dim];
                self.crossover(parent_a, parent_b, &mut child);
                self.mutate(&mut child);
                population.push(child);
            }
        }

        if let Some(best) = population
            .iter()
            .min_by(|a, b| loss_fn(a).partial_cmp(&loss_fn(b)).unwrap_or(std::cmp::Ordering::Equal))
        {
            params.copy_from_slice(best);
        }
        Ok(())
    }
}
