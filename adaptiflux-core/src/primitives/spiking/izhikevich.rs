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

use crate::primitives::base::{Primitive, PrimitiveMessage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IzhikevichParams {
    pub a: crate::utils::types::Param,
    pub b: crate::utils::types::Param,
    pub c: crate::utils::types::Param,
    pub d: crate::utils::types::Param,
    pub dt: crate::utils::types::Param,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IzhikevichState {
    pub v: crate::utils::types::StateValue,
    pub u: crate::utils::types::StateValue,
}

#[derive(Debug, Clone)]
pub struct IzhikevichNeuron;

impl IzhikevichNeuron {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for IzhikevichNeuron {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for IzhikevichParams {
    fn default() -> Self {
        Self {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            dt: 0.1,
        }
    }
}

impl Primitive for IzhikevichNeuron {
    type State = IzhikevichState;
    type Params = IzhikevichParams;

    fn initialize(_params: Self::Params) -> Self::State {
        IzhikevichState { v: -65.0, u: -13.0 }
    }

    fn update(
        mut state: Self::State,
        params: &Self::Params,
        input: &[PrimitiveMessage],
    ) -> (Self::State, Vec<PrimitiveMessage>) {
        let mut outputs = Vec::new();

        let total_input_current: crate::utils::types::StateValue = input
            .iter()
            .filter_map(|msg| {
                if let PrimitiveMessage::InputCurrent(value) = msg {
                    Some(*value)
                } else {
                    None
                }
            })
            .sum();

        if state.v >= 30.0 {
            outputs.push(PrimitiveMessage::Spike {
                timestamp: 0,
                amplitude: 1.0,
            });
            state.v = params.c;
            state.u += params.d;
            return (state, outputs);
        }

        state.v += 0.04 * state.v * state.v + 5.0 * state.v + 140.0 - state.u + total_input_current;
        state.u += params.a * (params.b * state.v - state.u) * params.dt;

        if state.v >= 30.0 {
            outputs.push(PrimitiveMessage::Spike {
                timestamp: 0,
                amplitude: 1.0,
            });
            state.v = params.c;
            state.u += params.d;
        }

        (state, outputs)
    }
}

/// ⚡ SIMD Batch processing module (PERF-003: 3-4x speedup for N>4 neurons)
pub mod simd {
    use wide::f32x4;
    use wide::CmpGt;
    use crate::utils::types::StateValue;

    /// SIMD batch parameters (packed for efficient SIMD operations)
    #[derive(Debug, Clone)]
    pub struct IzhikevichBatchParams {
        pub a: StateValue,
        pub b: StateValue,
        pub c: StateValue,
        pub d: StateValue,
        pub dt: StateValue,
    }

    impl IzhikevichBatchParams {
        /// Create from scalar parameters
        pub fn from_scalar(a: StateValue, b: StateValue, c: StateValue, d: StateValue, dt: StateValue) -> Self {
            Self { a, b, c, d, dt }
        }
    }

    impl Default for IzhikevichBatchParams {
        fn default() -> Self {
            Self::from_scalar(0.02, 0.2, -65.0, 8.0, 0.1)
        }
    }

    /// SIMD batch processing for Izhikevich neurons (4 neurons per SIMD operation)
    /// Processes neurons in groups of 4 using AVX/SSE/NEON instructions simultaneously.
    #[derive(Debug, Clone)]
    pub struct IzhikevichBatch {
        /// Membrane potential (V) - packed 4 at a time
        pub v: Vec<f32x4>,
        /// Recovery variable (U) - packed 4 at a time
        pub u: Vec<f32x4>,
        /// Parameter A (packed) - 4 copies
        pub a: Vec<f32x4>,
        /// Parameter B (packed) - 4 copies
        pub b: Vec<f32x4>,
        /// Parameter C (packed) - 4 copies
        pub c: Vec<f32x4>,
        /// Parameter D (packed) - 4 copies
        pub d: Vec<f32x4>,
        /// Spike bitmasking (32-bit: 1 bit per neuron in group)
        pub spikes: Vec<u32>,
        /// Track how many neurons this batch contains (for partial final group)
        pub neuron_count: usize,
    }

    impl IzhikevichBatch {
        /// Create batch for N neurons (padded to multiple of 4)
        pub fn new(neuron_count: usize, params: &IzhikevichBatchParams) -> Self {
            let groups = (neuron_count + 3) / 4;
            
            Self {
                v: vec![f32x4::splat(params.c as f32); groups],
                u: vec![f32x4::splat(-13.0); groups],
                a: vec![f32x4::splat(params.a as f32); groups],
                b: vec![f32x4::splat(params.b as f32); groups],
                c: vec![f32x4::splat(params.c as f32); groups],
                d: vec![f32x4::splat(params.d as f32); groups],
                spikes: vec![0u32; groups],
                neuron_count,
            }
        }

        /// Update all neurons in batch
        pub fn update(&mut self, inputs: &[f32x4], params: &IzhikevichBatchParams) {
            let dt_simd = f32x4::splat(params.dt as f32);
            let const_004 = f32x4::splat(0.04);
            let const_5 = f32x4::splat(5.0);
            let const_140 = f32x4::splat(140.0);
            let threshold = f32x4::splat(30.0);

            for i in 0..self.v.len() {
                let v = self.v[i];
                let u = self.u[i];
                let i_input = if i < inputs.len() {
                    inputs[i]
                } else {
                    f32x4::splat(0.0)
                };

                // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
                let dv = const_004 * v * v + const_5 * v + const_140 - u + i_input;
                
                // du/dt = a * (b*v - u)
                let du = self.a[i] * (self.b[i] * v - u);

                let new_v = v + dv * dt_simd;
                let new_u = u + du * dt_simd;

                // Check spike threshold
                let spiked = new_v.cmp_gt(threshold);
                let spike_mask = spiked.move_mask() as u32;
                
                // Manual conditional reset: if spiked, use reset values; else use updated values
                let new_v_arr: [f32; 4] = new_v.to_array().into();
                let c_arr: [f32; 4] = self.c[i].to_array().into();
                let new_u_arr: [f32; 4] = new_u.to_array().into();
                let d_arr: [f32; 4] = self.d[i].to_array().into();
                
                let mut v_out = [0f32; 4];
                let mut u_out = [0f32; 4];
                
                for j in 0..4 {
                    let spike = ((spike_mask >> j) & 1) != 0; // true if this lane spiked
                    if spike {
                        v_out[j] = c_arr[j];
                        u_out[j] = new_u_arr[j] + d_arr[j];
                    } else {
                        v_out[j] = new_v_arr[j];
                        u_out[j] = new_u_arr[j];
                    }
                }
                
                self.v[i] = f32x4::from(v_out);
                self.u[i] = f32x4::from(u_out);
                self.spikes[i] = spike_mask;
            }
        }

        /// Check if neuron at index spiked
        pub fn did_spike(&self, neuron_idx: usize) -> bool {
            let group_idx = neuron_idx / 4;
            let idx_in_group = neuron_idx % 4;
            
            if group_idx >= self.spikes.len() {
                return false;
            }
            
            let mask = self.spikes[group_idx];
            (mask >> idx_in_group) & 1 == 1
        }

        /// Get spike count in this batch
        pub fn spike_count(&self) -> usize {
            self.spikes.iter().map(|m| m.count_ones() as usize).sum()
        }

        /// Get membrane potentials as Vec<StateValue>
        pub fn get_v(&self) -> Vec<StateValue> {
            let mut result = Vec::with_capacity(self.neuron_count);
            for (group_idx, group) in self.v.iter().enumerate() {
                let v_array = group.to_array();
                for (idx, &v) in v_array.iter().enumerate() {
                    if group_idx * 4 + idx < self.neuron_count {
                        result.push(v as StateValue);
                    }
                }
            }
            result
        }

        /// Get recovery variables as Vec<StateValue>
        pub fn get_u(&self) -> Vec<StateValue> {
            let mut result = Vec::with_capacity(self.neuron_count);
            for (group_idx, group) in self.u.iter().enumerate() {
                let u_array = group.to_array();
                for (idx, &u) in u_array.iter().enumerate() {
                    if group_idx * 4 + idx < self.neuron_count {
                        result.push(u as StateValue);
                    }
                }
            }
            result
        }
    }

    /// Helper: pack scalar values into f32x4 chunks for batch input
    pub fn pack_input_currents(currents: &[StateValue]) -> Vec<f32x4> {
        let mut packed = Vec::new();
        let mut buf = [0.0f32; 4];
        
        for (i, &curr) in currents.iter().enumerate() {
            buf[i % 4] = curr as f32;
            
            if (i + 1) % 4 == 0 || i == currents.len() - 1 {
                if i == currents.len() - 1 && currents.len() % 4 != 0 {
                    // Pad final group with zeros
                    for j in (currents.len() % 4)..4 {
                        buf[j] = 0.0;
                    }
                }
                packed.push(f32x4::from(buf));
                buf = [0.0f32; 4];
            }
        }
        
        packed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::simd::*;

    #[test]
    fn izhikevich_spike_after_high_current() {
        let params = IzhikevichParams::default();
        let state = IzhikevichNeuron::initialize(params.clone());
        let input = vec![PrimitiveMessage::InputCurrent(1000.0)];

        let (_new_state, outputs) = IzhikevichNeuron::update(state, &params, &input);

        assert!(!outputs.is_empty());
        assert!(matches!(outputs[0], PrimitiveMessage::Spike { .. }));
    }

    #[test]
    fn izhikevich_batch_spike_detection() {
        let params = IzhikevichBatchParams::from_scalar(0.02, 0.2, -65.0, 8.0, 0.1);
        let mut batch = IzhikevichBatch::new(8, &params);
        
        // High input current to trigger spike
        let inputs = vec![wide::f32x4::splat(1000.0); 2];
        batch.update(&inputs, &params);
        
        // At least some neurons should spike
        assert!(batch.spike_count() > 0);
    }

    #[test]
    fn izhikevich_batch_reset() {
        let params = IzhikevichBatchParams::from_scalar(0.02, 0.2, -65.0, 8.0, 0.1);
        let mut batch = IzhikevichBatch::new(4, &params);
        let inputs = vec![wide::f32x4::splat(1000.0)];
        
        batch.update(&inputs, &params);
        
        // If any neuron spiked, spike_count > 0
        let spike_count = batch.spike_count();
        assert!(spike_count > 0, "Should have spikes with high input");
        
        // Verify that spiked neurons have V reset to C (-65.0)
        let v_vals = batch.get_v();
        let mut reset_count = 0;
        for (i, &v) in v_vals.iter().enumerate() {
            if batch.did_spike(i) {
                // Allow small floating point error
                assert!((v - (-65.0)).abs() < 0.1, "Spiked neuron {} should reset to -65.0, got {}", i, v);
                reset_count += 1;
            }
        }
        assert!(reset_count > 0, "At least one neuron should have reset");
    }

    #[test]
    fn pack_input_currents_test() {
        let currents = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let packed = pack_input_currents(&currents);
        
        assert_eq!(packed.len(), 2);
        
        let arr0 = packed[0].to_array();
        assert_eq!(arr0[0], 1.0);
        assert_eq!(arr0[3], 4.0);
        
        let arr1 = packed[1].to_array();
        assert_eq!(arr1[0], 5.0);
        assert_eq!(arr1[1], 0.0); // Padded
    }
}
