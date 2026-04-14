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

pub mod online_adaptation;
pub mod parameter_tuning;
pub mod signal_integration;

pub use online_adaptation::{
    EvolutionaryOptimizerLearner, EvolutionaryTuneState, GradientDescentLearner,
    OnlineAdaptationEngine, OnlineLearner,
};

#[cfg(feature = "adaptiflux_optim")]
pub use online_adaptation::AdaptiveOptimizerLearner;
#[cfg(feature = "custom_optim")]
pub use online_adaptation::CustomOptimizerLearner;
pub use parameter_tuning::{
    evolutionary_scalar_update, evolutionary_tau_update, pid_gains_sgd_step, sgd_step,
};
pub use signal_integration::{
    collect_error_feedback_from_bus, merge_from_posted_errors, FeedbackSignal,
};
