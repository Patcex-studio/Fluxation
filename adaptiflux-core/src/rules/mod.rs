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

pub mod behavior;
pub mod consistency;
mod engine;
pub mod structural_plasticity;
pub mod topology;

pub use behavior::{BehaviorAction, BehaviorRule};
pub use consistency::{ConsistencyCheck, ConsistencyError};
pub use engine::RuleEngine;
pub use structural_plasticity::{
    ActivityDependentSynaptogenesisRule, ClusterGroupingPlasticityRule,
    GrowthFactorNeurogenesisRule, HomeostaticApoptosisRule, PlasticityRule, SynapticPruningRule,
};
pub use topology::{PruneReason, TopologyAction, TopologyRule};
