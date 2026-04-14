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

pub mod abstraction_layer;
pub mod aggregation_functions;
pub mod cluster_detection;

pub use abstraction_layer::{AbstractionLayerManager, AgentGroupAbstraction, AggregationFnKind};
pub use aggregation_functions::{aggregate_max, aggregate_mean, aggregate_sum, aggregate_vote};
pub use cluster_detection::detect_dense_groups;
