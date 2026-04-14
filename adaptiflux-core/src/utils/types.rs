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

use uuid::Uuid;

pub type ZoooidId = Uuid;

/// Core parameter type for all agent parameters and optimization
pub type Param = f32;

/// Core state value type for all internal computations
pub type StateValue = f32;

/// Core gradient type for optimization
pub type Gradient = f32;

pub fn new_zoooid_id() -> ZoooidId {
    Uuid::new_v4()
}
