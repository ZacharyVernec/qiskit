// This code is an original file part of Zachary Vernec's fork of Qiskit.
//
// (C) Copyright Zachary Vernec 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// NOTE: This module re-exports the ErrorMap from the parent vf2 module to avoid
// duplicate #[pyclass] conflicts. The ErrorMap PyO3 class is registered only by
// the vf2 module's error_map_mod.

pub use crate::passes::vf2::ErrorMap;

/// Re-export the parent module's `error_map_mod` to avoid duplicate PyO3 class
/// registration.  The parent vf2 module is responsible for registering `ErrorMap`.
pub fn error_map_mod(
    m: &pyo3::prelude::Bound<pyo3::prelude::PyModule>,
) -> pyo3::prelude::PyResult<()> {
    crate::passes::vf2::error_map_mod(m)
}
