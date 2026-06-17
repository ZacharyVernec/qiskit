// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
// Copied from ../vf2/mod.rs
// Zachary Vernec modified this file to create a fixed-point VF2 variant

mod error_map;
mod vf2_layout;

pub use error_map::error_map_mod;
pub use vf2_layout::{
    Vf2PassConfiguration, Vf2PassReturn, vf2_layout_mod, vf2_layout_pass_average,
    vf2_layout_pass_exact,
};
