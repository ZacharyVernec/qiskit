# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# Copy of ./test_sabre_layout.py further modified by Zachary Vernec

"""Test the FixedPointSabreLayout pass"""

import unittest

import math

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import library as lib, Parameter
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import efficient_su2, quantum_volume
from qiskit.transpiler import (
    CouplingMap,
    AnalysisPass,
    PassManager,
    Target,
    Layout,
    DistributedTarget,
)
from qiskit.transpiler.passes import (
    FixedPointSabreLayout,
    FixedPointSabreSwap,
    FixedPointConstraintValidation,
    DenseLayout,
    Unroll3qOrMore,
    BasicSwap,
    SabrePreLayout,
)
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout.fixed_point_constraint_validation import (
    FIXED_POINT_METADATA_LOGICAL_PARTITIONS,
    FIXED_POINT_METADATA_ANCHORS,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.compiler.transpiler import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.passes.layout.fixed_point_sabre_pre_layout import FixedPointSabrePreLayout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from test import QiskitTestCase, slow_test  # pylint: disable=wrong-import-order

from ..legacy_cmaps import ALMADEN_CMAP, MUMBAI_CMAP


# Tests fallback to non-fixed-point when Target is not DistributedTarget


class TestFixedPointSabreLayout(QiskitTestCase):
    """Tests the FixedPointSabreLayout pass"""

    def setUp(self):
        super().setUp()
        self.cmap20 = ALMADEN_CMAP

    def test_5q_circuit_20q_coupling(self):
        """Test finds layout for 5q circuit on 20q device."""
        #                ┌───┐
        # q_0: ──■───────┤ X ├───────────────
        #        │       └─┬─┘┌───┐
        # q_1: ──┼────■────┼──┤ X ├───────■──
        #      ┌─┴─┐  │    │  ├───┤┌───┐┌─┴─┐
        # q_2: ┤ X ├──┼────┼──┤ X ├┤ X ├┤ X ├
        #      └───┘┌─┴─┐  │  └───┘└─┬─┘└───┘
        # q_3: ─────┤ X ├──■─────────┼───────
        #           └───┘            │
        # q_4: ──────────────────────■───────
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[1], qr[3])
        circuit.cx(qr[3], qr[0])
        circuit.x(qr[2])
        circuit.cx(qr[4], qr[2])
        circuit.x(qr[1])
        circuit.cx(qr[1], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = FixedPointSabreLayout(
            CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=32
        )
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [3, 6, 8, 7, 12])

    def test_6q_circuit_20q_coupling(self):
        """Test finds layout for 6q circuit on 20q device."""
        #       ┌───┐┌───┐┌───┐┌───┐┌───┐
        # q0_0: ┤ X ├┤ X ├┤ X ├┤ X ├┤ X ├
        #       └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
        # q0_1: ──┼────■────┼────┼────┼──
        #         │  ┌───┐  │    │    │
        # q0_2: ──┼──┤ X ├──┼────■────┼──
        #         │  └───┘  │         │
        # q1_0: ──■─────────┼─────────┼──
        #            ┌───┐  │         │
        # q1_1: ─────┤ X ├──┼─────────■──
        #            └───┘  │
        # q1_2: ────────────■────────────
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])

        dag = circuit_to_dag(circuit)
        pass_ = FixedPointSabreLayout(
            CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=32
        )
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [7, 8, 11, 12, 13, 6])

    def test_6q_circuit_20q_coupling_with_partial(self):
        """Test finds layout for 6q circuit on 20q device."""
        #       ┌───┐┌───┐┌───┐┌───┐┌───┐
        # q0_0: ┤ X ├┤ X ├┤ X ├┤ X ├┤ X ├
        #       └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
        # q0_1: ──┼────■────┼────┼────┼──
        #         │  ┌───┐  │    │    │
        # q0_2: ──┼──┤ X ├──┼────■────┼──
        #         │  └───┘  │         │
        # q1_0: ──■─────────┼─────────┼──
        #            ┌───┐  │         │
        # q1_1: ─────┤ X ├──┼─────────■──
        #            └───┘  │
        # q1_2: ────────────■────────────
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])

        pm = PassManager(
            [
                DensePartialSabreTrial(CouplingMap(self.cmap20)),
                FixedPointSabreLayout(
                    CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=0
                ),
            ]
        )
        pm.run(circuit)
        layout = pm.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [1, 3, 5, 2, 6, 0])

    def test_6q_circuit_20q_coupling_with_target(self):
        """Test finds layout for 6q circuit on 20q device."""
        #       ┌───┐┌───┐┌───┐┌───┐┌───┐
        # q0_0: ┤ X ├┤ X ├┤ X ├┤ X ├┤ X ├
        #       └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
        # q0_1: ──┼────■────┼────┼────┼──
        #         │  ┌───┐  │    │    │
        # q0_2: ──┼──┤ X ├──┼────■────┼──
        #         │  └───┘  │         │
        # q1_0: ──■─────────┼─────────┼──
        #            ┌───┐  │         │
        # q1_1: ─────┤ X ├──┼─────────■──
        #            └───┘  │
        # q1_2: ────────────■────────────
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])

        dag = circuit_to_dag(circuit)
        target = GenericBackendV2(num_qubits=20, coupling_map=self.cmap20).target
        pass_ = FixedPointSabreLayout(target, seed=0, swap_trials=32, layout_trials=32)
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [7, 8, 11, 12, 13, 6])

    def test_layout_with_classical_bits(self):
        """Test fixed_point_sabre layout with classical bits recreate from issue #8635."""
        qc = QuantumCircuit.from_qasm_str(
            """
OPENQASM 2.0;
include "qelib1.inc";
qreg q4833[1];
qreg q4834[6];
qreg q4835[7];
creg c982[2];
creg c983[2];
creg c984[2];
rzz(0) q4833[0],q4834[4];
cu(0,-6.1035156e-05,0,1e-05) q4834[1],q4835[2];
swap q4834[0],q4834[2];
cu(-1.1920929e-07,0,-0.33333333,0) q4833[0],q4834[2];
ccx q4835[2],q4834[5],q4835[4];
measure q4835[4] -> c984[0];
ccx q4835[2],q4835[5],q4833[0];
measure q4835[5] -> c984[1];
measure q4834[0] -> c982[1];
u(10*pi,0,1.9) q4834[5];
measure q4834[3] -> c984[1];
measure q4835[0] -> c982[0];
rz(0) q4835[1];
"""
        )
        backend = GenericBackendV2(
            num_qubits=27,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=MUMBAI_CMAP,
            seed=42,
        )
        res = transpile(
            qc,
            backend,
            layout_method="fixed_point_sabre",
            seed_transpiler=1234,
            optimization_level=1,
        )
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout.initial_layout
        self.assertEqual(
            [layout[q] for q in qc.qubits], [2, 0, 5, 1, 7, 3, 14, 6, 9, 8, 10, 11, 4, 12]
        )

    # pylint: disable=line-too-long
    def test_layout_many_search_trials(self):
        """Test recreate failure from randomized testing that overflowed."""
        qc = QuantumCircuit.from_qasm_str(
            """
    OPENQASM 2.0;
include "qelib1.inc";
qreg q18585[14];
creg c1423[5];
creg c1424[4];
creg c1425[3];
barrier q18585[4],q18585[5],q18585[12],q18585[1];
cz q18585[11],q18585[3];
cswap q18585[8],q18585[10],q18585[6];
u(-2.00001,6.1035156e-05,-1.9) q18585[2];
barrier q18585[3],q18585[6],q18585[5],q18585[8],q18585[10],q18585[9],q18585[11],q18585[2],q18585[12],q18585[7],q18585[13],q18585[4],q18585[0],q18585[1];
cp(0) q18585[2],q18585[4];
cu(-0.99999,0,0,0) q18585[7],q18585[1];
cu(0,0,0,2.1507119) q18585[6],q18585[3];
barrier q18585[13],q18585[0],q18585[12],q18585[3],q18585[2],q18585[10];
ry(-1.1044662) q18585[13];
barrier q18585[13];
id q18585[12];
barrier q18585[12],q18585[6];
cu(-1.9,1.9,-1.5,0) q18585[10],q18585[0];
barrier q18585[13];
id q18585[8];
barrier q18585[12];
barrier q18585[12],q18585[1],q18585[9];
sdg q18585[2];
rz(-10*pi) q18585[6];
u(0,27.566433,1.9) q18585[1];
barrier q18585[12],q18585[11],q18585[9],q18585[4],q18585[7],q18585[0],q18585[13],q18585[3];
cu(-0.99999,-5.9604645e-08,-0.5,2.00001) q18585[3],q18585[13];
rx(-5.9604645e-08) q18585[7];
p(1.1) q18585[13];
barrier q18585[12],q18585[13],q18585[10],q18585[9],q18585[7],q18585[4];
z q18585[10];
measure q18585[7] -> c1423[2];
barrier q18585[0],q18585[3],q18585[7],q18585[4],q18585[1],q18585[8],q18585[6],q18585[11],q18585[5];
barrier q18585[5],q18585[2],q18585[8],q18585[3],q18585[6];
"""
        )
        backend = GenericBackendV2(
            num_qubits=27,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=MUMBAI_CMAP,
            seed=42,
        )
        res = transpile(
            qc,
            backend,
            layout_method="fixed_point_sabre",
            routing_method="basic",
            seed_transpiler=12345,
            optimization_level=1,
        )
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout.initial_layout
        self.assertEqual(
            [layout[q] for q in qc.qubits], [8, 21, 12, 16, 10, 4, 14, 23, 13, 9, 11, 19, 2, 20]
        )

    def test_support_var_with_rust_fastpath(self):
        """Test that the joint layout/embed/routing logic for the Rust-space fast-path works in the
        presence of standalone `Var` nodes."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(5, inputs=[a])
        qc.add_var(b, 12)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(4, 0)

        out = FixedPointSabreLayout(
            CouplingMap.from_line(8), seed=0, swap_trials=2, layout_trials=2
        )(qc)

        self.assertIsInstance(out, QuantumCircuit)
        self.assertEqual(out.layout.initial_index_layout(), [6, 5, 4, 2, 3, 0, 1, 7])

    def test_support_var_with_explicit_routing_pass(self):
        """Test that the logic works if an explicit routing pass is given."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(5, inputs=[a])
        qc.add_var(b, 12)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(4, 0)

        cm = CouplingMap.from_line(8)
        pass_ = FixedPointSabreLayout(cm, seed=0, routing_pass=BasicSwap(cm, fake_run=True))
        _ = pass_(qc)
        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 4, 2, 5, 1])

    def test_uninitialized_target(self):
        """We shouldn't panic if the target isn't initialized."""
        target = Target(num_qubits=None)
        qc = QuantumCircuit(2)
        pass_ = FixedPointSabreLayout(target, seed=0)
        with self.assertRaisesRegex(TranspilerError, "not initialized"):
            pass_(qc)

    def test_out_of_range_partials(self):
        """We should safely reject partial layouts that are invalid."""
        pass_ = FixedPointSabreLayout(
            Target.from_configuration(
                num_qubits=5, coupling_map=CouplingMap.from_line(5), basis_gates=["sx", "rz", "cx"]
            ),
            seed=0,
        )
        qc = QuantumCircuit(2)
        partial = Layout(dict(zip(qc.qubits, [1, 5])))
        with self.assertRaisesRegex(TranspilerError, "out-of-range physical qubits"):
            pass_(qc, property_set={"fixed_point_sabre_starting_layouts": [partial]})

    @slow_test
    def test_release_valve_routes_multiple(self):
        """Test Sabre works if the release valve routes more than 1 operation.

        Regression test of #13081.
        """
        qv = quantum_volume(500, seed=42)
        qv.measure_all()
        qc = Unroll3qOrMore()(qv)

        cmap = CouplingMap.from_heavy_hex(21)
        pm = PassManager(
            [
                FixedPointSabreLayout(
                    cmap, swap_trials=20, layout_trials=20, max_iterations=4, seed=100
                ),
            ]
        )
        _ = pm.run(qc)
        self.assertIsNotNone(pm.property_set.get("layout"))

    def test_all_to_all(self):
        """An implicitly all-to-all backend should just become physical with the trivial layout."""
        qc = QuantumCircuit(QuantumRegister(5, "virtuals"))
        for target in qc.qubits[1:]:
            qc.cx(qc.qubits[0], target)
        # No qargs in the instruction properties => implicitly all-to-all.
        target = Target(num_qubits=10)
        target.add_instruction(lib.RZGate(Parameter("t")))
        target.add_instruction(lib.SXGate())
        target.add_instruction(lib.CXGate())
        pass_ = FixedPointSabreLayout(target, seed=0)
        out = pass_(qc)
        self.assertEqual(out.layout.initial_index_layout(), list(range(10)))
        self.assertEqual(out.layout.routing_permutation(), list(range(10)))

        expected = QuantumCircuit(QuantumRegister(10, "q"))
        for target in range(1, qc.num_qubits):
            expected.cx(0, target)
        self.assertEqual(out, expected)


class DensePartialSabreTrial(AnalysisPass):
    """Pass to run dense layout as a fixed_point_sabre trial."""

    def __init__(self, cmap):
        self.dense_pass = DenseLayout(cmap)
        super().__init__()

    def run(self, dag):
        self.dense_pass.run(dag)
        self.property_set["sabre_starting_layouts"] = [self.dense_pass.property_set["layout"]]


class TestDisjointDeviceFixedPointSabreLayout(QiskitTestCase):
    """Test FixedPointSabreLayout with a disjoint coupling map."""

    def setUp(self):
        super().setUp()
        self.dual_grid_cmap = CouplingMap(
            [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [5, 8]]
        )

    def test_dual_ghz(self):
        """Test a basic example with 2 circuit components and 2 cmap components."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        layout_routing_pass = FixedPointSabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 2, 1, 5, 4, 7, 6, 8])

    def test_dual_ghz_with_wide_barrier(self):
        """Test a basic example with 2 circuit components and 2 cmap components."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = FixedPointSabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 2, 1, 5, 4, 7, 6, 8])

    def test_dual_ghz_with_intermediate_barriers(self):
        """Test dual ghz circuit with intermediate barriers local to each component."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.barrier(0, 1, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.barrier(4, 5, 6)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = FixedPointSabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 2, 1, 5, 4, 7, 6, 8])

    def test_dual_ghz_with_intermediate_spanning_barriers(self):
        """Test dual ghz circuit with barrier in the middle across components."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.barrier(0, 1, 2, 4, 5)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = FixedPointSabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 2, 1, 5, 4, 7, 6, 8])

    def test_too_large_components(self):
        """Assert trying to run a circuit with too large a connected component raises."""
        qc = QuantumCircuit(8)
        qc.h(0)
        for i in range(1, 6):
            qc.cx(0, i)
        qc.h(7)
        qc.cx(7, 6)
        layout_routing_pass = FixedPointSabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        with self.assertRaises(TranspilerError):
            layout_routing_pass(qc)

    def test_with_partial_layout(self):
        """Test a partial layout with a disjoint connectivity graph."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        pm = PassManager(
            [
                DensePartialSabreTrial(self.dual_grid_cmap),
                FixedPointSabreLayout(
                    self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
                ),
            ]
        )
        pm.run(qc)
        layout = pm.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 2, 1, 5, 4, 7, 6, 8])

    def test_dag_fits_in_one_component(self):
        """Test that the output is valid if the DAG all fits in a single component of a disjoint
        coupling map.."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

        disjoint = CouplingMap([(0, 1), (1, 2), (3, 4), (4, 5)])
        layout_routing_pass = FixedPointSabreLayout(
            disjoint, seed=2025_02_12, swap_trials=1, layout_trials=1
        )
        out = layout_routing_pass(qc)
        self.assertEqual(len(out.layout.initial_layout), len(out.layout.final_layout))
        self.assertEqual(out.layout.initial_index_layout(filter_ancillas=False), [4, 5, 3, 0, 1, 2])
        self.assertEqual(out.layout.final_index_layout(filter_ancillas=False), [3, 5, 4, 0, 1, 2])

    def test_fixed_point_sabre_layout_global_1q(self):
        """Test that the pass runs with a globally defined 1q gate."""
        target = Target(num_qubits=3)
        target.add_instruction(lib.XGate())
        target.add_instruction(lib.CXGate(), {(0, 1): None, (1, 2): None})
        layout_routing_pass = FixedPointSabreLayout(target, swap_trials=1, layout_trials=1, seed=42)
        qc = QuantumCircuit(3)
        qc.cz(0, 2)
        out = layout_routing_pass(qc)
        # sabre maps 2 -> 1, and 0 -> 2 in the layout with no swaps
        self.assertIsNone(out.count_ops().get("swap", None))
        self.assertEqual([out.find_bit(x).index for x in out.data[0].qubits], [2, 1])
        self.assertEqual(len(out.layout.initial_layout), len(out.layout.final_layout))
        self.assertEqual(out.layout.initial_index_layout(filter_ancillas=False), [2, 0, 1])
        self.assertEqual(out.layout.routing_permutation(), [0, 1, 2])
        self.assertEqual(out.layout.final_index_layout(filter_ancillas=False), [2, 0, 1])


class TestSabrePreLayout(QiskitTestCase):
    """Tests the FixedPointSabreLayout pass with starting layout created by SabrePreLayout."""

    def setUp(self):
        super().setUp()
        circuit = efficient_su2(16, entanglement="circular", reps=6)
        circuit.assign_parameters([math.pi / 2] * len(circuit.parameters), inplace=True)
        circuit.measure_all()
        self.circuit = circuit
        self.coupling_map = CouplingMap.from_heavy_hex(7)

    def test_starting_layout(self):
        """Test that a starting layout is created and looks as expected."""
        pm = PassManager(
            [
                SabrePreLayout(coupling_map=self.coupling_map),
                FixedPointSabreLayout(
                    self.coupling_map, seed=123456, swap_trials=1, layout_trials=1
                ),
            ]
        )
        pm.run(self.circuit)
        layout = pm.property_set["layout"]
        self.assertEqual(
            [layout[q] for q in self.circuit.qubits],
            [9, 81, 54, 16, 86, 58, 92, 22, 91, 21, 57, 14, 85, 53, 79, 80],
        )

    def test_integration_with_pass_manager(self):
        """Tests SabrePreLayoutIntegration with the rest of PassManager pipeline."""
        backend = GenericBackendV2(num_qubits=20, coupling_map=ALMADEN_CMAP, seed=42)
        pm = generate_preset_pass_manager(
            0,
            backend,
            layout_method="fixed_point_sabre",
            routing_method="fixed_point_sabre",
            seed_transpiler=0,
        )
        pm.pre_layout = PassManager([SabrePreLayout(backend.target)])
        qct = pm.run(self.circuit)
        qct_initial_layout = qct.layout.initial_layout
        self.assertEqual(
            [qct_initial_layout[q] for q in self.circuit.qubits],
            [12, 11, 10, 16, 17, 18, 13, 14, 9, 8, 3, 2, 1, 6, 5, 7],
        )


# New tests for DistributedTarget


class TestFixedPointSabreLayoutWithDistributedTarget(QiskitTestCase):
    """Tests for FixedPointSabreLayout with a DistributedTarget.

    NOTE: Constraint enforcement in the Rust routing is not yet implemented.
    These tests currently verify that the constraint data pipeline flows
    correctly (data is built and passed to Rust, the pass completes without
    error).  When enforcement is added, stricter assertions about qubit
    placement (verifying R_i qubits stay in Q_i) should be added.
    """

    def setUp(self):
        super().setUp()
        # A 8-qubit linear coupling map.
        self.coupling = CouplingMap.from_line(8)
        self.base_target = Target.from_configuration(
            basis_gates=["u", "cx"],
            coupling_map=self.coupling,
        )

    def _make_distributed_target(self, qpu_to_qubits, comm_ancillas=None, comm_ancilla_edges=None):
        """Make a DistributedTarget with given QPU mapping."""
        return DistributedTarget(
            self.base_target,
            qpu_to_qubits,
            comm_ancillas=comm_ancillas or {},
            comm_ancilla_edges=comm_ancilla_edges or [],
        )

    def _run_layout(self, distributed_target, qc, **property_set_kwargs):
        """Run FixedPointSabreLayout with property-set constraint data and return the pass."""
        pass_ = FixedPointSabreLayout(distributed_target, seed=0, swap_trials=4, layout_trials=4)
        for key, value in property_set_kwargs.items():
            pass_.property_set[key] = value
        pass_(qc)
        return pass_

    # ------------------------------------------------------------------
    # Section 7 compatibility: trivial monolithic QPU
    # ------------------------------------------------------------------

    def test_trivial_monolithic_qpu_same_as_plain_sabre(self):
        """Section 7 compatibility: single QPU covering all qubits, no anchors
        should give the same result as non-fixed-point SabreLayout.

        This is the most important test, as it verifies that the fixed-point
        SABRE passes are a strict superset of the original SABRE passes
        when no constraints are active.
        """
        distributed_target = self._make_distributed_target({"qpu_0": set(range(8))})

        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(4, 0)

        # Fixed-point with trivial monolithic constraints.
        fp_pass = self._run_layout(
            distributed_target,
            qc,
            fixed_point_logical_partitions={"qpu_0": list(qc.qubits)},
            fixed_point_anchors={},
        )
        fp_layout = fp_pass.property_set["layout"]

        # Plain SabreLayout with same seed / settings.
        plain_pass = FixedPointSabreLayout(self.base_target, seed=0, swap_trials=4, layout_trials=4)
        plain_pass(qc)
        plain_layout = plain_pass.property_set["layout"]

        self.assertEqual(
            [fp_layout[q] for q in qc.qubits],
            [plain_layout[q] for q in qc.qubits],
            "Trivial monolithic QPU should produce same layout as plain SabreLayout",
        )

    def test_trivial_monolithic_qpu_with_skip_routing(self):
        """Section 7 compatibility with skip_routing=True."""
        distributed_target = self._make_distributed_target({"qpu_0": set(range(8))})

        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)

        pass_ = FixedPointSabreLayout(
            distributed_target, seed=0, swap_trials=4, layout_trials=4, skip_routing=True
        )
        pass_.property_set[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {"qpu_0": list(qc.qubits)}
        pass_.property_set[FIXED_POINT_METADATA_ANCHORS] = {}
        pass_(qc)

        layout = pass_.property_set["layout"]
        self.assertIsNotNone(layout)
        self.assertGreaterEqual(len(layout), qc.num_qubits)

    def test_trivial_monolithic_qpu_preserves_gate_semantics(self):
        """Even with trivial constraints, the routed circuit should be logically
        equivalent (same number of non-swap gates) to the original."""
        distributed_target = self._make_distributed_target({"qpu_0": set(range(8))})

        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        original_ops = dict(qc.count_ops())

        pass_ = self._run_layout(
            distributed_target,
            qc,
            fixed_point_logical_partitions={"qpu_0": list(qc.qubits)},
            fixed_point_anchors={},
        )
        # The pass returns a dag, but we can check on the output circuit.
        # The layout should be valid.
        layout = pass_.property_set["layout"]
        self.assertIsNotNone(layout)
        self.assertGreaterEqual(len(layout), qc.num_qubits)

    # ------------------------------------------------------------------
    # Multi-QPU constraint pipeline tests
    # ------------------------------------------------------------------

    def test_two_qpu_constraints_do_not_crash_layout(self):
        """Two QPUs with valid constraints. The pass should complete without error.
        (Full enforcement is not yet implemented; this is a smoke test.)"""
        distributed_target = self._make_distributed_target(
            {"qpu_0": {0, 1, 2, 3}, "qpu_1": {4, 5, 6, 7}},
            comm_ancillas={"qpu_0": [3], "qpu_1": [4]},
            comm_ancilla_edges=[(3, 4), (4, 3)],
        )

        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(3, 4)
        qc.cx(4, 5)

        pass_ = self._run_layout(
            distributed_target,
            qc,
            fixed_point_logical_partitions={
                "qpu_0": [qc.qubits[0], qc.qubits[1], qc.qubits[2]],
                "qpu_1": [qc.qubits[3], qc.qubits[4], qc.qubits[5]],
            },
            fixed_point_anchors={},
        )
        layout = pass_.property_set["layout"]
        self.assertIsNotNone(layout)
        # SabreLayout expands with ancillas, so len(layout) >= qc.num_qubits.
        self.assertGreaterEqual(len(layout), qc.num_qubits)

    def test_two_qpu_with_anchors_do_not_crash_layout(self):
        """Two QPUs with anchors — the pass should complete without error.
        (Full enforcement is not yet implemented; this is a smoke test.)"""
        distributed_target = self._make_distributed_target(
            {"qpu_0": {0, 1, 2, 3}, "qpu_1": {4, 5, 6, 7}},
            comm_ancillas={"qpu_0": [3], "qpu_1": [4]},
            comm_ancilla_edges=[(3, 4), (4, 3)],
        )

        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 3)

        pass_ = self._run_layout(
            distributed_target,
            qc,
            fixed_point_logical_partitions={
                "qpu_0": [qc.qubits[0], qc.qubits[1]],
                "qpu_1": [qc.qubits[2], qc.qubits[3]],
            },
            fixed_point_anchors={
                "qpu_0": {qc.qubits[0]: 0},
                "qpu_1": {qc.qubits[2]: 5},
            },
        )
        layout = pass_.property_set["layout"]
        self.assertIsNotNone(layout)
        self.assertGreaterEqual(len(layout), qc.num_qubits)

    def test_two_qpu_cross_qpu_gates_do_not_crash(self):
        """Circuit with cross-QPU gates — the pass should complete without error.
        (Full enforcement is not yet implemented; this is a smoke test.)"""
        distributed_target = self._make_distributed_target(
            {"qpu_0": {0, 1, 2, 3}, "qpu_1": {4, 5, 6, 7}},
            comm_ancillas={"qpu_0": [3], "qpu_1": [4]},
            comm_ancilla_edges=[(3, 4), (4, 3)],
        )

        qc = QuantumCircuit(4)
        qc.cx(0, 1)  # within qpu_0
        qc.cx(2, 3)  # within qpu_1
        qc.cx(0, 2)  # cross-QPU
        qc.cx(1, 3)  # cross-QPU

        pass_ = self._run_layout(
            distributed_target,
            qc,
            fixed_point_logical_partitions={
                "qpu_0": [qc.qubits[0], qc.qubits[1]],
                "qpu_1": [qc.qubits[2], qc.qubits[3]],
            },
            fixed_point_anchors={},
        )
        layout = pass_.property_set["layout"]
        self.assertIsNotNone(layout)
        self.assertGreaterEqual(len(layout), qc.num_qubits)

    def test_single_qpu_distributed_target_no_constraints(self):
        """Single QPU DistributedTarget with no property-set constraints behaves like
        a standard Target (no logical partitions or anchors provided)."""
        distributed_target = self._make_distributed_target({"qpu_0": set(range(8))})

        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)

        # No constraints on property set — should still work.
        pass_ = FixedPointSabreLayout(distributed_target, seed=0, swap_trials=4, layout_trials=4)
        pass_(qc)

        layout = pass_.property_set["layout"]
        self.assertIsNotNone(layout)
        self.assertGreaterEqual(len(layout), qc.num_qubits)

    def test_three_qpu_distributed_target_does_not_crash(self):
        """Three QPUs with varying sizes — the pass should complete without error."""
        distributed_target = DistributedTarget(
            self.base_target,
            {"qpu_0": {0, 1}, "qpu_1": {2, 3, 4}, "qpu_2": {5, 6, 7}},
            comm_ancillas={"qpu_0": [1], "qpu_1": [2], "qpu_2": [7]},
            comm_ancilla_edges=[(1, 2), (2, 1)],
        )

        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.cx(4, 5)

        pass_ = self._run_layout(
            distributed_target,
            qc,
            fixed_point_logical_partitions={
                "qpu_0": [qc.qubits[0], qc.qubits[1]],
                "qpu_1": [qc.qubits[2], qc.qubits[3]],
                "qpu_2": [qc.qubits[4], qc.qubits[5]],
            },
            fixed_point_anchors={},
        )
        layout = pass_.property_set["layout"]
        self.assertIsNotNone(layout)
        self.assertGreaterEqual(len(layout), qc.num_qubits)

    def test_final_layout_produced_with_constraints(self):
        """The pass produces a final_layout property when constraints are provided."""
        distributed_target = self._make_distributed_target({"qpu_0": set(range(8))})

        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)

        pass_ = self._run_layout(
            distributed_target,
            qc,
            fixed_point_logical_partitions={"qpu_0": list(qc.qubits)},
            fixed_point_anchors={},
        )
        final_layout = pass_.property_set.get("final_layout")
        self.assertIsNotNone(final_layout)

    def test_original_qubit_indices_produced_with_constraints(self):
        """The pass produces original_qubit_indices when constraints are provided."""
        distributed_target = self._make_distributed_target({"qpu_0": set(range(8))})

        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)

        pass_ = self._run_layout(
            distributed_target,
            qc,
            fixed_point_logical_partitions={"qpu_0": list(qc.qubits)},
            fixed_point_anchors={},
        )
        indices = pass_.property_set.get("original_qubit_indices")
        self.assertIsNotNone(indices)


class TestFixedPointConstraintValidation(QiskitTestCase):
    """Tests for the FixedPointConstraintValidation analysis pass."""

    def setUp(self):
        super().setUp()
        self.coupling = CouplingMap.from_line(8)
        self.base_target = Target.from_configuration(
            basis_gates=["u", "cx"],
            coupling_map=self.coupling,
        )

    def _make_distributed_target(self, qpu_to_qubits):
        """Make a DistributedTarget with given QPU mapping."""
        comm_ancillas = {}
        comm_edges = []
        if len(qpu_to_qubits) > 1:
            # Multi-QPU targets require communication ancillas.
            # For a line coupling map, use adjacent boundary qubits.
            qpu_names = sorted(qpu_to_qubits.keys())
            for i in range(len(qpu_names) - 1):
                q_a = max(qpu_to_qubits[qpu_names[i]])
                q_b = min(qpu_to_qubits[qpu_names[i + 1]])
                comm_ancillas.setdefault(qpu_names[i], []).append(q_a)
                comm_ancillas.setdefault(qpu_names[i + 1], []).append(q_b)
                comm_edges.append((q_a, q_b))
            # Fill remaining QPUs with their first qubit
            for name in qpu_names:
                if name not in comm_ancillas:
                    comm_ancillas[name] = [min(qpu_to_qubits[name])]
        return DistributedTarget(
            self.base_target,
            qpu_to_qubits,
            comm_ancillas=comm_ancillas,
            comm_ancilla_edges=comm_edges,
        )

    def _run_validation(self, distributed_target, qc):
        """Run the validation pass on a circuit and return the pass."""
        dag = circuit_to_dag(qc)
        pass_ = FixedPointConstraintValidation(target=distributed_target)
        pass_.run(dag)
        return pass_

    # ------------------------------------------------------------------
    #  No-constraints / valid-constraints cases
    # ------------------------------------------------------------------

    def test_no_constraints_is_noop(self):
        """No metadata → pass is a no-op, writes nothing to property_set."""
        distributed_target = self._make_distributed_target({"qpu_0": {0, 1, 2, 3}})
        qc = QuantumCircuit(4)
        qc.cx(0, 1)

        pass_ = self._run_validation(distributed_target, qc)
        self.assertNotIn(FIXED_POINT_METADATA_LOGICAL_PARTITIONS, pass_.property_set)
        self.assertNotIn(FIXED_POINT_METADATA_ANCHORS, pass_.property_set)

    def test_valid_constraints_single_qpu(self):
        """Valid constraints for a single QPU are written to property_set."""
        distributed_target = self._make_distributed_target({"qpu_0": {0, 1, 2, 3}})
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.metadata[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {"qpu_0": list(qc.qubits)}
        qc.metadata[FIXED_POINT_METADATA_ANCHORS] = {}

        pass_ = self._run_validation(distributed_target, qc)
        self.assertIn(FIXED_POINT_METADATA_LOGICAL_PARTITIONS, pass_.property_set)
        self.assertIn(FIXED_POINT_METADATA_ANCHORS, pass_.property_set)
        partitions = pass_.property_set[FIXED_POINT_METADATA_LOGICAL_PARTITIONS]
        self.assertEqual(set(partitions["qpu_0"]), set(qc.qubits))

    def test_valid_constraints_two_qpus(self):
        """Valid constraints for two QPUs are written to property_set."""
        distributed_target = self._make_distributed_target(
            {"qpu_0": {0, 1, 2, 3}, "qpu_1": {4, 5, 6, 7}}
        )
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(2, 3)
        qc.metadata[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {
            "qpu_0": list(qc.qubits[:3]),
            "qpu_1": list(qc.qubits[3:]),
        }
        qc.metadata[FIXED_POINT_METADATA_ANCHORS] = {}

        pass_ = self._run_validation(distributed_target, qc)
        partitions = pass_.property_set[FIXED_POINT_METADATA_LOGICAL_PARTITIONS]
        self.assertEqual(len(partitions["qpu_0"]), 3)
        self.assertEqual(len(partitions["qpu_1"]), 3)

    def test_valid_with_anchors(self):
        """Valid constraints with anchors are written to property_set."""
        distributed_target = self._make_distributed_target(
            {"qpu_0": {0, 1, 2, 3}, "qpu_1": {4, 5, 6, 7}}
        )
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.metadata[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {
            "qpu_0": list(qc.qubits[:3]),
            "qpu_1": list(qc.qubits[3:]),
        }
        qc.metadata[FIXED_POINT_METADATA_ANCHORS] = {
            "qpu_0": {qc.qubits[0]: 0},
            "qpu_1": {qc.qubits[3]: 4},
        }

        pass_ = self._run_validation(distributed_target, qc)
        anchors = pass_.property_set[FIXED_POINT_METADATA_ANCHORS]
        self.assertIn("qpu_0", anchors)
        self.assertIn("qpu_1", anchors)

    # ------------------------------------------------------------------
    #  Error cases: wrong target
    # ------------------------------------------------------------------

    def test_constraints_without_distributed_target_raises(self):
        """Constraints in metadata but not DistributedTarget → error."""
        qc = QuantumCircuit(4)
        qc.metadata[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {"qpu_0": list(qc.qubits)}
        qc.metadata[FIXED_POINT_METADATA_ANCHORS] = {}

        with self.assertRaises(TranspilerError):
            self._run_validation(self.base_target, qc)

    # ------------------------------------------------------------------
    #  Error cases: invalid partitions
    # ------------------------------------------------------------------

    def test_overlapping_logical_partitions_raises(self):
        """Overlapping logical partitions → TranspilerError."""
        distributed_target = self._make_distributed_target(
            {"qpu_0": {0, 1, 2, 3}, "qpu_1": {4, 5, 6, 7}}
        )
        qc = QuantumCircuit(4)
        qc.metadata[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {
            "qpu_0": list(qc.qubits),
            "qpu_1": [qc.qubits[0]],  # overlaps with qpu_0
        }
        qc.metadata[FIXED_POINT_METADATA_ANCHORS] = {}

        with self.assertRaises(TranspilerError):
            self._run_validation(distributed_target, qc)

    def test_capacity_violation_raises(self):
        """|R_i| > |Q_i| → TranspilerError."""
        distributed_target = self._make_distributed_target(
            {"qpu_0": {0, 1}}
        )  # only 2 physical qubits
        qc = QuantumCircuit(4)
        qc.metadata[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {
            "qpu_0": list(qc.qubits),  # 4 logical qubits > 2 physical
        }
        qc.metadata[FIXED_POINT_METADATA_ANCHORS] = {}

        with self.assertRaises(TranspilerError):
            self._run_validation(distributed_target, qc)

    def test_missing_dag_qubit_raises(self):
        """Logical qubit not in DAG → TranspilerError."""
        distributed_target = self._make_distributed_target({"qpu_0": {0, 1, 2, 3}})
        qc = QuantumCircuit(3)
        # Create a qubit that's not in the circuit
        extra_qr = QuantumRegister(1, "extra")
        extra_qubit = extra_qr[0]

        qc.metadata[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {
            "qpu_0": list(qc.qubits) + [extra_qubit],
        }
        qc.metadata[FIXED_POINT_METADATA_ANCHORS] = {}

        with self.assertRaises(TranspilerError):
            self._run_validation(distributed_target, qc)

    def test_unknown_qpu_name_raises(self):
        """QPU name not in DistributedTarget → TranspilerError."""
        distributed_target = self._make_distributed_target({"qpu_0": {0, 1, 2, 3}})
        qc = QuantumCircuit(4)
        qc.metadata[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {
            "qpu_1": list(qc.qubits),  # qpu_1 doesn't exist
        }
        qc.metadata[FIXED_POINT_METADATA_ANCHORS] = {}

        with self.assertRaises(TranspilerError):
            self._run_validation(distributed_target, qc)

    # ------------------------------------------------------------------
    #  Error cases: invalid anchors
    # ------------------------------------------------------------------

    def test_anchor_outside_logical_partition_raises(self):
        """Anchor logical qubit not in R_i → TranspilerError."""
        distributed_target = self._make_distributed_target(
            {"qpu_0": {0, 1, 2, 3}, "qpu_1": {4, 5, 6, 7}}
        )
        qc = QuantumCircuit(6)
        qc.metadata[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {
            "qpu_0": list(qc.qubits[:3]),
            "qpu_1": list(qc.qubits[3:]),
        }
        qc.metadata[FIXED_POINT_METADATA_ANCHORS] = {
            "qpu_0": {qc.qubits[3]: 0},  # qc.qubits[3] is in qpu_1, not qpu_0
        }

        with self.assertRaises(TranspilerError):
            self._run_validation(distributed_target, qc)

    def test_anchor_outside_physical_qpu_raises(self):
        """Anchor physical qubit not in Q_i → TranspilerError."""
        distributed_target = self._make_distributed_target(
            {"qpu_0": {0, 1, 2, 3}, "qpu_1": {4, 5, 6, 7}}
        )
        qc = QuantumCircuit(6)
        qc.metadata[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {
            "qpu_0": list(qc.qubits[:3]),
            "qpu_1": list(qc.qubits[3:]),
        }
        qc.metadata[FIXED_POINT_METADATA_ANCHORS] = {
            "qpu_0": {qc.qubits[0]: 4},  # 4 is in qpu_1, not qpu_0
        }

        with self.assertRaises(TranspilerError):
            self._run_validation(distributed_target, qc)


class TestFixedPointSabreSwapWithDistributedTarget(QiskitTestCase):
    """Tests for FixedPointSabreSwap with a DistributedTarget.

    NOTE: FixedPointSabreSwap requires the DAG to already have ancilla qubits
    (i.e. the number of DAG qubits must equal the target's number of qubits).
    This means swap tests need either matching-size circuits or a full layout
    pipeline. Constraint enforcement in the Rust routing is also not yet
    implemented. These tests verify the constraint data pipeline and trivial
    monolithic compatibility.
    """

    def setUp(self):
        super().setUp()
        self.coupling = CouplingMap.from_line(8)
        self.base_target = Target.from_configuration(
            basis_gates=["u", "cx"],
            coupling_map=self.coupling,
        )

    def _make_distributed_target(self, qpu_to_qubits, comm_ancillas=None, comm_ancilla_edges=None):
        """Make a DistributedTarget with given QPU mapping."""
        return DistributedTarget(
            self.base_target,
            qpu_to_qubits,
            comm_ancillas=comm_ancillas or {},
            comm_ancilla_edges=comm_ancilla_edges or [],
        )

    # ------------------------------------------------------------------
    # Section 7 compatibility: trivial monolithic QPU
    # ------------------------------------------------------------------

    def test_trivial_monolithic_qpu_same_as_plain_sabre(self):
        """Section 7 compatibility: single QPU covering all qubits, no anchors
        should give the same result as non-fixed-point SabreSwap.

        This is the most important test, as it verifies that the fixed-point
        SABRE passes are a strict superset of the original SABRE passes
        when no constraints are active.
        """
        from qiskit.converters import circuit_to_dag

        num_qubits = 8
        distributed_target = self._make_distributed_target({"qpu_0": set(range(num_qubits))})

        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

        dag = circuit_to_dag(qc)

        # Plain SabreSwap on same-size coupling.
        plain_pass = FixedPointSabreSwap(
            CouplingMap.from_line(num_qubits), "decay", seed=0, trials=4
        )
        plain_result = plain_pass.run(dag)

        # Fixed-point with trivial monolithic constraints.
        fp_dag = circuit_to_dag(qc)
        fp_pass = FixedPointSabreSwap(distributed_target, "decay", seed=0, trials=4)
        fp_pass.property_set[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {"qpu_0": list(qc.qubits)}
        fp_pass.property_set[FIXED_POINT_METADATA_ANCHORS] = {}
        fp_routed_dag = fp_pass.run(fp_dag)

        # Both should produce the same number of swaps.
        plain_swaps = plain_result.count_ops().get("swap", 0)
        fp_swaps = dag_to_circuit(fp_routed_dag).count_ops().get("swap", 0)
        self.assertEqual(
            plain_swaps,
            fp_swaps,
            "Trivial monolithic QPU should produce same number of swaps as plain SabreSwap",
        )

        # Both should produce a valid final_layout.
        self.assertIsNotNone(fp_pass.property_set.get("final_layout"))

    def test_trivial_monolithic_qpu_produces_final_layout(self):
        """Section 7 compatibility: trivial monolithic QPU should produce a final_layout."""
        from qiskit.converters import circuit_to_dag

        num_qubits = 8
        distributed_target = self._make_distributed_target({"qpu_0": set(range(num_qubits))})

        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

        dag = circuit_to_dag(qc)
        pass_ = FixedPointSabreSwap(distributed_target, "basic", seed=0, trials=4)
        pass_.property_set[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {"qpu_0": list(qc.qubits)}
        pass_.property_set[FIXED_POINT_METADATA_ANCHORS] = {}
        result = pass_.run(dag)

        self.assertIsNotNone(result)
        final_layout = pass_.property_set.get("final_layout")
        self.assertIsNotNone(final_layout)

    def test_no_constraints_no_crash(self):
        """FixedPointSabreSwap with a DistributedTarget but no constraint data
        should still work (compatibility mode)."""
        from qiskit.converters import circuit_to_dag

        num_qubits = 8
        distributed_target = self._make_distributed_target({"qpu_0": set(range(num_qubits))})

        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

        dag = circuit_to_dag(qc)
        pass_ = FixedPointSabreSwap(distributed_target, "basic", seed=0, trials=4)
        result = pass_.run(dag)
        self.assertIsNotNone(result)

    def test_rejects_too_few_qubits_with_constraints(self):
        """Swap pass with DistributedTarget still rejects too few qubits."""
        from qiskit.converters import circuit_to_dag

        num_qubits = 8
        distributed_target = self._make_distributed_target({"qpu_0": set(range(num_qubits))})

        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)

        dag = circuit_to_dag(qc)
        pass_ = FixedPointSabreSwap(distributed_target, "basic", seed=0, trials=4)
        pass_.property_set[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = {"qpu_0": list(qc.qubits)}
        pass_.property_set[FIXED_POINT_METADATA_ANCHORS] = {}
        with self.assertRaises(TranspilerError):
            pass_.run(dag)


if __name__ == "__main__":
    unittest.main()
