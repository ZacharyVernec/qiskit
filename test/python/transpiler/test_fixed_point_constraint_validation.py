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

"""Test the FixedPointConstraintValidation pass"""

import unittest

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import Target, CouplingMap, DistributedTarget
from qiskit.transpiler.passes.layout.fixed_point_constraint_validation import (
    FixedPointConstraintValidation,
)
from qiskit.transpiler.exceptions import TranspilerError
from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestFixedPointConstraintValidation(QiskitTestCase):
    """Tests for the FixedPointConstraintValidation analysis pass."""

    def setUp(self):
        super().setUp()
        # A 8-qubit linear target for the base Target.
        self.coupling = CouplingMap.from_line(8)
        self.base_target = Target.from_configuration(
            basis_gates=["u", "cx"],
            coupling_map=self.coupling,
        )
        # Two QPUs of 4 qubits each, fully covering the 8-qubit line.
        self.qpu_to_qubits = {
            "qpu_0": {0, 1, 2, 3},
            "qpu_1": {4, 5, 6, 7},
        }
        self.comm_ancillas = {
            "qpu_0": [3],
            "qpu_1": [4],
        }
        self.comm_ancilla_edges = [(3, 4), (4, 3)]

    def _make_dt(self, qpu_to_qubits=None, comm_ancillas=None, comm_ancilla_edges=None):
        """Make a DistributedTarget with defaults."""
        return DistributedTarget(
            self.base_target,
            qpu_to_qubits or self.qpu_to_qubits,
            comm_ancillas=comm_ancillas or self.comm_ancillas,
            comm_ancilla_edges=comm_ancilla_edges or self.comm_ancilla_edges,
        )

    @staticmethod
    def _run(pass_, dag, **property_set_kwargs):
        """Helper to run the validation pass with property-set inputs."""
        for key, value in property_set_kwargs.items():
            pass_.property_set[key] = value
        pass_(dag)
        return pass_

    # ------------------------------------------------------------------
    # No-op when target is NOT a DistributedTarget
    # ------------------------------------------------------------------

    def test_no_op_with_plain_target(self):
        """When the target is a plain Target (not DistributedTarget), the pass
        is a no-op and writes fixed_point_normalized=True."""
        qc = QuantumCircuit(QuantumRegister(4, "q"))
        pass_ = self._run(FixedPointConstraintValidation(self.base_target), qc)
        self.assertTrue(pass_.property_set["fixed_point_normalized"])

    def test_no_op_with_coupling_map(self):
        """When the target is a CouplingMap, the pass is a no-op."""
        qc = QuantumCircuit(QuantumRegister(4, "q"))
        pass_ = self._run(FixedPointConstraintValidation(self.coupling), qc)
        self.assertTrue(pass_.property_set["fixed_point_normalized"])

    # ------------------------------------------------------------------
    # Success cases — valid constraints
    # ------------------------------------------------------------------

    def test_valid_two_qpu_no_anchors(self):
        """Valid constraints with two QPUs and no anchors should succeed."""
        dt = self._make_dt()
        qr = QuantumRegister(6, "q")
        qc = QuantumCircuit(qr)
        pass_ = self._run(
            FixedPointConstraintValidation(dt),
            qc,
            fixed_point_logical_partitions={
                "qpu_0": [qr[0], qr[1], qr[2]],
                "qpu_1": [qr[3], qr[4], qr[5]],
            },
            fixed_point_anchors={},
        )
        self.assertTrue(pass_.property_set["fixed_point_normalized"])

    def test_valid_two_qpu_with_anchors(self):
        """Valid constraints with two QPUs and anchors should succeed."""
        dt = self._make_dt()
        qr = QuantumRegister(4, "q")
        qc = QuantumCircuit(qr)
        pass_ = self._run(
            FixedPointConstraintValidation(dt),
            qc,
            fixed_point_logical_partitions={
                "qpu_0": [qr[0], qr[1]],
                "qpu_1": [qr[2], qr[3]],
            },
            fixed_point_anchors={
                "qpu_0": {qr[0]: 0},
                "qpu_1": {qr[2]: 5},
            },
        )
        self.assertTrue(pass_.property_set["fixed_point_normalized"])

    def test_valid_subset_of_qpus(self):
        """Constraints that only use a subset of DistributedTarget QPUs are valid."""
        dt = self._make_dt()
        qr = QuantumRegister(3, "q")
        qc = QuantumCircuit(qr)
        pass_ = self._run(
            FixedPointConstraintValidation(dt),
            qc,
            fixed_point_logical_partitions={
                "qpu_0": [qr[0], qr[1], qr[2]],
            },
            fixed_point_anchors={},
        )
        self.assertTrue(pass_.property_set["fixed_point_normalized"])

    def test_valid_full_logical_set_single_qpu(self):
        """All logical qubits assigned to a single QPU is valid."""
        dt = self._make_dt()
        qr = QuantumRegister(4, "q")
        qc = QuantumCircuit(qr)
        pass_ = self._run(
            FixedPointConstraintValidation(dt),
            qc,
            fixed_point_logical_partitions={
                "qpu_0": [qr[0], qr[1], qr[2], qr[3]],
            },
            fixed_point_anchors={},
        )
        self.assertTrue(pass_.property_set["fixed_point_normalized"])

    # ------------------------------------------------------------------
    # Precondition 6: Unknown QPU names
    # ------------------------------------------------------------------

    def test_unknown_qpu_in_logical_partitions(self):
        """A QPU name in logical_partitions not known to DistributedTarget raises."""
        dt = self._make_dt()
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        with self.assertRaises(TranspilerError) as ctx:
            self._run(
                FixedPointConstraintValidation(dt),
                qc,
                fixed_point_logical_partitions={
                    "qpu_unknown": [qr[0], qr[1]],
                },
                fixed_point_anchors={},
            )
        self.assertIn("qpu_unknown", str(ctx.exception))
        self.assertIn("not a known QPU", str(ctx.exception))

    def test_unknown_qpu_in_anchors(self):
        """A QPU name in anchors not known to DistributedTarget raises."""
        dt = self._make_dt()
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        with self.assertRaises(TranspilerError) as ctx:
            self._run(
                FixedPointConstraintValidation(dt),
                qc,
                fixed_point_logical_partitions={
                    "qpu_0": [qr[0], qr[1]],
                },
                fixed_point_anchors={
                    "qpu_unknown": {qr[0]: 0},
                },
            )
        self.assertIn("qpu_unknown", str(ctx.exception))
        self.assertIn("not a known QPU", str(ctx.exception))

    # ------------------------------------------------------------------
    # Precondition 1: R_i qubits must exist in the DAG
    # ------------------------------------------------------------------

    def test_logical_qubit_not_in_dag(self):
        """A qubit in R_i that is not in the DAG raises."""
        dt = self._make_dt()
        qr = QuantumRegister(2, "q")
        qr_other = QuantumRegister(1, "r")
        qc = QuantumCircuit(qr)
        with self.assertRaises(TranspilerError) as ctx:
            self._run(
                FixedPointConstraintValidation(dt),
                qc,
                fixed_point_logical_partitions={
                    "qpu_0": [qr[0], qr[1], qr_other[0]],
                },
                fixed_point_anchors={},
            )
        self.assertIn("not a qubit in the DAG", str(ctx.exception))

    # ------------------------------------------------------------------
    # Precondition 2: R_i must be pairwise disjoint
    # ------------------------------------------------------------------

    def test_logical_qubit_in_multiple_partitions(self):
        """A qubit appearing in more than one R_i raises."""
        dt = self._make_dt()
        qr = QuantumRegister(4, "q")
        qc = QuantumCircuit(qr)
        with self.assertRaises(TranspilerError) as ctx:
            self._run(
                FixedPointConstraintValidation(dt),
                qc,
                fixed_point_logical_partitions={
                    "qpu_0": [qr[0], qr[1], qr[2]],
                    "qpu_1": [qr[2], qr[3]],  # qr[2] overlaps
                },
                fixed_point_anchors={},
            )
        self.assertIn("appears in more than one QPU partition", str(ctx.exception))

    # ------------------------------------------------------------------
    # Precondition 3: |R_i| <= |Q_i|
    # ------------------------------------------------------------------

    def test_too_many_logical_qubits_for_qpu(self):
        """|R_i| > |Q_i| should raise."""
        dt = DistributedTarget(
            self.base_target,
            {"qpu_0": {0, 1}},
            comm_ancillas={"qpu_0": [0]},
        )
        qr = QuantumRegister(3, "q")
        qc = QuantumCircuit(qr)
        with self.assertRaises(TranspilerError) as ctx:
            self._run(
                FixedPointConstraintValidation(dt),
                qc,
                fixed_point_logical_partitions={
                    "qpu_0": [qr[0], qr[1], qr[2]],  # 3 logical > 2 physical
                },
                fixed_point_anchors={},
            )
        self.assertIn(">", str(ctx.exception))

    # ------------------------------------------------------------------
    # Precondition 4: Anchor membership
    # ------------------------------------------------------------------

    def test_anchor_logical_not_in_r(self):
        """An anchor logical qubit not in R_i raises."""
        dt = self._make_dt()
        qr = QuantumRegister(3, "q")
        qc = QuantumCircuit(qr)
        with self.assertRaises(TranspilerError) as ctx:
            self._run(
                FixedPointConstraintValidation(dt),
                qc,
                fixed_point_logical_partitions={
                    "qpu_0": [qr[0], qr[1]],
                },
                fixed_point_anchors={
                    "qpu_0": {qr[2]: 0},  # qr[2] not in R_qpu_0
                },
            )
        self.assertIn("not in R_", str(ctx.exception))

    def test_anchor_physical_not_in_q(self):
        """An anchor physical qubit not in Q_i raises."""
        dt = self._make_dt()
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        with self.assertRaises(TranspilerError) as ctx:
            self._run(
                FixedPointConstraintValidation(dt),
                qc,
                fixed_point_logical_partitions={
                    "qpu_0": [qr[0], qr[1]],
                },
                fixed_point_anchors={
                    "qpu_0": {qr[0]: 7},  # 7 is in qpu_1, not qpu_0
                },
            )
        self.assertIn("not in Q_", str(ctx.exception))

    # ------------------------------------------------------------------
    # Precondition 5: Anchor cardinality
    # ------------------------------------------------------------------

    def test_anchors_at_max_cardinality(self):
        """Anchors at max cardinality min(|R_i|, |Q_i|) is valid."""
        dt = DistributedTarget(
            self.base_target,
            {"qpu_0": {0, 1}},
            comm_ancillas={"qpu_0": [0]},
        )
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        pass_ = self._run(
            FixedPointConstraintValidation(dt),
            qc,
            fixed_point_logical_partitions={
                "qpu_0": [qr[0], qr[1]],
            },
            fixed_point_anchors={
                "qpu_0": {qr[0]: 0, qr[1]: 1},
            },
        )
        self.assertTrue(pass_.property_set["fixed_point_normalized"])

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_logical_partitions_is_valid(self):
        """Empty logical_partitions with a DistributedTarget is valid."""
        dt = self._make_dt()
        qr = QuantumRegister(4, "q")
        qc = QuantumCircuit(qr)
        pass_ = self._run(
            FixedPointConstraintValidation(dt),
            qc,
            fixed_point_logical_partitions={},
            fixed_point_anchors={},
        )
        self.assertTrue(pass_.property_set["fixed_point_normalized"])

    def test_three_qpu_capacity_violation(self):
        """|R_i| > |Q_i| raises with three QPUs."""
        dt = DistributedTarget(
            self.base_target,
            {
                "qpu_0": {0, 1, 2},
                "qpu_1": {3, 4, 5, 6},
                "qpu_2": {7},
            },
            comm_ancillas={
                "qpu_0": [2],
                "qpu_1": [3],
                "qpu_2": [7],
            },
            comm_ancilla_edges=[(2, 3), (3, 2)],
        )
        qr = QuantumRegister(7, "q")
        qc = QuantumCircuit(qr)
        with self.assertRaises(TranspilerError) as ctx:
            self._run(
                FixedPointConstraintValidation(dt),
                qc,
                fixed_point_logical_partitions={
                    "qpu_0": [qr[0], qr[1]],
                    "qpu_1": [qr[2], qr[3], qr[4]],
                    "qpu_2": [qr[5], qr[6]],  # 2 logical > 1 physical
                },
                fixed_point_anchors={},
            )
        self.assertIn(">", str(ctx.exception))

    def test_no_anchors_property_set(self):
        """When fixed_point_anchors is not in the property set, it defaults to empty."""
        dt = self._make_dt()
        qr = QuantumRegister(4, "q")
        qc = QuantumCircuit(qr)
        pass_ = self._run(
            FixedPointConstraintValidation(dt),
            qc,
            fixed_point_logical_partitions={
                "qpu_0": [qr[0], qr[1]],
                "qpu_1": [qr[2], qr[3]],
            },
        )
        self.assertTrue(pass_.property_set["fixed_point_normalized"])

    def test_no_logical_partitions_property_set(self):
        """When fixed_point_logical_partitions is not set, it defaults to empty."""
        dt = self._make_dt()
        qr = QuantumRegister(4, "q")
        qc = QuantumCircuit(qr)
        pass_ = self._run(FixedPointConstraintValidation(dt), qc)
        self.assertTrue(pass_.property_set["fixed_point_normalized"])

    # ------------------------------------------------------------------
    # Trivial monolithic case: single QPU covering all qubits, no anchors
    # (Section 7 compatibility mode)
    # ------------------------------------------------------------------

    def test_trivial_monolithic_qpu_no_anchors(self):
        """Single QPU covering all physical qubits with all logical qubits,
        no anchors. This is the compatibility mode from Section 7 of the spec."""
        dt = DistributedTarget(
            self.base_target,
            {"qpu_0": set(range(8))},
        )
        qr = QuantumRegister(8, "q")
        qc = QuantumCircuit(qr)
        pass_ = self._run(
            FixedPointConstraintValidation(dt),
            qc,
            fixed_point_logical_partitions={
                "qpu_0": list(qr),
            },
            fixed_point_anchors={},
        )
        self.assertTrue(pass_.property_set["fixed_point_normalized"])


if __name__ == "__main__":
    unittest.main()
