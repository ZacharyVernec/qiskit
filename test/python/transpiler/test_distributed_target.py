# This code is an original file part of Zachary Vernec's fork of Qiskit.
#
# (C) Copyright Zachary Vernec 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the DistributedTarget class"""

import unittest

from qiskit.circuit.library import XGate
from qiskit.transpiler import (
    Target,
    CouplingMap,
    DistributedTarget,
)
from test import QiskitTestCase


class TestDistributedTarget(QiskitTestCase):
    """Tests for the DistributedTarget class."""

    def setUp(self):
        super().setUp()
        # A 8-qubit linear target.
        self.coupling = CouplingMap.from_line(8)
        self.target = Target.from_configuration(
            basis_gates=["u", "cx"],
            coupling_map=self.coupling,
        )

    def test_basic_construction(self):
        """Test basic construction with two QPUs."""
        qpu_to_qubits = {
            "qpu_0": {0, 1, 2, 3},
            "qpu_1": {4, 5, 6, 7},
        }
        comm_ancillas = {
            "qpu_0": [3],
            "qpu_1": [4],
        }
        comm_ancilla_edges = [(3, 4), (4, 3)]

        dt = DistributedTarget(
            self.target,
            qpu_to_qubits,
            comm_ancillas=comm_ancillas,
            comm_ancilla_edges=comm_ancilla_edges,
        )

        self.assertEqual(dt.num_qubits, 8)
        self.assertEqual(dt.qpu_to_qubits["qpu_0"], {0, 1, 2, 3})
        self.assertEqual(dt.qpu_to_qubits["qpu_1"], {4, 5, 6, 7})
        self.assertEqual(dt.qubits_to_qpu[0], "qpu_0")
        self.assertEqual(dt.qubits_to_qpu[7], "qpu_1")
        self.assertEqual(dt.comm_ancillas["qpu_0"], [3])
        self.assertEqual(dt.comm_ancillas["qpu_1"], [4])
        self.assertEqual(dt.comm_ancilla_edges, [(3, 4), (4, 3)])
        self.assertEqual(dt.qpus, ["qpu_0", "qpu_1"])
        # Gate data should be preserved.
        self.assertIn("u", dt)
        self.assertIn("cx", dt)

    def test_empty_qpu_qubit_set_raises(self):
        """An empty QPU qubit set should raise ValueError."""
        qpu_to_qubits = {
            "qpu_0": set(),
            "qpu_1": {0, 1},
        }
        comm_ancillas = {
            "qpu_0": [],
            "qpu_1": [0],
        }
        with self.assertRaises(ValueError) as ctx:
            DistributedTarget(
                self.target,
                qpu_to_qubits,
                comm_ancillas=comm_ancillas,
            )
        self.assertIn("empty qubit set", str(ctx.exception))

    def test_missing_comm_ancillas_raises(self):
        """Not providing comm_ancillas should raise ValueError."""
        qpu_to_qubits = {"qpu_0": {0, 1, 2, 3}}
        with self.assertRaises(ValueError) as ctx:
            DistributedTarget(self.target, qpu_to_qubits)
        self.assertIn("comm_ancillas", str(ctx.exception))

    def test_three_qpus(self):
        """Test construction with three QPUs."""
        qpu_to_qubits = {
            "qpu_0": {0, 1, 2},
            "qpu_1": {3, 4, 5},
            "qpu_2": {6, 7},
        }
        comm_ancillas = {
            "qpu_0": [2],
            "qpu_1": [3],
            "qpu_2": [7],
        }
        dt = DistributedTarget(
            self.target,
            qpu_to_qubits,
            comm_ancillas=comm_ancillas,
        )
        self.assertEqual(len(dt.qpu_to_qubits), 3)
        self.assertEqual(len(dt.qubits_to_qpu), 8)
        self.assertEqual(dt.qpus, ["qpu_0", "qpu_1", "qpu_2"])

    def test_inverse_consistency(self):
        """qpu_to_qubits and qubits_to_qpu are inverses of each other."""
        qpu_to_qubits = {
            "qpu_0": {0, 1, 2, 3},
            "qpu_1": {4, 5, 6, 7},
        }
        comm_ancillas = {
            "qpu_0": [3],
            "qpu_1": [4],
        }
        dt = DistributedTarget(
            self.target,
            qpu_to_qubits,
            comm_ancillas=comm_ancillas,
        )

        # qubits_to_qpu → qpu_to_qubits
        rebuilt: dict[str, set[int]] = {}
        for q, qpu in dt.qubits_to_qpu.items():
            rebuilt.setdefault(qpu, set()).add(q)
        self.assertEqual(dt.qpu_to_qubits, rebuilt)

        # qpu_to_qubits → qubits_to_qpu
        inverse: dict[int, str] = {}
        for qpu, qubits in dt.qpu_to_qubits.items():
            for q in qubits:
                inverse[q] = qpu
        self.assertEqual(dt.qubits_to_qpu, inverse)

    def test_not_all_qubits_mapped_to_qpu(self):
        """Not all physical qubits need to be mapped to a QPU."""
        # qubits 5, 6, 7 are not assigned to any QPU.
        qpu_to_qubits = {
            "qpu_0": {0, 1, 2, 3},
            "qpu_1": {4},
        }
        comm_ancillas = {
            "qpu_0": [3],
            "qpu_1": [4],
        }
        dt = DistributedTarget(
            self.target,
            qpu_to_qubits,
            comm_ancillas=comm_ancillas,
        )
        self.assertEqual(len(dt.qubits_to_qpu), 5)
        self.assertNotIn(5, dt.qubits_to_qpu)
        self.assertNotIn(6, dt.qubits_to_qpu)
        self.assertNotIn(7, dt.qubits_to_qpu)

    def test_empty_comm_ancilla_list_raises(self):
        """A QPU with an empty communication ancilla list should raise ValueError."""
        qpu_to_qubits = {
            "qpu_0": {0, 1},
            "qpu_1": {2, 3},
        }
        comm_ancillas = {
            "qpu_0": [0],
            "qpu_1": [],  # empty
        }
        with self.assertRaises(ValueError) as ctx:
            DistributedTarget(
                self.target,
                qpu_to_qubits,
                comm_ancillas=comm_ancillas,
            )
        self.assertIn("empty communication ancilla list", str(ctx.exception))

    # ------------------------------------------------------------------
    # Precondition violation tests
    # ------------------------------------------------------------------

    def test_overlapping_qpu_qubits_raises(self):
        """Overlapping qubit sets across QPUs should raise ValueError."""
        qpu_to_qubits = {
            "qpu_0": {0, 1, 2, 3},
            "qpu_1": {3, 4, 5, 6},  # qubit 3 overlaps with qpu_0
        }
        with self.assertRaises(ValueError) as ctx:
            DistributedTarget(self.target, qpu_to_qubits)
        self.assertIn("overlapping", str(ctx.exception))

    def test_qubit_out_of_range_raises(self):
        """Qubits outside the target's range should raise ValueError."""
        qpu_to_qubits = {
            "qpu_0": {0, 1, 2, 8},  # qubit 8 is out of range for an 8-qubit target
        }
        with self.assertRaises(ValueError) as ctx:
            DistributedTarget(self.target, qpu_to_qubits)
        self.assertIn("out of range", str(ctx.exception))

    def test_ancilla_not_in_qpu_raises(self):
        """An ancilla that doesn't belong to its QPU should raise ValueError."""
        qpu_to_qubits = {
            "qpu_0": {0, 1, 2, 3},
            "qpu_1": {4, 5, 6, 7},
        }
        comm_ancillas = {
            "qpu_0": [4],  # qubit 4 belongs to qpu_1
            "qpu_1": [4],
        }
        with self.assertRaises(ValueError) as ctx:
            DistributedTarget(
                self.target,
                qpu_to_qubits,
                comm_ancillas=comm_ancillas,
            )
        self.assertIn("not in that QPU", str(ctx.exception))

    def test_ancilla_for_unknown_qpu_raises(self):
        """An ancilla mapping for an unknown QPU should raise ValueError."""
        qpu_to_qubits = {"qpu_0": {0, 1}}
        comm_ancillas = {
            "qpu_0": [0],
            "qpu_1": [2],  # qpu_1 not in qpu_to_qubits
        }
        with self.assertRaises(ValueError) as ctx:
            DistributedTarget(self.target, qpu_to_qubits, comm_ancillas=comm_ancillas)
        self.assertIn("unknown QPU", str(ctx.exception))

    def test_invalid_comm_ancilla_edge_raises(self):
        """An edge that doesn't exist in the coupling map should raise ValueError."""
        qpu_to_qubits = {
            "qpu_0": {0, 1, 2, 3},
            "qpu_1": {4, 5, 6, 7},
        }
        comm_ancillas = {
            "qpu_0": [3],
            "qpu_1": [4],
        }
        comm_ancilla_edges = [(0, 7)]  # not connected in a line topology
        with self.assertRaises(ValueError) as ctx:
            DistributedTarget(
                self.target,
                qpu_to_qubits,
                comm_ancillas=comm_ancillas,
                comm_ancilla_edges=comm_ancilla_edges,
            )
        self.assertIn("not present", str(ctx.exception))

    def test_target_with_no_num_qubits_raises(self):
        """Target without a finite num_qubits should raise."""
        target_no_qubits = Target(description="no-limit target", num_qubits=None)
        qpu_to_qubits = {"qpu_0": {0, 1}}
        with self.assertRaises(ValueError) as ctx:
            DistributedTarget(target_no_qubits, qpu_to_qubits)
        self.assertIn("finite num_qubits", str(ctx.exception))

    def test_gate_data_is_preserved(self):
        """Verify that all gates from the source target are present in the
        DistributedTarget."""
        qpu_to_qubits = {"qpu_0": {0, 1, 2, 3}, "qpu_1": {4, 5, 6, 7}}
        comm_ancillas = {
            "qpu_0": [3],
            "qpu_1": [4],
        }
        dt = DistributedTarget(
            self.target,
            qpu_to_qubits,
            comm_ancillas=comm_ancillas,
        )
        # Check operation names match.
        self.assertEqual(set(dt.operation_names), set(self.target.operation_names))
        # Check instruction_supported works.
        self.assertTrue(dt.instruction_supported("cx", (0, 1)))
        self.assertFalse(dt.instruction_supported("cx", (0, 7)))  # not connected
        # Check that build_coupling_map returns the same coupling map.
        cm_orig = self.target.build_coupling_map()
        cm_dt = dt.build_coupling_map()
        self.assertEqual(set(cm_orig.get_edges()), set(cm_dt.get_edges()))

    def test_repr(self):
        """Test the repr."""
        qpu_to_qubits = {"qpu_0": {0, 1}}
        comm_ancillas = {"qpu_0": [0]}
        dt = DistributedTarget(
            self.target,
            qpu_to_qubits,
            comm_ancillas=comm_ancillas,
        )
        s = repr(dt)
        self.assertIn("DistributedTarget", s)
        self.assertIn("num_qubits=8", s)
        self.assertIn("qpus=1", s)

    def test_comm_ancilla_edges_with_no_coupling_map_raises(self):
        """If the target has no coupling map, providing comm ancilla edges should raise."""
        # A target with no coupling map (all-to-all, None coupling)
        target_no_cmap = Target(
            description="all-to-all",
            num_qubits=4,
        )
        target_no_cmap.add_instruction(XGate())
        qpu_to_qubits = {"qpu_0": {0, 1}, "qpu_1": {2, 3}}
        comm_ancillas = {
            "qpu_0": [1],
            "qpu_1": [2],
        }
        with self.assertRaises(ValueError) as ctx:
            DistributedTarget(
                target_no_cmap,
                qpu_to_qubits,
                comm_ancillas=comm_ancillas,
                comm_ancilla_edges=[(0, 2)],
            )
        self.assertIn("no coupling-map connectivity", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
