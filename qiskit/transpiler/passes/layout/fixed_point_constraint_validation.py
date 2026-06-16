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

"""
Pre-layout validation of fixed-point SABRE constraints.

Writes the validated/normalised constraint data to the property set
under ``fixed_point_normalized`` (a bool) on success.
"""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.distributed_target import DistributedTarget


class FixedPointConstraintValidation(AnalysisPass):
    """Validate fixed-point SABRE constraints before layout/routing.

    Reads:
      * ``target`` — must be a :class:`~DistributedTarget` if constraints are present.
      * ``fixed_point_logical_partitions`` — ``Dict[str, List[Qubit]]`` from the property set.
      * ``fixed_point_anchors`` — ``Dict[str, Dict[Qubit, int]]`` from the property set.

    Validates Section 3 preconditions from the fixed-point SABRE specification:

    1. All :math:`R_i` qubits exist in the DAG.
    2. Logical sets are pairwise disjoint across QPUs.
    3. :math:`|R_i| \\le |Q_i|` (full :math:`Q_i`, including comm ancillas).
    4. Anchor :math:`r_{i,j} \\in R_i` and :math:`q_{i,j} \\in Q_i`.
    5. Anchor cardinality :math:`\\le \\min(|R_i|, |Q_i|)`.
    6. QPU names in the property set match :class:`DistributedTarget` QPUs (or are a subset).

    Writes ``fixed_point_normalized`` (``True``) to the property set on success.
    Raises :class:`~TranspilerError` on failure.

    This pass is a no-op (writes ``fixed_point_normalized=True``) when the target
    is **not** a :class:`~DistributedTarget`.
    """

    def __init__(self, coupling_map):
        """
        Args:
            coupling_map (Union[DistributedTarget, Target, CouplingMap]): The
                target or coupling map.  Must be a :class:`~DistributedTarget`
                for constraint validation to have effect.
        """
        super().__init__()
        if isinstance(coupling_map, DistributedTarget):
            self.distributed_target = coupling_map
        else:
            self.distributed_target = None

    def run(self, dag):
        """Run the validation pass on *dag*.

        Args:
            dag (DAGCircuit): The circuit DAG to validate constraints against.

        Raises:
            TranspilerError: If any precondition is violated.
        """
        if self.distributed_target is None:
            # No DistributedTarget → no constraints to validate.
            self.property_set["fixed_point_normalized"] = True
            return

        dt = self.distributed_target
        logical_partitions = self.property_set.get("fixed_point_logical_partitions", {})
        anchors_raw = self.property_set.get("fixed_point_anchors", {})

        dag_qubits = set(dag.qubits)
        dag_index = {q: i for i, q in enumerate(dag.qubits)}

        # --- Precondition 6: QPU names must be known to the DistributedTarget. ---
        for qpu in logical_partitions:
            if qpu not in dt.qpu_to_qubits:
                raise TranspilerError(
                    f"Fixed-point constraint error: QPU '{qpu}' in "
                    "fixed_point_logical_partitions is not a known QPU of the "
                    f"DistributedTarget. Known QPUs: {dt.qpus}"
                )

        for qpu in anchors_raw:
            if qpu not in dt.qpu_to_qubits:
                raise TranspilerError(
                    f"Fixed-point constraint error: QPU '{qpu}' in "
                    "fixed_point_anchors is not a known QPU of the "
                    f"DistributedTarget. Known QPUs: {dt.qpus}"
                )

        # Track all logical qubits across groups for disjointness check.
        seen_logical: set = set()

        for qpu in dt.qpus:
            r_qubits = logical_partitions.get(qpu, [])
            q_qubits = dt.qpu_to_qubits[qpu]
            anchors = anchors_raw.get(qpu, {})

            # --- Precondition 1: R_i qubits exist in the DAG. ---
            for q in r_qubits:
                if q not in dag_qubits:
                    raise TranspilerError(
                        f"Fixed-point constraint error: logical qubit {q} "
                        f"in partition for QPU '{qpu}' is not a qubit in the DAG. "
                        f"DAG qubits: {sorted(dag_qubits, key=lambda x: dag_index[x])}"
                    )

            # --- Precondition 2: R_i are pairwise disjoint. ---
            for q in r_qubits:
                if q in seen_logical:
                    raise TranspilerError(
                        f"Fixed-point constraint error: logical qubit {q} "
                        f"appears in more than one QPU partition (already seen "
                        f"in another group). Partitions must be pairwise disjoint."
                    )
                seen_logical.add(q)

            # Compute Q_i^eff = full Q_i (including comm ancillas, per user's spec).
            # Capacity check uses full |Q_i|.
            effective_q_size = len(q_qubits)

            # --- Precondition 3: |R_i| <= |Q_i|. ---
            if len(r_qubits) > effective_q_size:
                raise TranspilerError(
                    f"Fixed-point constraint error: |R_{qpu}| = {len(r_qubits)} > "
                    f"|Q_{qpu}| = {effective_q_size}. "
                    f"Each QPU can accommodate at most as many logical qubits "
                    f"as it has physical qubits."
                )

            # --- Precondition 4: Anchor membership. ---
            for r, p in anchors.items():
                if r not in r_qubits:
                    raise TranspilerError(
                        f"Fixed-point constraint error: anchor logical qubit {r} "
                        f"for QPU '{qpu}' is not in R_{qpu} = {r_qubits}."
                    )
                if p not in q_qubits:
                    raise TranspilerError(
                        f"Fixed-point constraint error: anchor physical qubit {p} "
                        f"for QPU '{qpu}' is not in Q_{qpu} = {sorted(q_qubits)}."
                    )

            # --- Precondition 5: Anchor cardinality. ---
            if len(anchors) > min(len(r_qubits), effective_q_size):
                raise TranspilerError(
                    f"Fixed-point constraint error: too many anchors ({len(anchors)}) "
                    f"for QPU '{qpu}'. Maximum allowed is "
                    f"min(|R_{qpu}|, |Q_{qpu}|) = {min(len(r_qubits), effective_q_size)}."
                )

        # --- Precondition 7 (implicit): Comm ancillas belong to Q_i. ---
        # (Already validated by DistributedTarget on construction.)

        self.property_set["fixed_point_normalized"] = True
