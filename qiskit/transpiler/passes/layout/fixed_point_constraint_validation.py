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

Reads circuit-level constraint data from :attr:`.DAGCircuit.metadata` (which
round-trips from :attr:`.QuantumCircuit.metadata`), validates all Section 3
preconditions from the fixed-point SABRE specification, and writes
canonicalized constraints to the property set for downstream layout and
routing passes.

Metadata keys (set by the user on ``circuit.metadata``)
--------------------------------------------------------

* ``"fixed_point_logical_partitions"`` — ``Dict[str, List[Qubit]]``
  QPU name → list of logical qubits belonging to that QPU.
* ``"fixed_point_anchors"`` — ``Dict[str, Dict[Qubit, int]]``
  QPU name → mapping from logical ``Qubit`` to physical qubit index.

The pass writes the same keys (with validated values) into the property set
for downstream consumption by :class:`.FixedPointSabreLayout`,
:class:`.FixedPointSabreSwap`, :class:`.FixedPointVF2Layout`, and
:class:`.FixedPointVF2PostLayout`.

When no ``fixed_point_logical_partitions`` are present in the DAG metadata
the pass is a no-op (the downstream passes will fall back to standard
SABRE / VF2 behaviour).
"""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.distributed_target import DistributedTarget
from qiskit.transpiler.target import Target

# ---------------------------------------------------------------------------
# Public metadata key constants (importable by users and other passes)
# ---------------------------------------------------------------------------

#: Key in ``circuit.metadata`` and ``dag.metadata`` for logical partitions.
FIXED_POINT_METADATA_LOGICAL_PARTITIONS = "fixed_point_logical_partitions"

#: Key in ``circuit.metadata`` and ``dag.metadata`` for anchor mappings.
FIXED_POINT_METADATA_ANCHORS = "fixed_point_anchors"


class FixedPointConstraintValidation(AnalysisPass):
    """Validate fixed-point SABRE constraints before layout and routing.

    This analysis pass is intended to run in the ``pre_layout`` stage of a
    :class:`.StagedPassManager`.  It reads constraint data from
    :attr:`.DAGCircuit.metadata`, validates every precondition required by
    the fixed-point SABRE specification (Section 3), and writes the
    validated constraints back to the property set under the same keys.

    **Reads (from DAG metadata):**

    * ``dag.metadata["fixed_point_logical_partitions"]`` —
      ``Dict[str, List[Qubit]]``
    * ``dag.metadata["fixed_point_anchors"]`` —
      ``Dict[str, Dict[Qubit, int]]``

    **Writes (to property set, on success):**

    * ``property_set[FIXED_POINT_METADATA_LOGICAL_PARTITIONS]`` — validated copy
    * ``property_set[FIXED_POINT_METADATA_ANCHORS]`` — validated copy

    **Validated preconditions** (see the algorithm specification):

    1. All logical qubits in every :math:`R_i` exist in the DAG.
    2. Logical sets :math:`R_i` are pairwise disjoint across QPUs.
    3. :math:`|R_i| \\le |Q_i|` for every constrained QPU.
    4. Anchor :math:`r_{i,j} \\in R_i` and :math:`q_{i,j} \\in Q_i`.
    5. Anchor cardinality :math:`\\le \\min(|R_i|, |Q_i|)`.
    6. Every QPU named in the constraints exists in the
       :class:`.DistributedTarget`.

    If no ``fixed_point_logical_partitions`` are present in the DAG metadata
    this pass is a no-op.  If constraints are present but the *target* is
    not a :class:`.DistributedTarget` a :class:`.TranspilerError` is raised.
    """

    def __init__(self, target: Target | None = None):
        """
        Args:
            target: The compilation target.  Must be a
                :class:`.DistributedTarget` for constraint validation to
                take effect.  If ``None`` and the DAG metadata contains
                constraints, a :class:`.TranspilerError` is raised at
                run time.
        """
        super().__init__()
        self.target = target

    # ------------------------------------------------------------------
    #  Precondition helper: individual anchor map
    # ------------------------------------------------------------------

    def _validate_anchors(
        self,
        anchors: dict,
        r_qubits: list,
        q_qubits: set,
        qpu: str,
        effective_q_size: int,
    ) -> None:
        """Validate that every anchor pair respects membership and cardinality.

        Args:
            anchors: ``Dict[Qubit, int]`` for a single QPU.
            r_qubits: The logical qubits assigned to that QPU.
            q_qubits: The physical qubit *set* for that QPU.
            qpu: The QPU name (for error messages).
            effective_q_size: ``len(q_qubits)``.

        Raises:
            TranspilerError: If any anchor is invalid.
        """
        for r, p in anchors.items():
            # Precondition 4: anchor membership.
            if r not in r_qubits:
                raise TranspilerError(
                    f"Fixed-point constraint error: anchor logical qubit "
                    f"{r} for QPU '{qpu}' is not in R_{qpu} = "
                    f"{[str(q) for q in r_qubits]}."
                )
            if p not in q_qubits:
                raise TranspilerError(
                    f"Fixed-point constraint error: anchor physical qubit "
                    f"{p} for QPU '{qpu}' is not in Q_{qpu} = "
                    f"{sorted(q_qubits)}."
                )

        # Precondition 5: anchor cardinality.
        max_anchors = min(len(r_qubits), effective_q_size)
        if len(anchors) > max_anchors:
            raise TranspilerError(
                f"Fixed-point constraint error: too many anchors "
                f"({len(anchors)}) for QPU '{qpu}'.  Maximum allowed is "
                f"min(|R_{qpu}|, |Q_{qpu}|) = {max_anchors}."
            )

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------

    def run(self, dag):
        """Run the validation pass on *dag*.

        Args:
            dag: The circuit DAG to validate constraints against.

        Raises:
            TranspilerError: If any precondition is violated.
        """
        # --- Read constraints from DAG metadata --------------------------------
        metadata = dag.metadata or {}
        logical_partitions: dict = metadata.get(FIXED_POINT_METADATA_LOGICAL_PARTITIONS, {})
        anchors_raw: dict = metadata.get(FIXED_POINT_METADATA_ANCHORS, {})

        if not logical_partitions:
            # No constraints — downstream passes operate in compatibility mode.
            return

        # --- Target must be a DistributedTarget when constraints are present ---
        target = self.target
        if target is None:
            target = self.property_set.get("target")
        if not isinstance(target, DistributedTarget):
            raise TranspilerError(
                "Fixed-point constraint error: DAG metadata contains "
                f"'{FIXED_POINT_METADATA_LOGICAL_PARTITIONS}' but the target is not a "
                "DistributedTarget.  Fixed-point SABRE / VF2 requires a "
                f"DistributedTarget when constraints are present."
            )

        dt: DistributedTarget = target
        dag_qubits = set(dag.qubits)
        seen_logical: set = set()  # For pairwise-disjointness check (precond 2).

        # --- Precondition 6: QPU names must be known --------------------------
        for qpu in logical_partitions:
            if qpu not in dt.qpu_to_qubits:
                raise TranspilerError(
                    f"Fixed-point constraint error: QPU '{qpu}' in "
                    f"'{FIXED_POINT_METADATA_LOGICAL_PARTITIONS}' is not a known QPU "
                    f"of the DistributedTarget.  Known QPUs: {dt.qpus}"
                )
        for qpu in anchors_raw:
            if qpu not in dt.qpu_to_qubits:
                raise TranspilerError(
                    f"Fixed-point constraint error: QPU '{qpu}' in "
                    f"'{FIXED_POINT_METADATA_ANCHORS}' is not a known QPU "
                    f"of the DistributedTarget.  Known QPUs: {dt.qpus}"
                )

        # --- Validate each constrained QPU ------------------------------------
        for qpu, r_qubits in logical_partitions.items():
            q_qubits: set = dt.qpu_to_qubits[qpu]
            effective_q_size = len(q_qubits)
            anchors: dict = anchors_raw.get(qpu, {})

            # Precondition 1: all R_i qubits exist in the DAG.
            for q in r_qubits:
                if q not in dag_qubits:
                    raise TranspilerError(
                        f"Fixed-point constraint error: logical qubit {q} "
                        f"in partition R_{qpu} is not a qubit in the DAG.  "
                        f"DAG qubits: {[str(x) for x in dag.qubits]}"
                    )

            # Precondition 2: R_i are pairwise disjoint.
            for q in r_qubits:
                if q in seen_logical:
                    raise TranspilerError(
                        f"Fixed-point constraint error: logical qubit {q} "
                        f"appears in more than one QPU partition.  "
                        f"Partitions must be pairwise disjoint."
                    )
                seen_logical.add(q)

            # Precondition 3: |R_i| <= |Q_i|.
            if len(r_qubits) > effective_q_size:
                raise TranspilerError(
                    f"Fixed-point constraint error: |R_{qpu}| = "
                    f"{len(r_qubits)} > |Q_{qpu}| = {effective_q_size}.  "
                    f"Each QPU can accommodate at most as many logical "
                    f"qubits as it has physical qubits."
                )

            # Preconditions 4 & 5: anchor membership and cardinality.
            if anchors:
                self._validate_anchors(anchors, r_qubits, q_qubits, qpu, effective_q_size)

        # --- Write validated constraints to property set ----------------------
        # Downstream passes (FixedPointSabreLayout, FixedPointSabreSwap,
        # FixedPointVF2Layout, FixedPointVF2PostLayout) read from property_set
        # using these same keys.
        self.property_set[FIXED_POINT_METADATA_LOGICAL_PARTITIONS] = logical_partitions
        self.property_set[FIXED_POINT_METADATA_ANCHORS] = anchors_raw
