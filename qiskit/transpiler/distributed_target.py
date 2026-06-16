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
A target object representing a distributed (multi-QPU) quantum device for
the fixed-point SABRE transpiler passes.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Set, Tuple
import copy

from qiskit.transpiler.target import Target


class DistributedTarget(Target):
    """
    A :class:`.Target` subclass that represents a distributed (multi-QPU) quantum device.

    In addition to the standard :class:`.Target` properties (instruction set, durations,
    coupling map, etc.), a ``DistributedTarget`` holds:

    * A mapping from QPU identifiers to sets of physical qubits, and the inverse mapping.
    * A list of physical qubits designated as *communication ancillas* for each QPU.
    * A list of coupling-map edges that connect communication ancillas (potentially
      across QPUs).

    These extra data structures inform the fixed-point SABRE layout and routing passes
    about how qubits are partitioned across QPUs and which qubits are available for
    inter-QPU communication.

    .. note::

        As with :class:`.Target`, the core data lives in Rust and subclass overrides are
        not visible to Rust-based transpiler passes.  This class is a Python-space
        container for the extra distributed-device metadata, intended for use by
        the fixed-point SABRE passes and custom transpiler stage plugins.

    Validity preconditions (checked on initialisation)
    ---------------------------------------------------

    These follow the disjointness and domain-consistency requirements of the
    fixed-point SABRE specification:

    1. **Pairwise disjointness** — The qubit sets assigned to different QPUs must be
       pairwise disjoint.
    2. **Domain consistency** — Every qubit in every QPU's set must be a valid physical
       qubit index for this target (i.e. in ``range(num_qubits)``).
    3. **Non-empty QPU sets** — Every QPU must have at least one qubit assigned.
    4. **Ancilla subset** — The communication ancillas declared for a QPU must be a
       subset of that QPU's qubits, and every QPU must have at least one
       communication ancilla.  When there is only a single QPU, communication
       ancillas are optional (no inter-QPU communication is needed).
    5. **Edge validity** — Every communication-ancilla edge must be present in the
       coupling map of the target.

    Not all physical qubits in the target need to be mapped to a QPU; unassigned
    qubits are treated as free for unconstrained routing.  Likewise, not all edges
    between communication ancillas need to be listed — only those considered
    viable for inter-QPU communication."""

    __slots__ = (
        "_qpu_to_qubits",
        "_qubits_to_qpu",
        "_comm_ancillas",
        "_comm_ancilla_edges",
    )

    def __new__(
        cls,
        target: Target,
        qpu_to_qubits: Dict[str, Set[int]],
        comm_ancillas: Optional[Dict[str, List[int]]] = None,
        comm_ancilla_edges: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Create a new :class:`DistributedTarget`.

        Args:
            target: An existing :class:`.Target` describing the combined device
                (all QPUs).  Its properties (gate set, durations, coupling map, etc.)
                are shallow-copied into the new ``DistributedTarget``.
            qpu_to_qubits: A mapping from QPU identifiers (e.g. ``"qpu_0"``) to the
                set of physical qubit indices belonging to that QPU.
            comm_ancillas: A mapping from QPU identifiers to the list of
                physical qubit indices that are designated as communication ancillas
                for that QPU.  Required when there is more than one QPU; optional
                (and unused) for a single QPU.
            comm_ancilla_edges: An optional list of ``(u, v)`` edges in the coupling
                map that connect two communication ancillas.  Defaults to an empty
                list.

        Raises:
            ValueError: If any validity precondition is violated.
        """
        if target.num_qubits is None:
            raise ValueError("'target' must have a finite num_qubits")

        # 1. Pairwise disjointness — QPU qubit sets must not overlap.
        all_qpu_qubits: Dict[str, Set[int]] = {}
        for qpu, qubits in qpu_to_qubits.items():
            qubit_set = set(qubits)
            all_qpu_qubits[qpu] = qubit_set
            for other_qpu, other_qubits in all_qpu_qubits.items():
                if other_qpu == qpu:
                    continue
                overlap = qubit_set & other_qubits
                if overlap:
                    raise ValueError(
                        f"QPUs '{qpu}' and '{other_qpu}' have overlapping qubit sets: "
                        f"{sorted(overlap)}"
                    )

        # 2. Non-empty QPU sets — every QPU must have at least one qubit.
        for qpu, qubits in all_qpu_qubits.items():
            if not qubits:
                raise ValueError(f"QPU '{qpu}' has an empty qubit set.")

        # 3. Domain consistency — every qubit must be in range.
        num_qubits = target.num_qubits
        for qpu, qubits in all_qpu_qubits.items():
            for q in qubits:
                if q < 0 or q >= num_qubits:
                    raise ValueError(
                        f"Qubit {q} in QPU '{qpu}' is out of range "
                        f"[0, {num_qubits}) for this target."
                    )

        # Build inverse map.
        qubits_to_qpu: Dict[int, str] = {}
        for qpu, qubits in all_qpu_qubits.items():
            for q in qubits:
                qubits_to_qpu[q] = qpu

        # 4. Ancilla subset and non-empty — ancillas must belong to their QPU's
        # qubit set; every QPU must have at least one communication ancilla
        # unless there is only a single QPU (no inter-QPU communication needed).
        n_qpus = len(all_qpu_qubits)
        if comm_ancillas is None:
            if n_qpus > 1:
                raise ValueError(
                    "'comm_ancillas' argument is required when there is more than "
                    "one QPU."
                )
            comm_ancillas = {}
        for qpu in comm_ancillas:
            if qpu not in all_qpu_qubits:
                raise ValueError(
                    f"Communication ancillas specified for unknown QPU '{qpu}'."
                )
        if n_qpus > 1:
            for qpu in all_qpu_qubits:
                if qpu not in comm_ancillas:
                    raise ValueError(
                        f"QPU '{qpu}' has no entry in 'comm_ancillas'; "
                        "every QPU must have at least one communication ancilla."
                    )
                ancillas = comm_ancillas[qpu]
                if not ancillas:
                    raise ValueError(
                        f"QPU '{qpu}' has an empty communication ancilla list; "
                        "every QPU must have at least one."
                    )
                qpu_qubits = all_qpu_qubits[qpu]
                for anc in ancillas:
                    if anc not in qpu_qubits:
                        raise ValueError(
                            f"Communication ancilla {anc} for QPU '{qpu}' is not in "
                            f"that QPU's qubit set {sorted(qpu_qubits)}."
                        )
        else:
            # Single QPU: if ancillas were provided, validate subset membership.
            for qpu, ancillas in comm_ancillas.items():
                qpu_qubits = all_qpu_qubits[qpu]
                for anc in ancillas:
                    if anc not in qpu_qubits:
                        raise ValueError(
                            f"Communication ancilla {anc} for QPU '{qpu}' is not in "
                            f"that QPU's qubit set {sorted(qpu_qubits)}."
                        )

        # 5. Edge validity — every comm-ancilla edge must exist in the coupling map.
        if comm_ancilla_edges is None:
            comm_ancilla_edges = []
        coupling_map = target.build_coupling_map()
        if coupling_map is None:
            if comm_ancilla_edges:
                raise ValueError(
                    "Communication ancilla edges were specified, but the target has "
                    "no coupling-map connectivity constraints."
                )
        else:
            for u, v in comm_ancilla_edges:
                if not coupling_map.graph.has_edge(u, v):
                    raise ValueError(
                        f"Communication ancilla edge ({u}, {v}) is not present in "
                        "the target's coupling map."
                    )

        # Set up the Rust-side Target data via super().__new__.
        # We give the Rust target the same shape as the input target.
        out = super(DistributedTarget, cls).__new__(
            cls,
            description=target.description,
            num_qubits=target.num_qubits,
            dt=target.dt,
            granularity=target.granularity,
            min_length=target.min_length,
            pulse_alignment=target.pulse_alignment,
            acquire_alignment=target.acquire_alignment,
            qubit_properties=copy.copy(target.qubit_properties),
            concurrent_measurements=(
                copy.deepcopy(target.concurrent_measurements)
                if target.concurrent_measurements is not None
                else None
            ),
        )
        return out

    def __init__(
        self,
        target: Target,
        qpu_to_qubits: Dict[str, Set[int]],
        comm_ancillas: Optional[Dict[str, List[int]]] = None,
        comm_ancilla_edges: Optional[List[Tuple[int, int]]] = None,
    ):
        """Initialise the Python-side distributed metadata and copy gate data from ``target``."""
        super().__init__()
        # Copy the gate map from the source target.  `add_instruction` mirrors
        # data to both the Rust and Python sides correctly.
        for gate_name, qarg_props in target._gate_map.items():
            instruction = target._gate_name_map[gate_name]
            self.add_instruction(instruction, copy.deepcopy(qarg_props), name=gate_name)

        # Store the distributed-specific metadata.
        self._qpu_to_qubits: Dict[str, Set[int]] = {
            qpu: set(qubits) for qpu, qubits in qpu_to_qubits.items()
        }
        self._qubits_to_qpu: Dict[int, str] = {
            q: qpu for qpu, qubits in qpu_to_qubits.items() for q in qubits
        }
        self._comm_ancillas: Dict[str, List[int]] = (
            dict(comm_ancillas) if comm_ancillas else {}
        )
        self._comm_ancilla_edges: List[Tuple[int, int]] = (
            list(comm_ancilla_edges) if comm_ancilla_edges else []
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def qpu_to_qubits(self) -> Dict[str, Set[int]]:
        """Mapping from QPU identifier to the set of physical qubits on that QPU."""
        return self._qpu_to_qubits

    @property
    def qubits_to_qpu(self) -> Dict[int, str]:
        """Inverse mapping from physical qubit index to its QPU identifier."""
        return self._qubits_to_qpu

    @property
    def comm_ancillas(self) -> Dict[str, List[int]]:
        """Mapping from QPU identifier to its list of communication ancilla qubits."""
        return self._comm_ancillas

    @property
    def comm_ancilla_edges(self) -> List[Tuple[int, int]]:
        """List of coupling-map edges that connect two communication ancillas."""
        return self._comm_ancilla_edges

    @property
    def qpus(self) -> List[str]:
        """Sorted list of QPU identifiers."""
        return sorted(self._qpu_to_qubits.keys())

    def __repr__(self):
        return (
            f"DistributedTarget(num_qubits={self.num_qubits}, "
            f"qpus={len(self._qpu_to_qubits)}, "
            f"description={self.description!r})"
        )



