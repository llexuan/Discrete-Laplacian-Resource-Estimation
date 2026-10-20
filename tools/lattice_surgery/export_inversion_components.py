"""Export the exact logical components used by the full inverse estimate."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


def export_components(metrics_path: Path, phases_path: Path, output: Path) -> dict:
    """Match saved gate counts before exporting W and each distinct projector."""
    from qiskit import QuantumCircuit, qasm2, transpile
    from laplacian_qsvt.laplacian.laplacian_inversion_encode import (
        build_projector_rotation_clifford_t,
        num_k_qubits,
    )
    from laplacian_qsvt.laplacian.laplacian_nd_resource import CT_BASIS
    from laplacian_qsvt.laplacian.laplacian_w_resource import build_w_operator
    from run_layout_comparison import normalize_qasm

    metrics = json.loads(metrics_path.read_text())
    phases_metadata = json.loads(phases_path.read_text())
    logical = metrics["logical_resources"]
    phases = [float(phi) for phi in phases_metadata["phases"]]
    unique = logical["projector_rotation_contribution"]["unique_rotation_resources"]
    actual = Counter(repr(phi) for phi in phases)
    expected = {repr(float(item["phi"])): item["occurrences"] for item in unique}
    if actual != expected or len(phases) != logical["num_projector_rotations"]:
        raise ValueError("Phase input does not match the saved logical estimate")
    dims, n = logical["dimensions"], logical["target_n"]
    m = 2 + num_k_qubits(dims)
    qubits = logical["logical_qubits"]
    walk_qubits = 1 + m + dims * n
    if qubits != walk_qubits + 1:
        raise ValueError("Expected one clean phase flag in addition to W registers")
    output.mkdir(parents=True, exist_ok=True)
    snapshot_logical_inputs(metrics_path, phases_path, output)
    inputs = output / "inputs" / "components"
    inputs.mkdir(parents=True, exist_ok=True)
    components = []

    def emit(identifier, circuit, mapping, occurrences, reference, **metadata):
        compiled = transpile(
            circuit, basis_gates=CT_BASIS, optimization_level=0, seed_transpiler=0
        )
        counts = dict(compiled.count_ops())
        if counts != reference["gate_counts"]:
            raise ValueError(
                f"{identifier} gate counts differ from the logical report; regenerate that report with this environment first"
            )
        if compiled.num_qubits != len(mapping):
            raise ValueError(f"{identifier} decomposition changed its workspace")
        padded = QuantumCircuit(qubits)
        padded.compose(compiled, qubits=mapping, inplace=True)
        qasm, size = normalize_qasm(qasm2.dumps(padded))
        if size != qubits:
            raise ValueError("Component lost resident logical qubits")
        path = inputs / f"{identifier}.qasm"
        path.write_text(qasm)
        components.append(
            {
                "id": identifier,
                "occurrences": occurrences,
                "qasm": str(path.resolve()),
                "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
                "logical_qubits": qubits,
                "gate_counts": counts,
                "t_count": counts.get("t", 0) + counts.get("tdg", 0),
                "dropped_global_phase": float(padded.global_phase),
                **metadata,
            }
        )
        print(
            f"Exported {identifier}: {sum(counts.values()):,} gates, {occurrences} occurrences",
            flush=True,
        )

    eps = logical["eps_per_rotation"]
    selector_eps = (
        logical["walk_contribution"]["per_w"]["rotation_synthesis"]["eps_per_rotation"]
        if dims == 3 else None
    )
    walk = build_w_operator(n, dims, synthesis_eps=selector_eps)
    emit(
        "W",
        walk,
        list(range(walk_qubits)),
        logical["num_w_applications"],
        logical["walk_contribution"]["per_w"],
        kind="walk",
    )
    mapping = list(range(1 + m)) + [walk_qubits] + list(range(1 + m, 2 * m))
    for index, item in enumerate(unique):
        circuit, synthesis = build_projector_rotation_clifford_t(item["phi"], m, eps)
        if abs(synthesis.projective_error - item["projective_error"]) > 1e-12:
            raise ValueError(
                "Projector synthesis differs from the saved logical estimate"
            )
        emit(
            f"R_{index:03d}",
            circuit,
            mapping,
            item["occurrences"],
            item,
            kind="projector",
            phi=item["phi"],
        )
    totals = Counter()
    for component in components:
        for gate, count in component["gate_counts"].items():
            totals[gate] += count * component["occurrences"]
    if dict(totals) != logical["gate_counts"]:
        raise ValueError(
            "Exported component totals do not match the full logical estimate"
        )
    manifest = {
        "schema_version": 1,
        "scope": "full_inverse_component_architecture_inputs",
        "logical_metrics_source": str(metrics_path.resolve()),
        "logical_metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "phases_source": str(phases_path.resolve()),
        "phases_sha256": hashlib.sha256(phases_path.read_bytes()).hexdigest(),
        "dimensions": dims,
        "target_n": n,
        "logical_qubits": qubits,
        "num_w_applications": logical["num_w_applications"],
        "num_projector_rotations": logical["num_projector_rotations"],
        "logical_gate_counts": dict(totals),
        "algorithm": {
            key: metrics.get(key)
            for key in ("target_function", "kappa", "epsilon", "error_summary")
        },
        "minimum_kappa_for_full_nonzero_spectrum": logical[
            "minimum_kappa_for_full_nonzero_spectrum"
        ],
        "components": components,
        "registers": {"phase_flag": walk_qubits, "projector_local_to_global": mapping},
        "accounting": "One W and each unique phase rotation, weighted by occurrences; all components retain the full resident register.",
    }
    (output / "inputs" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def snapshot_logical_inputs(metrics_path: Path, phases_path: Path, output: Path) -> None:
    """Keep the exact source bytes so hashes identify the case's saved baseline."""
    from laplacian_qsvt.laplacian.laplacian_inversion_encode import render_resource_report

    logical_dir = output / "logical"
    logical_dir.mkdir(parents=True, exist_ok=True)
    (output / "inputs").mkdir(parents=True, exist_ok=True)
    (logical_dir / "inversion_resource_metrics.json").write_bytes(metrics_path.read_bytes())
    (output / "inputs" / "phases.json").write_bytes(phases_path.read_bytes())
    metrics = json.loads(metrics_path.read_text())
    if "verification" in metrics and "error_summary" in metrics:
        report = render_resource_report(metrics)
    else:
        logical = metrics["logical_resources"]
        report = (
            "Full inverse sequence: logical resource baseline\n"
            f"Dimensions / n per axis: {logical['dimensions']} / {logical['target_n']}\n"
            f"Logical qubits: {logical['logical_qubits']:,}\n"
            f"W applications: {logical['num_w_applications']:,}\n"
            f"Projector rotations: {logical['num_projector_rotations']:,}\n"
            f"Clifford gates: {logical['clifford_gate_count']:,}\n"
            f"T/T-dagger gates: {logical['t_count']:,}\n"
            "The supplied baseline does not include dense verification metadata.\n"
        )
    (logical_dir / "inversion_resource_report.txt").write_text(report)
