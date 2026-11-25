"""Export the exact logical components used by the full inverse estimate."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


def export_components(metrics_path: Path, phases_path: Path, output: Path) -> dict:
    """Match saved gate counts before exporting W and each distinct projector."""
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("operator_family") == "preconditioned_1d":
        return export_compiled_components(metrics_path, phases_path, output, metrics)
    from qiskit import QuantumCircuit, qasm2, transpile
    from laplacian_qsvt.laplacian.laplacian_inversion_encode import (
        build_projector_rotation_clifford_t,
        num_k_qubits,
    )
    from laplacian_qsvt.laplacian.laplacian_nd_resource import CT_BASIS
    from laplacian_qsvt.laplacian.laplacian_w_resource import build_w_operator
    from run_layout_comparison import normalize_qasm

    phases_metadata = json.loads(phases_path.read_text())
    logical = metrics["logical_resources"]
    phases = [float(phi) for phi in phases_metadata["phases"]]
    unique = logical["projector_rotation_contribution"]["unique_rotation_resources"]
    actual = Counter(repr(phi) for phi in phases)
    expected = {repr(float(item["phi"])): item["occurrences"] for item in unique}
    if actual != expected or len(phases) != logical["num_projector_rotations"]:
        raise ValueError("Phase input does not match the saved logical estimate")
    dims, n = logical["dimensions"], logical["target_n"]
    preconditioned = metrics.get("case") == "preconditioned_1d_conditioned_kernel"
    m = logical["signal_ancillas"] if preconditioned else 2 + num_k_qubits(dims)
    qubits = logical["logical_qubits"]
    walk_qubits = 1 + m + (n if preconditioned else dims * n)
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
    if preconditioned:
        from laplacian_qsvt.preconditioned.inversion import target_components

        parts = target_components(n, logical["block_encoding"]["inner_psi"], eps)
        # W_cond's inner phase rotations use the resident flag, so it spans every qubit.
        emit(
            "W",
            parts["W"],
            list(range(qubits)),
            logical["num_w_applications"],
            logical["walk_contribution"]["per_w"],
            kind="walk",
        )
    else:
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
    if preconditioned:
        for item in logical["boundary_contribution"]["components"]:
            emit(
                item["id"],
                parts[item["id"]],
                item["system_qubits"],
                item["occurrences"],
                item,
                kind="boundary",
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
    if preconditioned:
        manifest.update(
            case=metrics["case"],
            case_title=metrics["case_title"],
            block_encoding={
                key: logical["block_encoding"][key]
                for key in ("inner_degree", "u_x_calls_per_w", "select_mcx_gates")
            },
            scope_notes=[
                "Conditioned kernel only: costs exclude preparing C^{-1} b and recovering the Poisson solution from L_cond^{-1}.",
                "W_cond nests an inner degree-"
                f"{logical['block_encoding']['inner_degree']} sinc^2 QSVT over U_X; the QFT pair is applied once at the boundaries.",
            ],
        )
        manifest["algorithm"]["inner_sinc2"] = metrics.get("inner")
        if "unfiltered_inverse" in metrics:
            manifest["algorithm"]["unfiltered_inverse"] = metrics["unfiltered_inverse"]
            manifest["scope_notes"].append("The scaled inverse is verified in the odd QSP quadrature. Coherent quadrature extraction is outside this estimate.")
        manifest["accounting"] = "One W_cond, each unique phase rotation and the QFT/QFT^dag boundaries, weighted by occurrences; all components retain the full resident register."
    (output / "inputs" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def export_compiled_components(
    metrics_path: Path, phases_path: Path, output: Path, metrics: dict
) -> dict:
    """Validate and import the exact compiled conditioned-operator components.

    Rebuilding an ordinary-Laplacian W from dimensions alone would silently
    substitute a different operator. The logical producer therefore supplies
    hashed, fully resident Clifford+T QASM for every component, including the
    Fourier/extraction boundary operations counted once per inverse sequence.
    """
    from qiskit import qasm2
    from laplacian_qsvt.laplacian.laplacian_nd_resource import CT_BASIS
    from run_layout_comparison import normalize_qasm

    logical = metrics["logical_resources"]
    if logical["dimensions"] != 1:
        raise ValueError("The preconditioned_1d exporter requires dimensions=1")
    source_components = metrics["compiled_components"]
    identifiers = [item["id"] for item in source_components]
    if len(set(identifiers)) != len(identifiers) or "W" not in identifiers:
        raise ValueError("Compiled component identifiers must be unique and include W")
    for item in source_components:
        identifier = item["id"]
        if not identifier or not all(char.isalnum() or char == "_" for char in identifier):
            raise ValueError(f"Invalid compiled component identifier: {identifier!r}")
        if item["kind"] not in {"walk", "projector", "boundary"}:
            raise ValueError(f"Unsupported component kind: {item['kind']}")
        if not isinstance(item["occurrences"], int) or item["occurrences"] < 1:
            raise ValueError("Component occurrences must be positive integers")
    if sum(c["kind"] == "walk" for c in source_components) != 1:
        raise ValueError("Expected one shared compiled W component")
    if next(c for c in source_components if c["id"] == "W")["kind"] != "walk":
        raise ValueError("Component W must have kind=walk")
    phases = [float(phi) for phi in json.loads(phases_path.read_text())["phases"]]
    expected = Counter()
    for item in source_components:
        if item["kind"] == "projector":
            expected[repr(float(item["phi"]))] += item["occurrences"]
    if Counter(repr(phi) for phi in phases) != expected or len(phases) != logical["num_projector_rotations"]:
        raise ValueError("Phase input does not match the saved logical estimate")
    if sum(c["occurrences"] for c in source_components if c["kind"] == "walk") != logical["num_w_applications"]:
        raise ValueError("Walk multiplicity does not match the saved logical estimate")

    # Validate every input before replacing any published inputs or snapshots.
    validated = []
    totals = Counter()
    qubits = logical["logical_qubits"]
    for item in source_components:
        source = Path(item["qasm"])
        if not source.is_absolute():
            source = metrics_path.parent / source
        source_bytes = source.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != item["qasm_sha256"]:
            raise ValueError(f"{item['id']} compiled QASM hash differs from the logical report")
        circuit = qasm2.loads(source_bytes.decode())
        counts = dict(circuit.count_ops())
        if set(counts) - set(CT_BASIS):
            raise ValueError(f"{item['id']} is not an explicit Clifford+T circuit")
        if counts != item["gate_counts"]:
            raise ValueError(f"{item['id']} gate counts differ from the logical report")
        t_count = counts.get("t", 0) + counts.get("tdg", 0)
        if t_count != item["t_count"]:
            raise ValueError(f"{item['id']} T count differs from the logical report")
        _, size = normalize_qasm(source_bytes.decode())
        if size != qubits or circuit.num_qubits != qubits:
            raise ValueError(f"{item['id']} does not retain the full resident register")
        validated.append((item, source_bytes))
        for gate, count in counts.items():
            totals[gate] += count * item["occurrences"]
    if dict(totals) != logical["gate_counts"]:
        raise ValueError("Exported component totals do not match the full logical estimate")

    inputs = output / "inputs" / "components"
    inputs.mkdir(parents=True, exist_ok=True)
    snapshot_logical_inputs(metrics_path, phases_path, output)
    components = []
    for item, source_bytes in validated:
        path = inputs / f"{item['id']}.qasm"
        # Preserve the producer's hashed bytes, including when it already wrote
        # directly to this path. The worker normalizes text after hash checking.
        path.write_bytes(source_bytes)
        components.append({
            **item,
            "qasm": str(path.resolve()),
            "qasm_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "logical_qubits": qubits,
        })
        print(f"Exported {item['id']}: {sum(item['gate_counts'].values()):,} gates, {item['occurrences']} occurrences", flush=True)
    manifest = {
        "schema_version": 1,
        "operator_family": metrics["operator_family"],
        "scope": "preconditioned_1d_inverse_component_architecture_inputs",
        "architecture_scope": metrics.get("architecture_scope", "1D conditioned inverse kernel on the nonzero-frequency subspace"),
        "scope_notes": metrics.get("scope_notes", []),
        "logical_metrics_source": str(metrics_path.resolve()),
        "logical_metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "phases_source": str(phases_path.resolve()),
        "phases_sha256": hashlib.sha256(phases_path.read_bytes()).hexdigest(),
        "dimensions": 1,
        "target_n": logical["target_n"],
        "logical_qubits": qubits,
        "num_w_applications": logical["num_w_applications"],
        "num_projector_rotations": logical["num_projector_rotations"],
        "num_boundary_components": sum(c["occurrences"] for c in components if c["kind"] == "boundary"),
        "logical_gate_counts": dict(totals),
        "algorithm": {key: metrics.get(key) for key in ("target_function", "kappa", "epsilon", "error_summary")},
        "minimum_kappa_for_full_nonzero_spectrum": logical.get("minimum_kappa_for_full_nonzero_spectrum"),
        "components": components,
        "accounting": "One conditioned W and each unique projector weighted by occurrences, plus explicitly compiled boundary components; all components retain the full resident register.",
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
    if metrics.get("operator_family") == "preconditioned_1d":
        from laplacian_qsvt.laplacian.preconditioned_inversion_resource import render_resource_report as render_preconditioned_report

        report = render_preconditioned_report(metrics)
    elif metrics.get("case") == "preconditioned_1d_conditioned_kernel" and "verification" in metrics:
        from laplacian_qsvt.preconditioned.inversion import (
            render_resource_report as render_preconditioned_report,
        )

        report = render_preconditioned_report(metrics)
    elif "verification" in metrics and "error_summary" in metrics:
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
