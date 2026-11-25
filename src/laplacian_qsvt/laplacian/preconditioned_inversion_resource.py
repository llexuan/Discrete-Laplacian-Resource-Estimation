"""Compile and count a coherent 1D conditioned inverse from reusable components."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.circuit.library import QFTGate

from ..paths import PHASES_DIR, OUTPUTS_DIR
from ..synthesis.circuit_synthesis import (
    arbitrary_rotation_count, circuit_resources, compile_rotations, expand_to_rotations,
)


def weighted_resources(components: list[dict]) -> dict:
    totals = Counter()
    depths = t_depths = 0
    for component in components:
        weight = component["occurrences"]
        totals.update({gate: weight * count for gate, count in component["gate_counts"].items()})
        depths += weight * component["total_depth"]
        t_depths += weight * component["t_depth"]
    t_count = totals.get("t", 0) + totals.get("tdg", 0)
    return {
        "gate_counts": dict(totals),
        "clifford_gate_count": sum(totals.values()) - t_count,
        "t_count": t_count,
        "total_gates": sum(totals.values()),
        "total_depth": depths,
        "t_depth": t_depths,
    }


def compile_components(components: list[dict], output: Path, synthesis_tol: float) -> tuple[list[dict], dict]:
    """Allocate one budget across all weighted rotations, then emit actual QASM."""
    if not math.isfinite(synthesis_tol) or not 0 < synthesis_tol < 1:
        raise ValueError("synthesis_tol must be finite and between zero and one")
    expanded = []
    num_rotations = 0
    for component in components:
        print(f"Expanding {component['id']}...", flush=True)
        circuit = expand_to_rotations(component["circuit"])
        rotations = arbitrary_rotation_count(circuit)
        print(f"  {len(circuit.data):,} expanded gates; {rotations:,} arbitrary Rz", flush=True)
        num_rotations += rotations * component["occurrences"]
        expanded.append((component, circuit))
    allowance = synthesis_tol / max(1, num_rotations)
    epsilon = 10.0 ** math.floor(math.log10(allowance))
    print(f"Weighted arbitrary rotations: {num_rotations:,}; per-rotation budget {epsilon:.1e}", flush=True)
    directory = output / "inputs" / "components"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    summed_error = 0.0
    for component, expanded_circuit in expanded:
        print(f"Synthesizing {component['id']} (epsilon per Rz {epsilon:.1e})...", flush=True)
        compiled, synthesis = compile_rotations(expanded_circuit, epsilon)
        record = {key: value for key, value in component.items() if key != "circuit"}
        record.update(circuit_resources(compiled))
        qasm = qasm2.dumps(compiled) + "\n"
        path = directory / f"{component['id']}.qasm"
        path.write_text(qasm)
        record.update(
            qasm=str(path.resolve()),
            qasm_sha256=hashlib.sha256(qasm.encode()).hexdigest(),
            dropped_global_phase=float(compiled.global_phase),
            rotation_synthesis=synthesis,
        )
        records.append(record)
        summed_error += component["occurrences"] * synthesis["summed_operator_error_bound"]
        print(f"  {record['total_gates']:,} gates, {record['t_count']:,} T/Tdg, weight {record['occurrences']}", flush=True)
    if summed_error > synthesis_tol:
        raise ValueError(f"Measured synthesis bound {summed_error} exceeds {synthesis_tol}")
    return records, {
        "num_approximated_rotations": num_rotations,
        "synthesis_tolerance": synthesis_tol,
        "eps_per_rotation": epsilon,
        "measured_synthesis_error_bound": summed_error,
        "qasm_global_phase": "One scalar phase per complete component is omitted by QASM2; no later controls are applied to exported components.",
    }


def add_fourier_boundaries(components: list[dict], system_qubits: list[int], total_qubits: int) -> list[dict]:
    forward = QuantumCircuit(total_qubits, name="Fourier_input")
    forward.append(QFTGate(len(system_qubits)), system_qubits)
    backward = forward.inverse()
    return [
        {"id": "Fourier_input", "kind": "boundary", "occurrences": 1, "circuit": forward},
        *components,
        {"id": "Fourier_output", "kind": "boundary", "occurrences": 1, "circuit": backward},
    ]


def approximation_summary(inner: dict, phases_metadata: dict, synthesis: dict) -> dict:
    coeffs = np.asarray(phases_metadata["cheb_coeffs"], dtype=float)
    kappa = float(phases_metadata["kappa"])
    scale = float(phases_metadata["returned_scale"])
    grid = np.linspace(1 / kappa, 1.0, 100001)
    values = np.polynomial.chebyshev.chebval(grid, coeffs)
    sampled = float(np.max(np.abs(values - scale / grid)))
    lipschitz = float(np.dot(np.arange(len(coeffs), dtype=float) ** 2, np.abs(coeffs)))
    spacing = float(grid[1] - grid[0])
    # The Chebyshev derivative bound is global on [-1,1]. Every point is
    # within spacing/2 of a grid point; add the target inverse's derivative.
    polynomial_bound = sampled + 0.5 * spacing * (lipschitz + scale * kappa**2)
    inner_error = (float(inner["safety_scale"]) * float(inner["uniform_polynomial_error_bound_unscaled"])
                   + float(inner["phase_vs_poly_sampled_max_abs_error"]))
    propagated = lipschitz * inner_error
    phase_error = float(phases_metadata.get("phase_vs_poly_max_abs_error", 0))
    return {
        "outer_inverse_sampled_max_abs_error": sampled,
        "outer_inverse_grid_lipschitz_bound": polynomial_bound,
        "outer_chebyshev_derivative_bound": lipschitz,
        "inner_operator_error_bound": inner_error,
        "inner_error_reference": "L_cond / inner_normalization, including sampled inner phase residual",
        "inner_error_propagated_bound": propagated,
        "outer_phase_sampled_error": phase_error,
        "rotation_synthesis_error_bound": synthesis["measured_synthesis_error_bound"],
        "total_error_estimate": polynomial_bound + propagated + phase_error + synthesis["measured_synthesis_error_bound"],
        "interpretation": "Absolute error of the scaled conditioned inverse block. Analytic polynomial bounds evaluated in floating point; phase fitting and circuit checks are numerical, not an interval-arithmetic certificate.",
        "propagation": "sum(k^2*abs(c_k))*inner_operator_error; synthesized unitary errors sum over all weighted component uses",
    }


def render_resource_report(metrics: dict) -> str:
    logical = metrics["logical_resources"]
    inner, errors = metrics["inner_polynomial"], metrics["error_summary"]
    lines = [
        "=== 1D preconditioned Laplacian inverse: logical resource report ===", "",
        "Scope: one coherent scaled conditioned inverse kernel; nonzero Fourier modes.",
        "Includes inner and outer polynomial signal extraction and input/output QFTs.",
        "Original-equation input conditioning, state preparation, readout and repetition costs are excluded.", "",
        f"System qubits         : {logical['target_n']}",
        f"Grid points           : {2 ** logical['target_n']:,}",
        f"Resident logical qubits: {logical['logical_qubits']:,}",
        f"Inner polynomial degree: {inner['degree']}",
        f"Inner normalization alpha: {inner['normalization']:.12f}",
        f"Effective inverse scale: {metrics['effective_inverse_scale']:.12f} multiplying L_cond^-1",
        f"Outer polynomial degree: {metrics['degree']}",
        f"Conditioned queries (W): {logical['num_w_applications']}",
        f"Outer phase rotations : {logical['num_projector_rotations']}",
        f"Distinct original angles: {logical['projector_rotation_contribution']['num_unique_rotations']}",
        f"Distinct phase circuits: {sum(c['kind'] == 'projector' for c in metrics['compiled_components'])}",
        "", "LOGICAL RESOURCES", "",
        f"Clifford gates        : {logical['clifford_gate_count']:,}",
        f"T/T-dagger gates      : {logical['t_count']:,}",
        f"Total gates           : {logical['total_gates']:,}",
        f"Depth upper bound     : {logical['total_depth']:,}",
        f"T-depth upper bound   : {logical['t_depth']:,}",
        "Depths are serial sums of compiled component depths.", "",
        f"{'Component':<25} {'Clifford':>15} {'T/T-dagger':>15} {'Depth':>15}",
    ]
    for kind, label in (("walk", "All conditioned queries"), ("projector", "All outer phases"), ("boundary", "All boundary gates")):
        group = weighted_resources([c for c in metrics["compiled_components"] if c["kind"] == kind])
        lines.append(f"{label:<25} {group['clifford_gate_count']:>15,} {group['t_count']:>15,} {group['total_depth']:>15,}")
    lines += ["", "SPECTRUM AND ACCURACY", "",
        f"Ideal conditioned spectrum: [{4 / math.pi**2:.10f}, 1]",
        f"Encoded spectrum lower bound: {metrics['encoded_spectrum_lower_bound']:.10f}",
        f"Outer inversion domain: [{1 / metrics['kappa']:.10f}, 1]",
        f"Full nonzero spectrum covered: {metrics['spectral_coverage_passed']}",
        "Zero mode: g(0)=1 continuous extension; original periodic nullspace is not inverted.",
        f"Inner operator error bound: {errors['inner_operator_error_bound']:.6e}",
        f"Outer inverse sampled error: {errors['outer_inverse_sampled_max_abs_error']:.6e}",
        f"Outer inverse grid/Lipschitz bound: {errors['outer_inverse_grid_lipschitz_bound']:.6e}",
        f"Inner error after outer polynomial: {errors['inner_error_propagated_bound']:.6e}",
        f"Whole-sequence synthesis budget: {logical['synthesis_tolerance']:.6e}",
        f"Measured synthesis error sum: {logical['measured_synthesis_error_bound']:.6e}",
        f"Total absolute error estimate: {errors['total_error_estimate']:.6e}",
        errors["interpretation"], "", "SMALL-GRID CIRCUIT VERIFICATION", "",
        json.dumps(metrics["verification"], indent=2), "",
        "Native architecture projections retain the existing abstract magic-state and correction model.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-n", type=int, default=20)
    parser.add_argument("--verify-n", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--sinc-error", type=float, default=1e-7)
    parser.add_argument("--synthesis-tol", type=float, default=1e-8)
    parser.add_argument("--phases-json", type=Path, default=PHASES_DIR / "phases_inversion.json")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    from .preconditioned_1d import (
        build_preconditioned_walk, build_signed_projector_rotation,
        make_inner_polynomial, reflection_form_phases, verify_conditioned_inverse,
    )
    args = build_parser().parse_args()
    if args.target_n < 2:
        raise ValueError("--target-n must be at least two")
    if any(n < 2 or n > 4 for n in args.verify_n):
        raise ValueError("--verify-n must be between two and four (small circuit checks)")
    output = (args.output_dir or OUTPUTS_DIR / "preconditioned_laplacian" / f"1d_n{args.target_n}").resolve()
    phase_metadata = json.loads(args.phases_json.read_text())
    phases = [float(value) for value in phase_metadata["phases"]]
    if not np.all(np.isfinite(phases)) or len(phases) < 2:
        raise ValueError("Invalid outer phase sequence")
    if not np.allclose(phases, list(reversed(phases)), atol=1e-12, rtol=0):
        raise ValueError("Coherent extraction requires symmetric outer phases")
    degree = len(phases) - 1
    if phase_metadata["degree"] != degree:
        raise ValueError("Outer phase count does not match polynomial degree")
    inner = make_inner_polynomial(args.sinc_error)
    print(f"Inner sinc polynomial degree {inner['degree']}; outer inverse degree {degree}", flush=True)
    lower = inner["encoded_spectral_lower_bound"] - inner["phase_vs_poly_sampled_max_abs_error"]
    kappa = float(phase_metadata["kappa"])
    if lower < 1 / kappa:
        raise ValueError("Encoded conditioned spectrum is outside the inverse polynomial domain")
    verification = []
    for q in dict.fromkeys(args.verify_n):
        print(f"Verifying actual coherent circuits at N={2**q}...", flush=True)
        check = verify_conditioned_inverse(q, inner, phases, phase_metadata["cheb_coeffs"])
        if not check["passed"]:
            raise ValueError(f"Conditioned circuit verification failed: {check}")
        verification.append(check)
    walk = build_preconditioned_walk(args.target_n, inner)
    m = walk.metadata["ancilla_count"]
    total_qubits = walk.num_qubits + 2
    padded_walk = QuantumCircuit(total_qubits, name="W_conditioned")
    padded_walk.compose(walk, list(range(2, total_qubits)), inplace=True)
    start = QuantumCircuit(total_qubits, name="coherent_input")
    start.s(0)
    start.h(0)
    start.rz(-degree * math.pi, 0)
    end = QuantumCircuit(total_qubits, name="coherent_output")
    end.h(0)
    end.sdg(0)
    end.x(0)
    converted = reflection_form_phases(phases)
    projectors = {}
    sequence = ["Fourier_input", "Coherent_input"]
    for index, (phi, shifted) in enumerate(zip(phases, converted)):
        key = (repr(phi), repr(shifted))
        if key not in projectors:
            projectors[key] = {
                "id": f"R_{len(projectors):03d}", "kind": "projector", "occurrences": 0,
                "phi": phi, "implemented_reflection_phase": shifted,
                "circuit": build_signed_projector_rotation(shifted, m, args.target_n),
            }
        projectors[key]["occurrences"] += 1
        sequence.append(projectors[key]["id"])
        if index < degree:
            sequence.append("W")
    sequence.extend(["Coherent_output", "Fourier_output"])
    components = [
        {"id": "W", "kind": "walk", "occurrences": degree, "circuit": padded_walk},
        *projectors.values(),
        {"id": "Coherent_input", "kind": "boundary", "occurrences": 1, "circuit": start},
        {"id": "Coherent_output", "kind": "boundary", "occurrences": 1, "circuit": end},
    ]
    system = list(range(m + 2, total_qubits))
    components = add_fourier_boundaries(components, system, total_qubits)
    compiled, synthesis = compile_components(components, output, args.synthesis_tol)
    walk_record = next(item for item in compiled if item["kind"] == "walk")
    projector_records = [item for item in compiled if item["kind"] == "projector"]
    logical = {
        **weighted_resources(compiled), **synthesis,
        "dimensions": 1, "target_n": args.target_n,
        "grid_points_per_dimension": 2 ** args.target_n,
        "logical_qubits": total_qubits,
        "num_w_applications": degree,
        "num_projector_rotations": len(phases),
        "num_w_selector_rotations": degree * walk_record["rotation_synthesis"]["arbitrary_rotations"],
        "target_spectral_gap": lower,
        "minimum_kappa_for_full_nonzero_spectrum": 1 / lower,
        "depth_accounting": "serial_component_sum_upper_bound",
        "walk_contribution": {"per_w": walk_record, **weighted_resources([walk_record])},
        "projector_rotation_contribution": {
            **weighted_resources(projector_records),
            "num_unique_rotations": len(set(phases)),
            "unique_rotation_resources": projector_records,
        },
    }
    metrics = {
        "schema_version": 1, "operator_family": "preconditioned_1d",
        "scope": "coherent_1d_conditioned_inverse_kernel",
        "architecture_scope": "1D conditioned inverse kernel, including coherent polynomial extraction and QFT boundaries",
        "scope_notes": [
            "Costs cover one application of the scaled conditioned inverse on nonzero Fourier modes.",
            "Input conditioning for the original equation, source preparation, measurement and success-probability repetition costs are excluded.",
            "The zero-frequency eigenvalue uses the continuous extension g(0)=1; the original Laplacian nullspace remains a separate application constraint.",
        ],
        "target_function": "scaled inverse of Fourier-conditioned 1D periodic Laplacian",
        "kappa": kappa, "epsilon": phase_metadata["epsilon"],
        "degree": degree, "effective_degree": degree,
        "effective_inverse_scale": float(phase_metadata["returned_scale"]) * inner["normalization"],
        "inner_polynomial": inner, "outer_phase_source": str(args.phases_json.resolve()),
        "encoded_spectrum_lower_bound": lower, "spectral_coverage_passed": True,
        "verification": verification, "logical_resources": logical,
        "error_summary": approximation_summary(inner, phase_metadata, synthesis),
        "compiled_components": compiled, "sequence_component_ids": sequence,
        "registers": {"outer_extraction": 0, "outer_predicate": 1,
                      "inner_signal_ancillas": list(range(2, m + 2)), "system": system},
        "query_convention": "W is the Hermitian U_cond query. Endpoint and interior projector shifts implement the equivalent reflection-form QSP sequence.",
    }
    logical_dir = output / "logical"
    logical_dir.mkdir(parents=True, exist_ok=True)
    (output / "inputs" / "phases.json").write_bytes(args.phases_json.read_bytes())
    (output / "inputs" / "inner_phases.json").write_text(json.dumps(inner, indent=2) + "\n")
    (logical_dir / "inversion_resource_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = logical_dir / "inversion_resource_report.txt"
    report.write_text(render_resource_report(metrics))
    print(f"Saved: {report}", flush=True)


if __name__ == "__main__":
    main()
