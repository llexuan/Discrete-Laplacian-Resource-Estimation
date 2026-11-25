"""Checks for phase-specific component export and architectural cost composition."""

from __future__ import annotations

import contextlib
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/lattice_surgery"))

from export_inversion_components import export_components
from inversion_architecture import (
    aggregate_components, minimum_distance_proxy, publish_results, render_comparison,
    render_report, space_compact,
)
from estimate_sliced_resources import logical_error_per_round
from run_inversion_architecture import build_parser
from setup_floorplanner import DEFAULT_CHECKOUT
from laplacian_qsvt.laplacian.laplacian_inversion_encode import (
    build_projector_rotation,
    estimate_full_qsp_resources,
)


def small_inputs(directory, dims=2):
    phases = [0.13, -0.24, 0.13]
    metrics = {"logical_resources": estimate_full_qsp_resources(phases, dims, 1, 1e-5)}
    path = directory / "logical.json"
    phase_path = directory / "phases.json"
    path.write_text(json.dumps(metrics))
    phase_path.write_text(json.dumps({"phases": phases}))
    return path, phase_path, metrics


class InverseExportTests(unittest.TestCase):
    def compiled_conditioned_inputs(self, directory):
        """A small explicit artifact fixture, independent of operator synthesis."""
        components = []
        totals = Counter()
        for identifier, kind, occurrences in (
            ("W", "walk", 2), ("R_000", "projector", 3),
            ("QFT_IN", "boundary", 1), ("QFT_OUT", "boundary", 1),
        ):
            circuit = QuantumCircuit(5)
            circuit.h(0)
            if kind != "boundary":
                circuit.t(1)
                circuit.cx(0, 2)
            qasm = directory / f"{identifier}.qasm"
            qasm.write_text(qasm2.dumps(circuit))
            counts = dict(circuit.count_ops())
            component = {
                "id": identifier, "kind": kind, "occurrences": occurrences,
                "qasm": qasm.name,
                "qasm_sha256": hashlib.sha256(qasm.read_bytes()).hexdigest(),
                "gate_counts": counts, "t_count": counts.get("t", 0),
            }
            if kind == "projector":
                component["phi"] = 0.13
            components.append(component)
            for gate, count in counts.items():
                totals[gate] += count * occurrences
        metrics = {
            "operator_family": "preconditioned_1d",
            "architecture_scope": "Conditioned inverse kernel on nonzero modes",
            "scope_notes": ["Source conditioning is excluded."],
            "compiled_components": components,
            "kappa": 2.5,
            "logical_resources": {
                "dimensions": 1, "target_n": 3, "logical_qubits": 5,
                "num_w_applications": 2, "num_projector_rotations": 3,
                "gate_counts": dict(totals),
            },
        }
        source = directory / "logical.json"
        source.write_text(json.dumps(metrics))
        phases = directory / "phases.json"
        phases.write_text(json.dumps({"phases": [0.13] * 3}))
        return source, phases, metrics

    def test_preconditioned_export_validates_artifacts_and_keeps_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, phases, metrics = self.compiled_conditioned_inputs(directory)
            with contextlib.redirect_stdout(io.StringIO()), patch("export_inversion_components.snapshot_logical_inputs"):
                manifest = export_components(source, phases, directory / "case")
            self.assertEqual(manifest["operator_family"], "preconditioned_1d")
            self.assertEqual(manifest["num_boundary_components"], 2)
            self.assertEqual(manifest["logical_gate_counts"], metrics["logical_resources"]["gate_counts"])
            self.assertEqual([c["kind"] for c in manifest["components"]], ["walk", "projector", "boundary", "boundary"])
            for item in manifest["components"]:
                path = Path(item["qasm"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["qasm_sha256"])
                self.assertEqual(qasm2.loads(path.read_text()).num_qubits, 5)

    def test_preconditioned_export_rejects_tampering_and_false_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, phases, metrics = self.compiled_conditioned_inputs(directory)
            qasm = directory / "W.qasm"
            qasm.write_text(qasm.read_text() + "x q[3];\n")
            with self.assertRaisesRegex(ValueError, "QASM hash"):
                export_components(source, phases, directory / "bad")
            metrics["compiled_components"][0]["qasm_sha256"] = hashlib.sha256(qasm.read_bytes()).hexdigest()
            source.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "gate counts"):
                export_components(source, phases, directory / "bad")
            self.assertFalse((directory / "bad").exists())

    def test_preconditioned_export_rejects_wrong_phase_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, phases, _ = self.compiled_conditioned_inputs(directory)
            phases.write_text(json.dumps({"phases": [0.13, 0.13, 0.14]}))
            with self.assertRaisesRegex(ValueError, "Phase input does not match"):
                export_components(source, phases, directory / "bad")

    def test_preconditioned_export_keeps_endpoint_variants_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            directory = output / "inputs" / "components"
            directory.mkdir(parents=True)
            source, phases, metrics = self.compiled_conditioned_inputs(directory)
            original = (directory / "W.qasm").read_bytes()
            metrics["compiled_components"][1]["occurrences"] = 2
            variant = {**metrics["compiled_components"][1], "id": "R_END", "occurrences": 1}
            metrics["compiled_components"].append(variant)
            source.write_text(json.dumps(metrics))
            with contextlib.redirect_stdout(io.StringIO()), patch("export_inversion_components.snapshot_logical_inputs"):
                manifest = export_components(source, phases, output)
            self.assertEqual((directory / "W.qasm").read_bytes(), original)
            self.assertEqual(sum(c["kind"] == "projector" for c in manifest["components"]), 2)
            self.assertEqual(sum(c["occurrences"] for c in manifest["components"] if c["kind"] == "projector"), 3)

    def test_projector_export_restores_flag_and_preserves_dirty_system_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, phases, metrics = small_inputs(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = export_components(source, phases, directory / "components")
            self.assertEqual(manifest["logical_qubits"], 7)
            self.assertEqual(
                [c["occurrences"] for c in manifest["components"]], [2, 2, 1]
            )
            component = manifest["components"][1]
            actual = Operator(qasm2.loads(Path(component["qasm"]).read_text())).data
            # Flag is high-order qubit 6. Both borrowed system qubits are retained;
            # the flag-zero block must equal projector x identity on arbitrary data.
            actual = actual[:64, :64]
            target = np.kron(
                np.eye(4), Operator(build_projector_rotation(component["phi"], 3)).data
            )
            overlap = np.trace(target.conj().T @ actual)
            phase = overlap / abs(overlap)
            self.assertLessEqual(np.max(abs(actual - phase * target)), 1e-5)
            self.assertEqual(
                manifest["logical_gate_counts"],
                metrics["logical_resources"]["gate_counts"],
            )

    def test_3d_export_keeps_resident_qubits_and_rejects_wrong_phase_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, phases, _ = small_inputs(directory, dims=3)
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = export_components(source, phases, directory / "components")
            self.assertEqual(manifest["logical_qubits"], 9)
            self.assertTrue(
                all(
                    qasm2.loads(Path(c["qasm"]).read_text()).num_qubits == 9
                    for c in manifest["components"]
                )
            )
            phases.write_text(json.dumps({"phases": [0.13, -0.24, 0.31]}))
            with self.assertRaisesRegex(ValueError, "Phase input does not match"):
                export_components(source, phases, directory / "bad")


class CompositionTests(unittest.TestCase):
    def test_conditioned_boundary_cost_is_included_without_extra_projector(self):
        args, manifest, records = self.fixture()
        manifest.update(
            operator_family="preconditioned_1d", dimensions=1,
            architecture_scope="Conditioned inverse kernel on nonzero modes",
            scope_notes=["Source conditioning is excluded."],
            num_boundary_components=2,
            minimum_kappa_for_full_nonzero_spectrum=None,
            algorithm={"kappa": 2.5},
        )
        manifest["components"].append({"id": "BOUNDARY", "kind": "boundary", "occurrences": 2, "t_count": 3})
        manifest["logical_gate_counts"]["t"] += 6
        records.append({
            "status": "success", "slices": 5,
            "resources": {"area": 30},
            "instruction_counts": {"RequestMagicState": 3},
            "artifact_directory": "evidence", "input_sha256": "hash",
        })
        result = aggregate_components(manifest, records, 30, args)
        self.assertEqual(result["schedule_projection"]["slices"], 45)
        self.assertEqual(result["schedule_projection"]["contributions"]["boundary"]["slices"], 10)
        self.assertEqual(result["weighted_instruction_counts"]["RequestMagicState"], 34)
        self.assertEqual(result["unique_projectors"], 2)
        self.assertTrue(result["assumptions"]["explicit_sequence_boundary_gates_included"])
        self.assertFalse(any("full nonzero target spectrum" in warning for warning in result["warnings"]))
        result.update(method="compact", layout_resources={"height": 3, "width": 10, "area": 30}, layout_preparation="Native compact", compiler={"native_commit": "test"}, evidence_directory="evidence")
        report = render_report(result)
        self.assertIn("1D preconditioned inverse architecture", report)
        self.assertIn("Explicit boundary contribution, slices: 10", report)
        comparison = render_comparison({"methods": {"compact": result}, "run_directory": "test"})
        self.assertIn("1D PRECONDITIONED INVERSE", comparison)
        self.assertIn("Explicit boundary slices", comparison)
        self.assertIn("Source conditioning is excluded.", comparison)

    def test_magic_state_count_mismatch_cannot_produce_estimate(self):
        args, manifest, records = self.fixture()
        records[0]["instruction_counts"]["RequestMagicState"] -= 1
        with self.assertRaisesRegex(ValueError, "magic-state requests"):
            aggregate_components(manifest, records, 30, args)

    def test_failed_or_unselected_methods_replace_previous_visible_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "EDPC_resource_report.txt").write_text("Old successful score: 123")
            comparison = {
                "run_directory": str(output / "runs" / "failed-run"),
                "methods": {"compact": {"method": "compact", "status": "failed", "reason": "Routing failed"}},
            }
            text = publish_results(output, comparison)
            self.assertIn("INCOMPLETE / FAILED METHODS", text)
            self.assertIn("Routing failed", text)
            self.assertIn("Status: NOT_RUN", (output / "EDPC_resource_report.txt").read_text())
            self.assertNotIn("123", (output / "EDPC_resource_report.txt").read_text())
            self.assertEqual(json.loads((output / "data/comparison.json").read_text())["run_id"], "failed-run")

    def fixture(self):
        args = build_parser().parse_args(["--fixed-code-distance", "--code-distance", "9"])
        components = [
            {"id": "W", "kind": "walk", "occurrences": 2, "t_count": 10},
            {"id": "R_0", "kind": "projector", "occurrences": 2, "t_count": 2},
            {"id": "R_1", "kind": "projector", "occurrences": 1, "t_count": 4},
        ]
        manifest = {
            "components": components,
            "logical_qubits": 7,
            "dimensions": 2,
            "target_n": 1,
            "num_w_applications": 2,
            "num_projector_rotations": 3,
            "logical_gate_counts": {"t": 28, "h": 7},
        }
        records = [
            {
                "status": "success",
                "slices": slices,
                "resources": {"area": 30},
                "instruction_counts": {"RequestMagicState": c["t_count"]},
                "artifact_directory": "evidence",
                "input_sha256": "hash",
            }
            for c, slices in zip(components, [10, 4, 7])
        ]
        return args, manifest, records

    def test_time_is_weighted_but_resident_qubits_are_not_multiplied(self):
        args, manifest, records = self.fixture()
        result = aggregate_components(manifest, records, 30, args)
        self.assertEqual(result["schedule_projection"]["slices"], 35)
        self.assertEqual(
            result["physical_projection"]["physical_qubits"], 30 * 2 * 9**2
        )
        self.assertEqual(result["physical_projection"]["core_time_ns"], 35 * 9 * 1000)
        self.assertEqual(result["weighted_instruction_counts"]["RequestMagicState"], 28)
        self.assertAlmostEqual(
            result["physical_projection"]["failure_probability_proxy"], 0.00945
        )
        self.assertEqual(
            result["physical_projection"]["minimum_odd_code_distance_proxy"], 9
        )

    def test_failed_measurement_and_wrong_frequency_cannot_produce_totals(self):
        args, manifest, records = self.fixture()
        records[1]["status"] = "error"
        with self.assertRaisesRegex(ValueError, "All component compilations"):
            aggregate_components(manifest, records, 30, args)
        records[1]["status"] = "success"
        manifest["components"][1]["occurrences"] = 1
        with self.assertRaisesRegex(ValueError, "multiplicities"):
            aggregate_components(manifest, records, 30, args)

    def test_each_method_selects_its_own_minimum_distance_and_rescales_resources(self):
        args, manifest, records = self.fixture()
        args.auto_code_distance = True
        args.code_distance = 3
        distances = []
        for area in (3, 30, 300):
            for record in records:
                record["resources"]["area"] = area
            result = aggregate_components(manifest, records, area, args)
            physical = result["physical_projection"]
            d = result["assumptions"]["code_distance"]
            distances.append(d)
            self.assertLessEqual(physical["failure_probability_proxy"], 0.01)
            for smaller in range(3, d, 2):
                self.assertGreater(area * 35 * smaller * logical_error_per_round(0.001, smaller), 0.01)
            self.assertEqual(physical["physical_qubits"], 2 * area * d**2)
            self.assertEqual(physical["runtime_ns"], 35 * d * 1000)
            self.assertTrue(physical["meets_target_failure_probability"])
        self.assertEqual(distances, [7, 9, 13])
        self.assertEqual(args.code_distance, 3)

    def test_distance_search_handles_bounds_and_unreachable_budgets(self):
        self.assertEqual(minimum_distance_proxy(1050, 0.001, 0.01), 9)
        self.assertEqual(minimum_distance_proxy(1050, 0.001, 0.01, minimum=10), 11)
        self.assertIsNone(minimum_distance_proxy(1050, 0.01, 0.01))
        args, manifest, records = self.fixture()
        args.auto_code_distance = True
        args.physical_error_rate = 0.01
        with self.assertRaisesRegex(ValueError, "No admissible odd code distance"):
            aggregate_components(manifest, records, 30, args)

    def test_default_distance_policy_is_independent_with_optional_fixed_override(self):
        args = build_parser().parse_args([])
        self.assertTrue(args.auto_code_distance)
        self.assertEqual(args.code_distance, 3)
        args, manifest, records = self.fixture()
        args.code_distance = 3
        result = aggregate_components(manifest, records, 30, args)
        self.assertEqual(result["assumptions"]["distance_selection_mode"], "fixed")
        self.assertEqual(result["assumptions"]["code_distance"], 3)
        self.assertFalse(result["physical_projection"]["meets_target_failure_probability"])

    def test_factory_footprint_is_added_once_with_explicit_factory_count(self):
        args, manifest, records = self.fixture()
        args.factory_cycle_rounds = 30
        with self.assertRaisesRegex(ValueError, "specify --num-factories"):
            aggregate_components(manifest, records, 30, args)
        args.num_factories, args.factory_physical_qubits = 2, 1000
        result = aggregate_components(manifest, records, 30, args)
        self.assertEqual(result["physical_projection"]["physical_qubits"], 4860 + 2000)
        self.assertEqual(
            result["physical_projection"]["runtime_ns"], (28 * 30 / 2) * 1000
        )

    def test_compact_spacing_preserves_data_and_factory_footprint(self):
        source = "QQr00\nrrr00\nQAr00\n"
        spaced = space_compact(source)
        for symbol in "QA0":
            self.assertEqual(source.count(symbol), spaced.count(symbol))
        self.assertGreater(len(spaced.splitlines()[0]), len(source.splitlines()[0]))

    def test_litinski_factory_sizes_for_supply_time_and_combined_failure(self):
        args, manifest, records = self.fixture()
        args.factory_model = "litinski-15to1-two-level"
        args.auto_code_distance, args.code_distance = True, 3
        # At d=11 the layout alone passes this target, the combined proxy fails.
        rounds = 28 * 82.5
        layout_at_11 = 30 * rounds * logical_error_per_round(0.001, 11)
        args.target_failure_probability = layout_at_11 + 14 * 2.7e-12
        r = aggregate_components(manifest, records, 30, args)
        self.assertEqual(r["assumptions"]["code_distance"], 13)
        p = r["physical_projection"]
        self.assertEqual(p["physical_qubits"], 30 * 2 * 13**2 + 30_700)
        self.assertEqual(p["runtime_ns"], rounds * 1000)
        self.assertTrue(p["factory_model"]["is_throughput_limited"])
        self.assertAlmostEqual(p["combined_failure_proxy"], 30 * rounds * logical_error_per_round(0.001, 13) + 28 * 2.7e-12)
        self.assertLessEqual(p["combined_failure_proxy"], args.target_failure_probability)

    def test_litinski_profile_rejects_incompatible_assumptions(self):
        args, manifest, records = self.fixture()
        args.factory_model = "litinski-15to1-two-level"
        args.factory_cycle_rounds = 10
        with self.assertRaisesRegex(ValueError, "already specifies"):
            aggregate_components(manifest, records, 30, args)
        args.factory_cycle_rounds, args.physical_error_rate = None, 1e-4
        with self.assertRaisesRegex(ValueError, "0.001"):
            aggregate_components(manifest, records, 30, args)


@unittest.skipUnless(
    (DEFAULT_CHECKOUT / ".deps/adapter/liblsqecc_adapter").is_file(),
    "Run setup_floorplanner.py first",
)
class NativeInverseTests(unittest.TestCase):
    def test_all_three_methods_produce_component_based_full_sequence_reports(self):
        self.check_all_methods(dims=2)

    def test_3d_all_three_methods_produce_full_sequence_reports(self):
        self.check_all_methods(dims=3)

    def check_all_methods(self, dims):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, phases, _ = small_inputs(directory, dims=dims)
            output = directory / "architecture"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/lattice_surgery/run_inversion_architecture.py"),
                    "--logical-metrics",
                    str(source),
                    "--phases-json",
                    str(phases),
                    "--output-dir",
                    str(output),
                    "--budget",
                    "13",
                    "--auto-code-distance",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            comparison = json.loads((output / "data/comparison.json").read_text())
            text = (output / "comparison_report.txt").read_text()
            for label in ("METHOD CONFIGURATIONS", "EXAMPLE", "SHARED ASSUMPTIONS", "RESULTS", "INTERPRETATION", "MODEL LIMITS"):
                self.assertIn(label, text)
            self.assertTrue((output / "logical/inversion_resource_report.txt").is_file())
            self.assertEqual((output / "logical/inversion_resource_metrics.json").read_bytes(), source.read_bytes())
            manifest = json.loads((output / "inputs/manifest.json").read_text())
            self.assertTrue(all(Path(c["qasm"]).is_file() for c in manifest["components"]))
            self.assertFalse((output / "manifest.json").exists())
            reports = [
                json.loads((output / "data" / f"{prefix}_resource_estimate.json").read_text())
                for prefix in ("compact", "EDPC", "annealed")
            ]
            self.assertTrue(all(r["status"] == "success" for r in reports))
            self.assertEqual(len({r["compiler"]["binary_sha256"] for r in reports}), 1)
            for report in reports:
                self.assertEqual(report["assumptions"]["distance_selection_mode"], "per_method_minimum")
                self.assertEqual(report["assumptions"]["code_distance"], report["physical_projection"]["minimum_odd_code_distance_proxy"])
                self.assertEqual(report["run_id"], comparison["run_id"])
                self.assertEqual(report["num_w_applications"], 2)
                self.assertEqual(report["num_projector_rotations"], 3)
                self.assertEqual(report["unique_projectors"], 2)
                self.assertEqual(report["dimensions"], dims)
                self.assertEqual(report["logical_qubits"], 7 if dims == 2 else 9)
                self.assertEqual(
                    report["schedule_projection"]["slices"],
                    sum(
                        c["occurrences"] * c["one_component_slices"]
                        for c in report["components"]
                    ),
                )
                self.assertFalse(
                    report["assumptions"]["component_boundary_costs_included"]
                )
                self.assertLessEqual(
                    report["physical_projection"]["failure_probability_proxy"],
                    report["assumptions"]["target_failure_probability"],
                )
                self.assertFalse(list(output.rglob("*compiler_report.txt")))


if __name__ == "__main__":
    unittest.main()
