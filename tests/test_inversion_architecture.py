"""Checks for phase-specific component export and architectural cost composition."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from qiskit import qasm2
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/lattice_surgery"))

from export_inversion_components import export_components
from inversion_architecture import (
    aggregate_components, minimum_distance_proxy, publish_results, space_compact,
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


@unittest.skipUnless(
    (DEFAULT_CHECKOUT / ".deps/adapter/liblsqecc_adapter").is_file(),
    "Run setup_floorplanner.py first",
)
class NativeInverseTests(unittest.TestCase):
    def test_all_three_methods_produce_component_based_full_sequence_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            source, phases, _ = small_inputs(directory)
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
            for label in ("EXAMPLE", "SHARED ASSUMPTIONS", "RESULTS", "INTERPRETATION", "METHOD DETAILS AND LIMITS"):
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
                self.assertEqual(report["logical_qubits"], 7)
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
