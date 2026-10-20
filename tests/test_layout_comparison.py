"""Regression checks for input semantics, winner replay, and matched resource accounting."""

from __future__ import annotations

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

from compiler_options import backend_options
from estimate_sliced_resources import run_checked
from export_qiskit_qasm import export_circuit
from run_layout_comparison import DEFAULT_CHECKOUT, normalize_qasm


class LayoutInputTests(unittest.TestCase):
    def test_normalization_preserves_unitary_and_rejects_unsupported_inputs(self):
        source = 'OPENQASM 2.0; include "qelib1.inc"; qreg data[2]; h data[1]; cx data[1],data[0]; tdg data[0];'
        normalized, count = normalize_qasm(source)
        self.assertEqual(count, 2)
        self.assertEqual(normalize_qasm(source + "barrier data;")[0], normalized)
        np.testing.assert_allclose(
            Operator(qasm2.loads(source)).data, Operator(qasm2.loads(normalized)).data
        )
        for suffix in (
            "measure data[0] -> c[0];",
            "rx(0.3) data[0];",
            "qreg extra[1];",
            "cx data[0],data[0];",
            "t data[2];",
            "barrier data[0]\nreset data[0];",
        ):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                normalize_qasm(source + suffix)

    def test_qiskit_export_preserves_entangling_circuit_with_rotation_budget(self):
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(0.317, 1)
        circuit.rz(-0.219, 0)
        qasm, metadata = export_circuit(circuit, 1e-5)
        actual = Operator(qasm2.loads(qasm)).data
        expected = Operator(circuit).data
        overlap = np.trace(expected.conj().T @ actual)
        phase = overlap / abs(overlap)
        self.assertLessEqual(np.linalg.norm(actual - phase * expected, 2), 1e-5)
        self.assertLessEqual(metadata["measured_synthesis_error_bound"], 1e-5)
        self.assertTrue(
            set(metadata["gate_counts"])
            <= {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx", "cz"}
        )

    def test_custom_geometry_requires_correct_preprocessing(self):
        with self.assertRaises(ValueError):
            backend_options("compact", geometry_file=Path("layout.txt"))
        options = backend_options("edpc", 3, True, Path("layout.txt"))
        self.assertIn("--geometry-file", options)
        self.assertNotIn("--nostagger", options)
        self.assertEqual(options[options.index("--disttime") + 1], "3")

    def test_graceful_compiler_failure_cannot_become_a_successful_score(self):
        partial = subprocess.CompletedProcess(
            ["slicer"], 0, stdout="partial schedule", stderr="Routing failed"
        )
        with patch("estimate_sliced_resources.subprocess.run", return_value=partial):
            with self.assertRaisesRegex(RuntimeError, "Routing failed"):
                run_checked(["slicer"])


@unittest.skipUnless(
    (DEFAULT_CHECKOUT / ".deps/adapter/liblsqecc_adapter").is_file(),
    "Run setup_floorplanner.py for native integration tests",
)
class NativeLayoutTests(unittest.TestCase):
    def test_three_layouts_replay_and_use_identical_resource_assumptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "outputs"
            original = (ROOT / "tools/lattice_surgery/input_circuit.qasm").read_bytes()
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/lattice_surgery/run_layout_comparison.py"),
                    "--output-dir",
                    str(output),
                    "--budget",
                    "16",
                    "--seed",
                    "61",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(list(output.rglob("*compiler_report.txt")))
            summary = json.loads((output / "layout_comparison.json").read_text())
            reports = [
                json.loads((output / filename).read_text())
                for filename in (
                    "compact_resource_estimate.json",
                    "EDPC_resource_estimate.json",
                    "annealed_resource_estimate.json",
                )
            ]
            self.assertEqual(
                len({report["compiler"]["binary_sha256"] for report in reports}), 1
            )
            for report in reports:
                self.assertEqual(report["assumptions"], reports[0]["assumptions"])
                schedule = report["schedule"]
                physical = report["schedule_aware_estimate"]
                self.assertEqual(
                    schedule["num_slices"], schedule["sliced_lli_nonempty_lines"]
                )
                self.assertEqual(
                    physical["qec_rounds_core"], schedule["num_slices"] * 9
                )
                self.assertEqual(
                    physical["layout_physical_qubits"],
                    schedule["layout_tiles"] * 2 * 9**2,
                )
            winner = reports[-1]
            self.assertTrue(winner["optimization"]["winner_replay_verified"])
            self.assertEqual(
                winner["optimization"]["best_tile_slices"],
                winner["schedule"]["tile_slices"],
            )
            self.assertTrue(
                all(row["status"] == "success" for row in summary["layouts"].values())
            )
            self.assertEqual(
                (ROOT / "tools/lattice_surgery/input_circuit.qasm").read_bytes(),
                original,
            )
            # Independently check the mapped QASM under its recorded wire permutation.
            bundle = output / "annealed_layout"
            identity = json.loads((bundle / "identity.json").read_text())[
                "logical_to_compiler_index"
            ]
            target = qasm2.loads((bundle / "original-circuit.qasm").read_text())
            mapped = qasm2.loads((bundle / "mapped-circuit.qasm").read_text())
            expected = QuantumCircuit(target.num_qubits)
            expected.compose(target, qubits=identity, inplace=True)
            np.testing.assert_allclose(
                Operator(mapped).data, Operator(expected).data, atol=1e-12
            )

    def test_bad_input_fails_before_publishing_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            qasm = path / "bad.qasm"
            qasm.write_text("OPENQASM 2.0; qreg q[1]; rx(0.3) q[0];")
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/lattice_surgery/run_layout_comparison.py"),
                    "--qasm",
                    str(qasm),
                    "--output-dir",
                    str(path / "outputs"),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported statement", result.stderr)
            self.assertFalse(
                (path / "outputs/annealed_resource_estimate.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
