"""Regression checks for the full geometric inverse and its small endpoint phases."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from laplacian_qsvt.phases.inversion_phase_rotations import main, unfiltered_bounds
from laplacian_qsvt.synthesis.rotation_synthesis import synthesize_rz


class UnfilteredInverseTests(unittest.TestCase):
    def test_degree_45_phase_file_and_analytic_certificate(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "phases.json"
            args = [
                "phases", "--unfiltered-b", "23", "--kappa", "2.5", "--epsilon", "0.01",
                "--output-phases", str(output), "--output-plot", str(output.with_suffix(".png")),
            ]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                main()
            phases = json.loads(output.read_text())

        self.assertFalse(phases["use_rect_filter"])
        self.assertEqual(phases["effective_degree"], 45)
        self.assertEqual(len(phases["phases"]), 46)
        self.assertEqual(len(set(phases["phases"])), 23)
        self.assertLess(phases["phase_vs_poly_max_abs_error"], 1e-12)
        x = np.array([-1.0, -0.4, -0.2, -1e-9, 0.0, 0.2, 0.4, 1.0])
        geometric = x / 5 * sum((1 - x**2) ** j for j in range(23))
        actual = np.polynomial.chebyshev.chebval(x, phases["cheb_coeffs"])
        np.testing.assert_allclose(actual, geometric, rtol=0.0, atol=1e-13)
        bounds = phases["unfiltered_inverse"]
        self.assertEqual((bounds["b_min"], bounds["b_max"]), (23, 25))
        self.assertAlmostEqual(bounds["uniform_inverse_error_bound"], 0.5 * 0.84**23)
        self.assertLess(bounds["global_magnitude_bound"], 1)

    def test_rejects_exponents_outside_certified_interval(self):
        for b in [22, 26]:
            with self.subTest(b=b), self.assertRaisesRegex(ValueError, "certified interval"):
                unfiltered_bounds(b, 2.5, 0.01)

    def test_below_tolerance_phase_retains_its_error(self):
        result = synthesize_rz(-5.7e-13, 1e-12)
        self.assertEqual(result.gate_counts, {})
        self.assertAlmostEqual(result.projective_error, 2.85e-13, delta=1e-25)

    def test_small_resolved_phase_compiles_within_budget(self):
        result = synthesize_rz(-1.99478088121983e-10, 1e-12)
        self.assertGreater(result.t_count, 0)
        self.assertLessEqual(result.projective_error, 1e-12)
