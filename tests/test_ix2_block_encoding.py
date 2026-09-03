"""Tests for phase-metadata and matrix-polynomial helpers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.polynomial.chebyshev import chebval

from laplacian_qsvt.demos.ix2_block_encoding import (
    chebyshev_matrix_eval,
    load_phase_metadata,
)


class Ix2HelperTests(unittest.TestCase):
    def test_load_phase_metadata(self) -> None:
        payload = {
            "target_function": "matrix_inversion_1_over_x",
            "degree": 3,
            "effective_degree": 3,
            "parity": 1,
            "max_scale": 0.9,
            "phases": [0.1, 0.2, 0.3, 0.4],
            "cheb_coeffs": [0.0, 0.5, 0.0, -0.1],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_phase_metadata(path)

        self.assertEqual(loaded[:4], ("matrix_inversion_1_over_x", 3, 3, 1))
        self.assertEqual(loaded[4], payload["phases"])
        self.assertEqual(loaded[5], 0.9)
        np.testing.assert_allclose(loaded[6], payload["cheb_coeffs"])

    def test_chebyshev_matrix_eval_matches_eigenvalue_evaluation(self) -> None:
        matrix = np.array([[0.2, 0.1], [0.1, -0.4]], dtype=float)
        coefficients = np.array([0.3, -0.2, 0.5, 0.1], dtype=float)

        result = chebyshev_matrix_eval(coefficients, matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        expected = (
            eigenvectors
            @ np.diag(chebval(eigenvalues, coefficients))
            @ eigenvectors.conj().T
        )

        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
