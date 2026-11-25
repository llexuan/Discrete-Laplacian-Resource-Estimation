"""Circuit-semantic checks for folded frequencies and coherent QSVT extraction."""
from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
from numpy.polynomial import chebyshev as C
from pyqsp import angle_sequence
from qiskit.quantum_info import Operator

from laplacian_qsvt.laplacian.preconditioned_1d import (
    build_folded_frequency_block_encoding,
    build_preconditioned_block_encoding,
    build_real_polynomial_block_encoding,
    build_signed_projector_rotation,
    frequency_sector_circuit,
    make_inner_polynomial,
    verify_conditioned_inverse,
)


class Preconditioned1DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inner = make_inner_polynomial()
        cls.outer_coefficients = [0.0, 0.3, 0.0, 0.2]
        with contextlib.redirect_stdout(io.StringIO()):
            cls.outer_phases = angle_sequence.QuantumSignalProcessingPhases(
                cls.outer_coefficients, method="sym_qsp", chebyshev_basis=True)[0].tolist()

    def test_both_frequency_branches_and_padding(self) -> None:
        # q=2 has only one active selector and cannot catch the multi-flag bug.
        for q in (3, 4):
            circuit = build_folded_frequency_block_encoding(q)
            m = circuit.metadata["ancilla_count"]
            matrix = Operator(circuit).data
            indices = np.arange(1 << q) << m
            folded = 2 * np.minimum(np.arange(1 << q), (1 << q) - np.arange(1 << q)) / (1 << q)
            np.testing.assert_allclose(matrix[np.ix_(indices, indices)],
                                       np.diag(folded), rtol=0, atol=1e-12)
            np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0, atol=1e-12)
            # The optimization used in verification agrees with actual gates.
            for k in range(1 << q):
                sector = Operator(frequency_sector_circuit(circuit, q, k)).data
                full_indices = (k << m) + np.arange(1 << m)
                np.testing.assert_allclose(sector, matrix[np.ix_(full_indices, full_indices)],
                                           rtol=0, atol=1e-12)

    def test_polynomial_error_bound_and_normalization(self) -> None:
        x = np.linspace(-1, 1, 10001)
        target = np.sinc(x / 2) ** 2
        unscaled = C.chebval(x, self.inner["cheb_coeffs"]) * self.inner["normalization"]
        error = target - unscaled
        self.assertGreaterEqual(float(np.min(error)), -1e-14)
        self.assertLessEqual(float(np.max(error)), self.inner["uniform_polynomial_error_bound_unscaled"])
        self.assertGreater(self.inner["encoded_spectral_lower_bound"], 0.4)
        self.assertEqual(self.inner["degree"], 14)

    def test_signed_phase_includes_branch_phase_and_cleans_predicate(self) -> None:
        phi, m = 0.371, 3
        matrix = Operator(build_signed_projector_rotation(phi, m, 0)).data
        for ancilla in range(1 << m):
            for extraction in (0, 1):
                index = (ancilla << 2) | extraction
                desired = np.exp(1j * phi * (1 - 2 * extraction) * (1 if ancilla == 0 else -1))
                expected = np.zeros(len(matrix), dtype=complex)
                expected[index] = desired
                np.testing.assert_allclose(matrix[:, index], expected, rtol=0, atol=1e-12)

    def test_conditioned_unitary_is_real_hermitian_and_nested_response_is_correct(self) -> None:
        for q in (3, 4):
            checks = verify_conditioned_inverse(q, self.inner, self.outer_phases,
                                                self.outer_coefficients)
            self.assertTrue(checks["passed"], checks)
            self.assertTrue(checks["coherent_real_extraction_verified"])

    def test_full_outer_circuit_matches_polynomial_without_numerical_quadrature(self) -> None:
        q = 2
        signal = build_preconditioned_block_encoding(q, self.inner)
        m = signal.metadata["ancilla_count"]
        outer = build_real_polynomial_block_encoding(signal, m, self.outer_phases)
        matrix = Operator(outer).data
        indices = np.arange(1 << q) << (m + 2)
        theta = np.array([0, 0.5, 1, 0.5])
        target = C.chebval(C.chebval(theta, self.inner["cheb_coeffs"]), self.outer_coefficients)
        np.testing.assert_allclose(matrix[np.ix_(indices, indices)], np.diag(target),
                                   rtol=0, atol=1e-11)
        np.testing.assert_allclose(matrix.imag, 0, rtol=0, atol=1e-11)
        np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
