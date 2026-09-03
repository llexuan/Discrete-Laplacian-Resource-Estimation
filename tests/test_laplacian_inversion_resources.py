"""Tests for full-QSP projector synthesis and compositional resources."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit.quantum_info import Operator

from laplacian_qsvt.laplacian.laplacian_inversion_encode import (
    build_projector_rotation,
    build_projector_rotation_clifford_t,
    estimate_full_qsp_resources,
)


class LaplacianInversionResourceTests(unittest.TestCase):
    def test_projector_rotation_matches_dense_reference(self) -> None:
        phi = 0.312
        epsilon = 1e-6
        m = 3
        synthesized, synthesis = build_projector_rotation_clifford_t(
            phi, m, epsilon
        )

        full = Operator(synthesized).data
        signal_dimension = 2 ** (1 + m)
        dirty_dimension = 2 ** (m - 1)
        flag_zero_indices = [
            signal + (dirty << (m + 2))
            for dirty in range(dirty_dimension)
            for signal in range(signal_dimension)
        ]
        flag_zero_block = full[np.ix_(flag_zero_indices, flag_zero_indices)]
        target = Operator(build_projector_rotation(phi, m)).data
        target_with_dirty = np.kron(np.eye(dirty_dimension), target)

        self.assertLessEqual(synthesis.projective_error, epsilon)
        np.testing.assert_allclose(
            flag_zero_block, target_with_dirty, rtol=0.0, atol=epsilon
        )

    def test_full_qsp_resource_totals_include_walks_and_rotations(self) -> None:
        phases = [0.13, -0.24, 0.31]
        estimate = estimate_full_qsp_resources(
            phases, dims=2, target_n=1, synthesis_tol=1e-5
        )
        walk = estimate["walk_contribution"]
        rotations = estimate["projector_rotation_contribution"]

        self.assertEqual(estimate["num_w_applications"], 2)
        self.assertEqual(estimate["num_projector_rotations"], 3)
        self.assertEqual(estimate["logical_qubits"], 7)
        self.assertEqual(
            estimate["t_count"], walk["t_count"] + rotations["t_count"]
        )
        self.assertEqual(
            estimate["clifford_gate_count"],
            walk["clifford_gate_count"] + rotations["clifford_gate_count"],
        )
        self.assertLessEqual(
            estimate["measured_synthesis_error_bound"],
            estimate["synthesis_tolerance"],
        )

    def test_3d_budget_includes_every_walk_selector_rotation(self) -> None:
        phases = [0.13, -0.24, 0.31]
        estimate = estimate_full_qsp_resources(
            phases, dims=3, target_n=1, synthesis_tol=1e-5
        )

        self.assertEqual(estimate["num_w_applications"], 2)
        self.assertEqual(estimate["num_w_selector_rotations"], 8)
        self.assertEqual(estimate["num_approximated_rotations"], 11)
        self.assertEqual(estimate["logical_qubits"], 9)
        self.assertIsNotNone(
            estimate["walk_contribution"]["per_w"]["rotation_synthesis"]
        )
        self.assertLessEqual(
            estimate["measured_synthesis_error_bound"],
            estimate["synthesis_tolerance"],
        )


if __name__ == "__main__":
    unittest.main()
