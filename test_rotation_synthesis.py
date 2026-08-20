"""Regression tests for pyLIQTR-backed Clifford+T rotation synthesis."""

from __future__ import annotations

import unittest

import numpy as np
from qiskit.quantum_info import Operator

from laplacian_nd_resource import (
    build_prep_selector,
    estimate_resources,
    selector_rotation_angle_3d,
    verify_block_encoding_nd,
)
from laplacian_w_resource import verify_w_signal_block
from rotation_synthesis import synthesize_ry


class RotationSynthesisTests(unittest.TestCase):
    def test_ry_is_explicit_clifford_t_with_requested_error(self) -> None:
        epsilon = 1e-6
        result = synthesize_ry(selector_rotation_angle_3d(), epsilon)

        self.assertLessEqual(result.projective_error, epsilon)
        self.assertGreater(result.t_count, 0)
        self.assertTrue(
            set(result.gate_counts).issubset(
                {"h", "t", "tdg", "s", "sdg", "x", "y", "z"}
            )
        )

    def test_generated_inverse_is_exact(self) -> None:
        result = synthesize_ry(selector_rotation_angle_3d(), 1e-6)
        forward = Operator(result.circuit).data
        inverse = Operator(result.circuit.inverse()).data

        np.testing.assert_allclose(
            inverse @ forward,
            np.eye(2, dtype=complex),
            rtol=0.0,
            atol=1e-12,
        )

    def test_3d_prep_contains_no_arbitrary_rotation_after_synthesis(self) -> None:
        prep = build_prep_selector(3, synthesis_eps=1e-6)
        operation_names = {instr.operation.name for instr in prep.data}

        self.assertTrue({"rx", "ry", "rz"}.isdisjoint(operation_names))

    def test_full_resource_count_includes_emitted_synthesis(self) -> None:
        estimate = estimate_resources(1, 3, 1e-6)
        synthesis = estimate["rotation_synthesis"]

        self.assertEqual(
            synthesis["synthesized_t"],
            estimate["n_arbitrary_rotations"] * synthesis["t_per_rotation"],
        )
        self.assertEqual(
            estimate["t_count"],
            estimate["gate_counts"].get("t", 0)
            + estimate["gate_counts"].get("tdg", 0),
        )

    def test_synthesized_block_encoding_and_walk_remain_accurate(self) -> None:
        epsilon = 1e-6
        block_error = verify_block_encoding_nd(1, 3, synthesis_eps=epsilon)
        signal_error, unitarity_error = verify_w_signal_block(
            1, 3, synthesis_eps=epsilon
        )

        self.assertLessEqual(block_error, 2 * epsilon)
        self.assertLessEqual(signal_error, 4 * epsilon)
        self.assertLessEqual(unitarity_error, 1e-11)


if __name__ == "__main__":
    unittest.main()
