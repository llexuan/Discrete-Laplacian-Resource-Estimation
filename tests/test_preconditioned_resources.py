"""Accounting and basis-boundary checks for the conditioned inverse kernel."""

import unittest

import numpy as np
from qiskit.quantum_info import Operator

from laplacian_qsvt.laplacian.preconditioned_inversion_resource import (
    add_fourier_boundaries, weighted_resources,
)


class ConditionedResourceTests(unittest.TestCase):
    def test_boundaries_return_system_to_position_basis(self):
        components = add_fourier_boundaries([], [2, 3, 4], 5)
        forward = Operator(components[0]["circuit"]).data
        backward = Operator(components[1]["circuit"]).data
        np.testing.assert_allclose(backward @ forward, np.eye(32), atol=1e-12)
        self.assertEqual([c["occurrences"] for c in components], [1, 1])

    def test_weighted_accounting_includes_boundaries_once(self):
        records = [
            {"kind": "walk", "occurrences": 117, "gate_counts": {"t": 2, "cx": 3}, "total_depth": 4, "t_depth": 2},
            {"kind": "projector", "occurrences": 118, "gate_counts": {"tdg": 1}, "total_depth": 1, "t_depth": 1},
            {"kind": "boundary", "occurrences": 1, "gate_counts": {"h": 5}, "total_depth": 2, "t_depth": 0},
        ]
        counts = weighted_resources(records)
        self.assertEqual(counts["t_count"], 352)
        self.assertEqual(counts["clifford_gate_count"], 356)
        self.assertEqual(counts["total_depth"], 588)


if __name__ == "__main__":
    unittest.main()
