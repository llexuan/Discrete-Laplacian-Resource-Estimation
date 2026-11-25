"""Phase-sensitive checks for the conditioned pipeline's circuit compiler."""

import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from laplacian_qsvt.synthesis.circuit_synthesis import (
    arbitrary_rotation_count, compile_rotations, expand_to_rotations,
)


class CircuitSynthesisTests(unittest.TestCase):
    def test_controlled_rotation_preserves_relative_and_global_phases(self):
        circuit = QuantumCircuit(2)
        circuit.global_phase = 0.173
        circuit.h(0)
        circuit.cry(0.3841, 0, 1)
        circuit.crz(-0.291, 1, 0)
        expanded = expand_to_rotations(circuit)
        compiled, meta = compile_rotations(expanded, 1e-6)
        error = np.linalg.norm(Operator(compiled).data - Operator(circuit).data, 2)
        self.assertLessEqual(error, meta["summed_operator_error_bound"] + 1e-12)
        self.assertEqual(meta["arbitrary_rotations"], arbitrary_rotation_count(expanded))

    def test_exact_rotations_need_no_approximation(self):
        circuit = QuantumCircuit(1)
        for turns in (-11, -2, 0, 1, 4, 9):
            circuit.rz(turns * np.pi / 4, 0)
        compiled, meta = compile_rotations(circuit, 1e-6)
        np.testing.assert_allclose(Operator(compiled).data, Operator(circuit).data, atol=1e-13)
        self.assertEqual(meta["arbitrary_rotations"], 0)

    def test_borrowed_ancillas_work_for_every_input_state(self):
        circuit = QuantumCircuit(8)
        circuit.mcx(list(range(4)), 4)
        expanded = expand_to_rotations(circuit)
        compiled, _ = compile_rotations(expanded, 1e-6)
        np.testing.assert_allclose(Operator(compiled).data, Operator(circuit).data, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
