"""Checks for the nested block encoding of the 1D preconditioned Laplacian."""

from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/lattice_surgery"))

from laplacian_qsvt.laplacian.laplacian_nd_resource import CT_BASIS
from laplacian_qsvt.preconditioned.gates import (
    RotationEmitter,
    append_mcx,
    append_multiplexed_ry,
    state_preparation_tree,
)
from laplacian_qsvt.preconditioned.inversion import (
    dense_checks,
    estimate_logical_resources,
    walk_matrix,
)
from laplacian_qsvt.preconditioned.sinc2 import (
    minimum_odd_d,
    p2d,
    sinc2_phases,
    sinc2_target,
    truncation_bound,
)
from laplacian_qsvt.preconditioned.u_x import (
    ancilla_zero_isometry,
    build_prep,
    build_u_x,
    draw_prep_select,
    frequency_of_system_states,
    prep_state_error,
    walk_layout,
    x_hat_values,
)
from laplacian_qsvt.preconditioned.walk import build_walk

OUTER = json.loads((ROOT / "outputs/phases/phases_inversion.json").read_text())


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class PrimitiveTests(unittest.TestCase):
    def test_multiplexed_ry_applies_each_branch_angle(self):
        angles = [0.3, -1.1, 2.0, 0.7]
        qc = QuantumCircuit(3)
        append_multiplexed_ry(qc, RotationEmitter(), 0, [1, 2], angles)
        matrix = Operator(qc).data
        for branch, angle in enumerate(angles):
            column = matrix[:, branch << 1]
            expected = np.zeros(8, dtype=complex)
            expected[branch << 1] = math.cos(angle / 2)
            expected[(branch << 1) | 1] = math.sin(angle / 2)
            np.testing.assert_allclose(column, expected, atol=1e-12)

    def test_pruned_tree_prepares_weights_with_zero_branches(self):
        weights = [0.1, 0.0, 0.25, 0.05, 0.3, 0.3, 0.0, 0.0]
        qc = QuantumCircuit(3)
        emitter = RotationEmitter()
        for bit, controls, angles in state_preparation_tree(weights, 3):
            append_multiplexed_ry(qc, emitter, bit, controls, angles)
        np.testing.assert_allclose(
            np.abs(Statevector(qc).data) ** 2, weights, atol=1e-12
        )

    def test_mcx_with_open_controls_and_dirty_qubits_is_exact(self):
        for k in (3, 4, 5):
            qc = QuantumCircuit(2 * k)
            states = [1, 0] * k
            append_mcx(qc, list(range(k)), k, list(range(k + 1, 2 * k)), states[:k])
            reference = QuantumCircuit(2 * k)
            append_mcx(reference, list(range(k)), k, [], states[:k], logical=True)
            np.testing.assert_allclose(
                Operator(qc).data, Operator(reference).data, atol=1e-10
            )
            compiled = transpile(qc, basis_gates=CT_BASIS, optimization_level=0)
            self.assertTrue(set(compiled.count_ops()) <= set(CT_BASIS))


class BlockEncodingTests(unittest.TestCase):
    def test_u_x_is_hermitian_and_encodes_two_m_over_n(self):
        for n in (3, 4, 5):
            for order in ("natural", "bit_reversed"):
                lay = walk_layout(n, order)
                emitter = RotationEmitter()
                prep = build_prep(lay, emitter)
                self.assertLess(prep_state_error(lay, Operator(prep).data), 1e-12)
                u_x = Operator(build_u_x(lay, emitter, prep=prep)).data
                embed = ancilla_zero_isometry(lay)
                theta = x_hat_values(n)[frequency_of_system_states(lay)]
                np.testing.assert_allclose(
                    embed.conj().T @ u_x @ embed, np.diag(theta), atol=1e-12
                )
                np.testing.assert_allclose(u_x, u_x.conj().T, atol=1e-12)

    def test_prep_and_select_drawings_name_every_register(self):
        drawings = draw_prep_select(4)
        for token in ("a:", "p:", "J_0:", "J_1:", "k_3:"):
            self.assertIn(token, drawings["SELECT"])
        self.assertIn("p:", drawings["PREP"])
        controls = [
            getattr(instruction.operation, "num_ctrl_qubits", 0)
            for instruction in drawings["select_circuit"].data
        ]
        self.assertEqual(controls.count(4), 3)

    def test_sinc2_polynomial_is_bounded_and_within_the_taylor_bound(self):
        with quiet():
            meta = sinc2_phases(5)
        theta = np.linspace(-1, 1, 2001)
        self.assertLessEqual(np.max(np.abs(p2d(theta, 5) - sinc2_target(theta))), truncation_bound(5))
        self.assertLessEqual(meta["max_abs_poly_on_domain"], 1.0 + 1e-15)
        self.assertGreater(meta["minimum_on_unit_interval"], 0.4)
        self.assertTrue(meta["phases_symmetric"])
        self.assertLess(meta["phase_vs_poly_max_abs_error"], 1e-12)
        self.assertEqual(minimum_odd_d(2.2e-5), 5)

    def test_nested_sequence_matches_classical_conditioned_inverse(self):
        with quiet():
            meta = sinc2_phases(5)
        result = dense_checks(
            3, meta["psi"], 5, OUTER["phases"], OUTER["cheb_coeffs"], OUTER["parity"], OUTER["kappa"]
        )
        for key in (
            "u_x_block_error", "u_lcond_hermitian_part_error", "walk_signal_block_error",
            "u_prime_adjoint_branch_error", "qft_bit_reversed_dft_error",
            "qsp_block_max_entry_error", "position_vs_classical_polynomial_error",
        ):
            self.assertLess(result[key], 1e-11, key)
        self.assertLess(result["position_vs_exact_conditioned_kernel_error"], 0.02)

    def test_clifford_t_walk_stays_within_its_synthesis_bound(self):
        with quiet():
            psi = sinc2_phases(5)["psi"]
        lay = walk_layout(3)
        emitter = RotationEmitter(1e-5)
        walk, _ = build_walk(lay, psi, emitter)
        compiled = transpile(walk, basis_gates=CT_BASIS, optimization_level=0)
        self.assertTrue(set(compiled.count_ops()) <= set(CT_BASIS))
        bound = sum(r["projective_error"] for r in emitter.records)
        error = np.linalg.norm(Operator(walk).data - walk_matrix(lay, psi), 2)
        self.assertLessEqual(error, bound + 1e-10)


class ResourceCompositionTests(unittest.TestCase):
    def test_totals_are_component_sums_and_export_matches(self):
        from export_inversion_components import export_components

        with quiet():
            meta = sinc2_phases(5)
        phases = [0.13, -0.24, 0.13]
        logical = estimate_logical_resources(4, meta["psi"], phases, 1e-4)
        walk = logical["walk_contribution"]["per_w"]
        boundary = logical["boundary_contribution"]
        projectors = logical["projector_rotation_contribution"]
        self.assertEqual(
            logical["t_count"],
            2 * walk["t_count"] + projectors["t_count"] + boundary["t_count"],
        )
        self.assertEqual(logical["logical_qubits"], walk["logical_qubits"])
        self.assertLessEqual(logical["measured_synthesis_error_bound"], 1e-4)
        metrics = {
            "case": "preconditioned_1d_conditioned_kernel",
            "case_title": "test",
            "logical_resources": logical | {"minimum_kappa_for_full_nonzero_spectrum": 2.47},
        }
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "logical.json").write_text(json.dumps(metrics))
            (directory / "phases.json").write_text(json.dumps({"phases": phases}))
            with quiet():
                manifest = export_components(
                    directory / "logical.json", directory / "phases.json", directory / "out"
                )
        kinds = [c["kind"] for c in manifest["components"]]
        self.assertEqual(kinds, ["walk", "projector", "projector", "boundary", "boundary"])
        self.assertEqual(manifest["logical_gate_counts"], logical["gate_counts"])


if __name__ == "__main__":
    unittest.main()
