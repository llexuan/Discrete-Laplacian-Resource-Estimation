"""Optional editable Qiskit input; export with export_qiskit_qasm.py."""

from qiskit import QuantumCircuit


def build_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.t(0)
    return circuit
