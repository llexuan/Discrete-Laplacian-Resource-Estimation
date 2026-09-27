# Lattice-surgery compiler outputs

This directory is reserved for the isolated compiler smoke test in
`tools/lattice_surgery/`.

The tracked outputs are:

- `compiler.lli`: layout-independent lattice-surgery instructions.
- `compiler_sliced.lli`: instructions grouped by scheduled timestep.
- `compiler_report.txt`: Pauli/Litinski transformations and exact commands.

Each compiler run overwrites these files. The editable QASM input is tracked at
`tools/lattice_surgery/input_circuit.qasm`.

.venv/bin/python \
  tools/lattice_surgery/run_lattice_surgery_compiler.py
