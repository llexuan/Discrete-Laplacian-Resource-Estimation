# Lattice-surgery compiler outputs

This directory is reserved for the isolated compiler smoke test in
`tools/lattice_surgery/`.

The tracked outputs are:

- `compiler.lli`: layout-independent lattice-surgery instructions.
- `compiler_sliced.lli`: instructions grouped by scheduled timestep.
- `compiler_report.txt`: Pauli/Litinski transformations and exact commands.
- `slice_resource_estimate.json`: schedule-aware machine-readable resources.
- `slice_resource_report.txt`: readable timing and space report.

Each compiler run overwrites these files. The editable QASM input is tracked at
`tools/lattice_surgery/input_circuit.qasm`.

For a controlled layout comparison, the `edpc` and `compact` presets both use
wave scheduling, local compilation, twist-based S gates, `--disttime 1`,
`--nostagger`, no CNOT corrections, the graph-search router, and Dijkstra.
Only `-L edpc` versus `-L compact` differs.

EDPC LLI and resource report:

```shell
.venv/bin/python tools/lattice_surgery/run_lattice_surgery_compiler.py \
  --preset edpc

.venv/bin/python tools/lattice_surgery/estimate_sliced_resources.py \
  --preset edpc \
  --code-distance 9 \
  --cycle-time-ns 1000 \
  --output-json outputs/compiler/slice_resource_estimate.json \
  --output-report outputs/compiler/slice_resource_report.txt
```

Compact LLI and resource report:

```shell
.venv/bin/python tools/lattice_surgery/run_lattice_surgery_compiler.py \
  --preset compact \
  --lli outputs/compiler/compact.lli \
  --sliced-lli outputs/compiler/compact_sliced.lli \
  --report outputs/compiler/compact_compiler_report.txt

.venv/bin/python tools/lattice_surgery/estimate_sliced_resources.py \
  --preset compact \
  --code-distance 9 \
  --cycle-time-ns 1000 \
  --output-json outputs/compiler/compact_resource_estimate.json \
  --output-report outputs/compiler/compact_resource_report.txt
```

SLICER=../liblsqecc/build/lsqecc_slicer

# EDPC unsliced LLI
"$SLICER" -I qasm \
  -i tools/lattice_surgery/input_circuit.qasm \
  -L edpc -P wave --local \
  --disttime 1 --nostagger \
  --cnotcorrections never \
  -r graph_search -g djikstra \
  --printlli before --graceful \
  > outputs/compiler/compiler.lli

# EDPC sliced LLI
"$SLICER" -I qasm \
  -i tools/lattice_surgery/input_circuit.qasm \
  -L edpc -P wave --local \
  --disttime 1 --nostagger \
  --cnotcorrections never \
  -r graph_search -g djikstra \
  --printlli sliced --graceful \
  > outputs/compiler/compiler_sliced.lli

SLICER=../liblsqecc/build/lsqecc_slicer

# Compact unsliced LLI
"$SLICER" -I qasm \
  -i tools/lattice_surgery/input_circuit.qasm \
  -L compact -P wave --local \
  --disttime 1 --nostagger \
  --cnotcorrections never \
  -r graph_search -g djikstra \
  --printlli before --graceful \
  > outputs/compiler/compact.lli

# Compact sliced LLI
"$SLICER" -I qasm \
  -i tools/lattice_surgery/input_circuit.qasm \
  -L compact -P wave --local \
  --disttime 1 --nostagger \
  --cnotcorrections never \
  -r graph_search -g djikstra \
  --printlli sliced --graceful \
  > outputs/compiler/compact_sliced.lli