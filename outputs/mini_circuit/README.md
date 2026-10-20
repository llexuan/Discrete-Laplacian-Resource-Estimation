# Mini-circuit lattice-surgery results

Start with [comparison_report.txt](comparison_report.txt), then the annealed,
compact and EDPC TXT reports. These results concern the small QASM test circuit.
Full inversion results are under `../laplacian_inversion/2d_n20/`.

- `data/`: machine-readable comparison and resource estimates.
- `artifacts/`: unsliced/sliced LLI and the winning annealed layout.
- `runs/`: local compiler/search history (ignored by Git).

From the repository root, regenerate with:

```bash
PYTHONPATH=src .venv/bin/python tools/lattice_surgery/run_layout_comparison.py
```

The editable input remains `tools/lattice_surgery/input_circuit.py` / `.qasm`.
