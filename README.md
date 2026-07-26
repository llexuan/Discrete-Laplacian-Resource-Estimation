# One-Qubit Phase Rotation (pyqsp)

This folder contains a script that synthesizes QSP phase angles for a one-qubit
sequence so that the resulting polynomial response approximates:

\[
P(a) \approx f(a), \quad a \in [0,1].
\]

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python one_qubit_phase_rotations.py --degree 12 "cos(x)"
python one_qubit_phase_rotations.py --degree 20 "exp(-x**2)"
```

Outputs:
- `phases_target.json`: synthesized phase list (radians) and metadata.
- `qsp_target_fit.png`: verification plot of QSP response vs scaled target.
