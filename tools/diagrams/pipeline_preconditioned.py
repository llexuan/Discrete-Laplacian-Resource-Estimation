"""Write the preconditioned-1D pipeline overview diagram as SVG.

Inline markup: ``_{...}`` renders a subscript and ``^{...}`` a superscript.
"""

import re
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[2] / "outputs" / "plots" / "pipeline_preconditioned_1d.svg"

W, H = 1100, 700
BOX_W, BOX_H = 318, 116
COLS = [20, 391, 762]
ROWS = [118, 272, 426]

INK = "#12324a"
BODY = "#2b3c4d"
TEAL = "#0f8a7a"
BLUE = ("#eaf2f9", "#b7c9d9")
GREEN = ("#e7f4ef", "#b5d3c7")
ARROW = "#4f6474"

BOXES = [
    (0, 0, BLUE, "1  Define the operator", [
        "1D: q system qubits, N = 2^{q} grid points",
        "L_{cond} eigenvalues sinc^{2}(πm_{k}/N), m_{k} = min(k, N−k)",
        "Spectrum in [4/π^{2}, 1]; κ ≤ 2.47 for every N",
    ]),
    (0, 1, BLUE, "2  Build the block encoding", [
        "QFT moves the register to the frequency basis |k⟩",
        "U_{X} = PREP^{†} · SELECT · PREP → diag(2m_{k}/N)",
        "Shared ancilla: 2 + ⌈log_{2}(q−1)⌉ ancillas, α = 1",
    ]),
    (0, 2, BLUE, "3  Construct the walk W", [
        "Nested: inner sinc^{2} QSVT on U_{X} gives U_{Lcond}",
        "Direct: walk built on U_{X} itself",
        "Signal projector covers PREP / SELECT ancillas",
    ]),
    (1, 0, GREEN, "4  Choose the inverse target", [
        "Nested: existing P_{MI} (κ = 2.5) applied to L_{cond}",
        "Direct: h(θ) = θ^{2}/sin^{2}(πθ/2) = (4/π^{2}) L_{cond}^{−1}",
        "Zero mode excluded; all nonzero modes covered",
    ]),
    (1, 1, GREEN, "5  Generate QSP phases", [
        "Nested: 118 P_{MI} phases + inner sinc^{2} phases",
        "Direct: one even polynomial, degree ≈ 10–20",
        "Polynomial and synthesis errors budgeted together",
    ]),
    (1, 2, BLUE, "6  Compose the inverse sequence", [
        "QFT → R(φ_{0}) → W → … → W → R(φ_{r}) → QFT^{†}",
        "r ≈ 117 × deg(sinc^{2}) nested; r ≈ 10–20 direct",
        "QFT charged once, at the sequence boundaries",
    ]),
    (2, 0, GREEN, "Verification at small N", [
        "N = 8, 16: U_{X} signal block = diag(2m_{k}/N)",
        "Polynomial response vs sinc^{2} or h(θ)",
        "Compare with the classical conditioned inverse",
    ]),
    (2, 1, BLUE, "8  Report logical resources", [
        "Logical qubits; Clifford count; T-count; depths",
        "Labeled: 1D conditioned kernel, nonzero modes",
        "QFT / U_{X} / projector breakdown; old L̃ as baseline",
    ]),
    (2, 2, BLUE, "7  Compile to Clifford+T", [
        "QFT controlled phases and PREP R_{y} rotations",
        "SELECT: J-controlled CNOTs (no shift MCX gates)",
        "pyLIQTR synthesis of every arbitrary rotation",
    ]),
]

MARKUP = re.compile(r"([_^])\{([^}]*)\}")


def rich(text, size):
    """Convert inline sub/superscript markup into SVG tspans."""
    small = round(size * 0.72, 1)
    out, pos = [], 0
    for m in MARKUP.finditer(text):
        out.append(escape(text[pos:m.start()]))
        shift = 0.28 * size if m.group(1) == "_" else -0.38 * size
        out.append(
            f'<tspan dy="{shift:.1f}" font-size="{small}">{escape(m.group(2))}</tspan>'
            f'<tspan dy="{-shift:.1f}">\u200b</tspan>'
        )
        pos = m.end()
    out.append(escape(text[pos:]))
    return "".join(out)


def text(x, y, s, size, color, weight="normal", extra=""):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
        f'font-weight="{weight}" {extra}>{rich(s, size)}</text>'
    )


def box(row, col, colors, title, lines):
    x, y = COLS[col], ROWS[row]
    fill, stroke = colors
    parts = [
        f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="7" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>',
        text(x + 16, y + 28, title, 16, INK, "bold"),
    ]
    for i, line in enumerate(lines):
        parts.append(text(x + 16, y + 56 + i * 20, line, 12.5, BODY))
    return "\n".join(parts)


def arrow(x1, y1, x2, y2):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ARROW}" '
        f'stroke-width="1.5" marker-end="url(#head)"/>'
    )


def build():
    mid = lambda r: ROWS[r] + BOX_H / 2
    cx = COLS[2] + BOX_W / 2
    arrows = [
        arrow(COLS[0] + BOX_W + 6, mid(0), COLS[1] - 8, mid(0)),
        arrow(COLS[1] + BOX_W + 6, mid(0), COLS[2] - 8, mid(0)),
        arrow(cx, ROWS[0] + BOX_H + 6, cx, ROWS[1] - 8),
        arrow(COLS[0] + BOX_W + 6, mid(1), COLS[1] - 8, mid(1)),
        arrow(COLS[1] + BOX_W + 6, mid(1), COLS[2] - 8, mid(1)),
        arrow(cx, ROWS[1] + BOX_H + 6, cx, ROWS[2] - 8),
        arrow(COLS[2] - 6, mid(2), COLS[1] + BOX_W + 8, mid(2)),
    ]

    by = ROWS[2] + BOX_H + 36
    bottom = [
        f'<rect x="{COLS[0]}" y="{by}" width="{COLS[1] + BOX_W - COLS[0] - 80}" height="64" rx="7" '
        f'fill="#f3f6f9" stroke="#b7c9d9" stroke-width="1.4" stroke-dasharray="6 4"/>',
        text(COLS[0] + 16, by + 26, "Lattice-surgery architecture workflow", 16, INK, "bold"),
        text(COLS[0] + 16, by + 48,
             "New QFT, U_{X} and projector components → compact / EDPC / annealed layouts → qubits, runtime",
             12.5, BODY),
        text(COLS[2] - 40, by + 18, "Not included in the conditioned-kernel estimate:", 14, INK, "bold"),
        text(COLS[2] - 40, by + 40, "Preparing C^{−1}b (carries the original κ ≈ 10^{11}),", 12.5, "#5b6b78"),
        text(COLS[2] - 40, by + 58, "general |b⟩ preparation, output readout, 2D / 3D extension.", 12.5, "#5b6b78"),
    ]

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"
     font-family="DejaVu Sans, Helvetica Neue, Arial, sans-serif">
<defs>
  <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
    <path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/>
  </marker>
</defs>
<rect width="{W}" height="{H}" fill="#ffffff"/>
{text(20, 30, "DISCRETE LAPLACIAN / QSVT RESOURCE ESTIMATION — PRECONDITIONED UPDATE", 12.5, TEAL, "bold", 'letter-spacing="0.6"')}
{text(20, 68, "Preconditioned 1D Laplacian: frequency register to gate costs", 29, INK, "bold")}
{text(20, 96, "The block encoding U_{X} feeds either a nested sinc^{2} + P_{MI} sequence or one direct inverse polynomial (pending PI decision).", 13.5, "#5b6b78")}
{chr(10).join(box(*b) for b in BOXES)}
{chr(10).join(arrows)}
{chr(10).join(bottom)}
<line x1="20" y1="{H - 14}" x2="{W - 20}" y2="{H - 14}" stroke="#c9d6e0" stroke-width="1"/>
</svg>
"""


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    svg = build()
    OUT.write_text(svg, encoding="utf-8")
    # Quick Look thumbnails are square; a padded copy renders without cropping.
    pad = (W - H) // 2
    square = svg.replace(
        f'height="{H}" viewBox="0 0 {W} {H}"',
        f'height="{W}" viewBox="0 {-pad} {W} {W}"', 1,
    ).replace(f'<rect width="{W}" height="{H}"', f'<rect y="{-pad}" width="{W}" height="{W}"', 1)
    OUT.with_name(OUT.stem + "_square.svg").write_text(square, encoding="utf-8")
    print(OUT)
