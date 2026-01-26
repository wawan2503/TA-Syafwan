from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from flask import Flask, render_template, request

app = Flask(__name__)

# === Fixed parameters from the assignment sheet ===
FREQ_OPTIONS_GHZ = [1.8, 2.1, 2.3, 2.4, 3.3]
H_MM = 1.6            # substrate thickness (mm)
Z0_OHM = 50.0         # input impedance (ohm)
ER = 4.4              # dielectric constant
C = 3e8               # speed of light (m/s)


@dataclass
class Results:
    freq_ghz: float
    wp_mm: float
    lp_mm: float
    wg_mm: float
    lg_mm: float
    wf_mm: float
    lf_mm: float
    eps_eff_patch: float
    eps_eff_line: float


def _eps_eff(er: float, h_m: float, w_m: float) -> float:
    """Effective dielectric constant (Hammerstad-style simple form)."""
    if w_m <= 0:
        return float("nan")
    return (er + 1) / 2 + (er - 1) / 2 * (1 / math.sqrt(1 + 12 * h_m / w_m))


def patch_dimensions(freq_hz: float, er: float, h_m: float, mode: str = "module") -> tuple[float, float, float]:
    """
    Returns (Wp, Lp, eps_eff_patch) in meters.

    mode:
      - "module": uses the Wp formula as written in the PDF (sqrt((er+1)/2))
      - "standard": uses the common textbook formula (sqrt(2/(er+1)))
    """
    if mode == "module":
        wp = (C / (2 * freq_hz)) * math.sqrt((er + 1) / 2)
    else:
        wp = (C / (2 * freq_hz)) * math.sqrt(2 / (er + 1))

    eps_eff = _eps_eff(er, h_m, wp)
    delta_l = 0.412 * h_m * ((eps_eff + 0.3) * (wp / h_m + 0.264)) / ((eps_eff - 0.258) * (wp / h_m + 0.8))
    leff = C / (2 * freq_hz * math.sqrt(eps_eff))
    lp = leff - 2 * delta_l
    return wp, lp, eps_eff


def microstrip_width_for_z0(z0: float, er: float, h_m: float) -> tuple[float, float]:
    """
    Returns (W, eps_eff_line) in meters for a microstrip line with characteristic impedance z0.
    Uses the B-formula shown in the PDF (and the usual A-case for W/h <= 2).
    """
    A = z0 / 60 * math.sqrt((er + 1) / 2) + ((er - 1) / (er + 1)) * (0.23 + 0.11 / er)
    wh = (8 * math.exp(A)) / (math.exp(2 * A) - 2)

    if wh > 2:
        B = 377 * math.pi / (2 * z0 * math.sqrt(er))
        wh = (2 / math.pi) * (
            B - 1 - math.log(2 * B - 1)
            + ((er - 1) / (2 * er)) * (math.log(B - 1) + 0.39 - 0.61 / er)
        )

    w = wh * h_m
    eps_eff_line = _eps_eff(er, h_m, w)
    return w, eps_eff_line


def compute_all(freq_ghz: float, mode: str = "module") -> Results:
    freq_hz = freq_ghz * 1e9
    h_m = H_MM * 1e-3

    wp_m, lp_m, eps_eff_patch = patch_dimensions(freq_hz, ER, h_m, mode=mode)

    # Ground plane
    wg_m = 6 * h_m + wp_m
    lg_m = 6 * h_m + lp_m

    # Feed line width (50 ohm microstrip)
    wf_m, eps_eff_line = microstrip_width_for_z0(Z0_OHM, ER, h_m)

    # Feed line length (quarter guided wavelength)
    lambda_g = C / (freq_hz * math.sqrt(eps_eff_line))
    lf_m = lambda_g / 4

    to_mm = 1e3
    return Results(
        freq_ghz=freq_ghz,
        wp_mm=wp_m * to_mm,
        lp_mm=lp_m * to_mm,
        wg_mm=wg_m * to_mm,
        lg_mm=lg_m * to_mm,
        wf_mm=wf_m * to_mm,
        lf_mm=lf_m * to_mm,
        eps_eff_patch=eps_eff_patch,
        eps_eff_line=eps_eff_line,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    selected_freq = request.form.get("freq", str(FREQ_OPTIONS_GHZ[0]))
    mode = request.form.get("mode", "module")  # module | standard

    try:
        freq = float(selected_freq)
        if freq not in FREQ_OPTIONS_GHZ:
            freq = FREQ_OPTIONS_GHZ[0]
    except ValueError:
        freq = FREQ_OPTIONS_GHZ[0]

    results: Optional[Results] = None
    svg_results: Optional[Results] = None
    error: Optional[str] = None

    if request.method == "POST":
        try:
            results = compute_all(freq, mode=mode)
            svg_results = results
        except Exception as e:
            error = f"Gagal menghitung: {e}"

    return render_template(
        "index.html",
        freq_options=FREQ_OPTIONS_GHZ,
        fixed=dict(h_mm=H_MM, z0=Z0_OHM, er=ER, c=C),
        selected_freq=freq,
        mode=mode,
        results=results,
        svg_results=svg_results,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
