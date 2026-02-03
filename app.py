from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ===== Default / fixed (kamu boleh ubah kalau mau dinamis) =====
FREQ_OPTIONS_GHZ = [1.8, 2.1, 2.3, 2.4, 3.3]

C = 3e8          # m/s
ER = 4.4         # FR-4 (umum)
H_MM = 1.6       # mm
Z0_OHM = 50.0    # ohm

PATCH_WIDTH_SCALE = 2.0  # scale patch width (visual requirement)

RETURN_LOSS_FILE = Path(__file__).with_name("return loss.xlsx")


def format_freq_key(value: float | str) -> str:
    """Normalize frequency (float/string) to canonical key used for graph data."""
    if value is None:
        return ""

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return stripped
        try:
            value = float(stripped)
        except ValueError:
            return stripped

    if isinstance(value, (int, float)):
        text_val = f"{float(value):.4f}".rstrip("0").rstrip(".")
        return text_val or str(value)

    return str(value)




def s11_db_to_vswr(db_value: float | int | str) -> float:
    """Convert nilai S11 / return loss (dB) menjadi VSWR."""
    try:
        value = float(db_value)
    except (TypeError, ValueError):
        return float("nan")

    # nilai negatif diasumsikan sudah S11 (dB); positif dianggap return loss (dB)
    if value <= 0:
        gamma = 10 ** (value / 20.0)
    else:
        gamma = 10 ** (-value / 20.0)

    gamma = min(abs(gamma), 0.999999)
    denominator = 1 - gamma
    if denominator <= 0:
        return float("inf")

    return (1 + gamma) / denominator


def load_return_loss_data(path: Path) -> dict[str, dict[str, list[float]]]:
    """Read Excel workbook and return chart data keyed by frequency string."""

    dataset: dict[str, dict[str, list[float]]] = {}

    if not path.exists():
        print(f"[return-loss] file '{path.name}' tidak ditemukan")
        return dataset

    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:  # pragma: no cover
        print(f"[return-loss] gagal membuka '{path.name}': {exc}")
        return dataset

    for sheet in workbook.sheet_names:
        try:
            df = pd.read_excel(workbook, sheet_name=sheet, header=None)
        except Exception as exc:  # pragma: no cover
            print(f"[return-loss] gagal membaca sheet '{sheet}': {exc}")
            continue

        df = df.iloc[:, :2].dropna()
        if df.empty:
            continue

        pairs = [
            (float(freq_val), float(rl_val))
            for freq_val, rl_val in zip(df.iloc[:, 0], df.iloc[:, 1])
        ]
        pairs.sort(key=lambda item: item[0])

        freq_axis = [freq for freq, _ in pairs]
        return_loss = [rl for _, rl in pairs]
        vswr = [s11_db_to_vswr(value) for value in return_loss]

        try:
            freq_numeric = float(sheet)
        except ValueError:
            freq_numeric = None

        freq_key = format_freq_key(freq_numeric if freq_numeric is not None else sheet)

        dataset[freq_key] = {
            'freq_key': freq_key,
            'freq_ghz': freq_numeric,
            'frequency_axis': freq_axis,
            'return_loss': return_loss,
            'vswr': vswr,
            'sheet': sheet,
        }

    return dataset


RETURN_LOSS_DATA = load_return_loss_data(RETURN_LOSS_FILE)


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


# =========================
# Rumus umum microstrip
# =========================

def eps_eff_hammerstad(er: float, w_h: float) -> float:
    """
    Effective permittivity microstrip (Hammerstad-Jensen approximation).
    w_h = W/h
    """
    if w_h <= 0:
        return float("nan")

    # base term
    ee = (er + 1) / 2 + (er - 1) / 2 * (1 / math.sqrt(1 + 12 / w_h))

    # correction for narrow lines (w_h < 1)
    if w_h < 1:
        ee += 0.04 * (1 - w_h) ** 2

    return ee


def microstrip_z0(er: float, w_h: float) -> float:
    """
    Characteristic impedance Z0 for microstrip line (common closed-form).
    Uses eps_eff_hammerstad(er, w_h).
    """
    ee = eps_eff_hammerstad(er, w_h)
    if w_h <= 0 or math.isnan(ee) or ee <= 0:
        return float("nan")

    if w_h <= 1:
        # narrow line
        return (60 / math.sqrt(ee)) * math.log(8 / w_h + 0.25 * w_h)
    else:
        # wide line
        return (120 * math.pi) / (math.sqrt(ee) * (w_h + 1.393 + 0.667 * math.log(w_h + 1.444)))


def solve_w_h_for_z0(er: float, z0: float, tol: float = 1e-6, max_iter: int = 120) -> float:
    """
    Solve W/h given target Z0 (ohm) using bisection.
    """
    # bounds for W/h (very small to very large)
    lo = 1e-6
    hi = 100.0

    # ensure function crosses target (Z0 decreases as W/h increases generally)
    z_lo = microstrip_z0(er, lo)
    z_hi = microstrip_z0(er, hi)

    # If bounds are weird, expand hi
    expand = 0
    while (math.isnan(z_lo) or math.isnan(z_hi) or z_lo < z0 or z_hi > z0) and expand < 20:
        # We want z_lo > z0 and z_hi < z0 ideally
        hi *= 2
        z_hi = microstrip_z0(er, hi)
        expand += 1

    # Bisection
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        z_mid = microstrip_z0(er, mid)

        if math.isnan(z_mid):
            # move a bit
            lo = mid
            continue

        if abs(z_mid - z0) < 1e-6:
            return mid

        # Z0 decreases as w_h increases
        if z_mid > z0:
            lo = mid
        else:
            hi = mid

        if (hi - lo) < tol:
            break

    return (lo + hi) / 2


# =========================
# Rumus umum patch antenna
# =========================

def patch_width(c: float, f_hz: float, er: float) -> float:
    """
    Patch width (rectangular) - formula umum:
      W = c / (2 f) * sqrt(2/(er + 1))
    """
    return (c / (2 * f_hz)) * math.sqrt(2 / (er + 1))


def patch_length(c: float, f_hz: float, er: float, h_m: float, w_m: float) -> tuple[float, float]:
    """
    Patch length (rectangular) - formula umum:
      eps_eff = (er+1)/2 + (er-1)/2 * 1/sqrt(1 + 12h/W)
      ΔL/h = 0.412 * ((eps_eff+0.3)(W/h+0.264))/((eps_eff-0.258)(W/h+0.8))
      Leff = c/(2f*sqrt(eps_eff))
      L = Leff - 2ΔL
    returns (L, eps_eff)
    """
    w_h = w_m / h_m
    eps_eff = (er + 1) / 2 + (er - 1) / 2 * (1 / math.sqrt(1 + 12 / w_h))

    delta_l = 0.412 * h_m * ((eps_eff + 0.3) * (w_h + 0.264)) / ((eps_eff - 0.258) * (w_h + 0.8))
    leff = c / (2 * f_hz * math.sqrt(eps_eff))
    L = leff - 2 * delta_l

    return L, eps_eff


def compute_all(freq_ghz: float) -> Results:
    f_hz = freq_ghz * 1e9
    h_m = H_MM * 1e-3

    # --- Patch W & L (rumus umum) ---
    wp_m = patch_width(C, f_hz, ER) * PATCH_WIDTH_SCALE
    lp_m, eps_patch = patch_length(C, f_hz, ER, h_m, wp_m)

    # --- Ground plane (umum: tambah 3h tiap sisi => +6h total) ---
    wg_m = wp_m + 6 * h_m
    lg_m = lp_m + 6 * h_m

    # --- Feedline width Wf for Z0 (rumus umum microstrip) ---
    w_h_line = solve_w_h_for_z0(ER, Z0_OHM)
    wf_m = w_h_line * h_m
    eps_line = eps_eff_hammerstad(ER, w_h_line)

    # --- Feedline length: quarter guided wavelength ---
    lambda_g = C / (f_hz * math.sqrt(eps_line))
    lf_m = 0.25 * lambda_g

    to_mm = 1e3
    return Results(
        freq_ghz=freq_ghz,
        wp_mm=wp_m * to_mm,
        lp_mm=lp_m * to_mm,
        wg_mm=wg_m * to_mm,
        lg_mm=lg_m * to_mm,
        wf_mm=wf_m * to_mm,
        lf_mm=lf_m * to_mm,
        eps_eff_patch=eps_patch,
        eps_eff_line=eps_line,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    results: Optional[Results] = None
    error: Optional[str] = None

    selected_freq = FREQ_OPTIONS_GHZ[0]

    if request.method == "POST":
        try:
            selected_freq = float(request.form.get("freq", str(FREQ_OPTIONS_GHZ[0])))
            if selected_freq not in FREQ_OPTIONS_GHZ:
                selected_freq = FREQ_OPTIONS_GHZ[0]
            results = compute_all(selected_freq)
        except Exception as e:
            error = f"Gagal menghitung: {e}"

    safe_freq = selected_freq if selected_freq in FREQ_OPTIONS_GHZ else FREQ_OPTIONS_GHZ[0]
    svg_results = results if results is not None else compute_all(safe_freq)
    show_svg_labels = results is not None
    selected_freq_key = format_freq_key(safe_freq)

    return render_template(
        "index.html",
        freq_options=FREQ_OPTIONS_GHZ,
        fixed=dict(h_mm=H_MM, z0=Z0_OHM, er=ER, c=C),
        selected_freq=selected_freq,
        results=results,
        svg_results=svg_results,
        show_svg_labels=show_svg_labels,
        error=error,
        return_loss_data=RETURN_LOSS_DATA,
        selected_freq_key=selected_freq_key,
    )


if __name__ == "__main__":
    app.run(debug=True)
