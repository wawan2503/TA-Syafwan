from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

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

RETURN_LOSS_URL = (
    "https://docs.google.com/spreadsheets/d/1MqD1i4gFATr4pEF78kUzb_IFn4WGJgSZ/export?format=xlsx"
)
VSWR_URL = (
    "https://docs.google.com/spreadsheets/d/1VbDwD5FOYmovukIXw-x5rZ1WPV3Ayw80/export?format=xlsx"
)



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


def fetch_excel_bytes(url: str, timeout: float = 30.0, tag: str = "excel") -> io.BytesIO | None:
    """Download XLSX bytes from a Google Sheets export URL."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,"
            "application/octet-stream"
        ),
    }

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return io.BytesIO(response.read())
    except URLError as exc:  # pragma: no cover
        print(f"[{tag}] gagal mengunduh data dari '{url}': {exc}")
        return None


def _open_excel_workbook(source: Path | str, label: str) -> Optional[pd.ExcelFile]:
    """Open Excel workbook either from a local path or remote URL."""
    workbook: Optional[pd.ExcelFile] = None

    if isinstance(source, Path):
        display_name = source.name
        if not source.exists():
            print(f"[{label}] file '{display_name}' tidak ditemukan")
            return None

        try:
            workbook = pd.ExcelFile(source)
        except Exception as exc:  # pragma: no cover
            print(f"[{label}] gagal membuka '{display_name}': {exc}")
            return None
    else:
        display_name = str(source)
        buffer = fetch_excel_bytes(display_name, tag=label)
        if buffer is None:
            return None

        try:
            workbook = pd.ExcelFile(buffer)
        except Exception as exc:  # pragma: no cover
            print(f"[{label}] gagal membaca sumber '{display_name}': {exc}")
            return None

    return workbook


def load_return_loss_data(source: Path | str) -> dict[str, dict[str, list[float]]]:
    """Read Excel workbook and return chart data keyed by frequency string."""

    dataset: dict[str, dict[str, list[float]]] = {}
    workbook = _open_excel_workbook(source, "return-loss")
    if workbook is None:
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
            "freq_key": freq_key,
            "freq_ghz": freq_numeric,
            "frequency_axis": freq_axis,
            "return_loss": return_loss,
            "vswr": vswr,
            "sheet": sheet,
        }

    return dataset


def load_vswr_data(source: Path | str) -> dict[str, dict[str, list[float]]]:
    """Read VSWR workbook and return chart data keyed by frequency string."""

    dataset: dict[str, dict[str, list[float]]] = {}
    workbook = _open_excel_workbook(source, "vswr")
    if workbook is None:
        return dataset

    for sheet in workbook.sheet_names:
        try:
            df = pd.read_excel(workbook, sheet_name=sheet, header=None)
        except Exception as exc:  # pragma: no cover
            print(f"[vswr] gagal membaca sheet '{sheet}': {exc}")
            continue

        df = df.iloc[:, :2].dropna()
        if df.empty:
            continue

        pairs = [
            (float(freq_val), float(vswr_val))
            for freq_val, vswr_val in zip(df.iloc[:, 0], df.iloc[:, 1])
        ]
        pairs.sort(key=lambda item: item[0])

        freq_axis = [freq for freq, _ in pairs]
        vswr_values = [value for _, value in pairs]

        try:
            freq_numeric = float(sheet)
        except ValueError:
            freq_numeric = None

        freq_key = format_freq_key(freq_numeric if freq_numeric is not None else sheet)

        dataset[freq_key] = {
            "freq_key": freq_key,
            "freq_ghz": freq_numeric,
            "frequency_axis": freq_axis,
            "vswr": vswr_values,
            "sheet": sheet,
        }

    return dataset


def load_chart_data(return_loss_source: Path | str, vswr_source: Path | str) -> dict[str, dict[str, list[float]]]:
    """Combine return-loss and VSWR datasets for chart consumption."""

    dataset = load_return_loss_data(return_loss_source)
    vswr_dataset = load_vswr_data(vswr_source)

    for freq_key, vswr_entry in vswr_dataset.items():
        base = dataset.setdefault(
            freq_key,
            {
                "freq_key": freq_key,
                "freq_ghz": vswr_entry.get("freq_ghz"),
                "frequency_axis": vswr_entry.get("frequency_axis", []),
                "return_loss": [],
                "vswr": [],
                "sheet": vswr_entry.get("sheet"),
            },
        )

        base["vswr"] = vswr_entry.get("vswr", [])
        if not base.get("frequency_axis"):
            base["frequency_axis"] = vswr_entry.get("frequency_axis", [])
        if base.get("freq_ghz") is None:
            base["freq_ghz"] = vswr_entry.get("freq_ghz")

    return dataset


RETURN_LOSS_DATA = load_chart_data(RETURN_LOSS_URL, VSWR_URL)


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
    """Effective permittivity microstrip (Hammerstad-Jensen approximation)."""
    if w_h <= 0:
        return float("nan")

    ee = (er + 1) / 2 + (er - 1) / 2 * (1 / math.sqrt(1 + 12 / w_h))

    if w_h < 1:
        ee += 0.04 * (1 - w_h) ** 2

    return ee


def microstrip_z0(er: float, w_h: float) -> float:
    """Characteristic impedance Z0 for microstrip line."""
    ee = eps_eff_hammerstad(er, w_h)
    if w_h <= 0 or math.isnan(ee) or ee <= 0:
        return float("nan")

    if w_h <= 1:
        return (60 / math.sqrt(ee)) * math.log(8 / w_h + 0.25 * w_h)

    return (120 * math.pi) / (math.sqrt(ee) * (w_h + 1.393 + 0.667 * math.log(w_h + 1.444)))


def microstrip_w_h_for_z0(er: float, z0: float) -> float:
    """Closed-form approximation for W/h given target Z0 (ohm)."""
    if er <= 0 or z0 <= 0:
        return float("nan")

    a = (z0 / 60) * math.sqrt((er + 1) / 2) + ((er - 1) / (er + 1)) * (0.23 + 0.11 / er)
    w_h = (8 * math.exp(a)) / (math.exp(2 * a) - 2)
    if w_h <= 2:
        return w_h

    b = (377 * math.pi) / (2 * z0 * math.sqrt(er))
    return (2 / math.pi) * (
        b
        - 1
        - math.log(2 * b - 1)
        + ((er - 1) / (2 * er)) * (math.log(b - 1) + 0.39 - 0.61 / er)
    )


def solve_w_h_for_z0(er: float, z0: float, tol: float = 1e-6, max_iter: int = 120) -> float:
    """Solve W/h given target Z0 (ohm) using bisection."""
    lo = 1e-6
    hi = 100.0

    z_lo = microstrip_z0(er, lo)
    z_hi = microstrip_z0(er, hi)

    expand = 0
    while (math.isnan(z_lo) or math.isnan(z_hi) or z_lo < z0 or z_hi > z0) and expand < 20:
        hi *= 2
        z_hi = microstrip_z0(er, hi)
        expand += 1

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        z_mid = microstrip_z0(er, mid)

        if math.isnan(z_mid):
            lo = mid
            continue

        if abs(z_mid - z0) < 1e-6:
            return mid

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
    """Patch width (rectangular) - formula umum."""
    return (c / (2 * f_hz)) * math.sqrt(2 / (er + 1))


def patch_length(c: float, f_hz: float, er: float, h_m: float, w_m: float) -> tuple[float, float]:
    """Patch length (rectangular) - formula umum."""
    w_h = w_m / h_m
    eps_eff = eps_eff_hammerstad(er, w_h)

    delta_l = 0.412 * h_m * ((eps_eff + 0.3) * (w_h + 0.264)) / ((eps_eff - 0.258) * (w_h + 0.8))
    leff = c / (2 * f_hz * math.sqrt(eps_eff))
    L = leff - 2 * delta_l

    return L, eps_eff


def compute_all(freq_ghz: float) -> Results:
    f_hz = freq_ghz * 1e9
    h_m = H_MM * 1e-3

    wp_m = patch_width(C, f_hz, ER) * PATCH_WIDTH_SCALE
    lp_m, eps_patch = patch_length(C, f_hz, ER, h_m, wp_m)

    w_h_line = microstrip_w_h_for_z0(ER, Z0_OHM)
    if math.isnan(w_h_line) or w_h_line <= 0:
        w_h_line = solve_w_h_for_z0(ER, Z0_OHM)
    wf_m = w_h_line * h_m
    eps_line = eps_eff_hammerstad(ER, w_h_line)

    lambda_g = C / (f_hz * math.sqrt(eps_line))
    lf_m = 0.25 * lambda_g

    wg_m = wp_m + 6 * h_m
    lg_m = lp_m + lf_m + 6 * h_m

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


def _parse_freq_from_query(default_freq: float) -> float:
    try:
        q = request.args.get("freq")
        if q is not None:
            return float(q)
    except Exception:
        pass
    return default_freq


@app.route("/", methods=["GET"])
def landing():
    selected_freq = _parse_freq_from_query(FREQ_OPTIONS_GHZ[0])
    safe_freq = selected_freq if selected_freq in FREQ_OPTIONS_GHZ else FREQ_OPTIONS_GHZ[0]
    svg_results = compute_all(safe_freq)

    return render_template(
        "landing.html",
        freq_options=FREQ_OPTIONS_GHZ,
        fixed=dict(h_mm=H_MM, z0=Z0_OHM, er=ER, c=C),
        selected_freq=safe_freq,
        svg_results=svg_results,
    )


@app.route("/calculator", methods=["GET", "POST"])
def calculator():
    results: Optional[Results] = None
    error: Optional[str] = None

    selected_freq = FREQ_OPTIONS_GHZ[0]

    if request.method == "POST":
        try:
            selected_freq = float(request.form.get("freq", str(FREQ_OPTIONS_GHZ[0])))
            if selected_freq not in FREQ_OPTIONS_GHZ:
                selected_freq = FREQ_OPTIONS_GHZ[0]
            results = compute_all(selected_freq)
        except Exception as exc:
            error = f"Gagal menghitung: {exc}"
    else:
        selected_freq = _parse_freq_from_query(selected_freq)

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
