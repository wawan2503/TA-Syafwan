from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from flask import Flask, render_template, request

app = Flask(__name__)

# ===== Default / fixed (kamu boleh ubah kalau mau dinamis) =====
FREQ_OPTIONS_GHZ = [1.8, 2.1, 2.3, 2.4, 3.3]

C = 3e8          # m/s
ER = 4.4         # FR-4 (umum)
H_MM = 1.6       # mm
Z0_OHM = 50.0    # ohm

# Mode bilangan untuk rumus resonansi fo (default: mode dominan TM10)
MODE_M = 1
MODE_N = 0

BASE_DIR = Path(__file__).resolve().parent


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


def _safe_float(value: str | float | int | None) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).replace(',', '.'))
    except (TypeError, ValueError):
        return None


def _slugify(text: str) -> str:
    cleaned = ''.join(ch if ch.isalnum() else '-' for ch in text.lower())
    while '--' in cleaned:
        cleaned = cleaned.replace('--', '-')
    return cleaned.strip('-') or 'default'


def _line_looks_numeric(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    first = stripped[0]
    if first not in '-+.0123456789':
        return False
    if all(ch == '-' for ch in stripped):
        return False
    return True


def _parse_numeric_columns(path: Path, column_indexes: tuple[int, int]) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    max_index = max(column_indexes)
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as handle:
            for line in handle:
                if not _line_looks_numeric(line):
                    continue
                cleaned = line.replace(',', '.').split()
                if len(cleaned) <= max_index:
                    continue
                try:
                    x_val = float(cleaned[column_indexes[0]])
                    y_val = float(cleaned[column_indexes[1]])
                except ValueError:
                    continue
                xs.append(x_val)
                ys.append(y_val)
    except OSError as exc:  # pragma: no cover
        print(f"[chart-data] gagal membaca '{path.name}': {exc}")
    return xs, ys


def _find_measurement_file(folder: Path, tokens: Iterable[str]) -> Optional[Path]:
    tokens_lower = [token.lower() for token in tokens]
    for candidate in sorted(folder.glob('*.txt')):
        name_lower = candidate.name.lower()
        if any(token in name_lower for token in tokens_lower):
            return candidate
    return None


def _load_measurement_dataset(base_dir: Path, measurement_cfg: list[Dict[str, Any]]) -> dict[str, Any]:
    dataset: dict[str, Any] = {
        'default_scenario': '',
        'default_freq_key': format_freq_key(FREQ_OPTIONS_GHZ[0]) if FREQ_OPTIONS_GHZ else '',
        'measurement_order': [cfg['key'] for cfg in measurement_cfg],
        'measurements': {
            cfg['key']: {
                'title': cfg['title'],
                'description': cfg['description'],
                'x_label': cfg['x_label'],
                'y_label': cfg['y_label'],
                'color': cfg['color'],
                'background': cfg['background'],
                'tension': cfg.get('tension', 0.25),
            }
            for cfg in measurement_cfg
        },
        'scenario_options': [],
        'scenarios': {},
    }

    if not base_dir.exists():
        return dataset

    scenario_dirs = [path for path in base_dir.iterdir() if path.is_dir()]
    scenario_dirs.sort(key=lambda item: item.name.lower())

    for scenario_dir in scenario_dirs:
        scenario_key = _slugify(scenario_dir.name)
        freq_dirs = [path for path in scenario_dir.iterdir() if path.is_dir()]
        freq_dirs.sort(key=lambda item: (_safe_float(item.name) is None, _safe_float(item.name) or 0.0, item.name))
        scenario_entry: dict[str, Any] = {
            'label': scenario_dir.name,
            'frequencies': {},
            'default_freq_key': '',
        }

        for freq_dir in freq_dirs:
            freq_value = _safe_float(freq_dir.name)
            freq_key = format_freq_key(freq_value if freq_value is not None else freq_dir.name)
            freq_label = f"{freq_key} GHz" if freq_value is not None else freq_dir.name
            freq_entry: dict[str, Any] = {
                'label': freq_label,
                'freq': freq_value,
                'measurements': {},
            }

            for cfg in measurement_cfg:
                file_path = _find_measurement_file(freq_dir, cfg['file_tokens'])
                if file_path is None:
                    continue
                x_vals, y_vals = _parse_numeric_columns(file_path, cfg['column_indexes'])
                if not x_vals or not y_vals:
                    continue
                freq_entry['measurements'][cfg['key']] = {
                    'x': x_vals,
                    'y': y_vals,
                }

            if freq_entry['measurements']:
                if not scenario_entry['default_freq_key']:
                    scenario_entry['default_freq_key'] = freq_key
                scenario_entry['frequencies'][freq_key] = freq_entry

        if scenario_entry['frequencies']:
            dataset['scenarios'][scenario_key] = scenario_entry
            dataset['scenario_options'].append({'key': scenario_key, 'label': scenario_dir.name})
            if not dataset['default_scenario']:
                dataset['default_scenario'] = scenario_key

    return dataset


AWR_MEASUREMENT_CONFIG = [
    {
        'key': 'return_loss',
        'title': 'Return Loss (AWR)',
        'description': 'Kurva S11 hasil ekspor AWR (Data RL*.txt).',
        'x_label': 'Frekuensi (GHz)',
        'y_label': 'Return Loss (dB)',
        'color': 'rgba(37,99,235,1)',
        'background': 'rgba(37,99,235,0.15)',
        'file_tokens': ['rl'],
        'column_indexes': (0, 1),
    },
    {
        'key': 'vswr',
        'title': 'VSWR (AWR)',
        'description': 'Nilai VSWR dari file Data VSWR*.txt.',
        'x_label': 'Frekuensi (GHz)',
        'y_label': 'VSWR',
        'color': 'rgba(16,185,129,1)',
        'background': 'rgba(16,185,129,0.15)',
        'file_tokens': ['vswr'],
        'column_indexes': (0, 1),
    },
    {
        'key': 'gain',
        'title': 'Gain (AWR)',
        'description': 'Distribusi |PPC TPwr| terhadap sudut (Data Gain*.txt).',
        'x_label': 'Sudut (deg)',
        'y_label': 'Gain (dB)',
        'color': 'rgba(236,72,153,1)',
        'background': 'rgba(236,72,153,0.15)',
        'file_tokens': ['gain'],
        'column_indexes': (0, 1),
    },
]


CST_MEASUREMENT_CONFIG = [
    {
        'key': 'return_loss',
        'title': 'Return Loss (CST)',
        'description': 'Data S11 dari RL *.txt (CST).',
        'x_label': 'Frekuensi (GHz)',
        'y_label': 'Return Loss (dB)',
        'color': 'rgba(59,130,246,1)',
        'background': 'rgba(59,130,246,0.15)',
        'file_tokens': ['rl'],
        'column_indexes': (0, 1),
    },
    {
        'key': 'vswr',
        'title': 'VSWR (CST)',
        'description': 'VSWR linear hasil simulasi CST.',
        'x_label': 'Frekuensi (GHz)',
        'y_label': 'VSWR',
        'color': 'rgba(16,185,129,1)',
        'background': 'rgba(16,185,129,0.15)',
        'file_tokens': ['vswr'],
        'column_indexes': (0, 1),
    },
    {
        'key': 'gain',
        'title': 'Gain (CST)',
        'description': 'Abs(Gain) pada phi = 90 deg dari file Gain *.txt.',
        'x_label': 'Theta (deg)',
        'y_label': 'Gain (dBi)',
        'color': 'rgba(249,115,22,1)',
        'background': 'rgba(249,115,22,0.18)',
        'file_tokens': ['gain'],
        'column_indexes': (0, 2),
    },
    {
        'key': 'polar',
        'title': 'Polar (CST)',
        'description': 'Abs(Gain) pada phi = 0 deg dari file Polar *.txt.',
        'x_label': 'Theta (deg)',
        'y_label': 'Gain (dBi)',
        'color': 'rgba(14,165,233,1)',
        'background': 'rgba(14,165,233,0.15)',
        'file_tokens': ['polar'],
        'column_indexes': (0, 2),
    },
]


def load_awr_chart_data(base_dir: Path) -> dict[str, Any]:
    return _load_measurement_dataset(base_dir, AWR_MEASUREMENT_CONFIG)


def load_cst_chart_data(base_dir: Path) -> dict[str, Any]:
    return _load_measurement_dataset(base_dir, CST_MEASUREMENT_CONFIG)


AWR_CHART_DATA = load_awr_chart_data(BASE_DIR / 'AWR')
CST_CHART_DATA = load_cst_chart_data(BASE_DIR / 'CST')



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
    lambda0_mm: float


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
    return (c / (2 * f_hz)) * math.sqrt((er + 1) / 2)


def patch_length(c: float, f_hz: float, er: float, h_m: float, w_m: float) -> tuple[float, float]:
    """Patch length (rectangular) - formula umum."""
    if h_m <= 0:
        return float("nan"), float("nan")

    w_h = w_m / h_m if h_m > 0 else float("nan")
    eps_eff = eps_eff_hammerstad(er, w_h)

    sqrt_eps = math.sqrt(eps_eff) if not math.isnan(eps_eff) and eps_eff > 0 else float("nan")
    leff = float("nan")
    if not math.isnan(sqrt_eps):
        freq_factor_sq = ((2 * f_hz * sqrt_eps) / c) ** 2
        n_term = 0.0
        if MODE_N and w_m > 0:
            n_term = (MODE_N / w_m) ** 2
        denom = freq_factor_sq - n_term
        if MODE_M > 0 and denom > 0:
            leff = MODE_M / math.sqrt(denom)

    delta_l = float("nan")
    if not math.isnan(w_h) and w_h > 0 and not math.isnan(eps_eff):
        denom_delta = (eps_eff - 0.258) * (w_h + 0.8)
        if denom_delta != 0:
            delta_l = 0.412 * h_m * ((eps_eff + 0.3) * (w_h + 0.264)) / denom_delta

    L = float("nan")
    if not math.isnan(leff) and not math.isnan(delta_l):
        L = leff - 2 * delta_l

    return L, eps_eff


def compute_all(freq_ghz: float) -> Results:
    f_hz = freq_ghz * 1e9
    h_m = H_MM * 1e-3

    wp_m = patch_width(C, f_hz, ER)
    lp_m, eps_patch = patch_length(C, f_hz, ER, h_m, wp_m)

    b = (377 * math.pi) / (2 * Z0_OHM * math.sqrt(ER))
    try:
        w_h_line = (2 / math.pi) * (
            b
            - 1
            - math.log(2 * b - 1)
            + ((ER - 1) / (2 * ER)) * (math.log(b - 1) + 0.039 - 0.61 / ER)
        )
    except ValueError:
        w_h_line = float("nan")
    if math.isnan(w_h_line) or w_h_line <= 0:
        w_h_line = solve_w_h_for_z0(ER, Z0_OHM)
    wf_m = w_h_line * h_m
    eps_line = eps_eff_hammerstad(ER, w_h_line)

    lambda0 = C / f_hz
    lf_m = lambda0 / 10.0

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
        lambda0_mm=lambda0 * to_mm,
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

    return render_template(
        "index.html",
        freq_options=FREQ_OPTIONS_GHZ,
        fixed=dict(h_mm=H_MM, z0=Z0_OHM, er=ER, c=C),
        selected_freq=selected_freq,
        results=results,
        svg_results=svg_results,
        show_svg_labels=show_svg_labels,
        error=error,
        awr_chart_data=AWR_CHART_DATA,
        cst_chart_data=CST_CHART_DATA,
    )


if __name__ == "__main__":
    app.run(debug=True)
