import re
import sys
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NUM_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_frequency_ghz(text: str) -> Optional[float]:
    """Best-effort frequency extraction in GHz from a CST-like header.

    Supports forms like:
      - "f = 1.8"
      - "f = 1.8 GHz"
      - "Frequency = 1.8 GHz"
      - "f = 1.8e+009 Hz" (convert to GHz)
      - "Freq: 1800 MHz" (convert to GHz)
    """
    # Try unit-aware matches first
    patterns = [
        r"(?i)\b(?:f|freq(?:uency)?)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*GHz\b",
        r"(?i)\b(?:f|freq(?:uency)?)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*MHz\b",
        r"(?i)\b(?:f|freq(?:uency)?)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*Hz\b",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            if pat.endswith("GHz\\b"):
                return val
            if pat.endswith("MHz\\b"):
                return val / 1_000.0
            if pat.endswith("Hz\\b"):
                return val / 1_000_000_000.0

    # Fallback: unitless f=1.8 style (assume GHz)
    m = re.search(r"(?i)\b(?:f|freq(?:uency)?)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def load_polar_txt(path: Path) -> Tuple[pd.DataFrame, Optional[float]]:
    """Load CST Polar.txt extracting numeric rows only.

    Returns (df, freq_ghz). The DataFrame has columns:
      ['Theta', 'Phi', 'Abs(Dir)', 'Abs(Theta)']
    """
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Attempt frequency detection from header area
    freq_ghz = _parse_frequency_ghz(text)

    rows: List[List[float]] = []
    for line in text.splitlines():
        # Skip dashed separators quickly
        if set(line.strip()) <= {"-", "_", "="} and len(line.strip()) >= 3:
            continue
        nums = NUM_PATTERN.findall(line)
        if len(nums) >= 4:
            try:
                vals = list(map(float, nums[:4]))
            except ValueError:
                continue
            rows.append(vals)

    if not rows:
        raise ValueError(f"Tidak menemukan baris numerik yang valid di file: {path}")

    df = pd.DataFrame(rows, columns=["Theta", "Phi", "Abs(Dir)", "Abs(Theta)"])
    return df, freq_ghz


def build_cst_oriented_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Split Phi=90 and Phi=270, map angles to CST-like left/right halves, then merge."""
    # Use tolerance because Phi might be 90.0/270.0 with precision noise
    tol = 1e-2
    is_90 = np.isclose(df["Phi"], 90.0, atol=tol)
    is_270 = np.isclose(df["Phi"], 270.0, atol=tol)

    df_90 = df.loc[is_90, ["Theta", "Abs(Dir)", "Abs(Theta)"]].copy()
    df_270 = df.loc[is_270, ["Theta", "Abs(Dir)", "Abs(Theta)"]].copy()

    # Mapping angles to plot layout
    # Left side (360° -> 180°): Phi=90, angle = 360 - Theta
    df_90["Plot_Angle"] = 360.0 - df_90["Theta"].astype(float)
    # Right side (0° -> 180°): Phi=270, angle = Theta
    df_270["Plot_Angle"] = df_270["Theta"].astype(float)

    merged = pd.concat([df_90, df_270], ignore_index=True)
    # Normalize to [0, 360)
    merged["Plot_Angle"] = merged["Plot_Angle"] % 360.0
    merged.sort_values("Plot_Angle", inplace=True)
    merged["Angle_Rad"] = np.deg2rad(merged["Plot_Angle"].to_numpy())

    # Rename for convenience in plotting
    merged.rename(columns={"Abs(Dir)": "Dir_dBi", "Abs(Theta)": "Theta_dBi"}, inplace=True)
    return merged


def render_cst_style_polar(plot_df: pd.DataFrame, freq_ghz: Optional[float]) -> None:
    # Figure and polar axes
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7.5, 7.5), dpi=150)

    # Orientation: 0° at North, clockwise rotation like CST
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Radial limits and ticks
    ax.set_rlim(-40, 10)
    ax.set_rticks([-40, -20, 0])
    ax.set_rlabel_position(135)  # move radial labels to left for readability

    # Symmetric angle labels (CST-like)
    right_deg = [0, 30, 60, 90, 120, 150, 180]
    left_deg = [330, 300, 270, 240, 210]  # equivalent to -30, -60, -90, -120, -150
    xticks_deg = right_deg + left_deg
    xtick_labels = ["0", "30", "60", "90", "120", "150", "180", "30", "60", "90", "120", "150"]
    ax.set_xticks(np.deg2rad(xticks_deg))
    ax.set_xticklabels(xtick_labels)

    # Grid styling to feel CST-like
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)

    # Main curves
    ax.plot(
        plot_df["Angle_Rad"].to_numpy(),
        plot_df["Dir_dBi"].to_numpy(),
        color="red",
        linewidth=2.2,
        solid_joinstyle="round",
        label="Abs(Dir)",
    )
    ax.plot(
        plot_df["Angle_Rad"].to_numpy(),
        plot_df["Theta_dBi"].to_numpy(),
        color="#556B2F",
        linewidth=1.6,
        linestyle="--",
        dash_capstyle="round",
        label="Abs(Theta)",
    )

    # Beamwidth lines (from -40 to 10 dBi)
    r = np.linspace(-40, 10, 100)
    angles_deg = [0.0, 78.5, 360.0 - 78.5]
    for ang in angles_deg:
        th = np.deg2rad(ang)
        ax.plot(np.full_like(r, th), r, color="blue", linewidth=1.3)

    # Quadrant labels
    ax.text(np.deg2rad(150), 8, "Phi= 90", color="black", ha="center", va="center", fontsize=10)
    ax.text(np.deg2rad(30), 8, "Phi=270", color="black", ha="center", va="center", fontsize=10)

    # Title and axis label
    ax.set_title("Farfield Gain Abs (Phi=90)", fontsize=12, pad=18)
    plt.figtext(0.5, 0.02, "Theta / Degree vs. dBi", ha="center", va="center", fontsize=10)

    # Frequency and info block (bottom-right)
    if freq_ghz is None:
        freq_ghz = 1.8
    info = (
        f"Frequency = {freq_ghz:.3g} GHz\n"
        f"Main lobe magnitude = 6.45 dBi\n"
        f"Main lobe direction = 0.0 deg\n"
        f"Angular width (3 dB) = 157.1 deg\n"
        f"Side lobe level = -1.3 dB"
    )
    plt.figtext(0.72, 0.05, info, ha="left", va="bottom", fontsize=9)

    # Manual legend text (top-right), in red
    plt.figtext(0.78, 0.96, f" farfield (f={freq_ghz:.1f}) [1]", color="red", ha="left", va="top", fontsize=10)

    plt.tight_layout()
    plt.show()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Plot CST-like Polar Diagram from Polar.txt")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(Path("CST") / "Polar.txt"),
        help="Path ke file Polar.txt (default: CST/Polar.txt)",
    )
    args = parser.parse_args(argv)

    src = Path(args.path)
    if not src.exists():
        print(f"File tidak ditemukan: {src}", file=sys.stderr)
        print("Pastikan file Polar.txt berada di folder 'CST' atau berikan path secara eksplisit.", file=sys.stderr)
        return 2

    try:
        df, freq_ghz = load_polar_txt(src)
        plot_df = build_cst_oriented_dataframe(df)
    except Exception as e:
        print(f"Gagal memuat/parse data: {e}", file=sys.stderr)
        return 1

    render_cst_style_polar(plot_df, freq_ghz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

