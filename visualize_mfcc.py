#!/usr/bin/env python3
"""
AcoustEEEcare — MFCC .npy Visualizer
======================================
Loads a .npy file saved by the BLE receiver and produces:
  1. MFCC heatmap         — C1–C19 matrix (dark inferno style, C0 excluded)
  2. Per-coefficient mean ± std bar chart
  3. First 5 coefficients over time (line plot)
  4. Coefficient 0 spectrogram-style envelope

Usage:
    pip install numpy matplotlib
    python visualize_mfcc.py path/to/mfcc_001_20240101_120000.npy

Or drop it in the same folder as your recordings and run without args
to pick the most recent .npy automatically.
"""

import sys
import glob
import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Output folder for saved figures ──────────────────────────────────────────
FIGURES_DIR = Path("recordings_mfcc/figures")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize AcoustEEEcare MFCC .npy files"
    )
    parser.add_argument(
        "npy_file",
        nargs="?",
        help="Path to .npy file. If omitted, the most recent "
             ".npy in ./recordings_mfcc/ is used."
    )
    parser.add_argument(
        "--sr", type=int, default=2000,
        help="Decimated sample rate in Hz (default: 2000)"
    )
    parser.add_argument(
        "--hop", type=int, default=15,
        help="Hop size in samples at decimated rate (default: 15)"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the figure as a PNG instead of displaying it"
    )
    return parser.parse_args()


def find_latest_npy() -> Path:
    candidates = sorted(
        glob.glob("recordings_mfcc/mfcc_*.npy"),
        key=os.path.getmtime
    )
    if not candidates:
        print("ERROR: No .npy files found in ./recordings_mfcc/")
        print("       Pass a path explicitly: python visualize_mfcc.py file.npy")
        sys.exit(1)
    return Path(candidates[-1])


# ── Load + validate ───────────────────────────────────────────────────────────

def load_mfcc(path: Path) -> np.ndarray:
    arr = np.load(str(path))
    if arr.ndim != 2:
        print(f"ERROR: Expected 2-D array, got shape {arr.shape}")
        sys.exit(1)
    print(f"Loaded: {path}")
    print(f"  Shape : {arr.shape}  (frames × coefficients)")
    print(f"  Range : [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"  NaN   : {np.isnan(arr).sum()}   Inf: {np.isinf(arr).sum()}")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


# ── Time axis ─────────────────────────────────────────────────────────────────

def frame_times(n_frames: int, hop_samples: int, sr: int) -> np.ndarray:
    """Centre time of each frame in seconds."""
    return np.arange(n_frames) * hop_samples / sr


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_all(arr: np.ndarray, path: Path, sr: int, hop: int, save: bool):
    n_frames, n_coeffs = arr.shape
    times = frame_times(n_frames, hop, sr)
    duration = times[-1]

    # ── Dark theme setup ──────────────────────────────────────────────────────
    plt.style.use("dark_background")
    BG       = "#0d0d0d"
    PANEL_BG = "#111111"
    GRID_CLR = "#2a2a2a"
    TEXT_CLR = "#cccccc"
    ACCENT   = "#e74c3c"
    ACCENT2  = "#3498db"
    ACCENT3  = "#2ecc71"

    fig = plt.figure(figsize=(16, 11), facecolor=BG)
    fig.suptitle(
        f"MFCC Visualizer — {path.name}\n"
        f"{n_frames} frames × {n_coeffs} coefficients  |  "
        f"~{duration:.1f} s  |  decimated @ {sr} Hz  |  hop = {hop} samples",
        fontsize=12, fontweight="bold", y=0.98, color=TEXT_CLR
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        top=0.91, bottom=0.07,
        left=0.07, right=0.97
    )

    def style_ax(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_CLR, labelsize=8)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(TEXT_CLR)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # ── 1. MFCC heatmap (C1–C19, inferno, black bg) ───────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1)

    # Exclude C0 — use C1 onwards
    arr_display = arr[:, 1:].T          # shape: (n_coeffs-1, n_frames)
    vmin = np.percentile(arr[:, 1:], 2)
    vmax = np.percentile(arr[:, 1:], 98)

    im = ax1.imshow(
        arr_display,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="inferno",
        extent=[times[0], times[-1], 0.5, n_coeffs - 0.5]
    )
    cbar = fig.colorbar(im, ax=ax1, pad=0.01, fraction=0.02)
    cbar.set_label("Coefficient value", fontsize=8, color=TEXT_CLR)
    cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_CLR)

    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_ylabel("MFCC coefficient index", fontsize=9)
    ax1.set_title("MFCC Matrix — C1–C19 (C0 excluded to preserve contrast)", fontsize=10, fontweight="bold")
    ax1.set_yticks(range(1, n_coeffs, 2))
    ax1.tick_params(labelsize=8)

    # ── 2. Per-coefficient mean ± std ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)

    means     = arr.mean(axis=0)
    stds      = arr.std(axis=0)
    coeff_idx = np.arange(n_coeffs)
    colors    = [ACCENT if m >= 0 else ACCENT2 for m in means]

    ax2.bar(coeff_idx, means, color=colors, alpha=0.85, zorder=3, label="Mean")
    ax2.errorbar(
        coeff_idx, means, yerr=stds,
        fmt="none", ecolor=TEXT_CLR, elinewidth=1.2,
        capsize=3, zorder=4, label="±1 std"
    )
    ax2.axhline(0, color=TEXT_CLR, linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Coefficient index", fontsize=9)
    ax2.set_ylabel("Value", fontsize=9)
    ax2.set_title("Per-Coefficient Mean ± Std Dev", fontsize=10, fontweight="bold")
    ax2.set_xticks(coeff_idx)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=8, facecolor="#1a1a1a", labelcolor=TEXT_CLR)
    ax2.grid(axis="y", color=GRID_CLR, alpha=0.5, zorder=0)

    # ── 3. First 5 coefficients over time ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)

    n_show  = min(5, n_coeffs)
    palette = [ACCENT, ACCENT2, ACCENT3, "#f39c12", "#9b59b6"]

    for i in range(n_show):
        ax3.plot(
            times, arr[:, i],
            color=palette[i], linewidth=0.8,
            alpha=0.85, label=f"C{i}"
        )
    ax3.set_xlabel("Time (s)", fontsize=9)
    ax3.set_ylabel("Value", fontsize=9)
    ax3.set_title(f"Coefficients C0–C{n_show-1} Over Time", fontsize=10, fontweight="bold")
    ax3.legend(fontsize=8, ncol=n_show, loc="upper right",
               facecolor="#1a1a1a", labelcolor=TEXT_CLR)
    ax3.grid(color=GRID_CLR, alpha=0.4)
    ax3.tick_params(labelsize=8)

    # ── 4. C0 envelope + rolling statistics ───────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    style_ax(ax4)

    c0     = arr[:, 0]
    window = max(1, int(0.5 * sr / hop))

    if len(c0) >= window:
        roll_mean = np.convolve(c0, np.ones(window) / window, mode="same")
        roll_std  = np.array([
            c0[max(0, i - window // 2): i + window // 2].std()
            for i in range(len(c0))
        ])
    else:
        roll_mean = c0.copy()
        roll_std  = np.zeros_like(c0)

    ax4.plot(times, c0, color="#555555", linewidth=0.6, alpha=0.7, label="C0 raw")
    ax4.plot(times, roll_mean, color=ACCENT, linewidth=1.6,
             label=f"C0 rolling mean (~{window} frames)")
    ax4.fill_between(
        times,
        roll_mean - roll_std,
        roll_mean + roll_std,
        color=ACCENT, alpha=0.15,
        label="±1 std band"
    )
    ax4.axhline(c0.mean(), color=TEXT_CLR, linewidth=0.9,
                linestyle=":", label=f"Global mean ({c0.mean():.2f})")
    ax4.set_xlabel("Time (s)", fontsize=9)
    ax4.set_ylabel("C0 value", fontsize=9)
    ax4.set_title("Coefficient 0 — Raw + Rolling Mean ± Std", fontsize=10, fontweight="bold")
    ax4.legend(fontsize=8, loc="upper right",
               facecolor="#1a1a1a", labelcolor=TEXT_CLR)
    ax4.grid(color=GRID_CLR, alpha=0.4)
    ax4.tick_params(labelsize=8)

    # ── Output ────────────────────────────────────────────────────────────────
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / (path.stem + ".png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"Saved figure -> {out}")

    if not save:
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.npy_file:
        npy_path = Path(args.npy_file)
        if not npy_path.exists():
            print(f"ERROR: File not found: {npy_path}")
            sys.exit(1)
    else:
        npy_path = find_latest_npy()
        print(f"Auto-selected: {npy_path}")

    arr = load_mfcc(npy_path)
    plot_all(arr, npy_path, sr=args.sr, hop=args.hop, save=args.save)


if __name__ == "__main__":
    main()