"""
AcoustEEEcare — Audio Diagnostic Visualizer
=============================================
Just press Run in VS Code — no terminal arguments needed.

- Automatically picks the LATEST .wav file in RECORDINGS_DIR
- Saves the analysis PNG into a subfolder:
    D:\zephyrdev\acousteeecare\recordings\analysis\
- To analyze a specific file, set WAV_FILE below instead of leaving it None

Settings to change:
    RECORDINGS_DIR  →  folder where your .wav files are saved
    WAV_FILE        →  set to a specific filename to override auto-latest,
                       e.g. "recording_001_20260501_104853.wav"
                       leave as None to always use the newest file
    IS_TONE         →  True when using diagnostic tone, False for real voice
    EXPECTED_RATE   →  must match firmware SAMPLING_RATE (default 8000)
    TONE_HZ         →  must match firmware DIAG_TONE_HZ (default 1000)
"""

import os
import glob
import wave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
from scipy.fft import fft, fftfreq

# ══════════════════════════════════════════════════════════════
# ▼▼▼  EDIT THESE SETTINGS  ▼▼▼
# ══════════════════════════════════════════════════════════════

RECORDINGS_DIR = r"D:/zephyrdev/acousteeecare/recordings"

# Set to a filename string to pin a specific file, e.g.:
# WAV_FILE = "recording_001_20260501_104853.wav"
# Leave as None to always auto-pick the newest .wav in RECORDINGS_DIR
WAV_FILE = None

IS_TONE       = True   # True = diagnostic tone mode, False = real voice
EXPECTED_RATE = 8000   # must match firmware SAMPLING_RATE
TONE_HZ       = 1000   # must match firmware DIAG_TONE_HZ

# ══════════════════════════════════════════════════════════════
# ▲▲▲  EDIT THESE SETTINGS  ▲▲▲
# ══════════════════════════════════════════════════════════════


# ── Auto-select latest WAV ─────────────────────────────────────────────────────
def resolve_wav_path() -> str:
    if WAV_FILE is not None:
        path = os.path.join(RECORDINGS_DIR, WAV_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"WAV_FILE not found: {path}")
        return path

    pattern = os.path.join(RECORDINGS_DIR, "*.wav")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No .wav files found in:\n  {RECORDINGS_DIR}\n"
            "Make sure RECORDINGS_DIR is set correctly above."
        )
    latest = max(files, key=os.path.getmtime)
    return latest


# ── Output folder ──────────────────────────────────────────────────────────────
def make_output_path(wav_path: str) -> str:
    """Save PNG into an 'analysis' subfolder next to the recordings."""
    folder     = os.path.dirname(wav_path)
    out_folder = os.path.join(folder, "analysis")
    os.makedirs(out_folder, exist_ok=True)
    basename   = os.path.splitext(os.path.basename(wav_path))[0]
    return os.path.join(out_folder, f"{basename}_analysis.png")


# ── Audio loading ──────────────────────────────────────────────────────────────
def load_wav(path: str):
    with wave.open(path, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames    = wf.getnframes()
        raw         = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        samples = samples[::n_channels]
    return samples, sample_rate, n_frames


# ── Gap detection ──────────────────────────────────────────────────────────────
def detect_gaps(samples: np.ndarray, threshold=50, min_gap_samples=8):
    """
    Find regions where absolute value stays near zero for several consecutive
    samples — a strong sign of DMA buffer gaps caused by overruns.
    Returns list of (start_sample, end_sample) tuples.
    """
    near_zero = np.abs(samples) < threshold
    gaps      = []
    in_gap    = False
    gap_start = 0
    for i, z in enumerate(near_zero):
        if z and not in_gap:
            in_gap    = True
            gap_start = i
        elif not z and in_gap:
            in_gap = False
            if (i - gap_start) >= min_gap_samples:
                gaps.append((gap_start, i))
    if in_gap and (len(samples) - gap_start) >= min_gap_samples:
        gaps.append((gap_start, len(samples)))
    return gaps


# ── RMS helpers ────────────────────────────────────────────────────────────────
def rms_envelope(samples: np.ndarray, window: int):
    n_windows = len(samples) // window
    trimmed   = samples[:n_windows * window].reshape(n_windows, window)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def half_buffer_energy(samples: np.ndarray, half_buf=256):
    """Energy per firmware half-buffer (HALF_BUF_SAMPLES = 256)."""
    n      = (len(samples) // half_buf) * half_buf
    blocks = samples[:n].reshape(-1, half_buf)
    return np.sqrt(np.mean(blocks ** 2, axis=1))


# ── Main analysis ──────────────────────────────────────────────────────────────
def analyze(wav_path: str):
    print(f"\n{'='*60}")
    print(f"  AcoustEEEcare Audio Diagnostic")
    print(f"{'='*60}")
    print(f"  File      : {wav_path}")

    samples, file_rate, _ = load_wav(wav_path)
    duration = len(samples) / file_rate

    print(f"  Rate      : {file_rate} Hz  (expected {EXPECTED_RATE} Hz)")
    print(f"  Samples   : {len(samples)}")
    print(f"  Duration  : {duration:.3f} s")
    print(f"  Min/Max   : {samples.min():.0f} / {samples.max():.0f}")
    print(f"  RMS       : {np.sqrt(np.mean(samples**2)):.1f}")

    if file_rate != EXPECTED_RATE:
        print(f"\n  ⚠  Rate mismatch! WAV={file_rate} Hz but expected {EXPECTED_RATE} Hz")
        print(f"     Playback will be at wrong speed.")

    time_axis = np.arange(len(samples)) / file_rate

    # Gap detection
    gaps = detect_gaps(samples, threshold=50, min_gap_samples=8)
    print(f"\n  Gaps detected (near-zero runs >= 8 samples): {len(gaps)}")
    for i, (s, e) in enumerate(gaps[:10]):
        print(f"    gap {i+1:>3}: sample {s}-{e}  "
              f"({(e-s)/file_rate*1000:.1f} ms)  @ {s/file_rate:.3f} s")
    if len(gaps) > 10:
        print(f"    ... and {len(gaps)-10} more")

    # Half-buffer energy
    hb_energy = half_buffer_energy(samples, half_buf=256)
    hb_times  = np.arange(len(hb_energy)) * 256 / file_rate

    # RMS envelope (32 ms windows)
    win       = max(1, int(file_rate * 0.032))
    rms_env   = rms_envelope(samples, win)
    rms_times = np.arange(len(rms_env)) * win / file_rate

    # FFT
    N         = min(len(samples), 65536)
    fft_vals  = np.abs(fft(samples[:N])) / N
    fft_freqs = fftfreq(N, d=1.0 / file_rate)
    pos_mask  = fft_freqs > 0
    fft_freqs = fft_freqs[pos_mask]
    fft_vals  = fft_vals[pos_mask]

    # Spectrogram
    nperseg             = min(256, len(samples))
    f_spec, t_spec, Sxx = signal.spectrogram(
        samples, fs=file_rate, nperseg=nperseg,
        noverlap=nperseg // 2, scaling="spectrum"
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Tone analysis
    peak_freq = None
    if IS_TONE:
        peak_idx  = np.argmax(fft_vals)
        peak_freq = fft_freqs[peak_idx]
        ratio     = peak_freq / TONE_HZ
        print(f"\n  Tone analysis:")
        print(f"    Expected : {TONE_HZ} Hz")
        print(f"    Detected : {peak_freq:.1f} Hz")
        print(f"    Ratio    : {ratio:.3f}  (1.000 = perfect)")
        if ratio < 0.85:
            print(f"    WARNING: Tone is LOW  -> effective sample rate ~{file_rate*ratio:.0f} Hz")
            print(f"             Firmware is sampling SLOWER than expected")
        elif ratio > 1.15:
            print(f"    WARNING: Tone is HIGH -> effective sample rate ~{file_rate*ratio:.0f} Hz")
            print(f"             Firmware is sampling FASTER than expected")
        else:
            print(f"    OK: Tone frequency correct — sample rate is accurate")

    # Verdict
    verdict = []
    if len(gaps) == 0:
        verdict.append("OK: No gaps")
    else:
        verdict.append(f"FAIL: {len(gaps)} gap(s) — likely DMA overruns")

    rms_std  = np.std(rms_env)
    rms_mean = np.mean(rms_env)
    if rms_mean > 0 and rms_std / rms_mean < 0.3:
        verdict.append("OK: Steady amplitude")
    else:
        verdict.append("FAIL: Unsteady amplitude — stuttering detected")

    print(f"\n  Verdict:")
    for v in verdict:
        print(f"    {v}")

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle(
        f"AcoustEEEcare Audio Diagnostic\n"
        f"{os.path.basename(wav_path)}   |   "
        f"{file_rate} Hz  {duration:.2f} s  {len(gaps)} gap(s)  "
        + ("[ TONE MODE ]" if IS_TONE else "[ VOICE MODE ]"),
        fontsize=13, fontweight="bold", color="white"
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1a1a1a")
        ax.set_title(title, fontsize=9, color="#cccccc", pad=6)
        ax.set_xlabel(xlabel, fontsize=8, color="#999999")
        ax.set_ylabel(ylabel, fontsize=8, color="#999999")
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # Plot 1 — Full waveform
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, samples, color="#2196F3", linewidth=0.35, alpha=0.85)
    ax1.axhline(0, color="#555555", linewidth=0.5, linestyle="--")
    for s, e in gaps:
        ax1.axvspan(s / file_rate, e / file_rate, color="red", alpha=0.35)
    if gaps:
        ax1.axvspan(0, 0, color="red", alpha=0.35,
                    label=f"{len(gaps)} gap(s)  red shading = silence")
        ax1.legend(fontsize=8, loc="upper right",
                   facecolor="#222222", labelcolor="white")
    ax1.set_xlim(0, duration)
    style_ax(ax1,
             "1  Full Waveform   red shading = detected silent gaps (DMA overruns)",
             "Time (s)", "Amplitude (int16)")

    # Plot 2 — Half-buffer energy
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(hb_times, hb_energy, color="#4CAF50", linewidth=1.1)
    low_threshold = np.percentile(hb_energy, 10)
    low_idx = np.where(hb_energy < low_threshold * 0.5)[0]
    if len(low_idx):
        ax2.scatter(hb_times[low_idx], hb_energy[low_idx],
                    color="red", s=18, zorder=5,
                    label=f"{len(low_idx)} dropped buffer(s)")
        ax2.legend(fontsize=8, facecolor="#222222", labelcolor="white")
    ax2.set_xlim(0, duration)
    style_ax(ax2,
             "2  Energy per 256-sample Half-Buffer\nred dots = dropped DMA buffers",
             "Time (s)", "RMS Energy")

    # Plot 3 — RMS envelope
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(rms_times, rms_env, color="#FF9800", linewidth=1.1)
    ax3.set_xlim(0, duration)
    style_ax(ax3,
             "3  RMS Envelope  (32 ms windows)\nshould be flat/steady for diagnostic tone",
             "Time (s)", "RMS")

    # Plot 4 — FFT spectrum
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(fft_freqs, fft_vals, color="#9C27B0", linewidth=0.8)
    ax4.set_xlim(0, file_rate / 2)
    if IS_TONE and peak_freq is not None:
        ax4.axvline(TONE_HZ,   color="red",     linewidth=1.5,
                    linestyle="--", label=f"Expected: {TONE_HZ} Hz")
        ax4.axvline(peak_freq, color="#00FF88",  linewidth=1.5,
                    linestyle="--", label=f"Actual: {peak_freq:.1f} Hz")
        ax4.legend(fontsize=8, facecolor="#222222", labelcolor="white")
    style_ax(ax4,
             "4  FFT Spectrum\ntone should be ONE sharp peak at 1000 Hz",
             "Frequency (Hz)", "Magnitude")

    # Plot 5 — Spectrogram
    ax5 = fig.add_subplot(gs[2, 1])
    im  = ax5.pcolormesh(t_spec, f_spec, Sxx_db, shading="gouraud",
                         cmap="inferno", vmin=np.percentile(Sxx_db, 10))
    cbar = fig.colorbar(im, ax=ax5)
    cbar.set_label("Power (dB)", color="#999999", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#888888", labelsize=7)
    if IS_TONE:
        ax5.axhline(TONE_HZ, color="cyan", linewidth=1.0,
                    linestyle="--", alpha=0.8, label=f"{TONE_HZ} Hz")
        ax5.legend(fontsize=8, loc="upper right",
                   facecolor="#222222", labelcolor="white")
    style_ax(ax5,
             "5  Spectrogram\ntone = solid horizontal line   gaps = dark vertical bands",
             "Time (s)", "Frequency (Hz)")

    # Bottom summary bar
    summary = (
        f"Rate: {file_rate} Hz  |  Duration: {duration:.2f} s  |  "
        f"Samples: {len(samples)}  |  RMS: {np.sqrt(np.mean(samples**2)):.1f}  |  "
        f"Gaps: {len(gaps)}"
    )
    if IS_TONE and peak_freq is not None:
        summary += f"  |  Peak: {peak_freq:.1f} Hz (expected {TONE_HZ} Hz)"
    summary += "   —   " + "   ·   ".join(verdict)

    fig.text(0.5, 0.005, summary, ha="center", fontsize=8.5, color="#aaaaaa",
             bbox=dict(boxstyle="round", facecolor="#1e1e1e", alpha=0.9))

    # Save
    out_path = make_output_path(wav_path)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Plot saved to:")
    print(f"  {out_path}")
    print(f"{'='*60}\n")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wav_path = resolve_wav_path()
    print(f"\n  Auto-selected WAV: {os.path.basename(wav_path)}")
    analyze(wav_path)