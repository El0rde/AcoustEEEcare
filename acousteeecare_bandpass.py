"""
AcoustEEEcare — Bandpass Filter & Diagnostic Visualizer
=========================================================
Just press Run in VS Code — no terminal arguments needed.

- Automatically picks the LATEST .wav file in RECORDINGS_DIR
- Applies a Butterworth bandpass filter (10–200 Hz)
- Saves the FILTERED audio as a new .wav file
- Saves a side-by-side analysis PNG into a subfolder:
    <RECORDINGS_DIR>/analysis/

Settings to change:
    RECORDINGS_DIR  →  folder where your .wav files are saved
    WAV_FILE        →  set to a specific filename to override auto-latest,
                       leave as None to always use the newest file
    LOWCUT_HZ       →  lower bandpass edge  (default 10 Hz)
    HIGHCUT_HZ      →  upper bandpass edge  (default 200 Hz)
    FILTER_ORDER    →  Butterworth filter order (default 4)
    EXPECTED_RATE   →  must match firmware SAMPLING_RATE (default 8000)
"""

import os
import glob
import wave
import struct
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

LOWCUT_HZ    = 10     # lower bandpass edge in Hz
HIGHCUT_HZ   = 200    # upper bandpass edge in Hz
FILTER_ORDER = 4      # Butterworth order (higher = steeper roll-off)
EXPECTED_RATE = 8000  # must match firmware SAMPLING_RATE

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
    return max(files, key=os.path.getmtime)


# ── Output paths ───────────────────────────────────────────────────────────────
def make_output_paths(wav_path: str):
    """Returns (png_path, filtered_wav_path)."""
    folder     = os.path.dirname(wav_path)
    out_folder = os.path.join(folder, "analysis_bandpass")
    os.makedirs(out_folder, exist_ok=True)
    basename   = os.path.splitext(os.path.basename(wav_path))[0]
    png_path   = os.path.join(out_folder, f"{basename}_bandpass_analysis.png")
    filt_path  = os.path.join(out_folder, f"{basename}_bandpass_{LOWCUT_HZ}-{HIGHCUT_HZ}Hz.wav")
    return png_path, filt_path


# ── Audio loading ──────────────────────────────────────────────────────────────
def load_wav(path: str):
    with wave.open(path, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames    = wf.getnframes()
        sampwidth   = wf.getsampwidth()
        raw         = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        samples = samples[::n_channels]
    return samples, sample_rate


# ── Save WAV ───────────────────────────────────────────────────────────────────
def save_wav(path: str, samples: np.ndarray, sample_rate: int):
    """Save a float32 array back to a 16-bit signed WAV."""
    # Clip and convert to int16
    pcm = np.clip(samples, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ── Bandpass filter ────────────────────────────────────────────────────────────
def bandpass_filter(samples: np.ndarray, sample_rate: int,
                    lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter."""
    nyq  = sample_rate / 2.0
    low  = lowcut  / nyq
    high = highcut / nyq
    # Clamp to valid range
    low  = max(low,  1e-4)
    high = min(high, 1.0 - 1e-4)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, samples)   # zero-phase (no delay)


# ── FFT helper ─────────────────────────────────────────────────────────────────
def compute_fft(samples: np.ndarray, sample_rate: int):
    N         = min(len(samples), 65536)
    fft_vals  = np.abs(fft(samples[:N])) / N
    fft_freqs = fftfreq(N, d=1.0 / sample_rate)
    pos_mask  = fft_freqs > 0
    return fft_freqs[pos_mask], fft_vals[pos_mask]


# ── Gap detection ──────────────────────────────────────────────────────────────
def detect_gaps(samples: np.ndarray, threshold=50, min_gap_samples=8):
    near_zero = np.abs(samples) < threshold
    gaps, in_gap, gap_start = [], False, 0
    for i, z in enumerate(near_zero):
        if z and not in_gap:
            in_gap, gap_start = True, i
        elif not z and in_gap:
            in_gap = False
            if (i - gap_start) >= min_gap_samples:
                gaps.append((gap_start, i))
    if in_gap and (len(samples) - gap_start) >= min_gap_samples:
        gaps.append((gap_start, len(samples)))
    return gaps


# ── Main analysis ──────────────────────────────────────────────────────────────
def analyze(wav_path: str):
    print(f"\n{'='*62}")
    print(f"  AcoustEEEcare — Bandpass Filter & Diagnostic")
    print(f"{'='*62}")
    print(f"  File      : {wav_path}")

    samples, file_rate = load_wav(wav_path)
    duration           = len(samples) / file_rate

    print(f"  Rate      : {file_rate} Hz  (expected {EXPECTED_RATE} Hz)")
    print(f"  Samples   : {len(samples)}")
    print(f"  Duration  : {duration:.3f} s")
    print(f"  RMS (raw) : {np.sqrt(np.mean(samples**2)):.1f}")

    # ── Apply bandpass ─────────────────────────────────────────────────────────
    print(f"\n  Applying Butterworth bandpass  {LOWCUT_HZ}–{HIGHCUT_HZ} Hz  "
          f"(order {FILTER_ORDER}) …")
    filtered = bandpass_filter(samples, file_rate, LOWCUT_HZ, HIGHCUT_HZ, FILTER_ORDER)
    print(f"  RMS (filtered): {np.sqrt(np.mean(filtered**2)):.1f}")

    # ── Save filtered WAV ──────────────────────────────────────────────────────
    png_path, filt_wav_path = make_output_paths(wav_path)
    save_wav(filt_wav_path, filtered, file_rate)
    print(f"\n  Filtered WAV saved to:")
    print(f"  {filt_wav_path}")

    # ── Gap detection on raw signal ────────────────────────────────────────────
    gaps = detect_gaps(samples)
    print(f"\n  Gaps in raw signal (near-zero runs >= 8 samples): {len(gaps)}")
    for i, (s, e) in enumerate(gaps[:5]):
        print(f"    gap {i+1}: sample {s}–{e}  ({(e-s)/file_rate*1000:.1f} ms)"
              f"  @ {s/file_rate:.3f} s")
    if len(gaps) > 5:
        print(f"    … and {len(gaps)-5} more")

    # ── Compute FFTs ───────────────────────────────────────────────────────────
    time_axis   = np.arange(len(samples))  / file_rate
    time_filt   = np.arange(len(filtered)) / file_rate

    raw_freqs,  raw_fft  = compute_fft(samples,  file_rate)
    filt_freqs, filt_fft = compute_fft(filtered, file_rate)

    # Spectrograms
    nperseg = min(256, len(samples))
    f_r, t_r, Sxx_r = signal.spectrogram(samples,  fs=file_rate,
                                          nperseg=nperseg, noverlap=nperseg//2)
    f_f, t_f, Sxx_f = signal.spectrogram(filtered, fs=file_rate,
                                          nperseg=nperseg, noverlap=nperseg//2)
    Sxx_r_db = 10 * np.log10(Sxx_r + 1e-10)
    Sxx_f_db = 10 * np.log10(Sxx_f + 1e-10)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 15))
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle(
        f"AcoustEEEcare — Bandpass Filter Analysis\n"
        f"{os.path.basename(wav_path)}   |   "
        f"{file_rate} Hz  ·  {duration:.2f} s  ·  "
        f"Bandpass {LOWCUT_HZ}–{HIGHCUT_HZ} Hz  (Butterworth order {FILTER_ORDER})",
        fontsize=13, fontweight="bold", color="white", y=0.98
    )

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.32,
                           top=0.93, bottom=0.06)

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#1a1a1a")
        ax.set_title(title, fontsize=9, color="#cccccc", pad=6)
        ax.set_xlabel(xlabel, fontsize=8, color="#999999")
        ax.set_ylabel(ylabel, fontsize=8, color="#999999")
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # ── Row 0: Full waveforms side-by-side ────────────────────────────────────
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_raw.plot(time_axis, samples, color="#2196F3", linewidth=0.35, alpha=0.85)
    ax_raw.axhline(0, color="#555555", linewidth=0.5, linestyle="--")
    for s, e in gaps:
        ax_raw.axvspan(s / file_rate, e / file_rate, color="red", alpha=0.30)
    ax_raw.set_xlim(0, duration)
    style_ax(ax_raw, "1  Raw Waveform", "Time (s)", "Amplitude (int16)")

    ax_flt = fig.add_subplot(gs[0, 1])
    ax_flt.plot(time_filt, filtered, color="#00BCD4", linewidth=0.35, alpha=0.85)
    ax_flt.axhline(0, color="#555555", linewidth=0.5, linestyle="--")
    ax_flt.set_xlim(0, duration)
    style_ax(ax_flt,
             f"2  Filtered Waveform  ({LOWCUT_HZ}–{HIGHCUT_HZ} Hz bandpass)",
             "Time (s)", "Amplitude (int16)")

    # ── Row 1: FFT spectra ─────────────────────────────────────────────────────
    ax_rfft = fig.add_subplot(gs[1, 0])
    ax_rfft.plot(raw_freqs, raw_fft, color="#9C27B0", linewidth=0.8)
    ax_rfft.axvspan(0,          LOWCUT_HZ,  color="#FF5722", alpha=0.18, label="filtered out")
    ax_rfft.axvspan(HIGHCUT_HZ, file_rate/2, color="#FF5722", alpha=0.18)
    ax_rfft.axvline(LOWCUT_HZ,  color="#FF5722", linewidth=1.2, linestyle="--",
                    label=f"{LOWCUT_HZ} Hz")
    ax_rfft.axvline(HIGHCUT_HZ, color="#FF9800", linewidth=1.2, linestyle="--",
                    label=f"{HIGHCUT_HZ} Hz")
    ax_rfft.set_xlim(0, file_rate / 2)
    ax_rfft.legend(fontsize=7, facecolor="#222222", labelcolor="white")
    style_ax(ax_rfft, "3  FFT Spectrum — Raw", "Frequency (Hz)", "Magnitude")

    ax_ffft = fig.add_subplot(gs[1, 1])
    ax_ffft.plot(filt_freqs, filt_fft, color="#4CAF50", linewidth=0.8)
    ax_ffft.axvline(LOWCUT_HZ,  color="#FF5722", linewidth=1.2, linestyle="--",
                    label=f"{LOWCUT_HZ} Hz")
    ax_ffft.axvline(HIGHCUT_HZ, color="#FF9800", linewidth=1.2, linestyle="--",
                    label=f"{HIGHCUT_HZ} Hz")
    ax_ffft.set_xlim(0, file_rate / 2)
    ax_ffft.legend(fontsize=7, facecolor="#222222", labelcolor="white")
    style_ax(ax_ffft,
             f"4  FFT Spectrum — Filtered ({LOWCUT_HZ}–{HIGHCUT_HZ} Hz)",
             "Frequency (Hz)", "Magnitude")

    # ── Row 2: Filter frequency response ──────────────────────────────────────
    nyq = file_rate / 2.0
    b, a = signal.butter(FILTER_ORDER,
                         [LOWCUT_HZ / nyq, HIGHCUT_HZ / nyq], btype="band")
    w, h = signal.freqz(b, a, worN=4096, fs=file_rate)

    ax_resp = fig.add_subplot(gs[2, :])
    ax_resp.plot(w, 20 * np.log10(np.abs(h) + 1e-10),
                 color="#FF9800", linewidth=1.3)
    ax_resp.axvline(LOWCUT_HZ,  color="#FF5722", linewidth=1.2, linestyle="--",
                    label=f"Low cut  {LOWCUT_HZ} Hz")
    ax_resp.axvline(HIGHCUT_HZ, color="#4CAF50", linewidth=1.2, linestyle="--",
                    label=f"High cut {HIGHCUT_HZ} Hz")
    ax_resp.axhline(-3, color="#888888", linewidth=0.8, linestyle=":",
                    label="-3 dB")
    ax_resp.set_xlim(0, min(file_rate / 2, 500))   # zoom to 0–500 Hz for clarity
    ax_resp.set_ylim(-80, 5)
    ax_resp.legend(fontsize=8, facecolor="#222222", labelcolor="white")
    style_ax(ax_resp,
             f"5  Filter Frequency Response — Butterworth Bandpass "
             f"{LOWCUT_HZ}–{HIGHCUT_HZ} Hz  (order {FILTER_ORDER})",
             "Frequency (Hz)", "Gain (dB)")

    # ── Row 3: Spectrograms ────────────────────────────────────────────────────
    vmin = np.percentile(Sxx_r_db, 10)

    ax_sg_r = fig.add_subplot(gs[3, 0])
    im1 = ax_sg_r.pcolormesh(t_r, f_r, Sxx_r_db, shading="gouraud",
                              cmap="inferno", vmin=vmin)
    fig.colorbar(im1, ax=ax_sg_r).ax.yaxis.set_tick_params(
        color="#888888", labelsize=7)
    ax_sg_r.set_ylim(0, min(file_rate / 2, 500))
    style_ax(ax_sg_r, "6  Spectrogram — Raw", "Time (s)", "Frequency (Hz)")

    ax_sg_f = fig.add_subplot(gs[3, 1])
    im2 = ax_sg_f.pcolormesh(t_f, f_f, Sxx_f_db, shading="gouraud",
                              cmap="inferno", vmin=vmin)
    fig.colorbar(im2, ax=ax_sg_f).ax.yaxis.set_tick_params(
        color="#888888", labelsize=7)
    ax_sg_f.axhline(LOWCUT_HZ,  color="cyan",   linewidth=0.9,
                    linestyle="--", alpha=0.8)
    ax_sg_f.axhline(HIGHCUT_HZ, color="#00FF88", linewidth=0.9,
                    linestyle="--", alpha=0.8)
    ax_sg_f.set_ylim(0, min(file_rate / 2, 500))
    style_ax(ax_sg_f,
             f"7  Spectrogram — Filtered  (cyan={LOWCUT_HZ} Hz  green={HIGHCUT_HZ} Hz)",
             "Time (s)", "Frequency (Hz)")

    # ── Bottom summary bar ─────────────────────────────────────────────────────
    summary = (
        f"Rate: {file_rate} Hz  |  Duration: {duration:.2f} s  |  "
        f"Samples: {len(samples)}  |  "
        f"RMS raw: {np.sqrt(np.mean(samples**2)):.1f}  |  "
        f"RMS filtered: {np.sqrt(np.mean(filtered**2)):.1f}  |  "
        f"Gaps (raw): {len(gaps)}  |  "
        f"Bandpass: {LOWCUT_HZ}–{HIGHCUT_HZ} Hz  order {FILTER_ORDER}"
    )
    fig.text(0.5, 0.005, summary, ha="center", fontsize=8.5, color="#aaaaaa",
             bbox=dict(boxstyle="round", facecolor="#1e1e1e", alpha=0.9))

    # ── Save plot ──────────────────────────────────────────────────────────────
    plt.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Plot saved to:")
    print(f"  {png_path}")
    print(f"{'='*62}\n")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wav_path = resolve_wav_path()
    print(f"\n  Auto-selected WAV: {os.path.basename(wav_path)}")
    analyze(wav_path)
