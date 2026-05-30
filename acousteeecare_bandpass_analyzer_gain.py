"""
AcoustEEEcare — Bandpass Filter & Diagnostic Visualizer
=========================================================
Just press Run in VS Code — no terminal arguments needed.

- Automatically picks the LATEST .wav file in RECORDINGS_DIR
- Applies TWO Butterworth bandpass filters:
    • Main bandpass  (LOWCUT_HZ  – HIGHCUT_HZ,  default 10–200 Hz)
    • Respiratory    (RESP_LOWCUT_HZ – RESP_HIGHCUT_HZ, default 100–1000 Hz)
- Amplifies each result so it is actually audible on headphones/speakers
- Optionally pitch-shifts the MAIN bandpass output for infrasonic content
- Saves the MAIN FILTERED audio as a new .wav file
- Saves a side-by-side analysis PNG into a subfolder:
    <RECORDINGS_DIR>/analysis_bandpass_gain_for_mfcc/

Settings to change:
    RECORDINGS_DIR    → folder where your .wav files are saved
    WAV_FILE          → pin a specific filename, or None = auto-newest
    LOWCUT_HZ         → main bandpass lower edge
    HIGHCUT_HZ        → main bandpass upper edge
    RESP_LOWCUT_HZ    → respiratory bandpass lower edge  (default 100 Hz)
    RESP_HIGHCUT_HZ   → respiratory bandpass upper edge  (default 1000 Hz)
    FILTER_ORDER      → Butterworth filter order (shared)
    EXPECTED_RATE     → must match firmware SAMPLING_RATE
    AMPLIFY_GAIN      → linear gain for MAIN (heart) bandpass — applied to
                        BOTH the saved WAV and the figure plots
    RESP_AMPLIFY_GAIN → linear gain for RESPIRATORY bandpass — applied to
                        the figure plots (resp WAV is not saved separately)
    PITCH_SHIFT       → output sample-rate multiplier for MAIN bandpass WAV

Heart sound presets:
    Pulse-only (carotid/wrist):    LOW=10,  HIGH=200,  GAIN=50,  PITCH=4.0
    Audible heart sounds (S1/S2):  LOW=30,  HIGH=600,  GAIN=30,  PITCH=1.0
    Lung sounds (wheezes/crackles):LOW=100, HIGH=2000, GAIN=10,  PITCH=1.0
    Full stethoscope band:         LOW=20,  HIGH=1000, GAIN=20,  PITCH=1.0
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

RECORDINGS_DIR = r"D:\zephyrdev\AcoustEEEcare\recording_v84_oversampled_gain1_4_10us_battery_thin_silicon_2"

WAV_FILE = None

# ── Main bandpass ──────────────────────────────────────────────
LOWCUT_HZ    = 10
HIGHCUT_HZ   = 200

# ── Respiratory bandpass ───────────────────────────────────────
RESP_LOWCUT_HZ  = 100
RESP_HIGHCUT_HZ = 1000

FILTER_ORDER  = 4
EXPECTED_RATE = 8000

# ── Gain settings ─────────────────────────────────────────────
AMPLIFY_GAIN      = 1000.0
RESP_AMPLIFY_GAIN = 1000.0
PITCH_SHIFT       = 1.0

# ══════════════════════════════════════════════════════════════
# ▲▲▲  EDIT THESE SETTINGS  ▲▲▲
# ══════════════════════════════════════════════════════════════


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


def make_output_paths(wav_path: str):
    folder     = os.path.dirname(wav_path)
    out_folder = os.path.join(folder, "analysis_bandpass_gain_for_mfcc")
    os.makedirs(out_folder, exist_ok=True)
    basename   = os.path.splitext(os.path.basename(wav_path))[0]
    png_path   = os.path.join(out_folder, f"{basename}_bandpass_analysis.png")
    heart_parts = [f"heart_{LOWCUT_HZ}-{HIGHCUT_HZ}Hz", f"x{AMPLIFY_GAIN:g}"]
    if PITCH_SHIFT != 1.0:
        heart_parts.append(f"pitch{PITCH_SHIFT:g}")
    heart_wav_path = os.path.join(out_folder, f"{basename}_{'_'.join(heart_parts)}.wav")
    resp_parts = [f"resp_{RESP_LOWCUT_HZ}-{RESP_HIGHCUT_HZ}Hz", f"x{RESP_AMPLIFY_GAIN:g}"]
    resp_wav_path = os.path.join(out_folder, f"{basename}_{'_'.join(resp_parts)}.wav")
    return png_path, heart_wav_path, resp_wav_path


def load_wav(path: str):
    with wave.open(path, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_rate = wf.getframerate()
        n_frames    = wf.getnframes()
        raw         = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if n_channels > 1:
        samples = samples[::n_channels]
    return samples, sample_rate


def save_wav(path: str, samples: np.ndarray, sample_rate: int,
             gain: float = 1.0, pitch_shift: float = 1.0):
    # Measure clipping BEFORE applying clip, so the warning is accurate
    clipped = np.sum(np.abs(samples * gain) > 32767)
    pcm = np.clip(samples * gain, -32768, 32767).astype(np.int16)
    output_rate = int(round(sample_rate * pitch_shift))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(output_rate)
        wf.writeframes(pcm.tobytes())
    if clipped > 0:
        pct = 100.0 * clipped / len(samples)
        print(f"  NOTE: {clipped} samples ({pct:.2f}%) clipped after gain x{gain}")


def bandpass_filter(samples: np.ndarray, sample_rate: int,
                    lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
    nyq  = sample_rate / 2.0
    low  = max(lowcut  / nyq, 1e-4)
    high = min(highcut / nyq, 1.0 - 1e-4)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, samples)


def compute_fft(samples: np.ndarray, sample_rate: int):
    N         = min(len(samples), 65536)
    fft_vals  = np.abs(fft(samples[:N])) / N
    fft_freqs = fftfreq(N, d=1.0 / sample_rate)
    pos_mask  = fft_freqs > 0
    return fft_freqs[pos_mask], fft_vals[pos_mask]


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


def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor("#1a1a1a")
    ax.set_title(title, fontsize=9, color="#cccccc", pad=6)
    ax.set_xlabel(xlabel, fontsize=8, color="#999999")
    ax.set_ylabel(ylabel, fontsize=8, color="#999999")
    ax.tick_params(colors="#888888", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


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

    # ── Apply main bandpass ────────────────────────────────────────────────────
    print(f"\n  Applying main bandpass  {LOWCUT_HZ}–{HIGHCUT_HZ} Hz  "
          f"(order {FILTER_ORDER}) …")
    filtered_raw  = bandpass_filter(samples, file_rate, LOWCUT_HZ, HIGHCUT_HZ, FILTER_ORDER)
    rms_filt      = np.sqrt(np.mean(filtered_raw**2))
    filtered      = np.clip(filtered_raw * AMPLIFY_GAIN, -32768, 32767)
    rms_filt_gain = np.sqrt(np.mean(filtered**2))
    print(f"  RMS (filtered, pre-gain)       : {rms_filt:.1f}")
    print(f"  RMS (filtered, gain x{AMPLIFY_GAIN:g})  : {rms_filt_gain:.1f}  (clipped at 32767)")

    # ── Apply respiratory bandpass ─────────────────────────────────────────────
    print(f"\n  Applying respiratory bandpass  {RESP_LOWCUT_HZ}–{RESP_HIGHCUT_HZ} Hz  "
          f"(order {FILTER_ORDER}) …")
    resp_filtered_raw = bandpass_filter(samples, file_rate,
                                        RESP_LOWCUT_HZ, RESP_HIGHCUT_HZ, FILTER_ORDER)
    rms_resp      = np.sqrt(np.mean(resp_filtered_raw**2))
    resp_filtered = np.clip(resp_filtered_raw * RESP_AMPLIFY_GAIN, -32768, 32767)
    rms_resp_gain = np.sqrt(np.mean(resp_filtered**2))
    print(f"  RMS (resp, pre-gain)           : {rms_resp:.1f}")
    print(f"  RMS (resp, gain x{RESP_AMPLIFY_GAIN:g})      : {rms_resp_gain:.1f}  (clipped at 32767)")

    if PITCH_SHIFT != 1.0:
        print(f"  Pitch shift (main WAV)     : x{PITCH_SHIFT:g}  "
              f"(WAV header rate = {int(round(file_rate * PITCH_SHIFT))} Hz)")

    # ── Save WAVs — pass unclipped raw arrays so save_wav can report clipping ─
    png_path, heart_wav_path, resp_wav_path = make_output_paths(wav_path)
    save_wav(heart_wav_path, filtered_raw, file_rate,
             gain=AMPLIFY_GAIN, pitch_shift=PITCH_SHIFT)
    print(f"\n  Heart WAV saved to:\n  {heart_wav_path}")
    save_wav(resp_wav_path, resp_filtered_raw, file_rate,
             gain=RESP_AMPLIFY_GAIN, pitch_shift=1.0)
    print(f"  Respiratory WAV saved to:\n  {resp_wav_path}")

    # ── Gap detection ──────────────────────────────────────────────────────────
    gaps = detect_gaps(samples)
    print(f"\n  Gaps in raw signal (near-zero runs >= 8 samples): {len(gaps)}")
    for i, (s, e) in enumerate(gaps[:5]):
        print(f"    gap {i+1}: sample {s}–{e}  ({(e-s)/file_rate*1000:.1f} ms)"
              f"  @ {s/file_rate:.3f} s")
    if len(gaps) > 5:
        print(f"    … and {len(gaps)-5} more")

    # ── Time axes ─────────────────────────────────────────────────────────────
    time_axis = np.arange(len(samples))       / file_rate
    time_filt = np.arange(len(filtered))      / file_rate
    time_resp = np.arange(len(resp_filtered)) / file_rate

    # ── FFTs ──────────────────────────────────────────────────────────────────
    raw_freqs,  raw_fft  = compute_fft(samples,       file_rate)
    filt_freqs, filt_fft = compute_fft(filtered,      file_rate)
    resp_freqs, resp_fft = compute_fft(resp_filtered,  file_rate)

    # ── Spectrograms ───────────────────────────────────────────────────────────
    nperseg = min(256, len(samples))
    f_r, t_r, Sxx_r = signal.spectrogram(samples,       fs=file_rate,
                                          nperseg=nperseg, noverlap=nperseg//2)
    f_f, t_f, Sxx_f = signal.spectrogram(filtered,      fs=file_rate,
                                          nperseg=nperseg, noverlap=nperseg//2)
    f_p, t_p, Sxx_p = signal.spectrogram(resp_filtered, fs=file_rate,
                                          nperseg=nperseg, noverlap=nperseg//2)
    Sxx_r_db = 10 * np.log10(Sxx_r + 1e-10)
    Sxx_f_db = 10 * np.log10(Sxx_f + 1e-10)
    Sxx_p_db = 10 * np.log10(Sxx_p + 1e-10)

    # ── Shared y-axis limits for waveform row ─────────────────────────────────
    # All three waveform plots share the same y-axis so the gain amplification
    # is visually apparent: the raw signal sits small, the gain-amplified
    # filtered signals appear proportionally taller in the same frame.
    raw_peak  = float(np.max(np.abs(samples)))
    filt_peak = float(np.max(np.abs(filtered)))
    resp_peak = float(np.max(np.abs(resp_filtered)))
    shared_ylim = max(raw_peak, filt_peak, resp_peak) * 1.05
    shared_ylim = max(shared_ylim, 1.0)   # avoid zero-height axis on silence

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0f0f0f")

    title_extras = []
    if AMPLIFY_GAIN != 1.0:
        title_extras.append(f"gain x{AMPLIFY_GAIN:g}")
    if PITCH_SHIFT != 1.0:
        title_extras.append(f"pitch x{PITCH_SHIFT:g}")
    extra_str = ("  ·  " + "  ·  ".join(title_extras)) if title_extras else ""

    fig.suptitle(
        f"AcoustEEEcare — Bandpass Filter Analysis\n"
        f"{os.path.basename(wav_path)}   |   "
        f"{file_rate} Hz  ·  {duration:.2f} s  ·  "
        f"Main: {LOWCUT_HZ}–{HIGHCUT_HZ} Hz  ·  "
        f"Respiratory: {RESP_LOWCUT_HZ}–{RESP_HIGHCUT_HZ} Hz  "
        f"(Butterworth order {FILTER_ORDER}){extra_str}",
        fontsize=13, fontweight="bold", color="white", y=0.98
    )

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.30,
                           top=0.93, bottom=0.06)

    # ── Row 0: Waveforms — shared y-axis so gain is visible ───────────────────
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_raw.plot(time_axis, samples, color="#2196F3", linewidth=0.35, alpha=0.85)
    ax_raw.axhline(0, color="#555555", linewidth=0.5, linestyle="--")
    for s, e in gaps:
        ax_raw.axvspan(s / file_rate, e / file_rate, color="red", alpha=0.30)
    ax_raw.set_xlim(0, duration)
    ax_raw.set_ylim(-shared_ylim, shared_ylim)
    style_ax(ax_raw, "1  Raw Waveform  (shared y-axis — gain makes cols 2 & 3 taller)",
             "Time (s)", "Amplitude (int16)")

    ax_flt = fig.add_subplot(gs[0, 1])
    ax_flt.plot(time_filt, filtered, color="#00BCD4", linewidth=0.35, alpha=0.85)
    ax_flt.axhline(0, color="#555555", linewidth=0.5, linestyle="--")
    ax_flt.set_xlim(0, duration)
    ax_flt.set_ylim(-shared_ylim, shared_ylim)
    style_ax(ax_flt,
             f"2  Main Bandpass  ({LOWCUT_HZ}–{HIGHCUT_HZ} Hz)  ×gain {AMPLIFY_GAIN:g}",
             "Time (s)", "Amplitude (gain-scaled, shared y)")

    ax_res = fig.add_subplot(gs[0, 2])
    ax_res.plot(time_resp, resp_filtered, color="#FF9800", linewidth=0.35, alpha=0.85)
    ax_res.axhline(0, color="#555555", linewidth=0.5, linestyle="--")
    ax_res.set_xlim(0, duration)
    ax_res.set_ylim(-shared_ylim, shared_ylim)
    style_ax(ax_res,
             f"3  Respiratory Bandpass  ({RESP_LOWCUT_HZ}–{RESP_HIGHCUT_HZ} Hz)  ×gain {RESP_AMPLIFY_GAIN:g}",
             "Time (s)", "Amplitude (gain-scaled, shared y)")

    # ── Row 1: FFT spectra ─────────────────────────────────────────────────────
    ax_rfft = fig.add_subplot(gs[1, 0])
    ax_rfft.plot(raw_freqs, raw_fft, color="#9C27B0", linewidth=0.8)
    ax_rfft.axvspan(0,           LOWCUT_HZ,   color="#FF5722", alpha=0.18, label="main filtered out")
    ax_rfft.axvspan(HIGHCUT_HZ,  file_rate/2, color="#FF5722", alpha=0.18)
    ax_rfft.axvline(LOWCUT_HZ,   color="#FF5722",  linewidth=1.2, linestyle="--",
                    label=f"main {LOWCUT_HZ} Hz")
    ax_rfft.axvline(HIGHCUT_HZ,  color="#FF9800",  linewidth=1.2, linestyle="--",
                    label=f"main {HIGHCUT_HZ} Hz")
    ax_rfft.axvline(RESP_LOWCUT_HZ,  color="#00BCD4", linewidth=1.0, linestyle=":",
                    label=f"resp {RESP_LOWCUT_HZ} Hz")
    ax_rfft.axvline(RESP_HIGHCUT_HZ, color="#4CAF50", linewidth=1.0, linestyle=":",
                    label=f"resp {RESP_HIGHCUT_HZ} Hz")
    ax_rfft.set_xlim(0, file_rate / 2)
    ax_rfft.legend(fontsize=6, facecolor="#222222", labelcolor="white")
    style_ax(ax_rfft, "4  FFT Spectrum — Raw", "Frequency (Hz)", "Magnitude")

    ax_ffft = fig.add_subplot(gs[1, 1])
    ax_ffft.plot(filt_freqs, filt_fft, color="#00BCD4", linewidth=0.8)
    ax_ffft.axvline(LOWCUT_HZ,  color="#FF5722", linewidth=1.2, linestyle="--",
                    label=f"{LOWCUT_HZ} Hz")
    ax_ffft.axvline(HIGHCUT_HZ, color="#FF9800", linewidth=1.2, linestyle="--",
                    label=f"{HIGHCUT_HZ} Hz")
    ax_ffft.set_xlim(0, file_rate / 2)
    ax_ffft.legend(fontsize=7, facecolor="#222222", labelcolor="white")
    style_ax(ax_ffft,
             f"5  FFT Spectrum — Main Bandpass ({LOWCUT_HZ}–{HIGHCUT_HZ} Hz)",
             "Frequency (Hz)", "Magnitude")

    ax_rfft2 = fig.add_subplot(gs[1, 2])
    ax_rfft2.plot(resp_freqs, resp_fft, color="#FF9800", linewidth=0.8)
    ax_rfft2.axvline(RESP_LOWCUT_HZ,  color="#00BCD4", linewidth=1.2, linestyle="--",
                     label=f"{RESP_LOWCUT_HZ} Hz")
    ax_rfft2.axvline(RESP_HIGHCUT_HZ, color="#4CAF50", linewidth=1.2, linestyle="--",
                     label=f"{RESP_HIGHCUT_HZ} Hz")
    ax_rfft2.set_xlim(0, file_rate / 2)
    ax_rfft2.legend(fontsize=7, facecolor="#222222", labelcolor="white")
    style_ax(ax_rfft2,
             f"6  FFT Spectrum — Respiratory ({RESP_LOWCUT_HZ}–{RESP_HIGHCUT_HZ} Hz)",
             "Frequency (Hz)", "Magnitude")

    # ── Row 2: Filter frequency responses ─────────────────────────────────────
    nyq = file_rate / 2.0
    b_main, a_main = signal.butter(FILTER_ORDER,
                                   [LOWCUT_HZ / nyq, HIGHCUT_HZ / nyq], btype="band")
    w_main, h_main = signal.freqz(b_main, a_main, worN=4096, fs=file_rate)
    b_resp, a_resp = signal.butter(FILTER_ORDER,
                                   [RESP_LOWCUT_HZ / nyq, RESP_HIGHCUT_HZ / nyq], btype="band")
    w_resp, h_resp = signal.freqz(b_resp, a_resp, worN=4096, fs=file_rate)

    ax_resp_curve = fig.add_subplot(gs[2, :])
    ax_resp_curve.plot(w_main, 20 * np.log10(np.abs(h_main) + 1e-10),
                       color="#00BCD4", linewidth=1.5,
                       label=f"Main bandpass  {LOWCUT_HZ}–{HIGHCUT_HZ} Hz")
    ax_resp_curve.plot(w_resp, 20 * np.log10(np.abs(h_resp) + 1e-10),
                       color="#FF9800", linewidth=1.5, linestyle="--",
                       label=f"Respiratory bandpass  {RESP_LOWCUT_HZ}–{RESP_HIGHCUT_HZ} Hz")
    for freq, col in [(LOWCUT_HZ, "#FF5722"), (HIGHCUT_HZ, "#4CAF50"),
                      (RESP_LOWCUT_HZ, "#FF9800"), (RESP_HIGHCUT_HZ, "#FFEB3B")]:
        ax_resp_curve.axvline(freq, color=col, linewidth=0.9, linestyle=":",
                              alpha=0.7, label=f"{freq} Hz")
    ax_resp_curve.axhline(-3, color="#888888", linewidth=0.8, linestyle=":", label="-3 dB")
    x_max = min(file_rate / 2, max(HIGHCUT_HZ, RESP_HIGHCUT_HZ) * 1.3)
    ax_resp_curve.set_xlim(0, x_max)
    ax_resp_curve.set_ylim(-80, 5)
    ax_resp_curve.legend(fontsize=8, facecolor="#222222", labelcolor="white", ncol=3)
    style_ax(ax_resp_curve,
             f"7  Filter Frequency Responses — Butterworth (order {FILTER_ORDER})  "
             f"|  Main: {LOWCUT_HZ}–{HIGHCUT_HZ} Hz  ·  "
             f"Respiratory: {RESP_LOWCUT_HZ}–{RESP_HIGHCUT_HZ} Hz",
             "Frequency (Hz)", "Gain (dB)")

    # ── Row 3: Spectrograms ────────────────────────────────────────────────────
    vmin = np.percentile(Sxx_r_db, 10)

    ax_sg_r = fig.add_subplot(gs[3, 0])
    im1 = ax_sg_r.pcolormesh(t_r, f_r, Sxx_r_db, shading="gouraud",
                              cmap="inferno", vmin=vmin)
    fig.colorbar(im1, ax=ax_sg_r).ax.yaxis.set_tick_params(color="#888888", labelsize=7)
    ax_sg_r.set_ylim(0, min(file_rate / 2, 1100))
    style_ax(ax_sg_r, "8  Spectrogram — Raw", "Time (s)", "Frequency (Hz)")

    ax_sg_f = fig.add_subplot(gs[3, 1])
    im2 = ax_sg_f.pcolormesh(t_f, f_f, Sxx_f_db, shading="gouraud",
                              cmap="inferno", vmin=vmin)
    fig.colorbar(im2, ax=ax_sg_f).ax.yaxis.set_tick_params(color="#888888", labelsize=7)
    ax_sg_f.axhline(LOWCUT_HZ,  color="cyan",    linewidth=0.9, linestyle="--", alpha=0.8)
    ax_sg_f.axhline(HIGHCUT_HZ, color="#00FF88", linewidth=0.9, linestyle="--", alpha=0.8)
    ax_sg_f.set_ylim(0, min(file_rate / 2, 1100))
    style_ax(ax_sg_f,
             f"9  Spectrogram — Main BP  (cyan={LOWCUT_HZ} Hz  green={HIGHCUT_HZ} Hz)",
             "Time (s)", "Frequency (Hz)")

    ax_sg_p = fig.add_subplot(gs[3, 2])
    im3 = ax_sg_p.pcolormesh(t_p, f_p, Sxx_p_db, shading="gouraud",
                              cmap="inferno", vmin=vmin)
    fig.colorbar(im3, ax=ax_sg_p).ax.yaxis.set_tick_params(color="#888888", labelsize=7)
    ax_sg_p.axhline(RESP_LOWCUT_HZ,  color="#FF9800", linewidth=0.9, linestyle="--", alpha=0.8)
    ax_sg_p.axhline(RESP_HIGHCUT_HZ, color="#FFEB3B", linewidth=0.9, linestyle="--", alpha=0.8)
    ax_sg_p.set_ylim(0, min(file_rate / 2, 1100))
    style_ax(ax_sg_p,
             f"10  Spectrogram — Respiratory BP  "
             f"(orange={RESP_LOWCUT_HZ} Hz  yellow={RESP_HIGHCUT_HZ} Hz)",
             "Time (s)", "Frequency (Hz)")

    # ── Bottom summary bar ─────────────────────────────────────────────────────
    summary = (
        f"Rate: {file_rate} Hz  |  Duration: {duration:.2f} s  |  "
        f"Samples: {len(samples)}  |  "
        f"RMS raw: {np.sqrt(np.mean(samples**2)):.1f}  |  "
        f"RMS main BP (pre-gain): {rms_filt:.1f}  →  x{AMPLIFY_GAIN:g}  →  {rms_filt_gain:.1f}  |  "
        f"RMS resp BP (pre-gain): {rms_resp:.1f}  →  x{RESP_AMPLIFY_GAIN:g}  →  {rms_resp_gain:.1f}  |  "
        f"Gaps (raw): {len(gaps)}  |  "
        f"Main BP: {LOWCUT_HZ}–{HIGHCUT_HZ} Hz  |  "
        f"Resp BP: {RESP_LOWCUT_HZ}–{RESP_HIGHCUT_HZ} Hz  |  "
        f"Order: {FILTER_ORDER}  |  "
        f"Pitch: x{PITCH_SHIFT:g}"
    )
    fig.text(0.5, 0.005, summary, ha="center", fontsize=8, color="#aaaaaa",
             bbox=dict(boxstyle="round", facecolor="#1e1e1e", alpha=0.9))

    plt.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n  Plot saved to:\n  {png_path}")
    print(f"{'='*62}\n")
    plt.show()


if __name__ == "__main__":
    wav_path = resolve_wav_path()
    print(f"\n  Auto-selected WAV: {os.path.basename(wav_path)}")
    analyze(wav_path)