"""
AcoustEEEcare — Spectral Noise Removal
=======================================
Removes internal ADC/hardware noise from a microphone recording using
spectral subtraction, then applies bandpass filtering (10–200 Hz).

Outputs:
  - <mic>_denoised.wav       — cleaned audio file
  - <mic>_noise_analysis.png — 4-panel comparison figure

Usage:
  python noise_remover.py <noise.wav> <mic_recording.wav> [output_dir]

Example:
  python noise_remover.py internal_noise.wav mic_recording.wav ./results

Pipeline:
  1. Load both WAVs, resample to TARGET_SR
  2. Spectral subtraction  (noise floor estimated from noise profile)
  3. DC removal            (mean subtraction)
  4. Bandpass filter       (10–200 Hz, 4th-order Butterworth)
  5. Amplitude normalize
  6. Visualize waveform, spectrogram, MFCC (before vs after)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, resample_poly
import librosa
import librosa.display
import sys
import os
from math import gcd

# ─────────────────────────── CONFIGURATION ──────────────────────────────────

TARGET_SR         = 8000        # match firmware sample rate
BANDPASS_LOW_HZ   = 10
BANDPASS_HIGH_HZ  = 200
FILTER_ORDER      = 4

# MFCC / spectrogram params
N_FFT             = 1024
WIN_MS            = 25
OVERLAP_RATIO     = 0.30
N_MELS            = 20
N_MFCC            = 13

WIN_LENGTH        = int(TARGET_SR * WIN_MS / 1000)
HOP_LENGTH        = int(WIN_LENGTH * (1 - OVERLAP_RATIO))

# Spectral subtraction tuning
# alpha: over-subtraction factor (1.0–3.0). Higher = more aggressive removal
# beta:  spectral floor (prevents musical noise). 0.001–0.05 typical
ALPHA             = 2.0
BETA              = 0.01

# ──────────────────────────── I/O UTILITIES ─────────────────────────────────

def load_wav(path: str):
    """Load WAV and convert to float32 in [-1, 1]."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(audio, target_sr // g, orig_sr // g)


def save_wav(path, audio, sr):
    out = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sr, (out * 32767).astype(np.int16))
    print(f"  [✓] WAV saved → {path}")

# ─────────────────────────── PROCESSING STAGES ──────────────────────────────

def remove_dc(audio):
    return audio - np.mean(audio)


def bandpass_filter(audio, sr, low=10, high=200, order=4):
    nyq  = sr / 2.0
    low  = np.clip(low  / nyq, 1e-6, 1.0 - 1e-6)
    high = np.clip(high / nyq, 1e-6, 1.0 - 1e-6)
    sos  = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, audio)


def normalize_amplitude(audio):
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio


def spectral_subtraction(mic_audio, noise_audio, sr,
                          alpha=ALPHA, beta=BETA,
                          n_fft=N_FFT, hop=None):
    """
    Frequency-domain spectral subtraction.

    Steps:
      1. STFT of both signals.
      2. Estimate noise power spectrum from the noise profile.
      3. Subtract alpha * noise power from mic magnitude spectrum.
      4. Apply spectral floor (beta * noise power) to avoid musical noise.
      5. Reconstruct via iSTFT using original mic phase.

    Parameters
    ----------
    alpha : over-subtraction factor (aggressiveness)
    beta  : spectral floor multiplier (musical-noise suppression)
    """
    if hop is None:
        hop = n_fft // 4

    # STFT
    mic_stft   = librosa.stft(mic_audio,   n_fft=n_fft, hop_length=hop)
    noise_stft = librosa.stft(noise_audio, n_fft=n_fft, hop_length=hop)

    mic_mag, mic_phase     = np.abs(mic_stft),   np.angle(mic_stft)
    noise_mag              = np.abs(noise_stft)

    # Mean noise power spectrum across all noise frames
    noise_power = np.mean(noise_mag ** 2, axis=1, keepdims=True)  # (bins, 1)

    # Spectral subtraction: max(|X|^2 - alpha*|N|^2, beta*|N|^2)
    mic_power   = mic_mag ** 2
    clean_power = np.maximum(mic_power - alpha * noise_power,
                             beta      * noise_power)
    clean_mag   = np.sqrt(clean_power)

    # Reconstruct with original mic phase
    clean_stft  = clean_mag * np.exp(1j * mic_phase)
    clean_audio = librosa.istft(clean_stft, hop_length=hop, length=len(mic_audio))

    return clean_audio.astype(np.float32)

# ─────────────────────────── ANALYSIS HELPERS ───────────────────────────────

def compute_spectrogram(audio):
    D = librosa.stft(audio, n_fft=N_FFT,
                     hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
    return librosa.amplitude_to_db(np.abs(D), ref=np.max)


def compute_mfcc(audio):
    return librosa.feature.mfcc(
        y=audio, sr=TARGET_SR,
        n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=N_MELS, window='hann',
    )

# ─────────────────────────── PALETTE ────────────────────────────────────────

P = {
    "bg"      : "#090c11",
    "panel"   : "#0e121a",
    "text"    : "#e2eaf6",
    "subtext" : "#4a5a72",
    "grid"    : "#161d2b",
    "accent"  : "#00ffe7",
    "raw"     : "#38bdf8",   # sky blue  — raw mic
    "noise"   : "#f87171",   # red       — noise profile
    "clean"   : "#34d399",   # emerald   — after spectral sub
    "final"   : "#f472b6",   # pink      — after bandpass + norm
    "cmap_raw"   : "Blues",
    "cmap_noise" : "Reds",
    "cmap_clean" : "Greens",
    "cmap_final" : "RdPu",
}

# ─────────────────────────── FIGURE ─────────────────────────────────────────

def plot_all(noise_audio, mic_raw, mic_clean, mic_final, out_path):
    """
    5-column figure:
      Col 0: Noise Profile
      Col 1: Raw Mic
      Col 2: After Spectral Subtraction
      Col 3: After Bandpass (10–200 Hz)
      Col 4: Bandpass + Normalized  (final)

    3 rows: Waveform | Spectrogram | MFCC
    """
    signals = [noise_audio, mic_raw, mic_clean, mic_final]
    labels  = [
        "① NOISE PROFILE",
        "② RAW MIC",
        "③ SPECTRAL SUBTRACTION",
        "④ BANDPASS + NORMALIZED",
    ]
    colors    = [P["noise"], P["raw"], P["clean"], P["final"]]
    spec_maps = [P["cmap_noise"], P["cmap_raw"], P["cmap_clean"], P["cmap_final"]]
    mfcc_maps = ["Oranges", "Blues_r", "Greens_r", "RdPu"]

    specs = [compute_spectrogram(s) for s in signals]
    mfccs = [compute_mfcc(s)        for s in signals]
    times = [np.linspace(0, len(s) / TARGET_SR, len(s)) for s in signals]

    ncols = len(signals)
    fig = plt.figure(figsize=(24, 13), facecolor=P["bg"])
    fig.suptitle(
        "AcoustEEEcare  ·  Spectral Noise Removal  →  Bandpass Filter  →  Normalize",
        color=P["text"], fontsize=12, fontfamily="monospace",
        fontweight="bold", y=0.978,
    )

    gs = gridspec.GridSpec(
        3, ncols, figure=fig,
        hspace=0.52, wspace=0.30,
        left=0.05, right=0.975,
        top=0.935, bottom=0.07,
    )

    for col, (sig, label, color, scmap, mcmap) in \
            enumerate(zip(signals, labels, colors, spec_maps, mfcc_maps)):

        t = times[col]

        # Column header
        pos0 = gs[0, col].get_position(fig)
        cx   = (pos0.x0 + pos0.x1) / 2
        fig.text(cx, 0.948, label,
                 ha='center', va='bottom',
                 color=color, fontsize=8.5,
                 fontfamily="monospace", fontweight="bold")

        # ── Row 0: Waveform ──────────────────────────────────────────
        ax_w = fig.add_subplot(gs[0, col])
        ax_w.set_facecolor(P["panel"])
        ax_w.plot(t, sig, color=color, linewidth=0.5, alpha=0.9)
        ax_w.fill_between(t, sig, alpha=0.07, color=color)
        ax_w.axhline(0, color=P["subtext"], linewidth=0.6,
                     linestyle='--', alpha=0.4)
        _style(ax_w, "WAVEFORM",
               xlabel="Time (s)",
               ylabel="Amplitude" if col == 0 else "")

        # ── Row 1: Spectrogram ───────────────────────────────────────
        ax_s = fig.add_subplot(gs[1, col])
        img_s = librosa.display.specshow(
            specs[col], sr=TARGET_SR, hop_length=HOP_LENGTH,
            x_axis='time', y_axis='hz',
            ax=ax_s, cmap=scmap,
        )
        _style(ax_s, "SPECTROGRAM",
               xlabel="Time (s)",
               ylabel="Frequency (Hz)" if col == 0 else "")
        _cbar(fig, img_s, ax_s, "dB")

        # ── Row 2: MFCC ─────────────────────────────────────────────
        ax_m = fig.add_subplot(gs[2, col])
        img_m = librosa.display.specshow(
            mfccs[col], sr=TARGET_SR, hop_length=HOP_LENGTH,
            x_axis='time',
            ax=ax_m, cmap=mcmap,
        )
        _style(ax_m, f"MFCC  ({N_MFCC} coeffs)",
               xlabel="Time (s)",
               ylabel="Coefficient" if col == 0 else "")
        ax_m.set_yticks(range(N_MFCC))
        ax_m.set_yticklabels([str(i) for i in range(N_MFCC)], fontsize=5)
        _cbar(fig, img_m, ax_m)

    # Row labels
    for lbl, yp in zip(["WAVEFORM", "SPECTROGRAM", "MFCC"], [0.77, 0.50, 0.245]):
        fig.text(0.005, yp, lbl, ha='left', va='center',
                 color=P["subtext"], fontsize=7.5,
                 fontfamily="monospace", fontweight="bold", rotation=90)

    # Footer
    footer = (
        f"SR={TARGET_SR} Hz  |  FFT={N_FFT}  |  Win={WIN_MS} ms  |  "
        f"Hop={HOP_LENGTH} smp  |  Mel={N_MELS}  |  MFCC={N_MFCC}  |  "
        f"Spectral Sub: α={ALPHA} β={BETA}  |  "
        f"Bandpass: {BANDPASS_LOW_HZ}–{BANDPASS_HIGH_HZ} Hz  order={FILTER_ORDER}"
    )
    fig.text(0.5, 0.012, footer,
             ha='center', va='bottom',
             color=P["subtext"], fontsize=7, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor=P["panel"],
                       edgecolor=P["grid"], alpha=0.9))

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=P["bg"])
    plt.close(fig)
    print(f"  [✓] Figure saved → {out_path}")


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(P["panel"])
    ax.tick_params(colors=P["subtext"], labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(P["grid"])
    ax.grid(color=P["grid"], linewidth=0.4, alpha=0.8)
    ax.set_title(title, color=P["text"], fontsize=8,
                 fontfamily="monospace", pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, color=P["subtext"], fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color=P["subtext"], fontsize=7)


def _cbar(fig, img, ax, label=""):
    cb = fig.colorbar(img, ax=ax)
    cb.ax.yaxis.set_tick_params(color=P["subtext"], labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=P["subtext"])
    cb.outline.set_edgecolor(P["grid"])
    if label:
        cb.set_label(label, color=P["subtext"], fontsize=6)

# ─────────────────────────── MAIN ───────────────────────────────────────────

def process(noise_path: str, mic_path: str, output_dir: str = "."):
    sep = "─" * 62
    print(f"\n{sep}")
    print(f"  AcoustEEEcare — Spectral Noise Removal")
    print(sep)
    print(f"  Noise profile : {noise_path}")
    print(f"  Mic recording : {mic_path}")
    print(f"  Output dir    : {output_dir}")
    print(sep)

    # ── 1. Load ──────────────────────────────────────────────────────
    noise_raw, noise_sr = load_wav(noise_path)
    mic_raw,   mic_sr   = load_wav(mic_path)

    noise_audio = resample_audio(noise_raw, noise_sr, TARGET_SR)
    mic_audio   = resample_audio(mic_raw,   mic_sr,   TARGET_SR)

    print(f"  Noise profile : {len(noise_audio)} samples @ {TARGET_SR} Hz  "
          f"({len(noise_audio)/TARGET_SR:.2f} s)")
    print(f"  Mic recording : {len(mic_audio)} samples @ {TARGET_SR} Hz  "
          f"({len(mic_audio)/TARGET_SR:.2f} s)")

    # ── 2. DC remove (both) ──────────────────────────────────────────
    noise_audio = remove_dc(noise_audio)
    mic_audio   = remove_dc(mic_audio)
    print(f"  DC removed    : noise mean={np.mean(noise_audio):.2e}  "
          f"mic mean={np.mean(mic_audio):.2e}")

    # ── 3. Spectral subtraction ──────────────────────────────────────
    print(f"  Running spectral subtraction  (α={ALPHA}, β={BETA}) ...")
    mic_clean = spectral_subtraction(mic_audio, noise_audio, TARGET_SR)
    snr_before = _snr(mic_audio,   noise_audio)
    snr_after  = _snr(mic_clean,   noise_audio)
    print(f"  SNR estimate  : before={snr_before:.1f} dB  after={snr_after:.1f} dB")

    # ── 4. Bandpass filter ───────────────────────────────────────────
    mic_bp = bandpass_filter(mic_clean, TARGET_SR,
                              BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ, FILTER_ORDER)
    print(f"  Bandpass [{BANDPASS_LOW_HZ}–{BANDPASS_HIGH_HZ} Hz] applied")

    # ── 5. Normalize ─────────────────────────────────────────────────
    mic_final = normalize_amplitude(mic_bp)
    print(f"  Normalized    : peak = {np.max(np.abs(mic_final)):.4f}")

    # ── 6. Save WAV ──────────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(mic_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    out_wav = os.path.join(output_dir, f"{base}_denoised.wav")
    save_wav(out_wav, mic_final, TARGET_SR)

    # ── 7. Plot ──────────────────────────────────────────────────────
    out_fig = os.path.join(output_dir, f"{base}_noise_analysis.png")
    plot_all(noise_audio, mic_audio, mic_clean, mic_final, out_fig)

    print(sep)
    return out_wav, out_fig


def _snr(signal, noise):
    """Simple broadband SNR estimate in dB."""
    sig_pwr   = np.mean(signal ** 2)
    noise_pwr = np.mean(noise  ** 2)
    if noise_pwr == 0:
        return float('inf')
    return 10 * np.log10(sig_pwr / noise_pwr)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage  : python noise_remover.py <noise.wav> <mic_recording.wav> [output_dir]")
        print("Example: python noise_remover.py internal_noise.wav mic_recording.wav ./results")
        sys.exit(1)

    noise_file  = sys.argv[1]
    mic_file    = sys.argv[2]
    out_dir     = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(mic_file) or "."

    process(noise_file, mic_file, out_dir)