#!/usr/bin/env python3
"""
gen_mfcc_tables.py
==================
Generates a C source file containing flash-constant MFCC tables for one
pipeline (heart or lung) from a JSON winning-config file.

Usage:
    python tools/gen_mfcc_tables.py \
        --config artifacts/heart_winning_config.json \
        --name heart \
        --bandpass-low 10 --bandpass-high 200 \
        --sample-rate 8000 \
        --output src/generated/heart_mfcc_tables.c

    python tools/gen_mfcc_tables.py \
        --config artifacts/lung_winning_config.json \
        --name lung \
        --bandpass-low 100 --bandpass-high 1000 \
        --output src/generated/lung_mfcc_tables.c

All arrays are marked const so they live in .rodata (flash) on the target.
CMSIS biquad sign convention: feedback coefficients are NEGATED (-a1, -a2).
"""

import json
import argparse
import numpy as np
from scipy.signal import butter


def main():
    ap = argparse.ArgumentParser(
        description="Generate MFCC flash-constant C tables from a winning config JSON."
    )
    ap.add_argument("--config", required=True,
                    help="Path to winning_config.json (heart or lung)")
    ap.add_argument("--name", required=True, choices=["heart", "lung"],
                    help="Pipeline name (used in C symbol names)")
    ap.add_argument("--bandpass-low", type=float, required=True,
                    help="Bandpass lower cutoff in Hz (heart=10, lung=100)")
    ap.add_argument("--bandpass-high", type=float, required=True,
                    help="Bandpass upper cutoff in Hz (heart=200, lung=1000)")
    ap.add_argument("--sample-rate", type=int, default=8000,
                    help="ADC capture rate in Hz (default 8000)")
    ap.add_argument("--output", required=True,
                    help="Path for the generated .c file")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    target_rate    = cfg["target_rate"]
    frame_ms       = cfg["frame_ms"]
    hop_pct        = cfg["hop_pct"]
    fft_size       = cfg["fft_size"]
    n_mel          = cfg["num_mel"]
    n_mfcc         = cfg["num_mfcc"]

    frame_samples   = round(target_rate * frame_ms / 1000)
    hop_samples     = round(frame_samples * hop_pct)
    n_fft_bins      = fft_size // 2 + 1
    # n_frames: matches firmware formula 1 + (total_decimated - frame) / hop
    total_decimated = target_rate * 10   # 10-second recording
    n_frames        = 1 + (total_decimated - frame_samples) // hop_samples
    decimate_factor = args.sample_rate // target_rate
    # Mel filterbank spans from bandpass_low to Nyquist of target rate
    f_low_mel       = float(args.bandpass_low)
    f_high_mel      = float(target_rate) / 2.0

    print(f"[gen_mfcc_tables] Pipeline: {args.name}")
    print(f"  target_rate={target_rate}, frame_samples={frame_samples}, hop_samples={hop_samples}")
    print(f"  fft_size={fft_size}, n_fft_bins={n_fft_bins}, n_mel={n_mel}, n_mfcc={n_mfcc}")
    print(f"  n_frames_expected={n_frames}, decimate_factor={decimate_factor}")
    print(f"  bandpass={args.bandpass_low}-{args.bandpass_high} Hz @ {args.sample_rate} Hz")
    print(f"  mel range={f_low_mel}-{f_high_mel} Hz")

    # ----------------------------------------------------------------
    # 1. Bandpass SOS coefficients (4th-order Butterworth)
    #    CMSIS sign convention: negate a1, a2 (feedback coefficients)
    # ----------------------------------------------------------------
    sos = butter(4, [args.bandpass_low, args.bandpass_high],
                 btype="band", fs=args.sample_rate, output="sos")
    assert len(sos) == 4, f"Expected 4 SOS sections, got {len(sos)}"

    # ----------------------------------------------------------------
    # 2. Periodic Hamming window: w[n] = 0.54 - 0.46*cos(2pi*n/N)
    #    divisor is N (not N-1), matching the firmware's build_hamming()
    # ----------------------------------------------------------------
    n_arr = np.arange(frame_samples, dtype=np.float64)
    hamming = 0.54 - 0.46 * np.cos(2.0 * np.pi * n_arr / frame_samples)

    # ----------------------------------------------------------------
    # 3. HTK mel filterbank (no Slaney normalization)
    #    Matches dsp_mfcc.c build_mel_filterbank() exactly.
    # ----------------------------------------------------------------
    mel_lo = 2595.0 * np.log10(1.0 + f_low_mel / 700.0)
    mel_hi = 2595.0 * np.log10(1.0 + f_high_mel / 700.0)
    mel_pts = np.linspace(mel_lo, mel_hi, n_mel + 2)
    hz_pts  = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bin_pts = hz_pts / (float(target_rate) / 2.0) * (n_fft_bins - 1)

    mel_fb = np.zeros((n_mel, n_fft_bins), dtype=np.float32)
    for m in range(n_mel):
        f_left   = bin_pts[m]
        f_center = bin_pts[m + 1]
        f_right  = bin_pts[m + 2]
        for k in range(n_fft_bins):
            kf = float(k)
            if f_left <= kf <= f_center:
                d = f_center - f_left
                mel_fb[m, k] = float((kf - f_left) / d) if d > 0 else 0.0
            elif f_center < kf <= f_right:
                d = f_right - f_center
                mel_fb[m, k] = float((f_right - kf) / d) if d > 0 else 0.0

    # ----------------------------------------------------------------
    # 4. Orthonormal DCT-II matrix
    #    Matches dsp_mfcc.c build_dct_matrix() exactly.
    # ----------------------------------------------------------------
    inv_N  = np.sqrt(1.0 / n_mel)
    inv_2N = np.sqrt(2.0 / n_mel)
    dct = np.zeros((n_mfcc, n_mel), dtype=np.float32)
    for k in range(n_mfcc):
        scale = inv_N if k == 0 else inv_2N
        for nn in range(n_mel):
            dct[k, nn] = float(scale * np.cos(np.pi * k * (2 * nn + 1) / (2 * n_mel)))

    # ----------------------------------------------------------------
    # 5. Emit C file
    # ----------------------------------------------------------------
    nm = args.name   # short alias

    with open(args.output, "w") as out:
        out.write(f"/* AUTO-GENERATED by tools/gen_mfcc_tables.py from {args.config} */\n")
        out.write(f"/* DO NOT EDIT -- re-run gen_mfcc_tables.py to regenerate.      */\n")
        out.write(f"/*                                                               */\n")
        out.write(f"/* Pipeline : {nm:<50} */\n")
        out.write(f"/* Bandpass : {args.bandpass_low}-{args.bandpass_high} Hz @ {args.sample_rate} Hz{'':<33} */\n")
        out.write(f"/* n_mel    : {n_mel:<50} */\n")
        out.write(f"/* n_mfcc   : {n_mfcc:<50} */\n")
        out.write(f"\n")
        out.write(f'#include "{nm}_mfcc_config.h"\n')
        out.write(f"\n")

        # Bandpass coefficients
        out.write(f"/* Butterworth-4 bandpass {args.bandpass_low}-{args.bandpass_high} Hz @ {args.sample_rate} Hz\n")
        out.write(f" * CMSIS DF2T format per section: b0, b1, b2, -a1, -a2 */\n")
        out.write(f"const float32_t {nm}_bp_coeffs[BP_N_STAGES * 5] = {{\n")
        for sec_idx, sec in enumerate(sos):
            b0, b1, b2, a0, a1, a2 = sec
            out.write(f"    /* section {sec_idx} */ "
                      f"{b0:+.10f}f, {b1:+.10f}f, {b2:+.10f}f,  "
                      f"{-a1:+.10f}f, {-a2:+.10f}f,\n")
        out.write(f"}};\n\n")

        # Hamming window
        out.write(f"/* Periodic Hamming window, N={frame_samples} */\n")
        out.write(f"const float32_t {nm}_hamming[{frame_samples}] = {{\n")
        for i, v in enumerate(hamming):
            out.write(f"    {float(v):+.10f}f,")
            if (i + 1) % 4 == 0:
                out.write("\n")
        if frame_samples % 4 != 0:
            out.write("\n")
        out.write(f"}};\n\n")

        # Mel filterbank
        out.write(f"/* HTK mel filterbank: {n_mel} filters x {n_fft_bins} FFT bins\n")
        out.write(f" * Row-major: mel_fb[m * n_fft_bins + k] */\n")
        out.write(f"const float32_t {nm}_mel_fb[{n_mel} * {n_fft_bins}] = {{\n")
        for m in range(n_mel):
            out.write(f"    /* mel bin {m:2d} */\n    ")
            for k in range(n_fft_bins):
                out.write(f"{float(mel_fb[m, k]):+.8f}f,")
                if (k + 1) % 8 == 0 and k + 1 < n_fft_bins:
                    out.write("\n    ")
            out.write("\n")
        out.write(f"}};\n\n")

        # DCT matrix
        out.write(f"/* Orthonormal DCT-II matrix: {n_mfcc} x {n_mel}\n")
        out.write(f" * Row-major: dct[k * n_mel + n] */\n")
        out.write(f"const float32_t {nm}_dct[{n_mfcc} * {n_mel}] = {{\n")
        for k in range(n_mfcc):
            out.write(f"    /* mfcc {k:2d} */ ")
            for nn in range(n_mel):
                out.write(f"{float(dct[k, nn]):+.10f}f, ")
            out.write("\n")
        out.write(f"}};\n\n")

        # Config struct
        out.write(f"/* Pipeline config struct -- points to flash tables above */\n")
        out.write(f"const dsp_mfcc_config_t {nm}_mfcc_config = {{\n")
        out.write(f"    .decimate_factor    = {decimate_factor},\n")
        out.write(f"    .target_rate        = {target_rate},\n")
        out.write(f"    .frame_samples      = {frame_samples},\n")
        out.write(f"    .hop_samples        = {hop_samples},\n")
        out.write(f"    .n_frames_expected  = {n_frames},\n")
        out.write(f"    .fft_size           = {fft_size},\n")
        out.write(f"    .n_fft_bins         = {n_fft_bins},\n")
        out.write(f"    .n_mel              = {n_mel},\n")
        out.write(f"    .n_mfcc             = {n_mfcc},\n")
        out.write(f"    .f_low_hz           = {f_low_mel:.1f}f,\n")
        out.write(f"    .f_high_hz          = {f_high_mel:.1f}f,\n")
        out.write(f"    .bp_coeffs          = {nm}_bp_coeffs,\n")
        out.write(f"    .hamming            = {nm}_hamming,\n")
        out.write(f"    .mel_fb             = {nm}_mel_fb,\n")
        out.write(f"    .dct                = {nm}_dct,\n")
        out.write(f"}};\n")

    print(f"  -> Written to {args.output}")


if __name__ == "__main__":
    main()
