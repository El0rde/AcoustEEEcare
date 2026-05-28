# generate_bpf_coeffs.py  — corrected postShift=2 version
from scipy.signal import butter, sosfreqz
import numpy as np

fs     = 8000
f_low  = 10
f_high = 1000
order  = 2

sos = butter(order, [f_low, f_high], btype='bandpass', fs=fs, output='sos')

# postShift=2 gives coefficients headroom of 2 bits before Q15 saturation.
# The CMSIS accumulator left-shifts by 2 at output, recovering the scale.
# This prevents overflow when Stage 2 b=[1,-2,1] processes residual DC.
post_shift = 2
scale = 32768.0 / (2 ** post_shift)   # = 8192.0

def q15(x):
    v = int(round(x * scale))
    if v > 32767 or v < -32768:
        print(f"  WARNING: coefficient {x:.6f} → {v} clipped!")
    return max(-32768, min(32767, v))

print("=" * 60)
print("Paste these into src/main.c → heart_bpf_coeffs[]")
print("=" * 60)
print(f"\n/* SAADC pre-filter: 10-1000 Hz Butterworth-2 @ 8 kHz */")
print(f"/* postShift=2: headroom for Stage-2 [1,-2,1] numerator */")
print(f"#define HEART_BPF_STAGES  {len(sos)}\n")
print("static const q15_t heart_bpf_coeffs[5 * HEART_BPF_STAGES] = {")

for i, stage in enumerate(sos):
    b0, b1, b2, a0, a1, a2 = stage
    print(f"    /* Stage {i+1} */")
    print(f"     {q15(b0/a0)}, {q15(b1/a0)}, {q15(b2/a0)}, "
          f"{q15(-a1/a0)}, {q15(-a2/a0)},")
    print(f"    // float: b=[{b0/a0:.6f},{b1/a0:.6f},{b2/a0:.6f}] "
          f"-a=[{-a1/a0:.6f},{-a2/a0:.6f}]")

print("};\n")
print(f"// Use postShift = {post_shift} in arm_biquad_cascade_df1_init_q15()")

# Gain check
w, h = sosfreqz(sos, worN=[5,10,20,100,500,950,1000,1500,2000], fs=fs)
print("\nGain check (ideal filter, independent of Q15 scaling):")
for freq, gain in zip([5,10,20,100,500,950,1000,1500,2000], np.abs(h)):
    bar = '█' * int(gain * 20)
    print(f"  {freq:5d} Hz → gain = {gain:.4f}  {bar}")
print("=" * 60)