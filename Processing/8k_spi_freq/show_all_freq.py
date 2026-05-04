import numpy as np
import librosa
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
INPUT_FILE      = '16bit_8000_10s_SD_20260426_135524.wav'
NOISE_DURATION  = 0.5
N_HARMONICS     = 19
PEAK_THRESHOLD  = 0.03
DISPLAY_SECONDS = 3.0
BANDWIDTH_HZ    = 10.0
FILTER_ORDER    = 2

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD + DC REMOVAL
# ══════════════════════════════════════════════════════════════════════════════
y, sr = librosa.load(INPUT_FILE, sr=None, mono=True)
y     = y - y.mean()
N     = len(y)
t     = np.arange(N) / sr
duration = N / sr
print(f"Loaded: {duration:.2f}s @ {sr} Hz")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FIND FUNDAMENTAL
# ══════════════════════════════════════════════════════════════════════════════
noise    = y[:int(NOISE_DURATION * sr)]
autocorr = np.correlate(noise, noise, mode='full')[len(noise)-1:]
autocorr /= autocorr[0]
peaks, _ = find_peaks(autocorr[1:400], height=0.3)
period_samples = peaks[0] + 1
f0 = sr / period_samples
print(f"Fundamental: {f0:.4f} Hz")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FFT — detect harmonics
# ══════════════════════════════════════════════════════════════════════════════
hann  = np.hanning(N)
Y     = np.fft.rfft(y * hann)
Y_mag = np.abs(Y) / N
Y_mag[1:-1] *= 2
freqs = np.fft.rfftfreq(N, 1/sr)

def get_amplitude_at(f_target, freqs, Y_mag, window=2.0):
    mask = (freqs >= f_target - window) & (freqs <= f_target + window)
    return float(np.max(Y_mag[mask])) if mask.any() else 0.0

harm_freqs_all = [k * f0 for k in range(1, N_HARMONICS+1) if k * f0 < sr/2]
max_amp = max(get_amplitude_at(f, freqs, Y_mag) for f in harm_freqs_all)

detected = []
for k, fk in enumerate(harm_freqs_all, start=1):
    amp = get_amplitude_at(fk, freqs, Y_mag)
    rel = amp / max_amp if max_amp > 0 else 0
    if rel >= PEAK_THRESHOLD:
        detected.append({'harmonic': k, 'freq': fk, 'amp': amp, 'rel': rel})

print(f"\nDetected {len(detected)} harmonics:")
for d in detected:
    print(f"  H{d['harmonic']:02d}  {d['freq']:8.3f} Hz  rel={d['rel']:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. BANDPASS ISOLATE
# ══════════════════════════════════════════════════════════════════════════════
def bandpass(signal, fc, bw, fs, order=2):
    lo = max(fc - bw / 2, 1.0)
    hi = min(fc + bw / 2, fs / 2 - 1.0)
    if lo >= hi:
        return np.zeros_like(signal)
    Wn = [lo / (fs/2), hi / (fs/2)]
    if any(w <= 0 or w >= 1 for w in Wn):
        return np.zeros_like(signal)
    try:
        b, a = butter(order, Wn, btype='band')
        out  = filtfilt(b, a, signal)
        if not np.isfinite(out).all() or np.max(np.abs(out)) > 10:
            return np.zeros_like(signal)
        return out
    except Exception:
        return np.zeros_like(signal)

print("\nIsolating harmonics...")
isolated = [bandpass(y, d['freq'], BANDWIDTH_HZ, sr, FILTER_ORDER)
            for d in detected]
print("Done.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. INTERACTIVE OVERLAY VIEWER
# ══════════════════════════════════════════════════════════════════════════════
n_show  = min(int(DISPLAY_SECONDS * sr), N)
t_show  = t[:n_show]
y_show  = y[:n_show]
n_det   = len(detected)
cmap    = plt.cm.plasma
colors  = [cmap(i / max(n_det - 1, 1)) for i in range(n_det)]

idx = [0]   # mutable index for callbacks

fig = plt.figure(figsize=(16, 10), facecolor='#0e0e0e')
fig.suptitle('', color='white', fontsize=13)

gs = gridspec.GridSpec(3, 1, figure=fig,
                       height_ratios=[2.5, 1.2, 0.15],
                       hspace=0.45)

# — Main overlay plot —
ax_main = fig.add_subplot(gs[0])
ax_main.set_facecolor('#1a1a1a')
ax_main.tick_params(colors='#aaa', labelsize=8)
for sp in ax_main.spines.values():
    sp.set_edgecolor('#333')
ax_main.set_xlabel('Time (s)', color='#aaa', fontsize=9)
ax_main.set_ylabel('Amplitude', color='#aaa', fontsize=9)

line_orig, = ax_main.plot(t_show, y_show,
                           color='#4fc3f7', lw=0.6, alpha=0.5,
                           label='Original (full signal)')
line_harm, = ax_main.plot([], [],
                           color='red', lw=1.2, alpha=0.9,
                           label='Isolated harmonic')
ax_main.set_xlim(0, DISPLAY_SECONDS)
legend = ax_main.legend(facecolor='#222', labelcolor='white',
                         fontsize=9, framealpha=0.8,
                         loc='upper right')

# — FFT subplot — shows marker on current frequency —
ax_fft = fig.add_subplot(gs[1])
ax_fft.set_facecolor('#1a1a1a')
ax_fft.tick_params(colors='#aaa', labelsize=7)
for sp in ax_fft.spines.values():
    sp.set_edgecolor('#333')
ax_fft.plot(freqs, 20*np.log10(Y_mag + 1e-10),
            color='#ff7043', lw=0.6, alpha=0.85)
ax_fft.set_xlim(0, min(harm_freqs_all[-1] * 1.2, sr/2))
ax_fft.set_xlabel('Frequency (Hz)', color='#aaa', fontsize=8)
ax_fft.set_ylabel('Magnitude (dB)', color='#aaa', fontsize=8)

# Plot all harmonic markers dimly
for i, d in enumerate(detected):
    ax_fft.axvline(d['freq'], color=colors[i], lw=0.8, alpha=0.25)

# Active marker (will be updated)
vline = ax_fft.axvline(detected[0]['freq'], color='white', lw=1.8, alpha=0.95)

# — Navigation instructions —
ax_nav = fig.add_subplot(gs[2])
ax_nav.set_facecolor('#0e0e0e')
ax_nav.axis('off')
nav_text = ax_nav.text(0.5, 0.5,
    '◀  LEFT arrow — previous  |  RIGHT arrow — next  |  Q — quit',
    color='#666', fontsize=9, ha='center', va='center',
    transform=ax_nav.transAxes)

# ── Update function ───────────────────────────────────────────────────────────
def update(i):
    d      = detected[i]
    y_band = isolated[i]
    color  = colors[i]
    rms    = np.sqrt(np.mean(y_band**2))
    peak   = np.max(np.abs(y_band))

    # Update waveform
    line_harm.set_data(t_show, y_band[:n_show])
    line_harm.set_color(color)

    # Rescale y-axis to fit both signals
    combined_peak = max(np.max(np.abs(y_show)), peak) * 1.15
    ax_main.set_ylim(-combined_peak, combined_peak)

    # Update FFT marker
    vline.set_xdata([d['freq'], d['freq']])
    vline.set_color(color)

    # Update title
    fig.suptitle(
        f"H{d['harmonic']}  —  {d['freq']:.3f} Hz  "
        f"|  Rel Amp: {d['rel']:.3f}  "
        f"|  RMS: {rms:.5f}  "
        f"|  Peak: {peak:.5f}  "
        f"  [{i+1} / {n_det}]",
        color=color, fontsize=12
    )

    # Highlight active legend entry
    legend.get_texts()[1].set_text(
        f"H{d['harmonic']} — {d['freq']:.2f} Hz (isolated)"
    )

    fig.canvas.draw_idle()

# ── Key navigation ────────────────────────────────────────────────────────────
def on_key(event):
    if event.key == 'right':
        idx[0] = (idx[0] + 1) % n_det
        update(idx[0])
    elif event.key == 'left':
        idx[0] = (idx[0] - 1) % n_det
        update(idx[0])
    elif event.key == 'q':
        plt.close()

fig.canvas.mpl_connect('key_press_event', on_key)

# Initial render
update(0)
plt.show()