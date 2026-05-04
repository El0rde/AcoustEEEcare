import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks, iirnotch, sosfiltfilt
from scipy.signal import tf2sos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
INPUT_FILE     = 'soul_sister_raw.wav'
OUTPUT_FILE    = 'clean.wav'
PLOT_FILE      = 'analysis.png'

NOISE_DURATION = 0.5    # seconds of pure noise at start
N_HARMONICS    = 19     # how many harmonics to subtract
# Sliding window: re-estimate phase every CHUNK_S seconds with OVERLAP_S overlap
CHUNK_S        = 1.0
OVERLAP_S      = 0.25

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD + DC REMOVAL
# ══════════════════════════════════════════════════════════════════════════════
y_orig, sr = librosa.load(INPUT_FILE, sr=None, mono=True)
y = y_orig - y_orig.mean()
print(f"Loaded: {len(y)/sr:.2f}s @ {sr} Hz  |  DC removed: {y_orig.mean():.5f}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FIND EXACT NOISE FUNDAMENTAL via autocorrelation of noise window
# ══════════════════════════════════════════════════════════════════════════════
noise = y[:int(NOISE_DURATION * sr)]
autocorr = np.correlate(noise, noise, mode='full')[len(noise)-1:]
autocorr /= autocorr[0]
peaks, _ = find_peaks(autocorr[1:400], height=0.3)
period_samples = peaks[0] + 1
f0 = sr / period_samples
print(f"Noise fundamental: {f0:.4f} Hz  (period = {period_samples} samples)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FIT HARMONIC AMPLITUDES + PHASES FROM NOISE WINDOW (global coefficients)
# ══════════════════════════════════════════════════════════════════════════════
t_noise = np.arange(len(noise)) / sr
harm_freqs = [k * f0 for k in range(1, N_HARMONICS+1) if k * f0 < sr/2]

def build_matrix(t_arr, freqs):
    cols = []
    for fk in freqs:
        cols.append(np.cos(2*np.pi*fk*t_arr))
        cols.append(np.sin(2*np.pi*fk*t_arr))
    return np.column_stack(cols)

X_noise = build_matrix(t_noise, harm_freqs)
coeffs_global, _, _, _ = np.linalg.lstsq(X_noise, noise, rcond=None)
print(f"Fitted {len(harm_freqs)} harmonics from noise window")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SLIDING WINDOW SUBTRACTION
#    Re-fit phase in each chunk so drift is tracked, but amplitude is anchored
#    to global noise-window coefficients (prevents signal from being used to
#    estimate noise when signal is present).
#    
#    Strategy: use global coeffs for amplitude, only allow phase to update
#    by doing a constrained re-fit within each chunk.
# ══════════════════════════════════════════════════════════════════════════════
chunk  = int(CHUNK_S * sr)
hop    = int((CHUNK_S - OVERLAP_S) * sr)
t_full = np.arange(len(y)) / sr

noise_model = np.zeros(len(y))
weight_sum  = np.zeros(len(y))

# Hann window for smooth overlap-add
win = np.hanning(chunk)

n_chunks = 0
for start in range(0, len(y) - chunk + 1, hop):
    end    = start + chunk
    t_chunk = t_full[start:end]
    y_chunk = y[start:end]

    X_chunk = build_matrix(t_chunk, harm_freqs)

    # Only refit if this chunk looks like noise (RMS close to noise floor)
    chunk_rms   = np.sqrt(np.mean(y_chunk**2))
    noise_rms   = np.sqrt(np.mean(noise**2))

    if chunk_rms < 1.8 * noise_rms:
        # Mostly noise — refit coefficients locally
        c, _, _, _ = np.linalg.lstsq(X_chunk, y_chunk, rcond=None)
    else:
        # Signal present — use global noise coefficients to avoid cancelling signal
        c = coeffs_global.copy()

    noise_chunk = X_chunk @ c
    noise_model[start:end] += noise_chunk * win
    weight_sum[start:end]  += win
    n_chunks += 1

# Normalize by overlap weight
mask = weight_sum > 1e-6
noise_model[mask] /= weight_sum[mask]

y_clean = y - noise_model

print(f"Processed {n_chunks} chunks")
print(f"Noise model RMS:  {np.sqrt(np.mean(noise_model**2)):.6f}")
print(f"Before RMS: {np.sqrt(np.mean(y**2)):.6f}")
print(f"After  RMS: {np.sqrt(np.mean(y_clean**2)):.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. RESIDUAL NOTCH — catch any remaining harmonic bleed
# ══════════════════════════════════════════════════════════════════════════════
for fk in harm_freqs[:8]:
    b, a = iirnotch(fk, Q=50, fs=sr)
    y_clean = sosfiltfilt(tf2sos(b, a), y_clean)

# ══════════════════════════════════════════════════════════════════════════════
# 6. NORMALIZE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
noise_floor_after = np.sqrt(np.mean(y_clean[:int(0.5*sr)]**2))
peak_s, peak_e    = int(5.5*sr), int(6.5*sr)
peak_rms_after    = np.sqrt(np.mean(y_clean[peak_s:peak_e]**2))
print(f"\nResidual noise floor: {noise_floor_after:.6f}")
print(f"Peak region RMS:      {peak_rms_after:.6f}")
print(f"SNR at peak: {20*np.log10(peak_rms_after/noise_floor_after):.1f} dB")

peak = np.max(np.abs(y_clean))
if peak > 0:
    y_clean = y_clean / peak * 0.95

sf.write(OUTPUT_FILE, y_clean, sr)
print(f"Saved -> {OUTPUT_FILE}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
times    = np.linspace(0, len(y)/sr, len(y))
times_cl = np.linspace(0, len(y_clean)/sr, len(y_clean))

fig = plt.figure(figsize=(18, 18), facecolor='#0e0e0e')
fig.suptitle('Harmonic Noise Cancellation — Analysis', color='white', fontsize=15, y=0.99)
gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.52, wspace=0.35)

C_RAW, C_CLN, C_NOISE, C_MODEL = '#4fc3f7', '#a5d6a7', '#ef9a9a', '#ffb74d'

def ax_style(ax, title):
    ax.set_facecolor('#1a1a1a')
    ax.set_title(title, color='white', fontsize=10, pad=6)
    ax.tick_params(colors='#aaa', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

ax0 = fig.add_subplot(gs[0, 0])
ax_style(ax0, 'Waveform — Raw (DC removed)')
ax0.plot(times, y, color=C_RAW, lw=0.4, alpha=0.85)
ax0.set_xlabel('Time (s)', color='#aaa', fontsize=8)
ax0.set_ylabel('Amplitude', color='#aaa', fontsize=8)

ax1 = fig.add_subplot(gs[0, 1])
ax_style(ax1, 'Waveform — Cleaned')
ax1.plot(times_cl, y_clean, color=C_CLN, lw=0.4, alpha=0.85)
ax1.set_xlabel('Time (s)', color='#aaa', fontsize=8)
ax1.set_ylabel('Amplitude', color='#aaa', fontsize=8)

ax2 = fig.add_subplot(gs[1, :])
ax_style(ax2, 'Waveform Overlay — first 3s')
zoom = int(3 * sr)
ax2.plot(times[:zoom], y[:zoom],          color=C_RAW,   lw=0.5, alpha=0.6, label='Raw')
ax2.plot(times[:zoom], noise_model[:zoom],color=C_MODEL, lw=0.5, alpha=0.7, label='Noise model')
ax2.plot(times_cl[:zoom], y_clean[:zoom], color=C_CLN,   lw=0.6, alpha=0.9, label='Clean')
ax2.axvline(NOISE_DURATION, color=C_NOISE, lw=1, ls='--', label=f'Noise window')
ax2.legend(fontsize=8, facecolor='#222', labelcolor='white', framealpha=0.8)
ax2.set_xlabel('Time (s)', color='#aaa', fontsize=8)
ax2.set_ylabel('Amplitude', color='#aaa', fontsize=8)

ax3 = fig.add_subplot(gs[2, 0])
ax_style(ax3, 'FFT — Raw vs Clean vs Noise Model (first 0.5s)')
win_s = int(0.5 * sr)
hann  = np.hanning(win_s)
fft_raw   = np.abs(np.fft.rfft(y[:win_s]       * hann))
fft_clean = np.abs(np.fft.rfft(y_clean[:win_s] * hann))
fft_model = np.abs(np.fft.rfft(noise_model[:win_s] * hann))
fft_f     = np.fft.rfftfreq(win_s, 1/sr)
ax3.semilogy(fft_f, fft_raw,   color=C_RAW,   lw=0.8, alpha=0.7, label='Raw')
ax3.semilogy(fft_f, fft_model, color=C_MODEL, lw=0.8, alpha=0.8, label='Noise model')
ax3.semilogy(fft_f, fft_clean, color=C_CLN,   lw=0.8, alpha=0.9, label='Clean')
ax3.set_xlim(0, min(2000, sr/2))
ax3.set_xlabel('Frequency (Hz)', color='#aaa', fontsize=8)
ax3.set_ylabel('Magnitude (log)', color='#aaa', fontsize=8)
ax3.legend(fontsize=8, facecolor='#222', labelcolor='white', framealpha=0.8)

ax4 = fig.add_subplot(gs[2, 1])
ax_style(ax4, f'Harmonic Amplitudes (noise fundamental={f0:.1f} Hz)')
amps = [np.sqrt(coeffs_global[2*i]**2 + coeffs_global[2*i+1]**2) for i in range(len(harm_freqs))]
ax4.bar(harm_freqs, amps, width=3, color=C_NOISE, alpha=0.8)
ax4.set_xlabel('Frequency (Hz)', color='#aaa', fontsize=8)
ax4.set_ylabel('Amplitude', color='#aaa', fontsize=8)
ax4.set_xlim(0, min(1400, sr/2))

n_fft_plot = 2048
hop_plot   = n_fft_plot // 4
ax5 = fig.add_subplot(gs[3, 0])
ax_style(ax5, 'Spectrogram — Raw (dB)')
S_raw = np.abs(librosa.stft(y, n_fft=n_fft_plot, hop_length=hop_plot))
img5 = librosa.display.specshow(librosa.amplitude_to_db(S_raw, ref=np.max),
                                 sr=sr, hop_length=hop_plot, x_axis='time', y_axis='log',
                                 ax=ax5, cmap='magma')
plt.colorbar(img5, ax=ax5, format='%+2.0f dB').ax.tick_params(colors='#aaa', labelsize=7)

ax6 = fig.add_subplot(gs[3, 1])
ax_style(ax6, 'Spectrogram — Cleaned (dB)')
S_cln = np.abs(librosa.stft(y_clean, n_fft=n_fft_plot, hop_length=hop_plot))
img6 = librosa.display.specshow(librosa.amplitude_to_db(S_cln, ref=np.max),
                                  sr=sr, hop_length=hop_plot, x_axis='time', y_axis='log',
                                  ax=ax6, cmap='magma')
plt.colorbar(img6, ax=ax6, format='%+2.0f dB').ax.tick_params(colors='#aaa', labelsize=7)

ax7 = fig.add_subplot(gs[4, :])
ax_style(ax7, 'RMS Energy over time — Raw vs Clean')
frame = 256
raw_rms   = [np.sqrt(np.mean(y[i:i+frame]**2))       for i in range(0, len(y)-frame, frame)]
clean_rms = [np.sqrt(np.mean(y_clean[i:i+frame]**2)) for i in range(0, len(y_clean)-frame, frame)]
t_rms = np.arange(len(raw_rms)) * frame / sr
ax7.plot(t_rms, raw_rms,   color=C_RAW, lw=0.8, alpha=0.8, label='Raw')
ax7.plot(t_rms, clean_rms, color=C_CLN, lw=0.8, alpha=0.9, label='Clean')
ax7.axhline(noise_floor_after, color=C_NOISE, lw=1, ls='--', label='Noise floor after')
ax7.legend(fontsize=8, facecolor='#222', labelcolor='white', framealpha=0.8)
ax7.set_xlabel('Time (s)', color='#aaa', fontsize=8)
ax7.set_ylabel('RMS', color='#aaa', fontsize=8)

plt.savefig(PLOT_FILE, dpi=150, bbox_inches='tight', facecolor='#0e0e0e')
print(f"Plot saved -> {PLOT_FILE}")
plt.show()