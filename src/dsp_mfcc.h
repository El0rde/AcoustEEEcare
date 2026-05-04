/**
 * dsp_mfcc.h
 * ==========
 * On-device DSP + MFCC pipeline for AcoustEEEcare on nRF52840/Zephyr.
 *
 * Pipeline (locked to firmware plan §8 + §9):
 *   1. int16 @ 8 kHz  →  float32 via /32768.0f
 *   2. Butterworth-4 bandpass 10–200 Hz (CMSIS-DSP arm_biquad_cascade_df2T_f32)
 *   3. Decimate ÷4  →  int16 @ 2 kHz   (plain keep-every-4th)
 *   4. MFCC per frame (25 ms / 30% hop, 512-pt FFT, 20 mel bins, 20 coefficients)
 *      a. int16→float32 + periodic Hamming window (divisor N)
 *      b. Zero-pad to 512, arm_rfft_fast_f32
 *      c. Power spectrum: Re² + Im²
 *      d. HTK mel filterbank (no Slaney normalization)
 *      e. Natural log, floor 1e-6
 *      f. Orthonormal DCT-II, first 20 coefficients
 *
 * All constants are computed at init time from the same formulas as
 * mfcc_core.c — no malloc, no heap; all buffers are static.
 *
 * MFCC parameters (from winning_config.json, plan §9.2):
 *   target_rate  = 2000 Hz
 *   frame_ms     = 25 ms  → frame_samples = 50
 *   hop_pct      = 0.30   → hop_samples   = 15
 *   fft_size     = 512
 *   n_mel        = 20
 *   n_mfcc       = 20
 *   n_frames     = 1 + (20000 - 50) / 15 = 1331  (for 10 s @ 2 kHz)
 *
 * v6.3 STREAMING CHANGE:
 *   The large g_mfcc_out[1331][20] output array (104 KB BSS) has been
 *   removed. Instead, dsp_mfcc_run() accepts a callback that is invoked
 *   once per frame as it is computed. The caller streams the frame over
 *   BLE immediately, reusing a single 80-byte scratch buffer.
 *
 *   Similarly, dsp_mfcc_run() no longer accepts a full PCM batch buffer.
 *   Instead, call dsp_mfcc_feed_chunk() for each 512-sample half-buffer
 *   as it arrives from the SAADC, then call dsp_mfcc_finish() after
 *   recording to compute and stream all MFCC frames.
 *
 * Usage:
 *   dsp_mfcc_init();                       // call once at boot
 *   dsp_mfcc_reset();                      // call before each recording
 *   dsp_mfcc_feed_chunk(buf, 512);         // call per SAADC half-buffer
 *   dsp_mfcc_finish(my_frame_callback);    // call after recording ends
 */

#ifndef DSP_MFCC_H
#define DSP_MFCC_H

#include <stdint.h>

/* ── MFCC parameters ────────────────────────────────────────────── */
#define DSP_SAMPLING_RATE    8000
#define DSP_TARGET_RATE      2000
#define DSP_DECIMATE_FACTOR  4        /* 8000 / 2000 */

#define MFCC_FRAME_MS        25
#define MFCC_HOP_PCT_X100    30       /* 30% → hop = 15 samples at 2 kHz */
#define MFCC_FFT_SIZE        512
#define MFCC_N_MEL           20
#define MFCC_N_MFCC          20
#define MFCC_FRAME_SAMPLES   50       /* round(2000 * 25 / 1000) */
#define MFCC_HOP_SAMPLES     15       /* round(50 * 0.30) */
#define MFCC_N_FRAMES        1331     /* 1 + (20000 - 50) / 15 */
#define MFCC_N_FFT_BINS      257      /* FFT_SIZE/2 + 1 */

/* ── Decimated buffer (10 s @ 2 kHz = 20000 int16) ──────────────── */
#define DECIMATED_SAMPLES    20000    /* 10 s × 2000 Hz */

/**
 * Per-frame MFCC callback, invoked once per frame by dsp_mfcc_finish().
 *
 * @param frame_idx  Frame index (0 … MFCC_N_FRAMES-1).
 * @param coeffs     Pointer to MFCC_N_MFCC float32 values.
 *                   Valid only for the duration of the callback —
 *                   copy or transmit; do not store the pointer.
 */
typedef void (*dsp_mfcc_frame_cb_t)(int frame_idx, const float *coeffs);

/**
 * Initialise CMSIS-DSP instances, compute Hamming window,
 * mel filterbank, and DCT matrix. Call once at boot.
 * Returns 0 on success, negative on error.
 */
int dsp_mfcc_init(void);

/**
 * Reset filter state and decimated-buffer index for a new recording.
 * Call once just before the first dsp_mfcc_feed_chunk() of each session.
 */
void dsp_mfcc_reset(void);

/**
 * Feed one SAADC half-buffer into the pipeline.
 * Bandpass-filters the chunk (filter state persists across calls),
 * then decimates ÷4 into the internal s_decimated[] buffer.
 *
 * @param pcm_8k   int16 samples at 8 kHz.
 * @param n        Number of samples (typically HALF_BUF_SAMPLES = 512).
 */
void dsp_mfcc_feed_chunk(const int16_t *pcm_8k, int n);

/**
 * After recording is complete, compute all MFCC frames and invoke
 * cb() once per frame.  Runs synchronously (~700 ms on Cortex-M4F
 * @ 64 MHz).
 *
 * @param cb   Called for every frame.  Must not be NULL.
 * @return     Number of frames computed, or negative on error.
 */
int dsp_mfcc_finish(dsp_mfcc_frame_cb_t cb);

#endif /* DSP_MFCC_H */