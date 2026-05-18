/**
 * dsp_mfcc.c
 * ==========
 * On-device DSP + MFCC pipeline. Zephyr/nRF52840 version.
 *
 * v7.2 — SCRATCH BUFFER & VALIDATION FIXES:
 *   - MAX_FFT_SIZE bumped 512 -> 2048 (lung config uses fft_size=2048)
 *   - MAX_N_FFT_BINS bumped 257 -> 1025 (= 2048/2 + 1)
 *   - dsp_mfcc_init() now hard-validates cfg against MAX_* constants.
 *     Previously, oversize configs silently overflowed BSS scratch.
 *   - Added one-time scratch-budget log at first init.
 *
 * v7.1 CIRCULAR-WINDOW OPTIMIZATION (unchanged):
 *   - Sliding window is a true circular buffer indexed by window_head.
 *   - Per-decimated-sample memmove eliminated.
 */

#include "dsp_mfcc.h"
#include <arm_math.h>
#include <string.h>
#include <math.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(dsp_mfcc);

/* ================================================================
 * SHARED SCRATCH BUFFERS
 *
 * v7.2: Sized for the largest config across heart and lung.
 *   Heart : fft_size=512, n_fft_bins=257, n_mel=25, n_mfcc=25
 *   Lung  : fft_size=2048, n_fft_bins=1025, n_mel=26, n_mfcc=26,
 *           frame_samples=1200
 *
 * Memory cost (~25 KB scratch, in BSS):
 *   bp_scratch  : 512 * 4 =  2,048 B
 *   fft_in      : 2048 * 4 = 8,192 B
 *   fft_out     : 2048 * 4 = 8,192 B
 *   power       : 1025 * 4 = 4,100 B
 *   mel_buf     : 30 * 4 =     120 B
 *   mfcc_out    : 30 * 4 =     120 B
 *                  total : 22,772 B
 * ================================================================ */
#define BP_SCRATCH_MAX   512
#define MAX_FFT_SIZE     2048
#define MAX_N_FFT_BINS   1025
#define MAX_N_MEL        30
#define MAX_N_MFCC       30

/*
 * CMSIS biquad coefficients: [b0, b1, b2, -a1, -a2] per section.
 *
 * The window is filled in circular fashion. The OLDEST sample lives
 * at index window_head (it's the next slot to be overwritten). Reading
 * proceeds: window_head, window_head+1, ..., frame_samples-1,
 * 0, 1, ..., window_head-1.
 *
 * NOTE: The previous values were placeholders that attenuated the passband
 * by 40–70 dB, causing the MFCC log-mel output to saturate at the 1e-6 floor
 * (c0 ≈ -61.78 across all frames). These are the real coefficients.
 */
static const float32_t s_bp_coeffs[BP_N_STAGES * 5] = {
    /* section 0: b0,            b1,            b2,            -a1,           -a2          */
    +0.0000256870f, +0.0000513741f, +0.0000256870f, +1.7521353989f, -0.7703674421f,
    /* section 1 */
    +1.0000000000f, +2.0000000000f, +1.0000000000f, +1.8745313518f, -0.8973206556f,
    /* section 2 */
    +1.0000000000f, -2.0000000000f, +1.0000000000f, +1.9844767549f, -0.9845500024f,
    /* section 3 */
    +1.0000000000f, -2.0000000000f, +1.0000000000f, +1.9943824564f, -0.9944455175f,
};

/* State: 2 state variables per section, DF2T form */
static float32_t s_bp_state[BP_N_STAGES * 2];
static arm_biquad_cascade_df2T_instance_f32 s_bp_inst;

/* ══════════════════════════════════════════════════════════════════
 * SECTION 2: DECIMATED BUFFER
 * 10 s × 2000 Hz = 20000 int16 samples → 40 KB in BSS.
 * ══════════════════════════════════════════════════════════════════ */
static int16_t  s_decimated[DECIMATED_SAMPLES];
static int       s_dec_idx;   /* write cursor, reset by dsp_mfcc_reset() */

/* ══════════════════════════════════════════════════════════════════
 * SECTION 3: MFCC CONSTANTS (computed once at init)
 * ══════════════════════════════════════════════════════════════════ */

/* Periodic Hamming window: w[n] = 0.54 - 0.46*cos(2πn/N), divisor N */
static float32_t s_hamming[MFCC_FRAME_SAMPLES];

/* Mel filterbank: [N_MEL][N_FFT_BINS] row-major */
static float32_t s_mel_fb[MFCC_N_MEL * MFCC_N_FFT_BINS];

/* Orthonormal DCT-II matrix: [N_MFCC][N_MEL] row-major */
static float32_t s_dct[MFCC_N_MFCC * MFCC_N_MEL];

/* ══════════════════════════════════════════════════════════════════
 * SECTION 4: SCRATCH BUFFERS (reused every frame / every chunk)
 *
 * s_bp_scratch  — float32 version of one SAADC half-buffer (512 samples, 2 KB)
 * s_fft_in      — windowed + zero-padded frame              (512 floats,  2 KB)
 * s_fft_out     — RFFT complex output                       (512 floats,  2 KB)
 * s_power       — Re²+Im² power spectrum                    (257 floats,  1 KB)
 * s_mel         — mel filterbank energies                   (20  floats, 80  B)
 * s_mfcc_frame  — one frame of MFCC output (replaces g_mfcc_out) (20 floats, 80 B)
 *
 * Total scratch: ~7 KB  (was 104 KB + 7 KB before)
 * ══════════════════════════════════════════════════════════════════ */
#define BP_SCRATCH_MAX  512
static float32_t s_bp_scratch[BP_SCRATCH_MAX];

static float32_t s_fft_in[MFCC_FFT_SIZE];
static float32_t s_fft_out[MFCC_FFT_SIZE];
static float32_t s_power[MFCC_N_FFT_BINS];
static float32_t s_mel[MFCC_N_MEL];
static float32_t s_mfcc_frame[MFCC_N_MFCC];   /* single-frame output, 80 bytes */

/* CMSIS-DSP RFFT instance */
static arm_rfft_fast_instance_f32 s_rfft;

/* ══════════════════════════════════════════════════════════════════
 * INTERNAL: build Hamming window
 * ══════════════════════════════════════════════════════════════════ */
static void build_hamming(void)
{
    for (int i = 0; i < MFCC_FRAME_SAMPLES; i++) {
        s_hamming[i] = 0.54f - 0.46f * cosf(
            2.0f * 3.14159265f * (float)i / (float)MFCC_FRAME_SAMPLES);
    }
}

/* ================================================================
 * PUBLIC: dsp_mfcc_init()
 *
 * v7.2: hard-validates cfg fields against compiled-in scratch limits.
 *       Returns negative errno on any violation -- caller MUST check.
 * ================================================================ */
int dsp_mfcc_init(dsp_mfcc_pipeline_t *p, const dsp_mfcc_config_t *cfg)
{
    const float f_low  = 10.0f;
    const float f_high = (float)DSP_TARGET_RATE / 2.0f;  /* 1000 Hz */
    const float mel_lo = hz_to_mel(f_low);
    const float mel_hi = hz_to_mel(f_high);

    float mel_pts[MFCC_N_MEL + 2];
    float hz_pts [MFCC_N_MEL + 2];
    float bin_pts[MFCC_N_MEL + 2];

    for (int i = 0; i < MFCC_N_MEL + 2; i++) {
        mel_pts[i] = mel_lo + (float)i * (mel_hi - mel_lo) / (float)(MFCC_N_MEL + 1);
        hz_pts[i]  = mel_to_hz(mel_pts[i]);
        bin_pts[i] = hz_pts[i] / ((float)DSP_TARGET_RATE / 2.0f) * (float)(MFCC_N_FFT_BINS - 1);
    }

    memset(s_mel_fb, 0, sizeof(s_mel_fb));

    for (int m = 0; m < MFCC_N_MEL; m++) {
        float f_left   = bin_pts[m];
        float f_center = bin_pts[m + 1];
        float f_right  = bin_pts[m + 2];
        float *row     = s_mel_fb + m * MFCC_N_FFT_BINS;

        for (int k = 0; k < MFCC_N_FFT_BINS; k++) {
            float kf  = (float)k;
            float val = 0.0f;
            if (kf >= f_left && kf <= f_center) {
                float d = f_center - f_left;
                val = (d > 0.0f) ? (kf - f_left) / d : 0.0f;
            } else if (kf > f_center && kf <= f_right) {
                float d = f_right - f_center;
                val = (d > 0.0f) ? (f_right - kf) / d : 0.0f;
            }
            row[k] = val;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 * INTERNAL: build orthonormal DCT-II matrix
 * Matches mfcc_core.c build_dct_matrix() exactly.
 * ══════════════════════════════════════════════════════════════════ */
static void build_dct_matrix(void)
{
    const float inv_N  = sqrtf(1.0f / (float)MFCC_N_MEL);
    const float inv_2N = sqrtf(2.0f / (float)MFCC_N_MEL);
    const float pi     = 3.14159265358979323846f;

    for (int k = 0; k < MFCC_N_MFCC; k++) {
        float *row   = s_dct + k * MFCC_N_MEL;
        float  scale = (k == 0) ? inv_N : inv_2N;
        for (int n = 0; n < MFCC_N_MEL; n++) {
            row[n] = scale * cosf(pi * (float)k * (2.0f * (float)n + 1.0f)
                                  / (2.0f * (float)MFCC_N_MEL));
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 * PUBLIC: dsp_mfcc_init()
 * ══════════════════════════════════════════════════════════════════ */
int dsp_mfcc_init(void)
{
    arm_biquad_cascade_df2T_init_f32(
        &s_bp_inst, BP_N_STAGES,
        (float32_t *)s_bp_coeffs, s_bp_state);

    build_hamming();
    build_mel_filterbank();
    build_dct_matrix();

    arm_status st = arm_rfft_fast_init_f32(&s_rfft, MFCC_FFT_SIZE);
    if (st != ARM_MATH_SUCCESS) {
        LOG_ERR("arm_rfft_fast_init_f32 failed (status=%d)", (int)st);
        return -1;
    }

    /* ── v7.2: scratch-buffer bounds checks ─────────────────────── */
    if (cfg->frame_samples > MAX_FFT_SIZE) {
        LOG_ERR("frame_samples (%d) > MAX_FFT_SIZE (%d) -- "
                "increase MAX_FFT_SIZE in dsp_mfcc.c",
                cfg->frame_samples, MAX_FFT_SIZE);
        return -3;
    }
    if (cfg->fft_size > MAX_FFT_SIZE) {
        LOG_ERR("fft_size (%d) > MAX_FFT_SIZE (%d) -- "
                "increase MAX_FFT_SIZE in dsp_mfcc.c",
                cfg->fft_size, MAX_FFT_SIZE);
        return -4;
    }
    if (cfg->n_fft_bins > MAX_N_FFT_BINS) {
        LOG_ERR("n_fft_bins (%d) > MAX_N_FFT_BINS (%d) -- "
                "increase MAX_N_FFT_BINS in dsp_mfcc.c",
                cfg->n_fft_bins, MAX_N_FFT_BINS);
        return -5;
    }
    if (cfg->n_mel > MAX_N_MEL) {
        LOG_ERR("n_mel (%d) > MAX_N_MEL (%d) -- "
                "increase MAX_N_MEL in dsp_mfcc.c",
                cfg->n_mel, MAX_N_MEL);
        return -6;
    }
    if (cfg->n_mfcc > MAX_N_MFCC) {
        LOG_ERR("n_mfcc (%d) > MAX_N_MFCC (%d) -- "
                "increase MAX_N_MFCC in dsp_mfcc.c",
                cfg->n_mfcc, MAX_N_MFCC);
        return -7;
    }
    if (cfg->n_fft_bins != cfg->fft_size / 2 + 1) {
        LOG_ERR("n_fft_bins (%d) != fft_size/2 + 1 (%d) -- "
                "config self-inconsistent",
                cfg->n_fft_bins, cfg->fft_size / 2 + 1);
        return -8;
    }

    memset(p, 0, sizeof(*p));
    p->cfg = cfg;

    arm_biquad_cascade_df2T_init_f32(
        &p->bp_inst, BP_N_STAGES,
        (float32_t *)cfg->bp_coeffs, p->bp_state);

    arm_status st = arm_rfft_fast_init_f32(&p->rfft, (uint16_t)cfg->fft_size);
    if (st != ARM_MATH_SUCCESS) {
        LOG_ERR("arm_rfft_fast_init_f32 failed (fft_size=%d, status=%d) -- "
                "CMSIS-DSP needs CONFIG_CMSIS_DSP_TRANSFORM=y AND lookup "
                "tables compiled for that size",
                cfg->fft_size, (int)st);
        return -2;
    }

    LOG_INF("DSP+MFCC init OK: rate=%d frame=%d hop=%d fft=%d "
            "n_fft_bins=%d mel=%d mfcc=%d",
            cfg->target_rate, cfg->frame_samples, cfg->hop_samples,
            cfg->fft_size, cfg->n_fft_bins, cfg->n_mel, cfg->n_mfcc);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * PUBLIC: dsp_mfcc_reset()
 * Call once before each new recording session.
 * ══════════════════════════════════════════════════════════════════ */
void dsp_mfcc_reset(void)
{
    memset(s_bp_state, 0, sizeof(s_bp_state));
    s_dec_idx = 0;
}

/* ══════════════════════════════════════════════════════════════════
 * PUBLIC: dsp_mfcc_feed_chunk()
 * Bandpass + decimate one SAADC half-buffer into s_decimated[].
 * Call from saadc_event_handler for each NRFX_SAADC_EVT_DONE.
 * ══════════════════════════════════════════════════════════════════ */
void dsp_mfcc_feed_chunk(const int16_t *pcm_8k, int n)
{
    /* Clamp to scratch buffer size just in case */
    if (n > BP_SCRATCH_MAX) {
        n = BP_SCRATCH_MAX;
    }

    /* int16 → float32 */
    for (int i = 0; i < n; i++) {
        s_bp_scratch[i] = (float32_t)pcm_8k[i] / 32768.0f;
    }

    /* Bandpass in-place (state carries over between chunks) */
    arm_biquad_cascade_df2T_f32(&s_bp_inst, s_bp_scratch, s_bp_scratch,
                                (uint32_t)n);

    /* Decimate ÷4 → int16 into s_decimated[] */
    for (int i = 0; i < n; i += DSP_DECIMATE_FACTOR) {
        if (s_dec_idx < DECIMATED_SAMPLES) {
            s_decimated[s_dec_idx++] = (int16_t)(s_bp_scratch[i] * 32768.0f);
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 * INTERNAL: compute one MFCC frame from int16 signal at 2 kHz.
 * @param sig_2k   Pointer to frame start in s_decimated[].
 * @param out_f32  Output: MFCC_N_MFCC floats written to s_mfcc_frame[].
 * ══════════════════════════════════════════════════════════════════ */
static void compute_one_frame(const int16_t *sig_2k, float32_t *out_f32)
{
    /* Step 1: int16→float32, apply Hamming window, zero-pad to FFT size */
    for (int i = 0; i < MFCC_FRAME_SAMPLES; i++) {
        s_fft_in[i] = ((float32_t)sig_2k[i] / 32768.0f) * s_hamming[i];
    }
    memset(s_fft_in + MFCC_FRAME_SAMPLES, 0,
           (MFCC_FFT_SIZE - MFCC_FRAME_SAMPLES) * sizeof(float32_t));

    /* Step 2: RFFT — output is interleaved Re/Im, Nyquist in slot [1] */
    arm_rfft_fast_f32(&s_rfft, s_fft_in, s_fft_out, 0);

    /* Step 3: power spectrum Re²+Im²
     * CMSIS rfft_fast layout: [DC_re, Nyq_re, re1, im1, re2, im2, ...] */
    s_power[0]                = s_fft_out[0] * s_fft_out[0];
    s_power[MFCC_N_FFT_BINS - 1] = s_fft_out[1] * s_fft_out[1];
    for (int k = 1; k < MFCC_N_FFT_BINS - 1; k++) {
        float32_t re = s_fft_out[2 * k];
        float32_t im = s_fft_out[2 * k + 1];
        s_power[k] = re * re + im * im;
    }

    /* Step 4: mel filterbank dot products */
    for (int m = 0; m < MFCC_N_MEL; m++) {
        float32_t  energy = 0.0f;
        const float32_t *row = s_mel_fb + m * MFCC_N_FFT_BINS;
        for (int k = 0; k < MFCC_N_FFT_BINS; k++) {
            energy += row[k] * s_power[k];
        }
        /* Step 5: log with floor */
        s_mel[m] = logf(energy > 1e-6f ? energy : 1e-6f);
    }

    /* Step 6: DCT-II */
    for (int k = 0; k < MFCC_N_MFCC; k++) {
        float32_t  c   = 0.0f;
        const float32_t *row = s_dct + k * MFCC_N_MEL;
        for (int n = 0; n < MFCC_N_MEL; n++) {
            c += row[n] * s_mel[n];
        }
        out_f32[k] = c;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * PUBLIC: dsp_mfcc_finish()
 *
 * After recording is done (all chunks fed), compute MFCC frames
 * from s_decimated[] and invoke cb() once per frame.
 * No large output buffer needed — s_mfcc_frame[20] is reused each time.
 * ══════════════════════════════════════════════════════════════════ */
int dsp_mfcc_finish(dsp_mfcc_frame_cb_t cb)
{
    if (!cb) {
        LOG_ERR("dsp_mfcc_finish: callback is NULL");
        return -1;
    }

    LOG_INF("Bandpass+decimate done: %d samples at 2 kHz", s_dec_idx);

    int n_frames = 0;

    for (int f = 0; f < MFCC_N_FRAMES; f++) {
        int start = f * MFCC_HOP_SAMPLES;
        if (start + MFCC_FRAME_SAMPLES > s_dec_idx) {
            LOG_WRN("MFCC: ran out of samples at frame %d (start=%d, dec_idx=%d)",
                    f, start, s_dec_idx);
            break;
        }

        compute_one_frame(&s_decimated[start], s_mfcc_frame);
        cb(f, s_mfcc_frame);
        n_frames++;
    }

    LOG_INF("MFCC done: %d frames × %d coefficients", n_frames, MFCC_N_MFCC);
    return n_frames;
}