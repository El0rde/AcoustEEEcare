/**
 * dsp_mfcc.c
 * ==========
 * Multi-instance, streaming DSP + MFCC pipeline. Zephyr/nRF52840.
 *
 * v7.1 CIRCULAR-WINDOW OPTIMIZATION:
 *   - Sliding window is now a true circular buffer indexed by window_head.
 *     The per-decimated-sample memmove of (frame_samples-1)*2 bytes is
 *     eliminated. For the lung pipeline this saves ~792 KB/s of memmove
 *     bandwidth and the associated branch/cache pressure.
 *   - Frame compute walks the circular buffer in two segments (head..end,
 *     0..head-1). Each segment is contiguous so the inner loop stays
 *     tight and branch-free.
 *
 * Everything else (shared scratch buffers, public API, frame layout)
 * is unchanged from v7.0.
 */

#include "dsp_mfcc.h"
#include <arm_math.h>
#include <string.h>
#include <math.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(dsp_mfcc);

/* ================================================================
 * SHARED SCRATCH BUFFERS
 * ================================================================ */

#define BP_SCRATCH_MAX   512
#define MAX_FFT_SIZE     512
#define MAX_N_FFT_BINS   257
#define MAX_N_MEL        25
#define MAX_N_MFCC       25

static float32_t bp_scratch[BP_SCRATCH_MAX];
static float32_t fft_in    [MAX_FFT_SIZE];
static float32_t fft_out   [MAX_FFT_SIZE];
static float32_t power     [MAX_N_FFT_BINS];
static float32_t mel_buf   [MAX_N_MEL];
static float32_t mfcc_out  [MAX_N_MFCC];

/* ================================================================
 * INTERNAL: compute one MFCC frame from the circular window.
 *
 * The window is filled in circular fashion. The OLDEST sample lives
 * at index window_head (it's the next slot to be overwritten).
 * Reading proceeds: window_head, window_head+1, ..., frame_samples-1,
 * 0, 1, ..., window_head-1.
 *
 * We read in at most two contiguous segments to keep the inner loop
 * tight.
 * ================================================================ */
static void compute_one_frame_streaming(dsp_mfcc_pipeline_t *p)
{
    const dsp_mfcc_config_t *cfg = p->cfg;
    const int frame_samples = cfg->frame_samples;
    const int fft_size      = cfg->fft_size;
    const int n_fft_bins    = cfg->n_fft_bins;
    const int n_mel         = cfg->n_mel;
    const int n_mfcc        = cfg->n_mfcc;
    const int16_t *win      = p->window;
    const float32_t *ham    = cfg->hamming;
    const int head          = p->window_head;

    /* Step 1: int16->float32 * Hamming, walking the ring in 2 segments.
     * Segment A: [head, frame_samples)  -> outputs fft_in[0 .. seg_a-1]
     * Segment B: [0, head)              -> outputs fft_in[seg_a .. frame_samples-1]
     */
    int seg_a = frame_samples - head;   /* may be == frame_samples if head==0 */
    int i, ham_i = 0;

    for (i = 0; i < seg_a; i++) {
        fft_in[ham_i] = ((float32_t)win[head + i] / 32768.0f) * ham[ham_i];
        ham_i++;
    }
    for (i = 0; i < head; i++) {
        fft_in[ham_i] = ((float32_t)win[i] / 32768.0f) * ham[ham_i];
        ham_i++;
    }

    /* Zero-pad to fft_size */
    memset(fft_in + frame_samples, 0,
           (fft_size - frame_samples) * sizeof(float32_t));

    /* Step 2: RFFT */
    arm_rfft_fast_f32(&p->rfft, fft_in, fft_out, 0);

    /* Step 3: power spectrum */
    power[0]              = fft_out[0] * fft_out[0];
    power[n_fft_bins - 1] = fft_out[1] * fft_out[1];
    for (int k = 1; k < n_fft_bins - 1; k++) {
        float32_t re = fft_out[2 * k];
        float32_t im = fft_out[2 * k + 1];
        power[k] = re * re + im * im;
    }

    /* Step 4 + 5: mel filterbank + log */
    for (int m = 0; m < n_mel; m++) {
        float32_t energy = 0.0f;
        const float32_t *row = cfg->mel_fb + m * n_fft_bins;
        for (int k = 0; k < n_fft_bins; k++) {
            energy += row[k] * power[k];
        }
        mel_buf[m] = logf(energy > 1e-6f ? energy : 1e-6f);
    }

    /* Step 6: DCT-II */
    for (int k = 0; k < n_mfcc; k++) {
        float32_t c = 0.0f;
        const float32_t *row = cfg->dct + k * n_mel;
        for (int n = 0; n < n_mel; n++) {
            c += row[n] * mel_buf[n];
        }
        mfcc_out[k] = c;
    }
}

/* ================================================================
 * PUBLIC: dsp_mfcc_init()
 * ================================================================ */
int dsp_mfcc_init(dsp_mfcc_pipeline_t *p, const dsp_mfcc_config_t *cfg)
{
    if (!p || !cfg) {
        return -1;
    }

    memset(p, 0, sizeof(*p));
    p->cfg = cfg;

    arm_biquad_cascade_df2T_init_f32(
        &p->bp_inst, BP_N_STAGES,
        (float32_t *)cfg->bp_coeffs, p->bp_state);

    arm_status st = arm_rfft_fast_init_f32(&p->rfft, (uint16_t)cfg->fft_size);
    if (st != ARM_MATH_SUCCESS) {
        LOG_ERR("arm_rfft_fast_init_f32 failed (fft_size=%d, status=%d)",
                cfg->fft_size, (int)st);
        return -2;
    }

    LOG_INF("DSP+MFCC init OK: rate=%d frame=%d hop=%d mel=%d mfcc=%d",
            cfg->target_rate, cfg->frame_samples, cfg->hop_samples,
            cfg->n_mel, cfg->n_mfcc);
    return 0;
}

/* ================================================================
 * PUBLIC: dsp_mfcc_reset()
 * ================================================================ */
void dsp_mfcc_reset(dsp_mfcc_pipeline_t *p)
{
    const dsp_mfcc_config_t *cfg = p->cfg;

    memset(p->bp_state, 0, sizeof(p->bp_state));

    if (p->window) {
        memset(p->window, 0, cfg->frame_samples * sizeof(int16_t));
    }

    p->window_fill   = 0;
    p->window_head   = 0;
    p->decim_phase   = cfg->decimate_factor - 1;
    p->hop_counter   = cfg->frame_samples;
    p->frames_emitted = 0;
}

/* ================================================================
 * PUBLIC: dsp_mfcc_set_callback()
 * ================================================================ */
void dsp_mfcc_set_callback(dsp_mfcc_pipeline_t *p,
                           void (*cb)(int idx, const float *c, void *user),
                           void *user)
{
    p->frame_cb  = cb;
    p->user_data = user;
}

/* ================================================================
 * PUBLIC: dsp_mfcc_feed_chunk()
 *
 * Streaming with circular window:
 *   - decim_phase tracks decimation
 *   - each kept sample is written at window[window_head]; head++ mod frame_samples
 *   - window_fill saturates at frame_samples (so we know when we have
 *     a full window for the first time)
 *   - hop_counter decrements per decimated sample; frame fires when
 *     counter <= 0 and window is full
 * ================================================================ */
void dsp_mfcc_feed_chunk(dsp_mfcc_pipeline_t *p, const int16_t *pcm_8k, int n)
{
    const dsp_mfcc_config_t *cfg = p->cfg;
    const int frame_samples = cfg->frame_samples;

    if (n > BP_SCRATCH_MAX) {
        n = BP_SCRATCH_MAX;
    }

    /* int16 -> float32 */
    for (int i = 0; i < n; i++) {
        bp_scratch[i] = (float32_t)pcm_8k[i] / 32768.0f;
    }

    /* Bandpass in-place */
    arm_biquad_cascade_df2T_f32(&p->bp_inst, bp_scratch, bp_scratch,
                                (uint32_t)n);

    /* Decimate + circular window + frame trigger */
    for (int i = 0; i < n; i++) {
        p->decim_phase++;
        if (p->decim_phase < cfg->decimate_factor) {
            continue;
        }
        p->decim_phase = 0;

        int16_t s = (int16_t)(bp_scratch[i] * 32768.0f);

        /* Circular write -- O(1), no memmove */
        p->window[p->window_head] = s;
        p->window_head++;
        if (p->window_head >= frame_samples) {
            p->window_head = 0;
        }
        if (p->window_fill < frame_samples) {
            p->window_fill++;
        }

        p->hop_counter--;
        if (p->hop_counter <= 0 && p->window_fill == frame_samples) {
            compute_one_frame_streaming(p);

            if (p->frame_cb) {
                p->frame_cb(p->frames_emitted, mfcc_out, p->user_data);
            }

            p->frames_emitted++;
            p->hop_counter = cfg->hop_samples;
        }
    }
}

/* ================================================================
 * PUBLIC: dsp_mfcc_finish()
 * ================================================================ */
int dsp_mfcc_finish(dsp_mfcc_pipeline_t *p)
{
    if (!p || !p->cfg) {
        return -1;
    }

    LOG_INF("MFCC finish: %d frames emitted (expected %d), n_mfcc=%d",
            p->frames_emitted, p->cfg->n_frames_expected, p->cfg->n_mfcc);

    if (p->frames_emitted != p->cfg->n_frames_expected) {
        LOG_WRN("Frame count mismatch! got=%d expected=%d",
                p->frames_emitted, p->cfg->n_frames_expected);
    }

    return p->frames_emitted;
}