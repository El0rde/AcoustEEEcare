/**
 * dsp_mfcc.h
 * ==========
 * Multi-instance DSP + MFCC pipeline for AcoustEEEcare on nRF52840/Zephyr.
 *
 * v7.1 CIRCULAR-WINDOW OPTIMIZATION:
 *   Adds window_head to the pipeline struct so the sliding window
 *   can be addressed as a circular buffer (no per-sample memmove).
 *   Public API unchanged from v7.0.
 */

#ifndef DSP_MFCC_H
#define DSP_MFCC_H

#include <stdint.h>
#include <arm_math.h>

/* ---- Universal constants (same for both pipelines) ---- */
#define DSP_SAMPLING_RATE    8000   /* 8 kHz SAADC capture rate */
#define HALF_BUF_SAMPLES     512    /* SAADC ping-pong half-buffer size */
#define BP_N_STAGES          4      /* Butterworth-4 -> 4 biquad sections */

/* ---- Pipeline configuration struct ---- */
typedef struct dsp_mfcc_config {
    /* Decimation */
    int decimate_factor;
    int target_rate;

    /* Frame structure */
    int frame_samples;
    int hop_samples;
    int n_frames_expected;

    /* FFT */
    int fft_size;
    int n_fft_bins;

    /* MFCC */
    int n_mel;
    int n_mfcc;
    float f_low_hz;
    float f_high_hz;

    /* Pointers to flash-constant tables */
    const float32_t *bp_coeffs;
    const float32_t *hamming;
    const float32_t *mel_fb;
    const float32_t *dct;
} dsp_mfcc_config_t;

/* ---- Pipeline state struct ---- */
typedef struct dsp_mfcc_pipeline {
    const dsp_mfcc_config_t *cfg;

    /* Bandpass biquad state */
    arm_biquad_cascade_df2T_instance_f32 bp_inst;
    float32_t bp_state[BP_N_STAGES * 2];

    /* FFT instance */
    arm_rfft_fast_instance_f32 rfft;

    /* Streaming state: circular window over decimated samples */
    int16_t  *window;       /* caller-allocated, size = cfg->frame_samples */
    int       window_fill;  /* samples currently in window (0..frame_samples) */
    int       window_head;  /* circular write index (0..frame_samples-1).
                             * Also points at OLDEST sample once window full,
                             * because that slot is the next to be overwritten. */
    int       hop_counter;  /* samples until next frame boundary */
    int       decim_phase;  /* decimation phase (0..decimate_factor-1) */

    /* Frame counter */
    int       frames_emitted;

    /* Per-frame callback */
    void    (*frame_cb)(int frame_idx, const float *coeffs, void *user);
    void     *user_data;
} dsp_mfcc_pipeline_t;

/* ---- Public API (unchanged from v7.0) ---- */
int  dsp_mfcc_init(dsp_mfcc_pipeline_t *p, const dsp_mfcc_config_t *cfg);
void dsp_mfcc_reset(dsp_mfcc_pipeline_t *p);
void dsp_mfcc_set_callback(dsp_mfcc_pipeline_t *p,
                           void (*cb)(int idx, const float *c, void *user),
                           void *user);
void dsp_mfcc_feed_chunk(dsp_mfcc_pipeline_t *p, const int16_t *pcm_8k, int n);
int  dsp_mfcc_finish(dsp_mfcc_pipeline_t *p);

#endif /* DSP_MFCC_H */