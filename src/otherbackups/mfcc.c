#include <stdint.h>
#include <zephyr/kernel.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/logging/log.h>
#include <math.h>
#include <string.h>
#include "fft_size1024_bins513.h"        
#include "bandpass_resample.h"

LOG_MODULE_REGISTER(mfcc, LOG_LEVEL_INF);

#ifndef M_PI
#  define M_PI 3.14159265358979323846f
#endif

/* ─── MFCC parameters ───────────────────────────────────────── */
#define NUM_MFCC            16
/* SAMPLE_RATE is now 8 kHz after downsampling from 16 kHz DMIC */
/* #define SAMPLE_RATE      16000 */   /* ← old: DMIC native rate             */
#define SAMPLE_RATE         8000        /* ← new: rate after bandpass_resample */
#define DMIC_SAMPLE_RATE    16000       /* DMIC hardware always runs at 16 kHz */
#define WINDOWING_MS        30
#define OVERLAP_MS          13
#define FRAME_SAMPLES       ((int)(WINDOWING_MS * SAMPLE_RATE / 1000))
#define OVERLAP_SAMPLES     ((int)(OVERLAP_MS * SAMPLE_RATE / 1000))  
#define HOP_SIZE            (FRAME_SAMPLES - OVERLAP_SAMPLES)          
#define FFT_SIZE            1024
#define NUM_BINS            (FFT_SIZE / 2 + 1)
#define FREQ_LOW            10.0f    
#define FREQ_HIGH           200.0f   

#define NUM_MEL_FILTERS     40          // Default MATLAB
// #define PRE_EMPHASIS        0.97f    // Not used in MATLAB
// #define LOW_FREQ_HZ         80.0f

/* ─── DMIC / slab ────────────────────────────────────────────── */
/* DMIC delivers HOP_SIZE samples at 8 kHz per hop.
 * But the mic runs at 16 kHz, so we must read 2× as many raw samples
 * before downsampling — DMIC_HOP_SAMPLES is the raw 16 kHz block size. */
#define DMIC_HOP_SAMPLES    (HOP_SIZE * 2)
/* #define BLOCK_SIZE       (HOP_SIZE * sizeof(int16_t)) */       /* ← old: 8 kHz sized block  */
#define BLOCK_SIZE          (DMIC_HOP_SAMPLES * sizeof(int16_t)) /* ← new: 16 kHz sized block */
#define BLOCK_COUNT  3
K_MEM_SLAB_DEFINE_STATIC(audio_slab, BLOCK_SIZE, BLOCK_COUNT, 4);

/* ─── Mel helpers ────────────────────────────────────────────── */
// static inline float hz_to_mel(float hz)  { return 1127.0f * logf(1.0f + hz / 700.0f); }
// static inline float mel_to_hz(float mel) { return 700.0f  * (expf(mel / 1127.0f) - 1.0f); }

/* ─── Static buffers ─────────────────────────────────────────────────────── */
static float        mel_fb[NUM_MEL_FILTERS][NUM_BINS];
static creal_T      fft_buf[NUM_BINS];  
static float        power[NUM_BINS];
static float        log_energy[NUM_MEL_FILTERS];
static float        mfcc_out[NUM_MFCC + 1];
static int16_t      overlap_buf[FRAME_SAMPLES];
static int          samples_filled = 0;
/* fft.c takes int16, but we need float precision for energy. */
static float        windowed_f[FFT_SIZE];        /* 1024 — zero-padded float  */
static int16_t      windowed_i[FFT_SIZE];        /* 1024 — zero-padded int16  */
/* static int16_t   filtered_buf[FRAME_SAMPLES]; */  /* ← old: per-frame bandpass buffer;
                                                       *   filtering now happens per-hop
                                                       *   in the main loop via bandpass_resample() */
static int16_t      resampled_hop[HOP_SIZE];     /* 8 kHz hop after bandpass + decimate */

/* ─── Slaney band edges — matches MATLAB's default mfcc() band edges ────── */
#define SLANEY_NUM_EDGES  42    // MATLAB Default

static void compute_slaney_band_edges(float *edges)
{
    /* Direct port of MATLAB's slaneybandedges():
     *   First 13 edges: linear spacing starting at 133.33 Hz, step 66.67 Hz
     *   Edges 14-42:    geometric spacing, each multiplied by 1.0711703       */
    float factor = 133.33333333333333f;

    for (int i = 1; i <= 13; i++)
        edges[i - 1] = factor + (factor / 2.0f) * (float)(i - 1);

    for (int i = 14; i <= 42; i++)
        edges[i - 1] = edges[i - 2] * 1.0711703f;
}

/* ─── build_mel_filterbank() ─────────────────────────────────── */
// static void build_mel_filterbank(void)
// {
//     ... (old commented-out implementation) ...
// }

static void build_mel_filterbank(void)
{
    float band_edges_hz[SLANEY_NUM_EDGES];
    compute_slaney_band_edges(band_edges_hz);

    memset(mel_fb, 0, sizeof(mel_fb));

    for (int filter = 0; filter < NUM_MEL_FILTERS; filter++)
    {
        float lo     = band_edges_hz[filter];
        float center = band_edges_hz[filter + 1];
        float hi     = band_edges_hz[filter + 2];
        float bw_rise = center - lo;
        float bw_fall = hi - center;

        for (int k = 0; k < NUM_BINS; k++)
        {
            /* Hz frequency of bin k — matches MATLAB:
             * linFq = (0:NFFT-1)/NFFT * fs
             * NOTE: SAMPLE_RATE is now 8000 — filterbank is built for 8 kHz  */
            float hz = (float)k / (float)FFT_SIZE * (float)SAMPLE_RATE;

            if (hz > lo && hz <= center && bw_rise > 0.0f)
                mel_fb[filter][k] = (hz - lo) / bw_rise;
            else if (hz > center && hz < hi && bw_fall > 0.0f)
                mel_fb[filter][k] = (hi - hz) / bw_fall;
        }

        /* Bandwidth normalization — matches MATLAB 'Bandwidth':
         * weightPerBand = (bandEdges(m+2) - bandEdges(m)) / 2              */
        float bandwidth = (hi - lo) / 2.0f;
        if (bandwidth > 0.0f)
            for (int k = 0; k < NUM_BINS; k++)
                mel_fb[filter][k] /= bandwidth;
    }
}

/* ─── compute_mfcc() ─────────────────────────────────────────────────────── */
static void compute_mfcc(const int16_t *pcm)
{
    /* 1. Hamming window — periodic, matches hamming(N,"periodic")
     *    Formula: 0.54 - 0.46 * cos(2*pi*n / N)   <-- divide by N, not N-1
     *
     *    NOTE: bandpass_filter() removed — filtering + resampling now happens
     *    per-hop in main() via bandpass_resample() before frames are assembled.
     *    pcm here is already bandpassed + downsampled to 8 kHz.               */
    /* bandpass_filter(pcm, filtered_buf, FRAME_SAMPLES); */  /* ← removed: done in main loop */
    for (int n = 0; n < FRAME_SAMPLES; n++) {
        /* float signal = (float)filtered_buf[n]; */  /* ← old: used filtered_buf */
        float signal = (float)pcm[n];                 /* ← new: pcm already filtered */
        float window = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * n / (float)FRAME_SAMPLES);
        windowed_f[n]  = signal * window;
        windowed_i[n]  = (int16_t)(signal * window);
    }
    for (int n = FRAME_SAMPLES; n < FFT_SIZE; n++) {
        windowed_f[n] = 0.0f;
        windowed_i[n] = 0;
    }

    /* 2. Log frame energy — matches MATLAB: E = sum(y.^2); logE = log(E)
     *    y in MATLAB is the windowed int16-scaled signal before FFT
     *    Use logf (natural log) here — this is the frame energy, NOT mel energy */
    float frame_energy = 0.0f;
    for (int n = 0; n < FRAME_SAMPLES; n++) {
        float s = windowed_f[n] / 32768.0f;     /* normalize to [-1,1] range    */
        frame_energy += s * s;
    }
    /* MATLAB floor: E(E==0) = realmin  →  realmin('single') = 1.1755e-38 */
    mfcc_out[0] = logf(frame_energy < 1.1755e-38f ? 1.1755e-38f : frame_energy);

    /* 3. MATLAB FFT — takes raw int16, casts to double internally */
    fft_entry(windowed_i, fft_buf);

    /* 4. Magnitude spectrum — abs(fft output), bins 0..NUM_BINS-1 */
    for (int k = 0; k < NUM_BINS; k++) {
        float re  = (float)fft_buf[k].re;
        float im  = (float)fft_buf[k].im;
        power[k]  = sqrtf(re * re + im * im);
    }

    /* 5. Mel filterbank energy + log10 — matches cepstralCoefficients log10() */
    for (int m = 0; m < NUM_MEL_FILTERS; m++) {
        float e = 0.0f;
        for (int k = 0; k < NUM_BINS; k++)
            e += mel_fb[m][k] * power[k];
        /* MATLAB floor: realmin('double') = 2.2251e-308, use float safe floor */
        log_energy[m] = log10f(e < 1e-37f ? 1e-37f : e);
    }

    /* 6. DCT-II orthonormal — matches audio.internal.createDCTmatrix()
     *    scale_dc   = sqrt(1/M)
     *    scale_rest = sqrt(2/M)
     *    MATLAB returns coeffs 1..NumCoeffs (0-indexed: skips DC component 0) */
    float scale_dc   = sqrtf(1.0f / (float)NUM_MEL_FILTERS);
    float scale_rest = sqrtf(2.0f / (float)NUM_MEL_FILTERS);

    for (int n = 0; n < NUM_MFCC; n++) {
        float sum = 0.0f;
        for (int m = 0; m < NUM_MEL_FILTERS; m++)
            sum += log_energy[m] * cosf((float)M_PI * (float)n * ((float)m + 0.5f) / (float)NUM_MEL_FILTERS);
        mfcc_out[n + 1] = (n == 0 ? scale_dc : scale_rest) * sum;
    }
}

/* ─── main() ─────────────────────────────────────────────────── */
int main(void)
{
    LOG_INF("MFCC on XIAO nRF52840 Sense (Zephyr)");

    fft_entry_initialize();
    /* bandpass_resample_init(); */               /* ← old: no mode argument    */
    bandpass_resample_init(BP_MODE_10_200HZ);     /* ← new: select active mode;
                                                   *   swap to BP_MODE_100_1000HZ
                                                   *   for 100–1000 Hz band     */
    build_mel_filterbank();
    LOG_INF("Filterbank ready. Streaming MFCCs...");

    // const struct device *dmic_dev = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));
    // if (!device_is_ready(dmic_dev)) {
    //     LOG_ERR("DMIC device not ready");
    //     return -ENODEV;
    // }

    struct pcm_stream_cfg stream = {
        /* .pcm_rate = SAMPLE_RATE, */            /* ← old: would set 8000 on HW */
        .pcm_rate   = DMIC_SAMPLE_RATE,           /* ← new: HW always at 16 kHz  */
        .pcm_width  = 16,
        .block_size = BLOCK_SIZE,                 /* now sized for 16 kHz hop     */
        .mem_slab   = &audio_slab,
    };

    struct dmic_cfg cfg = {
        .io = {
            .min_pdm_clk_freq = 1000000,
            .max_pdm_clk_freq = 3500000,
            .min_pdm_clk_dc   = 40,
            .max_pdm_clk_dc   = 60,
        },
        .streams = &stream,
        .channel = {
            .req_num_streams = 1,
            .req_num_chan    = 1,
            // .req_chan_map_lo = dmic_build_channel_map(0, 0, PDM_CHAN_LEFT),
        },
    };

    int ret = dmic_configure(dmic_dev, &cfg);
    if (ret < 0) { LOG_ERR("dmic_configure failed: %d", ret); return ret; }

    ret = dmic_trigger(dmic_dev, DMIC_TRIGGER_START);
    if (ret < 0) { LOG_ERR("dmic_trigger START failed: %d", ret); return ret; }

    while (1)
    {
        void    *buf;
        uint32_t size;

        ret = dmic_read(dmic_dev, 0, &buf, &size, K_FOREVER);
        if (ret < 0) { LOG_ERR("dmic_read failed: %d", ret); break; }

        /* buf holds DMIC_HOP_SAMPLES int16 samples at 16 kHz */
        const int16_t *new_hop_16k = (const int16_t *)buf;

        /* Bandpass filter + downsample 16 kHz → 8 kHz into resampled_hop */
        bandpass_resample(new_hop_16k, resampled_hop, DMIC_HOP_SAMPLES);

        /* Slide overlap buffer and append the new 8 kHz hop
         * (was: directly copying the raw 16 kHz new_hop) */
        /* memmove(overlap_buf, overlap_buf + HOP_SIZE, (FRAME_SAMPLES - HOP_SIZE) * sizeof(int16_t)); */ /* ← same logic, kept for clarity */
        /* memcpy(overlap_buf + (FRAME_SAMPLES - HOP_SIZE), new_hop, HOP_SIZE * sizeof(int16_t));       */ /* ← old: copied raw 16 kHz samples */
        memmove(overlap_buf,
                overlap_buf + HOP_SIZE,
                (FRAME_SAMPLES - HOP_SIZE) * sizeof(int16_t));
        memcpy(overlap_buf + (FRAME_SAMPLES - HOP_SIZE),
               resampled_hop,                     /* ← new: copy 8 kHz filtered hop */
               HOP_SIZE * sizeof(int16_t));

        samples_filled += HOP_SIZE;
        if (samples_filled > FRAME_SAMPLES)
            samples_filled = FRAME_SAMPLES;

        if (samples_filled < FRAME_SAMPLES) {
            /* Not enough data yet — need 3 hops before first frame */
            k_mem_slab_free(&audio_slab, &buf);
            continue;
        }

        compute_mfcc(overlap_buf);

        k_mem_slab_free(&audio_slab, &buf);
    }

    fft_entry_terminate();
    dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
    return 0;
}