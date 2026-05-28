/*
 * AcoustEEEcare — SAADC BLE + TFLite Micro Edition
 * ==================================================
 * v8.4 — In-cycle keepalive fix
 *
 * CHANGES FROM v8.3:
 *
 *   [STAB FIX 5] PING keepalive moved INSIDE record_and_stream().
 *                In v8.3 the PING code lived in the main loop AFTER the
 *                record_and_stream() call. Because record_and_stream() is a
 *                blocking function that runs for ~40-50 s, the main loop
 *                never reached the PING code during a recording cycle.
 *                Android's BLE stack has its own independent supervision
 *                timer (~20 s). With no traffic from the MCU during the
 *                20-35 s Phase-2 inference window, Android declared the
 *                link dead from its side — the MCU LED stayed green (MCU
 *                never saw its own timeout expire) but the Flutter app
 *                showed "Check connection." This was the root cause of the
 *                disconnect-after-26-recordings bug.
 *
 *                Fix: a keepalive_ping_if_due() helper is called at every
 *                natural pause inside record_and_stream() — after SAADC
 *                capture, during the DSP-done wait (polled in 2 s slices),
 *                between the two inferences, and after lung inference.
 *                This guarantees at least one PING per 3 s throughout the
 *                entire ~50 s blocking cycle.
 *
 *   [STAB FIX 6] BLE_SUPERVISION_TIMEOUT raised 4000 → 6000 (40 s → 60 s).
 *                Worst-case full cycle: 10 s capture + 6 s drain + 35 s
 *                inference = 51 s. 60 s gives comfortable margin for both
 *                the MCU-side timer and the Android-side timer (Android
 *                honours the connection parameter the MCU requests).
 *
 *   [STAB FIX 7] nack_buf_seq[] cleared between recordings.
 *                Previously only nack_buf_len[] was zeroed at the start of
 *                each REC. Stale seq numbers in the 4-slot NACK window from
 *                previous recordings could cause a late NACK to match a
 *                slot from a prior session and retransmit wrong data.
 *
 * ALL PREVIOUS FIXES (v8.3 / v8.2 / v8.1 / v7.9 / … / v6.8) PRESERVED.
 * ==================================================
 */

/* ── SD card is permanently disabled in this build ─────────────── */
#define USE_SD  false

#define BLE_AUDIO_LIVE  1   /* stream raw audio to phone live */
#define DSP_OFFLINE     0   /* MFCC computed live during capture */

/* ══════════════════════════════════════════════════════════════════
 * TUNEABLE TIMING CONSTANTS
 * All timing values in one place — edit here, takes effect everywhere.
 * ══════════════════════════════════════════════════════════════════ */

/* How long to wait for the previous recording's BLE TX drain to finish
 * before starting the next recording. This is a ceiling — in practice
 * drain completes in ~200-500 ms. Raise if you see "Pre-REC drain wait
 * timed out" in logs; lower if you want faster back-to-back recordings.
 * [STAB FIX 2] Raised from 2000 to 6000 ms: in a busy RF environment
 * the 80 KB audio drain can exceed 2 s, causing a timeout reset that
 * left stale packets in the Android BLE stack and broke rec 2+. */
#define PRE_REC_DRAIN_TIMEOUT_MS   6000

/* How long to wait for the BLE TX drain at the END of each recording,
 * after saadc_stop_streaming(). Same ceiling — drain is usually <1 s.
 * Raise if BLE throughput is poor (busy RF environment). */
#define TX_DONE_TIMEOUT_MS         5000

/* Keepalive PING interval when idle and connected (ms).
 * Prevents supervision timeout during idle between recordings.
 * Must be less than (supervision_timeout - connection_interval_max)
 * = (40000 ms - 30 ms). 3000 ms is conservative and safe. */
#define PING_INTERVAL_MS           3000

/* BLE supervision timeout is in 10 ms units.
 * [STAB FIX 1] Raised from 2000 (20 s) to 4000 (40 s).
 * [STAB FIX 6] Raised from 4000 (40 s) to 6000 (60 s).
 * Worst-case full cycle: 10 s capture + 6 s drain + 35 s inference = 51 s.
 * 60 s gives comfortable margin. Android honours the MCU's requested
 * supervision timeout, so this also extends the Android-side timer. */
#define BLE_SUPERVISION_TIMEOUT    6000

/* Small delay after sending START: header, before SAADC begins.
 * Gives the phone-side receiver time to prepare its buffer.
 * 10 ms is sufficient; only raise if receiver logs show a missing START. */
#define START_HEADER_DELAY_MS      10

/* ══════════════════════════════════════════════════════════════════
 * INCLUDES
 * ══════════════════════════════════════════════════════════════════ */
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>

#include <arm_math.h>   /* CMSIS-DSP Q15 biquad filter */

#include <zephyr/bluetooth/services/nus.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/irq.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/printk.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/sys/ring_buffer.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/sys/crc.h>          /* crc16_ccitt */
#include "zephyr/kernel/thread_stack.h"

#if defined(CONFIG_THREAD_ANALYZER)
#include <zephyr/debug/thread_analyzer.h>
#endif
#if defined(CONFIG_SYS_HEAP_RUNTIME_STATS)
#include <zephyr/sys/sys_heap.h>
extern struct sys_heap _system_heap;
#endif

#include <nrfx.h>
#include <drivers/nrfx_errors.h>
#include <nrfx_saadc.h>

#include "dsp_mfcc.h"
#include "heart_mfcc_config.h"
#include "lung_mfcc_config.h"
#include "cnn_norm_stats.h"

#include "tflm_inference.h"

LOG_MODULE_REGISTER(AcoustEEEcare);

/* ══════════════════════════════════════════════════════════════════
 * FORWARD DECLARATIONS
 * ══════════════════════════════════════════════════════════════════ */
extern const struct gpio_dt_spec red_led;
extern const struct gpio_dt_spec green_led;
extern const struct gpio_dt_spec blue_led;
static inline void led_on (const struct gpio_dt_spec *l);
static inline void led_off(const struct gpio_dt_spec *l);
static inline void led_set_white(void);
static inline void led_set_red(void);
static inline void led_set_green(void);
static inline void led_set_blue(void);
static inline void led_set_yellow(void);
static inline void led_set_purple(void);
static inline void led_set_cyan(void);
static int  saadc_start_streaming(void);
static void saadc_stop_streaming(void);
static void saadc_init_once(void);   /* [FIX F] one-time IRQ setup */

/* ══════════════════════════════════════════════════════════════════
 * SHARED TENSOR ARENA
 * 72 KB — arena_used_bytes logged after AllocateTensors().
 * ══════════════════════════════════════════════════════════════════ */
#define TENSOR_ARENA_BYTES  (72u * 1024u)

static uint8_t tensor_arena[TENSOR_ARENA_BYTES] __aligned(16);

/* ── int8 MFCC feature buffers ──────────────────────────────────── */
#define HEART_FEAT_N   (665 * 25)
#define LUNG_FEAT_N    (324 * 26)
static int8_t heart_features[HEART_FEAT_N] __aligned(4);
static int8_t lung_features [LUNG_FEAT_N]  __aligned(4);
static volatile uint32_t heart_feat_count = 0;
static volatile uint32_t lung_feat_count  = 0;

#define HEART_Q_SCALE 0.39602566f
#define HEART_Q_ZP    38
#define LUNG_Q_SCALE  0.33742353f
#define LUNG_Q_ZP     28

/* Keep the model path matched to the MFCC training pipeline by default.
 * BLE preview gain is applied only to the streamed/saved WAV copy. */
#define USE_OUTER_Q15_BPF_FOR_DSP  0
#define BLE_PREVIEW_GAIN           4
#define BLE_PREVIEW_REMOVE_DC      1

/* ══════════════════════════════════════════════════════════════════
 * RAM USAGE REPORT
 * ══════════════════════════════════════════════════════════════════ */
extern char __bss_start;
extern char __bss_end __attribute__((weak));
extern char _end      __attribute__((weak));
extern char _image_ram_start;
extern char _image_ram_end;

/* 32 KB BLE audio ring — ~3 s slack at 8 kHz int16 */
/* BLE TX priority 4 — preempts main thread during inference */
#define BLE_TX_PRIORITY       4
#define BLE_TX_STACK_SIZE     3072   /* enlarged for CRC/NACK path */
#define DSP_THREAD_STACK_SIZE 4096

static struct k_thread ble_tx_thread_data;
static struct k_thread dsp_thread_data;
static dsp_mfcc_pipeline_t heart_pipeline;
static dsp_mfcc_pipeline_t lung_pipeline;
static uint32_t heart_frame_count;
static uint32_t lung_frame_count;

volatile uint32_t g_tflm_arena_used_bytes = 0;

#if defined(CONFIG_THREAD_ANALYZER)
static void thread_stack_dump_cb(const struct k_thread *thread, void *user_data)
{
    ARG_UNUSED(user_data);
    size_t unused;
    if (k_thread_stack_space_get(thread, &unused) != 0) return;
    const char *name = k_thread_name_get((k_tid_t)thread);
    LOG_INF("  %-12s  unused=%5u B", name ? name : "?", (unsigned)unused);
}
#endif

static void report_ram_usage(const char *label)
{
    uintptr_t bss_start   = (uintptr_t)&__bss_start;
    uintptr_t bss_end     = (&__bss_end != NULL)
                              ? (uintptr_t)&__bss_end
                              : (uintptr_t)&_end;
    uintptr_t image_start = (uintptr_t)&_image_ram_start;
    uintptr_t image_end   = (uintptr_t)&_image_ram_end;

    uint32_t bss_bytes   = (uint32_t)(bss_end   - bss_start);
    uint32_t image_bytes = (uint32_t)(image_end - image_start);

    LOG_INF("================ RAM USAGE @ %s ================",
            label ? label : "unlabeled");
    LOG_INF("BSS region:    [0x%08lx .. 0x%08lx)  %u B (%u KB)",
            (unsigned long)bss_start, (unsigned long)bss_end,
            bss_bytes, bss_bytes / 1024);
    LOG_INF("Image region:  [0x%08lx .. 0x%08lx)  %u B (%u KB)",
            (unsigned long)image_start, (unsigned long)image_end,
            image_bytes, image_bytes / 1024);
    LOG_INF("Tensor arena:  declared=%u B  last_used=%u B",
            (unsigned)TENSOR_ARENA_BYTES,
            (unsigned)g_tflm_arena_used_bytes);

#if defined(CONFIG_SYS_HEAP_RUNTIME_STATS)
    {
        struct sys_memory_stats hs;
        if (sys_heap_runtime_stats_get(&_system_heap, &hs) == 0) {
            LOG_INF("Heap:          free=%u  alloc=%u  max_alloc=%u",
                    (unsigned)hs.free_bytes,
                    (unsigned)hs.allocated_bytes,
                    (unsigned)hs.max_allocated_bytes);
        }
    }
#else
    LOG_INF("Heap:          (CONFIG_SYS_HEAP_RUNTIME_STATS=n; stats unavailable)");
#endif

    size_t unused;
    if (k_thread_stack_space_get(&ble_tx_thread_data, &unused) == 0) {
        LOG_INF("BLE TX stack:  %u / %u B used (%u%% headroom)",
                (unsigned)(BLE_TX_STACK_SIZE - unused),
                (unsigned)BLE_TX_STACK_SIZE,
                (unsigned)(unused * 100u / BLE_TX_STACK_SIZE));
    }
    if (k_thread_stack_space_get(&dsp_thread_data, &unused) == 0) {
        LOG_INF("DSP stack:     %u / %u B used (%u%% headroom)",
                (unsigned)(DSP_THREAD_STACK_SIZE - unused),
                (unsigned)DSP_THREAD_STACK_SIZE,
                (unsigned)(unused * 100u / DSP_THREAD_STACK_SIZE));
    }

#if defined(CONFIG_THREAD_ANALYZER)
    LOG_INF("All threads (unused stack bytes):");
    k_thread_foreach(thread_stack_dump_cb, NULL);
#endif

    LOG_INF("Heart frames:  %u  (expected %d)",
            heart_frame_count, heart_pipeline.cfg->n_frames_expected);
    LOG_INF("Lung  frames:  %u  (expected %d)",
            lung_frame_count,  lung_pipeline.cfg->n_frames_expected);
    LOG_INF("=========================================================");
}

/* ══════════════════════════════════════════════════════════════════
 * SAADC CONFIG
 * ══════════════════════════════════════════════════════════════════ */
#define SAADC_CC_VALUE      2000U
#define SAADC_IRQ_PRIORITY  5

static const nrfx_saadc_channel_t saadc_channel_cfg = {
    .channel_config = {
        .resistor_p = NRF_SAADC_RESISTOR_DISABLED,
        .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
        .gain       = NRF_SAADC_GAIN1_4,
        .reference  = NRF_SAADC_REFERENCE_VDD4,
        .acq_time   = NRF_SAADC_ACQTIME_10US,
        .mode       = NRF_SAADC_MODE_SINGLE_ENDED,
        .burst      = NRF_SAADC_BURST_DISABLED,
    },
    .pin_p         = NRF_SAADC_INPUT_AIN0,
    .pin_n         = NRF_SAADC_INPUT_DISABLED,
    .channel_index = 0,
};

static volatile uint32_t saadc_dma_overruns = 0;

#define ENABLE_DC_REMOVAL
static int32_t dc_estimate = 0;

/* ══════════════════════════════════════════════════════════════════
 * CMSIS-DSP Q15 BAND-PASS FILTER
 * Range: 10–1000 Hz, 2nd-order Butterworth @ 8 kHz
 *
 * Covers both MFCC pipelines:
 *   Heart MFCC internal BPF:  10–200  Hz (Layer 2)
 *   Lung  MFCC internal BPF: 100–1000 Hz (Layer 2)
 *
 * Coefficients generated with postShift=2 (see bpf_coeff_generator.py):
 *   from scipy.signal import butter
 *   sos = butter(2, [10, 1000], btype='bandpass', fs=8000, output='sos')
 *   scale = 32768.0 / (2 ** 2)   # postShift=2 for Stage-2 headroom
 *
 * Format: {b0, b1, b2, -a1, -a2} per biquad (CMSIS-DSP df1 Q15).
 *
 * [NOISE FIX 1] postShift changed from 1 → 2.
 * Previous postShift=1 gave Stage-2 b=[16384,-32768,16384] which
 * operated at the Q15 saturation boundary, amplifying residual DC
 * and out-of-band noise. postShift=2 halves all values, giving
 * 2 bits of accumulator headroom with identical frequency response.
 * ══════════════════════════════════════════════════════════════════ */
#define HEART_BPF_STAGES  2

static const q15_t heart_bpf_coeffs[5 * HEART_BPF_STAGES] = {
    /* Stage 1 — 10-1000 Hz Butterworth @ 8kHz (postShift=2) */
     787, 1574, 787, 7817, -2790,
    /* float: b=[0.096040,0.192079,0.096040] -a=[0.954203,-0.340608] */
    /* Stage 2 — 10-1000 Hz Butterworth @ 8kHz (postShift=2) */
     8192, -16384, 8192, 16293, -8102,
    /* float: b=[1.000000,-2.000000,1.000000] -a=[1.988896,-0.988958] */
};
static q15_t                        heart_bpf_state[4 * HEART_BPF_STAGES];
static arm_biquad_casd_df1_inst_q15 heart_bpf;

/* ══════════════════════════════════════════════════════════════════
 * [BUG 1 FIX] Custom advertising parameter — 100–150 ms interval,
 * indefinite (no 30-second fast-advertising expiry like FAST_1).
 * ══════════════════════════════════════════════════════════════════ */
static const struct bt_le_adv_param acousteeecare_adv_param =
    BT_LE_ADV_PARAM_INIT(
        BT_LE_ADV_OPT_CONN,
        BT_GAP_ADV_FAST_INT_MIN_2,   /* 100 ms */
        BT_GAP_ADV_FAST_INT_MAX_2,   /* 150 ms */
        NULL
    );

/* ══════════════════════════════════════════════════════════════════
 * AUDIO PARAMS
 * ══════════════════════════════════════════════════════════════════ */
#define SAMPLING_RATE        8000
#define DURATION_S           10
#define TOTAL_AUDIO_SAMPLES  (SAMPLING_RATE * DURATION_S)
#define TOTAL_AUDIO_BYTES    (TOTAL_AUDIO_SAMPLES * sizeof(int16_t))

#define HALF_BUF_SAMPLES     512
#define HALF_BUF_BYTES       (HALF_BUF_SAMPLES * sizeof(int16_t))

static int16_t ping_pong[2][HALF_BUF_SAMPLES] __aligned(4);
static volatile uint8_t  next_dma_buf     = 1;
static volatile bool     analog_recording = false;
static uint16_t          nus_chunk_size   = 244;

/* ══════════════════════════════════════════════════════════════════
 * BLE AUDIO RING BUFFER + TX THREAD
 * ══════════════════════════════════════════════════════════════════ */
static K_SEM_DEFINE(half_produced_sem, 0, K_SEM_MAX_LIMIT);

#if BLE_AUDIO_LIVE
/* 32 KB — ~3 s headroom at 16 KB/s */
#define AUDIO_RING_BYTES  (32 * 1024)
RING_BUF_DECLARE(audio_ring, AUDIO_RING_BYTES);

static K_SEM_DEFINE(audio_data_sem, 0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(tx_done_sem,    0, 1);

static atomic_t  ring_drops;
static uint32_t  ring_high_water = 0;

static uint16_t tx_seq = 0;

/* Prevents the BLE TX thread from sending "finished\n" before the
 * recording session has started. Set true when analog_recording is first
 * observed true; reset to false at the start of each REC.
 * [FIX E] atomic_t for correct cross-thread visibility. */
static atomic_t ble_tx_saw_recording = ATOMIC_INIT(0);

/* NACK retransmit window — keeps last 4 chunks for retransmit. */
#define NACK_WINDOW  4

static uint8_t  nack_buf[NACK_WINDOW][251]   __aligned(4);
static uint16_t nack_buf_seq[NACK_WINDOW];
static uint16_t nack_buf_len[NACK_WINDOW];

static uint16_t  nack_seq_pending = 0xFFFF;
static K_MUTEX_DEFINE(nack_mutex);
#endif /* BLE_AUDIO_LIVE */

/* CRC-16 chunk header: [seq16][len16][crc16] = 6 bytes */
#define CHUNK_HEADER_BYTES  6

/* ══════════════════════════════════════════════════════════════════
 * DSP RING BUFFER + DSP THREAD
 * ══════════════════════════════════════════════════════════════════ */
/* 16 KB — matches BLE ring headroom */
#define DSP_RING_BYTES  (16 * 1024)
RING_BUF_DECLARE(dsp_ring, DSP_RING_BYTES);

static K_SEM_DEFINE(dsp_data_sem, 0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(dsp_done_sem, 0, 1);

static int16_t dsp_pop_buf[HALF_BUF_SAMPLES] __aligned(4);
static int16_t ble_preview_buf[HALF_BUF_SAMPLES] __aligned(4);

/* ══════════════════════════════════════════════════════════════════
 * DUAL MFCC PIPELINE INSTANCES
 * ══════════════════════════════════════════════════════════════════ */
static int16_t heart_window[60]   __aligned(4);
static int16_t lung_window[1200]  __aligned(4);

static inline int16_t sat_i16(int32_t x)
{
    if (x > 32767) return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

static void log_feature_stats(const char *tag, const int8_t *buf, uint32_t n)
{
    if (!buf || n == 0) {
        LOG_WRN("%s features empty", tag);
        return;
    }

    int8_t min_v = 127;
    int8_t max_v = -128;
    uint32_t clip_lo = 0;
    uint32_t clip_hi = 0;
    uint32_t checksum = 2166136261u;

    for (uint32_t i = 0; i < n; i++) {
        int8_t v = buf[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
        if (v == -128) clip_lo++;
        if (v == 127) clip_hi++;
        checksum ^= (uint8_t)v;
        checksum *= 16777619u;
    }

    LOG_INF("%s features: n=%u min=%d max=%d clip_lo=%u clip_hi=%u checksum=0x%08x",
            tag, n, (int)min_v, (int)max_v,
            clip_lo, clip_hi, checksum);
}

static void heart_frame_cb(int idx, const float *coeffs, void *user)
{
    (void)user; (void)idx;
    if (heart_frame_count >= 665) return;
    heart_frame_count++;
    const int n = heart_pipeline.cfg->n_mfcc;
    for (int i = 0; i < n; i++) {
        float v = (coeffs[i] - heart_mfcc_mean[i]) / (heart_mfcc_std[i] + CNN_NORM_EPS);
        int32_t q = (int32_t)lroundf(v / HEART_Q_SCALE) + HEART_Q_ZP;
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        if (heart_feat_count < HEART_FEAT_N)
            heart_features[heart_feat_count++] = (int8_t)q;
    }
}

static void lung_frame_cb(int idx, const float *coeffs, void *user)
{
    (void)user; (void)idx;
    if (lung_frame_count >= 324) return;
    lung_frame_count++;
    const int n = lung_pipeline.cfg->n_mfcc;
    for (int i = 0; i < n; i++) {
        float v = (coeffs[i] - lung_mfcc_mean[i]) / (lung_mfcc_std[i] + CNN_NORM_EPS);
        int32_t q = (int32_t)lroundf(v / LUNG_Q_SCALE) + LUNG_Q_ZP;
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        if (lung_feat_count < LUNG_FEAT_N)
            lung_features[lung_feat_count++] = (int8_t)q;
    }
}

#define DSP_THREAD_PRIORITY 7

static K_THREAD_STACK_DEFINE(ble_tx_stack,     BLE_TX_STACK_SIZE);
static K_THREAD_STACK_DEFINE(dsp_thread_stack, DSP_THREAD_STACK_SIZE);
static void ble_tx_thread_fn(void *a, void *b, void *c);
static void dsp_thread_fn(void *a, void *b, void *c);

/* ══════════════════════════════════════════════════════════════════
 * DSP THREAD
 * ══════════════════════════════════════════════════════════════════ */
static void dsp_thread_fn(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (true) {
        k_sem_take(&dsp_data_sem, K_FOREVER);

        while (analog_recording || ring_buf_size_get(&dsp_ring) > 0) {
            uint32_t avail = ring_buf_size_get(&dsp_ring);
            if (avail < HALF_BUF_BYTES) {
                if (!analog_recording) break;
                k_sleep(K_MSEC(1));
                continue;
            }
            uint32_t got = ring_buf_get(&dsp_ring,
                                        (uint8_t *)dsp_pop_buf,
                                        HALF_BUF_BYTES);
            if (got != HALF_BUF_BYTES) continue;

            dsp_mfcc_feed_chunk(&heart_pipeline, dsp_pop_buf, HALF_BUF_SAMPLES);
            dsp_mfcc_feed_chunk(&lung_pipeline,  dsp_pop_buf, HALF_BUF_SAMPLES);
        }

        k_sem_give(&dsp_done_sem);
    }
}

/* ══════════════════════════════════════════════════════════════════
 * SAADC EVENT HANDLER
 * ══════════════════════════════════════════════════════════════════ */
static void saadc_event_handler(nrfx_saadc_evt_t const *p_event)
{
    switch (p_event->type) {

    case NRFX_SAADC_EVT_BUF_REQ:
        (void)nrfx_saadc_buffer_set(ping_pong[next_dma_buf], HALF_BUF_SAMPLES);
        next_dma_buf ^= 1;
        break;

    case NRFX_SAADC_EVT_DONE: {
        int16_t *filled_buf = p_event->data.done.p_buffer;

#ifdef ENABLE_DC_REMOVAL
        for (uint32_t i = 0; i < HALF_BUF_SAMPLES; i++) {
            /* [NOISE FIX 2] DC removal time constant: >> 10 (~128 ms, ~1.2 Hz HPF).
             * Preserves low-frequency content down to 10 Hz before the BPF stage.
             * Do NOT revert to >> 9 (~64 ms / ~2.5 Hz) — that value bleeds into
             * the heart auscultation band and introduces audible AC ripple. */
            dc_estimate += ((int32_t)filled_buf[i] - dc_estimate) >> 10;
            filled_buf[i] = (int16_t)((int32_t)filled_buf[i] - dc_estimate);
        }
#endif

        /* Stream a louder monitor copy for WAV review only. The MFCC/model
         * path below still receives the un-gained DC-removed samples. */
#if BLE_AUDIO_LIVE
        int32_t preview_dc = 0;
#if BLE_PREVIEW_REMOVE_DC
        for (uint32_t i = 0; i < HALF_BUF_SAMPLES; i++) {
            preview_dc += filled_buf[i];
        }
        preview_dc /= (int32_t)HALF_BUF_SAMPLES;
#endif

        for (uint32_t i = 0; i < HALF_BUF_SAMPLES; i++) {
            int32_t preview_sample = (int32_t)filled_buf[i] - preview_dc;
            ble_preview_buf[i] =
                sat_i16(preview_sample * BLE_PREVIEW_GAIN);
        }

        uint32_t ble_written = ring_buf_put(&audio_ring,
                                            (const uint8_t *)ble_preview_buf,
                                            HALF_BUF_BYTES);
        if (ble_written != HALF_BUF_BYTES) {
            atomic_inc(&ring_drops);
            LOG_WRN_ONCE("audio_ring overflow — BLE stream will have gaps!");
        } else {
            uint32_t fill = ring_buf_size_get(&audio_ring);
            if (fill > ring_high_water) {
                ring_high_water = fill;
            }
        }
#endif

        /* The MFCC pipeline already applies the model-specific bandpass.
         * Keep this extra Q15 BPF disabled unless the models are trained
         * with the exact same outer filter in front of MFCC extraction. */
#if USE_OUTER_Q15_BPF_FOR_DSP
        arm_biquad_cascade_df1_q15(&heart_bpf,
                                   (q15_t *)filled_buf,
                                   (q15_t *)filled_buf,
                                   HALF_BUF_SAMPLES);
#endif

        /* DSP ring receives the un-gained model-path samples. */
        uint32_t dsp_written = ring_buf_put(&dsp_ring,
                                            (const uint8_t *)filled_buf,
                                            HALF_BUF_BYTES);
        if (dsp_written != HALF_BUF_BYTES) {
            LOG_WRN_ONCE("dsp_ring overflow — MFCC frames may be lost!");
        }
        k_sem_give(&dsp_data_sem);

        k_sem_give(&half_produced_sem);
#if BLE_AUDIO_LIVE
        k_sem_give(&audio_data_sem);
#endif
        break;
    }

    default:
        break;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * SAADC INIT / START / STOP
 * [FIX F] IRQ_CONNECT and irq_enable moved to saadc_init_once(),
 * called once from main(). IRQ_CONNECT expands to a linker-section
 * entry and must not be in a runtime loop.
 * ══════════════════════════════════════════════════════════════════ */
static inline void debug_flash_twice(void (*set_color)(void))
{
    led_off(&red_led); led_off(&green_led); led_off(&blue_led);
    k_busy_wait(150000);
    set_color();
    k_busy_wait(150000);
    led_off(&red_led); led_off(&green_led); led_off(&blue_led);
    k_busy_wait(150000);
    set_color();
    k_busy_wait(300000);
}

static bool saadc_was_initialized = false;

/* [FIX F] One-time IRQ registration — call from main() only. */
static void saadc_init_once(void)
{
    IRQ_CONNECT(SAADC_IRQn, SAADC_IRQ_PRIORITY, nrfx_saadc_irq_handler, NULL, 0);
    irq_enable(SAADC_IRQn);
}

static int saadc_init(void)
{
    nrfx_err_t err;

    if (saadc_was_initialized) {
        nrfx_saadc_uninit();
    }

    /* Reset Q15 band-pass filter state (zeroed on each REC).
     * [NOISE FIX 1] postShift=2 — was 1. Gives Stage-2 [1,-2,1]
     * numerator 2 bits of headroom before Q15 saturation. */
    memset(heart_bpf_state, 0, sizeof(heart_bpf_state));
    arm_biquad_cascade_df1_init_q15(&heart_bpf,
                                    HEART_BPF_STAGES,
                                    (q15_t *)heart_bpf_coeffs,
                                    heart_bpf_state,
                                    2 /* postShift — was 1, now 2 */);

    err = nrfx_saadc_init(SAADC_IRQ_PRIORITY);
    if (err != 0) { LOG_ERR("nrfx_saadc_init: 0x%08X", err); return -EIO; }

    err = nrfx_saadc_channel_config(&saadc_channel_cfg);
    if (err != 0) { LOG_ERR("channel_config: 0x%08X", err); return -EIO; }

    nrfx_saadc_adv_config_t adv_cfg  = NRFX_SAADC_DEFAULT_ADV_CONFIG;
    adv_cfg.oversampling              = NRF_SAADC_OVERSAMPLE_DISABLED;
    adv_cfg.burst                     = NRF_SAADC_BURST_DISABLED;
    adv_cfg.internal_timer_cc         = SAADC_CC_VALUE;
    adv_cfg.start_on_end              = true;

    err = nrfx_saadc_advanced_mode_set(BIT(saadc_channel_cfg.channel_index),
                                        NRF_SAADC_RESOLUTION_12BIT,
                                        &adv_cfg,
                                        saadc_event_handler);
    if (err != 0) { LOG_ERR("advanced_mode_set: 0x%08X", err); return -EIO; }

    saadc_was_initialized = true;
    return 0;
}

static int saadc_start_streaming(void)
{
    nrfx_err_t err;
    saadc_dma_overruns = 0;

    /* [NOISE FIX 3] Re-seed dc_estimate at the start of each recording.
     *
     * Problem (v8.1 and earlier): dc_estimate only seeded on first boot
     * (when it equals 0). On 2nd+ recordings with a repositioned electrode
     * or shifted battery voltage, the carried-over estimate was stale,
     * causing a ~128 ms DC convergence transient audible as a low-frequency
     * thump at the start of each recording.
     *
     * Fix: blend 50/50 between the carried-over estimate and the ADC
     * midpoint (2048). This gives fast convergence toward the true DC
     * level without the hard jump that caused the thump in v7.x.
     * First-boot (dc_estimate == 0 from BSS) seeds directly to 2048. */
    if (dc_estimate == 0) {
        dc_estimate = 2048;
    } else {
        dc_estimate = (dc_estimate + 2048) / 2;
    }

    dsp_mfcc_reset(&heart_pipeline);
    dsp_mfcc_reset(&lung_pipeline);
    memset(heart_features, 0, sizeof(heart_features));
    memset(lung_features,  0, sizeof(lung_features));
    heart_frame_count = 0;
    lung_frame_count  = 0;
    heart_feat_count  = 0;
    lung_feat_count   = 0;

    next_dma_buf = 1;

    if (saadc_init() != 0) return -EIO;

    err = nrfx_saadc_buffer_set(ping_pong[0], HALF_BUF_SAMPLES);
    if (err != 0) return -EIO;
    err = nrfx_saadc_mode_trigger();
    if (err != 0) return -EIO;

    LOG_INF("SAADC streaming started @ %d Hz", SAMPLING_RATE);
    return 0;
}

static void saadc_stop_streaming(void)
{
    if (saadc_was_initialized) {
        nrfx_saadc_uninit();
        saadc_was_initialized = false;
    }
    LOG_INF("SAADC stopped (overruns=%u, ble_drops=%u)",
            saadc_dma_overruns,
#if BLE_AUDIO_LIVE
            (uint32_t)atomic_get(&ring_drops)
#else
            0U
#endif
    );
}

/* ══════════════════════════════════════════════════════════════════
 * LED
 * ══════════════════════════════════════════════════════════════════ */
const struct gpio_dt_spec red_led   = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
const struct gpio_dt_spec green_led = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);
const struct gpio_dt_spec blue_led  = GPIO_DT_SPEC_GET(DT_ALIAS(led2), gpios);

static inline void led_on (const struct gpio_dt_spec *l) { gpio_pin_set_dt(l, 1); }
static inline void led_off(const struct gpio_dt_spec *l) { gpio_pin_set_dt(l, 0); }

static inline void led_set_white(void)  { led_on(&red_led);  led_on(&green_led);  led_on(&blue_led);  }
static inline void led_set_red(void)    { led_on(&red_led);  led_off(&green_led); led_off(&blue_led); }
static inline void led_set_green(void)  { led_off(&red_led); led_on(&green_led);  led_off(&blue_led); }
static inline void led_set_cyan(void)   { led_off(&red_led); led_on(&green_led);  led_on(&blue_led);  }
static inline void led_set_yellow(void) { led_on(&red_led);  led_on(&green_led);  led_off(&blue_led); }
static inline void led_set_purple(void) { led_on(&red_led);  led_off(&green_led); led_on(&blue_led);  }
static inline void led_set_blue(void)   { led_off(&red_led); led_off(&green_led); led_on(&blue_led);  }

static void led_error_flash(void (*color)(void))
{
    for (int i = 0; i < 3; i++) {
        color();
        k_sleep(K_MSEC(200));
        led_off(&red_led); led_off(&green_led); led_off(&blue_led);
        k_sleep(K_MSEC(200));
    }
    color();
}

/* ══════════════════════════════════════════════════════════════════
 * FLAGS
 * ══════════════════════════════════════════════════════════════════ */
static volatile bool start_recording = false;
static volatile bool is_connected    = false;
static volatile bool mtu_exchanged   = false;

/* ══════════════════════════════════════════════════════════════════
 * BLE ADVERTISING
 * ══════════════════════════════════════════════════════════════════ */
#define DEVICE_NAME      "AcoustEEEcare"
#define DEVICE_NAME_LEN  (sizeof(DEVICE_NAME) - 1)

/* [BUG 4A FIX] UUID in primary ad; name in scan response. */
static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
    BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_SRV_VAL),
};
static const struct bt_data sd_adv[] = {
    BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),
};

/* ══════════════════════════════════════════════════════════════════
 * CONNECTION CALLBACKS
 * ══════════════════════════════════════════════════════════════════ */
static struct k_work_delayable adv_restart_work;

static void mtu_exchange_cb(struct bt_conn *conn, uint8_t err,
                             struct bt_gatt_exchange_params *params)
{
    if (!err) {
        uint16_t mtu = bt_gatt_get_mtu(conn);
        nus_chunk_size = mtu - 3;
        LOG_INF("MTU=%d -> NUS chunk=%d, payload=%d",
                mtu, nus_chunk_size,
                nus_chunk_size - CHUNK_HEADER_BYTES);
    } else {
        LOG_WRN("MTU exchange failed (%d), using default=%d", err, nus_chunk_size);
    }
    mtu_exchanged = true;

    /* [STAB FIX 4] Notify the Flutter app that MTU exchange is complete and
     * it is now safe to send a REC command.  Flutter waits for this "READY"
     * message before enabling the Record button (Option A handshake).
     * Without this the Flutter Option-B 2-second fallback timer was the only
     * path to unlock the UI, and it is inherently racy on slow connections. */
    bt_nus_send(NULL, "READY\n", 6);
    LOG_INF("MTU exchange complete — sent READY to phone");
}

static struct bt_gatt_exchange_params exchange_params = { .func = mtu_exchange_cb };

static void adv_restart_work_handler(struct k_work *work)
{
    int err = bt_le_adv_start(&acousteeecare_adv_param, ad, ARRAY_SIZE(ad),
                               sd_adv, ARRAY_SIZE(sd_adv));
    if (err) LOG_ERR("adv restart failed: %d", err);
}

static void connected(struct bt_conn *conn, uint8_t err)
{
    if (err) { LOG_WRN("Connection failed (%u)", err); return; }
    is_connected  = true;
    mtu_exchanged = false;
    led_set_green();

    static const struct bt_conn_le_phy_param phy = {
        .options     = BT_CONN_LE_PHY_OPT_NONE,
        .pref_tx_phy = BT_GAP_LE_PHY_2M,
        .pref_rx_phy = BT_GAP_LE_PHY_2M,
    };
    bt_conn_le_phy_update(conn, &phy);

    /* Supervision timeout uses BLE units of 10 ms.
     * [STAB FIX 6] Raised to BLE_SUPERVISION_TIMEOUT (6000 = 60 s).
     * Android honours the MCU's requested timeout, so this also extends
     * the Android-side supervision timer to 60 s. */
    static const struct bt_le_conn_param fast_conn = {
        .interval_min = 12, .interval_max = 24, .latency = 0,
        .timeout = BLE_SUPERVISION_TIMEOUT,
    };
    bt_conn_le_param_update(conn, &fast_conn);
    bt_gatt_exchange_mtu(conn, &exchange_params);
    LOG_INF("Connected");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
    if (analog_recording) saadc_stop_streaming();
    analog_recording = false;
    is_connected     = false;
    start_recording  = false;
    mtu_exchanged    = false;
    nus_chunk_size   = 244;
    led_set_red();
    LOG_WRN("Disconnected (reason=%d)", reason);
    k_work_schedule(&adv_restart_work, K_MSEC(500));
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
    .connected    = connected,
    .disconnected = disconnected,
};

/* ══════════════════════════════════════════════════════════════════
 * NUS CALLBACKS — handles REC command and NACK retransmit requests
 * ══════════════════════════════════════════════════════════════════ */
static void received(struct bt_conn *conn, const void *data, uint16_t len, void *ctx)
{
    ARG_UNUSED(conn); ARG_UNUSED(ctx);

    if (len == 3 && memcmp(data, "REC", 3) == 0) {
        start_recording = true;
        return;
    }

#if BLE_AUDIO_LIVE
    const char *msg = (const char *)data;
    if (len > 5 && memcmp(msg, "NACK:", 5) == 0) {
        char seq_str[8] = {0};
        uint16_t copy_len = (len - 5 < sizeof(seq_str) - 1)
                            ? (len - 5) : (sizeof(seq_str) - 1);
        memcpy(seq_str, msg + 5, copy_len);
        uint16_t requested = (uint16_t)strtoul(seq_str, NULL, 10);
        k_mutex_lock(&nack_mutex, K_FOREVER);
        nack_seq_pending = requested;
        k_mutex_unlock(&nack_mutex);
        LOG_WRN("NACK received for seq=%u — will retransmit", requested);
    }
#endif
}

static struct bt_nus_cb nus_listener = { .received = received };

/* ══════════════════════════════════════════════════════════════════
 * BLE TX THREAD
 * Chunk header: [seq16][len16][crc16] = 6 bytes.
 * NACK retransmit window: last NACK_WINDOW (4) chunks kept.
 * ══════════════════════════════════════════════════════════════════ */
static void ble_tx_thread_fn(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

#if !BLE_AUDIO_LIVE
    while (true) {
        k_sleep(K_FOREVER);
    }
#else
    uint8_t  chunk[251] __aligned(4);
    bool     finished_sent = false;

    while (true) {
        k_sem_take(&audio_data_sem, K_FOREVER);

        if (analog_recording) {
            finished_sent = false;
            atomic_set(&ble_tx_saw_recording, 1);
        }

        while (true) {
            /* Check for pending NACK — search window for matching seq */
            k_mutex_lock(&nack_mutex, K_FOREVER);
            uint16_t pending_nack = nack_seq_pending;
            if (pending_nack != 0xFFFF) {
                nack_seq_pending = 0xFFFF;
            }
            k_mutex_unlock(&nack_mutex);

            if (pending_nack != 0xFFFF) {
                bool retransmitted = false;
                for (int i = 0; i < NACK_WINDOW; i++) {
                    if (nack_buf_len[i] > 0 && nack_buf_seq[i] == pending_nack) {
                        LOG_INF("Retransmitting seq=%u (%u B) from window[%d]",
                                pending_nack, nack_buf_len[i], i);
                        int err;
                        int retry_ms = 2;
                        do {
                            err = bt_nus_send(NULL, nack_buf[i], nack_buf_len[i]);
                            if (err == -ENOMEM || err == -EAGAIN) {
                                k_sleep(K_MSEC(retry_ms));
                                if (retry_ms < 20) retry_ms *= 2;
                            }
                        } while (err == -ENOMEM || err == -EAGAIN);
                        retransmitted = true;
                        break;
                    }
                }
                if (!retransmitted) {
                    LOG_WRN("NACK seq=%u not in retransmit window — cannot retransmit",
                            pending_nack);
                }
                continue;
            }

            uint16_t payload_bytes = nus_chunk_size - CHUNK_HEADER_BYTES;
            if (payload_bytes < 2) { k_sleep(K_MSEC(5)); break; }

            uint32_t available = ring_buf_size_get(&audio_ring);

            if (available == 0) {
                if (!analog_recording && !finished_sent &&
                    atomic_get(&ble_tx_saw_recording)) {
                    int err;
                    int retry_ms = 2;
                    do {
                        err = bt_nus_send(NULL, "finished\n", 9);
                        if (err == -ENOMEM || err == -EAGAIN) {
                            k_sleep(K_MSEC(retry_ms));
                            if (retry_ms < 20) retry_ms *= 2;
                        }
                    } while (err == -ENOMEM || err == -EAGAIN);

                    finished_sent = true;
                    LOG_INF("BLE audio stream done (seq=%u)", tx_seq);
                    k_sem_give(&tx_done_sem);
                }
                break;
            }

            bool partial = (available < payload_bytes);
            if (partial && analog_recording) break;

            uint16_t send_bytes = partial ? (uint16_t)available : payload_bytes;

            /* Build chunk: [seq16][len16][crc16][payload] */
            ring_buf_get(&audio_ring, &chunk[CHUNK_HEADER_BYTES], send_bytes);
            uint16_t crc = crc16_ccitt(0xFFFF, &chunk[CHUNK_HEADER_BYTES], send_bytes);
            sys_put_le16(tx_seq,     &chunk[0]);
            sys_put_le16(send_bytes, &chunk[2]);
            sys_put_le16(crc,        &chunk[4]);

            uint16_t total_len = CHUNK_HEADER_BYTES + send_bytes;

            /* Store chunk in circular NACK window */
            int slot = tx_seq % NACK_WINDOW;
            if (total_len <= sizeof(nack_buf[slot])) {
                memcpy(nack_buf[slot], chunk, total_len);
                nack_buf_seq[slot] = tx_seq;
                nack_buf_len[slot] = total_len;
            }

            int err;
            int retry_ms = 2;
            do {
                err = bt_nus_send(NULL, chunk, total_len);
                if (err == -ENOMEM || err == -EAGAIN) {
                    k_sleep(K_MSEC(retry_ms));
                    if (retry_ms < 20) retry_ms *= 2;
                }
            } while (err == -ENOMEM || err == -EAGAIN);

            tx_seq = (tx_seq + 1) & 0xFFFF;
        }
    }
#endif /* BLE_AUDIO_LIVE */
}

/* ══════════════════════════════════════════════════════════════════
 * BLE RESULT SEND HELPER
 * ══════════════════════════════════════════════════════════════════ */
static void nus_send_blocking(const char *buf, uint16_t len)
{
    int err;
    int retry_ms = 2;
    do {
        err = bt_nus_send(NULL, buf, len);
        if (err == -ENOMEM || err == -EAGAIN) {
            k_sleep(K_MSEC(retry_ms));
            if (retry_ms < 20) retry_ms *= 2;
        }
    } while (err == -ENOMEM || err == -EAGAIN);
}

/* ══════════════════════════════════════════════════════════════════
 * [STAB FIX 5] IN-CYCLE KEEPALIVE HELPER
 *
 * Called from inside record_and_stream() at every natural pause point.
 * Sends PING\n if PING_INTERVAL_MS has elapsed since the last ping.
 * This is the ONLY correct place for the keepalive during a recording
 * cycle — the main loop cannot fire while record_and_stream() is
 * blocking it.
 * ══════════════════════════════════════════════════════════════════ */
static void keepalive_ping_if_due(int64_t *last_ping_ms)
{
    if (!is_connected) return;
    int64_t now = k_uptime_get();
    if ((now - *last_ping_ms) >= PING_INTERVAL_MS) {
        bt_nus_send(NULL, "PING\n", 5);
        *last_ping_ms = now;
        LOG_DBG("PING sent (in-cycle keepalive)");
    }
}

/* ══════════════════════════════════════════════════════════════════
 * RECORD AND STREAM
 * ══════════════════════════════════════════════════════════════════ */

/* Atomic flag — true while record_and_stream() is executing.
 * Prevents re-entry from a rapid REC command during TX drain or
 * inference (when analog_recording is false but session is live). */
static atomic_t in_session = ATOMIC_INIT(0);

static void record_and_stream(void)
{
    atomic_set(&in_session, 1);
    led_set_purple();

    /* [STAB FIX 5] Keepalive timestamp — passed to keepalive_ping_if_due()
     * at each natural pause point inside this blocking function. */
    int64_t ping_ms = k_uptime_get();

    const uint32_t total_halves =
        (TOTAL_AUDIO_SAMPLES + HALF_BUF_SAMPLES - 1) / HALF_BUF_SAMPLES;

    k_sem_reset(&half_produced_sem);

    /* [FIX A] drops declared unconditionally — always in scope for LOG_INF */
    uint32_t drops = 0;

#if BLE_AUDIO_LIVE
    if (atomic_get(&ble_tx_saw_recording)) {
        if (k_sem_take(&tx_done_sem, K_MSEC(PRE_REC_DRAIN_TIMEOUT_MS)) != 0) {
            LOG_WRN("Pre-REC drain wait timed out — resetting anyway");
        }
    }

    k_sem_reset(&audio_data_sem);
    ring_buf_reset(&audio_ring);
    atomic_set(&ring_drops, 0);
    ring_high_water = 0;
    tx_seq          = 0;
    atomic_set(&ble_tx_saw_recording, 0);
    k_sem_reset(&tx_done_sem);

    /* [STAB FIX 7] Clear both len AND seq arrays so stale seq numbers from
     * previous recordings cannot cause a late NACK to match the wrong slot
     * and retransmit incorrect data. */
    memset(nack_buf_len, 0, sizeof(nack_buf_len));
    memset(nack_buf_seq, 0, sizeof(nack_buf_seq));

    k_mutex_lock(&nack_mutex, K_FOREVER);
    nack_seq_pending = 0xFFFF;
    k_mutex_unlock(&nack_mutex);
#endif

    ring_buf_reset(&dsp_ring);
    k_sem_reset(&dsp_done_sem);

    led_set_cyan();

#if BLE_AUDIO_LIVE
    {
        char hdr[32];
        int hlen = snprintf(hdr, sizeof(hdr), "START:%u\n",
                            (uint32_t)TOTAL_AUDIO_BYTES);
        bt_nus_send(NULL, hdr, hlen);
        k_sleep(K_MSEC(START_HEADER_DELAY_MS));
    }
#endif

    analog_recording = true;
    k_sem_give(&dsp_data_sem);

    /* ════════════════════════════════════════════════════════════
     * PHASE 1: SAADC capture
     * ════════════════════════════════════════════════════════════ */
    LOG_INF("Phase 1: SAADC capture — %d s @ %d Hz", DURATION_S, SAMPLING_RATE);

    if (saadc_start_streaming() != 0) {
        analog_recording = false;
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:SAADC", 9);
        atomic_set(&in_session, 0);
        return;
    }

    led_set_white();

    for (uint32_t h = 0; h < total_halves; h++) {
        if (k_sem_take(&half_produced_sem, K_MSEC(500)) != 0) {
            LOG_ERR("SAADC timeout at half-buffer %u/%u", h, total_halves);
            analog_recording = false;
            saadc_stop_streaming();
            bt_nus_send(NULL, "ERR:TIMEOUT", 11);
            atomic_set(&in_session, 0);
            return;
        }
    }

    saadc_stop_streaming();
    analog_recording = false;

    led_set_green();

    /* [STAB FIX 5] Ping after SAADC capture completes.
     * The 10 s capture window had audio chunks keeping the link alive,
     * but we reset the ping timer here so the next ping fires on schedule
     * from the start of the inference window. */
    keepalive_ping_if_due(&ping_ms);

    /* Wait for DSP thread to finish MFCC extraction.
     * [STAB FIX 5] Poll in 2 s slices and send a keepalive each slice
     * so the BLE supervision timer never sees more than 3 s of silence
     * during DSP completion (which can take up to ~30 s in theory). */
    {
        bool dsp_done = false;
        int64_t dsp_deadline = k_uptime_get() + 30000;
        while (k_uptime_get() < dsp_deadline) {
            if (k_sem_take(&dsp_done_sem, K_MSEC(2000)) == 0) {
                dsp_done = true;
                break;
            }
            /* Slice elapsed — send keepalive and keep waiting */
            keepalive_ping_if_due(&ping_ms);
        }
        if (!dsp_done) {
            LOG_WRN("DSP thread did not finish within 30 s");
        }
    }

#if BLE_AUDIO_LIVE
    k_sem_give(&audio_data_sem);   /* drain ring tail + send finished */
    if (k_sem_take(&tx_done_sem, K_MSEC(TX_DONE_TIMEOUT_MS)) != 0) {
        LOG_WRN("tx_done_sem timeout — BLE drain did not complete within %d ms",
                TX_DONE_TIMEOUT_MS);
    }

    /* Drop warning — sent after TX drain so host has audio first */
    drops = (uint32_t)atomic_get(&ring_drops);
    if (drops > 0) {
        char warn[32];
        int  wlen = snprintf(warn, sizeof(warn), "WARN:DROPS:%u\n", drops);
        nus_send_blocking(warn, (uint16_t)wlen);
        LOG_WRN("Audio had %u ring-buffer drop(s) — recording may be degraded",
                drops);
    }
    LOG_INF("BLE ring high-water: %u / %u B", ring_high_water, AUDIO_RING_BYTES);
#endif

    /* [STAB FIX 5] Ping before entering the long inference window. */
    keepalive_ping_if_due(&ping_ms);

    /* ═════════════════════════════════════════════════════════════
     * PHASE 2: On-device inference from RAM feature buffers
     * ═════════════════════════════════════════════════════════════ */
    int hf = (int)heart_frame_count, lf = (int)lung_frame_count;
    LOG_INF("Captured frames: heart=%d lung=%d", hf, lf);

    if (hf < 660 || hf > 670 || lf < 320 || lf > 330) {
        LOG_ERR("Frame count off (heart=%d/665 lung=%d/324)", hf, lf);
        bt_nus_send(NULL, "ERR:FRAMES\n", 11);
        led_error_flash(led_set_yellow);
        led_set_green();
        atomic_set(&in_session, 0);
        return;
    }

    LOG_INF("Feature counts: heart=%u/%u lung=%u/%u",
            (unsigned)heart_feat_count, (unsigned)HEART_FEAT_N,
            (unsigned)lung_feat_count,  (unsigned)LUNG_FEAT_N);
    log_feature_stats("heart", heart_features, heart_feat_count);
    log_feature_stats("lung",  lung_features,  lung_feat_count);

    heart_result_t hr;
    lung_result_t  rr;

    nus_send_blocking("INF:START\n", 10);

    led_set_blue();

    /* Heart inference — can take 15-25 s on the nRF SoC.
     * [STAB FIX 5] Ping immediately before so Android's timer resets
     * at the worst possible moment (just before the long silent window). */
    keepalive_ping_if_due(&ping_ms);
    run_heart_inference_ram(tensor_arena, TENSOR_ARENA_BYTES,
                            heart_features, 665, 25, &hr);
    nus_send_blocking("INF:HEART_DONE\n", 15);

    /* [STAB FIX 5] Ping between the two inferences — this is the most
     * critical call site. Heart inference can take ~20 s; without this
     * ping Android will have received nothing for that entire window and
     * will have already declared the link dead before lung inference starts. */
    keepalive_ping_if_due(&ping_ms);

    k_yield();   /* [BUG 2C FIX] Let BLE stack breathe between inferences */

    led_set_purple();
    run_lung_inference_ram(tensor_arena, TENSOR_ARENA_BYTES,
                           lung_features, 324, 26, &rr);
    nus_send_blocking("INF:LUNG_DONE\n", 14);

    /* [STAB FIX 5] Ping after lung inference completes before sending results. */
    keepalive_ping_if_due(&ping_ms);

    char msg[48]; int mlen;
    if (hr.rc == 0) {
        mlen = snprintf(msg, sizeof(msg), "HR:%.0f\n", (double)hr.value);
        nus_send_blocking(msg, mlen);
    } else {
        nus_send_blocking("ERR:HEART_INF\n", 14);
    }
    k_sleep(K_MSEC(20));
    if (rr.rc == 0) {
        mlen = snprintf(msg, sizeof(msg), "RR:%.0f\n", (double)rr.value);
        nus_send_blocking(msg, mlen);
    } else {
        nus_send_blocking("ERR:LUNG_INF\n", 13);
    }

    LOG_INF("Done. HR=%.0f RR=%.0f drops=%u",
            (double)hr.value, (double)rr.value, drops);
    led_set_green();
    atomic_set(&in_session, 0);
}

/* ══════════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════════ */
int main(void)
{
    int err;

    gpio_pin_configure_dt(&red_led,   GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&green_led, GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&blue_led,  GPIO_OUTPUT_ACTIVE);
    led_set_white();

    k_work_init_delayable(&adv_restart_work, adv_restart_work_handler);

    /* [FIX F] One-time SAADC IRQ registration */
    saadc_init_once();

    /* ── MFCC pipeline init ──────────────────────────────────────── */
    if (dsp_mfcc_init(&heart_pipeline, &heart_mfcc_config) != 0) {
        LOG_ERR("heart dsp_mfcc_init failed");
        led_error_flash(led_set_yellow);
        return -1;
    }
    heart_pipeline.window = heart_window;
    dsp_mfcc_set_callback(&heart_pipeline, heart_frame_cb, NULL);

    if (dsp_mfcc_init(&lung_pipeline, &lung_mfcc_config) != 0) {
        LOG_ERR("lung dsp_mfcc_init failed");
        led_error_flash(led_set_yellow);
        return -1;
    }
    lung_pipeline.window = lung_window;
    dsp_mfcc_set_callback(&lung_pipeline, lung_frame_cb, NULL);

    /* ── BLE init ────────────────────────────────────────────────── */
    err = bt_enable(NULL);
    if (err) { LOG_ERR("bt_enable: %d", err); return err; }

    err = bt_nus_cb_register(&nus_listener, NULL);
    if (err) { LOG_ERR("bt_nus_cb_register: %d", err); return err; }

    err = bt_le_adv_start(&acousteeecare_adv_param, ad, ARRAY_SIZE(ad),
                          sd_adv, ARRAY_SIZE(sd_adv));
    if (err) { LOG_ERR("bt_le_adv_start: %d", err); return err; }

    /* ── Thread creation ─────────────────────────────────────────── */
    k_thread_create(&ble_tx_thread_data, ble_tx_stack,
                    K_THREAD_STACK_SIZEOF(ble_tx_stack),
                    ble_tx_thread_fn, NULL, NULL, NULL,
                    BLE_TX_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&ble_tx_thread_data, "ble_tx");

    k_thread_create(&dsp_thread_data, dsp_thread_stack,
                    K_THREAD_STACK_SIZEOF(dsp_thread_stack),
                    dsp_thread_fn, NULL, NULL, NULL,
                    DSP_THREAD_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&dsp_thread_data, "dsp");

    led_set_red();
    LOG_INF("AcoustEEEcare v8.4 ready — waiting for BLE connection");

    report_ram_usage("boot (idle)");

    /* [BUG 3 FIX] Track last ping time for idle keepalive.
     * Note: the in-cycle keepalive (keepalive_ping_if_due inside
     * record_and_stream) manages its own timestamp. This variable
     * covers idle periods between recordings only. */
    int64_t last_ping_ms = 0;

    while (true) {
        k_sleep(K_MSEC(100));
        if (is_connected && mtu_exchanged && start_recording &&
            !atomic_get(&in_session)) {
            start_recording = false;
            LOG_INF("REC command — starting %d s @ %d Hz",
                    DURATION_S, SAMPLING_RATE);
            record_and_stream();
            report_ram_usage("after REC completed");
            last_ping_ms = k_uptime_get();
        }

        /* Idle keepalive: send PING every 3 s when connected and between
         * recordings. During a recording cycle, keepalive_ping_if_due()
         * inside record_and_stream() handles keepalives directly — this
         * code is unreachable while record_and_stream() blocks. */
        if (is_connected && !analog_recording && !atomic_get(&in_session)) {
            int64_t now = k_uptime_get();
            if ((now - last_ping_ms) > PING_INTERVAL_MS) {
                bt_nus_send(NULL, "PING\n", 5);
                last_ping_ms = now;
            }
        }
    }

    return 0;
}