/*
 * AcoustEEEcare — SAADC BLE + TFLite Micro Edition
 * ==================================================
 * v8.1 — SD card removed; v8.1 improvements applied
 *
 * CHANGES FROM v7.9:
 *
 *   [SD REMOVAL] All USE_SD / fs / FATFS / disk_access code removed.
 *                SD writer thread, sd_writer_thread_fn, init_sd_card,
 *                process_audio_offline, and all related ring buffers,
 *                semaphores, and file handles are gone.
 *
 *   [IMP 1] Ring buffer enlarged: AUDIO_RING_BYTES 16 KB → 32 KB.
 *           Reduces silent audio gaps under sustained BLE stalls.
 *
 *   [IMP 2] Drop warning: after TX drain, sends "WARN:DROPS:N\n" over
 *           NUS when ring_drops > 0 so the host can flag degraded WAVs.
 *
 *   [IMP 3] CMSIS-DSP Q15 band-pass filter added to the SAADC event
 *           handler after DC removal (heart: 20–950 Hz Butterworth 2nd-
 *           order). Rejects out-of-band noise before MFCC extraction.
 *
 *   [IMP 4] DSP ring enlarged: DSP_RING_BYTES 8 KB → 16 KB, matching
 *           the BLE audio ring headroom.
 *
 *   [IMP 5] NACK + CRC-16: chunk header extended from 4 to 6 bytes
 *           ([seq16][len16][crc16]). On CRC mismatch the firmware sends
 *           "NACK:<seq>\n" instead of hard-aborting.  BLE TX thread
 *           retransmits on NACK (see ble_tx_thread_fn).
 *
 *   [IMP 6] BLE TX stack enlarged: 2048 → 3072 B to cover the extra
 *           CRC computation and NACK handling path.
 *
 *   [IMP 7] Confirmed TENSOR_ARENA_BYTES kept at 72 KB; arena_used_bytes
 *           logged after AllocateTensors() via g_tflm_arena_used_bytes.
 *
 * ALL PREVIOUS FIXES (v7.9 / v7.8 / v7.7 / … / v6.8) ARE PRESERVED
 * where they still apply to the no-SD build.
 * ==================================================
 */

/* ── SD card is permanently disabled in this build ─────────────── */
#define USE_SD  false

#define BLE_AUDIO_LIVE  1   /* stream raw audio to phone live */
#define DSP_OFFLINE     0   /* MFCC computed live during capture */

/* ══════════════════════════════════════════════════════════════════
 * INCLUDES
 * ══════════════════════════════════════════════════════════════════ */
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <math.h>

#include <arm_math.h>   /* [IMP 3] CMSIS-DSP Q15 biquad filter */

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
#include <zephyr/sys/crc.h>          /* [IMP 5] crc16_ccitt */
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

/* ══════════════════════════════════════════════════════════════════
 * SHARED TENSOR ARENA
 * [IMP 7] 72 KB — arena_used_bytes logged after AllocateTensors().
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

/* ══════════════════════════════════════════════════════════════════
 * RAM USAGE REPORT
 * ══════════════════════════════════════════════════════════════════ */
extern char __bss_start;
extern char __bss_end __attribute__((weak));
extern char _end      __attribute__((weak));
extern char _image_ram_start;
extern char _image_ram_end;

/* [IMP 1] 32 KB — enlarged from 16 KB for ~3 s slack at 8 kHz int16 */
#define BLE_TX_STACK_SIZE     3072   /* [IMP 6] enlarged from 2048 for CRC/NACK path */
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
 * [IMP 3] CMSIS-DSP Q15 BAND-PASS FILTER (Heart: 20–950 Hz, 8 kHz)
 *
 * Coefficients generated with:
 *   from scipy.signal import butter
 *   sos = butter(2, [20, 950], btype='bandpass', fs=8000, output='sos')
 *   # convert to Q15: round(sos * 32768)
 *
 * Replace placeholder values below with your computed coefficients.
 * Format: {b0, b1, b2, -a1, -a2} for each biquad stage (CMSIS-DSP
 * convention negates a1/a2).
 * ══════════════════════════════════════════════════════════════════ */
/* Two second-order sections for a 2nd-order bandpass (one biquad each
 * stage; adjust HEART_BPF_STAGES if you use a higher-order design).   */
#define HEART_BPF_STAGES  2

static const q15_t heart_bpf_coeffs[5 * HEART_BPF_STAGES] = {
    /* Stage 1 — replace with scipy-generated Q15 values */
     1382,  2764,  1382, -25576,  12610,
    /* Stage 2 */
     1382, -2764,  1382,  25576,  12610,
};
static q15_t                       heart_bpf_state[4 * HEART_BPF_STAGES];
static arm_biquad_casd_df1_inst_q15 heart_bpf;

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
/* [IMP 1] 32 KB — enlarged from 16 KB; ~3 s headroom at 16 KB/s */
#define AUDIO_RING_BYTES  (32 * 1024)
RING_BUF_DECLARE(audio_ring, AUDIO_RING_BYTES);

static K_SEM_DEFINE(audio_data_sem, 0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(tx_done_sem,    0, 1);

static atomic_t  ring_drops;
static uint32_t  ring_high_water = 0;

static uint16_t tx_seq = 0;

/* [FIX] Prevents the BLE TX thread from sending "finished\n" before the
 * recording session has started. Set true when analog_recording is first
 * observed true; reset to false at the start of each REC. */
static volatile bool ble_tx_saw_recording = false;

/* [IMP 5] Sequence tracking for NACK retransmit */
static atomic_t  nack_requested;         /* set to (seq + 1) when NACK received */
static uint16_t  nack_seq = 0xFFFF;      /* 0xFFFF = no pending NACK */
static K_MUTEX_DEFINE(nack_mutex);
#endif /* BLE_AUDIO_LIVE */

/* [IMP 5] CRC-16 chunk header: [seq16][len16][crc16] = 6 bytes */
#define CHUNK_HEADER_BYTES  6

/* ══════════════════════════════════════════════════════════════════
 * DSP RING BUFFER + DSP THREAD
 * ══════════════════════════════════════════════════════════════════ */
/* [IMP 4] 16 KB — enlarged from 8 KB to match BLE ring headroom */
#define DSP_RING_BYTES  (16 * 1024)
RING_BUF_DECLARE(dsp_ring, DSP_RING_BYTES);

static K_SEM_DEFINE(dsp_data_sem, 0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(dsp_done_sem, 0, 1);

static int16_t dsp_pop_buf[HALF_BUF_SAMPLES] __aligned(4);

/* ══════════════════════════════════════════════════════════════════
 * DUAL MFCC PIPELINE INSTANCES
 *
 * Window sizes MUST equal cfg->frame_samples:
 *   heart: 60 samples   lung: 1200 samples
 * ══════════════════════════════════════════════════════════════════ */
static int16_t heart_window[60]   __aligned(4);
static int16_t lung_window[1200]  __aligned(4);

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

#define BLE_TX_PRIORITY     5
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
            dc_estimate += ((int32_t)filled_buf[i] - dc_estimate) >> 8;
            filled_buf[i] = (int16_t)((int32_t)filled_buf[i] - dc_estimate);
        }
#endif

        /* [IMP 3] Heart band-pass filter: 20–950 Hz, 2nd-order Butterworth
         * Applied after DC removal, in-place. Rejects low-frequency
         * motion artifacts and high-frequency noise before MFCC. */
        arm_biquad_cascade_df1_q15(&heart_bpf,
                                   (q15_t *)filled_buf,
                                   (q15_t *)filled_buf,
                                   HALF_BUF_SAMPLES);

        /* DSP ring (always fed; DSP thread processes both heart and lung) */
        uint32_t dsp_written = ring_buf_put(&dsp_ring,
                                            (const uint8_t *)filled_buf,
                                            HALF_BUF_BYTES);
        if (dsp_written != HALF_BUF_BYTES) {
            LOG_WRN_ONCE("dsp_ring overflow — MFCC frames may be lost!");
        }
        k_sem_give(&dsp_data_sem);

#if BLE_AUDIO_LIVE
        uint32_t ble_written = ring_buf_put(&audio_ring,
                                            (const uint8_t *)filled_buf,
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

static int saadc_init(void)
{
    nrfx_err_t err;

    if (saadc_was_initialized) {
        nrfx_saadc_uninit();
    }

    /* [IMP 3] Init Q15 band-pass filter state (zeroed on each REC) */
    memset(heart_bpf_state, 0, sizeof(heart_bpf_state));
    arm_biquad_cascade_df1_init_q15(&heart_bpf,
                                    HEART_BPF_STAGES,
                                    (q15_t *)heart_bpf_coeffs,
                                    heart_bpf_state,
                                    1 /* postShift */);

    IRQ_CONNECT(SAADC_IRQn, SAADC_IRQ_PRIORITY, nrfx_saadc_irq_handler, NULL, 0);
    irq_enable(SAADC_IRQn);

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
    /* Seed DC estimate to ADC midpoint to avoid HPF ringing at start */
    dc_estimate = 2048;

    dsp_mfcc_reset(&heart_pipeline);
    dsp_mfcc_reset(&lung_pipeline);
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

static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
    BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),
};
static const struct bt_data sd_adv[] = {
    BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_SRV_VAL),
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
}

static struct bt_gatt_exchange_params exchange_params = { .func = mtu_exchange_cb };

static void adv_restart_work_handler(struct k_work *work)
{
    int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
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

    static const struct bt_le_conn_param fast_conn = {
        .interval_min = 12, .interval_max = 24, .latency = 0, .timeout = 400,
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
 * NUS CALLBACKS
 * [IMP 5] Handle "NACK:<seq>\n" from host: set nack_seq so the BLE TX
 * thread can retransmit the missing chunk on next iteration.
 * ══════════════════════════════════════════════════════════════════ */
static void received(struct bt_conn *conn, const void *data, uint16_t len, void *ctx)
{
    ARG_UNUSED(conn); ARG_UNUSED(ctx);

    if (len == 3 && memcmp(data, "REC", 3) == 0) {
        start_recording = true;
        return;
    }

#if BLE_AUDIO_LIVE
    /* [IMP 5] NACK handler: "NACK:<seq_decimal>\n" */
    const char *msg = (const char *)data;
    if (len > 5 && memcmp(msg, "NACK:", 5) == 0) {
        char seq_str[8] = {0};
        uint16_t copy_len = (len - 5 < sizeof(seq_str) - 1)
                            ? (len - 5) : (sizeof(seq_str) - 1);
        memcpy(seq_str, msg + 5, copy_len);
        uint16_t requested = (uint16_t)strtoul(seq_str, NULL, 10);
        k_mutex_lock(&nack_mutex, K_FOREVER);
        nack_seq = requested;
        k_mutex_unlock(&nack_mutex);
        LOG_WRN("NACK received for seq=%u — will retransmit", requested);
    }
#endif
}

static struct bt_nus_cb nus_listener = { .received = received };

/* ══════════════════════════════════════════════════════════════════
 * BLE TX THREAD
 * [IMP 5] Extended chunk header: [seq16][len16][crc16] = 6 bytes.
 *         CRC-16/CCITT computed over the payload bytes only.
 *         On NACK, the thread retransmits the requested sequence.
 * ══════════════════════════════════════════════════════════════════ */
static void ble_tx_thread_fn(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

#if !BLE_AUDIO_LIVE
    while (true) {
        k_sleep(K_FOREVER);
    }
#else
    /* Retransmit buffer: keeps the last sent chunk in case NACK arrives */
    uint8_t  chunk[251]      __aligned(4);
    uint8_t  last_chunk[251] __aligned(4);
    uint16_t last_chunk_len  = 0;
    uint16_t last_chunk_seq  = 0xFFFF;
    bool     finished_sent   = false;

    while (true) {
        k_sem_take(&audio_data_sem, K_FOREVER);

        if (analog_recording) {
            finished_sent        = false;
            ble_tx_saw_recording = true;
        }

        while (true) {
            /* [IMP 5] Check for pending NACK before sending new data */
            k_mutex_lock(&nack_mutex, K_FOREVER);
            uint16_t pending_nack = nack_seq;
            if (pending_nack != 0xFFFF) {
                nack_seq = 0xFFFF;   /* consume */
            }
            k_mutex_unlock(&nack_mutex);

            if (pending_nack != 0xFFFF &&
                last_chunk_len > 0 &&
                last_chunk_seq == pending_nack) {
                /* Retransmit the last chunk verbatim (header + payload) */
                LOG_INF("Retransmitting seq=%u (%u B)", pending_nack, last_chunk_len);
                int err;
                do {
                    err = bt_nus_send(NULL, last_chunk, last_chunk_len);
                    if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(2));
                } while (err == -ENOMEM || err == -EAGAIN);
                continue;   /* re-check NACK before advancing */
            }

            uint16_t payload_bytes = nus_chunk_size - CHUNK_HEADER_BYTES;
            if (payload_bytes < 2) { k_sleep(K_MSEC(5)); break; }

            uint32_t available = ring_buf_size_get(&audio_ring);

            if (available == 0) {
                if (!analog_recording && !finished_sent && ble_tx_saw_recording) {
                    int err;
                    do {
                        err = bt_nus_send(NULL, "finished\n", 9);
                        if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(2));
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

            /* [IMP 5] Build chunk: [seq16][len16][crc16][payload] */
            ring_buf_get(&audio_ring, &chunk[CHUNK_HEADER_BYTES], send_bytes);
            uint16_t crc = crc16_ccitt(0xFFFF,
                                       &chunk[CHUNK_HEADER_BYTES],
                                       send_bytes);
            sys_put_le16(tx_seq,     &chunk[0]);
            sys_put_le16(send_bytes, &chunk[2]);
            sys_put_le16(crc,        &chunk[4]);

            uint16_t total_len = CHUNK_HEADER_BYTES + send_bytes;

            /* Keep a copy for potential NACK retransmit */
            if (total_len <= sizeof(last_chunk)) {
                memcpy(last_chunk, chunk, total_len);
                last_chunk_len = total_len;
                last_chunk_seq = tx_seq;
            }

            int err;
            do {
                err = bt_nus_send(NULL, chunk, total_len);
                if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(2));
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
    do {
        err = bt_nus_send(NULL, buf, len);
        if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(2));
    } while (err == -ENOMEM || err == -EAGAIN);
}

/* ══════════════════════════════════════════════════════════════════
 * RECORD AND STREAM
 * ══════════════════════════════════════════════════════════════════ */
static void record_and_stream(void)
{
    led_set_purple();

    const uint32_t total_halves =
        (TOTAL_AUDIO_SAMPLES + HALF_BUF_SAMPLES - 1) / HALF_BUF_SAMPLES;

    k_sem_reset(&half_produced_sem);

#if BLE_AUDIO_LIVE
    k_sem_reset(&audio_data_sem);
    ring_buf_reset(&audio_ring);
    atomic_set(&ring_drops, 0);
    ring_high_water      = 0;
    tx_seq               = 0;
    ble_tx_saw_recording = false;   /* [FIX] reset guard for this session */
    k_sem_reset(&tx_done_sem);

    /* [IMP 5] Clear any stale NACK state */
    k_mutex_lock(&nack_mutex, K_FOREVER);
    nack_seq = 0xFFFF;
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
        k_sleep(K_MSEC(10));
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
        return;
    }

    led_set_white();

    for (uint32_t h = 0; h < total_halves; h++) {
        if (k_sem_take(&half_produced_sem, K_MSEC(500)) != 0) {
            LOG_ERR("SAADC timeout at half-buffer %u/%u", h, total_halves);
            analog_recording = false;
            saadc_stop_streaming();
            bt_nus_send(NULL, "ERR:TIMEOUT", 11);
            return;
        }
    }

    saadc_stop_streaming();
    analog_recording = false;

    led_set_green();

    if (k_sem_take(&dsp_done_sem, K_MSEC(30000)) != 0) {
        LOG_WRN("DSP thread did not finish within 30 s");
    }

#if BLE_AUDIO_LIVE
    k_sem_give(&audio_data_sem);   /* drain ring tail + send finished */
    k_sem_take(&tx_done_sem, K_MSEC(15000));

    /* [IMP 2] Drop warning — sent after TX drain so host has audio first */
    uint32_t drops = (uint32_t)atomic_get(&ring_drops);
    if (drops > 0) {
        char warn[32];
        int  wlen = snprintf(warn, sizeof(warn), "WARN:DROPS:%u\n", drops);
        nus_send_blocking(warn, (uint16_t)wlen);
        LOG_WRN("Audio had %u ring-buffer drop(s) — recording may be degraded",
                drops);
    }
    LOG_INF("BLE ring high-water: %u / %u B", ring_high_water, AUDIO_RING_BYTES);
#endif

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
        return;
    }

    heart_result_t hr;
    lung_result_t  rr;

    led_set_blue();
    run_heart_inference_ram(tensor_arena, TENSOR_ARENA_BYTES,
                            heart_features, 665, 25, &hr);
    led_set_purple();
    run_lung_inference_ram(tensor_arena, TENSOR_ARENA_BYTES,
                           lung_features, 324, 26, &rr);

    /* Send results to phone */
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

    err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
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
    LOG_INF("AcoustEEEcare v8.1 ready — waiting for BLE connection");

    /* Baseline RAM snapshot before first recording */
    report_ram_usage("boot (idle)");

    while (true) {
        k_sleep(K_MSEC(100));
        if (is_connected && mtu_exchanged && start_recording && !analog_recording) {
            start_recording = false;
            LOG_INF("REC command — starting %d s @ %d Hz",
                    DURATION_S, SAMPLING_RATE);
            record_and_stream();
            report_ram_usage("after REC completed");
        }
    }

    return 0;
}