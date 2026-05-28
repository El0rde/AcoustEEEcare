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
 * PATCH FIXES (applied over v8.1):
 *
 *   [FIX A] drops variable scope: moved outside #if BLE_AUDIO_LIVE block
 *           so LOG_INF at end of record_and_stream() always compiles.
 *
 *   [FIX B] NACK retransmit window expanded from 1 to NACK_WINDOW (4)
 *           slots. Firmware can now retransmit any of the last 4 chunks,
 *           covering cases where NACK arrives after subsequent sends.
 *
 *   [FIX C] DC removal time constant changed from >> 8 (~32 ms) to
 *           >> 10 (~128 ms) for a gentler HPF that preserves low-frequency
 *           content before BPF.
 *
 *   [FIX D] BLE TX thread priority lowered from 5 to 4 so it preempts
 *           the main thread during inference and keeps the audio ring
 *           draining.
 *
 *   [FIX E] ble_tx_saw_recording converted from volatile bool to atomic_t
 *           to ensure correct cross-thread visibility without races.
 *
 *   [FIX F] saadc_init() IRQ_CONNECT / irq_enable moved to a one-time
 *           saadc_init_once() called from main(). IRQ_CONNECT is a
 *           compile-time linker macro and must not be called repeatedly.
 *
 *   [R2+R5] Capture-path Q15 BPF (heart_bpf_coeffs) removed; band
 *           selection now done per-pipeline in dsp_mfcc.c.
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
#include <stdlib.h>

/* arm_math.h removed: Q15 biquad filter no longer used (R2+R5 fix). */

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
static void saadc_init_once(void);   /* [FIX F] one-time IRQ setup */

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
/* [FIX D] BLE_TX_PRIORITY lowered to 4 (was 5 = same as main).
 *         This ensures BLE TX preempts the main thread during
 *         inference so the audio ring keeps draining. */
#define BLE_TX_PRIORITY       4
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
        /* [FIX 1] Increased acquisition time from 10 µs to 40 µs.
         * The bias source impedance is 10k||10k = 5 kΩ. Nordic's datasheet
         * requires ~40 µs acquisition for a 5 kΩ source to let the sampling
         * capacitor fully settle. At 10 µs the cap was starved, reading
         * ~813 codes (0.65 V) instead of the true ~2048 codes (1.65 V).
         * At 40 µs: 40 µs acq + conversion overhead << 125 µs period, so
         * 8 kHz output rate is preserved with margin. */
        .acq_time   = NRF_SAADC_ACQTIME_40US,
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
/* [FIX 3] Flag: has dc_estimate been seeded from the first real sample?
 * Set false at start of each recording, true on first EVT_DONE. */
static bool dc_seeded = false;

/* [R2+R5 FIX] The capture-path Q15 band-pass filter (was 20–950 Hz, 8 kHz)
 * and its declarations have been removed.
 *
 * Reason (R5): 20–950 Hz matched neither the heart (10–200 Hz) nor the
 * lung (100–1000 Hz) training band, distorting the signal before the
 * correct per-pipeline float band-passes in dsp_mfcc.c could apply.
 *
 * Reason (R2): Running arm_biquad_cascade_df1_q15 inside the SAADC ISR
 * delayed BLE link-layer interrupts, contributing to disconnects.
 * DC removal was also moved to the DSP worker thread (see below).
 *
 * Band selection is now handled entirely by the per-pipeline float
 * bp_coeffs in dsp_mfcc.c (heart: 10–200 Hz, lung: 100–1000 Hz). */

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
 * observed true; reset to false at the start of each REC.
 * [FIX E] Converted from volatile bool to atomic_t for correct cross-thread
 * visibility without data races. */
static atomic_t ble_tx_saw_recording = ATOMIC_INIT(0);

/* [IMP 5] / [FIX B] NACK retransmit window.
 * Expanded from 1 slot to NACK_WINDOW (4) slots so we can retransmit
 * any of the last 4 chunks, covering NACK arrival after subsequent sends. */
#define NACK_WINDOW  4

static uint8_t  nack_buf[NACK_WINDOW][251]   __aligned(4);
static uint16_t nack_buf_seq[NACK_WINDOW];
static uint16_t nack_buf_len[NACK_WINDOW];

static uint16_t  nack_seq_pending = 0xFFFF;
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

            /* [R2 FIX] DC removal moved here from the SAADC ISR so the ISR
             * stays minimal and BLE link-layer interrupts are not delayed.
             * Time constant >> 10 (~128 ms at 8 kHz) — same as before. */
#ifdef ENABLE_DC_REMOVAL
            for (uint32_t i = 0; i < HALF_BUF_SAMPLES; i++) {
                dc_estimate += ((int32_t)dsp_pop_buf[i] - dc_estimate) >> 10;
                dsp_pop_buf[i] = (int16_t)((int32_t)dsp_pop_buf[i] - dc_estimate);
            }
#endif

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

        /* [FIX 3] Seed dc_estimate from the first real sample on each
         * recording. dc_estimate is set to 0 as a sentinel in
         * saadc_start_streaming(); here, on the very first EVT_DONE, we
         * capture the raw ADC value (true DC ~2048 after Fix 1) so the
         * IIR starts without a large initial error and the first MFCC
         * frames are clean. The sentinel 0 is safe because a true DC of
         * 0 would only arise with the chip powered off. */
        if (!dc_seeded) {
            dc_estimate = (int32_t)filled_buf[0];
            dc_seeded   = true;
        }
        /* Reset seed flag when streaming stops so next recording re-seeds */

#if BLE_AUDIO_LIVE
        /* Stream the captured SAADC samples before DSP mutates the buffer.
         * ring_buf_put() copies the bytes, so the in-place filtering below
         * still feeds the MFCC pipelines without altering the BLE copy. */
        uint32_t ble_written = ring_buf_put(&audio_ring,
                                            (const uint8_t *)filled_buf,
                                            HALF_BUF_BYTES);
        if (ble_written != HALF_BUF_BYTES) {
            atomic_inc(&ring_drops);
            LOG_WRN_ONCE("audio_ring overflow - BLE stream will have gaps!");
        } else {
            uint32_t fill = ring_buf_size_get(&audio_ring);
            if (fill > ring_high_water) {
                ring_high_water = fill;
            }
        }
#endif

        /* [R2 FIX] DC removal and band-pass are now done in the DSP worker
         * thread (below), NOT here in the ISR.  A long-running ISR delays
         * BLE link-layer interrupts and is a primary cause of disconnects.
         * The ISR is now minimal: copy raw DMA buffer → dsp_ring, signal. */

        /* [R5 FIX] The capture-path Q15 band-pass (was 20–950 Hz) has been
         * removed entirely.  That filter matched neither the heart (10–200 Hz)
         * nor the lung (100–1000 Hz) training band and distorted the signal
         * before the correct per-pipeline float band-passes in dsp_mfcc.c
         * could apply their own bands.  Per-pipeline filtering in dsp_mfcc.c
         * already handles band selection correctly. */

        /* DSP ring — raw, unfiltered samples */
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

    /* [R2+R5 FIX] Q15 BPF init removed: filter was wrong-band and ISR-blocking. */

    /* [FIX F] IRQ_CONNECT removed from here — now in saadc_init_once() */

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
    /* [FIX 3] Seed DC estimate from the first real ADC sample, not a
     * hardcoded midpoint. Before Fix 1, the ADC read ~813 (not 2048), so
     * seeding at 2048 caused a -1383 code transient and ~128 ms of HPF
     * ringing that corrupted the first several MFCC frames every recording.
     * dc_estimate is set here to 0 as a sentinel; saadc_event_handler will
     * overwrite it with ping_pong[0][0] on the very first EVT_DONE. */
    dc_estimate = 0;  /* overwritten by first sample — see saadc_event_handler */

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
    dc_seeded = false;   /* [FIX 3] allow next recording to re-seed DC estimate */
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
        /* [FIX 4a] Supervision timeout raised from 400 (4 s) to 1000 (10 s).
         * During recording, the main thread runs SAADC + DSP + TFLM inference
         * concurrently with BLE at ~16 KB/s. The old 4 s timeout was tight
         * enough that a heavy inference phase could starve the link layer and
         * trigger a central drop. 10 s gives ample slack while staying well
         * inside the 32 s maximum (timeout units = 10 ms). */
        .interval_min = 12, .interval_max = 24, .latency = 0, .timeout = 1000,
    };
    bt_conn_le_param_update(conn, &fast_conn);
    bt_gatt_exchange_mtu(conn, &exchange_params);
    LOG_INF("Connected");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
    /* [FIX 4c] On disconnect, do NOT abort an in-progress recording.
     * Previously this called saadc_stop_streaming() and cleared
     * analog_recording, which discarded all captured audio. Now we let
     * the SAADC and DSP pipelines run to completion so the recording is
     * preserved. The BLE TX thread will stall (ring backs up) but the
     * 32 KB audio ring provides ~2 s of headroom. On reconnect the host
     * can re-request results or the firmware resumes streaming the ring.
     * We DO clear start_recording so no new REC is auto-started. */
    if (analog_recording) {
        LOG_WRN("Disconnected mid-recording (reason=%d) — preserving capture, "
                "ring will absorb until reconnect", reason);
        /* SAADC keeps running; do not call saadc_stop_streaming() here */
    }
    is_connected    = false;
    start_recording = false;
    mtu_exchanged   = false;
    nus_chunk_size  = 244;
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
 * [IMP 5] Handle "NACK:<seq>\n" from host.
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
 * [IMP 5] Extended chunk header: [seq16][len16][crc16] = 6 bytes.
 * [FIX B] NACK retransmit window: keeps last NACK_WINDOW chunks
 *         so retransmit works even if NACK arrives after later sends.
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
            /* [FIX E] Use atomic_set instead of plain bool assignment */
            atomic_set(&ble_tx_saw_recording, 1);
        }

        while (true) {
            /* [FIX B] Check for pending NACK — search window for matching seq */
            k_mutex_lock(&nack_mutex, K_FOREVER);
            uint16_t pending_nack = nack_seq_pending;
            if (pending_nack != 0xFFFF) {
                nack_seq_pending = 0xFFFF;   /* consume */
            }
            k_mutex_unlock(&nack_mutex);

            if (pending_nack != 0xFFFF) {
                /* Search NACK_WINDOW for the requested seq */
                bool retransmitted = false;
                for (int i = 0; i < NACK_WINDOW; i++) {
                    if (nack_buf_len[i] > 0 && nack_buf_seq[i] == pending_nack) {
                        LOG_INF("Retransmitting seq=%u (%u B) from window[%d]",
                                pending_nack, nack_buf_len[i], i);
                        int err;
                        do {
                            err = bt_nus_send(NULL, nack_buf[i], nack_buf_len[i]);
                            if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(2));
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
                /* [FIX E] Use atomic_get for ble_tx_saw_recording */
                if (!analog_recording && !finished_sent &&
                    atomic_get(&ble_tx_saw_recording)) {
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
            uint16_t crc = crc16_ccitt(0xFFFF, &chunk[CHUNK_HEADER_BYTES], send_bytes);
            sys_put_le16(tx_seq,     &chunk[0]);
            sys_put_le16(send_bytes, &chunk[2]);
            sys_put_le16(crc,        &chunk[4]);

            uint16_t total_len = CHUNK_HEADER_BYTES + send_bytes;

            /* [FIX B] Store chunk in circular NACK window */
            int slot = tx_seq % NACK_WINDOW;
            if (total_len <= sizeof(nack_buf[slot])) {
                memcpy(nack_buf[slot], chunk, total_len);
                nack_buf_seq[slot] = tx_seq;
                nack_buf_len[slot] = total_len;
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

    /* [FIX A] drops declared unconditionally so LOG_INF at end always compiles.
     * Value is populated inside #if BLE_AUDIO_LIVE below. */
    uint32_t drops = 0;

#if BLE_AUDIO_LIVE
    k_sem_reset(&audio_data_sem);
    ring_buf_reset(&audio_ring);
    atomic_set(&ring_drops, 0);
    ring_high_water = 0;
    tx_seq          = 0;
    /* [FIX E] Use atomic_set */
    atomic_set(&ble_tx_saw_recording, 0);
    k_sem_reset(&tx_done_sem);

    /* Clear NACK window */
    memset(nack_buf_len, 0, sizeof(nack_buf_len));
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

    /* [FIX A] drops is now always in scope here */
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
