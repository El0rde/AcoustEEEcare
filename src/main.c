/*
 * AcoustEEEcare — SAADC BLE + SD Card Edition
 * ============================================================
<<<<<<< Updated upstream
 * v6.6 — Fixed Zephyr FS API usage (fat_fs type, fs_data pointer)
 *
 * CHANGES FROM v6.5:
 *   Added #include <ff.h> back under USE_SD (was removed in v6.5).
 *     Zephyr's own shell.c (subsys/fs/shell.c) includes <ff.h> when
 *     CONFIG_FAT_FILESYSTEM_ELM is set.  The FATFS typedef lives in
 *     ff.h — without it the compiler cannot size the fat_fs variable.
 *     Note: ff.h is provided by the fatfs module (west managed); it
 *     is NOT a file you write yourself.
 *
 *   Removed redundant k_sem_give(&sd_data_sem) from ISR.
 *     The semaphore is given once per recording in record_and_stream()
 *     to wake the SD writer thread.  Giving it on every half-buffer
 *     floods the semaphore counter and masks real backpressure.
 *
 *   Added LOG_WRN_ONCE() for ring drop events so silent data loss is
 *     visible in the log.
 *
 * REMAINING DESIGN (unchanged from v6.5):
 *   - No fs_write() in ISR — only ring_buf_put() + k_sem_give()
 *   - Two ring buffers: audio_ring (16 KB BLE), sd_ring (32 KB SD)
 *   - SD writer thread drains sd_ring in 512-byte aligned sectors
 *   - XOR checksum appended at end of file
 *   - SPI SD at 4 MHz for broad card compatibility
 *
 * TARGET HARDWARE
 *   Seeed XIAO nRF52840
 *   Analog MEMS microphone on AIN0 (P0.02), cap-coupled
 *   SD card on SPI2, CS on P0.28 (adjust overlay to your wiring)
 *
 * COMPILE SWITCH
 *   #define USE_SD  true   — record to SD + BLE stream
 *   #define USE_SD  false  — BLE-only (v6.3 behaviour, no SD required)
 * ============================================================
 */

#define USE_SD  false   /* set false for BLE-only mode */
=======
 * v7.5 — BSS-corruption fixes + RAM usage instrumentation
 *
 * CHANGES FROM v7.4:
 *
 *   [FIX 10] heart_window sized 50 -> 60 to match heart frame_samples=60.
 *            Previously memset(p->window, 0, 60*2)=120 bytes overflowed
 *            a 100-byte buffer on every dsp_mfcc_reset(), silently
 *            corrupting whatever sat next in BSS.
 *
 *   [FIX 11] lung_window sized 100 -> 1200 to match lung frame_samples=1200.
 *            The previous 200-byte buffer was being memset'd with
 *            2400 bytes on every reset — a 2200-byte BSS overflow.
 *            THIS WAS THE PRIMARY CAUSE OF THE SECOND-REC HANG.
 *
 *   [FIX 12] SD_WRITE_BUF_SIZE bumped 512 -> 4096 (one FAT cluster on
 *            most SD cards). Cuts per-write FATFS overhead ~8x, which
 *            is what was causing the sd_drops=36 (~36 KB of dropped
 *            audio on the first REC).
 *
 *   [FIX 13] AUDIO_SD_RING_BYTES bumped 16 KB -> 32 KB. Was exactly 1 s
 *            of audio at 8 kHz/int16. Any fs_write stall >1 s caused
 *            ring overflow. 32 KB gives ~2 s of headroom.
 *
 *   [FIX 14] audio.pcm file is pre-truncated to expected size right
 *            after fs_open. FATFS allocates clusters up front so
 *            subsequent writes skip FAT updates entirely.
 *
 *   [FIX 15] report_ram_usage() now also dumps:
 *            - tensor_arena last "used" bytes (set by tflm_inference)
 *            - heap stats via sys_heap_runtime_stats_get()
 *            - per-thread stack high-water using k_thread_foreach()
 *            Called BEFORE recording starts AND after each phase.
 *
 *   [FIX 16] _bss_end symbol name resolved with weak fallback for
 *            both Zephyr linker script flavours (__bss_end vs _end).
 *
 * ALL PREVIOUS FIXES (v7.4 / v7.3 / v7.2 / v7.1 / v6.8) ARE PRESERVED.
 * ============================================================
 */

#define USE_SD  true

#define BLE_AUDIO_LIVE  0   /* must stay 0 for arena RAM to be available */
#define DSP_OFFLINE     1   /* must stay 1 for arena RAM to be available */


/* ── Model enable flags ─────────────────────────────────────────── */
/* #define ENABLE_HEART_MODEL  1    set to 0 while heart model absent */
/* #define ENABLE_LUNG_MODEL   1    set to 1 once lung model is ready */ 

>>>>>>> Stashed changes

/* ══════════════════════════════════════════════════════════════════
 * INCLUDES
 * ══════════════════════════════════════════════════════════════════ */
#include <stdint.h>
#include <string.h>
#include <stdio.h>
<<<<<<< Updated upstream
// #include <math.h>
=======
#include <errno.h>
>>>>>>> Stashed changes

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
#include "zephyr/kernel/thread_stack.h"
// #include "zephyr/sys/time_units.h"

/* v7.5: optional thread analyzer + heap stats for RAM tracking */
#if defined(CONFIG_THREAD_ANALYZER)
#include <zephyr/debug/thread_analyzer.h>
#endif
#if defined(CONFIG_SYS_HEAP_RUNTIME_STATS)
#include <zephyr/sys/sys_heap.h>
extern struct sys_heap _system_heap;   /* defined by Zephyr kernel */
#endif

#include <nrfx.h>
#include <drivers/nrfx_errors.h>
#include <nrfx_saadc.h>

#include "dsp_mfcc.h"

#if USE_SD
#include <zephyr/fs/fs.h>
#include <zephyr/storage/disk_access.h>
/*
 * ff.h provides the FATFS typedef (the ELM FatFs work-area struct).
 * It is supplied by the fatfs west module when CONFIG_FAT_FILESYSTEM_ELM=y.
 * Zephyr's own subsys/fs/shell.c includes it the same way.
 * Do NOT write your own ff.h — west pulls the real one from
 * https://github.com/zephyrproject-rtos/fatfs
 */
#include <ff.h>
#endif

LOG_MODULE_REGISTER(AcoustEEEcare);

/* ══════════════════════════════════════════════════════════════════
<<<<<<< Updated upstream
=======
 * FORWARD DECLARATIONS for debug beacons
 *
 * The LED helpers (led_set_*, led_on, led_off) and the LED gpio_dt_spec
 * variables are defined further down in the file. We forward-declare
 * them up here so earlier functions (debug_flash_twice, saadc_init,
 * saadc_start_streaming) can use them.
 *
 * Also declares saadc_start_streaming and saadc_stop_streaming so
 * record_and_stream can call them.
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
 *
 * [FIX 2] 100 KB — sized from the actual model:
 *   best_mcu_int8 input [1, 665, 25, 1], peak_pair_bytes = 83,712.
 *   With TFLM overhead (~8 KB) the realistic need is ~92 KB.
 *   100 KB gives ~8 KB headroom.  After AllocateTensors() runs once,
 *   check the "arena used" log line and tighten further if you want.
 * ══════════════════════════════════════════════════════════════════ */
#define TENSOR_ARENA_BYTES  (72u * 1024u)

/* __used so the linker keeps it even if ENABLE_*_MODEL=0,
 * preserving BSS layout for diagnostic builds. */
static uint8_t tensor_arena[TENSOR_ARENA_BYTES] __aligned(16) __used;

/* ══════════════════════════════════════════════════════════════════
 * RAM USAGE REPORT
 *
 * [FIX 3] Use linker-provided symbols.  __bss_start and _end (or
 * __bss_end depending on Zephyr's linker script flavour) bracket the
 * BSS region; both are defined by the Zephyr linker script.
 * _image_ram_end and _image_ram_start bracket the entire RAM image
 * (including noinit, where thread stacks live).
 * ══════════════════════════════════════════════════════════════════ */
/* v7.5 [FIX 16]: __bss_end isn't defined on every Zephyr linker
 * script flavour. Provide weak fallback alias to _end. */
extern char __bss_start;
extern char __bss_end __attribute__((weak));
extern char _end      __attribute__((weak));
extern char _image_ram_start;
extern char _image_ram_end;

#define BLE_TX_STACK_SIZE     1024
#if DSP_OFFLINE
#define DSP_THREAD_STACK_SIZE 1024
#else
#define DSP_THREAD_STACK_SIZE 4096
#endif
#define SD_WRITER_STACK_SIZE  2048

static struct k_thread ble_tx_thread_data;
static struct k_thread dsp_thread_data;
static struct k_thread sd_writer_thread_data;
static dsp_mfcc_pipeline_t heart_pipeline;
static dsp_mfcc_pipeline_t lung_pipeline;
static uint32_t heart_frame_count;
static uint32_t lung_frame_count;

/* v7.5 [FIX 15]: TFLM arena high-water (updated by tflm_inference.cc
 * via this extern). Lets report_ram_usage() show the actual usage. */
volatile uint32_t g_tflm_arena_used_bytes = 0;

#if defined(CONFIG_THREAD_ANALYZER)
static void thread_stack_dump_cb(const struct k_thread *thread, void *user_data)
{
    ARG_UNUSED(user_data);
    size_t unused;
    if (k_thread_stack_space_get(thread, &unused) != 0) {
        return;
    }
    const char *name = k_thread_name_get((k_tid_t)thread);
    LOG_INF("  %-12s  unused=%5u B", name ? name : "?", (unsigned)unused);
}
#endif

static void report_ram_usage(const char *label)
{
    uintptr_t bss_start   = (uintptr_t)&__bss_start;
    /* Prefer __bss_end if present, else fall back to _end. */
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

    /* ── Heap stats ─────────────────────────────────────────────── */
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
    LOG_INF("Heap:          (CONFIG_SYS_HEAP_RUNTIME_STATS=n; "
            "stats unavailable)");
#endif

    /* ── Per-thread stack high-water ────────────────────────────── */
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
#if USE_SD
    if (k_thread_stack_space_get(&sd_writer_thread_data, &unused) == 0) {
        LOG_INF("SD writer:     %u / %u B used (%u%% headroom)",
                (unsigned)(SD_WRITER_STACK_SIZE - unused),
                (unsigned)SD_WRITER_STACK_SIZE,
                (unsigned)(unused * 100u / SD_WRITER_STACK_SIZE));
    }
#endif

    /* ── All-thread analyzer (covers main, idle, BLE rx, etc.) ── */
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
>>>>>>> Stashed changes
 * SAADC CONFIG
 * ══════════════════════════════════════════════════════════════════ */
#define SAADC_CC_VALUE      2000U   /* 16 MHz / 2000 = 8 kHz */
#define SAADC_IRQ_PRIORITY  6

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

/* DC removal: high-pass IIR, alpha = 1/256.
 * Init to 0 — converges within ~256 samples (~32 ms at 8 kHz).
 * DO NOT init to 2048: that creates a large transient spike at the
 * start of every recording. */
#define ENABLE_DC_REMOVAL
static int32_t dc_estimate = 0;

/* ══════════════════════════════════════════════════════════════════
 * AUDIO PARAMS
 * ══════════════════════════════════════════════════════════════════ */
#define SAMPLING_RATE        8000
#define DURATION_S           10
#define TOTAL_AUDIO_SAMPLES  (SAMPLING_RATE * DURATION_S)            /* 80 000 */
#define TOTAL_AUDIO_BYTES    (TOTAL_AUDIO_SAMPLES * sizeof(int16_t)) /* 160 000 */

#define HALF_BUF_SAMPLES     512
#define HALF_BUF_BYTES       (HALF_BUF_SAMPLES * sizeof(int16_t))   /* 1 024 */

static int16_t ping_pong[2][HALF_BUF_SAMPLES];
static volatile uint8_t  next_dma_buf     = 1;
static volatile bool     analog_recording = false;
static uint16_t          nus_chunk_size   = 244;

/* ══════════════════════════════════════════════════════════════════
 * BLE AUDIO RING BUFFER + TX THREAD
 * ══════════════════════════════════════════════════════════════════ */
#define AUDIO_RING_BYTES  (16 * 1024)
RING_BUF_DECLARE(audio_ring, AUDIO_RING_BYTES);

static K_SEM_DEFINE(half_produced_sem, 0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(audio_data_sem,    0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(tx_done_sem,       0, 1);

static atomic_t  ring_drops;
static uint32_t  ring_high_water = 0;

static uint16_t tx_seq = 0;
#define CHUNK_HEADER_BYTES  4

#define BLE_TX_STACK_SIZE  2048
#define BLE_TX_PRIORITY    5
static K_THREAD_STACK_DEFINE(ble_tx_stack, BLE_TX_STACK_SIZE);
static struct k_thread ble_tx_thread_data;
static void ble_tx_thread_fn(void *a, void *b, void *c);

/* ══════════════════════════════════════════════════════════════════
 * SD CARD GLOBALS  (USE_SD true only)
 * ══════════════════════════════════════════════════════════════════ */
#if USE_SD

#define AUDIO_FILE_PATH      "/SD:/analog.pcm"
<<<<<<< Updated upstream
#define SD_CARD_MOUNT_POINT  "/SD:"
#define CHECKSUM_SIZE        sizeof(uint32_t)

/*
 * SD ring buffer — 32 KB.
 * At 16 000 bytes/s (8 kHz x 2 bytes) this holds 2 s of audio,
 * giving the FAT writer thread ample headroom even on slow cards.
 * The ISR fills it; the SD writer thread drains it.
 * NO fs_write() calls happen in the ISR.
 */
#define SD_RING_BYTES  (32 * 1024)
RING_BUF_DECLARE(sd_ring, SD_RING_BYTES);

static atomic_t  sd_ring_drops;
static uint32_t  sd_ring_high_water = 0;
=======
#define HEART_MFCC_FILE_PATH "/SD:/heart_mfcc.f32"
#define LUNG_MFCC_FILE_PATH  "/SD:/lung_mfcc.f32"
#define HR_RESULT_FILE_PATH  "/SD:/hr.txt"
#define RR_RESULT_FILE_PATH  "/SD:/rr.txt"

/* [FIX 7] Probe file path used by the 6-step init sequence. */
#define SD_INIT_CHECK_PATH   "/SD:/acoustchk"

#define CHECKSUM_SIZE        sizeof(uint32_t)

/* [FIX 7] SD init constants from the isolated test. */
#define SD_DISK_NAME         "SD"
#define SD_INIT_RETRIES      5
#define SD_INIT_RETRY_MS     1500
#define SD_RAIL_STABILISE_MS 250

/* Probe sentinel — self-documenting if the file is left on the card. */
static const char sd_probe_payload[] = "AcoustEEEcare-SD-probe-v7.4\n";

/* v7.5 [FIX 13]: 32 KB = ~2 s of audio at 8 kHz/int16.
 * Previous 16 KB was exactly 1 s — any fs_write stall >1 s caused
 * ring overflow and the sd_drops=36 we saw on the first REC. */
#define AUDIO_SD_RING_BYTES      (32 * 1024)
RING_BUF_DECLARE(audio_sd_ring,   AUDIO_SD_RING_BYTES);

#if !DSP_OFFLINE
#define HEART_MFCC_RING_BYTES    ( 8 * 1024)
#define LUNG_MFCC_RING_BYTES     (16 * 1024)
RING_BUF_DECLARE(heart_mfcc_ring, HEART_MFCC_RING_BYTES);
RING_BUF_DECLARE(lung_mfcc_ring,  LUNG_MFCC_RING_BYTES);
static uint32_t  heart_ring_high_water = 0;
static uint32_t  lung_ring_high_water  = 0;
#endif /* !DSP_OFFLINE */

static atomic_t  sd_ring_drops;
static uint32_t  audio_ring_high_water = 0;

#endif /* USE_SD — data declarations */

/* ══════════════════════════════════════════════════════════════════
 * DUAL MFCC PIPELINE INSTANCES
 *
 * v7.5 [FIX 10/11] Window sizes MUST equal cfg->frame_samples or
 * dsp_mfcc_reset() will overflow the buffer on every reset.
 *   heart: frame_samples = 60   (was sized 50 -> 120 B overflow)
 *   lung : frame_samples = 1200 (was sized 100 -> 2200 B overflow!)
 * The lung overflow was the primary cause of the second-REC hang.
 * ══════════════════════════════════════════════════════════════════ */
static int16_t heart_window[60]   __aligned(4);
static int16_t lung_window[1200]  __aligned(4);

#if USE_SD
static struct fs_file_t *offline_heart_fp = NULL;
static struct fs_file_t *offline_lung_fp  = NULL;
#endif

static void heart_frame_cb(int idx, const float *coeffs, void *user)
{
    (void)user; (void)idx;
    heart_frame_count++;

#if USE_SD
    const uint32_t bytes = heart_pipeline.cfg->n_mfcc * sizeof(float);
    if (offline_heart_fp != NULL) {
        ssize_t w = fs_write(offline_heart_fp, coeffs, bytes);
        if (w != (ssize_t)bytes) {
            LOG_WRN_ONCE("offline heart fs_write short: %d/%u", (int)w, bytes);
        }
    }
#if !DSP_OFFLINE
    else {
        uint32_t put = ring_buf_put(&heart_mfcc_ring,
                                    (const uint8_t *)coeffs, bytes);
        if (put != bytes) {
            atomic_inc(&sd_ring_drops);
        }
    }
#endif
#else
    (void)coeffs;
#endif
}

static void lung_frame_cb(int idx, const float *coeffs, void *user)
{
    (void)user; (void)idx;
    lung_frame_count++;

#if USE_SD
    const uint32_t bytes = lung_pipeline.cfg->n_mfcc * sizeof(float);
    if (offline_lung_fp != NULL) {
        ssize_t w = fs_write(offline_lung_fp, coeffs, bytes);
        if (w != (ssize_t)bytes) {
            LOG_WRN_ONCE("offline lung fs_write short: %d/%u", (int)w, bytes);
        }
    }
#if !DSP_OFFLINE
    else {
        uint32_t put = ring_buf_put(&lung_mfcc_ring,
                                    (const uint8_t *)coeffs, bytes);
        if (put != bytes) {
            atomic_inc(&sd_ring_drops);
        }
    }
#endif
#else
    (void)coeffs;
#endif
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

#if DSP_OFFLINE
    while (true) {
        k_sleep(K_FOREVER);
    }
#else
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
#endif /* DSP_OFFLINE */
}

/* ══════════════════════════════════════════════════════════════════
 * SD CARD GLOBALS
 * ══════════════════════════════════════════════════════════════════ */
#if USE_SD
>>>>>>> Stashed changes

/*
 * sd_data_sem — given ONCE per recording by record_and_stream() to
 * wake the SD writer thread.  NOT given per half-buffer from the ISR;
 * flooding the semaphore counter served no purpose and masked
 * backpressure.  The writer thread polls ring_buf_size_get() in its
 * own loop.
 */
static K_SEM_DEFINE(sd_data_sem, 0, K_SEM_MAX_LIMIT);
/* Semaphore: SD writer thread signals main thread when file is closed */
static K_SEM_DEFINE(sd_done_sem, 0, 1);

/* SD writer thread */
#define SD_WRITER_STACK_SIZE  2048
#define SD_WRITER_PRIORITY    6   /* lower priority than BLE TX (5) */
static K_THREAD_STACK_DEFINE(sd_writer_stack, SD_WRITER_STACK_SIZE);
static struct k_thread sd_writer_thread_data;
static void sd_writer_thread_fn(void *a, void *b, void *c);

/*
 * SD filesystem state.
 *
 * fat_fs  — FATFS work-area struct (typedef from ff.h / ELM FatFs).
 *            This is the actual buffer FatFs uses internally to track
 *            the mounted volume.  Its address goes into mp.fs_data.
 *
 * mp      — Zephyr fs_mount_t descriptor.
 *            .type    = FS_FATFS  → integer enum telling the VFS switch
 *                                   which filesystem ops to call.
 *            .fs_data = &fat_fs   → pointer to the FATFS work area.
 *
 * v6.5 BUG: mp.fs_data was set to &FS_FATFS — the address of an
 * integer constant — not &fat_fs.  This caused fs_mount() to write the
 * FatFs volume state into a random read-only location, producing a
 * hard-fault or silent memory corruption.
 */
static FATFS            fat_fs;          /* ELM FatFs work area — sized by ff.h */
static struct fs_mount_t mp = {
    .type      = FS_FATFS,              /* integer type ID — tells VFS "use FAT" */
    .mnt_point = SD_CARD_MOUNT_POINT,
    .fs_data   = &fat_fs,              /* pointer to FATFS work area — FIXED */
};
static bool             sd_mounted  = false;
static struct fs_file_t sd_file;
static uint32_t         sd_checksum = 0;   /* running XOR */

<<<<<<< Updated upstream
/* SD writer write buffer — 512 bytes aligns to FAT sector size */
#define SD_WRITE_BUF_SIZE  512
static uint8_t sd_write_buf[SD_WRITE_BUF_SIZE];
=======
/* [FIX 1] sd_write_buf forced to 4-byte alignment for nRF52 EasyDMA.
 * v7.5 [FIX 12]: bumped 512 -> 4096 (one FAT cluster on typical SDs).
 * Cuts per-write FATFS bookkeeping ~8x. Combined with FIX 14 file
 * pre-truncation, this should eliminate sd_drops entirely. */
#define SD_WRITE_BUF_SIZE  4096
static uint8_t sd_write_buf[SD_WRITE_BUF_SIZE] __aligned(4);
>>>>>>> Stashed changes

/* XOR checksum over raw bytes */
static uint32_t compute_checksum(const uint8_t *data, uint32_t len)
{
    uint32_t cs = 0;
    for (uint32_t i = 0; i < len; i++) cs ^= data[i];
    return cs;
}

<<<<<<< Updated upstream
/* ── SD card init ─────────────────────────────────────────────── */
static int init_sd_card(void)
{
    static const char *disk_pdrv = "SD";
    uint64_t memory_size_mb;
    uint32_t block_count, block_size;

    LOG_INF("Initialising SD card...");

    if (disk_access_init(disk_pdrv) != 0) {
        LOG_ERR("disk_access_init failed");
        return -1;
    }
    if (disk_access_ioctl(disk_pdrv, DISK_IOCTL_GET_SECTOR_COUNT, &block_count)) {
        LOG_ERR("Cannot get sector count");
        return -1;
    }
    if (disk_access_ioctl(disk_pdrv, DISK_IOCTL_GET_SECTOR_SIZE, &block_size)) {
        LOG_ERR("Cannot get sector size");
        return -1;
    }

    memory_size_mb = (uint64_t)block_count * block_size / (1024 * 1024);
    LOG_INF("SD card: %u MB", (uint32_t)memory_size_mb);

    /*
     * mp is fully initialised at declaration (type, mnt_point, fs_data).
     * Just call fs_mount() — no patching needed here.
     */
    if (fs_mount(&mp) != 0) {
        LOG_ERR("fs_mount failed");
        return -1;
    }

    LOG_INF("SD mounted at %s", SD_CARD_MOUNT_POINT);
    sd_mounted = true;
=======
/* ══════════════════════════════════════════════════════════════════
 * init_sd_card — 6-step hardened sequence (from isolated test v6.6)
 *
 * [FIX 7] Replaces the v7.3 single-shot init + verify_sd_writable().
 *
 *   Step 1  disk_access_init with retries + rail-stabilise delay
 *   Step 2  DISK_IOCTL_GET_SECTOR_COUNT — card responds sanity check
 *   Step 3  DISK_IOCTL_GET_SECTOR_SIZE  — must be 512 for FATFS
 *   Step 4  fs_mount() + CTRL_SYNC diagnostic on failure
 *   Step 5  fs_statvfs() free-space check (non-fatal warn if full)
 *   Step 6  probe write → seek → read → memcmp → unlink
 *           (supersedes old verify_sd_writable() single-byte sentinel)
 * ══════════════════════════════════════════════════════════════════ */
static int init_sd_card(void)
{
    int ret;

    LOG_INF("Initialising SD card (6-step sequence)...");

    /* ── Step 1: disk_access_init with retries ───────────────────── */
    LOG_INF("[1/6] Initialising disk '%s' ...", SD_DISK_NAME);
    ret = -EIO;
    for (int attempt = 0; attempt < SD_INIT_RETRIES; attempt++) {
        ret = disk_access_init(SD_DISK_NAME);
        if (ret == 0) {
            break;
        }
        LOG_WRN("  attempt %d/%d failed (%d), retrying ...",
                attempt + 1, SD_INIT_RETRIES, ret);
        LOG_INF("  Waiting %d ms for SD power rail to stabilise...",
                SD_RAIL_STABILISE_MS);
        k_msleep(SD_RAIL_STABILISE_MS);
        k_msleep(SD_INIT_RETRY_MS);
    }
    if (ret != 0) {
        LOG_ERR("[1/6] FAIL: disk_access_init returned %d after %d attempts",
                ret, SD_INIT_RETRIES);
        return ret;
    }
    LOG_INF("[1/6] disk_access_init OK");

    /* ── Step 2: sector count ────────────────────────────────────── */
    uint32_t sector_count = 0;
    ret = disk_access_ioctl(SD_DISK_NAME,
                            DISK_IOCTL_GET_SECTOR_COUNT, &sector_count);
    if (ret != 0 || sector_count == 0) {
        LOG_ERR("[2/6] FAIL: sector count ioctl ret=%d count=%u",
                ret, sector_count);
        return (ret != 0) ? ret : -EIO;
    }
    LOG_INF("[2/6] sector_count=%u  (~%u MiB)",
            sector_count,
            (uint32_t)((uint64_t)sector_count * 512u / (1024u * 1024u)));

    /* ── Step 3: sector size ─────────────────────────────────────── */
    uint32_t sector_size = 0;
    ret = disk_access_ioctl(SD_DISK_NAME,
                            DISK_IOCTL_GET_SECTOR_SIZE, &sector_size);
    if (ret != 0 || sector_size != 512u) {
        LOG_ERR("[3/6] FAIL: sector size ret=%d size=%u (expected 512)",
                ret, sector_size);
        return (ret != 0) ? ret : -EIO;
    }
    LOG_INF("[3/6] sector_size=%u", sector_size);

    /* ── Step 4: mount FAT filesystem ────────────────────────────── */
    ret = fs_mount(&mp);
    if (ret != 0) {
        LOG_ERR("[4/6] FAIL: fs_mount returned %d", ret);
        /* CTRL_SYNC distinguishes SPI wiring fault from wrong FS format. */
        int sync_ret = disk_access_ioctl(SD_DISK_NAME,
                                         DISK_IOCTL_CTRL_SYNC, NULL);
        LOG_INF("  CTRL_SYNC probe: %s",
                (sync_ret == 0)
                    ? "SPI OK — check card format "
                      "(exFAT needs CONFIG_FS_FATFS_EXFAT=y)"
                    : "SPI also failing — check wiring/power");
        return ret;
    }
    LOG_INF("[4/6] FAT mount OK  (%s)", SD_CARD_MOUNT_POINT);
    sd_mounted = true;

    /* ── Step 5: free space ──────────────────────────────────────── */
    struct fs_statvfs sbuf;
    ret = fs_statvfs(SD_CARD_MOUNT_POINT, &sbuf);
    if (ret != 0) {
        /* Non-fatal — card may still be usable. */
        LOG_WRN("[5/6] fs_statvfs failed (%d) — continuing anyway", ret);
    } else {
        uint64_t free_bytes = (uint64_t)sbuf.f_bfree * sbuf.f_frsize;
        LOG_INF("[5/6] free=%llu MiB",
                (unsigned long long)(free_bytes / (1024u * 1024u)));
        if (sbuf.f_bfree == 0) {
            LOG_WRN("  WARNING: SD card is full — writes will fail");
        }
    }

    /* ── Step 6: probe write → seek → read → memcmp → unlink ────── */
    LOG_INF("[6/6] Probe write/read roundtrip on %s ...", SD_INIT_CHECK_PATH);

    const size_t probe_len = sizeof(sd_probe_payload) - 1u; /* exclude NUL */
    char probe_rd[sizeof(sd_probe_payload)];

    struct fs_file_t probe;
    fs_file_t_init(&probe);

    ret = fs_open(&probe, SD_INIT_CHECK_PATH, FS_O_CREATE | FS_O_RDWR);
    if (ret != 0) {
        LOG_ERR("[6/6] FAIL: fs_open probe returned %d", ret);
        fs_unmount(&mp);
        sd_mounted = false;
        return ret;
    }

    ssize_t w = fs_write(&probe, sd_probe_payload, probe_len);
    if (w < 0 || (size_t)w != probe_len) {
        LOG_ERR("[6/6] FAIL: fs_write probe returned %d (expected %u)",
                (int)w, (unsigned)probe_len);
        fs_close(&probe);
        fs_unlink(SD_INIT_CHECK_PATH);
        fs_unmount(&mp);
        sd_mounted = false;
        return (w < 0) ? (int)w : -EIO;
    }

    fs_seek(&probe, 0, FS_SEEK_SET);

    ssize_t r = fs_read(&probe, probe_rd, probe_len);
    fs_close(&probe);
    fs_unlink(SD_INIT_CHECK_PATH);

    if (r < 0 || (size_t)r != probe_len) {
        LOG_ERR("[6/6] FAIL: fs_read probe returned %d (expected %u)",
                (int)r, (unsigned)probe_len);
        fs_unmount(&mp);
        sd_mounted = false;
        return (r < 0) ? (int)r : -EIO;
    }

    if (memcmp(sd_probe_payload, probe_rd, probe_len) != 0) {
        LOG_ERR("[6/6] FAIL: probe read-back mismatch — data corruption");
        fs_unmount(&mp);
        sd_mounted = false;
        return -EIO;
    }

    LOG_INF("[6/6] probe write+read+verify OK  → SD is writable");
>>>>>>> Stashed changes
    return 0;
}

/* ── SD writer thread ─────────────────────────────────────────── */
/*
 * Runs at PRIORITY 6 (below BLE TX at 5).
 * Lifecycle per recording:
 *   1. Waits on sd_data_sem (given once per REC by record_and_stream).
 *   2. Drains sd_ring in SD_WRITE_BUF_SIZE chunks while
 *      analog_recording is true or ring is non-empty.
 *   3. Flushes any partial tail buffer.
 *   4. Appends 4-byte XOR checksum.
 *   5. Closes the file and signals sd_done_sem.
 *   6. Goes back to waiting on sd_data_sem.
 *
 * The file is opened by record_and_stream() before SAADC starts,
 * so the writer thread never needs to open it.
 */
static void sd_writer_thread_fn(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (true) {
        /* Wait until a recording starts */
        k_sem_take(&sd_data_sem, K_FOREVER);

        if (!sd_mounted) {
            LOG_ERR("SD writer: not mounted, discarding data");
<<<<<<< Updated upstream
            /* Drain the ring so it doesn't back up */
            while (ring_buf_size_get(&sd_ring) > 0) {
                uint32_t avail = ring_buf_size_get(&sd_ring);
                uint32_t drain = avail < SD_WRITE_BUF_SIZE ? avail : SD_WRITE_BUF_SIZE;
                ring_buf_get(&sd_ring, sd_write_buf, drain);
=======
            /* [FIX 5] Properly drain the rings. */
            while (ring_buf_size_get(&audio_sd_ring) > 0
#if !DSP_OFFLINE
                || ring_buf_size_get(&heart_mfcc_ring) > 0
                || ring_buf_size_get(&lung_mfcc_ring)  > 0
#endif
            ) {
                uint32_t avail = ring_buf_size_get(&audio_sd_ring);
                if (avail > 0) {
                    uint32_t n = MIN(avail, (uint32_t)SD_WRITE_BUF_SIZE);
                    ring_buf_get(&audio_sd_ring, sd_write_buf, n);
                }
#if !DSP_OFFLINE
                avail = ring_buf_size_get(&heart_mfcc_ring);
                if (avail > 0) {
                    uint32_t n = MIN(avail, (uint32_t)SD_WRITE_BUF_SIZE);
                    ring_buf_get(&heart_mfcc_ring, sd_write_buf, n);
                }
                avail = ring_buf_size_get(&lung_mfcc_ring);
                if (avail > 0) {
                    uint32_t n = MIN(avail, (uint32_t)SD_WRITE_BUF_SIZE);
                    ring_buf_get(&lung_mfcc_ring, sd_write_buf, n);
                }
#endif
>>>>>>> Stashed changes
            }
            k_sem_give(&sd_done_sem);
            continue;
        }

        LOG_INF("SD writer: starting write loop");
        uint32_t total_written = 0;
        sd_checksum = 0;

        /* Drain loop: keep going while recording is active OR ring has data */
        while (analog_recording || ring_buf_size_get(&sd_ring) > 0) {
            uint32_t avail = ring_buf_size_get(&sd_ring);

<<<<<<< Updated upstream
            if (avail == 0) {
                /* Nothing to write yet — yield and retry */
=======
            bool did_work = false;

            /* ── Audio ring (always present) ── */
            {
                uint32_t avail = ring_buf_size_get(&audio_sd_ring);
                bool done = !analog_recording;
                if (avail >= SD_WRITE_BUF_SIZE || (done && avail > 0)) {
                    uint32_t n = MIN(avail, (uint32_t)SD_WRITE_BUF_SIZE);
                    ring_buf_get(&audio_sd_ring, sd_write_buf, n);
                    sd_checksum ^= compute_checksum(sd_write_buf, n);
                    ssize_t wr = fs_write(&sd_audio_file, sd_write_buf, n);
                    if (wr < 0) {
                        /* [FIX 4] Bail out instead of hammering the card. */
                        LOG_ERR("SD audio fs_write failed: %d — aborting writer",
                                (int)wr);
                        write_error = true;
                        break;
                    }
                    audio_written += (uint32_t)wr;
                    uint32_t fill = ring_buf_size_get(&audio_sd_ring);
                    if (fill > audio_ring_high_water) audio_ring_high_water = fill;
                    did_work = true;
                }
            }

#if !DSP_OFFLINE
            /* ── Heart MFCC ring (online DSP only) ── */
            {
                uint32_t avail = ring_buf_size_get(&heart_mfcc_ring);
                bool done = !analog_recording;
                if (avail >= SD_WRITE_BUF_SIZE || (done && avail > 0)) {
                    uint32_t n = MIN(avail, (uint32_t)SD_WRITE_BUF_SIZE);
                    ring_buf_get(&heart_mfcc_ring, sd_write_buf, n);
                    ssize_t wr = fs_write(&sd_heart_file, sd_write_buf, n);
                    if (wr < 0) {
                        LOG_ERR("SD heart fs_write failed: %d — aborting", (int)wr);
                        write_error = true;
                        break;
                    }
                    heart_written += (uint32_t)wr;
                    uint32_t fill = ring_buf_size_get(&heart_mfcc_ring);
                    if (fill > heart_ring_high_water) heart_ring_high_water = fill;
                    did_work = true;
                }
            }

            /* ── Lung MFCC ring (online DSP only) ── */
            {
                uint32_t avail = ring_buf_size_get(&lung_mfcc_ring);
                bool done = !analog_recording;
                if (avail >= SD_WRITE_BUF_SIZE || (done && avail > 0)) {
                    uint32_t n = MIN(avail, (uint32_t)SD_WRITE_BUF_SIZE);
                    ring_buf_get(&lung_mfcc_ring, sd_write_buf, n);
                    ssize_t wr = fs_write(&sd_lung_file, sd_write_buf, n);
                    if (wr < 0) {
                        LOG_ERR("SD lung fs_write failed: %d — aborting", (int)wr);
                        write_error = true;
                        break;
                    }
                    lung_written += (uint32_t)wr;
                    uint32_t fill = ring_buf_size_get(&lung_mfcc_ring);
                    if (fill > lung_ring_high_water) lung_ring_high_water = fill;
                    did_work = true;
                }
            }
#endif /* !DSP_OFFLINE */

            if (!did_work) {
>>>>>>> Stashed changes
                k_sleep(K_MSEC(1));
                continue;
            }

            /*
             * Write in SD_WRITE_BUF_SIZE aligned blocks while recording,
             * but allow partial writes once recording has stopped (tail flush).
             */
            bool recording_done = !analog_recording;
            uint32_t to_write   = avail < SD_WRITE_BUF_SIZE ? avail : SD_WRITE_BUF_SIZE;

            /* During recording: hold partial blocks so we write full sectors */
            if (!recording_done && to_write < SD_WRITE_BUF_SIZE) {
                k_sleep(K_MSEC(1));
                continue;
            }

            ring_buf_get(&sd_ring, sd_write_buf, to_write);
            sd_checksum ^= compute_checksum(sd_write_buf, to_write);

            ssize_t written = fs_write(&sd_file, sd_write_buf, to_write);
            if (written < 0) {
                LOG_ERR("SD write error: %d at byte %u", (int)written, total_written);
                /* Continue anyway — partial file is better than crash */
            } else {
                total_written += (uint32_t)written;
            }
        }

        /* Append XOR checksum */
        fs_write(&sd_file, &sd_checksum, CHECKSUM_SIZE);
        fs_close(&sd_file);

        LOG_INF("SD writer: %u audio bytes written, checksum=0x%08X",
                total_written, sd_checksum);
        LOG_INF("SD writer: sd_ring high water = %u / %u", sd_ring_high_water, SD_RING_BYTES);

        k_sem_give(&sd_done_sem);
    }
}

#endif /* USE_SD */

/* ══════════════════════════════════════════════════════════════════
 * SAADC EVENT HANDLER
 *
 * On NRFX_SAADC_EVT_DONE (one half-buffer of HALF_BUF_SAMPLES ready):
 *   1. DC removal — in-place IIR high-pass on filled_buf
 *   2. dsp_mfcc_feed_chunk() — on-the-fly bandpass + decimate
 *   3a. [USE_SD true]  push to sd_ring (SD writer thread drains it)
 *   3b. push to audio_ring for BLE TX thread (both modes)
 *
 * IMPORTANT:
 *   No fs_write() here. Only ring_buf_put() and k_sem_give().
 *   sd_data_sem is NOT given here — it is given once per recording
 *   by record_and_stream() to avoid flooding the semaphore counter.
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

        /* ── Step 1: DC removal ── */
#ifdef ENABLE_DC_REMOVAL
        for (uint32_t i = 0; i < HALF_BUF_SAMPLES; i++) {
            dc_estimate += ((int32_t)filled_buf[i] - dc_estimate) >> 8;
            filled_buf[i] = (int16_t)((int32_t)filled_buf[i] - dc_estimate);
        }
#endif

        /* ── Step 2: On-the-fly bandpass + decimate ── */
        dsp_mfcc_feed_chunk(filled_buf, HALF_BUF_SAMPLES);

#if USE_SD
        /* ── Step 3a: Push to SD ring buffer (ISR-safe, no fs_write) ── */
        if (analog_recording) {
            uint32_t sd_written = ring_buf_put(&sd_ring,
                                               (const uint8_t *)filled_buf,
                                               HALF_BUF_BYTES);
            if (sd_written != HALF_BUF_BYTES) {
                atomic_inc(&sd_ring_drops);
                LOG_WRN_ONCE("sd_ring overflow — audio will have gaps!");
            } else {
                uint32_t fill = ring_buf_size_get(&sd_ring);
                if (fill > sd_ring_high_water) {
                    sd_ring_high_water = fill;
                }
            }
            /*
             * Do NOT give sd_data_sem here.
             * It is given once per recording in record_and_stream()
             * to wake the SD writer thread.  The writer polls
             * ring_buf_size_get() internally — no per-chunk signal needed.
             */
        }
#endif

        /* ── Step 3b: Push to BLE audio ring (both modes) ── */
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

        k_sem_give(&half_produced_sem);
        k_sem_give(&audio_data_sem);
        break;
    }

    default:
        break;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * SAADC INIT / START / STOP
 * ══════════════════════════════════════════════════════════════════ */

/* Tiny helper to flash a color twice with off in between.
 * Used as a unique "I got here" marker that won't be confused
 * with the steady-state colors from outer beacons. */
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

    /* SUB-SUB I: red flash x2 — entering saadc_init */
    debug_flash_twice(led_set_red);

    /* Only uninit if we previously initialized. Calling nrfx_saadc_uninit()
     * on a fresh, never-initialized SAADC peripheral can hang waiting for
     * hardware state bits that were never set in the first place. */
    if (saadc_was_initialized) {
        nrfx_saadc_uninit();
    }

    /* SUB-SUB II: green flash x2 — nrfx_saadc_uninit returned (or skipped) */
    debug_flash_twice(led_set_green);

    IRQ_CONNECT(SAADC_IRQn, SAADC_IRQ_PRIORITY, nrfx_saadc_irq_handler, NULL, 0);
    irq_enable(SAADC_IRQn);

    /* SUB-SUB III: blue flash x2 — IRQ_CONNECT + irq_enable done */
    debug_flash_twice(led_set_blue);

    err = nrfx_saadc_init(SAADC_IRQ_PRIORITY);
    if (err != 0) { LOG_ERR("nrfx_saadc_init: 0x%08X", err); return -EIO; }

    /* SUB-SUB IV: yellow flash x2 — nrfx_saadc_init returned OK */
    debug_flash_twice(led_set_yellow);

    err = nrfx_saadc_channel_config(&saadc_channel_cfg);
    if (err != 0) { LOG_ERR("channel_config: 0x%08X", err); return -EIO; }

    /* SUB-SUB V: purple flash x2 — channel_config returned OK */
    debug_flash_twice(led_set_purple);

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

    /* Mark as initialized so future REC calls correctly uninit first. */
    saadc_was_initialized = true;

    /* SUB-SUB VI: cyan flash x2 — advanced_mode_set returned OK, exiting saadc_init */
    debug_flash_twice(led_set_cyan);

    return 0;
}

static int saadc_start_streaming(void)
{
    nrfx_err_t err;
    saadc_dma_overruns = 0;
    dc_estimate        = 0;   /* Reset to 0, not 2048 — avoids initial spike */

<<<<<<< Updated upstream
    dsp_mfcc_reset();
=======
    /* SUB-BEACON A: red — entered saadc_start_streaming */
    led_set_red();
    k_busy_wait(300000);

    dsp_mfcc_reset(&heart_pipeline);

    /* SUB-BEACON B: green — heart pipeline reset done */
    led_set_green();
    k_busy_wait(300000);

    dsp_mfcc_reset(&lung_pipeline);

    /* SUB-BEACON C: blue — lung pipeline reset done */
    led_set_blue();
    k_busy_wait(300000);

    heart_frame_count = 0;
    lung_frame_count  = 0;
    next_dma_buf = 1;
>>>>>>> Stashed changes

    if (saadc_init() != 0) return -EIO;
    next_dma_buf = 1;

    /* SUB-BEACON D: yellow — saadc_init() returned OK */
    led_set_yellow();
    k_busy_wait(300000);

    err = nrfx_saadc_buffer_set(ping_pong[0], HALF_BUF_SAMPLES);
    if (err != 0) return -EIO;

    /* SUB-BEACON E: purple — buffer_set returned OK */
    led_set_purple();
    k_busy_wait(300000);

    err = nrfx_saadc_mode_trigger();
    if (err != 0) return -EIO;

    /* SUB-BEACON F: cyan — mode_trigger returned OK */
    led_set_cyan();
    k_busy_wait(300000);

    LOG_INF("SAADC streaming started @ %d Hz", SAMPLING_RATE);
    return 0;
}

static void saadc_stop_streaming(void)
{
    if (saadc_was_initialized) {
        nrfx_saadc_uninit();
        saadc_was_initialized = false;
    }
    LOG_INF("SAADC stopped (overruns=%u, ble_drops=%u, sd_drops=%u)",
<<<<<<< Updated upstream
            saadc_dma_overruns,
            (uint32_t)atomic_get(&ring_drops)
=======
#if BLE_AUDIO_LIVE
            (uint32_t)atomic_get(&ring_drops)
#else
            0U
#endif

>>>>>>> Stashed changes
#if USE_SD
            , (uint32_t)atomic_get(&sd_ring_drops)
#else
            , 0U
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
                mtu, nus_chunk_size, nus_chunk_size - CHUNK_HEADER_BYTES);
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
        .interval_min = 6, .interval_max = 12, .latency = 0, .timeout = 400,
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
 * ══════════════════════════════════════════════════════════════════ */
static void received(struct bt_conn *conn, const void *data, uint16_t len, void *ctx)
{
    ARG_UNUSED(conn); ARG_UNUSED(ctx);
    if (len == 3 && memcmp(data, "REC", 3) == 0) {
        start_recording = true;
    }
}

static struct bt_nus_cb nus_listener = { .received = received };

/* ══════════════════════════════════════════════════════════════════
 * BLE TX THREAD
 * ══════════════════════════════════════════════════════════════════ */
static void ble_tx_thread_fn(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);
    uint8_t chunk[251];
    bool    finished_sent = false;

    while (true) {
        k_sem_take(&audio_data_sem, K_FOREVER);

        if (analog_recording) {
            finished_sent = false;
        }

        while (true) {
            uint16_t payload_bytes = nus_chunk_size - CHUNK_HEADER_BYTES;
            if (payload_bytes < 2) { k_sleep(K_MSEC(5)); break; }

            uint32_t available = ring_buf_size_get(&audio_ring);

            if (available == 0) {
                if (!analog_recording && !finished_sent) {
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
            sys_put_le16(tx_seq,      &chunk[0]);
            sys_put_le16(send_bytes,  &chunk[2]);
            ring_buf_get(&audio_ring, &chunk[4], send_bytes);

            int err;
            do {
                err = bt_nus_send(NULL, chunk, CHUNK_HEADER_BYTES + send_bytes);
                if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(2));
            } while (err == -ENOMEM || err == -EAGAIN);

            tx_seq = (tx_seq + 1) & 0xFFFF;
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 * MFCC FRAME CALLBACK
 * ══════════════════════════════════════════════════════════════════ */
static int s_mfcc_n_frames_total = MFCC_N_FRAMES;

static void on_mfcc_frame(int frame_idx, const float *coeffs)
{
    int err;

    if (frame_idx == 0) {
        char hdr[48];
        int hdr_len = snprintf(hdr, sizeof(hdr),
                               "MFCC_START:%d:%d\n",
                               s_mfcc_n_frames_total, MFCC_N_MFCC);
        do {
            err = bt_nus_send(NULL, hdr, hdr_len);
            if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(5));
        } while (err == -ENOMEM || err == -EAGAIN);

        k_sleep(K_MSEC(20));
    }

    uint8_t  pkt[CHUNK_HEADER_BYTES + MFCC_N_MFCC * sizeof(float)];
    uint16_t payload = MFCC_N_MFCC * sizeof(float);

    sys_put_le16((uint16_t)frame_idx, &pkt[0]);
    sys_put_le16(payload,              &pkt[2]);
    memcpy(&pkt[4], coeffs, payload);

    do {
        err = bt_nus_send(NULL, pkt, sizeof(pkt));
        if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(2));
    } while (err == -ENOMEM || err == -EAGAIN);
}

/* ══════════════════════════════════════════════════════════════════
 * RECORD AND STREAM
 *
 * USE_SD true:
 *   1. Open /SD:/analog.pcm
 *   2. Give sd_data_sem ONCE to wake the SD writer thread
 *   3. SAADC runs — ISR fills both sd_ring and audio_ring in parallel
 *   4. BLE TX thread streams audio live over NUS
 *   5. After total_halves: stop SAADC, set analog_recording=false
 *   6. Wait for SD writer thread to flush, append checksum, close file
 *   7. Stream MFCC over NUS
 *
 * USE_SD false:
 *   BLE stream only — no SD involvement.
 * ══════════════════════════════════════════════════════════════════ */
static void record_and_stream(void)
{
    /* DEBUG BEACON: record_and_stream entered */
    led_set_purple();
    k_busy_wait(200000);

    const uint32_t total_halves =
        (TOTAL_AUDIO_SAMPLES + HALF_BUF_SAMPLES - 1) / HALF_BUF_SAMPLES;

    /* ── Reset BLE audio ring ── */
    ring_buf_reset(&audio_ring);
    atomic_set(&ring_drops, 0);
    ring_high_water = 0;
    tx_seq          = 0;
    k_sem_reset(&half_produced_sem);
    k_sem_reset(&tx_done_sem);

#if USE_SD
    /* ── Reset SD ring ── */
    ring_buf_reset(&sd_ring);
    atomic_set(&sd_ring_drops, 0);
    sd_ring_high_water = 0;
    k_sem_reset(&sd_done_sem);

    if (!sd_mounted) {
        LOG_ERR("SD not mounted — aborting REC");
        bt_nus_send(NULL, "ERR:NOSD", 8);
        led_error_flash(led_set_yellow);
        return;
    }

<<<<<<< Updated upstream
    /* Delete old file, open new one */
    fs_unlink(AUDIO_FILE_PATH);
    fs_file_t_init(&sd_file);
    if (fs_open(&sd_file, AUDIO_FILE_PATH,
                FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC) < 0) {
        LOG_ERR("Cannot open SD file for recording");
        bt_nus_send(NULL, "ERR:SD_OPEN", 11);
        led_error_flash(led_set_yellow);
        return;
=======
    /* v7.5: silent unlink — -ENOENT on a fresh card is harmless and
     * was generating four <err> log lines per REC. We only care if
     * an *existing* file fails to delete. */
    {
        const char *cleanup_paths[] = {
            AUDIO_FILE_PATH,
            HEART_MFCC_FILE_PATH,
            LUNG_MFCC_FILE_PATH,
            HR_RESULT_FILE_PATH,
            RR_RESULT_FILE_PATH,
        };
        for (size_t i = 0; i < ARRAY_SIZE(cleanup_paths); i++) {
            int rc = fs_unlink(cleanup_paths[i]);
            if (rc < 0 && rc != -ENOENT) {
                LOG_WRN("unlink(%s): %d", cleanup_paths[i], rc);
            }
        }
>>>>>>> Stashed changes
    }

    /*
     * Wake the SD writer thread exactly once.
     * The writer loops internally on ring_buf_size_get() — it does not
     * need a semaphore signal per half-buffer.
     */
    k_sem_give(&sd_data_sem);
#endif

<<<<<<< Updated upstream
    /* Send audio length to host so it can pre-allocate */
    char hdr[32];
    snprintf(hdr, sizeof(hdr), "START:%u\n", (uint32_t)TOTAL_AUDIO_BYTES);
    bt_nus_send(NULL, hdr, strlen(hdr));
    k_sleep(K_MSEC(10));
=======
    /* DEBUG BEACON: about to fs_open audio */
    led_set_blue();
    k_busy_wait(200000);

    int rc_audio = fs_open(&sd_audio_file, AUDIO_FILE_PATH,
                            FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc_audio < 0) {
        LOG_ERR("fs_open(%s) failed: %d", AUDIO_FILE_PATH, rc_audio);
        bt_nus_send(NULL, "ERR:SD_OPEN", 11);
        led_error_flash(led_set_yellow);
        return;
    }

    /* v7.5 [FIX 14]: pre-allocate the full audio file so FATFS
     * allocates all clusters up front. Subsequent writes don't trigger
     * FAT-table updates mid-recording — biggest single contributor
     * to write-stall and the sd_drops issue we saw on the first REC. */
    {
        int rc_tr = fs_truncate(&sd_audio_file,
                                (off_t)(TOTAL_AUDIO_BYTES + CHECKSUM_SIZE));
        if (rc_tr < 0) {
            LOG_WRN("fs_truncate(audio.pcm) failed: %d (non-fatal)", rc_tr);
        } else {
            /* Seek back to start; truncate leaves position at end on
             * some FATFS versions. */
            fs_seek(&sd_audio_file, 0, FS_SEEK_SET);
            LOG_INF("audio.pcm pre-allocated to %u B",
                    (unsigned)(TOTAL_AUDIO_BYTES + CHECKSUM_SIZE));
        }
    }

    /* DEBUG BEACON: fs_open + fs_truncate done */
    led_set_yellow();
    k_busy_wait(200000);

#if !DSP_OFFLINE
    int rc_heart = fs_open(&sd_heart_file, HEART_MFCC_FILE_PATH,
                            FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc_heart < 0) {
        LOG_ERR("fs_open(%s) failed: %d", HEART_MFCC_FILE_PATH, rc_heart);
        fs_close(&sd_audio_file);
        bt_nus_send(NULL, "ERR:SD_OPEN", 11);
        led_error_flash(led_set_yellow);
        return;
    }

    int rc_lung = fs_open(&sd_lung_file, LUNG_MFCC_FILE_PATH,
                           FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc_lung < 0) {
        LOG_ERR("fs_open(%s) failed: %d", LUNG_MFCC_FILE_PATH, rc_lung);
        fs_close(&sd_audio_file);
        fs_close(&sd_heart_file);
        bt_nus_send(NULL, "ERR:SD_OPEN", 11);
        led_error_flash(led_set_yellow);
        return;
    }
#endif /* !DSP_OFFLINE */
#endif /* USE_SD */
>>>>>>> Stashed changes

    led_set_cyan();
    k_busy_wait(200000);   /* DEBUG BEACON: cyan visible */
    analog_recording = true;

    if (saadc_start_streaming() != 0) {
        analog_recording = false;
#if USE_SD
        fs_close(&sd_file);
#endif
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:SAADC", 9);
        return;
    }

<<<<<<< Updated upstream
    LOG_INF("Recording %d s @ %d Hz (USE_SD=%s)...", DURATION_S, SAMPLING_RATE,
            USE_SD ? "true" : "false");

    /* Count half-buffers */
=======
    /* DEBUG BEACON: SAADC started OK, about to wait for samples */
    led_set_white();
    k_busy_wait(200000);

>>>>>>> Stashed changes
    for (uint32_t h = 0; h < total_halves; h++) {
        if (k_sem_take(&half_produced_sem, K_MSEC(200)) != 0) {
            LOG_ERR("SAADC timeout at half-buffer %u/%u", h, total_halves);
            analog_recording = false;
            saadc_stop_streaming();
            k_sem_give(&audio_data_sem);
            k_sem_take(&tx_done_sem, K_MSEC(10000));
            bt_nus_send(NULL, "ERR:TIMEOUT", 11);
            return;
        }
    }

    saadc_stop_streaming();
    analog_recording = false;

<<<<<<< Updated upstream
    /* Wake BLE TX thread to drain remaining audio_ring and send "finished\n" */
    k_sem_give(&audio_data_sem);

    /* Wait for BLE audio stream to complete */
    if (k_sem_take(&tx_done_sem, K_MSEC(10000)) != 0) {
        LOG_WRN("BLE TX did not finish within 10 s");
=======
    /* DEBUG BEACON: Phase 1 SAADC loop completed */
    led_set_green();
    k_busy_wait(500000);

#if !DSP_OFFLINE
    if (k_sem_take(&dsp_done_sem, K_MSEC(30000)) != 0) {
        LOG_WRN("DSP thread did not finish within 30 s");
>>>>>>> Stashed changes
    }

#if USE_SD
    /*
     * SD writer thread detects analog_recording==false, drains the
     * remaining sd_ring, flushes the tail, appends checksum, closes
     * the file, then signals sd_done_sem.
     */
    LOG_INF("Waiting for SD writer to flush and close...");
    if (k_sem_take(&sd_done_sem, K_MSEC(15000)) != 0) {
        LOG_ERR("SD writer did not finish within 15 s — file may be truncated");
    } else {
        LOG_INF("SD file closed: %s (checksum=0x%08X)", AUDIO_FILE_PATH, sd_checksum);
        bt_nus_send(NULL, "SD:OK\n", 6);
    }
#endif

    LOG_INF("BLE audio stream complete — starting MFCC");
    led_set_purple();
<<<<<<< Updated upstream
=======
    report_ram_usage("after Phase 2 MFCC");
>>>>>>> Stashed changes

    /* ── MFCC stream ── */
    s_mfcc_n_frames_total = MFCC_N_FRAMES;
    int n_frames = dsp_mfcc_finish(on_mfcc_frame);

    if (n_frames <= 0) {
        LOG_ERR("dsp_mfcc_finish failed: %d", n_frames);
        bt_nus_send(NULL, "ERR:DSP", 7);
        led_error_flash(led_set_yellow);
        led_set_green();
        return;
    }

    k_sleep(K_MSEC(20));
    int err;
    do {
        err = bt_nus_send(NULL, "MFCC_END\n", 9);
        if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(5));
    } while (err == -ENOMEM || err == -EAGAIN);

    LOG_INF("MFCC stream complete: %d frames x %d coeffs", n_frames, MFCC_N_MFCC);
    led_set_green();
    LOG_INF("record_and_stream() complete.");
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

    if (dsp_mfcc_init() != 0) {
        LOG_ERR("dsp_mfcc_init failed");
        led_error_flash(led_set_yellow);
        return -1;
    }

    err = saadc_init();
    if (err < 0) {
        LOG_ERR("SAADC init failed: %d", err);
        led_error_flash(led_set_yellow);
        return err;
    }

    err = bt_enable(NULL);
    if (err) { LOG_ERR("bt_enable: %d", err); return err; }

    err = bt_nus_cb_register(&nus_listener, NULL);
    if (err) { LOG_ERR("bt_nus_cb_register: %d", err); return err; }

    err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                          sd_adv, ARRAY_SIZE(sd_adv));
    if (err) { LOG_ERR("bt_le_adv_start: %d", err); return err; }

    /* BLE TX thread */
    k_thread_create(&ble_tx_thread_data, ble_tx_stack,
                    K_THREAD_STACK_SIZEOF(ble_tx_stack),
                    ble_tx_thread_fn, NULL, NULL, NULL,
                    BLE_TX_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&ble_tx_thread_data, "ble_tx");

#if USE_SD
    /* SD writer thread — starts blocked on sd_data_sem */
    k_thread_create(&sd_writer_thread_data, sd_writer_stack,
                    K_THREAD_STACK_SIZEOF(sd_writer_stack),
                    sd_writer_thread_fn, NULL, NULL, NULL,
                    SD_WRITER_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&sd_writer_thread_data, "sd_writer");

    /* SD card init — non-fatal if card absent */
    if (init_sd_card() != 0) {
        LOG_ERR("SD card init failed — REC will return ERR:NOSD");
        LOG_ERR("  CS      — overlay: xiao_d pin 1; verify physical wire");
        LOG_ERR("  Format  — must be FAT32; exFAT needs CONFIG_FS_FATFS_EXFAT=y");
        LOG_ERR("  Power   — SD module needs stable 3.3 V");
        LOG_ERR("  SPI     — D8/D9/D10 wired to SCK/MISO/MOSI?");
        led_error_flash(led_set_yellow);
        /* sd_mounted stays false — ERR:NOSD sent on first REC */
    }
#endif

    led_set_red();
<<<<<<< Updated upstream
    LOG_INF("AcoustEEEcare v6.6 ready (USE_SD=%s) — waiting for BLE connection",
            USE_SD ? "true" : "false");
=======
    LOG_INF("AcoustEEEcare v7.5 ready — waiting for BLE connection");

    /* v7.5 [FIX 15]: baseline RAM snapshot, before any recording.
     * Compare subsequent snapshots against this to spot leaks/overflows. */
    report_ram_usage("boot (idle)");
>>>>>>> Stashed changes

    while (true) {
        k_sleep(K_MSEC(100));
        if (is_connected && mtu_exchanged && start_recording && !analog_recording) {
            start_recording = false;
            LOG_INF("REC command — starting %d s @ %d Hz", DURATION_S, SAMPLING_RATE);
            record_and_stream();
            /* v7.5 [FIX 15]: post-REC snapshot. If this drifts upward
             * across successive RECs, there's a leak somewhere. */
            report_ram_usage("after REC completed");
        }
    }

    return 0;
}

/*
 * ════════════════════════════════════════════════════════════════════
 * REQUIRED prj.conf ADDITIONS for USE_SD true
 * ════════════════════════════════════════════════════════════════════
 *
 * CONFIG_SPI=y
 * CONFIG_DISK_ACCESS=y
 * CONFIG_DISK_DRIVER_SDMMC=y
 * CONFIG_FAT_FILESYSTEM_ELM=y
 * CONFIG_FILE_SYSTEM=y
 * CONFIG_FILE_SYSTEM_MAX_TYPES=2
 * CONFIG_HEAP_MEM_POOL_SIZE=8192
 * CONFIG_MAIN_STACK_SIZE=4096
 *
 * ════════════════════════════════════════════════════════════════════
 * REQUIRED devicetree overlay  (e.g. boards/xiao_ble.overlay)
 * ════════════════════════════════════════════════════════════════════
 *
 * &spi2 {
 *     status = "okay";
 *     cs-gpios = <&gpio0 28 GPIO_ACTIVE_LOW>;   // adjust to your CS pin
 *
 *     sdhc: sdhc@0 {
 *         compatible = "zephyr,sdhc-spi-slot";
 *         reg = <0>;
 *         status = "okay";
 *         spi-max-frequency = <4000000>;          // 4 MHz — safe for all cards
 *         disk {
 *             compatible = "zephyr,sdmmc-disk";
 *             status = "okay";
 *         };
 *     };
 * };
 *
 * ════════════════════════════════════════════════════════════════════
 * MEMORY BUDGET (approximate, USE_SD true)
 * ════════════════════════════════════════════════════════════════════
 *
 *   Zephyr kernel + BLE stack     ~90 KB
 *   s_decimated[20000] (dsp_mfcc)  40 KB
 *   DSP scratch                     ~7 KB
 *   audio_ring  (BLE)               16 KB
 *   sd_ring     (SD writer)         32 KB
 *   ping_pong[2][512]                2 KB
 *   BLE TX thread stack              2 KB
 *   SD writer thread stack           2 KB
 *   sd_write_buf                   512  B
 *   fat_fs (FATFS work area)        ~4 KB  (sized by ELM FatFs internally)
 *   Remaining headroom             ~61 KB
 */