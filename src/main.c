/*
 * AcoustEEEcare — SAADC BLE + SD Card + TFLite Micro Edition
 * ============================================================
 * v7.9 — Abandon pre-alloc overwrite; use FS_O_TRUNC on every REC
 *
 * CHANGES FROM v7.8:
 *
 *   [Option A] fs_open for audio.pcm now uses FS_O_CREATE|FS_O_WRITE|FS_O_TRUNC
 *              on every recording, replacing the v7.7 pre-allocate-then-overwrite
 *              strategy. Opening without TRUNC + fs_seek(0) left FATFS writing
 *              directory metadata, keeping the card busy when the first real
 *              fs_write arrived — causing -EIO on every write regardless of delay.
 *              TRUNC gives FATFS a clean state. The ~50 ms first-write erase is
 *              absorbed by the 32 KB ring (~2 s headroom at 16 KB/s).
 *
 * CHANGES FROM v7.7 (preserved from v7.8):
 *
 *   [FIX 1] SAADC_IRQ_PRIORITY: 6 -> 5.
 *           Priority 7 (tried first) exceeded IRQ_PRIO_LOWEST on
 *           nRF52840 and failed to compile. Raised to 5 instead —
 *           higher priority means the SAADC ISR completes faster and
 *           yields the CPU sooner, reducing the window where it can
 *           interfere with SPI bus timing during SD card writes.
 *
 *   [FIX 2] Pre-write k_sleep: 2 ms -> 50 ms.
 *           Retry backoff k_sleep: 10 ms -> 50 ms.
 *           The Zephyr #52931 reporter found a printk (~10-20 ms) was
 *           sufficient to let the card finish its internal program
 *           cycle. 2 ms was clearly not enough for this card. 50 ms
 *           is safe: at 16 KB/s audio (one 512-byte write per 32 ms),
 *           the 32 KB ring holds ~2 s of headroom. Total worst-case
 *           write latency (50 ms sleep + 3 x 50 ms retry) = 200 ms,
 *           well within the ring's capacity.
 *
 *   [FIX 3] 100 ms settle delay at the start of the SD writer loop.
 *           (Kept for safety; rationale partially superseded by v7.9
 *           Option A which removes the fs_seek that triggered the
 *           metadata write in the first place.)
 *
 * ALL PREVIOUS FIXES (v7.7 / v7.6 / v7.5 / v7.4 / v7.3 / v7.2 /
 * v7.1 / v6.8) ARE PRESERVED.
 * ============================================================
 */


#define USE_SD  true

#define BLE_AUDIO_LIVE  0   /* must stay 0 for arena RAM to be available */
#define DSP_OFFLINE     1   /* must stay 1 for arena RAM to be available */

/* ── Model enable flags ─────────────────────────────────────────── */
/* ENABLE_HEART_MODEL and ENABLE_LUNG_MODEL are now defined globally
 * in CMakeLists.txt via target_compile_definitions, so they apply
 * to main.c AND tflm_inference.cc consistently. */

/* ══════════════════════════════════════════════════════════════════
 * INCLUDES
 * ══════════════════════════════════════════════════════════════════ */
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

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
#include "heart_mfcc_config.h"
#include "lung_mfcc_config.h"

#include "tflm_inference.h"

#if USE_SD
#include <zephyr/fs/fs.h>
#include <zephyr/storage/disk_access.h>
#include <ff.h>
#endif

LOG_MODULE_REGISTER(AcoustEEEcare);

/* ══════════════════════════════════════════════════════════════════
 * FORWARD DECLARATIONS
 *
 * The LED helpers (led_set_*, led_on, led_off) and the LED
 * gpio_dt_spec variables are defined further down. We forward-declare
 * them up here so earlier functions can use them.
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

static uint8_t tensor_arena[TENSOR_ARENA_BYTES] __aligned(16);

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
 * SAADC CONFIG
 * ══════════════════════════════════════════════════════════════════ */
#define SAADC_CC_VALUE      2000U
#define SAADC_IRQ_PRIORITY  5   /* v7.8 FIX 1: was 6 (same as SD_WRITER_PRIORITY).
                                 * Priority 7 exceeded IRQ_PRIO_LOWEST on nRF52840.
                                 * Raised to 5 instead — higher priority means the
                                 * SAADC ISR completes faster and releases CPU sooner,
                                 * reducing the window where it can interfere with the
                                 * SPI bus timing during SD writes. */

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
 * AUDIO PARAMS
 * ══════════════════════════════════════════════════════════════════ */
#define SAMPLING_RATE        8000
#define DURATION_S           10
#define TOTAL_AUDIO_SAMPLES  (SAMPLING_RATE * DURATION_S)
#define TOTAL_AUDIO_BYTES    (TOTAL_AUDIO_SAMPLES * sizeof(int16_t))

#define HALF_BUF_SAMPLES     512
#define HALF_BUF_BYTES       (HALF_BUF_SAMPLES * sizeof(int16_t))

/* [FIX 6] Ping-pong buffers passed to SAADC EasyDMA must be aligned. */
static int16_t ping_pong[2][HALF_BUF_SAMPLES] __aligned(4);
static volatile uint8_t  next_dma_buf     = 1;
static volatile bool     analog_recording = false;
static uint16_t          nus_chunk_size   = 244;

/* ══════════════════════════════════════════════════════════════════
 * BLE AUDIO RING BUFFER + TX THREAD
 * ══════════════════════════════════════════════════════════════════ */
static K_SEM_DEFINE(half_produced_sem, 0, K_SEM_MAX_LIMIT);

#if BLE_AUDIO_LIVE
#define AUDIO_RING_BYTES  (16 * 1024)
RING_BUF_DECLARE(audio_ring, AUDIO_RING_BYTES);

static K_SEM_DEFINE(audio_data_sem,    0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(tx_done_sem,       0, 1);

static atomic_t  ring_drops;
static uint32_t  ring_high_water = 0;

static uint16_t tx_seq = 0;
#endif /* BLE_AUDIO_LIVE */

#define CHUNK_HEADER_BYTES  4

/* ══════════════════════════════════════════════════════════════════
 * DSP RING BUFFER + DSP THREAD
 * ══════════════════════════════════════════════════════════════════ */
#if !DSP_OFFLINE
#define DSP_RING_BYTES  (32 * 1024)
RING_BUF_DECLARE(dsp_ring, DSP_RING_BYTES);

static K_SEM_DEFINE(dsp_data_sem, 0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(dsp_done_sem, 0, 1);

static int16_t dsp_pop_buf[HALF_BUF_SAMPLES] __aligned(4);
#endif /* !DSP_OFFLINE */

/* ══════════════════════════════════════════════════════════════════
 * SD CARD DATA DECLARATIONS
 * ══════════════════════════════════════════════════════════════════ */
#if USE_SD

#define SD_CARD_MOUNT_POINT  "/SD:"
#define AUDIO_FILE_PATH      "/SD:/analog.pcm"
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

static K_SEM_DEFINE(sd_data_sem, 0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(sd_done_sem, 0, 1);

#define SD_WRITER_PRIORITY    6
static K_THREAD_STACK_DEFINE(sd_writer_stack, SD_WRITER_STACK_SIZE);
static void sd_writer_thread_fn(void *a, void *b, void *c);

static FATFS             fat_fs;
static struct fs_mount_t mp = {
    .type      = FS_FATFS,
    .mnt_point = SD_CARD_MOUNT_POINT,
    .fs_data   = &fat_fs,
};
static bool             sd_mounted   = false;
static struct fs_file_t sd_audio_file;
#if !DSP_OFFLINE
static struct fs_file_t sd_heart_file;
static struct fs_file_t sd_lung_file;
#endif
static uint32_t         sd_checksum  = 0;

/* [FIX 1] sd_write_buf forced to 4-byte alignment for nRF52 EasyDMA.
 * v7.6 FIX E: dropped back to 512 (one disk sector) after the field
 * log showed "Only 1 blocks of 1 were written / Write failed / fs:
 * file write error (-5)" on the FIRST write of a 4 KB block. Cheap
 * SDHC cards on hand-wired breadboards stall during internal
 * wear-leveling and can fail multi-sector writes at 8 MHz SPI; the
 * card responded fine to the 28-byte probe write but rejected the
 * 4 KB production write. 512 B writes are atomic from the card's
 * perspective and survive the stalls. FATFS bookkeeping overhead
 * is higher per-write but the 32 KB ring absorbs it.
 *
 * v7.5 [FIX 12]: was 512 -> 4096 (one FAT cluster on typical SDs).
 * Cuts per-write FATFS bookkeeping ~8x. Combined with FIX 14 file
 * pre-truncation, this should eliminate sd_drops entirely. */
#define SD_WRITE_BUF_SIZE  512
static uint8_t sd_write_buf[SD_WRITE_BUF_SIZE] __aligned(4);

static uint32_t compute_checksum(const uint8_t *data, uint32_t len)
{
    uint32_t cs = 0;
    for (uint32_t i = 0; i < len; i++) cs ^= data[i];
    return cs;
}

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

    /* ── Step 7: pre-allocate analog.pcm ────────────────────────────
     * Write TOTAL_AUDIO_BYTES + CHECKSUM_SIZE bytes of zeros now, at
     * boot, so the SD card erases all required flash blocks before any
     * recording starts.  During recording the SD writer overwrites
     * these pre-allocated clusters — no block erase is triggered, so
     * every 512-byte write completes in ~1–5 ms instead of hanging the
     * SPI bus for ~10 s (the root cause of audio=0 B / sd_drops=126).
     *
     * zero_sector lives in BSS (static const → zero-initialised by the
     * C runtime), so it costs no extra RAM beyond what is already used.
     */
    LOG_INF("[7/7] Pre-allocating %s (%u B) — this may take a few seconds ...",
            AUDIO_FILE_PATH,
            (uint32_t)TOTAL_AUDIO_BYTES + (uint32_t)CHECKSUM_SIZE);
    {
        static const uint8_t zero_sector[SD_WRITE_BUF_SIZE]; /* BSS = 0 */
        struct fs_file_t pa;
        fs_file_t_init(&pa);

        ret = fs_open(&pa, AUDIO_FILE_PATH, FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
        if (ret != 0) {
            LOG_WRN("[7/7] pre-alloc fs_open failed (%d) — "
                    "recording writes may be slow", ret);
        } else {
            uint32_t remaining =
                (uint32_t)TOTAL_AUDIO_BYTES + (uint32_t)CHECKSUM_SIZE;
            bool pa_ok = true;

            while (remaining > 0 && pa_ok) {
                uint32_t n = MIN(remaining, (uint32_t)SD_WRITE_BUF_SIZE);
                ssize_t pw = fs_write(&pa, zero_sector, n);
                if (pw < 0) {
                    LOG_WRN("[7/7] pre-alloc write stalled at offset %u (%d) "
                            "— skipping remainder",
                            (uint32_t)TOTAL_AUDIO_BYTES +
                            (uint32_t)CHECKSUM_SIZE - remaining,
                            (int)pw);
                    pa_ok = false;
                } else {
                    remaining -= (uint32_t)pw;
                }
                /* Yield so BLE stack and logging stay responsive. */
                k_yield();
            }

            fs_close(&pa);

            if (pa_ok) {
                LOG_INF("[7/7] pre-alloc OK — %u B written to %s",
                        (uint32_t)TOTAL_AUDIO_BYTES + (uint32_t)CHECKSUM_SIZE,
                        AUDIO_FILE_PATH);
            }
        }
    }

    return 0;
}

/* ── SD writer thread ─────────────────────────────────────────── */
static void sd_writer_thread_fn(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (true) {
        k_sem_take(&sd_data_sem, K_FOREVER);

        if (!sd_mounted) {
            LOG_ERR("SD writer: not mounted, discarding data");
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
            }
            k_sem_give(&sd_done_sem);
            continue;
        }

        LOG_INF("SD writer: starting write loop");
        k_sleep(K_MSEC(100)); /* v7.8 FIX 3: after fs_open+fs_seek(0) on the
                                * pre-allocated file, FATFS writes FAT metadata
                                * (timestamps, dir entry). The card may still be
                                * busy from that update when the first fs_write
                                * hits. 100 ms guarantees the card has finished
                                * before we touch it. */
        uint32_t audio_written = 0;
#if !DSP_OFFLINE
        uint32_t heart_written = 0;
        uint32_t lung_written  = 0;
#endif
        sd_checksum = 0;
        bool write_error = false;   /* [FIX 4] sticky abort flag */

        while (!write_error &&
               (analog_recording ||
                ring_buf_size_get(&audio_sd_ring) > 0
#if !DSP_OFFLINE
                || ring_buf_size_get(&heart_mfcc_ring) > 0
                || ring_buf_size_get(&lung_mfcc_ring)  > 0
#endif
               )) {

            bool did_work = false;

            /* ── Audio ring (always present) ── */
            {
                uint32_t avail = ring_buf_size_get(&audio_sd_ring);
                bool done = !analog_recording;
                if (avail >= SD_WRITE_BUF_SIZE || (done && avail > 0)) {
                    uint32_t n = MIN(avail, (uint32_t)SD_WRITE_BUF_SIZE);
                    ring_buf_get(&audio_sd_ring, sd_write_buf, n);
                    sd_checksum ^= compute_checksum(sd_write_buf, n);
                    /* v7.6 FIX G: workaround for Zephyr SD-over-SPI issue
                     * #52931. The Zephyr SD subsystem polls card status
                     * immediately after a write, but on some cards the
                     * card hasn't actually finished its internal program
                     * cycle and returns "still busy" — which the subsys
                     * misreads as -EIO. The reporter found that simply
                     * inserting a delay before fs_write (they used a
                     * printk) made the bug disappear. A 2 ms k_sleep is
                     * cheap and lets the card finish the previous block.
                     * At 8 kHz int16 audio (16 KB/s) and 512-byte writes
                     * (one write every 32 ms), 2 ms is 6% overhead — far
                     * less than the retry latency we were eating. */
                    k_sleep(K_MSEC(50)); /* v7.8 FIX 2: was 2 ms — not enough for
                                          * this card's internal program cycle.
                                          * Zephyr #52931 reporter needed ~10-20 ms
                                          * (a printk); 50 ms is safe given the
                                          * 32 KB ring (~2 s headroom at 16 KB/s). */
                    /* v7.6 FIX F: retry transient -EIO before bailing. */
                    ssize_t wr = -1;
                    for (int retry = 0; retry < 3; retry++) {
                        wr = fs_write(&sd_audio_file, sd_write_buf, n);
                        if (wr >= 0) break;
                        LOG_WRN("SD audio fs_write transient err: %d "
                                "(retry %d/3)", (int)wr, retry + 1);
                        k_sleep(K_MSEC(50)); /* v7.8 FIX 2b: was 10 ms */
                    }
                    if (wr < 0) {
                        LOG_ERR("SD audio fs_write failed after retries: %d "
                                "— aborting writer", (int)wr);
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
                k_sleep(K_MSEC(1));
            }
        }

        /* Append checksum (best-effort; ignore error if card is dead). */
        if (!write_error) {
            ssize_t cw = fs_write(&sd_audio_file, &sd_checksum, CHECKSUM_SIZE);
            if (cw < 0) {
                LOG_WRN("SD audio checksum write failed: %d", (int)cw);
            }
        }
        fs_close(&sd_audio_file);
#if !DSP_OFFLINE
        fs_close(&sd_heart_file);
        fs_close(&sd_lung_file);
#endif

#if !DSP_OFFLINE
        LOG_INF("SD writer done: audio=%u B  heart=%u B  lung=%u B  err=%d",
                audio_written, heart_written, lung_written, (int)write_error);
        LOG_INF("Ring high-water: audio=%u  heart=%u  lung=%u",
                audio_ring_high_water, heart_ring_high_water, lung_ring_high_water);
#else
        LOG_INF("SD writer done: audio=%u B  err=%d (offline DSP -> MFCC later)",
                audio_written, (int)write_error);
        LOG_INF("Ring high-water: audio=%u", audio_ring_high_water);
#endif

        k_sem_give(&sd_done_sem);
    }
}

#if DSP_OFFLINE
/* ══════════════════════════════════════════════════════════════════
 * OFFLINE MFCC PROCESSING (Phase 2)
 * ══════════════════════════════════════════════════════════════════ */
static int process_audio_offline(int *out_heart_frames, int *out_lung_frames)
{
    int rc = 0;
    struct fs_file_t fa;
    struct fs_file_t fh;
    struct fs_file_t fl;

    fs_file_t_init(&fa);
    fs_file_t_init(&fh);
    fs_file_t_init(&fl);

    LOG_INF("Offline MFCC: opening files...");

    rc = fs_open(&fa, AUDIO_FILE_PATH, FS_O_READ);
    if (rc < 0) {
        LOG_ERR("Offline: fs_open(%s) failed: %d", AUDIO_FILE_PATH, rc);
        return rc;
    }

    rc = fs_open(&fh, HEART_MFCC_FILE_PATH,
                 FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc < 0) {
        LOG_ERR("Offline: fs_open(%s) failed: %d", HEART_MFCC_FILE_PATH, rc);
        fs_close(&fa);
        return rc;
    }

    rc = fs_open(&fl, LUNG_MFCC_FILE_PATH,
                 FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc < 0) {
        LOG_ERR("Offline: fs_open(%s) failed: %d", LUNG_MFCC_FILE_PATH, rc);
        fs_close(&fa);
        fs_close(&fh);
        return rc;
    }

    dsp_mfcc_reset(&heart_pipeline);
    dsp_mfcc_reset(&lung_pipeline);
    heart_frame_count = 0;
    lung_frame_count  = 0;

    offline_heart_fp = &fh;
    offline_lung_fp  = &fl;

    /* [FIX 6] 4-byte aligned read buffer for SD/FATFS underlying DMA. */
    int16_t  read_buf[HALF_BUF_SAMPLES] __aligned(4);
    uint32_t bytes_read_total = 0;
    int      chunk_idx        = 0;
    int64_t  t_start          = k_uptime_get();

    const uint32_t audio_payload_bytes = (uint32_t)TOTAL_AUDIO_BYTES;

    /* v7.6 FIX D: aggregate min/max/mean across the whole audio file
     * during offline DSP. If the SD write path silently produced a
     * zero-filled file (the root cause we suspected for HR=321), this
     * will print min=0 max=0 mean=0 and tell you immediately. Healthy
     * mic recording: min/max around ±a few thousand, mean near zero
     * after the bandpass settles. */
    int32_t  diag_sum = 0;
    int16_t  diag_mn  = INT16_MAX;
    int16_t  diag_mx  = INT16_MIN;
    uint32_t diag_n   = 0;

    while (bytes_read_total < audio_payload_bytes) {
        uint32_t want = MIN((uint32_t)sizeof(read_buf),
                            audio_payload_bytes - bytes_read_total);

        ssize_t got = fs_read(&fa, read_buf, want);
        if (got <= 0) {
            LOG_WRN("Offline: short/zero read at byte %u (got=%d) — "
                    "audio file may be truncated",
                    bytes_read_total, (int)got);
            break;
        }
        int samples = (int)(got / sizeof(int16_t));

        /* v7.6 FIX D: accumulate diagnostic stats. */
        for (int i = 0; i < samples; i++) {
            int16_t s = read_buf[i];
            if (s < diag_mn) diag_mn = s;
            if (s > diag_mx) diag_mx = s;
            diag_sum += s;
        }
        diag_n += (uint32_t)samples;

        dsp_mfcc_feed_chunk(&heart_pipeline, read_buf, samples);
        dsp_mfcc_feed_chunk(&lung_pipeline,  read_buf, samples);

        bytes_read_total += (uint32_t)got;
        chunk_idx++;

        if ((chunk_idx & 3) == 0) {
            k_yield();
        }
    }

    /* v7.6 FIX D: report stats. If min=max=0 the audio file is empty —
     * SD write path is broken. If min/max are tiny (<10) the mic input
     * is dead or DC-only. Healthy signal has |samples| in the hundreds
     * to low thousands after bandpass. */
    if (diag_n > 0) {
        LOG_INF("[DIAG] analog.pcm stats: min=%d max=%d mean=%d (n=%u)",
                (int)diag_mn, (int)diag_mx,
                (int)(diag_sum / (int32_t)diag_n), diag_n);
        if (diag_mn == 0 && diag_mx == 0) {
            LOG_ERR("[DIAG] analog.pcm is ALL ZEROS — SD write path failed; "
                    "MFCC will be garbage and model output will saturate");
        }
    }

    int hf = dsp_mfcc_finish(&heart_pipeline);
    int lf = dsp_mfcc_finish(&lung_pipeline);

    offline_heart_fp = NULL;
    offline_lung_fp  = NULL;

    fs_close(&fa);
    fs_close(&fh);
    fs_close(&fl);

    int64_t elapsed_ms = k_uptime_delta(&t_start);
    LOG_INF("Offline MFCC done: %u B audio in %lld ms (%u chunks)",
            bytes_read_total, elapsed_ms, chunk_idx);
    LOG_INF("Offline frames: heart=%u (finish=%d)  lung=%u (finish=%d)",
            heart_frame_count, hf, lung_frame_count, lf);

    if (out_heart_frames) *out_heart_frames = (hf >= 0) ? hf : (int)heart_frame_count;
    if (out_lung_frames)  *out_lung_frames  = (lf >= 0) ? lf : (int)lung_frame_count;

    if (heart_frame_count == 0 && lung_frame_count == 0) {
        LOG_ERR("Offline MFCC produced zero frames — audio file may be empty");
        return -EIO;
    }

    return 0;
}
#endif /* DSP_OFFLINE */

#endif /* USE_SD */

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

#if !DSP_OFFLINE
        uint32_t dsp_written = ring_buf_put(&dsp_ring,
                                            (const uint8_t *)filled_buf,
                                            HALF_BUF_BYTES);
        if (dsp_written != HALF_BUF_BYTES) {
            LOG_WRN_ONCE("dsp_ring overflow — MFCC frames may be lost!");
        }
        k_sem_give(&dsp_data_sem);
#endif

#if USE_SD
        if (analog_recording) {
            uint32_t sd_written = ring_buf_put(&audio_sd_ring,
                                               (const uint8_t *)filled_buf,
                                               HALF_BUF_BYTES);
            if (sd_written != HALF_BUF_BYTES) {
                atomic_inc(&sd_ring_drops);
                LOG_WRN_ONCE("audio_sd_ring overflow — audio will have gaps!");
            } else {
                uint32_t fill = ring_buf_size_get(&audio_sd_ring);
                if (fill > audio_ring_high_water) {
                    audio_ring_high_water = fill;
                }
            }
        }
#endif

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

/* Flashes a color twice with off in between — a unique "I got here"
 * marker that won't be confused with steady-state LED colors set
 * elsewhere in the firmware. */
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

/* Tracks whether nrfx_saadc has been successfully initialized.
 * Prevents nrfx_saadc_uninit() from being called on a fresh peripheral
 * (which hangs waiting for state bits that were never set). */
static bool saadc_was_initialized = false;

static int saadc_init(void)
{
    nrfx_err_t err;

    /* Only uninit if we previously initialized. Calling nrfx_saadc_uninit()
     * on a fresh, never-initialized SAADC peripheral hangs waiting for
     * hardware state bits that were never set in the first place.
     * This is THE root cause of the "stuck on cyan" hang. */
    if (saadc_was_initialized) {
        nrfx_saadc_uninit();
    }

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

    /* Mark as initialized so future REC calls correctly uninit first. */
    saadc_was_initialized = true;

    return 0;
}

static int saadc_start_streaming(void)
{
    nrfx_err_t err;
    saadc_dma_overruns = 0;
    /* v7.6 FIX A: dc_estimate must seed to the ADC midpoint (~2048 for
     * 12-bit single-ended @ VDD/4 ref), NOT 0. Starting at 0 means the
     * IIR HPF needs ~256 samples to converge, and during those samples
     * the bandpass filter sees a huge DC step that rings for hundreds
     * of additional samples. That ringing dominates the MFCC frames at
     * the start of the recording and pushes the model output into
     * saturation (HR=321 = output_q=127, the int8 ceiling).
     * Matches the value used in the known-good BLE-only firmware. */
    dc_estimate        = 2048;

    dsp_mfcc_reset(&heart_pipeline);
    dsp_mfcc_reset(&lung_pipeline);
    heart_frame_count = 0;
    lung_frame_count  = 0;

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
    LOG_INF("SAADC stopped (overruns=%u, ble_drops=%u, sd_drops=%u)",
            saadc_dma_overruns,
#if BLE_AUDIO_LIVE
            (uint32_t)atomic_get(&ring_drops)
#else
            0U
#endif
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

#if !BLE_AUDIO_LIVE
    while (true) {
        k_sleep(K_FOREVER);
    }
#else
    uint8_t chunk[251] __aligned(4);
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
#endif /* BLE_AUDIO_LIVE */
}

/* ══════════════════════════════════════════════════════════════════
 * RECORD AND STREAM
 * ══════════════════════════════════════════════════════════════════ */
static void record_and_stream(void)
{
    /* DEBUG BEACON: record_and_stream entered */
    led_set_purple();
    k_busy_wait(200000);

    const uint32_t total_halves =
        (TOTAL_AUDIO_SAMPLES + HALF_BUF_SAMPLES - 1) / HALF_BUF_SAMPLES;

    k_sem_reset(&half_produced_sem);

#if BLE_AUDIO_LIVE
    ring_buf_reset(&audio_ring);
    atomic_set(&ring_drops, 0);
    ring_high_water = 0;
    tx_seq          = 0;
    k_sem_reset(&tx_done_sem);
#endif

#if !DSP_OFFLINE
    ring_buf_reset(&dsp_ring);
    k_sem_reset(&dsp_done_sem);
#endif

#if USE_SD
    ring_buf_reset(&audio_sd_ring);
    atomic_set(&sd_ring_drops, 0);
    audio_ring_high_water = 0;
#if !DSP_OFFLINE
    ring_buf_reset(&heart_mfcc_ring);
    ring_buf_reset(&lung_mfcc_ring);
    heart_ring_high_water = 0;
    lung_ring_high_water  = 0;
#endif
    k_sem_reset(&sd_done_sem);

    if (!sd_mounted) {
        LOG_ERR("SD not mounted — aborting REC");
        bt_nus_send(NULL, "ERR:NOSD", 8);
        led_error_flash(led_set_yellow);
        return;
    }

    /* v7.5: silent unlink — -ENOENT on a fresh card is harmless and
     * was generating four <err> log lines per REC. We only care if
     * an *existing* file fails to delete. */
    {
        /* Do NOT unlink AUDIO_FILE_PATH — it was pre-allocated at boot.
         * Deleting it frees the clusters and the next open triggers a
         * fresh block erase during recording, reproducing the hang.   */
        const char *cleanup_paths[] = {
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
    }

    fs_file_t_init(&sd_audio_file);
#if !DSP_OFFLINE
    fs_file_t_init(&sd_heart_file);
    fs_file_t_init(&sd_lung_file);
#endif

    /* DEBUG BEACON: about to fs_open audio */
    led_set_blue();
    k_busy_wait(200000);

    /* v7.9 Option A: abandon the pre-allocate-then-overwrite strategy.
     * Opening without FS_O_TRUNC + fs_seek(0) caused FATFS to update
     * directory metadata, leaving the card busy exactly when the first
     * fs_write arrived — producing -EIO on every write attempt regardless
     * of how long we waited. FS_O_TRUNC gives FATFS a clean slate.
     * The first-write block erase may take ~50 ms but the 32 KB ring
     * (~2 s headroom at 16 KB/s) easily absorbs it.                   */
    int rc_audio = fs_open(&sd_audio_file, AUDIO_FILE_PATH,
                            FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc_audio < 0) {
        LOG_ERR("fs_open(%s) failed: %d", AUDIO_FILE_PATH, rc_audio);
        bt_nus_send(NULL, "ERR:SD_OPEN", 11);
        led_error_flash(led_set_yellow);
        return;
    }

    /* DEBUG BEACON: fs_open + seek done */
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

    led_set_cyan();
    k_busy_wait(200000);   /* DEBUG BEACON: cyan visible */
    analog_recording = true;

#if USE_SD
    k_sem_give(&sd_data_sem);
#endif
#if !DSP_OFFLINE
    k_sem_give(&dsp_data_sem);
#endif

    /* ════════════════════════════════════════════════════════════
     * PHASE 1: SAADC capture
     * ════════════════════════════════════════════════════════════ */
    LOG_INF("Phase 1: SAADC capture — %d s @ %d Hz", DURATION_S, SAMPLING_RATE);

    if (saadc_start_streaming() != 0) {
        analog_recording = false;
#if USE_SD
        fs_close(&sd_audio_file);
#if !DSP_OFFLINE
        fs_close(&sd_heart_file);
        fs_close(&sd_lung_file);
#endif
#endif
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:SAADC", 9);
        return;
    }

    /* DEBUG BEACON: SAADC started OK, about to wait for samples */
    led_set_white();
    k_busy_wait(200000);

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

    /* DEBUG BEACON: Phase 1 SAADC loop completed */
    led_set_green();
    k_busy_wait(500000);

#if !DSP_OFFLINE
    if (k_sem_take(&dsp_done_sem, K_MSEC(30000)) != 0) {
        LOG_WRN("DSP thread did not finish within 30 s");
    }
#endif

#if USE_SD
    LOG_INF("Phase 1 done — waiting for SD writer to flush...");
    bool sd_ok = false;
    if (k_sem_take(&sd_done_sem, K_MSEC(15000)) != 0) {
        LOG_ERR("SD writer timeout — file may be truncated");
    } else {
        LOG_INF("SD audio closed (checksum=0x%08X, drops=%u)",
                sd_checksum, (uint32_t)atomic_get(&sd_ring_drops));
        sd_ok = true;
    }

    if (!sd_ok) {
        bt_nus_send(NULL, "ERR:SD_WRITE", 12);
        led_error_flash(led_set_yellow);
        led_set_green();
        return;
    }

    {
        struct fs_dirent de;
        uint32_t expected_audio_size =
            (uint32_t)TOTAL_AUDIO_BYTES + (uint32_t)CHECKSUM_SIZE;

        if (fs_stat(AUDIO_FILE_PATH, &de) == 0) {
            LOG_INF("[DIAG] audio.pcm on-disk size: %u B  (expected %u B)",
                    (uint32_t)de.size, expected_audio_size);
            if ((uint32_t)de.size < expected_audio_size) {
                LOG_WRN("[DIAG] audio.pcm SHORTER than expected — "
                        "ring drops or SD write error likely");
                bt_nus_send(NULL, "ERR:SD_WRITE", 12);
                led_error_flash(led_set_yellow);
                led_set_green();
                return;
            }
        } else {
            LOG_ERR("[DIAG] fs_stat(%s) failed", AUDIO_FILE_PATH);
            bt_nus_send(NULL, "ERR:SD_WRITE", 12);
            led_error_flash(led_set_yellow);
            led_set_green();
            return;
        }
    }
#endif /* USE_SD */

    /* ════════════════════════════════════════════════════════════
     * PHASE 2: Offline MFCC
     * ════════════════════════════════════════════════════════════ */
    int hf = 0;
    int lf = 0;

#if DSP_OFFLINE && USE_SD
    LOG_INF("Phase 2: offline MFCC processing...");
    led_set_yellow();

    int mfcc_rc = process_audio_offline(&hf, &lf);
    if (mfcc_rc < 0) {
        LOG_ERR("Offline MFCC failed: %d", mfcc_rc);
        bt_nus_send(NULL, "ERR:DSP", 7);
        led_error_flash(led_set_yellow);
        led_set_green();
        return;
    }

    {
        struct fs_dirent de;

        uint32_t expected_heart =
            (uint32_t)hf * (uint32_t)heart_pipeline.cfg->n_mfcc * sizeof(float);
        uint32_t expected_lung  =
            (uint32_t)lf * (uint32_t)lung_pipeline.cfg->n_mfcc  * sizeof(float);

        if (fs_stat(HEART_MFCC_FILE_PATH, &de) == 0) {
            LOG_INF("[DIAG] heart_mfcc.f32: %u B  (expected %u B, hf=%d)",
                    (uint32_t)de.size, expected_heart, hf);
        } else {
            LOG_ERR("[DIAG] fs_stat(%s) failed", HEART_MFCC_FILE_PATH);
        }

        if (fs_stat(LUNG_MFCC_FILE_PATH, &de) == 0) {
            LOG_INF("[DIAG] lung_mfcc.f32:  %u B  (expected %u B, lf=%d)",
                    (uint32_t)de.size, expected_lung, lf);
        } else {
            LOG_ERR("[DIAG] fs_stat(%s) failed", LUNG_MFCC_FILE_PATH);
        }
    }
#endif /* DSP_OFFLINE && USE_SD */

    led_set_purple();
    report_ram_usage("after Phase 2 MFCC");

    if (hf <= 0 || lf <= 0) {
        LOG_ERR("MFCC pipeline error: heart=%d lung=%d", hf, lf);
        bt_nus_send(NULL, "ERR:DSP", 7);
        led_error_flash(led_set_yellow);
        led_set_green();
        return;
    }

#if USE_SD
    /* ════════════════════════════════════════════════════════════
     * PHASE 3: Heart model inference
     * ════════════════════════════════════════════════════════════ */
    heart_result_t heart_result;
    heart_result.rc         = -ENOTSUP;
    heart_result.value      = 0.f;
    heart_result.confidence = 0.f;
    heart_result.class_idx  = -1;

#if ENABLE_HEART_MODEL
    LOG_INF("Phase 3: heart inference (%d frames × %d coeffs)...",
            hf, heart_pipeline.cfg->n_mfcc);

    {
        uint32_t expected_heart_bytes =
            (uint32_t)hf * (uint32_t)heart_pipeline.cfg->n_mfcc * sizeof(float);
        struct fs_dirent de;
        bool skip_heart = false;

        if (fs_stat(HEART_MFCC_FILE_PATH, &de) < 0) {
            LOG_ERR("Heart MFCC file not found — skipping heart inference");
            skip_heart = true;
        } else if ((uint32_t)de.size == 0) {
            LOG_ERR("Heart MFCC file is empty — skipping heart inference");
            skip_heart = true;
        } else if ((uint32_t)de.size < expected_heart_bytes) {
            LOG_ERR("Heart MFCC file too small: %u B < expected %u B — "
                    "skipping heart inference",
                    (uint32_t)de.size, expected_heart_bytes);
            skip_heart = true;
        }

        if (!skip_heart) {
            led_set_blue();

            run_heart_inference(tensor_arena,
                                TENSOR_ARENA_BYTES,
                                HEART_MFCC_FILE_PATH,
                                hf,
                                heart_pipeline.cfg->n_mfcc,
                                &heart_result);

            if (heart_result.rc < 0) {
                LOG_ERR("Heart inference failed: %d", heart_result.rc);
                bt_nus_send(NULL, "ERR:HEART_INF", 13);
            } else {
                LOG_INF("Heart result: HR=%.0f BPM (confidence=%.3f, class=%d)",
                        (double)heart_result.value,
                        (double)heart_result.confidence,
                        heart_result.class_idx);
            }
        } else {
            heart_result.rc = -ENODATA;
            bt_nus_send(NULL, "ERR:HEART_INF", 13);
        }
    }
#else
    LOG_INF("Phase 3: heart model disabled (ENABLE_HEART_MODEL=0) — skipping");
#endif /* ENABLE_HEART_MODEL */

    /* ════════════════════════════════════════════════════════════
     * PHASE 4: Lung model inference
     * ════════════════════════════════════════════════════════════ */
    lung_result_t lung_result;
    lung_result.rc         = -ENOTSUP;
    lung_result.value      = 0.f;
    lung_result.confidence = 0.f;
    lung_result.class_idx  = -1;

#if ENABLE_LUNG_MODEL
    LOG_INF("Phase 4: lung inference (%d frames × %d coeffs)...",
            lf, lung_pipeline.cfg->n_mfcc);

    {
        uint32_t expected_lung_bytes =
            (uint32_t)lf * (uint32_t)lung_pipeline.cfg->n_mfcc * sizeof(float);
        struct fs_dirent de;
        bool skip_lung = false;

        if (fs_stat(LUNG_MFCC_FILE_PATH, &de) < 0) {
            LOG_ERR("Lung MFCC file not found — skipping lung inference");
            skip_lung = true;
        } else if ((uint32_t)de.size == 0) {
            LOG_ERR("Lung MFCC file is empty — skipping lung inference");
            skip_lung = true;
        } else if ((uint32_t)de.size < expected_lung_bytes) {
            LOG_ERR("Lung MFCC file too small: %u B < expected %u B — "
                    "skipping lung inference",
                    (uint32_t)de.size, expected_lung_bytes);
            skip_lung = true;
        }

        if (!skip_lung) {
            led_set_purple();

            run_lung_inference(tensor_arena,
                               TENSOR_ARENA_BYTES,
                               LUNG_MFCC_FILE_PATH,
                               lf,
                               lung_pipeline.cfg->n_mfcc,
                               &lung_result);

            if (lung_result.rc < 0) {
                LOG_ERR("Lung inference failed: %d", lung_result.rc);
                bt_nus_send(NULL, "ERR:LUNG_INF", 12);
            } else {
                LOG_INF("Lung result: RR=%.0f BPM (confidence=%.3f, class=%d)",
                        (double)lung_result.value,
                        (double)lung_result.confidence,
                        lung_result.class_idx);
            }
        } else {
            lung_result.rc = -ENODATA;
            bt_nus_send(NULL, "ERR:LUNG_INF", 12);
        }
    }
#else
    LOG_INF("Phase 4: lung model disabled (ENABLE_LUNG_MODEL=0) — skipping");
#endif /* ENABLE_LUNG_MODEL */
#endif /* USE_SD */

    /* ════════════════════════════════════════════════════════════
     * PHASE 5: BLE upload
     * ════════════════════════════════════════════════════════════ */
    LOG_INF("Phase 5: BLE upload...");
    led_set_green();

#if USE_SD
    struct file_stream_entry {
        const char *start_fmt;
        const char *end_msg;
        const char *path;
        uint32_t    file_bytes;
        bool        is_audio;
    };

    uint32_t hr_file_bytes = 0;
    uint32_t rr_file_bytes = 0;

    if (heart_result.rc == 0) {
        struct fs_dirent dirent;
        if (fs_stat(HR_RESULT_FILE_PATH, &dirent) == 0) {
            hr_file_bytes = (uint32_t)dirent.size;
        }
    }
    if (lung_result.rc == 0) {
        struct fs_dirent dirent;
        if (fs_stat(RR_RESULT_FILE_PATH, &dirent) == 0) {
            rr_file_bytes = (uint32_t)dirent.size;
        }
    }

    struct file_stream_entry stream_list[6];
    int stream_count = 0;

#if !BLE_AUDIO_LIVE
    stream_list[stream_count++] = (struct file_stream_entry){
        .start_fmt  = "START:%u\n",
        .end_msg    = "finished\n",
        .path       = AUDIO_FILE_PATH,
        .file_bytes = (uint32_t)TOTAL_AUDIO_BYTES,
        .is_audio   = true,
    };
#endif

    stream_list[stream_count++] = (struct file_stream_entry){
        .start_fmt  = "MFCC_HEART_START:%u\n",
        .end_msg    = "MFCC_HEART_END\n",
        .path       = HEART_MFCC_FILE_PATH,
        .file_bytes = (uint32_t)hf * (uint32_t)heart_pipeline.cfg->n_mfcc * sizeof(float),
        .is_audio   = false,
    };

    stream_list[stream_count++] = (struct file_stream_entry){
        .start_fmt  = "MFCC_LUNG_START:%u\n",
        .end_msg    = "MFCC_LUNG_END\n",
        .path       = LUNG_MFCC_FILE_PATH,
        .file_bytes = (uint32_t)lf * (uint32_t)lung_pipeline.cfg->n_mfcc * sizeof(float),
        .is_audio   = false,
    };

#if ENABLE_HEART_MODEL
    if (heart_result.rc == 0 && hr_file_bytes > 0) {
        stream_list[stream_count++] = (struct file_stream_entry){
            .start_fmt  = "RESULT_HEART_START:%u\n",
            .end_msg    = "RESULT_HEART_END\n",
            .path       = HR_RESULT_FILE_PATH,
            .file_bytes = hr_file_bytes,
            .is_audio   = false,
        };
    }
#endif

#if ENABLE_LUNG_MODEL
    if (lung_result.rc == 0 && rr_file_bytes > 0) {
        stream_list[stream_count++] = (struct file_stream_entry){
            .start_fmt  = "RESULT_LUNG_START:%u\n",
            .end_msg    = "RESULT_LUNG_END\n",
            .path       = RR_RESULT_FILE_PATH,
            .file_bytes = rr_file_bytes,
            .is_audio   = false,
        };
    }
#endif

    for (int fi = 0; fi < stream_count; fi++) {
        struct file_stream_entry *e = &stream_list[fi];

        char ctrl[64];
        int ctrl_len = snprintf(ctrl, sizeof(ctrl), e->start_fmt, e->file_bytes);
        int err;
        do {
            err = bt_nus_send(NULL, ctrl, ctrl_len);
            if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(5));
        } while (err == -ENOMEM || err == -EAGAIN);
        k_sleep(K_MSEC(20));

        struct fs_file_t f;
        fs_file_t_init(&f);
        if (fs_open(&f, e->path, FS_O_READ) < 0) {
            LOG_ERR("Cannot open %s for BLE send", e->path);
            bt_nus_send(NULL, "ERR:SD_READ", 11);
            continue;
        }

        /* [FIX 6] BLE TX packet buffer 4-byte aligned.
         * v7.6: chunked [seq16][len16][payload] is correct for ALL
         * files (audio, MFCC, results) — receiver expects this and
         * strips the header before writing to disk. */
        uint8_t  pkt[251] __aligned(4);
        uint16_t seq     = 0;
        uint16_t payload = nus_chunk_size - CHUNK_HEADER_BYTES;
        uint32_t bytes_sent = 0;

        while (bytes_sent < e->file_bytes) {
            uint32_t want = MIN((uint32_t)payload, e->file_bytes - bytes_sent);
            ssize_t  nr   = fs_read(&f, &pkt[CHUNK_HEADER_BYTES], want);
            if (nr <= 0) break;

            sys_put_le16(seq,          &pkt[0]);
            sys_put_le16((uint16_t)nr, &pkt[2]);
            do {
                err = bt_nus_send(NULL, pkt, CHUNK_HEADER_BYTES + (uint16_t)nr);
                if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(2));
            } while (err == -ENOMEM || err == -EAGAIN);
            seq = (seq + 1) & 0xFFFF;
            bytes_sent += (uint32_t)nr;
        }

        fs_close(&f);

        do {
            err = bt_nus_send(NULL, e->end_msg, strlen(e->end_msg));
            if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(5));
        } while (err == -ENOMEM || err == -EAGAIN);

        LOG_INF("BLE send done: %s (%u bytes)", e->path, bytes_sent);
        k_sleep(K_MSEC(20));

#if !BLE_AUDIO_LIVE
        if (e->is_audio) {
            bt_nus_send(NULL, "SD:OK\n", 6);
            k_sleep(K_MSEC(10));
        }
#endif
    }
#endif /* USE_SD */

    LOG_INF("record_and_stream() complete. HR=%.0f RR=%.0f",
            (double)heart_result.value,
            (double)lung_result.value);
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

    err = bt_enable(NULL);
    if (err) { LOG_ERR("bt_enable: %d", err); return err; }

    err = bt_nus_cb_register(&nus_listener, NULL);
    if (err) { LOG_ERR("bt_nus_cb_register: %d", err); return err; }

    err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                          sd_adv, ARRAY_SIZE(sd_adv));
    if (err) { LOG_ERR("bt_le_adv_start: %d", err); return err; }

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

#if USE_SD
    k_thread_create(&sd_writer_thread_data, sd_writer_stack,
                    K_THREAD_STACK_SIZEOF(sd_writer_stack),
                    sd_writer_thread_fn, NULL, NULL, NULL,
                    SD_WRITER_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&sd_writer_thread_data, "sd_writer");

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
    LOG_INF("AcoustEEEcare v7.7 ready — waiting for BLE connection");

    /* v7.5 [FIX 15]: baseline RAM snapshot, before any recording.
     * Compare subsequent snapshots against this to spot leaks/overflows. */
    report_ram_usage("boot (idle)");

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