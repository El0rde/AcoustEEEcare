/*
 * AcoustEEEcare — SAADC BLE + SD Card Edition
 * ============================================================
 * v6.8 — Fixed ERR:SD_OPEN (fs_open -2 / ENOENT after fs_mount)
 *
 * CHANGES FROM v6.7:
 *
 *   [FIX 7] ERR:SD_OPEN / fs_open -2 after successful fs_mount.
 *     Root cause: fs_mount() can return 0 (success) even when the FAT
 *     volume is present but the directory structure is not ready for
 *     file creation — for example when the card was ejected uncleanly,
 *     the FAT root is valid but the working area is in an unexpected
 *     state, or (on some SD/FatFs builds) the mount-point directory
 *     entry itself does not exist yet.
 *
 *     Three sub-fixes:
 *
 *     a) verify_sd_writable() — called once at the end of init_sd_card().
 *        Creates and immediately deletes a small sentinel file
 *        ("/SD:/acoustchk") to confirm the filesystem actually accepts
 *        writes.  If this fails, sd_mounted is set to false so the
 *        firmware returns ERR:NOSD (safe, informative) rather than
 *        reaching ERR:SD_OPEN later.
 *
 *        NOTE: The sentinel filename must NOT start with '~'.  ELM FatFs
 *        treats '~' as reserved for Windows 8.3 short-name generation and
 *        returns FR_INVALID_NAME (-ENOENT / -2) — the same error we were
 *        trying to catch.  Plain ASCII alphanumeric names are safe.
 *
 *     b) Per-file error logging in record_and_stream().
 *        Each fs_open() call is now separate with its own LOG_ERR that
 *        prints the exact path and errno, making future diagnosis easier.
 *
 *     c) fs_mkdir(SD_CARD_MOUNT_POINT) guard before file opens.
 *        Harmless on healthy cards (returns -EEXIST which is ignored);
 *        on cards where the mount point directory entry is missing it
 *        creates it so fs_open can succeed.
 *
 * CHANGES FROM v6.6 (carried forward from v6.7):
 *   [FIX 1] DSP moved out of SAADC ISR into a dedicated dsp_thread.
 *   [FIX 2] Eliminated double saadc_init() + next_dma_buf race.
 *   [FIX 3] USE_SD set to true.
 *   [FIX 4] half_produced_sem timeout raised 200 ms -> 500 ms.
 *   [FIX 5] Stale reference to sd_file removed.
 *   [FIX 6] SD ring buffer declarations moved above frame callbacks.
 *
 * OVERLAY NOTE (xiao_ble.overlay):
 *   spi-max-frequency = <4000000> is recommended for broad SD card
 *   compatibility.  10 MHz works with most modern cards but some cheaper
 *   cards mis-behave above 4–8 MHz, especially during write bursts.
 *   If you see SD write errors during recording, drop to 4000000 first.
 *
 * REMAINING DESIGN (unchanged from v6.7):
 *   - No fs_write() in ISR — only ring_buf_put() + k_sem_give()
 *   - Three ring buffers: audio_ring (16 KB BLE), audio_sd_ring
 *     (32 KB SD audio), heart_mfcc_ring (8 KB), lung_mfcc_ring (16 KB)
 *   - New dsp_ring (32 KB) for ISR->DSP thread handoff
 *   - SD writer thread drains sd rings in 512-byte aligned sectors
 *   - XOR checksum appended at end of audio file
 *
 * TARGET HARDWARE
 *   Seeed XIAO nRF52840
 *   Analog MEMS microphone on AIN0 (P0.02), cap-coupled
 *   SD card on SPI2, CS on P0.28 (adjust overlay to your wiring)
 * ============================================================
 */

#define USE_SD  true   /* must be true — MFCC callbacks discard data when false */

/* ────────────────────────────────────────────────────────────────────
 * BLE_AUDIO_LIVE — controls when audio is streamed over BLE.
 *
 *   1 = OLD behavior. Audio is pushed to audio_ring in the SAADC ISR
 *       and streamed over BLE in real time while recording.  This
 *       competes with SD writer + DSP thread for CPU and was found to
 *       starve all consumers when the lung MFCC pipeline at hop=5
 *       takes ~50% of the M4F.
 *
 *   0 = NEW behavior. Audio is NOT pushed to audio_ring during the
 *       SAADC capture window.  The SAADC ISR pushes only to dsp_ring
 *       (for MFCC) and audio_sd_ring (for SD).  After SAADC stops and
 *       the SD writer finishes, the audio file is opened from SD and
 *       streamed back over BLE just like the heart/lung MFCC files.
 *
 * Why 0 is the current default: this gives the SD writer and DSP
 * thread full CPU during the 10-second recording window, so we can
 * capture the winning-config ground-truth audio + MFCCs cleanly.
 * Once the MFCC params are tuned for the live workload, this can be
 * re-enabled.
 * ──────────────────────────────────────────────────────────────────── */
#define BLE_AUDIO_LIVE  0

/* ────────────────────────────────────────────────────────────────────
 * DSP_OFFLINE — controls WHEN the MFCC pipelines process audio.
 *
 *   0 = OLD behavior. SAADC ISR pushes every half-buffer into dsp_ring;
 *       the DSP thread drains the ring and runs both MFCC pipelines
 *       in real time, concurrently with audio capture.  Heart and lung
 *       MFCC frames are emitted to ring buffers and the SD writer
 *       drains those rings into heart_mfcc.f32 / lung_mfcc.f32 as the
 *       recording progresses.
 *
 *   1 = NEW (decoupled) behavior. The SAADC ISR does NOT push to
 *       dsp_ring during the recording window.  The DSP thread stays
 *       asleep.  Only audio capture + SD audio write happen during the
 *       10-second recording.  After SAADC stops and the audio file
 *       closes, the main thread re-reads the audio file from SD and
 *       feeds it through both MFCC pipelines at maximum CPU speed
 *       (no real-time constraint).  Frame callbacks write coefficients
 *       directly to the open heart/lung MFCC files via fs_write.
 *
 * Why 1 is the current default: the lung pipeline at hop=5 needs
 * ~48% of one M4F core to keep up in real time, which causes dsp_ring
 * to overflow and ~36% of frames to be lost.  Decoupling the MFCC
 * compute from real time gives bit-identical results with zero frame
 * loss, at the cost of ~15-30 s extra wall-clock per recording.
 * Total: ~10 s capture + ~30 s offline MFCC + ~12 s BLE upload.
 *
 * The MFCC math is IDENTICAL between modes: same dsp_mfcc.c, same
 * RFFT, same mel filterbank, same DCT.  Only the timing differs.
 * ──────────────────────────────────────────────────────────────────── */
#define DSP_OFFLINE  1

/* ══════════════════════════════════════════════════════════════════
 * INCLUDES
 * ══════════════════════════════════════════════════════════════════ */
#include <stdint.h>
#include <string.h>
#include <stdio.h>

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

#include <nrfx.h>
#include <drivers/nrfx_errors.h>
#include <nrfx_saadc.h>

#include "dsp_mfcc.h"
#include "heart_mfcc_config.h"
#include "lung_mfcc_config.h"

#if USE_SD
#include <zephyr/fs/fs.h>
#include <zephyr/storage/disk_access.h>
#include <ff.h>
#endif

LOG_MODULE_REGISTER(AcoustEEEcare);

/* ══════════════════════════════════════════════════════════════════
 * RAM USAGE REPORT
 * ══════════════════════════════════════════════════════════════════ */
extern char _end;

#define BLE_TX_STACK_SIZE     2048
#define DSP_THREAD_STACK_SIZE 4096   /* DSP thread needs room for RFFT scratch */
#define SD_WRITER_STACK_SIZE  2048

static struct k_thread ble_tx_thread_data;
static struct k_thread dsp_thread_data;
static struct k_thread sd_writer_thread_data;
static dsp_mfcc_pipeline_t heart_pipeline;
static dsp_mfcc_pipeline_t lung_pipeline;
static uint32_t heart_frame_count;
static uint32_t lung_frame_count;

static void report_ram_usage(void)
{
    const uintptr_t ram_start = 0x20000000u;
    const uintptr_t ram_end   = 0x20040000u;
    const uintptr_t bss_end   = (uintptr_t)&_end;

    uint32_t bss_used   = (uint32_t)(bss_end - ram_start);
    uint32_t free_above = (uint32_t)(ram_end  - bss_end);

    LOG_INF("==================== RAM USAGE REPORT ====================");
    LOG_INF("BSS+data used:   %u bytes (%u KB)", bss_used,  bss_used  / 1024);
    LOG_INF("Free above BSS:  %u bytes (%u KB)", free_above, free_above / 1024);
    LOG_INF("  (heap + stacks live here; TFLM arena ceiling is a fraction of this)");

    size_t unused;

    if (k_thread_stack_space_get(&ble_tx_thread_data, &unused) == 0) {
        LOG_INF("BLE TX stack:    %u / %u bytes used  (%u%% headroom)",
                (unsigned)(BLE_TX_STACK_SIZE - unused),
                (unsigned)BLE_TX_STACK_SIZE,
                (unsigned)(unused * 100u / BLE_TX_STACK_SIZE));
    }

    if (k_thread_stack_space_get(&dsp_thread_data, &unused) == 0) {
        LOG_INF("DSP thread stack:%u / %u bytes used  (%u%% headroom)",
                (unsigned)(DSP_THREAD_STACK_SIZE - unused),
                (unsigned)DSP_THREAD_STACK_SIZE,
                (unsigned)(unused * 100u / DSP_THREAD_STACK_SIZE));
    }

#if USE_SD
    if (k_thread_stack_space_get(&sd_writer_thread_data, &unused) == 0) {
        LOG_INF("SD writer stack: %u / %u bytes used  (%u%% headroom)",
                (unsigned)(SD_WRITER_STACK_SIZE - unused),
                (unsigned)SD_WRITER_STACK_SIZE,
                (unsigned)(unused * 100u / SD_WRITER_STACK_SIZE));
    }
#endif

    LOG_INF("Heart frames:    %u  (expected %d)",
            heart_frame_count, heart_pipeline.cfg->n_frames_expected);
    LOG_INF("Lung  frames:    %u  (expected %d)",
            lung_frame_count,  lung_pipeline.cfg->n_frames_expected);

    LOG_INF("==========================================================");
}

/* ══════════════════════════════════════════════════════════════════
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
 * Init to 0 — converges within ~256 samples (~32 ms at 8 kHz). */
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

/* ══════════════════════════════════════════════════════════════════
 * DSP RING BUFFER + DSP THREAD
 *
 * The ISR pushes raw int16 half-buffers here (ISR-safe ring_buf_put).
 * The DSP thread drains them and calls dsp_mfcc_feed_chunk() — keeping
 * all FPU-heavy work (RFFT, mel filterbank, DCT) out of interrupt context.
 *
 * Size: 32 KB = 20 half-buffers of headroom at 8 kHz (160 ms).
 * That is far more than the DSP thread scheduling latency.
 * ══════════════════════════════════════════════════════════════════ */
#define DSP_RING_BYTES  (32 * 1024)
RING_BUF_DECLARE(dsp_ring, DSP_RING_BYTES);

static K_SEM_DEFINE(dsp_data_sem, 0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(dsp_done_sem, 0, 1);   /* DSP thread signals main when finished */

/* Scratch buffer for DSP thread to pop samples into before feed_chunk */
static int16_t dsp_pop_buf[HALF_BUF_SAMPLES];

/* ══════════════════════════════════════════════════════════════════
 * SD CARD DATA DECLARATIONS  (USE_SD true only)
 *
 * [FIX 6] Moved up from the lower #if USE_SD block so that
 * heart_frame_cb() and lung_frame_cb() (below) can reference
 * heart_mfcc_ring, lung_mfcc_ring, and sd_ring_drops without
 * hitting undeclared-identifier errors.
 *
 * What lives here: path/size #defines, ring buffer declarations,
 * sd_ring_drops, and the three *_ring_high_water counters.
 *
 * What stays in the lower #if USE_SD block: semaphores, thread stack,
 * FATFS / fs_mount_t / file handles, compute_checksum(), init_sd_card(),
 * and sd_writer_thread_fn() — none of those are referenced by the
 * frame callbacks.
 * ══════════════════════════════════════════════════════════════════ */
#if USE_SD

#define SD_CARD_MOUNT_POINT  "/SD:"
#define AUDIO_FILE_PATH      "/SD:/analog.pcm"
#define HEART_MFCC_FILE_PATH "/SD:/heart_mfcc.f32"
#define LUNG_MFCC_FILE_PATH  "/SD:/lung_mfcc.f32"

/*
 * [FIX 7] Sentinel filename for write-verify.
 * Must be a plain alphanumeric name — do NOT use '~' as the first
 * character.  ELM FatFs reserves '~' for Windows 8.3 short-name
 * generation and returns FR_INVALID_NAME (-ENOENT, -2) for any
 * filename that starts with it.  That is the exact error we are
 * trying to detect, so using '~' caused verify_sd_writable() to
 * always fail even on a perfectly healthy, freshly-formatted card.
 */
#define SD_INIT_CHECK_PATH   "/SD:/acoustchk"

#define CHECKSUM_SIZE        sizeof(uint32_t)

#define AUDIO_SD_RING_BYTES      (32 * 1024)
#define HEART_MFCC_RING_BYTES    ( 8 * 1024)
#define LUNG_MFCC_RING_BYTES     (16 * 1024)

RING_BUF_DECLARE(audio_sd_ring,   AUDIO_SD_RING_BYTES);
RING_BUF_DECLARE(heart_mfcc_ring, HEART_MFCC_RING_BYTES);
RING_BUF_DECLARE(lung_mfcc_ring,  LUNG_MFCC_RING_BYTES);

static atomic_t  sd_ring_drops;
static uint32_t  audio_ring_high_water = 0;
static uint32_t  heart_ring_high_water = 0;
static uint32_t  lung_ring_high_water  = 0;

#endif /* USE_SD — data declarations */

/* ══════════════════════════════════════════════════════════════════
 * DUAL MFCC PIPELINE INSTANCES
 * ══════════════════════════════════════════════════════════════════ */
static int16_t heart_window[50];
static int16_t lung_window[100];

#if USE_SD
/* Offline-mode direct-to-file pointers.  When non-NULL, the frame
 * callbacks fs_write coefficients to these files instead of pushing
 * to the heart/lung MFCC ring buffers.  Set by process_audio_offline()
 * for the duration of the offline MFCC pass; reset to NULL afterwards.
 *
 * Important: these are only safe to read/write from a single thread
 * (the main thread, during offline processing).  In offline mode the
 * DSP thread never runs, so there is no concurrency. */
static struct fs_file_t *offline_heart_fp = NULL;
static struct fs_file_t *offline_lung_fp  = NULL;
#endif

/* Heart frame callback.
 * In online mode: invoked from DSP thread; pushes coeffs to heart_mfcc_ring.
 * In offline mode: invoked from main thread; writes coeffs to offline_heart_fp. */
static void heart_frame_cb(int idx, const float *coeffs, void *user)
{
    (void)user; (void)idx;
    heart_frame_count++;

#if USE_SD
    const uint32_t bytes = heart_pipeline.cfg->n_mfcc * sizeof(float);

    if (offline_heart_fp != NULL) {
        /* Offline mode: direct fs_write.  No ring buffer involved. */
        ssize_t w = fs_write(offline_heart_fp, coeffs, bytes);
        if (w != (ssize_t)bytes) {
            LOG_WRN_ONCE("offline heart fs_write short: %d/%u", (int)w, bytes);
        }
    } else {
        /* Online mode: push to ring for SD writer thread to drain. */
        uint32_t put = ring_buf_put(&heart_mfcc_ring,
                                    (const uint8_t *)coeffs, bytes);
        if (put != bytes) {
            atomic_inc(&sd_ring_drops);
        }
    }
#else
    (void)coeffs;
#endif
}

/* Lung frame callback. Same dispatch logic as heart_frame_cb. */
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
    } else {
        uint32_t put = ring_buf_put(&lung_mfcc_ring,
                                    (const uint8_t *)coeffs, bytes);
        if (put != bytes) {
            atomic_inc(&sd_ring_drops);
        }
    }
#else
    (void)coeffs;
#endif
}

#define BLE_TX_PRIORITY     5
#define DSP_THREAD_PRIORITY 7   /* below BLE TX (5) and SD writer (6) */

static K_THREAD_STACK_DEFINE(ble_tx_stack,     BLE_TX_STACK_SIZE);
static K_THREAD_STACK_DEFINE(dsp_thread_stack, DSP_THREAD_STACK_SIZE);
static void ble_tx_thread_fn(void *a, void *b, void *c);
static void dsp_thread_fn(void *a, void *b, void *c);

/* ══════════════════════════════════════════════════════════════════
 * DSP THREAD
 *
 * Drains dsp_ring and runs both MFCC pipelines. Signals dsp_done_sem
 * once recording stops and dsp_ring is fully drained.
 * ══════════════════════════════════════════════════════════════════ */
static void dsp_thread_fn(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (true) {
        k_sem_take(&dsp_data_sem, K_FOREVER);

        /* Drain dsp_ring while recording or while data remains */
        while (analog_recording || ring_buf_size_get(&dsp_ring) > 0) {
            uint32_t avail = ring_buf_size_get(&dsp_ring);
            if (avail < HALF_BUF_BYTES) {
                if (!analog_recording) break;   /* done — ring is empty */
                k_sleep(K_MSEC(1));
                continue;
            }

            uint32_t got = ring_buf_get(&dsp_ring,
                                        (uint8_t *)dsp_pop_buf,
                                        HALF_BUF_BYTES);
            if (got != HALF_BUF_BYTES) continue;   /* partial read — skip */

            dsp_mfcc_feed_chunk(&heart_pipeline, dsp_pop_buf, HALF_BUF_SAMPLES);
            dsp_mfcc_feed_chunk(&lung_pipeline,  dsp_pop_buf, HALF_BUF_SAMPLES);
        }

        k_sem_give(&dsp_done_sem);
    }
}

/* ══════════════════════════════════════════════════════════════════
 * SD CARD GLOBALS  (USE_SD true only)
 *
 * Semaphores, thread stack, FATFS, file handles, helpers, and the
 * writer thread body. Ring buffer + atomic declarations have been
 * moved to the earlier #if USE_SD block above (see [FIX 6]).
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
static struct fs_file_t sd_heart_file;
static struct fs_file_t sd_lung_file;
static uint32_t         sd_checksum  = 0;

#define SD_WRITE_BUF_SIZE  512
static uint8_t sd_write_buf[SD_WRITE_BUF_SIZE];

static uint32_t compute_checksum(const uint8_t *data, uint32_t len)
{
    uint32_t cs = 0;
    for (uint32_t i = 0; i < len; i++) cs ^= data[i];
    return cs;
}

/* ── [FIX 7a] SD write-verify: create + delete a sentinel file ── */
/*
 * verify_sd_writable()
 *
 * fs_mount() returning 0 does not guarantee the filesystem can actually
 * create files — it only means the FAT superblock was parsed.  On cards
 * that were ejected uncleanly, the first fs_open(FS_O_CREATE) can fail
 * with -ENOENT even though sd_mounted would be true.
 *
 * This function creates a tiny sentinel file, confirms the write
 * succeeded, then deletes it.  Called once at the end of init_sd_card().
 * If it fails, sd_mounted is forced to false so record_and_stream()
 * returns ERR:NOSD (clear) instead of ERR:SD_OPEN (confusing).
 *
 * IMPORTANT: SD_INIT_CHECK_PATH must not start with '~'.  ELM FatFs
 * reserves '~' for 8.3 short-name generation and returns -ENOENT for
 * such names — which would make this function always fail.
 *
 * Returns 0 on success, negative errno on failure.
 */
static int verify_sd_writable(void)
{
    struct fs_file_t f;
    fs_file_t_init(&f);

    int rc = fs_open(&f, SD_INIT_CHECK_PATH,
                     FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc < 0) {
        LOG_ERR("SD write-verify: fs_open(%s) failed: %d",
                SD_INIT_CHECK_PATH, rc);
        return rc;
    }

    const uint8_t sentinel = 0xA5;
    ssize_t w = fs_write(&f, &sentinel, 1);
    fs_close(&f);
    fs_unlink(SD_INIT_CHECK_PATH);

    if (w != 1) {
        LOG_ERR("SD write-verify: fs_write returned %d", (int)w);
        return (w < 0) ? (int)w : -EIO;
    }

    LOG_INF("SD write-verify: OK");
    return 0;
}

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

    if (fs_mount(&mp) != 0) {
        LOG_ERR("fs_mount failed");
        return -1;
    }

    LOG_INF("SD mounted at %s", SD_CARD_MOUNT_POINT);
    sd_mounted = true;

    /* [FIX 7a] Confirm the filesystem can actually create files.
     * If this fails the card is physically present but not write-ready
     * (corrupt FAT, write-protected, or wrong format).  Force
     * sd_mounted = false so the firmware gives ERR:NOSD, not
     * the cryptic ERR:SD_OPEN. */
    if (verify_sd_writable() != 0) {
        LOG_ERR("SD mounted but not writable — treating as absent");
        fs_unmount(&mp);
        sd_mounted = false;
        return -1;
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
            while (ring_buf_size_get(&audio_sd_ring)   > 0 ||
                   ring_buf_size_get(&heart_mfcc_ring) > 0 ||
                   ring_buf_size_get(&lung_mfcc_ring)  > 0) {
                ring_buf_get(&audio_sd_ring,   sd_write_buf,
                             MIN(ring_buf_size_get(&audio_sd_ring),   SD_WRITE_BUF_SIZE));
                ring_buf_get(&heart_mfcc_ring, sd_write_buf,
                             MIN(ring_buf_size_get(&heart_mfcc_ring), SD_WRITE_BUF_SIZE));
                ring_buf_get(&lung_mfcc_ring,  sd_write_buf,
                             MIN(ring_buf_size_get(&lung_mfcc_ring),  SD_WRITE_BUF_SIZE));
            }
            k_sem_give(&sd_done_sem);
            continue;
        }

        LOG_INF("SD writer: starting write loop (3 files)");
        uint32_t audio_written = 0;
        uint32_t heart_written = 0;
        uint32_t lung_written  = 0;
        sd_checksum = 0;

        while (analog_recording ||
               ring_buf_size_get(&audio_sd_ring)   > 0 ||
               ring_buf_size_get(&heart_mfcc_ring) > 0 ||
               ring_buf_size_get(&lung_mfcc_ring)  > 0) {

            bool did_work = false;

            /* ── Audio ring ── */
            {
                uint32_t avail = ring_buf_size_get(&audio_sd_ring);
                bool done = !analog_recording;
                if (avail >= SD_WRITE_BUF_SIZE || (done && avail > 0)) {
                    uint32_t n = MIN(avail, SD_WRITE_BUF_SIZE);
                    ring_buf_get(&audio_sd_ring, sd_write_buf, n);
                    sd_checksum ^= compute_checksum(sd_write_buf, n);
                    ssize_t w = fs_write(&sd_audio_file, sd_write_buf, n);
                    if (w > 0) audio_written += (uint32_t)w;
                    uint32_t fill = ring_buf_size_get(&audio_sd_ring);
                    if (fill > audio_ring_high_water) audio_ring_high_water = fill;
                    did_work = true;
                }
            }

            /* ── Heart MFCC ring ── */
            {
                uint32_t avail = ring_buf_size_get(&heart_mfcc_ring);
                bool done = !analog_recording;
                if (avail >= SD_WRITE_BUF_SIZE || (done && avail > 0)) {
                    uint32_t n = MIN(avail, SD_WRITE_BUF_SIZE);
                    ring_buf_get(&heart_mfcc_ring, sd_write_buf, n);
                    ssize_t w = fs_write(&sd_heart_file, sd_write_buf, n);
                    if (w > 0) heart_written += (uint32_t)w;
                    uint32_t fill = ring_buf_size_get(&heart_mfcc_ring);
                    if (fill > heart_ring_high_water) heart_ring_high_water = fill;
                    did_work = true;
                }
            }

            /* ── Lung MFCC ring ── */
            {
                uint32_t avail = ring_buf_size_get(&lung_mfcc_ring);
                bool done = !analog_recording;
                if (avail >= SD_WRITE_BUF_SIZE || (done && avail > 0)) {
                    uint32_t n = MIN(avail, SD_WRITE_BUF_SIZE);
                    ring_buf_get(&lung_mfcc_ring, sd_write_buf, n);
                    ssize_t w = fs_write(&sd_lung_file, sd_write_buf, n);
                    if (w > 0) lung_written += (uint32_t)w;
                    uint32_t fill = ring_buf_size_get(&lung_mfcc_ring);
                    if (fill > lung_ring_high_water) lung_ring_high_water = fill;
                    did_work = true;
                }
            }

            if (!did_work) {
                k_sleep(K_MSEC(1));
            }
        }

        /* Append audio checksum, close all three files */
        fs_write(&sd_audio_file, &sd_checksum, CHECKSUM_SIZE);
        fs_close(&sd_audio_file);
        fs_close(&sd_heart_file);
        fs_close(&sd_lung_file);

        LOG_INF("SD writer done: audio=%u B  heart=%u B  lung=%u B",
                audio_written, heart_written, lung_written);
        LOG_INF("Ring high-water: audio=%u  heart=%u  lung=%u",
                audio_ring_high_water, heart_ring_high_water, lung_ring_high_water);

        k_sem_give(&sd_done_sem);
    }
}

#if DSP_OFFLINE
/* ══════════════════════════════════════════════════════════════════
 * OFFLINE MFCC PROCESSING (Phase 2)
 *
 * Called after the SAADC capture phase has fully completed and the
 * audio file on SD has been closed.  Re-opens the audio file for
 * read, opens heart and lung MFCC files for write, then feeds the
 * audio through both MFCC pipelines on the main thread.  Frame
 * callbacks write coefficients directly to the open MFCC files
 * (no ring buffers, no DSP thread).
 *
 * Runs at maximum CPU speed -- no real-time constraint.  Typical
 * duration for 10 s of audio at heart hop=15 + lung hop=5: 15-30 s.
 *
 * Returns 0 on success, negative errno on failure.
 * ══════════════════════════════════════════════════════════════════ */
static int process_audio_offline(void)
{
    int rc = 0;
    struct fs_file_t fa;          /* audio file (read)       */
    struct fs_file_t fh;          /* heart MFCC file (write) */
    struct fs_file_t fl;          /* lung MFCC file (write)  */

    fs_file_t_init(&fa);
    fs_file_t_init(&fh);
    fs_file_t_init(&fl);

    LOG_INF("Offline MFCC: opening files…");

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

    /* Reset both pipelines for a fresh pass.  This zeros their biquad
     * state, sliding windows, decim_phase, hop_counter, and frames_emitted.
     * After this, frame_emitted will count the actual offline frames. */
    dsp_mfcc_reset(&heart_pipeline);
    dsp_mfcc_reset(&lung_pipeline);
    heart_frame_count = 0;
    lung_frame_count  = 0;

    /* Engage offline-callback dispatch: from now until we clear these,
     * the frame callbacks fs_write to fh / fl instead of pushing to
     * the heart_mfcc_ring / lung_mfcc_ring. */
    offline_heart_fp = &fh;
    offline_lung_fp  = &fl;

    /* Read the audio file in HALF_BUF_BYTES chunks (1024 B = 512 int16
     * samples) -- same granularity the SAADC ISR used originally.  We
     * deliberately stop after TOTAL_AUDIO_BYTES to skip the XOR checksum
     * that the SD writer appended at end-of-file. */
    int16_t  read_buf[HALF_BUF_SAMPLES];
    uint32_t bytes_read_total = 0;
    int      chunk_idx        = 0;
    int64_t  t_start           = k_uptime_get();

    while (bytes_read_total < TOTAL_AUDIO_BYTES) {
        uint32_t want = MIN((uint32_t)sizeof(read_buf),
                            TOTAL_AUDIO_BYTES - bytes_read_total);
        ssize_t  got  = fs_read(&fa, read_buf, want);
        if (got <= 0) {
            LOG_WRN("Offline: short read at byte %u (got=%d)",
                    bytes_read_total, (int)got);
            break;
        }
        int samples = (int)(got / sizeof(int16_t));

        /* Feed both pipelines.  These calls will synchronously invoke
         * heart_frame_cb / lung_frame_cb when a frame is ready, which
         * fs_write directly to fh / fl. */
        dsp_mfcc_feed_chunk(&heart_pipeline, read_buf, samples);
        dsp_mfcc_feed_chunk(&lung_pipeline,  read_buf, samples);

        bytes_read_total += (uint32_t)got;
        chunk_idx++;

        /* Yield periodically so the BLE stack and watchdog can run.
         * BLE supervision timeout is ~400 ms; processing 4 chunks
         * (~256 ms of audio, ~128 ms of CPU at ~50% load) keeps us
         * well under that ceiling. */
        if ((chunk_idx & 3) == 0) {
            k_yield();
        }
    }

    /* Disengage offline-callback dispatch BEFORE closing files, so
     * any stray late callback can't write to a closed handle. */
    offline_heart_fp = NULL;
    offline_lung_fp  = NULL;

    fs_close(&fa);
    fs_close(&fh);
    fs_close(&fl);

    int64_t elapsed_ms = k_uptime_delta(&t_start);
    LOG_INF("Offline MFCC done: %u B audio in %lld ms (%u chunks)",
            bytes_read_total, elapsed_ms, chunk_idx);
    LOG_INF("Offline frames: heart=%u  lung=%u",
            heart_frame_count, lung_frame_count);

    return 0;
}
#endif /* DSP_OFFLINE */

#endif /* USE_SD */

/* ══════════════════════════════════════════════════════════════════
 * SAADC EVENT HANDLER
 *
 * Kept intentionally minimal — only ISR-safe operations:
 *   1. DC removal (simple IIR, no FPU division)
 *   2. ring_buf_put into dsp_ring   -> wakes DSP thread
 *   3. ring_buf_put into audio_sd_ring (USE_SD)
 *   4. ring_buf_put into audio_ring  -> wakes BLE TX thread
 *
 * NO dsp_mfcc_feed_chunk() here. All RFFT/mel/DCT runs in dsp_thread.
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

        /* ── Step 2: Push to DSP ring (ISR-safe) — DSP thread does the math ── */
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
        /* ── Step 3: Push raw audio to SD audio ring (ISR-safe) ── */
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

        /* ── Step 4: Push to BLE audio ring ── */
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
static int saadc_init(void)
{
    nrfx_err_t err;
    nrfx_saadc_uninit();
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
    return 0;
}

static int saadc_start_streaming(void)
{
    nrfx_err_t err;
    saadc_dma_overruns = 0;
    dc_estimate        = 0;

    dsp_mfcc_reset(&heart_pipeline);
    dsp_mfcc_reset(&lung_pipeline);
    heart_frame_count = 0;
    lung_frame_count  = 0;

    /* [FIX 2] Set next_dma_buf BEFORE saadc_init() to prevent the race
     * where EVT_BUF_REQ fires between init and the assignment. */
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
    nrfx_saadc_uninit();
    LOG_INF("SAADC stopped (overruns=%u, ble_drops=%u, sd_drops=%u)",
            saadc_dma_overruns,
            (uint32_t)atomic_get(&ring_drops)
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
static const struct gpio_dt_spec red_led   = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
static const struct gpio_dt_spec green_led = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);
static const struct gpio_dt_spec blue_led  = GPIO_DT_SPEC_GET(DT_ALIAS(led2), gpios);

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
 * RECORD AND STREAM
 * ══════════════════════════════════════════════════════════════════ */
static void record_and_stream(void)
{
    const uint32_t total_halves =
        (TOTAL_AUDIO_SAMPLES + HALF_BUF_SAMPLES - 1) / HALF_BUF_SAMPLES;

    /* ── Reset BLE audio ring ── */
    ring_buf_reset(&audio_ring);
    atomic_set(&ring_drops, 0);
    ring_high_water = 0;
    tx_seq          = 0;
    k_sem_reset(&half_produced_sem);
    k_sem_reset(&tx_done_sem);

    /* ── Reset DSP ring ── */
    ring_buf_reset(&dsp_ring);
    k_sem_reset(&dsp_done_sem);

#if USE_SD
    /* ── Reset SD rings ── */
    ring_buf_reset(&audio_sd_ring);
    ring_buf_reset(&heart_mfcc_ring);
    ring_buf_reset(&lung_mfcc_ring);
    atomic_set(&sd_ring_drops, 0);
    audio_ring_high_water = 0;
    heart_ring_high_water = 0;
    lung_ring_high_water  = 0;
    k_sem_reset(&sd_done_sem);

    if (!sd_mounted) {
        LOG_ERR("SD not mounted — aborting REC");
        bt_nus_send(NULL, "ERR:NOSD", 8);
        led_error_flash(led_set_yellow);
        return;
    }

    /* [FIX 7b] Ensure mount-point directory entry exists.
     * Harmless on healthy cards (-EEXIST is silently ignored). */
    
    /*
    int mkdir_rc = fs_mkdir(SD_CARD_MOUNT_POINT);
    if (mkdir_rc < 0 && mkdir_rc != -EEXIST) {
        LOG_WRN("fs_mkdir(%s) returned %d (non-fatal)",
                SD_CARD_MOUNT_POINT, mkdir_rc);
    }
    */

    /* Open three files, truncating any previous recording.
     * [FIX 7c] Each fs_open is a separate call with its own error log. */
    fs_unlink(AUDIO_FILE_PATH);
    fs_unlink(HEART_MFCC_FILE_PATH);
    fs_unlink(LUNG_MFCC_FILE_PATH);

    fs_file_t_init(&sd_audio_file);
    fs_file_t_init(&sd_heart_file);
    fs_file_t_init(&sd_lung_file);

    int rc_audio = fs_open(&sd_audio_file, AUDIO_FILE_PATH,
                            FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc_audio < 0) {
        LOG_ERR("fs_open(%s) failed: %d", AUDIO_FILE_PATH, rc_audio);
        bt_nus_send(NULL, "ERR:SD_OPEN", 11);
        led_error_flash(led_set_yellow);
        return;
    }

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
    /* [FIX 8] sd_data_sem give MOVED — see below */
#endif

    /* Send audio length to host -- only needed when live-streaming.
     * When BLE_AUDIO_LIVE=0 the START: header is sent later as part of
     * the post-recording file streaming loop. */
#if BLE_AUDIO_LIVE
    char hdr[32];
    snprintf(hdr, sizeof(hdr), "START:%u\n", (uint32_t)TOTAL_AUDIO_BYTES);
    bt_nus_send(NULL, hdr, strlen(hdr));
    k_sleep(K_MSEC(10));
#endif

    led_set_cyan();

    /* [FIX 8] Set analog_recording = true BEFORE waking either drainer thread.
     * Previously the SD writer was woken before this flag was set; it saw
     * empty rings + analog_recording==false, decided the recording was
     * already over, and exited within 16 ms. The SAADC then dropped 10 s
     * of audio (sd_drops=4589) because nothing was draining audio_sd_ring.
     * Same race existed for the DSP thread but it was less visible because
     * dsp_data_sem is given again from the ISR on every half-buffer. */
    analog_recording = true;

#if USE_SD
    k_sem_give(&sd_data_sem);   /* now safe: writer will see recording in progress */
#endif
#if !DSP_OFFLINE
    /* In offline mode the DSP thread never processes anything;
     * it stays asleep on dsp_data_sem forever.  Only wake it in
     * online mode. */
    k_sem_give(&dsp_data_sem);
#endif

    if (saadc_start_streaming() != 0) {
        analog_recording = false;
#if USE_SD
        fs_close(&sd_audio_file);
        fs_close(&sd_heart_file);
        fs_close(&sd_lung_file);
#endif
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:SAADC", 9);
        return;
    }

    LOG_INF("Recording %d s @ %d Hz (USE_SD=%s)...", DURATION_S, SAMPLING_RATE,
            USE_SD ? "true" : "false");

    for (uint32_t h = 0; h < total_halves; h++) {
        if (k_sem_take(&half_produced_sem, K_MSEC(500)) != 0) {
            LOG_ERR("SAADC timeout at half-buffer %u/%u", h, total_halves);
            analog_recording = false;
            saadc_stop_streaming();
#if BLE_AUDIO_LIVE
            k_sem_give(&audio_data_sem);
            k_sem_take(&tx_done_sem, K_MSEC(10000));
#endif
            bt_nus_send(NULL, "ERR:TIMEOUT", 11);
            return;
        }
    }

    saadc_stop_streaming();
    analog_recording = false;

#if BLE_AUDIO_LIVE
    /* Wake BLE TX thread so it can drain audio_ring and send "finished\n". */
    k_sem_give(&audio_data_sem);

    if (k_sem_take(&tx_done_sem, K_MSEC(10000)) != 0) {
        LOG_WRN("BLE TX did not finish within 10 s");
    }
#endif

#if !DSP_OFFLINE
    if (k_sem_take(&dsp_done_sem, K_MSEC(30000)) != 0) {
        LOG_WRN("DSP thread did not finish within 30 s — MFCC may be incomplete");
    }
#endif

#if USE_SD
    LOG_INF("Waiting for SD writer to flush and close...");
    bool sd_ok = false;
    if (k_sem_take(&sd_done_sem, K_MSEC(15000)) != 0) {
        LOG_ERR("SD writer did not finish within 15 s — file may be truncated");
    } else {
        LOG_INF("SD files closed (audio checksum=0x%08X, drops=%u)",
                sd_checksum, (uint32_t)atomic_get(&sd_ring_drops));
        sd_ok = true;
#if BLE_AUDIO_LIVE
        /* Live-audio flow: audio was already streamed during recording
         * (terminated by "finished\n"), so SD:OK\n goes here, before
         * the MFCC streams.  When BLE_AUDIO_LIVE=0 we delay this send
         * until after the audio file has been streamed -- see the file
         * streaming loop below. */
        bt_nus_send(NULL, "SD:OK\n", 6);
#endif
    }
#endif

#if DSP_OFFLINE && USE_SD
    /* ── Phase 2: Offline MFCC processing ─────────────────────────────
     * SD writer has closed all three files; audio file contains all
     * 161792 B of captured PCM (plus a 4-byte XOR checksum at the tail
     * which we skip).  Re-read the audio file from SD, feed it through
     * both MFCC pipelines, and write coefficients directly to the
     * heart/lung files.  No real-time constraint -- runs as fast as
     * the M4F + SD reads allow. */
    if (sd_ok) {
        int rc = process_audio_offline();
        if (rc < 0) {
            LOG_ERR("Offline MFCC processing failed: %d", rc);
            bt_nus_send(NULL, "ERR:DSP", 7);
            led_error_flash(led_set_yellow);
            led_set_green();
            return;
        }
    } else {
        LOG_ERR("Skipping offline MFCC because SD writer didn't finish");
    }
#endif

    led_set_purple();

    int hf = dsp_mfcc_finish(&heart_pipeline);
    int lf = dsp_mfcc_finish(&lung_pipeline);

    report_ram_usage();

    if (hf <= 0 || lf <= 0) {
        LOG_ERR("MFCC pipeline error: heart=%d lung=%d", hf, lf);
        bt_nus_send(NULL, "ERR:DSP", 7);
        led_error_flash(led_set_yellow);
        led_set_green();
        return;
    }

#if USE_SD
    /* ── Stream SD files back over BLE ────────────────────────────────
     *
     * Order: [audio (if !BLE_AUDIO_LIVE)] -> heart MFCC -> lung MFCC
     *
     * The audio entry uses the same START:/finished\n protocol as the
     * old live-streaming path so the receiver's state machine doesn't
     * need to change -- the only difference is that all audio bytes
     * arrive AFTER the recording window instead of during it.
     *
     * Each entry knows its on-disk byte count up front:
     *   - audio: TOTAL_AUDIO_BYTES (= 160000 for 10 s @ 8 kHz / int16)
     *           NOTE: the file on SD also has a 4-byte XOR checksum
     *           appended (see sd_writer_thread_fn).  We deliberately
     *           do NOT send the checksum byte -- the receiver expects
     *           exactly TOTAL_AUDIO_BYTES of PCM.
     *   - heart: hf * n_mfcc * sizeof(float)
     *   - lung:  lf * n_mfcc * sizeof(float)
     */
    struct file_stream_entry {
        const char *start_fmt;     /* printf-format for START header   */
        const char *end_msg;       /* trailer message (or NULL)        */
        const char *path;          /* SD file path                     */
        uint32_t    file_bytes;    /* exact byte count to send         */
    };

    struct file_stream_entry stream_list[3];
    int stream_count = 0;
    (void)sd_ok;  /* may be unused when BLE_AUDIO_LIVE=1 */

#if !BLE_AUDIO_LIVE
    /* Audio file -- only when live-streaming is disabled */
    stream_list[stream_count++] = (struct file_stream_entry){
        .start_fmt  = "START:%u\n",
        .end_msg    = "finished\n",
        .path       = AUDIO_FILE_PATH,
        .file_bytes = (uint32_t)TOTAL_AUDIO_BYTES,
    };
#endif

    stream_list[stream_count++] = (struct file_stream_entry){
        .start_fmt  = "MFCC_HEART_START:%u\n",
        .end_msg    = "MFCC_HEART_END\n",
        .path       = HEART_MFCC_FILE_PATH,
        .file_bytes = (uint32_t)hf * (uint32_t)heart_pipeline.cfg->n_mfcc * sizeof(float),
    };
    stream_list[stream_count++] = (struct file_stream_entry){
        .start_fmt  = "MFCC_LUNG_START:%u\n",
        .end_msg    = "MFCC_LUNG_END\n",
        .path       = LUNG_MFCC_FILE_PATH,
        .file_bytes = (uint32_t)lf * (uint32_t)lung_pipeline.cfg->n_mfcc * sizeof(float),
    };

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

        uint8_t  pkt[251];
        uint16_t seq = 0;
        uint16_t payload = nus_chunk_size - CHUNK_HEADER_BYTES;
        uint32_t bytes_sent = 0;

        /* Read exactly e->file_bytes; do not send any trailing data
         * (e.g. the XOR checksum at the tail of the audio file). */
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
        /* If this was the audio entry (always index 0 when live audio
         * is disabled), now is the right time to send SD:OK\n -- after
         * the audio "finished\n" trailer and before the MFCC streams.
         * Matches the receiver's expected sequence: audio, finished,
         * SD:OK, heart, lung. */
        if (fi == 0 && sd_ok &&
            strcmp(e->path, AUDIO_FILE_PATH) == 0) {
            bt_nus_send(NULL, "SD:OK\n", 6);
            k_sleep(K_MSEC(10));
        }
#endif
    }
#endif

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

    /* BLE TX thread */
    k_thread_create(&ble_tx_thread_data, ble_tx_stack,
                    K_THREAD_STACK_SIZEOF(ble_tx_stack),
                    ble_tx_thread_fn, NULL, NULL, NULL,
                    BLE_TX_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&ble_tx_thread_data, "ble_tx");

    /* DSP thread — drains dsp_ring, runs MFCC pipelines off ISR */
    k_thread_create(&dsp_thread_data, dsp_thread_stack,
                    K_THREAD_STACK_SIZEOF(dsp_thread_stack),
                    dsp_thread_fn, NULL, NULL, NULL,
                    DSP_THREAD_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&dsp_thread_data, "dsp");

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
        led_error_flash(led_set_yellow);
    }
#endif

    led_set_red();
    LOG_INF("AcoustEEEcare v6.8 ready (USE_SD=%s) — waiting for BLE connection",
            USE_SD ? "true" : "false");

    while (true) {
        k_sleep(K_MSEC(100));
        if (is_connected && mtu_exchanged && start_recording && !analog_recording) {
            start_recording = false;
            LOG_INF("REC command — starting %d s @ %d Hz", DURATION_S, SAMPLING_RATE);
            record_and_stream();
        }
    }

    return 0;
}

/*
 * ════════════════════════════════════════════════════════════════════
 * REQUIRED prj.conf (USE_SD true — already set in your prj.conf)
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
 * OVERLAY NOTE
 * ════════════════════════════════════════════════════════════════════
 *
 * spi-max-frequency = <4000000> is safer for broad SD compatibility.
 * 10000000 (10 MHz) works with most cards but some cheaper cards
 * misbehave during write bursts above 4–8 MHz.  If SD write errors
 * appear during recording, drop to 4000000 first before debugging
 * firmware.
 *
 * ════════════════════════════════════════════════════════════════════
 * MEMORY BUDGET (approximate, v6.8)
 * ════════════════════════════════════════════════════════════════════
 *
 *   Zephyr kernel + BLE stack          ~90 KB
 *   DSP scratch (dsp_mfcc.c statics)    ~7 KB
 *   audio_ring  (BLE TX)                16 KB
 *   dsp_ring    (ISR->DSP thread)       32 KB
 *   audio_sd_ring                       32 KB
 *   heart_mfcc_ring                      8 KB
 *   lung_mfcc_ring                      16 KB
 *   ping_pong[2][512]                    2 KB
 *   dsp_pop_buf[512]                     1 KB
 *   BLE TX thread stack                  2 KB
 *   DSP thread stack                     4 KB
 *   SD writer thread stack               2 KB
 *   sd_write_buf                       512  B
 *   fat_fs (FATFS work area)            ~4 KB
 *   Remaining headroom                 ~38 KB
 * ════════════════════════════════════════════════════════════════════
 */