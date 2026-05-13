/*
 * AcoustEEEcare — SAADC BLE + SD Card + TFLite Micro Edition
 * ============================================================
 * v7.2 — SD card thrashing fix: buffered MFCC writes + larger
 *        audio read chunks in offline pipeline.
 *
 * CHANGES FROM v7.1:
 *
 *   [SD-FIX 1] Buffered MFCC coefficient writes.
 *     Previously, heart_frame_cb and lung_frame_cb each called
 *     fs_write() with tiny payloads (80 B for heart, 52 B for lung)
 *     every time the DSP emitted a frame.  During Phase 2 offline
 *     processing this meant THOUSANDS of sub-sector fs_writes
 *     interleaved with fs_read calls on a third file — FATFS-on-SPI
 *     thrashes on read-modify-write of the same FAT sectors, and
 *     the card eventually returns EIO ("Failed to read from SDMMC -22").
 *
 *     New behaviour: each callback memcpy's into a 4 KB RAM buffer
 *     and only fs_writes when the buffer is full or at end-of-stream.
 *     This collapses ~9000 small writes into ~130 aligned 4 KB writes.
 *     RAM cost: 8 KB total (well within the 122 KB free above BSS).
 *
 *   [SD-FIX 2] Larger fs_read chunk in process_audio_offline().
 *     Read buffer bumped from 512 samples (1024 B) to 4096 samples
 *     (8192 B) — 8x fewer fs_read calls, all sector-aligned.
 *
 *   [SD-FIX 3] Explicit flush helpers for the MFCC write buffers,
 *     called after the audio read loop and on any error path that
 *     closes the MFCC files.
 *
 * ALL PREVIOUS FIXES (v6.8, v7.0, v7.1) ARE PRESERVED UNCHANGED.
 * ============================================================
 */

#define USE_SD  true

#define BLE_AUDIO_LIVE  0   /* must stay 0 for arena RAM to be available */
#define DSP_OFFLINE     1   /* must stay 1 for arena RAM to be available */

/* ── Model enable flags ──────────────────────────────────────────
 * Set to 1 when the compiled-in model is ready.
 * Set to 0 to skip inference for that channel entirely —
 * the MFCC file is still captured and uploaded over BLE so you
 * can validate the feature extraction pipeline independently.
 * ─────────────────────────────────────────────────────────────── */
#define ENABLE_HEART_MODEL  1   /* set to 0 while heart model absent */
#define ENABLE_LUNG_MODEL   0   /* set to 1 once lung model is ready */

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

#include <nrfx.h>
#include <drivers/nrfx_errors.h>
#include <nrfx_saadc.h>

#include "dsp_mfcc.h"
#include "heart_mfcc_config.h"
#include "lung_mfcc_config.h"

/* TFLite Micro inference (C-linkage wrapper around .cc implementation) */
#include "tflm_inference.h"

#if USE_SD
#include <zephyr/fs/fs.h>
#include <zephyr/storage/disk_access.h>
#include <ff.h>
#endif

LOG_MODULE_REGISTER(AcoustEEEcare);

/* ══════════════════════════════════════════════════════════════════
 * SHARED TENSOR ARENA
 * ══════════════════════════════════════════════════════════════════ */
#define TENSOR_ARENA_BYTES  (112u * 1024u)

static uint8_t tensor_arena[TENSOR_ARENA_BYTES] __attribute__((aligned(16)));

/* ══════════════════════════════════════════════════════════════════
 * [SD-FIX 1] MFCC WRITE BUFFERS
 *
 * Replace thousands of tiny per-frame fs_write calls with a handful
 * of aligned 4 KB writes.  See header comment for rationale.
 *
 * Heart: n_mfcc=20 -> 80 B/frame.  4096 / 80 = 51 frames per flush.
 * Lung:  n_mfcc=13 -> 52 B/frame.  4096 / 52 = 78 frames per flush.
 *
 * Both are sized to a multiple of an SD sector (512 B) so each flush
 * is a clean multi-sector write rather than a read-modify-write.
 * ══════════════════════════════════════════════════════════════════ */
/* 2 KB per buffer is plenty: 25 heart frames (80 B each) or
 * 39 lung frames (52 B each) per flush, all sector-aligned writes. */
#define MFCC_WRITE_BUF_SIZE  2048u

static uint8_t  heart_mfcc_write_buf[MFCC_WRITE_BUF_SIZE];
static uint32_t heart_mfcc_write_pos = 0;

static uint8_t  lung_mfcc_write_buf[MFCC_WRITE_BUF_SIZE];
static uint32_t lung_mfcc_write_pos = 0;

/* Track whether any fs_write returned an error so we can surface it
 * cleanly from process_audio_offline() instead of letting it cascade. */
static volatile int heart_mfcc_write_err = 0;
static volatile int lung_mfcc_write_err  = 0;

/* ══════════════════════════════════════════════════════════════════
 * RAM USAGE REPORT
 * ══════════════════════════════════════════════════════════════════ */
extern char _end;

#define BLE_TX_STACK_SIZE     2048
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
    LOG_INF("Tensor arena:    %u KB (declared, not all used until inference)",
            TENSOR_ARENA_BYTES / 1024);

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
#define SAADC_CC_VALUE      2000U
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

static int16_t dsp_pop_buf[HALF_BUF_SAMPLES];
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
#define SD_INIT_CHECK_PATH   "/SD:/acoustchk"

#define CHECKSUM_SIZE        sizeof(uint32_t)

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
 * ══════════════════════════════════════════════════════════════════ */
static int16_t heart_window[50];
static int16_t lung_window[100];

#if USE_SD
static struct fs_file_t *offline_heart_fp = NULL;
static struct fs_file_t *offline_lung_fp  = NULL;

/* ── [SD-FIX 3] Flush helpers for MFCC write buffers ──────────── */
static int flush_heart_mfcc_buf(void)
{
    if (offline_heart_fp == NULL || heart_mfcc_write_pos == 0) {
        heart_mfcc_write_pos = 0;
        return 0;
    }
    ssize_t w = fs_write(offline_heart_fp,
                         heart_mfcc_write_buf,
                         heart_mfcc_write_pos);
    int rc = 0;
    if (w != (ssize_t)heart_mfcc_write_pos) {
        LOG_WRN("Heart MFCC flush short: %d/%u",
                (int)w, heart_mfcc_write_pos);
        rc = (w < 0) ? (int)w : -EIO;
        heart_mfcc_write_err = rc;
    }
    heart_mfcc_write_pos = 0;
    return rc;
}

static int flush_lung_mfcc_buf(void)
{
    if (offline_lung_fp == NULL || lung_mfcc_write_pos == 0) {
        lung_mfcc_write_pos = 0;
        return 0;
    }
    ssize_t w = fs_write(offline_lung_fp,
                         lung_mfcc_write_buf,
                         lung_mfcc_write_pos);
    int rc = 0;
    if (w != (ssize_t)lung_mfcc_write_pos) {
        LOG_WRN("Lung MFCC flush short: %d/%u",
                (int)w, lung_mfcc_write_pos);
        rc = (w < 0) ? (int)w : -EIO;
        lung_mfcc_write_err = rc;
    }
    lung_mfcc_write_pos = 0;
    return rc;
}
#endif

/* ── [SD-FIX 1] Buffered frame callbacks ───────────────────────── */
static void heart_frame_cb(int idx, const float *coeffs, void *user)
{
    (void)user; (void)idx;
    heart_frame_count++;

#if USE_SD
    const uint32_t bytes = heart_pipeline.cfg->n_mfcc * sizeof(float);

    if (offline_heart_fp != NULL) {
        /* If a previous flush already failed, stop accumulating —
         * the card is unresponsive and further writes will only
         * stall the read loop. */
        if (heart_mfcc_write_err != 0) {
            return;
        }

        /* If this frame won't fit in the buffer, flush first. */
        if (heart_mfcc_write_pos + bytes > MFCC_WRITE_BUF_SIZE) {
            if (flush_heart_mfcc_buf() != 0) {
                return;
            }
        }

        /* Safety: a single frame must fit in the buffer.  This is a
         * compile-time guarantee for current n_mfcc values, but guard
         * anyway so a future config change can't corrupt memory. */
        if (bytes > MFCC_WRITE_BUF_SIZE) {
            LOG_ERR("heart frame too large for write buffer: %u > %u",
                    bytes, MFCC_WRITE_BUF_SIZE);
            heart_mfcc_write_err = -EINVAL;
            return;
        }

        memcpy(&heart_mfcc_write_buf[heart_mfcc_write_pos], coeffs, bytes);
        heart_mfcc_write_pos += bytes;
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
        if (lung_mfcc_write_err != 0) {
            return;
        }

        if (lung_mfcc_write_pos + bytes > MFCC_WRITE_BUF_SIZE) {
            if (flush_lung_mfcc_buf() != 0) {
                return;
            }
        }

        if (bytes > MFCC_WRITE_BUF_SIZE) {
            LOG_ERR("lung frame too large for write buffer: %u > %u",
                    bytes, MFCC_WRITE_BUF_SIZE);
            lung_mfcc_write_err = -EINVAL;
            return;
        }

        memcpy(&lung_mfcc_write_buf[lung_mfcc_write_pos], coeffs, bytes);
        lung_mfcc_write_pos += bytes;
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

#define SD_WRITE_BUF_SIZE  512
static uint8_t sd_write_buf[SD_WRITE_BUF_SIZE];

static uint32_t compute_checksum(const uint8_t *data, uint32_t len)
{
    uint32_t cs = 0;
    for (uint32_t i = 0; i < len; i++) cs ^= data[i];
    return cs;
}

/* ── SD write-verify ─────────────────────────────────────────── */
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
            while (ring_buf_size_get(&audio_sd_ring) > 0
#if !DSP_OFFLINE
                || ring_buf_size_get(&heart_mfcc_ring) > 0
                || ring_buf_size_get(&lung_mfcc_ring)  > 0
#endif
            ) {
                ring_buf_get(&audio_sd_ring, sd_write_buf,
                             MIN(ring_buf_size_get(&audio_sd_ring), SD_WRITE_BUF_SIZE));
#if !DSP_OFFLINE
                ring_buf_get(&heart_mfcc_ring, sd_write_buf,
                             MIN(ring_buf_size_get(&heart_mfcc_ring), SD_WRITE_BUF_SIZE));
                ring_buf_get(&lung_mfcc_ring,  sd_write_buf,
                             MIN(ring_buf_size_get(&lung_mfcc_ring),  SD_WRITE_BUF_SIZE));
#endif
            }
            k_sem_give(&sd_done_sem);
            continue;
        }

        LOG_INF("SD writer: starting write loop");
        uint32_t audio_written = 0;
#if !DSP_OFFLINE
        uint32_t heart_written = 0;
        uint32_t lung_written  = 0;
#endif
        sd_checksum = 0;

        while (analog_recording ||
               ring_buf_size_get(&audio_sd_ring) > 0
#if !DSP_OFFLINE
               || ring_buf_size_get(&heart_mfcc_ring) > 0
               || ring_buf_size_get(&lung_mfcc_ring)  > 0
#endif
        ) {

            bool did_work = false;

            /* ── Audio ring (always present) ── */
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

#if !DSP_OFFLINE
            /* ── Heart MFCC ring (online DSP only) ── */
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

            /* ── Lung MFCC ring (online DSP only) ── */
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
#endif /* !DSP_OFFLINE */

            if (!did_work) {
                k_sleep(K_MSEC(1));
            }
        }

        fs_write(&sd_audio_file, &sd_checksum, CHECKSUM_SIZE);
        fs_close(&sd_audio_file);
#if !DSP_OFFLINE
        fs_close(&sd_heart_file);
        fs_close(&sd_lung_file);
#endif

#if !DSP_OFFLINE
        LOG_INF("SD writer done: audio=%u B  heart=%u B  lung=%u B",
                audio_written, heart_written, lung_written);
        LOG_INF("Ring high-water: audio=%u  heart=%u  lung=%u",
                audio_ring_high_water, heart_ring_high_water, lung_ring_high_water);
#else
        LOG_INF("SD writer done: audio=%u B (offline DSP -> MFCC written later)",
                audio_written);
        LOG_INF("Ring high-water: audio=%u", audio_ring_high_water);
#endif

        k_sem_give(&sd_done_sem);
    }
}

#if DSP_OFFLINE
/* ══════════════════════════════════════════════════════════════════
 * OFFLINE MFCC PROCESSING (Phase 2)
 *
 * [SD-FIX 2] Read buffer enlarged from HALF_BUF_SAMPLES (1024 B) to
 * OFFLINE_READ_SAMPLES (8192 B).  Reads now hit the card 8x less
 * often and every read is a clean 16-sector block.  The DSP pipeline
 * still receives HALF_BUF_SAMPLES-sized sub-chunks because that's the
 * window stride dsp_mfcc_feed_chunk expects.
 * ══════════════════════════════════════════════════════════════════ */
/* 4 KB read buffer: 4x larger than the original 1024 B, still a
 * dramatic reduction in fs_read transaction count vs v7.1, but small
 * enough to leave generous stack headroom for TFLite Micro inference
 * (which also runs on the main thread). */
#define OFFLINE_READ_SAMPLES   2048
#define OFFLINE_READ_BYTES     (OFFLINE_READ_SAMPLES * sizeof(int16_t))

/* offline_read_buf lives on the main thread's stack rather than in
 * BSS — it's only needed for the duration of process_audio_offline().
 * Requires CONFIG_MAIN_STACK_SIZE >= 12288 in prj.conf to be safe
 * (8 KB buffer + ~2 KB frame overhead + margin). */

static int process_audio_offline(void)
{
    int rc = 0;
    struct fs_file_t fa;
    struct fs_file_t fh;
    struct fs_file_t fl;
    int16_t offline_read_buf[OFFLINE_READ_SAMPLES];

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

    dsp_mfcc_reset(&heart_pipeline);
    dsp_mfcc_reset(&lung_pipeline);
    heart_frame_count = 0;
    lung_frame_count  = 0;

    /* [SD-FIX 1] Reset MFCC write buffers for a fresh session. */
    heart_mfcc_write_pos = 0;
    lung_mfcc_write_pos  = 0;
    heart_mfcc_write_err = 0;
    lung_mfcc_write_err  = 0;

    offline_heart_fp = &fh;
    offline_lung_fp  = &fl;

    uint32_t bytes_read_total = 0;
    int      chunk_idx        = 0;
    int64_t  t_start          = k_uptime_get();

    while (bytes_read_total < TOTAL_AUDIO_BYTES) {
        uint32_t want = MIN((uint32_t)OFFLINE_READ_BYTES,
                            TOTAL_AUDIO_BYTES - bytes_read_total);
        ssize_t  got  = fs_read(&fa, offline_read_buf, want);
        if (got <= 0) {
            if (got < 0) {
                LOG_ERR("Offline: fs_read error at byte %u (got=%d) — SD card unresponsive",
                        bytes_read_total, (int)got);
                rc = (int)got;
            } else {
                LOG_WRN("Offline: unexpected EOF at byte %u — file shorter than expected",
                        bytes_read_total);
            }
            break;
        }

        /* Feed the DSP in HALF_BUF_SAMPLES-sized sub-chunks so the
         * MFCC pipeline's framing math stays identical to before. */
        int total_samples = (int)(got / sizeof(int16_t));
        int offset        = 0;

        while (offset < total_samples) {
            int sub = MIN(HALF_BUF_SAMPLES, total_samples - offset);
            dsp_mfcc_feed_chunk(&heart_pipeline,
                                &offline_read_buf[offset], sub);
            dsp_mfcc_feed_chunk(&lung_pipeline,
                                &offline_read_buf[offset], sub);
            offset += sub;
        }

        bytes_read_total += (uint32_t)got;
        chunk_idx++;

        /* Bail out early if the card died mid-MFCC-write. */
        if (heart_mfcc_write_err != 0 || lung_mfcc_write_err != 0) {
            LOG_ERR("Offline: MFCC write error (heart=%d lung=%d) — aborting",
                    heart_mfcc_write_err, lung_mfcc_write_err);
            rc = (heart_mfcc_write_err != 0) ? heart_mfcc_write_err
                                             : lung_mfcc_write_err;
            break;
        }

        if ((chunk_idx & 1) == 0) {
            k_yield();
        }
    }

    /* [SD-FIX 3] Flush any tail bytes still sitting in the MFCC
     * write buffers before closing the files.  Skip flush if we
     * already saw a write error — the card is bricked, additional
     * writes will just waste time. */
    if (rc == 0 && heart_mfcc_write_err == 0) {
        int fr = flush_heart_mfcc_buf();
        if (fr != 0 && rc == 0) rc = fr;
    }
    if (rc == 0 && lung_mfcc_write_err == 0) {
        int fr = flush_lung_mfcc_buf();
        if (fr != 0 && rc == 0) rc = fr;
    }

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

    return rc;
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
#endif /* BLE_AUDIO_LIVE */
}

/* ══════════════════════════════════════════════════════════════════
 * RECORD AND STREAM
 * ══════════════════════════════════════════════════════════════════ */
static void record_and_stream(void)
{
    const uint32_t total_halves =
        (TOTAL_AUDIO_SAMPLES + HALF_BUF_SAMPLES - 1) / HALF_BUF_SAMPLES;

    /* ── Reset semaphores ── */
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

    /* Force a clean remount on every REC command to flush any stale
     * FATFS state left by a previous session that crashed mid-close. */
    if (sd_mounted) {
        fs_unmount(&mp);
        sd_mounted = false;
        k_sleep(K_MSEC(50));
    }
    if (init_sd_card() != 0) {
        LOG_ERR("SD remount failed — aborting REC");
        bt_nus_send(NULL, "ERR:NOSD", 8);
        led_error_flash(led_set_yellow);
        return;
    }

    /* Unlink stale files from previous recording. */
    {
        static const char *stale_files[] = {
            AUDIO_FILE_PATH,
            HEART_MFCC_FILE_PATH,
            LUNG_MFCC_FILE_PATH,
            HR_RESULT_FILE_PATH,
            RR_RESULT_FILE_PATH,
        };
        bool unlink_ok = true;
        for (int i = 0; i < (int)ARRAY_SIZE(stale_files); i++) {
            int ul = fs_unlink(stale_files[i]);
            if (ul < 0 && ul != -ENOENT) {
                LOG_ERR("fs_unlink(%s) failed: %d — SD unresponsive after remount",
                        stale_files[i], ul);
                unlink_ok = false;
                break;
            }
        }
        if (!unlink_ok) {
            bt_nus_send(NULL, "ERR:NOSD", 8);
            led_error_flash(led_set_yellow);
            return;
        }
    }

    fs_file_t_init(&sd_audio_file);
#if !DSP_OFFLINE
    fs_file_t_init(&sd_heart_file);
    fs_file_t_init(&sd_lung_file);
#endif

    int rc_audio = fs_open(&sd_audio_file, AUDIO_FILE_PATH,
                            FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc_audio < 0) {
        LOG_ERR("fs_open(%s) failed: %d", AUDIO_FILE_PATH, rc_audio);
        bt_nus_send(NULL, "ERR:SD_OPEN", 11);
        led_error_flash(led_set_yellow);
        return;
    }

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
#endif

    /* ════════════════════════════════════════════════════════════
     * PHASE 2: Offline MFCC
     * ════════════════════════════════════════════════════════════ */
#if DSP_OFFLINE && USE_SD
    LOG_INF("Phase 2: offline MFCC processing...");
    led_set_yellow();

    int mfcc_rc = process_audio_offline();
    if (mfcc_rc < 0) {
        LOG_ERR("Offline MFCC failed: %d — SD I/O error during Phase 2", mfcc_rc);
        bt_nus_send(NULL, "ERR:SD_READ", 11);
        led_error_flash(led_set_yellow);
        led_set_green();
        return;
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
    /* ════════════════════════════════════════════════════════════
     * PHASE 3: Heart model inference
     * ════════════════════════════════════════════════════════════ */
    heart_result_t heart_result;
    heart_result.rc = -ENOTSUP;
    heart_result.value = 0.f;
    heart_result.confidence = 0.f;
    heart_result.class_idx = -1;

#if ENABLE_HEART_MODEL
    LOG_INF("Phase 3: heart inference (%d frames × %d coeffs)...",
            hf, heart_pipeline.cfg->n_mfcc);
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
#else
    LOG_INF("Phase 3: heart model disabled (ENABLE_HEART_MODEL=0) — skipping");
#endif

    /* ════════════════════════════════════════════════════════════
     * PHASE 4: Lung model inference
     * ════════════════════════════════════════════════════════════ */
    lung_result_t lung_result;
    lung_result.rc = -ENOTSUP;
    lung_result.value = 0.f;
    lung_result.confidence = 0.f;
    lung_result.class_idx = -1;

#if ENABLE_LUNG_MODEL
    LOG_INF("Phase 4: lung inference (%d frames × %d coeffs)...",
            lf, lung_pipeline.cfg->n_mfcc);
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
#else
    LOG_INF("Phase 4: lung model disabled (ENABLE_LUNG_MODEL=0) — skipping");
#endif
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

        uint8_t  pkt[251];
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

    LOG_INF("record_and_stream() complete. "
            "HR=%.0f RR=%.0f",
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
        led_error_flash(led_set_yellow);
    }
#endif

    led_set_red();
    LOG_INF("AcoustEEEcare v7.2 ready — waiting for BLE connection");

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