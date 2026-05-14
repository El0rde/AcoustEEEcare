/*
 * AcoustEEEcare — SAADC BLE + SD Card Edition
 * ============================================================
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

#define USE_SD  true   /* set false for BLE-only mode */

/* ══════════════════════════════════════════════════════════════════
 * INCLUDES
 * ══════════════════════════════════════════════════════════════════ */
#include <stdint.h>
#include <string.h>
#include <stdio.h>
// #include <math.h>

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

/* SD writer write buffer — 512 bytes aligns to FAT sector size */
#define SD_WRITE_BUF_SIZE  512
static uint8_t sd_write_buf[SD_WRITE_BUF_SIZE];

/* XOR checksum over raw bytes */
static uint32_t compute_checksum(const uint8_t *data, uint32_t len)
{
    uint32_t cs = 0;
    for (uint32_t i = 0; i < len; i++) cs ^= data[i];
    return cs;
}

/* ── SD card init ─────────────────────────────────────────────── */
static int init_sd_card(void)
{
    static const char *disk_pdrv = "SD";
    uint64_t memory_size_mb;
    uint32_t block_count, block_size;
    int      ret;

    LOG_INF("=== SD INIT START ===");
    LOG_INF("disk_pdrv = \"%s\"", disk_pdrv);

    /* ── Step 1: disk_access_init with retry ── */
    for (int attempt = 0; attempt < 5; attempt++) {
        LOG_INF("[SD] disk_access_init attempt %d/5...", attempt + 1);
        ret = disk_access_init(disk_pdrv);
        LOG_INF("[SD] disk_access_init returned: %d", ret);
        if (ret == 0) {
            LOG_INF("[SD] disk_access_init OK on attempt %d", attempt + 1);
            break;
        }
        LOG_WRN("[SD] attempt %d failed (ret=%d), waiting 500 ms...", attempt + 1, ret);
        k_sleep(K_MSEC(500));
    }

    if (ret != 0) {
        LOG_ERR("[SD] disk_access_init FAILED after all retries (last ret=%d)", ret);
        LOG_ERR("[SD] Possible causes: card absent, SPI wiring fault, "
                "CS pin wrong, card not 3.3 V tolerant");
        return -1;
    }

    /* ── Step 2: DISK_IOCTL_GET_SECTOR_COUNT ── */
    LOG_INF("[SD] querying sector count...");
    ret = disk_access_ioctl(disk_pdrv, DISK_IOCTL_GET_SECTOR_COUNT, &block_count);
    LOG_INF("[SD] DISK_IOCTL_GET_SECTOR_COUNT ret=%d, block_count=%u", ret, block_count);
    if (ret != 0) {
        LOG_ERR("[SD] Cannot get sector count (ret=%d)", ret);
        return -1;
    }

    /* ── Step 3: DISK_IOCTL_GET_SECTOR_SIZE ── */
    LOG_INF("[SD] querying sector size...");
    ret = disk_access_ioctl(disk_pdrv, DISK_IOCTL_GET_SECTOR_SIZE, &block_size);
    LOG_INF("[SD] DISK_IOCTL_GET_SECTOR_SIZE ret=%d, block_size=%u", ret, block_size);
    if (ret != 0) {
        LOG_ERR("[SD] Cannot get sector size (ret=%d)", ret);
        return -1;
    }

    /* Sanity-check the values before using them */
    if (block_size == 0 || block_size > 4096) {
        LOG_ERR("[SD] Suspicious block_size=%u — card may not have initialised "
                "correctly", block_size);
        return -1;
    }
    if (block_count == 0) {
        LOG_ERR("[SD] block_count=0 — card reported empty, init likely failed silently");
        return -1;
    }

    memory_size_mb = (uint64_t)block_count * block_size / (1024 * 1024);
    LOG_INF("[SD] Card geometry: %u sectors x %u B = %u MB",
            block_count, block_size, (uint32_t)memory_size_mb);

    /* ── Step 4: fs_mount ── */
    LOG_INF("[SD] fs_mount: type=%d mnt_point=\"%s\" fs_data=%p",
            mp.type, mp.mnt_point, mp.fs_data);
    LOG_INF("[SD] fat_fs address: %p (size=%u B)", (void *)&fat_fs, (uint32_t)sizeof(fat_fs));

    ret = fs_mount(&mp);
    LOG_INF("[SD] fs_mount returned: %d", ret);

    if (ret != 0) {
        LOG_ERR("[SD] fs_mount FAILED (ret=%d)", ret);
        LOG_ERR("[SD] Possible causes: card not FAT32 formatted, "
                "corrupted FAT, card needs full format (not quick)");

        /*
         * Attempt DISK_IOCTL_CTRL_SYNC to see if the underlying driver
         * is at least responsive after a failed mount.
         */
        int sync_ret = disk_access_ioctl(disk_pdrv, DISK_IOCTL_CTRL_SYNC, NULL);
        LOG_INF("[SD] post-failure DISK_IOCTL_CTRL_SYNC ret=%d "
                "(0=driver alive, non-0=driver also dead)", sync_ret);
        return -1;
    }

    LOG_INF("[SD] fs_mount OK");

    /* ── Step 5: fs_statvfs — check free space ── */
    struct fs_statvfs sbuf;
    ret = fs_statvfs(SD_CARD_MOUNT_POINT, &sbuf);
    if (ret != 0) {
        LOG_WRN("[SD] fs_statvfs failed (ret=%d) — mount OK but FS may be "
                "damaged", ret);
    } else {
        LOG_INF("[SD] statvfs: f_bsize=%lu f_frsize=%lu f_blocks=%lu f_bfree=%lu",
                sbuf.f_bsize, sbuf.f_frsize, sbuf.f_blocks, sbuf.f_bfree);
        uint32_t free_mb = (uint32_t)((uint64_t)sbuf.f_bfree * sbuf.f_frsize
                                      / (1024 * 1024));
        LOG_INF("[SD] Free space: ~%u MB", free_mb);
        if (sbuf.f_bfree == 0) {
            LOG_WRN("[SD] Card appears FULL — writes will fail");
        }
    }

    /* ── Step 6: probe write/read roundtrip ── */
    LOG_INF("[SD] probe: writing test file " SD_CARD_MOUNT_POINT "/test.tmp ...");
    struct fs_file_t probe_file;
    fs_file_t_init(&probe_file);
    ret = fs_open(&probe_file, SD_CARD_MOUNT_POINT "/test.tmp",
                  FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    LOG_INF("[SD] probe fs_open ret=%d", ret);

    if (ret == 0) {
        static const uint8_t probe_data[] = { 0xAC, 0x0E, 0x5E, 0xEC };
        ssize_t wr = fs_write(&probe_file, probe_data, sizeof(probe_data));
        LOG_INF("[SD] probe fs_write ret=%d (expected %u)", (int)wr,
                (uint32_t)sizeof(probe_data));
        fs_close(&probe_file);

        /* Re-open and read back */
        ret = fs_open(&probe_file, SD_CARD_MOUNT_POINT "/test.tmp", FS_O_READ);
        LOG_INF("[SD] probe re-open for read ret=%d", ret);
        if (ret == 0) {
            uint8_t readback[4] = {0};
            ssize_t rd = fs_read(&probe_file, readback, sizeof(readback));
            LOG_INF("[SD] probe fs_read ret=%d data=[%02X %02X %02X %02X]",
                    (int)rd, readback[0], readback[1], readback[2], readback[3]);
            fs_close(&probe_file);

            if (rd == sizeof(probe_data) &&
                memcmp(readback, probe_data, sizeof(probe_data)) == 0) {
                LOG_INF("[SD] probe PASSED — card is readable and writable");
            } else {
                LOG_ERR("[SD] probe FAILED — readback mismatch, card may be "
                        "write-protected or have a bad sector at root");
            }
        }
        fs_unlink(SD_CARD_MOUNT_POINT "/test.tmp");
    } else {
        LOG_ERR("[SD] probe fs_open failed (ret=%d) — card mounted but not "
                "writable; check write-protect tab", ret);
    }

    LOG_INF("=== SD INIT COMPLETE ===");
    sd_mounted = true;
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
            /* Drain the ring so it doesn't back up */
            while (ring_buf_size_get(&sd_ring) > 0) {
                uint32_t avail = ring_buf_size_get(&sd_ring);
                uint32_t drain = avail < SD_WRITE_BUF_SIZE ? avail : SD_WRITE_BUF_SIZE;
                ring_buf_get(&sd_ring, sd_write_buf, drain);
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

            if (avail == 0) {
                /* Nothing to write yet — yield and retry */
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
    dc_estimate        = 0;   /* Reset to 0, not 2048 — avoids initial spike */

    dsp_mfcc_reset();

    if (saadc_init() != 0) return -EIO;
    next_dma_buf = 1;

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

    /* Delete old file, open new one */
    fs_unlink(AUDIO_FILE_PATH);
    fs_file_t_init(&sd_file);
    if (fs_open(&sd_file, AUDIO_FILE_PATH,
                FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC) < 0) {
        LOG_ERR("Cannot open SD file for recording");
        bt_nus_send(NULL, "ERR:SD_OPEN", 11);
        led_error_flash(led_set_yellow);
        return;
    }

    /*
     * Wake the SD writer thread exactly once.
     * The writer loops internally on ring_buf_size_get() — it does not
     * need a semaphore signal per half-buffer.
     */
    k_sem_give(&sd_data_sem);
#endif

    /* Send audio length to host so it can pre-allocate */
    char hdr[32];
    snprintf(hdr, sizeof(hdr), "START:%u\n", (uint32_t)TOTAL_AUDIO_BYTES);
    bt_nus_send(NULL, hdr, strlen(hdr));
    k_sleep(K_MSEC(10));

    led_set_cyan();
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

    LOG_INF("Recording %d s @ %d Hz (USE_SD=%s)...", DURATION_S, SAMPLING_RATE,
            USE_SD ? "true" : "false");

    /* Count half-buffers */
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

    /* Wake BLE TX thread to drain remaining audio_ring and send "finished\n" */
    k_sem_give(&audio_data_sem);

    /* Wait for BLE audio stream to complete */
    if (k_sem_take(&tx_done_sem, K_MSEC(10000)) != 0) {
        LOG_WRN("BLE TX did not finish within 10 s");
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
        led_error_flash(led_set_yellow);
    }
#endif

    led_set_red();
    LOG_INF("AcoustEEEcare v6.6 ready (USE_SD=%s) — waiting for BLE connection",
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