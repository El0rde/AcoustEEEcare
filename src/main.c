/*
 * AcoustEEEcare — SAADC BLE + SD Card + HR TFLite Edition
 * ============================================================
 * v7.2 — Fixed: removed direct interpreter/TENSOR_ARENA_SIZE access
 *               from main.c; arena logging now via hr_infer_arena_used()
 *
 * MODEL FACTS (HR_trial_144_int8.tflite):
 *   Input  [1, 1331, 20, 1] INT8 — 1331 MFCC frames × 20 coefficients
 *   Output [1, 1]           INT8 — single BPM value (0–121 BPM range)
 *   Ops    CONV_2D, MAX_POOL_2D, MEAN, FULLY_CONNECTED
 *   Size   16.4 KB flash
 *
 * IMPORTANT — MFCC CONFIG:
 *   Your dsp_mfcc must be configured with MFCC_N_MFCC = 20.
 *   MFCC_N_FRAMES should equal 1331; if it differs, hr_infer_run_int8()
 *   will zero-pad (fewer frames) or truncate (more frames) automatically.
 *
 * TARGET HARDWARE
 *   Seeed XIAO nRF52840
 *   Analog MEMS microphone on AIN0 (P0.02), cap-coupled
 *   SD card on SPI2, CS on P0.28
 * ============================================================
 */

#define USE_SD           true   /* set false for BLE-only mode          */
#define INTEGRATE_MODEL  true   /* set false to disable TFLite inference */

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

#if INTEGRATE_MODEL
#include "hr_infer.h"   /* HR TFLite inference interface */
#endif

#if USE_SD
#include <zephyr/fs/fs.h>
#include <zephyr/storage/disk_access.h>
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

/* DC removal: high-pass IIR, alpha = 1/256 */
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
#define AUDIO_RING_BYTES  (2 * 1024)
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
 * MFCC ACCUMULATION BUFFER  (INTEGRATE_MODEL only)
 *
 * on_mfcc_frame() fills this flat array frame-by-frame as
 * dsp_mfcc_finish() calls it. After finish(), hr_infer_run_int8()
 * reads the complete buffer.
 *
 * Layout: row-major [frame][coeff]
 *   s_mfcc_flat_q8[f * HR_MODEL_MFCC + c] = coeff c of frame f (INT8)
 *
 * Size: 1331 × 20 × 1 = 26 620 bytes ≈ 26.6 KB
 * ══════════════════════════════════════════════════════════════════ */
#if INTEGRATE_MODEL
static int8_t s_mfcc_flat_q8[HR_MODEL_FRAMES * HR_MODEL_MFCC];
static int    s_mfcc_collected = 0;
static bool   hr_model_ready  = false;
#endif

/* ══════════════════════════════════════════════════════════════════
 * SD CARD GLOBALS  (USE_SD true only)
 * ══════════════════════════════════════════════════════════════════ */
#if USE_SD

#define AUDIO_FILE_PATH      "/SD:/analog.pcm"
#define SD_CARD_MOUNT_POINT  "/SD:"
#define CHECKSUM_SIZE        sizeof(uint32_t)

#define SD_RING_BYTES  (8 * 1024)
RING_BUF_DECLARE(sd_ring, SD_RING_BYTES);

static atomic_t  sd_ring_drops;
static uint32_t  sd_ring_high_water = 0;

static K_SEM_DEFINE(sd_data_sem, 0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(sd_done_sem, 0, 1);

#define SD_WRITER_STACK_SIZE  1536
#define SD_WRITER_PRIORITY    6
static K_THREAD_STACK_DEFINE(sd_writer_stack, SD_WRITER_STACK_SIZE);
static struct k_thread sd_writer_thread_data;
static void sd_writer_thread_fn(void *a, void *b, void *c);

static FATFS            fat_fs;
static struct fs_mount_t mp = {
    .type      = FS_FATFS,
    .mnt_point = SD_CARD_MOUNT_POINT,
    .fs_data   = &fat_fs,
};
static bool             sd_mounted  = false;
static struct fs_file_t sd_file;
static uint32_t         sd_checksum = 0;

#define SD_WRITE_BUF_SIZE  512
static uint8_t sd_write_buf[SD_WRITE_BUF_SIZE];

static uint32_t compute_checksum(const uint8_t *data, uint32_t len)
{
    uint32_t cs = 0;
    for (uint32_t i = 0; i < len; i++) cs ^= data[i];
    return cs;
}

static int init_sd_card(void)
{
    static const char *disk_pdrv = "SD";
    uint64_t memory_size_mb;
    uint32_t block_count, block_size;

    LOG_INF("Initialising SD card...");

    k_sleep(K_MSEC(5000));

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
    return 0;
}

static void sd_writer_thread_fn(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

    while (true) {
        k_sem_take(&sd_data_sem, K_FOREVER);

        if (!sd_mounted) {
            LOG_ERR("SD writer: not mounted, discarding data");
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

        while (analog_recording || ring_buf_size_get(&sd_ring) > 0) {
            uint32_t avail = ring_buf_size_get(&sd_ring);

            if (avail == 0) {
                k_sleep(K_MSEC(1));
                continue;
            }

            bool recording_done = !analog_recording;
            uint32_t to_write   = avail < SD_WRITE_BUF_SIZE ? avail : SD_WRITE_BUF_SIZE;

            if (!recording_done && to_write < SD_WRITE_BUF_SIZE) {
                k_sleep(K_MSEC(1));
                continue;
            }

            ring_buf_get(&sd_ring, sd_write_buf, to_write);
            sd_checksum ^= compute_checksum(sd_write_buf, to_write);

            ssize_t written = fs_write(&sd_file, sd_write_buf, to_write);
            if (written < 0) {
                LOG_ERR("SD write error: %d at byte %u", (int)written, total_written);
            } else {
                total_written += (uint32_t)written;
            }
        }

        fs_write(&sd_file, &sd_checksum, CHECKSUM_SIZE);
        fs_close(&sd_file);

        LOG_INF("SD writer: %u audio bytes written, checksum=0x%08X",
                total_written, sd_checksum);
        LOG_INF("SD writer: sd_ring high water = %u / %u",
                sd_ring_high_water, SD_RING_BYTES);

        k_sem_give(&sd_done_sem);
    }
}

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

        dsp_mfcc_feed_chunk(filled_buf, HALF_BUF_SAMPLES);

#if USE_SD
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
        }
#endif

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
    dc_estimate        = 0;

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
 *
 * When INTEGRATE_MODEL true:
 *   Job 1 — quantize float coeffs to INT8 and accumulate into
 *            s_mfcc_flat_q8[] for later TFLite inference.
 *   Job 2 — stream the raw float frame over BLE NUS.
 *
 * When INTEGRATE_MODEL false:
 *   Job 2 only — stream frame over BLE NUS (v6.6 behaviour).
 * ══════════════════════════════════════════════════════════════════ */
static int s_mfcc_n_frames_total = MFCC_N_FRAMES;

static void on_mfcc_frame(int frame_idx, const float *coeffs)
{
    int err;

#if INTEGRATE_MODEL
    /*
     * Quantize float MFCC coefficient to INT8 using the model's input
     * quantization params (scale, zero-point) from hr_infer.h.
     * We write directly into s_mfcc_flat_q8[] here; hr_infer_run_int8()
     * will memcpy the whole buffer into the tensor after all frames
     * are collected.
     */
    if (hr_model_ready && frame_idx < HR_MODEL_FRAMES) {
        int n = MFCC_N_MFCC < HR_MODEL_MFCC ? MFCC_N_MFCC : HR_MODEL_MFCC;
        for (int i = 0; i < n; i++) {
            float   q  = coeffs[i] / HR_INPUT_SCALE + (float)HR_INPUT_ZP;
            int32_t qi = (int32_t)(q >= 0.0f ? q + 0.5f : q - 0.5f);
            s_mfcc_flat_q8[frame_idx * HR_MODEL_MFCC + i] =
                (int8_t)(qi < -128 ? -128 : qi > 127 ? 127 : qi);
        }
        s_mfcc_collected = frame_idx + 1;
    }
#endif /* INTEGRATE_MODEL */

    /* ── Job 2: BLE NUS stream ── */
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
 * Stage 1: Record audio → SD (USE_SD) + BLE stream
 * Stage 2: Compute MFCC + stream over BLE
 * Stage 3: TFLite HR inference → BPM over BLE  (INTEGRATE_MODEL only)
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

#if INTEGRATE_MODEL
    /* ── Reset MFCC accumulation buffer ── */
    s_mfcc_collected = 0;
    memset(s_mfcc_flat_q8, 0, sizeof(s_mfcc_flat_q8));
#endif

#if USE_SD
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

    fs_unlink(AUDIO_FILE_PATH);
    fs_file_t_init(&sd_file);
    if (fs_open(&sd_file, AUDIO_FILE_PATH,
                FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC) < 0) {
        LOG_ERR("Cannot open SD file for recording");
        bt_nus_send(NULL, "ERR:SD_OPEN", 11);
        led_error_flash(led_set_yellow);
        return;
    }

    k_sem_give(&sd_data_sem);
#endif

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

    LOG_INF("Recording %d s @ %d Hz (USE_SD=%s, INTEGRATE_MODEL=%s)...",
            DURATION_S, SAMPLING_RATE,
            USE_SD          ? "true" : "false",
            INTEGRATE_MODEL ? "true" : "false");

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

    k_sem_give(&audio_data_sem);

    if (k_sem_take(&tx_done_sem, K_MSEC(10000)) != 0) {
        LOG_WRN("BLE TX did not finish within 10 s");
    }

#if USE_SD
    LOG_INF("Waiting for SD writer to flush and close...");
    if (k_sem_take(&sd_done_sem, K_MSEC(15000)) != 0) {
        LOG_ERR("SD writer did not finish within 15 s — file may be truncated");
    } else {
        LOG_INF("SD file closed: %s (checksum=0x%08X)", AUDIO_FILE_PATH, sd_checksum);
        bt_nus_send(NULL, "SD:OK\n", 6);
    }
#endif

    /* ── Stage 2: MFCC stream ── */
    LOG_INF("BLE audio stream complete — starting MFCC");
    led_set_purple();

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

    LOG_INF("MFCC stream complete: %d frames x %d coeffs"
#if INTEGRATE_MODEL
            " (collected %d for TFLite)"
#endif
            , n_frames, MFCC_N_MFCC
#if INTEGRATE_MODEL
            , s_mfcc_collected
#endif
    );

#if INTEGRATE_MODEL
    /* ── Stage 3: TFLite HR inference ── */
    led_set_yellow();   /* Yellow = inference running */

    hr_result_t hr_result;
    if (hr_infer_run_int8(s_mfcc_flat_q8, s_mfcc_collected, &hr_result) == 0) {

        LOG_INF("HR inference complete: %.1f BPM (raw_int8=%d)",
                (double)hr_result.bpm, (int)hr_result.raw_output);

        /*
         * Log arena usage so you can tune TENSOR_ARENA_SIZE in hr_infer.cpp.
         * hr_infer_arena_used() is a thin C wrapper around
         * interpreter->arena_used_bytes() defined in hr_infer.cpp.
         * Once you have a stable number, set TENSOR_ARENA_SIZE to
         * that value + 4096 and rebuild.
         */
        LOG_INF("Arena used: %u bytes — set TENSOR_ARENA_SIZE to this + 4096",
                hr_infer_arena_used());

        char bpm_msg[24];
        int  bpm_len = snprintf(bpm_msg, sizeof(bpm_msg),
                                "BPM:%.1f\n", (double)hr_result.bpm);
        do {
            err = bt_nus_send(NULL, bpm_msg, bpm_len);
            if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(5));
        } while (err == -ENOMEM || err == -EAGAIN);

    } else {
        LOG_ERR("HR inference failed");
        bt_nus_send(NULL, "ERR:INFER\n", 10);
    }
#endif /* INTEGRATE_MODEL */

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

    /* DSP MFCC init */
    if (dsp_mfcc_init() != 0) {
        LOG_ERR("dsp_mfcc_init failed");
        led_error_flash(led_set_yellow);
        return -1;
    }

#if INTEGRATE_MODEL
    /*
     * TFLite HR model init.
     * Non-fatal: if it fails, hr_model_ready stays false and inference
     * is skipped. BLE stream and SD card still work normally.
     */
    if (hr_infer_init() != 0) {
        LOG_ERR("HR TFLite init failed — inference disabled");
        led_error_flash(led_set_yellow);
    } else {
        hr_model_ready = true;
        LOG_INF("HR model ready");
    }
#endif

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

    k_thread_create(&ble_tx_thread_data, ble_tx_stack,
                    K_THREAD_STACK_SIZEOF(ble_tx_stack),
                    ble_tx_thread_fn, NULL, NULL, NULL,
                    BLE_TX_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&ble_tx_thread_data, "ble_tx");

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
    LOG_INF("AcoustEEEcare v7.2 ready (USE_SD=%s, INTEGRATE_MODEL=%s) "
            "— waiting for BLE connection",
            USE_SD          ? "true" : "false",
            INTEGRATE_MODEL ? "true" : "false");

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
 * FEATURE FLAG COMBINATIONS
 * ════════════════════════════════════════════════════════════════════
 *
 *  USE_SD  INTEGRATE_MODEL   Behaviour
 *  ──────  ───────────────   ─────────────────────────────────────────
 *  false   false             BLE audio + MFCC stream only (v6.6)
 *  false   true              BLE audio + MFCC + TFLite BPM (no SD)
 *  true    false             BLE audio + MFCC + SD write (no inference)
 *  true    true              Full v7.2: all features enabled
 *
 * ════════════════════════════════════════════════════════════════════
 * REQUIRED prj.conf
 * ════════════════════════════════════════════════════════════════════
 *
 * # --- always required ---
 * CONFIG_HEAP_MEM_POOL_SIZE=8192
 * CONFIG_MAIN_STACK_SIZE=2048
 *
 * # --- USE_SD true ---
 * CONFIG_SPI=y
 * CONFIG_DISK_ACCESS=y
 * CONFIG_DISK_DRIVER_SDMMC=y
 * CONFIG_FAT_FILESYSTEM_ELM=y
 * CONFIG_FILE_SYSTEM=y
 * CONFIG_FILE_SYSTEM_MAX_TYPES=2
 *
 * # --- INTEGRATE_MODEL true ---
 * CONFIG_TENSORFLOW_LITE_MICRO=y
 * CONFIG_FPU=y
 * CONFIG_FPU_SHARING=y
 * CONFIG_CPP_EXCEPTIONS_MINIMAL=y
 * CONFIG_REQUIRES_FLOAT_PRINTF=y
 *
 * ════════════════════════════════════════════════════════════════════
 * MEMORY BUDGET (USE_SD true, INTEGRATE_MODEL true)
 * ════════════════════════════════════════════════════════════════════
 *
 *   Zephyr kernel + BLE stack         ~92 KB
 *   s_decimated (dsp_mfcc)             40 KB
 *   DSP scratch                         ~7 KB
 *   audio_ring                           2 KB
 *   sd_ring                              8 KB
 *   ping_pong[2][512]                    2 KB
 *   s_mfcc_flat_q8[1331*20] INT8      ~26.6 KB
 *   tensor_arena (TFLite, hr_infer.cpp) 80 KB  ← trim after reading arena log
 *   BLE TX thread stack                  2 KB
 *   SD writer thread stack             1.5 KB
 *   misc globals                        ~1 KB
 *   ─────────────────────────────────────────
 *   Total (before arena trim)        ~262 KB
 *
 *   After first successful inference, check the log:
 *     "Arena used: XXXXX bytes"
 *   Set TENSOR_ARENA_SIZE = XXXXX + 4096 in hr_infer.cpp and rebuild.
 *   Typical final total: ~220–240 KB.
 * ════════════════════════════════════════════════════════════════════
 */