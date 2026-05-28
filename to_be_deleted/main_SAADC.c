/*
 * AcoustEEEcare — SAADC Edition
 * ============================================================
 * Original DMIC/PDM pipeline replaced with SAADC on AIN0
 * for use with an analog MEMS microphone (cap-coupled, no
 * software DC removal needed).
 *
 * Target sample rate : 8 000 Hz  (125 µs per sample)
 * ADC resolution     : 12-bit
 * ADC gain           : 1/4
 * ADC reference      : VDD/4
 * Input              : AIN0  (P0.02 on XIAO nRF52840)
 *
 * BLE commands (NUS):
 *   "REC"  — record DURATION_S seconds to SD card
 *   "SEND" — verify checksum then stream SD audio over BLE
 *   "MFCC" — (commented out, pipeline preserved below)
 *
 * LED indicators:
 *   WHITE  — booting
 *   RED    — idle / advertising
 *   GREEN  — BLE connected / transfer done
 *   BLUE   — recording
 *   CYAN   — sending over BLE
 *   YELLOW — error flash
 * ============================================================
 */

#include <zephyr/bluetooth/services/nus.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/logging/log.h>
#include <zephyr/kernel.h>
#include <zephyr/fs/fs.h>
#include <zephyr/storage/disk_access.h>
#include <zephyr/drivers/adc.h>
#include <zephyr/sys/printk.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include "ff.h"

/* MFCC headers — kept but inactive until un-commented */
/*
#include "fft_size1024_bins513.h"
*/

#include <string.h>
#include <stdio.h>
#include <math.h>

LOG_MODULE_REGISTER(AcoustEEEcare);

#ifndef M_PI
#  define M_PI 3.14159265358979323846f
#endif

#ifndef MIN
#  define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

/* ══════════════════════════════════════════════════════════════
 * SAADC CONFIGURATION
 * ══════════════════════════════════════════════════════════════ */
#define ADC_NODE             DT_NODELABEL(adc)
#define ADC_RESOLUTION       12
#define ADC_GAIN             ADC_GAIN_1_4
#define ADC_REFERENCE        ADC_REF_VDD_1_4
#define ADC_ACQUISITION_TIME ADC_ACQ_TIME_DEFAULT
#define ADC_CHANNEL_MIC      0                          /* AIN0 — P0.02 */

static const struct device *adc_dev = DEVICE_DT_GET(ADC_NODE);
static int16_t adc_raw[1];                              /* single-sample read buffer */

static const struct adc_channel_cfg adc_channel_mic_cfg = {
    .gain             = ADC_GAIN,
    .reference        = ADC_REFERENCE,
    .acquisition_time = ADC_ACQUISITION_TIME,
    .channel_id       = ADC_CHANNEL_MIC,
    .input_positive   = SAADC_CH_PSELP_PSELP_AnalogInput0, /* AIN0 */
};

static const struct adc_sequence adc_seq = {
    .channels    = BIT(ADC_CHANNEL_MIC),
    .buffer      = adc_raw,        /* NOTE: adc_raw must remain writable at call-site */
    .buffer_size = sizeof(adc_raw),
    .resolution  = ADC_RESOLUTION,
};

/* ══════════════════════════════════════════════════════════════
 * AUDIO RECORDING PARAMS
 * ══════════════════════════════════════════════════════════════ */
#define SAMPLING_RATE      8000        /* Hz — SAADC target rate          */
#define SAMPLE_PERIOD_US   125         /* µs = 1 000 000 / SAMPLING_RATE  */
#define BIT_WIDTH          16          /* int16_t per sample               */
#define DURATION_S         10          /* recording window in seconds      */

#define TOTAL_AUDIO_SAMPLES  (SAMPLING_RATE * DURATION_S)           /* 80 000  */
#define TOTAL_AUDIO_BYTES    (TOTAL_AUDIO_SAMPLES * sizeof(int16_t)) /* 160 000 */

/*
 * SD write buffer — flush every WRITE_EVERY_N_SAMPLES samples to
 * avoid hammering the FAT layer on every single sample.
 */
#define WRITE_EVERY_N_SAMPLES  256
#define ACCUM_BUF_SIZE         (WRITE_EVERY_N_SAMPLES * sizeof(int16_t))
static uint8_t accum_buf[ACCUM_BUF_SIZE];

static uint16_t nus_chunk_size = 244;   /* updated after MTU exchange */

/* ══════════════════════════════════════════════════════════════
 * MFCC PARAMS  (preserved, inactive)
 * ══════════════════════════════════════════════════════════════ */
/*
#define TARGET_RATE         8000
#define DECIMATE_FACTOR     1          // no decimation needed at 8 kHz

#define BPF_LOW_HZ          10.0f
#define BPF_HIGH_HZ         200.0f

typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float x1, x2;
} Biquad;

#define NUM_BPF_BIQUADS  2
#define NUM_AA_BIQUADS   2
static Biquad bpf[NUM_BPF_BIQUADS];
static Biquad aa_lp[NUM_AA_BIQUADS];

static inline float biquad_process(Biquad *s, float x)
{
    float y = s->b0 * x + s->x1;
    s->x1   = s->b1 * x - s->a1 * y + s->x2;
    s->x2   = s->b2 * x - s->a2 * y;
    return y;
}

// ... (full biquad / BPF / LP / decimate helpers from original code 2
//      go here unchanged when re-enabling MFCC)

#define FRAME_MS            30
#define OVERLAP_MS          20
#define FRAME_SAMPLES       ((int)(FRAME_MS   * TARGET_RATE / 1000))
#define OVERLAP_SAMPLES     ((int)(OVERLAP_MS * TARGET_RATE / 1000))
#define HOP_SIZE            (FRAME_SAMPLES - OVERLAP_SAMPLES)
#define FFT_SIZE            1024
#define NUM_BINS            513
#define NUM_MEL_FILTERS     20
#define NUM_MFCC            13
#define LOW_FREQ_HZ         80.0f

#define MFCC_CSV_LINE_SIZE  (NUM_MFCC * 13 + 2)

static float   mel_fb[NUM_MEL_FILTERS][FFT_SIZE];
static creal_T fft_buf[NUM_BINS];
static float   mag_full[FFT_SIZE];
static float   log_energy[NUM_MEL_FILTERS];
static float   mfcc_out[NUM_MFCC];
static int16_t overlap_buf[FRAME_SAMPLES];
static int16_t windowed_i[FFT_SIZE];

// build_mel_filterbank(), compute_mfcc(), run_mfcc_stream() ...
// (restore from original code 2 — no changes needed since TARGET_RATE
//  is already 8 kHz and DECIMATE_FACTOR becomes 1)
*/

/* ══════════════════════════════════════════════════════════════
 * SD CARD
 * ══════════════════════════════════════════════════════════════ */
#define AUDIO_FILE_PATH      "/SD:/audio.pcm"
#define SD_CARD_MOUNT_POINT  "/SD:"

static FATFS fat_fs;
static struct fs_mount_t mp = {
    .type      = FS_FATFS,
    .mnt_point = SD_CARD_MOUNT_POINT,
};
static bool sd_mounted = false;

/* ══════════════════════════════════════════════════════════════
 * CHECKSUM
 * ══════════════════════════════════════════════════════════════ */
#define CHECKSUM_SIZE  sizeof(uint32_t)

static uint32_t compute_checksum(const uint8_t *data, uint32_t len)
{
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < len; i++) checksum ^= data[i];
    return checksum;
}

/* ══════════════════════════════════════════════════════════════
 * SD INITIALIZATION
 * ══════════════════════════════════════════════════════════════ */
static int init_sd_card(void)
{
    static const char *disk_pdrv = "SD";
    uint64_t memory_size_mb;
    uint32_t block_count, block_size;
    int err;

    printk("Initializing SD card...\n");
    err = disk_access_init(disk_pdrv);
    if (err != 0) { printk("disk_access_init failed: %d\n", err); return -1; }

    if (disk_access_ioctl(disk_pdrv, DISK_IOCTL_GET_SECTOR_COUNT, &block_count)) {
        printk("Unable to get sector count\n"); return -1;
    }
    if (disk_access_ioctl(disk_pdrv, DISK_IOCTL_GET_SECTOR_SIZE, &block_size)) {
        printk("Unable to get sector size\n"); return -1;
    }

    memory_size_mb = (uint64_t)block_count * block_size / (1024 * 1024);
    printk("Memory Size (MB): %u\n", (uint32_t)memory_size_mb);

    mp.fs_data = &fat_fs;
    err = fs_mount(&mp);
    if (err) { printk("Error mounting fat_fs [%d]\n", err); return err; }

    printk("Disk mounted successfully!\n");
    sd_mounted = true;
    return 0;
}

/* ══════════════════════════════════════════════════════════════
 * LED
 * ══════════════════════════════════════════════════════════════ */
static const struct gpio_dt_spec red_led   = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
static const struct gpio_dt_spec green_led = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);
static const struct gpio_dt_spec blue_led  = GPIO_DT_SPEC_GET(DT_ALIAS(led2), gpios);

static inline void led_on (const struct gpio_dt_spec *l) { gpio_pin_set_dt(l, 1); }
static inline void led_off(const struct gpio_dt_spec *l) { gpio_pin_set_dt(l, 0); }

static inline void led_set_white(void)  { led_on(&red_led);  led_on(&green_led);  led_on(&blue_led);  }
static inline void led_set_red(void)    { led_on(&red_led);  led_off(&green_led); led_off(&blue_led); }
static inline void led_set_green(void)  { led_off(&red_led); led_on(&green_led);  led_off(&blue_led); }
static inline void led_set_blue(void)   { led_off(&red_led); led_off(&green_led); led_on(&blue_led);  }
static inline void led_set_cyan(void)   { led_off(&red_led); led_on(&green_led);  led_on(&blue_led);  }
static inline void led_set_yellow(void) { led_on(&red_led);  led_on(&green_led);  led_off(&blue_led); }
static inline void led_set_purple(void) { led_on(&red_led);  led_off(&green_led); led_on(&blue_led);  }

static void led_error_flash(void (*error_color)(void))
{
    for (int i = 0; i < 3; i++) {
        error_color();
        k_sleep(K_MSEC(200));
        led_off(&red_led); led_off(&green_led); led_off(&blue_led);
        k_sleep(K_MSEC(200));
    }
    error_color();
}

/* ══════════════════════════════════════════════════════════════
 * FLAGS
 * ══════════════════════════════════════════════════════════════ */
static volatile bool start_recording = false;
static volatile bool start_sending   = false;
/* static volatile bool start_mfcc   = false; */   /* MFCC — commented out */
static volatile bool is_connected    = false;
static volatile bool mtu_exchanged   = false;

/* ══════════════════════════════════════════════════════════════
 * BLE ADVERTISING
 * ══════════════════════════════════════════════════════════════ */
#define DEVICE_NAME      "AcoustEEEcare"
#define DEVICE_NAME_LEN  (sizeof(DEVICE_NAME) - 1)

static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
    BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),
};
static const struct bt_data sd_adv[] = {
    BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_SRV_VAL),
};

/* ══════════════════════════════════════════════════════════════
 * CONNECTION CALLBACKS
 * ══════════════════════════════════════════════════════════════ */
static struct k_work_delayable adv_restart_work;

static void mtu_exchange_cb(struct bt_conn *conn, uint8_t err,
                             struct bt_gatt_exchange_params *params)
{
    if (!err) {
        uint16_t mtu = bt_gatt_get_mtu(conn);
        nus_chunk_size = mtu - 3;
        LOG_INF("MTU exchanged: %d, chunk: %d", mtu, nus_chunk_size);
    } else {
        LOG_WRN("MTU exchange failed (%d), using default %d", err, nus_chunk_size);
    }
    mtu_exchanged = true;
}

static struct bt_gatt_exchange_params exchange_params = { .func = mtu_exchange_cb };

static void adv_restart_work_handler(struct k_work *work)
{
    int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                               sd_adv, ARRAY_SIZE(sd_adv));
    if (err) LOG_ERR("adv restart failed: %d", err);
    else     LOG_INF("Advertising restarted");
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

    static const struct bt_le_conn_param conn_params = {
        .interval_min = 6, .interval_max = 12, .latency = 0, .timeout = 400,
    };
    bt_conn_le_param_update(conn, &conn_params);

    int mtu_err = bt_gatt_exchange_mtu(conn, &exchange_params);
    if (mtu_err) {
        LOG_WRN("MTU exchange request failed: %d, using default %d", mtu_err, nus_chunk_size);
        mtu_exchanged = true;
    }
    LOG_INF("Connected");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
    is_connected    = false;
    start_recording = false;
    start_sending   = false;
    /* start_mfcc   = false; */   /* MFCC — commented out */
    mtu_exchanged   = false;
    nus_chunk_size  = 244;
    led_set_red();
    LOG_WRN("Disconnected (%d)", reason);
    k_work_schedule(&adv_restart_work, K_MSEC(500));
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
    .connected    = connected,
    .disconnected = disconnected,
};

/* ══════════════════════════════════════════════════════════════
 * NUS CALLBACKS
 * ══════════════════════════════════════════════════════════════ */
static void received(struct bt_conn *conn, const void *data, uint16_t len, void *ctx)
{
    ARG_UNUSED(conn); ARG_UNUSED(ctx);
    LOG_INF("RX len=%d: %.*s", len, len, (const char *)data);

    if      (len == 3 && memcmp(data, "REC",  3) == 0) start_recording = true;
    else if (len == 4 && memcmp(data, "SEND", 4) == 0) start_sending   = true;
    /* MFCC command — commented out
    else if (len == 4 && memcmp(data, "MFCC", 4) == 0) start_mfcc      = true;
    */
}

static struct bt_nus_cb nus_listener = { .received = received };

/* ══════════════════════════════════════════════════════════════
 * SAADC RECORDING → SD
 *
 * Replaces record_audio_to_sd() which used DMIC.
 * Samples AIN0 at 8 kHz using k_busy_wait(SAMPLE_PERIOD_US)
 * for timing, writes int16 PCM to /SD:/audio.pcm, then appends
 * a 4-byte XOR checksum identical to the original implementation.
 *
 * NOTE: k_busy_wait() blocks the CPU for the sample period.
 *       At 8 kHz this is 125 µs per iteration — acceptable on
 *       nRF52840 at 64 MHz. If you add more work inside the loop
 *       subtract elapsed cycles from the wait (same pattern as
 *       code 1's mic_thread).
 * ══════════════════════════════════════════════════════════════ */
static int record_audio_to_sd(void)
{
    if (!sd_mounted) { LOG_ERR("SD not mounted"); return -ENODEV; }

    /* --- open output file ---------------------------------------- */
    fs_unlink(AUDIO_FILE_PATH);
    struct fs_file_t file;
    fs_file_t_init(&file);
    int ret = fs_open(&file, AUDIO_FILE_PATH, FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (ret < 0) {
        LOG_ERR("fs_open failed: %d", ret);
        led_error_flash(led_set_yellow);
        return ret;
    }

    LOG_INF("Recording %d s @ %d Hz via SAADC -> %s ...",
            DURATION_S, SAMPLING_RATE, AUDIO_FILE_PATH);

    led_set_blue();

    uint32_t total_written = 0;
    uint32_t checksum      = 0;
    uint32_t accum_pos     = 0;         /* byte offset into accum_buf */

    /*
     * adc_sequence cannot be const when passed to adc_read() because
     * the driver may update internal fields; use a local mutable copy.
     */
    struct adc_sequence seq = {
        .channels    = BIT(ADC_CHANNEL_MIC),
        .buffer      = adc_raw,
        .buffer_size = sizeof(adc_raw),
        .resolution  = ADC_RESOLUTION,
    };

    for (uint32_t i = 0; i < TOTAL_AUDIO_SAMPLES; i++) {
        uint32_t t_start = k_cycle_get_32();

        /* --- read one sample --------------------------------------- */
        if (adc_read(adc_dev, &seq) == 0) {
            int16_t sample = adc_raw[0];

            /* accumulate into write buffer */
            memcpy(accum_buf + accum_pos, &sample, sizeof(int16_t));
            accum_pos  += sizeof(int16_t);
            checksum   ^= compute_checksum((const uint8_t *)&sample, sizeof(int16_t));
        } else {
            /* on ADC read failure write a zero sample and log */
            int16_t zero = 0;
            memcpy(accum_buf + accum_pos, &zero, sizeof(int16_t));
            accum_pos += sizeof(int16_t);
            LOG_WRN("ADC read failed at sample %u", i);
        }

        /* --- flush to SD when buffer is full ----------------------- */
        if (accum_pos >= ACCUM_BUF_SIZE) {
            ssize_t written = fs_write(&file, accum_buf, accum_pos);
            if (written < 0) {
                LOG_ERR("fs_write failed at sample %u: %d", i, (int)written);
                led_error_flash(led_set_yellow);
                fs_close(&file);
                return (int)written;
            }
            total_written += (uint32_t)written;
            accum_pos      = 0;
        }

        /* --- busy-wait for remainder of sample period -------------- */
        uint32_t elapsed_us = k_cyc_to_us_floor32(k_cycle_get_32() - t_start);
        uint32_t wait_us    = (elapsed_us < SAMPLE_PERIOD_US)
                              ? (SAMPLE_PERIOD_US - elapsed_us) : 0;
        k_busy_wait(wait_us);
    }

    /* --- flush any remaining bytes --------------------------------- */
    if (accum_pos > 0) {
        ssize_t written = fs_write(&file, accum_buf, accum_pos);
        if (written < 0) {
            LOG_ERR("fs_write final flush failed: %d", (int)written);
            led_error_flash(led_set_yellow);
            fs_close(&file);
            return (int)written;
        }
        total_written += (uint32_t)written;
    }

    /* --- append 4-byte XOR checksum -------------------------------- */
    ssize_t cs_written = fs_write(&file, &checksum, CHECKSUM_SIZE);
    if (cs_written != CHECKSUM_SIZE) {
        LOG_ERR("Failed to write checksum: %d", (int)cs_written);
        led_error_flash(led_set_yellow);
        fs_close(&file);
        return -EIO;
    }

    fs_close(&file);
    LOG_INF("Recording done -- %u audio bytes + 4-byte checksum (0x%08X)",
            total_written, checksum);
    return 0;
}

/* ══════════════════════════════════════════════════════════════
 * SEND SD AUDIO OVER BLE  (unchanged from original code 2)
 * ══════════════════════════════════════════════════════════════ */
#define SEND_BUF_SIZE  512
static uint8_t send_buf[SEND_BUF_SIZE];

static void send_audio_from_sd(void)
{
    if (!sd_mounted) {
        LOG_ERR("SD not mounted");
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:NOSD", 8);
        return;
    }

    struct fs_file_t file;
    fs_file_t_init(&file);
    int ret = fs_open(&file, AUDIO_FILE_PATH, FS_O_READ);
    if (ret < 0) {
        LOG_ERR("fs_open for read failed: %d", ret);
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:NOFILE", 10);
        return;
    }

    struct fs_dirent entry;
    fs_stat(AUDIO_FILE_PATH, &entry);
    uint32_t file_size = entry.size;

    if (file_size <= CHECKSUM_SIZE) {
        LOG_ERR("File too small: %u bytes", file_size);
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:BADFILE", 11);
        fs_close(&file);
        return;
    }

    uint32_t audio_size = file_size - CHECKSUM_SIZE;

    /* --- verify checksum before sending ---------------------------- */
    uint32_t stored_checksum   = 0;
    uint32_t computed_checksum = 0;
    uint8_t  verify_buf[256];
    int32_t  bytes_remaining = (int32_t)audio_size;

    while (bytes_remaining > 0) {
        uint32_t to_read = MIN((uint32_t)bytes_remaining, sizeof(verify_buf));
        ssize_t  n       = fs_read(&file, verify_buf, to_read);
        if (n <= 0) {
            LOG_ERR("Checksum verify read failed: %d", (int)n);
            led_error_flash(led_set_yellow);
            bt_nus_send(NULL, "ERR:READFAIL", 12);
            fs_close(&file);
            return;
        }
        computed_checksum ^= compute_checksum(verify_buf, (uint32_t)n);
        bytes_remaining   -= (int32_t)n;
    }

    ssize_t cs_read = fs_read(&file, &stored_checksum, CHECKSUM_SIZE);
    if (cs_read != CHECKSUM_SIZE) {
        LOG_ERR("Failed to read stored checksum: %d", (int)cs_read);
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:CSFAIL", 10);
        fs_close(&file);
        return;
    }

    if (computed_checksum != stored_checksum) {
        LOG_ERR("Checksum MISMATCH: computed=0x%08X stored=0x%08X",
                computed_checksum, stored_checksum);
        led_error_flash(led_set_purple);
        bt_nus_send(NULL, "ERR:CORRUPT", 11);
        fs_close(&file);
        return;
    }

    LOG_INF("Checksum OK: 0x%08X -- starting BLE transfer", stored_checksum);

    ret = fs_seek(&file, 0, FS_SEEK_SET);
    if (ret < 0) {
        LOG_ERR("fs_seek failed: %d", ret);
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:SEEK", 8);
        fs_close(&file);
        return;
    }

    /* send START header with audio byte count */
    char hdr[32];
    snprintf(hdr, sizeof(hdr), "START:%u", audio_size);
    bt_nus_send(NULL, hdr, strlen(hdr));
    k_sleep(K_MSEC(10));

    uint32_t offset      = 0;
    uint32_t chunk_count = 0;
    uint16_t chunk_size  = MIN(nus_chunk_size, SEND_BUF_SIZE);
    bool     send_ok     = true;

    while (offset < audio_size) {
        uint32_t to_read  = MIN(audio_size - offset, chunk_size);
        ssize_t  read_len = fs_read(&file, send_buf, to_read);
        if (read_len <= 0) {
            LOG_ERR("fs_read failed at offset %u: %d", offset, (int)read_len);
            send_ok = false;
            break;
        }

        int err;
        do {
            err = bt_nus_send(NULL, send_buf, (uint16_t)read_len);
            if (err == -EAGAIN) k_sleep(K_MSEC(1));
        } while (err == -EAGAIN);

        if (err < 0) {
            LOG_ERR("bt_nus_send failed at offset %u: %d", offset, err);
            send_ok = false;
            break;
        }

        offset += (uint32_t)read_len;
        chunk_count++;
    }

    fs_close(&file);

    if (!send_ok) {
        LOG_ERR("Transfer aborted at offset %u / %u", offset, audio_size);
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:ABORT", 9);
    } else {
        bt_nus_send(NULL, "END", 3);
        LOG_INF("Transfer complete -- %u chunks, %u bytes", chunk_count, offset);
    }
}

/* ══════════════════════════════════════════════════════════════
 * MFCC STREAM  (preserved, fully commented out)
 *
 * To re-enable:
 *   1. Un-comment the MFCC PARAMS block near the top.
 *   2. Un-comment fft_entry_initialize() / fft_entry_terminate()
 *      and the include for fft_size1024_bins513.h.
 *   3. Restore compute_butter4_bpf(), compute_butter4_lp(),
 *      apply_bpf_and_decimate(), build_mel_filterbank(),
 *      compute_mfcc() from original code 2 — they are identical
 *      except DECIMATE_FACTOR becomes 1 (already at 8 kHz).
 *   4. Replace the dmic_read() loop inside run_mfcc_stream()
 *      with an SAADC polling loop that fills hop buffers of
 *      HOP_SIZE samples using the same k_busy_wait() timing
 *      pattern as record_audio_to_sd() above.
 *   5. Un-comment start_mfcc flag, "MFCC" NUS command, and
 *      the start_mfcc block in main().
 * ══════════════════════════════════════════════════════════════ */
/*
static void run_mfcc_stream(void)
{
    // ... restore from code 2, swap dmic_read() for SAADC polling
}
*/

/* ══════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════ */
int main(void)
{
    int err;

    /* --- GPIO / LED init ------------------------------------------ */
    gpio_pin_configure_dt(&red_led,   GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&green_led, GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&blue_led,  GPIO_OUTPUT_ACTIVE);
    led_set_white();

    k_work_init_delayable(&adv_restart_work, adv_restart_work_handler);

    /* --- SAADC init ----------------------------------------------- */
    if (!device_is_ready(adc_dev)) {
        LOG_ERR("SAADC device not ready");
        led_error_flash(led_set_yellow);
        return -ENODEV;
    }

    err = adc_channel_setup(adc_dev, &adc_channel_mic_cfg);
    if (err != 0) {
        LOG_ERR("Failed to setup SAADC channel: %d", err);
        led_error_flash(led_set_yellow);
        return err;
    }
    LOG_INF("SAADC ready on AIN0 @ %d Hz", SAMPLING_RATE);

    /* --- FFT / MFCC init (commented out) -------------------------- */
    /*
    fft_entry_initialize();
    compute_butter4_bpf(BPF_LOW_HZ, BPF_HIGH_HZ, (float)SAMPLING_RATE);
    compute_butter4_lp((float)TARGET_RATE / 2.0f * 0.9f, (float)SAMPLING_RATE);
    build_mel_filterbank();
    LOG_INF("MFCC pipeline ready");
    */

    /* --- BLE init ------------------------------------------------- */
    err = bt_enable(NULL);
    if (err) {
        LOG_ERR("bt_enable failed: %d", err);
        led_error_flash(led_set_yellow);
        return err;
    }

    err = bt_nus_cb_register(&nus_listener, NULL);
    if (err) {
        LOG_ERR("bt_nus_cb_register failed: %d", err);
        led_error_flash(led_set_yellow);
        return err;
    }

    err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                          sd_adv, ARRAY_SIZE(sd_adv));
    if (err) {
        LOG_ERR("bt_le_adv_start failed: %d", err);
        led_error_flash(led_set_yellow);
        return err;
    }

    /* --- SD card init --------------------------------------------- */
    if (init_sd_card() != 0) {
        LOG_ERR("Failed to initialize SD card");
        led_error_flash(led_set_yellow);
        return -1;
    }

    led_set_red();
    LOG_INF("AcoustEEEcare (SAADC) ready — waiting for BLE connection");

    /* ══ Main loop ══════════════════════════════════════════════════ */
    while (true) {
        k_sleep(K_MSEC(100));
        if (!is_connected) continue;

        /* --- REC command ------------------------------------------ */
        if (start_recording) {
            start_recording = false;

            int rec_ret = record_audio_to_sd();
            if (rec_ret < 0) {
                LOG_ERR("record_audio_to_sd failed: %d", rec_ret);
                bt_nus_send(NULL, "ERR:REC", 7);
                k_sleep(K_MSEC(2000));
            } else {
                bt_nus_send(NULL, "REC:OK", 6);
            }
            led_set_green();
        }

        /* --- SEND command ----------------------------------------- */
        if (start_sending) {
            start_sending = false;

            /* wait up to 500 ms for MTU exchange to complete */
            int waited = 0;
            while (!mtu_exchanged && waited < 500) {
                k_sleep(K_MSEC(10));
                waited += 10;
            }
            LOG_INF("Starting BLE transfer -- chunk: %d bytes", nus_chunk_size);
            led_set_cyan();
            send_audio_from_sd();
            if (is_connected) led_set_green();
        }

        /* --- MFCC command (commented out) ------------------------- */
        /*
        if (start_mfcc) {
            run_mfcc_stream();
            start_mfcc = false;
            if (is_connected) led_set_green();
        }
        */
    }

    return 0;
}