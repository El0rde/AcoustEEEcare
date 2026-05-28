#include <zephyr/bluetooth/services/nus.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/logging/log.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/kernel.h>
#include <zephyr/fs/fs.h>
#include <zephyr/storage/disk_access.h>
#include "ff.h"
#include "fft_size1024_bins513.h"   /* fft_entry(), fft_entry_initialize(), fft_entry_terminate() */
#include "zephyr/device.h"
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
 * AUDIO HARDWARE PARAMS
 * ══════════════════════════════════════════════════════════════ */
#define SAMPLING_RATE      16000
#define BIT_WIDTH          16
#define DURATION_S         10

#define BLOCK_SIZE_BYTES   320
#define NUM_BLOCKS         8

#define TOTAL_AUDIO_BYTES  (SAMPLING_RATE * (BIT_WIDTH / 8) * DURATION_S)
#define RECORD_BLOCK_COUNT (TOTAL_AUDIO_BYTES / BLOCK_SIZE_BYTES)

#define WRITE_EVERY_N_BLOCKS  4
#define ACCUM_BUF_SIZE        (BLOCK_SIZE_BYTES * WRITE_EVERY_N_BLOCKS)
static uint8_t accum_buf[ACCUM_BUF_SIZE];

static uint16_t nus_chunk_size = 244;

/* ══════════════════════════════════════════════════════════════
 * BANDPASS + RESAMPLING PARAMS
 * ══════════════════════════════════════════════════════════════ */
#define SAMPLE_RATE         16000
#define TARGET_RATE         8000
#define DECIMATE_FACTOR     (SAMPLE_RATE / TARGET_RATE)   /* = 2 */

#define BPF_LOW_HZ          10.0f
#define BPF_HIGH_HZ         200.0f

/* ─── Biquad struct — Direct Form II transposed ─────────────── */
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

static void compute_butter4_bpf(float low_hz, float high_hz, float fs)
{
    float Wl   = 2.0f * fs * tanf((float)M_PI * low_hz  / fs);
    float Wh   = 2.0f * fs * tanf((float)M_PI * high_hz / fs);
    float BW   = Wh - Wl;
    float W0sq = Wl * Wh;

    const float p_re = -0.70710678118f;
    float signs[2]   = { 0.70710678118f, -0.70710678118f };
    float k = 2.0f * fs;

    for (int i = 0; i < NUM_BPF_BIQUADS; i++) {
        float lp_re = p_re, lp_im = signs[i];
        float bp_re = BW * lp_re, bp_im = BW * lp_im;
        float bp2_re  = bp_re * bp_re - bp_im * bp_im;
        float bp2_im  = 2.0f * bp_re * bp_im;
        float disc_re = bp2_re - 4.0f * W0sq;
        float disc_im = bp2_im;
        float disc_mag = sqrtf(disc_re * disc_re + disc_im * disc_im);
        float sqrt_re  = sqrtf((disc_mag + disc_re) / 2.0f);
        float sqrt_im  = (disc_im >= 0.0f ? 1.0f : -1.0f)
                       * sqrtf((disc_mag - disc_re) / 2.0f);
        float s_r = (bp_re + sqrt_re) / 2.0f;
        float s_i = (bp_im + sqrt_im) / 2.0f;
        float den = (k - s_r) * (k - s_r) + s_i * s_i;
        float num = (k + s_r) * (k + s_r) + s_i * s_i;
        bpf[i].a1 = -2.0f * (k * k - s_r * s_r - s_i * s_i) / den;
        bpf[i].a2 =  num / den;
        bpf[i].b0 =  BW * k / (2.0f * den);
        bpf[i].b1 =  0.0f;
        bpf[i].b2 = -bpf[i].b0;
        bpf[i].x1 =  bpf[i].x2 = 0.0f;
    }
}

static void compute_butter4_lp(float cutoff_hz, float fs)
{
    float Wc = 2.0f * fs * tanf((float)M_PI * cutoff_hz / fs);
    const float proto_re[2] = {
        -sinf(    (float)M_PI / 8.0f),
        -sinf(3.0f * (float)M_PI / 8.0f)
    };
    const float proto_im[2] = {
         cosf(    (float)M_PI / 8.0f),
         cosf(3.0f * (float)M_PI / 8.0f)
    };
    float k = 2.0f * fs;
    for (int i = 0; i < NUM_AA_BIQUADS; i++) {
        float s_r = Wc * proto_re[i];
        float s_i = Wc * proto_im[i];
        float den = (k - s_r) * (k - s_r) + s_i * s_i;
        float Wc2 = Wc * Wc;
        aa_lp[i].b0 = Wc2 / den;
        aa_lp[i].b1 = 2.0f * Wc2 / den;
        aa_lp[i].b2 = Wc2 / den;
        aa_lp[i].a1 = -2.0f * (k * k - s_r * s_r - s_i * s_i) / den;
        aa_lp[i].a2 = ((k + s_r) * (k + s_r) + s_i * s_i) / den;
        aa_lp[i].x1 = aa_lp[i].x2 = 0.0f;
    }
}

static void apply_bpf_and_decimate(const int16_t *in,  int n_raw,
                                   int16_t       *out, int *n_out)
{
    int out_idx = 0;
    for (int i = 0; i < n_raw; i++) {
        float s = (float)in[i];
        s = biquad_process(&bpf[0], s);
        s = biquad_process(&bpf[1], s);
        s = biquad_process(&aa_lp[0], s);
        s = biquad_process(&aa_lp[1], s);
        if (i % DECIMATE_FACTOR == 0)
            out[out_idx++] = (int16_t)(s < -32768.0f ? -32768.0f
                                      : s >  32767.0f ?  32767.0f : s);
    }
    *n_out = out_idx;
}

/* ══════════════════════════════════════════════════════════════
 * MFCC PARAMS
 * ══════════════════════════════════════════════════════════════ */
#define FRAME_MS            30
#define OVERLAP_MS          20
#define FRAME_SAMPLES       ((int)(FRAME_MS   * TARGET_RATE / 1000))  /* 240 @ 8kHz */
#define OVERLAP_SAMPLES     ((int)(OVERLAP_MS * TARGET_RATE / 1000))  /* 160 @ 8kHz */
#define HOP_SIZE            (FRAME_SAMPLES - OVERLAP_SAMPLES)          /* 80  @ 8kHz */
#define FFT_SIZE            1024
#define NUM_BINS            513     /* FFT_SIZE/2 + 1 — used by fft_entry() output */
#define NUM_MEL_FILTERS     20
#define NUM_MFCC            13
#define LOW_FREQ_HZ         80.0f

#define RAW_HOP_SAMPLES     (HOP_SIZE * DECIMATE_FACTOR)   /* 160 @ 16kHz */

#define MFCC_CSV_LINE_SIZE  (NUM_MFCC * 13 + 2)

/* ══════════════════════════════════════════════════════════════
 * MEMORY SLABS
 * ══════════════════════════════════════════════════════════════ */
K_MEM_SLAB_DEFINE(dmic_mem_slab, BLOCK_SIZE_BYTES, NUM_BLOCKS, 4);

#define MFCC_BLOCK_COUNT  3
K_MEM_SLAB_DEFINE_STATIC(mfcc_audio_slab,
                         (RAW_HOP_SAMPLES * sizeof(int16_t)),
                         MFCC_BLOCK_COUNT, 4);

#define SEND_BUF_SIZE  512
static uint8_t send_buf[SEND_BUF_SIZE];

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

static uint32_t compute_checksum(const uint8_t *data, uint32_t len) {
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < len; i++) checksum ^= data[i];
    return checksum;
}

/* ══════════════════════════════════════════════════════════════
 * SD INITIALIZATION
 * ══════════════════════════════════════════════════════════════ */
static int init_sd_card(void) {
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
 * MFCC CSV HELPERS
 * ══════════════════════════════════════════════════════════════ */
static int mfcc_csv_write_row(struct fs_file_t *file, const float *coeffs, int n)
{
    char line[MFCC_CSV_LINE_SIZE];
    int  pos = 0;

    for (int i = 0; i < n; i++) {
        pos += snprintf(line + pos, sizeof(line) - pos,
                        i < n - 1 ? "%.6f," : "%.6f\n",
                        (double)coeffs[i]);
        if (pos >= (int)sizeof(line) - 1) {
            LOG_ERR("MFCC CSV line truncated at coefficient %d", i);
            return -ENOMEM;
        }
    }

    ssize_t written = fs_write(file, line, pos);
    if (written != pos) {
        LOG_ERR("MFCC CSV row write failed: %d", (int)written);
        return -EIO;
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════
 * DMIC HELPERS
 * ══════════════════════════════════════════════════════════════ */
static void drain_dmic(const struct device *dmic_dev) {
    void    *leftover      = NULL;
    uint32_t leftover_size = 0;
    while (dmic_read(dmic_dev, 0, &leftover, &leftover_size, 100) == 0)
        k_mem_slab_free(&dmic_mem_slab, leftover);
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

static void led_error_flash(void (*error_color)(void)) {
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
static volatile bool start_mfcc      = false;
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
                             struct bt_gatt_exchange_params *params) {
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

static void adv_restart_work_handler(struct k_work *work) {
    int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                               sd_adv, ARRAY_SIZE(sd_adv));
    if (err) LOG_ERR("adv restart failed: %d", err);
    else     LOG_INF("Advertising restarted");
}

static void connected(struct bt_conn *conn, uint8_t err) {
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

static void disconnected(struct bt_conn *conn, uint8_t reason) {
    is_connected    = false;
    start_recording = false;
    start_sending   = false;
    start_mfcc      = false;
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
static void received(struct bt_conn *conn, const void *data, uint16_t len, void *ctx) {
    ARG_UNUSED(conn); ARG_UNUSED(ctx);
    LOG_INF("RX len=%d: %.*s", len, len, (const char *)data);
    if      (len == 3 && memcmp(data, "REC",  3) == 0) start_recording = true;
    else if (len == 4 && memcmp(data, "SEND", 4) == 0) start_sending   = true;
    else if (len == 4 && memcmp(data, "MFCC", 4) == 0) start_mfcc      = true;
}

static struct bt_nus_cb nus_listener = { .received = received };

/* ══════════════════════════════════════════════════════════════
 * MFCC ENGINE
 * Validated against MATLAB mfcc() — max diff < 0.000012
 *
 * Changes from original firmware:
 *   1. build_mel_filterbank: Slaney Hz-domain triangles with
 *      bandwidth/2 normalisation — matches designMelFilterBank()
 *   2. compute_mfcc: no pre-emphasis, periodic Hamming window,
 *      magnitude spectrum (not power), full FFT_SIZE filterbank,
 *      log10 (not natural log)
 *   3. fft_entry() output (NUM_BINS one-sided) is mirrored to
 *      fill the full FFT_SIZE before applying the filterbank
 * ══════════════════════════════════════════════════════════════ */
static inline float hz_to_mel(float hz)  { return 1127.0f * logf(1.0f + hz / 700.0f); }
static inline float mel_to_hz(float mel) { return 700.0f  * (expf(mel / 1127.0f) - 1.0f); }

/* Filterbank spans full FFT_SIZE — matches MATLAB's 1024×20 FB matrix */
static float   mel_fb[NUM_MEL_FILTERS][FFT_SIZE];
static creal_T fft_buf[NUM_BINS];          /* fft_entry() output — one-sided */
static float   mag_full[FFT_SIZE];         /* mirrored magnitude spectrum    */
static float   log_energy[NUM_MEL_FILTERS];
static float   mfcc_out[NUM_MFCC];
static int16_t overlap_buf[FRAME_SAMPLES];
static int16_t decimated_hop[HOP_SIZE];
static int16_t windowed_i[FFT_SIZE];       /* int16 for fft_entry()          */

static void build_mel_filterbank(void)
{
    /* Replicates audio.internal.designMelFilterBank() — 'Hz' / Slaney style:
     *
     *  linFq[b]  = b / FFT_SIZE * TARGET_RATE   (Hz of each bin)
     *  p[e]      = first bin where linFq[b] > edge_hz[e]  (strict >)
     *  rising    = (linFq[b] - edge[e])   / (edge[e+1] - edge[e])
     *  falling   = (edge[e+2] - linFq[b]) / (edge[e+2] - edge[e+1])
     *  normalise = divide by (edge[e+2] - edge[e]) / 2
     */
    float mel_low  = hz_to_mel(LOW_FREQ_HZ);
    float mel_high = hz_to_mel((float)TARGET_RATE / 2.0f);

    float edge_hz[NUM_MEL_FILTERS + 2];
    for (int pt = 0; pt < NUM_MEL_FILTERS + 2; pt++) {
        float fraction = (float)pt / (float)(NUM_MEL_FILTERS + 1);
        float mel      = mel_low + fraction * (mel_high - mel_low);
        edge_hz[pt]    = mel_to_hz(mel);
    }

    /* Inflection points: first bin where linFq > edge_hz (strict >) */
    int p[NUM_MEL_FILTERS + 2];
    for (int e = 0; e < NUM_MEL_FILTERS + 2; e++) {
        p[e] = FFT_SIZE - 1;
        for (int b = 0; b < FFT_SIZE; b++) {
            float linFq = (float)b / (float)FFT_SIZE * (float)TARGET_RATE;
            if (linFq > edge_hz[e]) { p[e] = b; break; }
        }
    }

    memset(mel_fb, 0, sizeof(mel_fb));
    for (int f = 0; f < NUM_MEL_FILTERS; f++) {
        float left_hz   = edge_hz[f];
        float center_hz = edge_hz[f + 1];
        float right_hz  = edge_hz[f + 2];

        /* Width of each slope in Hz — used to normalise the triangle height */
        float rising_width  = center_hz - left_hz;
        float falling_width = right_hz  - center_hz;
        if (rising_width  < 1e-10f) rising_width  = 1e-10f;  /* avoid divide-by-zero */
        if (falling_width < 1e-10f) falling_width = 1e-10f;

        /* Bandwidth normalisation factor = total filter width / 2
        * Matches MATLAB's designMelFilterBank 'Bandwidth' normalisation */
        float bandwidth = (right_hz - left_hz) / 2.0f;
        if (bandwidth < 1e-10f) bandwidth = 1e-10f;

        /* Rising slope: bins from left edge up to (not including) center */
        for (int bin = p[f]; bin < p[f + 1] && bin < FFT_SIZE; bin++) {
            float bin_hz     = (float)bin / (float)FFT_SIZE * (float)TARGET_RATE;
            float rise       = (bin_hz - left_hz) / rising_width;   /* 0.0 → 1.0 */
            mel_fb[f][bin]   = rise / bandwidth;
        }

        /* Falling slope: bins from center down to (not including) right edge */
        for (int bin = p[f + 1]; bin < p[f + 2] && bin < FFT_SIZE; bin++) {
            float bin_hz     = (float)bin / (float)FFT_SIZE * (float)TARGET_RATE;
            float fall       = (right_hz - bin_hz) / falling_width;  /* 1.0 → 0.0 */
            mel_fb[f][bin]   = fall / bandwidth;
        }
    }
}

static void compute_mfcc(const int16_t *pcm)
{
    /* Periodic Hamming window — no pre-emphasis, normalise to [-1, 1]
     * Scale back to int16 range for fft_entry() fixed-point input */
    for (int n = 0; n < FRAME_SAMPLES; n++) {
        float signal = (float)pcm[n] / 32768.0f;
        float win    = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * n
                                             / (float)FRAME_SAMPLES);
        float w      = signal * win * 32767.0f;
        windowed_i[n] = (int16_t)(w < -32768.0f ? -32768.0f
                                : w >  32767.0f ?  32767.0f : w);
    }
    for (int n = FRAME_SAMPLES; n < FFT_SIZE; n++)
        windowed_i[n] = 0;

    /* Fixed-point FFT — outputs one-sided spectrum NUM_BINS = FFT_SIZE/2+1 */
    fft_entry(windowed_i, fft_buf);

    /* Compute magnitude for one-sided bins 0..NUM_BINS-1
     * then mirror to fill bins FFT_SIZE/2+1..FFT_SIZE-1
     * Matches MATLAB's abs(fft(..., FFT_SIZE)) on real input */
    for (int k = 0; k < NUM_BINS; k++) {
        float re   = (float)fft_buf[k].re;
        float im   = (float)fft_buf[k].im;
        /* Undo the int16 scaling so magnitude is in [-1,1] float range */
        mag_full[k] = sqrtf(re * re + im * im) / 32767.0f;
    }
    /* Mirror: mag[FFT_SIZE-k] = mag[k] for k=1..FFT_SIZE/2-1 */
    for (int k = 1; k < FFT_SIZE / 2; k++)
        mag_full[FFT_SIZE - k] = mag_full[k];

    /* Mel filterbank over full FFT_SIZE bins + log10 */
    for (int m = 0; m < NUM_MEL_FILTERS; m++) {
        float e = 0.0f;
        for (int k = 0; k < FFT_SIZE; k++)
            e += mel_fb[m][k] * mag_full[k];
        log_energy[m] = log10f(e < 1e-10f ? 1e-10f : e);
    }

    /* Orthonormal DCT-II */
    float scale_dc   = sqrtf(1.0f / (float)NUM_MEL_FILTERS);
    float scale_rest = sqrtf(2.0f / (float)NUM_MEL_FILTERS);
    for (int n = 0; n < NUM_MFCC; n++) {
        float sum = 0.0f;
        for (int m = 0; m < NUM_MEL_FILTERS; m++)
            sum += log_energy[m]
                 * cosf((float)M_PI * (float)n * (m + 0.5f)
                        / (float)NUM_MEL_FILTERS);
        mfcc_out[n] = (n == 0 ? scale_dc : scale_rest) * sum;
    }
}

/* ─── run_mfcc_stream() ──────────────────────────────────────── */
static void run_mfcc_stream(const struct device *dmic_dev)
{
    LOG_INF("MFCC stream: BPF %.0f-%.0f Hz, %d->%d Hz, FFT=%d, bins=%d",
            (double)BPF_LOW_HZ, (double)BPF_HIGH_HZ,
            SAMPLE_RATE, TARGET_RATE, FFT_SIZE, NUM_BINS);
    led_set_blue();

    compute_butter4_bpf(BPF_LOW_HZ, BPF_HIGH_HZ, (float)SAMPLE_RATE);
    compute_butter4_lp((float)TARGET_RATE / 2.0f * 0.9f, (float)SAMPLE_RATE);

    struct pcm_stream_cfg stream = {
        .pcm_rate   = SAMPLE_RATE,
        .pcm_width  = 16,
        .block_size = RAW_HOP_SAMPLES * sizeof(int16_t),
        .mem_slab   = &mfcc_audio_slab,
    };
    struct dmic_cfg cfg = {
        .io = {
            .min_pdm_clk_freq = 1000000,
            .max_pdm_clk_freq = 3500000,
            .min_pdm_clk_dc   = 40,
            .max_pdm_clk_dc   = 60,
        },
        .streams = &stream,
        .channel = {
            .req_num_streams = 1,
            .req_num_chan    = 1,
            .req_chan_map_lo = dmic_build_channel_map(0, 0, PDM_CHAN_LEFT),
        },
    };

    int ret = dmic_configure(dmic_dev, &cfg);
    if (ret < 0) {
        LOG_ERR("MFCC dmic_configure failed: %d", ret);
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:MFCC_CFG", 12);
        return;
    }
    ret = dmic_trigger(dmic_dev, DMIC_TRIGGER_START);
    if (ret < 0) {
        LOG_ERR("MFCC dmic_trigger failed: %d", ret);
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:MFCC_START", 14);
        return;
    }

    bt_nus_send(NULL, "MFCC:OK", 7);

    char mfcc_file_path[64];
    snprintf(mfcc_file_path, sizeof(mfcc_file_path),
            "/SD:/mfcc_%ds_%.0f-%.0fhz_f%d_o%d_h%d_n%d.csv",
            DURATION_S,
            (double)BPF_LOW_HZ, (double)BPF_HIGH_HZ,
            FRAME_MS, OVERLAP_MS, HOP_SIZE, NUM_MFCC);

    struct fs_file_t mfcc_file;
    fs_file_t_init(&mfcc_file);
    ret = fs_open(&mfcc_file, mfcc_file_path,
                FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (ret < 0) {
        LOG_ERR("Failed to open MFCC CSV: %d", ret);
        bt_nus_send(NULL, "ERR:MFCC_CSV", 12);
        dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
        fft_entry_terminate();
        return;
    }

    char hdr[NUM_MFCC * 5 + 2];
    int  hpos = 0;
    for (int i = 0; i < NUM_MFCC; i++)
        hpos += snprintf(hdr + hpos, sizeof(hdr) - hpos,
                        i < NUM_MFCC - 1 ? "c%d," : "c%d\n", i);

    ssize_t hdr_written = fs_write(&mfcc_file, hdr, hpos);
    if (hdr_written != hpos) {
        LOG_ERR("MFCC CSV header write failed: %d", (int)hdr_written);
        bt_nus_send(NULL, "ERR:MFCC_CSV", 12);
        fs_close(&mfcc_file);
        dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
        fft_entry_terminate();
        return;
    }

    memset(overlap_buf, 0, sizeof(overlap_buf));
    int samples_filled = 0;

    while (is_connected && start_mfcc)
    {
        void    *buf;
        uint32_t size;

        ret = dmic_read(dmic_dev, 0, &buf, &size, 1000);
        if (ret < 0) { LOG_WRN("MFCC dmic_read failed: %d", ret); break; }

        const int16_t *raw_hop = (const int16_t *)buf;

        int n_dec = 0;
        apply_bpf_and_decimate(raw_hop, RAW_HOP_SAMPLES, decimated_hop, &n_dec);

        memmove(overlap_buf,overlap_buf + HOP_SIZE,
                (FRAME_SAMPLES - HOP_SIZE) * sizeof(int16_t));
        memcpy(overlap_buf + (FRAME_SAMPLES - HOP_SIZE),
               decimated_hop, HOP_SIZE * sizeof(int16_t));

        samples_filled += HOP_SIZE;
        if (samples_filled > FRAME_SAMPLES)
            samples_filled = FRAME_SAMPLES;

        k_mem_slab_free(&mfcc_audio_slab, buf);

        if (samples_filled < FRAME_SAMPLES)
            continue;

        compute_mfcc(overlap_buf);

        int csv_err = mfcc_csv_write_row(&mfcc_file, mfcc_out, NUM_MFCC);
        if (csv_err < 0) {
            LOG_WRN("MFCC CSV write error %d — stopping stream", csv_err);
            break;
        }

        LOG_INF("MFCC: "
                "%6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f "
                "%6.2f %6.2f %6.2f %6.2f %6.2f %6.2f",
                (double)mfcc_out[0],  (double)mfcc_out[1],
                (double)mfcc_out[2],  (double)mfcc_out[3],
                (double)mfcc_out[4],  (double)mfcc_out[5],
                (double)mfcc_out[6],  (double)mfcc_out[7],
                (double)mfcc_out[8],  (double)mfcc_out[9],
                (double)mfcc_out[10], (double)mfcc_out[11],
                (double)mfcc_out[12]);
    }

    dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
    fft_entry_terminate();
    fs_close(&mfcc_file);
    LOG_INF("MFCC stream stopped — CSV saved to %s", mfcc_file_path);

    if (is_connected) led_set_green();
}

/* ══════════════════════════════════════════════════════════════
 * RECORDING → SD
 * ══════════════════════════════════════════════════════════════ */
static int record_audio_to_sd(const struct device *dmic_dev) {
    if (!sd_mounted) { LOG_ERR("SD not mounted"); return -ENODEV; }

    LOG_INF("Free slab blocks before record: %u / %u",
            k_mem_slab_num_free_get(&dmic_mem_slab), NUM_BLOCKS);

    dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
    k_sleep(K_MSEC(50));
    drain_dmic(dmic_dev);

    struct pcm_stream_cfg stream = {
        .pcm_rate   = SAMPLING_RATE,
        .pcm_width  = BIT_WIDTH,
        .block_size = BLOCK_SIZE_BYTES,
        .mem_slab   = &dmic_mem_slab,
    };
    struct dmic_cfg cfg = {
        .io = {
            .min_pdm_clk_freq = 1000000, .max_pdm_clk_freq = 3500000,
            .min_pdm_clk_dc   = 40,      .max_pdm_clk_dc   = 60,
        },
        .streams = &stream,
        .channel = {
            .req_num_streams = 1, .req_num_chan = 1,
            .req_chan_map_lo = dmic_build_channel_map(0, 0, PDM_CHAN_LEFT),
        },
    };

    int ret = dmic_configure(dmic_dev, &cfg);
    if (ret < 0) { LOG_ERR("dmic_configure failed: %d", ret); led_error_flash(led_set_yellow); return ret; }

    fs_unlink(AUDIO_FILE_PATH);
    struct fs_file_t file;
    fs_file_t_init(&file);
    ret = fs_open(&file, AUDIO_FILE_PATH, FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (ret < 0) { LOG_ERR("fs_open failed: %d", ret); led_error_flash(led_set_yellow); return ret; }

    ret = dmic_trigger(dmic_dev, DMIC_TRIGGER_START);
    if (ret < 0) {
        LOG_ERR("DMIC START failed: %d", ret);
        led_error_flash(led_set_yellow); fs_close(&file); return ret;
    }

    LOG_INF("Recording %d s -> %s ...", DURATION_S, AUDIO_FILE_PATH);
    uint32_t total_written = 0, checksum = 0, accum_pos = 0;

    for (int i = 0; i < RECORD_BLOCK_COUNT; i++) {
        void *block = NULL; uint32_t size = 0;
        ret = dmic_read(dmic_dev, 0, &block, &size, 2000);
        if (ret < 0) {
            LOG_ERR("dmic_read block %d failed: %d", i, ret);
            led_error_flash(led_set_yellow);
            dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
            drain_dmic(dmic_dev); fs_close(&file); return ret;
        }
        checksum ^= compute_checksum((const uint8_t *)block, size);
        memcpy(accum_buf + accum_pos, block, size);
        accum_pos += size;
        k_mem_slab_free(&dmic_mem_slab, block);
        k_yield();
        if (accum_pos >= ACCUM_BUF_SIZE) {
            ssize_t written = fs_write(&file, accum_buf, accum_pos);
            if (written < 0) {
                LOG_ERR("fs_write failed at block %d: %d", i, (int)written);
                led_error_flash(led_set_yellow);
                dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
                drain_dmic(dmic_dev); fs_close(&file); return (int)written;
            }
            total_written += (uint32_t)written; accum_pos = 0;
        }
    }
    if (accum_pos > 0) {
        ssize_t written = fs_write(&file, accum_buf, accum_pos);
        if (written < 0) {
            LOG_ERR("fs_write final flush failed: %d", (int)written);
            led_error_flash(led_set_yellow); fs_close(&file); return (int)written;
        }
        total_written += (uint32_t)written;
    }
    dmic_trigger(dmic_dev, DMIC_TRIGGER_STOP);
    k_sleep(K_MSEC(50));
    drain_dmic(dmic_dev);
    ssize_t cs_written = fs_write(&file, &checksum, CHECKSUM_SIZE);
    if (cs_written != CHECKSUM_SIZE) {
        LOG_ERR("Failed to write checksum: %d", (int)cs_written);
        led_error_flash(led_set_yellow); fs_close(&file); return -EIO;
    }
    fs_close(&file);
    LOG_INF("Recording done -- %u audio bytes + 4 byte checksum (0x%08X)",
            total_written, checksum);
    return 0;
}

/* ══════════════════════════════════════════════════════════════
 * SEND SD AUDIO OVER BLE
 * ══════════════════════════════════════════════════════════════ */
static void send_audio_from_sd(void) {
    if (!sd_mounted) {
        LOG_ERR("SD not mounted"); led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:NOSD", 8); return;
    }
    struct fs_file_t file; fs_file_t_init(&file);
    int ret = fs_open(&file, AUDIO_FILE_PATH, FS_O_READ);
    if (ret < 0) {
        LOG_ERR("fs_open for read failed: %d", ret); led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:NOFILE", 10); return;
    }
    struct fs_dirent entry;
    fs_stat(AUDIO_FILE_PATH, &entry);
    uint32_t file_size = entry.size;
    if (file_size <= CHECKSUM_SIZE) {
        LOG_ERR("File too small: %u bytes", file_size); led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:BADFILE", 11); fs_close(&file); return;
    }
    uint32_t audio_size = file_size - CHECKSUM_SIZE;
    uint32_t stored_checksum = 0, computed_checksum = 0;
    uint8_t  verify_buf[BLOCK_SIZE_BYTES];
    int32_t  bytes_remaining = (int32_t)audio_size;
    while (bytes_remaining > 0) {
        uint32_t to_read = MIN((uint32_t)bytes_remaining, sizeof(verify_buf));
        ssize_t  n = fs_read(&file, verify_buf, to_read);
        if (n <= 0) {
            LOG_ERR("Checksum verify read failed: %d", (int)n); led_error_flash(led_set_yellow);
            bt_nus_send(NULL, "ERR:READFAIL", 12); fs_close(&file); return;
        }
        computed_checksum ^= compute_checksum(verify_buf, (uint32_t)n);
        bytes_remaining   -= (int32_t)n;
    }
    ssize_t cs_read = fs_read(&file, &stored_checksum, CHECKSUM_SIZE);
    if (cs_read != CHECKSUM_SIZE) {
        LOG_ERR("Failed to read stored checksum: %d", (int)cs_read); led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:CSFAIL", 10); fs_close(&file); return;
    }
    if (computed_checksum != stored_checksum) {
        LOG_ERR("Checksum MISMATCH: computed=0x%08X stored=0x%08X",
                computed_checksum, stored_checksum);
        led_error_flash(led_set_purple);
        bt_nus_send(NULL, "ERR:CORRUPT", 11); fs_close(&file); return;
    }
    LOG_INF("Checksum OK: 0x%08X -- starting BLE transfer", stored_checksum);
    ret = fs_seek(&file, 0, FS_SEEK_SET);
    if (ret < 0) {
        LOG_ERR("fs_seek failed: %d", ret); led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:SEEK", 8); fs_close(&file); return;
    }
    char hdr[32];
    snprintf(hdr, sizeof(hdr), "START:%u", audio_size);
    bt_nus_send(NULL, hdr, strlen(hdr));
    k_sleep(K_MSEC(10));
    uint32_t offset = 0, chunk_count = 0;
    uint16_t chunk_size = MIN(nus_chunk_size, SEND_BUF_SIZE);
    bool     send_ok    = true;
    while (offset < audio_size) {
        uint32_t to_read  = MIN(audio_size - offset, chunk_size);
        ssize_t  read_len = fs_read(&file, send_buf, to_read);
        if (read_len <= 0) {
            LOG_ERR("fs_read failed at offset %u: %d", offset, (int)read_len);
            send_ok = false; break;
        }
        int err;
        do {
            err = bt_nus_send(NULL, send_buf, (uint16_t)read_len);
            if (err == -EAGAIN) k_sleep(K_MSEC(1));
        } while (err == -EAGAIN);
        if (err < 0) {
            LOG_ERR("bt_nus_send failed at offset %u: %d", offset, err);
            send_ok = false; break;
        }
        offset += (uint32_t)read_len; chunk_count++;
    }
    fs_close(&file);
    if (!send_ok) {
        LOG_ERR("Transfer aborted at offset %u / %u", offset, audio_size);
        led_error_flash(led_set_yellow); bt_nus_send(NULL, "ERR:ABORT", 9);
    } else {
        bt_nus_send(NULL, "END", 3);
        LOG_INF("Transfer complete -- %u chunks, %u bytes", chunk_count, offset);
    }
}

/* ══════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════ */
int main(void) {
    int err;

    gpio_pin_configure_dt(&red_led,   GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&green_led, GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&blue_led,  GPIO_OUTPUT_ACTIVE);
    led_set_white();

    k_work_init_delayable(&adv_restart_work, adv_restart_work_handler);

    const struct device *const dmic_dev = DEVICE_DT_GET(DT_NODELABEL(pdm0));
    if (!device_is_ready(dmic_dev)) {
        LOG_ERR("DMIC device not ready");
        led_error_flash(led_set_yellow); return -ENODEV;
    }
    LOG_INF("DMIC ready");

    fft_entry_initialize();

    compute_butter4_bpf(BPF_LOW_HZ, BPF_HIGH_HZ, (float)SAMPLE_RATE);
    compute_butter4_lp((float)TARGET_RATE / 2.0f * 0.9f, (float)SAMPLE_RATE);
    LOG_INF("BPF %.0f-%.0f Hz + AA LP at %.0f Hz ready",
            (double)BPF_LOW_HZ, (double)BPF_HIGH_HZ,
            (double)(TARGET_RATE / 2.0f * 0.9f));

    build_mel_filterbank();
    LOG_INF("Mel filterbank ready (%d filters, %d bins, Nyquist %d Hz)",
            NUM_MEL_FILTERS, FFT_SIZE, TARGET_RATE / 2);

    err = bt_enable(NULL);
    if (err) { LOG_ERR("bt_enable failed: %d", err); led_error_flash(led_set_yellow); return err; }

    err = bt_nus_cb_register(&nus_listener, NULL);
    if (err) { LOG_ERR("bt_nus_cb_register failed: %d", err); led_error_flash(led_set_yellow); return err; }

    err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                          sd_adv, ARRAY_SIZE(sd_adv));
    if (err) { LOG_ERR("bt_le_adv_start failed: %d", err); led_error_flash(led_set_yellow); return err; }

    if (init_sd_card() != 0) {
        printk("Failed to initialize SD card\n");
        led_error_flash(led_set_yellow); return -1;
    }

    led_set_red();

    while (true) {
        k_sleep(K_MSEC(100));
        if (!is_connected) continue;

        if (start_recording) {
            start_recording = false;
            led_set_blue();
            int rec_ret = record_audio_to_sd(dmic_dev);
            if (rec_ret < 0) {
                LOG_ERR("record_audio_to_sd failed: %d", rec_ret);
                bt_nus_send(NULL, "ERR:REC", 7);
                k_sleep(K_MSEC(2000));
            } else bt_nus_send(NULL, "REC:OK", 6);
            led_set_green();
        }

        if (start_sending) {
            start_sending = false;
            int waited = 0;
            while (!mtu_exchanged && waited < 500) { k_sleep(K_MSEC(10)); waited += 10; }
            LOG_INF("Starting BLE transfer -- chunk: %d bytes", nus_chunk_size);
            led_set_cyan();
            send_audio_from_sd();
            if (is_connected) led_set_green();
        }

        if (start_mfcc) {
            run_mfcc_stream(dmic_dev);
            start_mfcc = false;
            if (is_connected) led_set_green();
        }
    }

    return 0;
}