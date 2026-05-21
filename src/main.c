/*
 * AcoustEEEcare — SD-free BLE-streaming + TFLite Micro Edition
 * =============================================================
 * v8.1 — All 12 review issues from code-review session addressed.
 *
 * Fix log (mapped to review issue numbers):
 *   #1  Protocol: host strips .npy header before sending; firmware receives
 *       raw int8 only. Documented explicitly below and in nus_received().
 *   #2  Result delivery: HR/RR sent as "HR:<val>\n" BLE strings AND saved
 *       to results.txt on the host (receiver.py handles the file write).
 *       Documented explicitly so the two sides agree.
 *   #3  quant_sent lifted to file scope — reset on disconnect now works.
 *   #4  mfcc_rx_received/expected explicitly zeroed before HEART_DONE signal.
 *   #5  mfcc_rx_received protected by K_SPINLOCK; mfcc_rx_state is
 *       atomic_t (32-bit, aligned — safe on Cortex-M4).
 *   #6  END sentinel checks gated on current receive state.
 *   #7  dsp_mfcc.c truncation: BUILD_ASSERT added (see dsp_mfcc.c note).
 *       Here: HALF_BUF_SAMPLES verified at compile time.
 *   #8  saadc_event_handler forward declaration added.
 *   #9  When ENABLE_LUNG_MODEL=0, firmware sends "LUNG_DISABLED\n" so
 *       host can skip Phase 4 entirely.
 *   #10 seq16 validated; gap detection logs WRN and aborts transfer.
 *   #11 Flash budget: see build notes below.
 *   #12 MFCC counters reset before "HEART_DONE\n" is sent.
 *
 * PIPELINE:
 *   Phase 1 — SAADC capture + live BLE audio stream (START:N ... finished)
 *   Phase 2 — Receive MFCC_HEART_START:N, raw int8 chunks, MFCC_HEART_END
 *             (host strips .npy header; only raw int8 payload is sent)
 *   Phase 3 — Heart inference → send "HR:<val>\n"
 *   Phase 4 — Receive MFCC_LUNG_START:N, raw int8 chunks, MFCC_LUNG_END
 *             (skipped if ENABLE_LUNG_MODEL=0; host receives "LUNG_DISABLED\n")
 *   Phase 5 — Lung inference → send "RR:<val>\n"
 *             (host saves both values to results.txt)
 *
 * BUILD NOTE (Issue #11):
 *   Lung mel filterbank = 26*1025*4 = 106 600 B in flash.
 *   Run `west build -t rom_report` and confirm total < ~900 KB.
 *
 * PROTOCOL (NUS, little-endian chunked):
 *   nRF → host:  "QUANT_HEART:<scale_hex>:<zp>\n"
 *                "QUANT_LUNG:<scale_hex>:<zp>\n"   (or "LUNG_DISABLED\n")
 *                "START:<total_bytes>\n"
 *                [seq16 LE][len16 LE][pcm_bytes...]
 *                "finished\n"
 *                "HR:<value>\n"
 *                "HEART_DONE\n"
 *                "RR:<value>\n"                     (or absent if lung disabled)
 *                "ERR:<code>\n"
 *
 *   host → nRF:  "REC"
 *                "MFCC_HEART_START:<n_bytes>\n"
 *                [seq16 LE][len16 LE][raw_int8_bytes...]   (.npy header stripped)
 *                "MFCC_HEART_END\n"
 *                "MFCC_LUNG_START:<n_bytes>\n"             (skipped if LUNG_DISABLED)
 *                [seq16 LE][len16 LE][raw_int8_bytes...]
 *                "MFCC_LUNG_END\n"
 * =============================================================
 */

#define BLE_AUDIO_LIVE  1
#define DSP_OFFLINE     0

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
#include <zephyr/sys/atomic.h>
#include <zephyr/spinlock.h>
#include "zephyr/kernel/thread_stack.h"

#include <nrfx.h>
#include <drivers/nrfx_errors.h>
#include <nrfx_saadc.h>

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
/* Issue #8: forward declaration so saadc_init() can reference handler */
static void saadc_event_handler(nrfx_saadc_evt_t const *p_event);

/* ══════════════════════════════════════════════════════════════════
 * COMPILE-TIME SANITY (Issue #7)
 * ══════════════════════════════════════════════════════════════════ */
/* BP_SCRATCH_MAX in dsp_mfcc.c is 512. HALF_BUF_SAMPLES must not
 * exceed it or dsp_mfcc_feed_chunk() will silently truncate. */
#define BP_SCRATCH_MAX_SHADOW 512
BUILD_ASSERT(HALF_BUF_SAMPLES <= BP_SCRATCH_MAX_SHADOW,
             "HALF_BUF_SAMPLES must not exceed BP_SCRATCH_MAX in dsp_mfcc.c");

/* ══════════════════════════════════════════════════════════════════
 * TENSOR ARENA
 * ══════════════════════════════════════════════════════════════════ */
#define TENSOR_ARENA_BYTES  (72u * 1024u)
static uint8_t tensor_arena[TENSOR_ARENA_BYTES] __aligned(16);

volatile uint32_t g_tflm_arena_used_bytes = 0;

/* ══════════════════════════════════════════════════════════════════
 * MFCC RECEIVE BUFFER
 *
 * Issue #1: The host MUST strip the .npy magic/header (128 bytes)
 * before sending. This buffer receives raw int8 payload only.
 *
 * Heart: 665 * 25 = 16,625 bytes (int8)
 * Lung:  324 * 26 =  8,424 bytes (int8)
 * Sized for the larger (heart).
 * ══════════════════════════════════════════════════════════════════ */
#define HEART_MFCC_BYTES  (665 * 25)   /* 16,625 B */
#define LUNG_MFCC_BYTES   (324 * 26)   /*  8,424 B */
#define MFCC_RX_BUF_SIZE  HEART_MFCC_BYTES

static int8_t   mfcc_rx_buf[MFCC_RX_BUF_SIZE];
/* Issue #5: mfcc_rx_received guarded by spinlock; written in BLE
 * callback, read in main thread. */
static struct k_spinlock mfcc_rx_lock;
static uint32_t mfcc_rx_received;   /* bytes received so far  (guarded) */
static uint32_t mfcc_rx_expected;   /* bytes expected          (guarded) */
static uint16_t mfcc_rx_next_seq;   /* expected next seq16     (guarded) */

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

#define CHUNK_HEADER_BYTES  4

/* ══════════════════════════════════════════════════════════════════
 * BLE AUDIO RING BUFFER  (16 KB)
 * ══════════════════════════════════════════════════════════════════ */
#define AUDIO_RING_BYTES  (16 * 1024)
RING_BUF_DECLARE(audio_ring, AUDIO_RING_BYTES);

static K_SEM_DEFINE(half_produced_sem, 0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(audio_data_sem,    0, K_SEM_MAX_LIMIT);
static K_SEM_DEFINE(tx_done_sem,       0, 1);

static atomic_t  ring_drops;
static uint32_t  ring_high_water = 0;
static uint16_t  tx_seq          = 0;

/* ══════════════════════════════════════════════════════════════════
 * MFCC RECEIVE STATE MACHINE
 * ══════════════════════════════════════════════════════════════════ */
typedef enum {
    MFCC_STATE_IDLE,
    MFCC_STATE_RECV_HEART,
    MFCC_STATE_RECV_LUNG,
} mfcc_rx_state_t;

/* Issue #5: use atomic so main-thread reads are safe without full lock */
static atomic_t mfcc_rx_state_atomic;   /* stores mfcc_rx_state_t values */

static K_SEM_DEFINE(heart_mfcc_done_sem, 0, 1);
static K_SEM_DEFINE(lung_mfcc_done_sem,  0, 1);

/* Helper macros */
#define mfcc_rx_state_get()      ((mfcc_rx_state_t)atomic_get(&mfcc_rx_state_atomic))
#define mfcc_rx_state_set(s)     atomic_set(&mfcc_rx_state_atomic, (atomic_val_t)(s))

/* ══════════════════════════════════════════════════════════════════
 * FLAGS
 * ══════════════════════════════════════════════════════════════════ */
static volatile bool start_recording = false;
static volatile bool is_connected    = false;
static volatile bool mtu_exchanged   = false;

/* Issue #3: lifted to file scope so both branches of main() share it */
static bool quant_sent = false;

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
 * BLE ADVERTISING
 * ══════════════════════════════════════════════════════════════════ */
#define DEVICE_NAME      "AcoustEEEcare"
#define DEVICE_NAME_LEN  (sizeof(DEVICE_NAME) - 1)

static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
    BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),
};
static const struct bt_data sd_ble[] = {
    BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_SRV_VAL),
};

/* ══════════════════════════════════════════════════════════════════
 * NUS RECEIVE CALLBACK
 *
 * Issue #1: Binary chunks carry raw int8 MFCC data only.
 *           The host MUST strip the 128-byte .npy header before
 *           sending. No .npy header parsing is performed here.
 *
 * Issue #5: mfcc_rx_received is incremented under mfcc_rx_lock
 *           to prevent torn reads from the main thread.
 *
 * Issue #6: HEART_END and LUNG_END are only accepted in their
 *           matching receive state (not cross-matched).
 *
 * Issue #10: seq16 is validated; a gap aborts the transfer with
 *            ERR:SEQ_GAP so the host can retry.
 * ══════════════════════════════════════════════════════════════════ */
static void nus_received(struct bt_conn *conn,
                         const void *data, uint16_t len, void *ctx)
{
    ARG_UNUSED(conn); ARG_UNUSED(ctx);

    const uint8_t       *buf   = (const uint8_t *)data;
    mfcc_rx_state_t      state = mfcc_rx_state_get();

    /* ── Binary chunk path ── */
    if (state == MFCC_STATE_RECV_HEART || state == MFCC_STATE_RECV_LUNG) {

        /* Issue #6: gate END sentinel on current state only */
        if (state == MFCC_STATE_RECV_HEART &&
            len >= 14 && memcmp(buf, "MFCC_HEART_END", 14) == 0) {
            k_spinlock_key_t key = k_spin_lock(&mfcc_rx_lock);
            uint32_t received = mfcc_rx_received;
            k_spin_unlock(&mfcc_rx_lock, key);
            LOG_INF("MFCC_HEART_END received (%u / %u B)",
                    received, mfcc_rx_expected);
            mfcc_rx_state_set(MFCC_STATE_IDLE);
            k_sem_give(&heart_mfcc_done_sem);
            return;
        }

        if (state == MFCC_STATE_RECV_LUNG &&
            len >= 13 && memcmp(buf, "MFCC_LUNG_END", 13) == 0) {
            k_spinlock_key_t key = k_spin_lock(&mfcc_rx_lock);
            uint32_t received = mfcc_rx_received;
            k_spin_unlock(&mfcc_rx_lock, key);
            LOG_INF("MFCC_LUNG_END received (%u / %u B)",
                    received, mfcc_rx_expected);
            mfcc_rx_state_set(MFCC_STATE_IDLE);
            k_sem_give(&lung_mfcc_done_sem);
            return;
        }

        /* Binary chunk: [seq16 LE][len16 LE][payload] */
        if (len < CHUNK_HEADER_BYTES) {
            LOG_WRN("Short chunk (len=%u) — discarding", len);
            return;
        }

        uint16_t seq         = sys_get_le16(buf);
        uint16_t payload_len = sys_get_le16(buf + 2);

        if (payload_len + CHUNK_HEADER_BYTES > len) {
            LOG_WRN("Chunk payload_len=%u overruns packet len=%u",
                    payload_len, len);
            return;
        }

        /* Issue #10: sequence gap detection */
        k_spinlock_key_t key = k_spin_lock(&mfcc_rx_lock);
        if (seq != mfcc_rx_next_seq) {
            LOG_ERR("SEQ GAP: expected %u got %u — aborting MFCC transfer",
                    mfcc_rx_next_seq, seq);
            mfcc_rx_received = 0;
            mfcc_rx_expected = 0;
            k_spin_unlock(&mfcc_rx_lock, key);
            mfcc_rx_state_set(MFCC_STATE_IDLE);
            /* Notify host; ble_send_reliable not safe from callback —
             * use bt_nus_send directly with no retry (best-effort). */
            bt_nus_send(NULL, "ERR:SEQ_GAP\n", 12);
            return;
        }
        mfcc_rx_next_seq = (uint16_t)(seq + 1);

        const uint8_t *payload  = buf + CHUNK_HEADER_BYTES;
        uint32_t remaining = mfcc_rx_expected - mfcc_rx_received;
        uint32_t to_copy   = (payload_len < remaining) ? payload_len : remaining;

        if (to_copy > 0) {
            memcpy(mfcc_rx_buf + mfcc_rx_received, payload, to_copy);
            mfcc_rx_received += to_copy;
        }

        uint32_t received_snapshot = mfcc_rx_received;
        k_spin_unlock(&mfcc_rx_lock, key);

        if (received_snapshot >= mfcc_rx_expected) {
            LOG_INF("MFCC buffer full (%u B) — waiting for END sentinel",
                    received_snapshot);
        }
        return;
    }

    /* ── ASCII control message path ── */

    if (len == 3 && memcmp(buf, "REC", 3) == 0) {
        start_recording = true;
        return;
    }

    /* "MFCC_HEART_START:<N>\n" */
    if (len > 17 && memcmp(buf, "MFCC_HEART_START:", 17) == 0) {
        char tmp[16] = {0};
        uint16_t copy = (len - 17 < 15) ? (len - 17) : 15;
        memcpy(tmp, buf + 17, copy);

        k_spinlock_key_t key = k_spin_lock(&mfcc_rx_lock);
        mfcc_rx_expected = (uint32_t)strtoul(tmp, NULL, 10);
        mfcc_rx_received = 0;
        mfcc_rx_next_seq = 0;
        if (mfcc_rx_expected > MFCC_RX_BUF_SIZE) {
            LOG_ERR("MFCC_HEART_START: expected %u B > buf %u B — clamping",
                    mfcc_rx_expected, (uint32_t)MFCC_RX_BUF_SIZE);
            mfcc_rx_expected = MFCC_RX_BUF_SIZE;
        }
        k_spin_unlock(&mfcc_rx_lock, key);

        LOG_INF("MFCC_HEART_START: expecting %u B (raw int8, no .npy header)",
                mfcc_rx_expected);
        mfcc_rx_state_set(MFCC_STATE_RECV_HEART);
        return;
    }

    /* "MFCC_LUNG_START:<N>\n" */
    if (len > 16 && memcmp(buf, "MFCC_LUNG_START:", 16) == 0) {
        char tmp[16] = {0};
        uint16_t copy = (len - 16 < 15) ? (len - 16) : 15;
        memcpy(tmp, buf + 16, copy);

        k_spinlock_key_t key = k_spin_lock(&mfcc_rx_lock);
        mfcc_rx_expected = (uint32_t)strtoul(tmp, NULL, 10);
        mfcc_rx_received = 0;
        mfcc_rx_next_seq = 0;
        if (mfcc_rx_expected > MFCC_RX_BUF_SIZE) {
            LOG_ERR("MFCC_LUNG_START: expected %u B > buf %u B — clamping",
                    mfcc_rx_expected, (uint32_t)MFCC_RX_BUF_SIZE);
            mfcc_rx_expected = MFCC_RX_BUF_SIZE;
        }
        k_spin_unlock(&mfcc_rx_lock, key);

        LOG_INF("MFCC_LUNG_START: expecting %u B (raw int8, no .npy header)",
                mfcc_rx_expected);
        mfcc_rx_state_set(MFCC_STATE_RECV_LUNG);
        return;
    }

    LOG_WRN("Unknown NUS message (len=%u): %.*s", len, (int)len, (const char *)buf);
}

static struct bt_nus_cb nus_listener = { .received = nus_received };

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
        LOG_INF("MTU=%d -> NUS chunk=%d", mtu, nus_chunk_size);
    } else {
        LOG_WRN("MTU exchange failed (%d), using default=%d", err, nus_chunk_size);
    }
    mtu_exchanged = true;
}

static struct bt_gatt_exchange_params exchange_params = { .func = mtu_exchange_cb };

static void adv_restart_work_handler(struct k_work *work)
{
    bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                    sd_ble, ARRAY_SIZE(sd_ble));
}

static void send_quant_params(void)
{
    float   heart_scale = 0.0f, lung_scale = 0.0f;
    int32_t heart_zp    = 0,    lung_zp    = 0;
    char    msg[64];
    int     err;

    if (get_heart_quant_params(tensor_arena, TENSOR_ARENA_BYTES,
                               &heart_scale, &heart_zp) == 0) {
        int n = snprintf(msg, sizeof(msg), "QUANT_HEART:%a:%d\n",
                         (double)heart_scale, (int)heart_zp);
        do {
            err = bt_nus_send(NULL, msg, (uint16_t)n);
            if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(5));
        } while (err == -ENOMEM || err == -EAGAIN);
        LOG_INF("Sent: %s", msg);
    }

#if defined(ENABLE_LUNG_MODEL) && ENABLE_LUNG_MODEL
    if (get_lung_quant_params(tensor_arena, TENSOR_ARENA_BYTES,
                              &lung_scale, &lung_zp) == 0) {
        int n = snprintf(msg, sizeof(msg), "QUANT_LUNG:%a:%d\n",
                         (double)lung_scale, (int)lung_zp);
        do {
            err = bt_nus_send(NULL, msg, (uint16_t)n);
            if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(5));
        } while (err == -ENOMEM || err == -EAGAIN);
        LOG_INF("Sent: %s", msg);
    }
#else
    /* Issue #9: tell host lung is disabled so it skips Phase 4 entirely */
    do {
        err = bt_nus_send(NULL, "LUNG_DISABLED\n", 14);
        if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(5));
    } while (err == -ENOMEM || err == -EAGAIN);
    LOG_INF("Sent: LUNG_DISABLED");
#endif
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
    mfcc_rx_state_set(MFCC_STATE_IDLE);
    nus_chunk_size   = 244;
    /* Issue #3: quant_sent is now file-scope — reset properly here */
    quant_sent       = false;
    led_set_red();
    LOG_WRN("Disconnected (reason=%d)", reason);
    k_work_schedule(&adv_restart_work, K_MSEC(500));
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
    .connected    = connected,
    .disconnected = disconnected,
};

/* ══════════════════════════════════════════════════════════════════
 * BLE TX THREAD
 * ══════════════════════════════════════════════════════════════════ */
#define BLE_TX_STACK_SIZE  2048
#define BLE_TX_PRIORITY    5

static K_THREAD_STACK_DEFINE(ble_tx_stack, BLE_TX_STACK_SIZE);
static struct k_thread ble_tx_thread_data;

static void ble_tx_thread_fn(void *a, void *b, void *c)
{
    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);

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
}

/* ══════════════════════════════════════════════════════════════════
 * SAADC
 * ══════════════════════════════════════════════════════════════════ */
static bool saadc_was_initialized = false;

static int saadc_init(void)
{
    nrfx_err_t err;

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

    saadc_was_initialized = true;
    return 0;
}

/* Issue #8: definition follows declaration above */
static void saadc_event_handler(nrfx_saadc_evt_t const *p_event)
{
    switch (p_event->type) {

    case NRFX_SAADC_EVT_BUF_REQ:
        nrfx_saadc_buffer_set(ping_pong[next_dma_buf], HALF_BUF_SAMPLES);
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
        if (analog_recording) {
            uint32_t written = ring_buf_put(&audio_ring,
                                            (const uint8_t *)filled_buf,
                                            HALF_BUF_BYTES);
            if (written != HALF_BUF_BYTES) {
                atomic_inc(&ring_drops);
                LOG_WRN_ONCE("audio_ring overflow — BLE stream has gaps!");
            } else {
                uint32_t fill = ring_buf_size_get(&audio_ring);
                if (fill > ring_high_water) ring_high_water = fill;
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

static int saadc_start_streaming(void)
{
    nrfx_err_t err;
    saadc_dma_overruns = 0;
    dc_estimate        = 2048;
    next_dma_buf       = 1;

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
            (uint32_t)atomic_get(&ring_drops));
}

/* ══════════════════════════════════════════════════════════════════
 * ble_send_reliable
 * ══════════════════════════════════════════════════════════════════ */
static void ble_send_reliable(const void *data, uint16_t len)
{
    int err;
    do {
        err = bt_nus_send(NULL, data, len);
        if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(5));
    } while (err == -ENOMEM || err == -EAGAIN);
}

/* ══════════════════════════════════════════════════════════════════
 * MFCC RX SNAPSHOT HELPER (safe read from main thread)
 * ══════════════════════════════════════════════════════════════════ */
static uint32_t mfcc_rx_received_get(void)
{
    k_spinlock_key_t key = k_spin_lock(&mfcc_rx_lock);
    uint32_t v = mfcc_rx_received;
    k_spin_unlock(&mfcc_rx_lock, key);
    return v;
}

/* ══════════════════════════════════════════════════════════════════
 * RECORD AND STREAM
 * ══════════════════════════════════════════════════════════════════ */
static void record_and_stream(void)
{
    const uint32_t total_halves =
        (TOTAL_AUDIO_SAMPLES + HALF_BUF_SAMPLES - 1) / HALF_BUF_SAMPLES;

    ring_buf_reset(&audio_ring);
    atomic_set(&ring_drops, 0);
    ring_high_water = 0;
    tx_seq          = 0;
    k_sem_reset(&tx_done_sem);
    k_sem_reset(&half_produced_sem);
    k_sem_reset(&heart_mfcc_done_sem);
    k_sem_reset(&lung_mfcc_done_sem);
    mfcc_rx_state_set(MFCC_STATE_IDLE);

    /* Issue #4 + #12: explicit counter reset at start of session */
    {
        k_spinlock_key_t key = k_spin_lock(&mfcc_rx_lock);
        mfcc_rx_received = 0;
        mfcc_rx_expected = 0;
        mfcc_rx_next_seq = 0;
        k_spin_unlock(&mfcc_rx_lock, key);
    }

    led_set_cyan();

    /* ════════════════════════════════════════════════════════════
     * PHASE 1: SAADC capture + live BLE audio stream
     * ════════════════════════════════════════════════════════════ */
    LOG_INF("Phase 1: SAADC capture — %d s @ %d Hz", DURATION_S, SAMPLING_RATE);

    char ctrl[32];
    int  ctrl_len = snprintf(ctrl, sizeof(ctrl),
                             "START:%u\n", (uint32_t)TOTAL_AUDIO_BYTES);
    ble_send_reliable(ctrl, (uint16_t)ctrl_len);
    k_sleep(K_MSEC(20));

    analog_recording = true;

    if (saadc_start_streaming() != 0) {
        analog_recording = false;
        ble_send_reliable("ERR:SAADC", 9);
        led_error_flash(led_set_yellow);
        return;
    }

    led_set_white();

    for (uint32_t h = 0; h < total_halves; h++) {
        if (k_sem_take(&half_produced_sem, K_MSEC(500)) != 0) {
            LOG_ERR("SAADC timeout at half %u/%u", h, total_halves);
            analog_recording = false;
            saadc_stop_streaming();
            ble_send_reliable("ERR:TIMEOUT", 11);
            return;
        }
    }

    saadc_stop_streaming();
    analog_recording = false;

    led_set_green();

    LOG_INF("Phase 1: waiting for BLE TX drain...");
    if (k_sem_take(&tx_done_sem, K_MSEC(30000)) != 0) {
        LOG_WRN("BLE TX drain timeout — host may have missed tail");
    }

    LOG_INF("Phase 1 done. ring_high_water=%u drops=%u",
            ring_high_water, (uint32_t)atomic_get(&ring_drops));

    /* ════════════════════════════════════════════════════════════
     * PHASE 2: Receive heart MFCC from host (raw int8, .npy stripped)
     * ════════════════════════════════════════════════════════════ */
    LOG_INF("Phase 2: waiting for MFCC_HEART_START from host...");
    led_set_blue();

    if (k_sem_take(&heart_mfcc_done_sem, K_MSEC(60000)) != 0) {
        LOG_ERR("Timeout waiting for heart MFCC");
        ble_send_reliable("ERR:MFCC_HEART_TIMEOUT", 22);
        led_error_flash(led_set_yellow);
        return;
    }

    uint32_t heart_rx = mfcc_rx_received_get();
    LOG_INF("Phase 2: heart MFCC received (%u B)", heart_rx);

    if (heart_rx < HEART_MFCC_BYTES) {
        LOG_ERR("Heart MFCC incomplete: got %u, expected %u",
                heart_rx, (uint32_t)HEART_MFCC_BYTES);
        ble_send_reliable("ERR:MFCC_HEART_SHORT", 20);
        led_error_flash(led_set_yellow);
        return;
    }

    /* ════════════════════════════════════════════════════════════
     * PHASE 3: Heart inference
     * ════════════════════════════════════════════════════════════ */
    LOG_INF("Phase 3: heart inference...");
    led_set_purple();

    heart_result_t heart_result;

#if defined(ENABLE_HEART_MODEL) && ENABLE_HEART_MODEL
    run_heart_inference(tensor_arena, TENSOR_ARENA_BYTES,
                        mfcc_rx_buf,
                        665, 25,
                        &heart_result);

    if (heart_result.rc < 0) {
        LOG_ERR("Heart inference failed: %d", heart_result.rc);
        ble_send_reliable("ERR:HEART_INF", 13);
    } else {
        char result_msg[32];
        int  rlen = snprintf(result_msg, sizeof(result_msg),
                             "HR:%.0f\n", (double)heart_result.value);
        ble_send_reliable(result_msg, (uint16_t)rlen);
        LOG_INF("Heart result: HR=%.0f BPM", (double)heart_result.value);
    }
#else
    LOG_INF("Phase 3: heart model disabled — skipping");
    heart_result.rc    = -ENOTSUP;
    heart_result.value = 0.0f;
#endif

    /* ════════════════════════════════════════════════════════════
     * PHASE 4: Receive lung MFCC from host
     * Issue #12: reset counters BEFORE sending HEART_DONE so the
     * NUS callback won't see stale heart values if the host is fast.
     * Issue #9: skip entirely if ENABLE_LUNG_MODEL=0.
     * ════════════════════════════════════════════════════════════ */
#if defined(ENABLE_LUNG_MODEL) && ENABLE_LUNG_MODEL

    LOG_INF("Phase 4: waiting for MFCC_LUNG_START from host...");
    led_set_blue();

    /* Issue #12 + #4: counters cleared BEFORE HEART_DONE is sent */
    {
        k_spinlock_key_t key = k_spin_lock(&mfcc_rx_lock);
        mfcc_rx_received = 0;
        mfcc_rx_expected = 0;
        mfcc_rx_next_seq = 0;
        k_spin_unlock(&mfcc_rx_lock, key);
    }

    ble_send_reliable("HEART_DONE\n", 11);

    if (k_sem_take(&lung_mfcc_done_sem, K_MSEC(60000)) != 0) {
        LOG_ERR("Timeout waiting for lung MFCC");
        ble_send_reliable("ERR:MFCC_LUNG_TIMEOUT", 21);
        led_error_flash(led_set_yellow);
        return;
    }

    uint32_t lung_rx = mfcc_rx_received_get();
    LOG_INF("Phase 4: lung MFCC received (%u B)", lung_rx);

    if (lung_rx < LUNG_MFCC_BYTES) {
        LOG_ERR("Lung MFCC incomplete: got %u, expected %u",
                lung_rx, (uint32_t)LUNG_MFCC_BYTES);
        ble_send_reliable("ERR:MFCC_LUNG_SHORT", 19);
        led_error_flash(led_set_yellow);
        return;
    }

    /* ════════════════════════════════════════════════════════════
     * PHASE 5: Lung inference
     * ════════════════════════════════════════════════════════════ */
    LOG_INF("Phase 5: lung inference...");
    led_set_purple();

    lung_result_t lung_result;
    run_lung_inference(tensor_arena, TENSOR_ARENA_BYTES,
                       mfcc_rx_buf,
                       324, 26,
                       &lung_result);

    if (lung_result.rc < 0) {
        LOG_ERR("Lung inference failed: %d", lung_result.rc);
        ble_send_reliable("ERR:LUNG_INF", 12);
    } else {
        char result_msg[32];
        int  rlen = snprintf(result_msg, sizeof(result_msg),
                             "RR:%.0f\n", (double)lung_result.value);
        ble_send_reliable(result_msg, (uint16_t)rlen);
        LOG_INF("Lung result: RR=%.0f BPM", (double)lung_result.value);
    }

    led_set_green();
    LOG_INF("record_and_stream() complete. HR=%.0f RR=%.0f",
            (double)heart_result.value, (double)lung_result.value);

#else
    /* Issue #9: LUNG_DISABLED already sent in send_quant_params().
     * Still signal HEART_DONE so host knows heart result is final. */
    ble_send_reliable("HEART_DONE\n", 11);
    led_set_green();
    LOG_INF("record_and_stream() complete (lung disabled). HR=%.0f",
            (double)heart_result.value);
#endif
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

    err = bt_enable(NULL);
    if (err) { LOG_ERR("bt_enable: %d", err); return err; }

    err = bt_nus_cb_register(&nus_listener, NULL);
    if (err) { LOG_ERR("bt_nus_cb_register: %d", err); return err; }

    err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                          sd_ble, ARRAY_SIZE(sd_ble));
    if (err) { LOG_ERR("bt_le_adv_start: %d", err); return err; }

    k_thread_create(&ble_tx_thread_data, ble_tx_stack,
                    K_THREAD_STACK_SIZEOF(ble_tx_stack),
                    ble_tx_thread_fn, NULL, NULL, NULL,
                    BLE_TX_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&ble_tx_thread_data, "ble_tx");

    led_set_red();
    LOG_INF("AcoustEEEcare v8.1 ready — waiting for BLE connection");

    while (true) {
        k_sleep(K_MSEC(100));

        if (is_connected && mtu_exchanged && !analog_recording) {
            /* Issue #3: quant_sent is file-scope; reset in disconnected() */
            if (!quant_sent) {
                send_quant_params();
                quant_sent = true;
            }

            if (start_recording) {
                start_recording = false;
                LOG_INF("REC command — starting %d s @ %d Hz",
                        DURATION_S, SAMPLING_RATE);
                record_and_stream();
            }
        }
        /* No else branch needed — disconnect resets quant_sent directly */
    }

    return 0;
}