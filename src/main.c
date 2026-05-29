/*
 * AcoustEEEcare — SAADC BLE Edition
 * ============================================================
 * v6.7 — BLE-only; SD card support removed. Floating-pin noise fix.
 *
 * CHANGES FROM v6.6:
 *   Removed all SD card code (USE_SD, sd_ring, sd_writer_thread,
 *     fs_mount, ff.h, disk_access, etc.).  BLE-only behaviour matches
 *     the original v6.3 design.
 *
 *   Fixed floating-input noise on AIN0:
 *     resistor_p changed from NRF_SAADC_RESISTOR_DISABLED to
 *     NRF_SAADC_RESISTOR_PULLDOWN.  The nRF52840 SAADC has ~11 kΩ
 *     internal pull resistors.  When AIN0 is disconnected (floating),
 *     enabling the pulldown ties it to GND, which eliminates the
 *     random noise the floating node picks up.
 *     Re-enable the microphone and this resistor is still fine for
 *     cap-coupled MEMS mics (the bias resistor on the mic board wins).
 *     If you drive the pin from a low-impedance source you can change
 *     back to NRF_SAADC_RESISTOR_DISABLED.
 *
 *   Added NOISE_GATE_THRESHOLD as a second software layer.
 *     Set to 0 to disable (default). Raise to e.g. 10–20 raw ADC
 *     counts if residual noise is still audible on the host side.
 *
 * TARGET HARDWARE
 *   Seeed XIAO nRF52840
 *   Analog MEMS microphone on AIN0 (P0.02), cap-coupled (optional)
 * ============================================================
 */

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

LOG_MODULE_REGISTER(AcoustEEEcare);

/* ══════════════════════════════════════════════════════════════════
 * SAADC CONFIG
 * ══════════════════════════════════════════════════════════════════ */
#define SAADC_CC_VALUE      2000U   /* 16 MHz / 2000 = 8 kHz */
#define SAADC_IRQ_PRIORITY  6

static const nrfx_saadc_channel_t saadc_channel_cfg = {
    .channel_config = {
        .resistor_p = NRF_SAADC_RESISTOR_VDD1_2, /* FIX: was DISABLED.
                                                     * Pulls AIN0 to GND when
                                                     * nothing drives the pin,
                                                     * eliminating floating-
                                                     * input noise.  ~11 kΩ
                                                     * internal pull. */
        .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
        .gain       = NRF_SAADC_GAIN1_4,                 /* tune — see note */
        .reference  = NRF_SAADC_REFERENCE_VDD4,      /* 0.6 V, decoupled from VDD */
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

/*
 * Software noise gate — zeroes samples whose magnitude is below this
 * threshold (in raw ADC counts after DC removal).
 * Set to 0 to disable.  Try 10–20 if residual noise still reaches the host.
 */
#define NOISE_GATE_THRESHOLD  0

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
 * SAADC EVENT HANDLER
 *
 * On NRFX_SAADC_EVT_DONE (one half-buffer of HALF_BUF_SAMPLES ready):
 *   1. DC removal         — in-place IIR high-pass
 *   2. Noise gate         — optional zero-below-threshold (NOISE_GATE_THRESHOLD)
 *   3. dsp_mfcc_feed_chunk() — on-the-fly bandpass + decimate
 *   4. ring_buf_put()     — push to audio_ring for BLE TX thread
 *
 * No fs_write() here — only ring_buf_put() and k_sem_give().
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

        /* ── Step 2: Software noise gate ── */
#if NOISE_GATE_THRESHOLD > 0
        for (uint32_t i = 0; i < HALF_BUF_SAMPLES; i++) {
            if (filled_buf[i] > -NOISE_GATE_THRESHOLD &&
                filled_buf[i] <  NOISE_GATE_THRESHOLD) {
                filled_buf[i] = 0;
            }
        }
#endif

        /* ── Step 3: On-the-fly bandpass + decimate ── */
        dsp_mfcc_feed_chunk(filled_buf, HALF_BUF_SAMPLES);

        /* ── Step 4: Push to BLE audio ring ── */
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
    LOG_INF("SAADC stopped (overruns=%u, ble_drops=%u)",
            saadc_dma_overruns,
            (uint32_t)atomic_get(&ring_drops));
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
 *   1. Send START header over NUS so the host can pre-allocate.
 *   2. Start SAADC — ISR fills audio_ring.
 *   3. BLE TX thread streams audio live over NUS.
 *   4. After total_halves: stop SAADC, set analog_recording=false.
 *   5. Wait for BLE TX thread to drain ring and send "finished\n".
 *   6. Run MFCC and stream coefficients over NUS.
 * ══════════════════════════════════════════════════════════════════ */
static void record_and_stream(void)
{
    const uint32_t total_halves =
        (TOTAL_AUDIO_SAMPLES + HALF_BUF_SAMPLES - 1) / HALF_BUF_SAMPLES;

    ring_buf_reset(&audio_ring);
    atomic_set(&ring_drops, 0);
    ring_high_water = 0;
    tx_seq          = 0;
    k_sem_reset(&half_produced_sem);
    k_sem_reset(&tx_done_sem);

    /* Notify host of expected byte count */
    char hdr[32];
    snprintf(hdr, sizeof(hdr), "START:%u\n", (uint32_t)TOTAL_AUDIO_BYTES);
    bt_nus_send(NULL, hdr, strlen(hdr));
    k_sleep(K_MSEC(10));

    led_set_cyan();
    analog_recording = true;

    if (saadc_start_streaming() != 0) {
        analog_recording = false;
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:SAADC", 9);
        return;
    }

    LOG_INF("Recording %d s @ %d Hz...", DURATION_S, SAMPLING_RATE);

    /* Count half-buffers produced by the ISR */
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

    /* Wake BLE TX thread to drain the ring and send "finished\n" */
    k_sem_give(&audio_data_sem);

    if (k_sem_take(&tx_done_sem, K_MSEC(10000)) != 0) {
        LOG_WRN("BLE TX did not finish within 10 s");
    }

    LOG_INF("BLE audio stream complete — starting MFCC");
    LOG_INF("audio_ring high water = %u / %u", ring_high_water, AUDIO_RING_BYTES);
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

    k_thread_create(&ble_tx_thread_data, ble_tx_stack,
                    K_THREAD_STACK_SIZEOF(ble_tx_stack),
                    ble_tx_thread_fn, NULL, NULL, NULL,
                    BLE_TX_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&ble_tx_thread_data, "ble_tx");

    led_set_red();
    LOG_INF("AcoustEEEcare v6.7 ready — waiting for BLE connection");

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
 * MEMORY BUDGET (approximate, BLE-only)
 * ════════════════════════════════════════════════════════════════════
 *
 *   Zephyr kernel + BLE stack     ~90 KB
 *   s_decimated[20000] (dsp_mfcc)  40 KB
 *   DSP scratch                     ~7 KB
 *   audio_ring (BLE)                16 KB
 *   ping_pong[2][512]                2 KB
 *   BLE TX thread stack              2 KB
 *   Remaining headroom             ~93 KB   (32 KB freed vs USE_SD build)
 */