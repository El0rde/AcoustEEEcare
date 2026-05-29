/*
 * AcoustEEEcare — BLE-to-MFCC firmware (v8.1)
 * ==================================================================
 * SINGLE-THREADED REWRITE of v8.0.
 *
 * Why this rewrite:
 *   v8.0 used a separate DSP thread and BLE TX thread. On the target
 *   those worker threads never executed their loop bodies — proven by
 *   the logs: stage_overruns=160 (ISR filled every staging slot) with
 *   ring_drops=0 (nothing ever written to the audio ring), and 0 audio
 *   + 0 MFCC packets reaching the host, while the main-thread control
 *   strings (START / finished / MFCC_START / MFCC_END) all arrived.
 *   That pattern means only the worker threads were dead; main-thread
 *   BLE worked. Raising the DSP stack changed nothing because the
 *   thread never reached the FFT.
 *
 *   The previous *working* firmware (v5.9/v5.10) did all heavy work in
 *   the main thread and streamed reliably. v8.1 returns to that proven
 *   model: the SAADC ISR stages half-buffers; the MAIN thread drains
 *   them, sends audio over BLE inline, and feeds the MFCC pipeline.
 *   No worker threads, no audio ring, no frame queue.
 *
 * Second bug also fixed:
 *   v8.0's 16-slot MFCC queue would have dropped ~649 of 665 frames
 *   (frames are produced during capture but consumed only afterwards).
 *   v8.1 stores every frame in a full-size buffer indexed by frame
 *   number — zero drops — then streams them after "finished".
 *
 * Reliability fixes carried over from v8.0:
 *   FIX A  gain = GAIN4 (full-scale ~206 mV) for a quiet MEMS mic.
 *   FIX B  SAADC configured ONCE at boot; start/abort per recording
 *          (never uninit/re-init) → no BLE-starving nrfx churn.
 *          Supervision timeout 2 s.
 *   FIX C  every per-recording state reset at the start of each REC.
 *
 * Parity note (unchanged): the MFCC pipeline is fed RAW samples (only
 *   the bandpass removes DC), matching the search/training pipeline.
 *   The DC-removal IIR is applied ONLY to the copy streamed for
 *   listening, never to the MFCC input.
 *
 * Host protocol (unchanged — receiver_v80.py works as-is):
 *   "START:<bytes>:<ORGAN>\n"
 *   audio packets : [u16 seq LE][u16 len LE][pcm int16 LE ...]
 *   "finished\n"
 *   "MFCC_START:<n_frames>:<n_mfcc>\n"
 *   mfcc packets  : [u16 idx LE][u16 payload_bytes LE][float32 ...]
 *   "MFCC_END\n"
 * ==================================================================
 */

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
#include <zephyr/sys/byteorder.h>
#include <zephyr/sys/util.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>

#include <nrfx.h>
#include <drivers/nrfx_errors.h>
#include <nrfx_saadc.h>

#include "dsp_mfcc.h"
#include "heart_mfcc_config.h"
#include "lung_mfcc_config.h"

LOG_MODULE_REGISTER(AcoustEEEcare);

/* ══════════════════════════════════════════════════════════════════
 * ORGAN SELECTION
 * ══════════════════════════════════════════════════════════════════ */
typedef enum { ORGAN_HEART = 0, ORGAN_LUNG = 1 } organ_t;
static organ_t active_organ = ORGAN_HEART;

static dsp_mfcc_pipeline_t s_pipeline;

/* caller-allocated MFCC window; sized for the larger config (lung=1200) */
#define MAX_WINDOW_SAMPLES  1200
static int16_t s_window[MAX_WINDOW_SAMPLES];

static inline const dsp_mfcc_config_t *active_config(void)
{
    return (active_organ == ORGAN_LUNG) ? &lung_mfcc_config
                                        : &heart_mfcc_config;
}

/* ══════════════════════════════════════════════════════════════════
 * SAADC CONFIG  (FIX A: GAIN4)
 * ══════════════════════════════════════════════════════════════════ */
#define SAADC_CC_VALUE      2000U   /* 16 MHz / 2000 = 8 kHz */
#define SAADC_IRQ_PRIORITY  6

static const nrfx_saadc_channel_t saadc_channel_cfg = {
    .channel_config = {
        .resistor_p = NRF_SAADC_RESISTOR_DISABLED,
        .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
        .gain       = NRF_SAADC_GAIN2,              /* FIX A: was GAIN1_4 */
        .reference  = NRF_SAADC_REFERENCE_VDD4,
        .acq_time   = NRF_SAADC_ACQTIME_10US,
        .mode       = NRF_SAADC_MODE_SINGLE_ENDED,
        .burst      = NRF_SAADC_BURST_DISABLED,
    },
    .pin_p         = NRF_SAADC_INPUT_AIN0,
    .pin_n         = NRF_SAADC_INPUT_DISABLED,
    .channel_index = 0,
};

/* DC removal for the LISTENING path only (NOT fed to MFCC) */
static int32_t dc_estimate = 2048;

/* ══════════════════════════════════════════════════════════════════
 * AUDIO PARAMS
 * ══════════════════════════════════════════════════════════════════ */
#define SAMPLING_RATE        8000
#define DURATION_S           10
#define TOTAL_AUDIO_SAMPLES  (SAMPLING_RATE * DURATION_S)            /* 80 000 */
#define TOTAL_AUDIO_BYTES    (TOTAL_AUDIO_SAMPLES * sizeof(int16_t)) /* 160 000 */

#define HALF_BUF_SAMPLES     512
#define HALF_BUF_BYTES       (HALF_BUF_SAMPLES * sizeof(int16_t))

static int16_t ping_pong[2][HALF_BUF_SAMPLES];
static volatile uint8_t  next_dma_buf     = 1;
static volatile bool     analog_recording = false;
static uint16_t          nus_chunk_size   = 244;

/* ══════════════════════════════════════════════════════════════════
 * ISR -> MAIN STAGING RING
 *   The ISR copies each completed half-buffer into one of STAGE_SLOTS
 *   staging buffers and gives stage_sem. The MAIN thread drains them.
 *   8 slots = ~512 ms of slack, plenty to cover BLE send + FFT time.
 * ══════════════════════════════════════════════════════════════════ */
#define STAGE_SLOTS   8
static int16_t  stage_buf[STAGE_SLOTS][HALF_BUF_SAMPLES];
static uint16_t stage_len[STAGE_SLOTS];
static volatile uint8_t stage_wr;
static volatile uint8_t stage_rd;
static K_SEM_DEFINE(stage_sem, 0, STAGE_SLOTS);
static volatile uint32_t stage_overruns;

/* ══════════════════════════════════════════════════════════════════
 * MFCC FRAME STORE
 *   Every frame is stored by index during capture (no queue, no drops),
 *   then streamed after "finished". Sized for the largest config.
 * ══════════════════════════════════════════════════════════════════ */
#define MFCC_MAX_FRAMES  665   /* heart = 665, lung = 324 */
#define MFCC_MAX_COEFFS  26    /* lung = 26, heart = 25 */
static float        mfcc_store[MFCC_MAX_FRAMES][MFCC_MAX_COEFFS];
static volatile int mfcc_count;

#define CHUNK_HEADER_BYTES 4
static uint16_t tx_seq = 0;

/* ══════════════════════════════════════════════════════════════════
 * SAADC EVENT HANDLER (ISR) — copy + signal only
 * ══════════════════════════════════════════════════════════════════ */
static void saadc_event_handler(nrfx_saadc_evt_t const *p_event)
{
    switch (p_event->type) {

    case NRFX_SAADC_EVT_BUF_REQ:
        (void)nrfx_saadc_buffer_set(ping_pong[next_dma_buf], HALF_BUF_SAMPLES);
        next_dma_buf ^= 1;
        break;

    case NRFX_SAADC_EVT_DONE: {
        int16_t *filled = p_event->data.done.p_buffer;
        uint8_t wr  = stage_wr;
        uint8_t nxt = (wr + 1) % STAGE_SLOTS;
        if (nxt == stage_rd) { stage_overruns++; break; }
        memcpy(stage_buf[wr], filled, HALF_BUF_BYTES);
        stage_len[wr] = HALF_BUF_SAMPLES;
        stage_wr = nxt;
        k_sem_give(&stage_sem);
        break;
    }

    default:
        break;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * SAADC init once / start / stop  (FIX B)
 * ══════════════════════════════════════════════════════════════════ */
static int saadc_init_once(void)
{
    nrfx_err_t err;
    IRQ_CONNECT(SAADC_IRQn, SAADC_IRQ_PRIORITY, nrfx_saadc_irq_handler, NULL, 0);
    irq_enable(SAADC_IRQn);

    err = nrfx_saadc_init(SAADC_IRQ_PRIORITY);
    if (err != 0) { LOG_ERR("saadc_init: 0x%08X", err); return -EIO; }

    err = nrfx_saadc_channel_config(&saadc_channel_cfg);
    if (err != 0) { LOG_ERR("channel_config: 0x%08X", err); return -EIO; }

    nrfx_saadc_adv_config_t adv = NRFX_SAADC_DEFAULT_ADV_CONFIG;
    adv.oversampling      = NRF_SAADC_OVERSAMPLE_DISABLED;
    adv.burst             = NRF_SAADC_BURST_DISABLED;
    adv.internal_timer_cc = SAADC_CC_VALUE;
    adv.start_on_end      = true;

    err = nrfx_saadc_advanced_mode_set(BIT(saadc_channel_cfg.channel_index),
                                        NRF_SAADC_RESOLUTION_12BIT,
                                        &adv, saadc_event_handler);
    if (err != 0) { LOG_ERR("advanced_mode_set: 0x%08X", err); return -EIO; }

    LOG_INF("SAADC configured once @ %d Hz (GAIN4)", SAMPLING_RATE);
    return 0;
}

static int saadc_start(void)
{
    nrfx_err_t err;
    next_dma_buf = 1;
    err = nrfx_saadc_buffer_set(ping_pong[0], HALF_BUF_SAMPLES);
    if (err != 0) { LOG_ERR("buffer_set: 0x%08X", err); return -EIO; }
    err = nrfx_saadc_mode_trigger();
    if (err != 0) { LOG_ERR("mode_trigger: 0x%08X", err); return -EIO; }
    return 0;
}

static void saadc_stop(void) { nrfx_saadc_abort(); }

/* ══════════════════════════════════════════════════════════════════
 * LEDs
 * ══════════════════════════════════════════════════════════════════ */
static const struct gpio_dt_spec red_led   = GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);
static const struct gpio_dt_spec green_led = GPIO_DT_SPEC_GET(DT_ALIAS(led1), gpios);
static const struct gpio_dt_spec blue_led  = GPIO_DT_SPEC_GET(DT_ALIAS(led2), gpios);
static inline void led_on (const struct gpio_dt_spec *l) { gpio_pin_set_dt(l, 1); }
static inline void led_off(const struct gpio_dt_spec *l) { gpio_pin_set_dt(l, 0); }
static inline void led_red(void)   { led_on(&red_led);  led_off(&green_led); led_off(&blue_led); }
static inline void led_green(void) { led_off(&red_led); led_on(&green_led);  led_off(&blue_led); }
static inline void led_cyan(void)  { led_off(&red_led); led_on(&green_led);  led_on(&blue_led);  }
static inline void led_purple(void){ led_on(&red_led);  led_off(&green_led); led_on(&blue_led);  }
static inline void led_white(void) { led_on(&red_led);  led_on(&green_led);  led_on(&blue_led);  }

/* ══════════════════════════════════════════════════════════════════
 * FLAGS / BLE
 * ══════════════════════════════════════════════════════════════════ */
static volatile bool start_recording = false;
static volatile bool is_connected    = false;
static volatile bool mtu_exchanged   = false;

#define DEVICE_NAME      "AcoustEEEcare"
#define DEVICE_NAME_LEN  (sizeof(DEVICE_NAME) - 1)

static const struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
    BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),
};
static const struct bt_data sd_adv[] = {
    BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_SRV_VAL),
};

static struct k_work_delayable adv_restart_work;

static void mtu_exchange_cb(struct bt_conn *conn, uint8_t err,
                             struct bt_gatt_exchange_params *params)
{
    if (!err) {
        uint16_t mtu = bt_gatt_get_mtu(conn);
        nus_chunk_size = mtu - 3;
        LOG_INF("MTU=%d NUS chunk=%d", mtu, nus_chunk_size);
    }
    mtu_exchanged = true;
}
static struct bt_gatt_exchange_params exchange_params = { .func = mtu_exchange_cb };

static void adv_restart_work_handler(struct k_work *work)
{
    int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                               sd_adv, ARRAY_SIZE(sd_adv));
    if (err) LOG_ERR("adv restart: %d", err);
}

static void connected(struct bt_conn *conn, uint8_t err)
{
    if (err) { LOG_WRN("conn failed (%u)", err); return; }
    is_connected  = true;
    mtu_exchanged = false;
    led_green();

    static const struct bt_conn_le_phy_param phy = {
        .options     = BT_CONN_LE_PHY_OPT_NONE,
        .pref_tx_phy = BT_GAP_LE_PHY_2M,
        .pref_rx_phy = BT_GAP_LE_PHY_2M,
    };
    bt_conn_le_phy_update(conn, &phy);

    static const struct bt_le_conn_param cp = {
        .interval_min = 12, .interval_max = 24, .latency = 0, .timeout = 200,
    };
    bt_conn_le_param_update(conn, &cp);
    bt_gatt_exchange_mtu(conn, &exchange_params);
    LOG_INF("Connected");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
    if (analog_recording) saadc_stop();
    analog_recording = false;
    is_connected     = false;
    start_recording  = false;
    mtu_exchanged    = false;
    nus_chunk_size   = 244;
    led_red();
    LOG_WRN("Disconnected (reason=%d)", reason);
    k_work_schedule(&adv_restart_work, K_MSEC(500));
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
    .connected = connected, .disconnected = disconnected,
};

static void received(struct bt_conn *conn, const void *data, uint16_t len, void *ctx)
{
    ARG_UNUSED(conn); ARG_UNUSED(ctx);
    if (len == 3 && memcmp(data, "REC", 3) == 0) {
        start_recording = true;
    } else if (len == 5 && memcmp(data, "HEART", 5) == 0) {
        if (!analog_recording) { active_organ = ORGAN_HEART; LOG_INF("organ=HEART"); }
    } else if (len == 4 && memcmp(data, "LUNG", 4) == 0) {
        if (!analog_recording) { active_organ = ORGAN_LUNG;  LOG_INF("organ=LUNG"); }
    }
}
static struct bt_nus_cb nus_listener = { .received = received };

/* ══════════════════════════════════════════════════════════════════
 * BLE send helper (main-thread, blocking on backpressure)
 * ══════════════════════════════════════════════════════════════════ */
static void ble_send_blocking(const void *buf, uint16_t len)
{
    int err;
    do {
        err = bt_nus_send(NULL, buf, len);
        if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(2));
    } while (err == -ENOMEM || err == -EAGAIN);
}

/* ══════════════════════════════════════════════════════════════════
 * MFCC FRAME CALLBACK (runs in MAIN thread, inside feed_chunk)
 * Store by index — no queue, no drops.
 * ══════════════════════════════════════════════════════════════════ */
static void mfcc_frame_cb(int idx, const float *coeffs, void *user)
{
    ARG_UNUSED(user);
    if (idx < 0 || idx >= MFCC_MAX_FRAMES) return;
    int nc = active_config()->n_mfcc;
    if (nc > MFCC_MAX_COEFFS) nc = MFCC_MAX_COEFFS;
    memcpy(mfcc_store[idx], coeffs, nc * sizeof(float));
    if (idx + 1 > mfcc_count) mfcc_count = idx + 1;
}

/* ══════════════════════════════════════════════════════════════════
 * Send one block of PCM as audio packets (DC-removed for listening)
 * ══════════════════════════════════════════════════════════════════ */
static int16_t dc_copy[HALF_BUF_SAMPLES];

static void send_audio_block(const int16_t *raw, uint32_t n)
{
    /* DC-remove a copy (listening path only) */
    for (uint32_t i = 0; i < n; i++) {
        dc_estimate += ((int32_t)raw[i] - dc_estimate) >> 8;
        dc_copy[i]   = (int16_t)((int32_t)raw[i] - dc_estimate);
    }

    const uint8_t *p   = (const uint8_t *)dc_copy;
    uint32_t bytes     = n * sizeof(int16_t);
    uint32_t offset    = 0;
    uint8_t  chunk[251];

    while (bytes > 0) {
        uint16_t payload = nus_chunk_size - CHUNK_HEADER_BYTES;
        uint16_t send    = (bytes < payload) ? (uint16_t)bytes : payload;
        sys_put_le16(tx_seq, &chunk[0]);
        sys_put_le16(send,   &chunk[2]);
        memcpy(&chunk[4], p + offset, send);
        ble_send_blocking(chunk, CHUNK_HEADER_BYTES + send);
        tx_seq  = (tx_seq + 1) & 0xFFFF;
        offset += send;
        bytes  -= send;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * Stream all stored MFCC frames
 * ══════════════════════════════════════════════════════════════════ */
static void stream_mfcc(int n_frames_expected)
{
    const dsp_mfcc_config_t *cfg = active_config();

    char hdr[48];
    int hl = snprintf(hdr, sizeof(hdr), "MFCC_START:%d:%d\n",
                      n_frames_expected, cfg->n_mfcc);
    ble_send_blocking(hdr, hl);
    k_sleep(K_MSEC(20));

    uint16_t payload = cfg->n_mfcc * sizeof(float);
    uint8_t  pkt[CHUNK_HEADER_BYTES + MFCC_MAX_COEFFS * sizeof(float)];

    int sent = 0;
    for (int idx = 0; idx < n_frames_expected; idx++) {
        sys_put_le16((uint16_t)idx, &pkt[0]);
        sys_put_le16(payload,       &pkt[2]);
        memcpy(&pkt[4], mfcc_store[idx], payload);
        ble_send_blocking(pkt, CHUNK_HEADER_BYTES + payload);
        sent++;
    }

    k_sleep(K_MSEC(20));
    ble_send_blocking("MFCC_END\n", 9);
    LOG_INF("MFCC stream done: %d/%d frames (stored=%d)",
            sent, n_frames_expected, mfcc_count);
}

/* ══════════════════════════════════════════════════════════════════
 * RECORD AND STREAM (everything in the MAIN thread)
 * ══════════════════════════════════════════════════════════════════ */
static void record_and_stream(void)
{
    const dsp_mfcc_config_t *cfg = active_config();

    /* FIX C: reset all per-recording state */
    dc_estimate    = 2048;
    tx_seq         = 0;
    stage_wr = stage_rd = 0;
    stage_overruns = 0;
    mfcc_count     = 0;
    k_sem_reset(&stage_sem);

    if (dsp_mfcc_init(&s_pipeline, cfg) != 0) {
        LOG_ERR("dsp_mfcc_init failed");
        ble_send_blocking("ERR:DSP", 7);
        return;
    }
    s_pipeline.window = s_window;
    dsp_mfcc_reset(&s_pipeline);
    dsp_mfcc_set_callback(&s_pipeline, mfcc_frame_cb, NULL);

    char hdr[40];
    snprintf(hdr, sizeof(hdr), "START:%u:%s\n", (uint32_t)TOTAL_AUDIO_BYTES,
             active_organ == ORGAN_LUNG ? "LUNG" : "HEART");
    ble_send_blocking(hdr, strlen(hdr));
    k_sleep(K_MSEC(10));

    led_cyan();
    analog_recording = true;
    if (saadc_start() != 0) {
        analog_recording = false;
        ble_send_blocking("ERR:SAADC", 9);
        return;
    }

    LOG_INF("Recording %d s @ %d Hz (organ=%s)", DURATION_S, SAMPLING_RATE,
            active_organ == ORGAN_LUNG ? "LUNG" : "HEART");

    /* Drain staged buffers in the main thread until we've handled
     * exactly TOTAL_AUDIO_SAMPLES samples (sample-accurate: the last
     * partial chunk sends only its remaining bytes, and feeds the MFCC
     * pipeline exactly 80 000 samples so it emits exactly n_frames). */
    uint32_t samples_done = 0;
    bool     ok = true;

    while (samples_done < TOTAL_AUDIO_SAMPLES) {
        if (k_sem_take(&stage_sem, K_MSEC(500)) != 0) {
            LOG_ERR("staging timeout at %u/%u samples (overruns=%u)",
                    samples_done, (uint32_t)TOTAL_AUDIO_SAMPLES, stage_overruns);
            ok = false;
            break;
        }
        uint8_t  rd     = stage_rd;
        int16_t *raw    = stage_buf[rd];
        uint32_t avail  = stage_len[rd];
        uint32_t remain = TOTAL_AUDIO_SAMPLES - samples_done;
        uint32_t use    = (avail < remain) ? avail : remain;

        /* MFCC path — RAW samples (matches training) */
        dsp_mfcc_feed_chunk(&s_pipeline, raw, (int)use);

        /* Listening path — DC-removed audio over BLE */
        send_audio_block(raw, use);

        samples_done += use;
        stage_rd = (rd + 1) % STAGE_SLOTS;
    }

    saadc_stop();
    analog_recording = false;

    ble_send_blocking("finished\n", 9);
    LOG_INF("Audio done: %u samples, overruns=%u, mfcc_frames=%d",
            samples_done, stage_overruns, mfcc_count);

    if (!ok) {
        led_red();
        return;
    }

    /* MFCC: every frame was stored by index during capture. */
    led_purple();
    stream_mfcc(cfg->n_frames_expected);

    led_green();
    LOG_INF("record_and_stream complete.");
}

/* ══════════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════════ */
int main(void)
{
    gpio_pin_configure_dt(&red_led,   GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&green_led, GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&blue_led,  GPIO_OUTPUT_ACTIVE);
    led_white();

    k_work_init_delayable(&adv_restart_work, adv_restart_work_handler);

    if (saadc_init_once() != 0) { LOG_ERR("SAADC init failed"); return -1; }

    int err = bt_enable(NULL);
    if (err) { LOG_ERR("bt_enable: %d", err); return err; }

    err = bt_nus_cb_register(&nus_listener, NULL);
    if (err) { LOG_ERR("nus_cb_register: %d", err); return err; }

    err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
                          sd_adv, ARRAY_SIZE(sd_adv));
    if (err) { LOG_ERR("adv_start: %d", err); return err; }

    led_red();
    LOG_INF("AcoustEEEcare v8.1 ready — BLE-to-MFCC (single-threaded)");

    while (true) {
        k_sleep(K_MSEC(100));
        if (is_connected && mtu_exchanged && start_recording && !analog_recording) {
            start_recording = false;
            record_and_stream();
        }
    }
    return 0;
}

/*
 * ==================================================================
 * NEXT STEP — TFLITE MICRO INFERENCE (interface reference)
 *   HEART  in 'mfcc_input' int8 [1,665,25,1] scale=0.39602566 zp=38
 *          out 'rr_output' int8 [1,1]         scale=1.2597337  zp=-128
 *   LUNG   in 'mfcc_input' int8 [1,324,26,1] scale=0.33742353 zp=28
 *          out 'rr_output' int8 [1,1]         scale=0.1256319  zp=-128
 *   Ops: CONV_2D, RESHAPE, PAD, FULLY_CONNECTED.
 *   Quantize:  q = round(mfcc / scale) + zp, clamp [-128,127]
 *   Dequant :  rr = (out - zp) * scale
 *   Input layout frame-major: mfcc_store[idx][coeff] maps directly.
 * ==================================================================
 */