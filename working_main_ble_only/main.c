/*
 * AcoustEEEcare — SAADC BLE-only, Ping-Pong Stream Edition
 * ============================================================
 * v5.9 — Heart/Lung auscultation fixes
 *
 * FIXES IN THIS VERSION (v5.9):
 *
 *   Fix 1 (CRITICAL): Sample rate corrected — CC changed from 500 to 2000,
 *     oversampling and burst disabled.
 *     With NRF_SAADC_OVERSAMPLE_4X + NRF_SAADC_BURST_ENABLED + CC=500, the
 *     hardware's burst/oversample interaction doubled the effective sample
 *     period, producing only ~4000 unique samples/s instead of 8000.
 *     Symptom: 1 kHz diagnostic tone detected as 500 Hz; voice sounded
 *     slowed down / pitch-shifted an octave down.
 *     Fix: disable oversampling and burst; set CC=2000 (16 MHz / 8000 = 2000).
 *
 *   Fix 2: DC removal filter time constant slowed from >>5 to >>8.
 *     α = 1/32 (>>5) gives a ~40 Hz high-pass corner, attenuating heart
 *     sound fundamentals (S1, S2 at 20–150 Hz).
 *     α = 1/256 (>>8) moves the corner to ~5 Hz, preserving all heart
 *     and lung content while still removing slow DC drift.
 *
 *   Fix 3: HALF_BUF_SAMPLES doubled from 256 to 512.
 *     Doubles the main-thread deadline from 32 ms to 64 ms, preventing
 *     DMA overruns caused by BLE radio events stalling the main thread.
 *
 *   Fix 4: CONFIG_MAIN_THREAD_PRIORITY=5 (set in prj.conf — see below).
 *     Raises main thread above the BLE host work queue so it can preempt
 *     BLE work to drain half-buffers in time.
 *
 *   Note Fix 5 (gain) and Fix 6 (optional bandpass) from the fixes doc
 *   are hardware-dependent and left as comments/stubs for the operator.
 *
 * FIXES PRESERVED FROM v5.8:
 *
 *   Fix 1: Gain reduced from NRF_SAADC_GAIN1 to NRF_SAADC_GAIN1_2.
 *     With an external amplifier, GAIN1 caused hard clipping at +4095.
 *     Asymmetric clipping made the FFT report 500 Hz instead of 1000 Hz.
 *     GAIN1_2 doubles the input range to ~1.65 V FS.
 *     Drop to GAIN1_4 if still clipping.
 *
 *   Fix 2: Diagnostic tone phase now computed from tone_half_count.
 *     IRQ-order-independent phase, eliminating click/stutter at buffer joins.
 *
 * FIXES PRESERVED FROM v5.7:
 *
 *   Fix 1: DC removal filter not run in DIAGNOSTIC_TONE mode.
 *   Fix 2: tone_offset (now tone_half_count) moved to file scope, reset each REC.
 *   Fix 3: DC removal gated by #define ENABLE_DC_REMOVAL.
 *
 * FIXES PRESERVED FROM v5.6:
 *
 *   Fix 1: Gain NRF_SAADC_GAIN1_4 → NRF_SAADC_GAIN1.
 *   Fix 2: Reference NRF_SAADC_REFERENCE_INTERNAL → NRF_SAADC_REFERENCE_VDD4.
 *   Fix 3: DC removal IIR high-pass filter in EVT_DONE.
 *
 * FIXES PRESERVED FROM v5.5:
 *
 *   Fix 1 (CRITICAL): Handle NRFX_SAADC_EVT_BUF_REQ for continuous streaming.
 *
 * FIXES PRESERVED FROM v5.4:
 *
 *   Fix 2 (CRITICAL): saadc_init() calls nrfx_saadc_uninit() first.
 *   Fix 3 (CRITICAL): All nrfx error checks use err != 0.
 *   Fix 4: IRQ priority hardcoded to 6.
 *   Fix 5: disconnected() guards saadc_stop_streaming() with analog_recording.
 *   Fix 6: Buffer identity from p_buffer pointer comparison.
 *
 * prj.conf addition required for Fix 4 (main thread priority):
 *   CONFIG_MAIN_THREAD_PRIORITY=5
 *
 * See CHANGELOG at bottom for full version history.
 * ============================================================
 */

#include <stdint.h>
#include <zephyr/bluetooth/services/nus.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/drivers/gpio.h>
#include <nrfx.h>
#include <drivers/nrfx_errors.h>
#include <nrfx_saadc.h>
#include <zephyr/irq.h>
#include <zephyr/logging/log.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include "zephyr/kernel/thread_stack.h"
#include "zephyr/sys/time_units.h"

#include <string.h>
#include <stdio.h>
#include <math.h>

LOG_MODULE_REGISTER(AcoustEEEcare);

/* ══════════════════════════════════════════════════════════════
 * SAADC CONFIG
 *
 * v5.9 nrfx_saadc config:
 * Channel:      AIN0 / P0.02
 * Resolution:   12-bit
 * Gain:         1/2  (FS = ~1.65 V with VDD4 ref + external amp)
 * Reference:    VDD/4 (~0.825 V)
 * Acquisition:  10 us (low-Z op-amp output)
 * Oversample:   DISABLED  ← v5.9: was 4X (caused half-rate output)
 * Burst:        DISABLED  ← v5.9: was ENABLED (caused half-rate output)
 * Sample timer: Internal CC=2000 → 8 kHz output  ← v5.9: was CC=500
 * DMA:          Double-buffered ping-pong
 *
 * Fix #1 math:
 *   16,000,000 Hz (internal timer) / 2000 (CC) = 8000 Hz exactly.
 *   No oversampling, no burst — one hardware sample per CC tick.
 *
 * NOTE: AC coupling (100 nF series cap + 100 kΩ pull-down on AIN0)
 * is strongly recommended. For auscultation use 470 nF / 100 kΩ for a
 * ~3.4 Hz hardware HPF corner that preserves all heart sound content.
 * ══════════════════════════════════════════════════════════════ */

/* v5.9 FIX 1: CC=2000 for true 8 kHz with no oversampling.
 * Was 500U — that value was for 4× oversampled mode but even then
 * the burst+oversample combination doubled the effective period. */
#define SAADC_CC_VALUE      2000U

/* FIX (v5.4): Priority 6 — nrfx default 7 exceeds Zephyr+BLE limit */
#define SAADC_IRQ_PRIORITY  6

static const nrfx_saadc_channel_t saadc_channel_cfg = {
    .channel_config = {
        .resistor_p = NRF_SAADC_RESISTOR_DISABLED,
        .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
        .gain       = NRF_SAADC_GAIN1_4,           /* v5.8: was GAIN1 — clipping with external amp.
                                                     * GAIN1_2 doubles the input range (~1.65 V FS
                                                     * with VDD4 ref).  Drop to GAIN1_4 if still
                                                     * clipping (amp output swing too large).
                                                     * For chest auscultation with small mic signal,
                                                     * try GAIN2 or GAIN4 per Fix #3 in fixes doc. */
        .reference  = NRF_SAADC_REFERENCE_VDD4,    /* v5.6: was NRF_SAADC_REFERENCE_INTERNAL */
        .acq_time   = NRF_SAADC_ACQTIME_10US,
        .mode       = NRF_SAADC_MODE_SINGLE_ENDED,
        .burst      = NRF_SAADC_BURST_DISABLED,     /* v5.9 FIX 1: was NRF_SAADC_BURST_ENABLED —
                                                     * burst+oversample caused half-rate output */
    },
    .pin_p         = NRF_SAADC_INPUT_AIN0,
    .pin_n         = NRF_SAADC_INPUT_DISABLED,
    .channel_index = 0,
};

static volatile uint32_t saadc_dma_overruns = 0;

/* v5.6: File-scope DC estimate for IIR high-pass filter in EVT_DONE.
 * v5.7: Gated by ENABLE_DC_REMOVAL.
 * v5.9: Filter shift changed from >>5 to >>8 (corner ~5 Hz) to preserve
 * heart sound fundamentals (S1, S2 at 20–150 Hz). Enable this define if
 * your mic hardware has no AC coupling / you see DC offset. */

#define ENABLE_DC_REMOVAL 
static int32_t dc_estimate = 0;

/* ══════════════════════════════════════════════════════════════
 * AUDIO PARAMS
 * ══════════════════════════════════════════════════════════════ */
#define SAMPLING_RATE        8000
#define SAMPLE_PERIOD_US     125
#define DURATION_S           10
#define TOTAL_AUDIO_SAMPLES  (SAMPLING_RATE * DURATION_S)
#define TOTAL_AUDIO_BYTES    (TOTAL_AUDIO_SAMPLES * sizeof(int16_t))

/* v5.9 FIX 3: Doubled from 256 to 512.
 * Doubles main-thread DMA drain deadline from 32 ms to 64 ms,
 * preventing overruns caused by BLE radio events stalling main thread.
 * RAM cost: 2 × 512 × 2 = 2 KB total — acceptable on nRF52840. */
#define HALF_BUF_SAMPLES     512
#define HALF_BUF_BYTES       (HALF_BUF_SAMPLES * sizeof(int16_t))

static int16_t ping_pong[2][HALF_BUF_SAMPLES];

static K_SEM_DEFINE(half_ready_sem, 0, 1);
static volatile uint8_t  ready_half       = 0;
static volatile bool     analog_recording = false;
static uint16_t          nus_chunk_size   = 244;

/* v5.5: Index of the next buffer to hand to the driver on EVT_BUF_REQ. */
static volatile uint8_t  next_dma_buf     = 1;

/* #define DIAGNOSTIC_TONE */

#ifdef DIAGNOSTIC_TONE
#define DIAG_TONE_HZ    1000
#define DIAG_AMPLITUDE  2000
/* v5.8: tone_half_count — deterministic phase, reset each recording. */
static uint32_t tone_half_count = 0;
static void inject_diagnostic_tone(int16_t *buf, uint32_t start_sample)
{
    for (uint32_t i = 0; i < HALF_BUF_SAMPLES; i++) {
        float phase = 2.0f * 3.14159265f * DIAG_TONE_HZ *
                      (float)(start_sample + i) / (float)SAMPLING_RATE;
        buf[i] = (int16_t)(DIAG_AMPLITUDE * sinf(phase));
    }
}
#endif

static void log_buffer_stats(const int16_t *buf, uint8_t half_idx)
{
    int32_t sum = 0;
    int16_t mn = buf[0], mx = buf[0];
    for (uint32_t i = 0; i < HALF_BUF_SAMPLES; i++) {
        if (buf[i] < mn) mn = buf[i];
        if (buf[i] > mx) mx = buf[i];
        sum += buf[i];
    }
    LOG_DBG("half[%u]: min=%d max=%d mean=%d",
            half_idx, mn, mx, (int16_t)(sum / (int32_t)HALF_BUF_SAMPLES));
}

/* ══════════════════════════════════════════════════════════════
 * SAADC EVENT HANDLER (IRQ context)
 * ══════════════════════════════════════════════════════════════ */
static void saadc_event_handler(nrfx_saadc_evt_t const *p_event)
{
    switch (p_event->type) {

    case NRFX_SAADC_EVT_BUF_REQ:
        (void)nrfx_saadc_buffer_set(ping_pong[next_dma_buf],
                                    HALF_BUF_SAMPLES);
        next_dma_buf ^= 1;
        break;

    case NRFX_SAADC_EVT_DONE: {
        int16_t *filled_buf  = p_event->data.done.p_buffer;
        uint8_t  filled_half = (filled_buf == ping_pong[0]) ? 0 : 1;

#ifdef DIAGNOSTIC_TONE
        /* v5.8: Deterministic phase from tone_half_count × HALF_BUF_SAMPLES. */
        inject_diagnostic_tone(filled_buf, tone_half_count * HALF_BUF_SAMPLES);
        tone_half_count++;
#else
        /* Real mic path — DC removal only runs on actual ADC samples.
         * v5.7: gated by ENABLE_DC_REMOVAL.
         * v5.9 FIX 2: shift changed from >>5 (~40 Hz corner) to >>8 (~5 Hz
         * corner) so heart sound fundamentals at 20–150 Hz are preserved. */
#ifdef ENABLE_DC_REMOVAL
        for (uint32_t i = 0; i < HALF_BUF_SAMPLES; i++) {
            dc_estimate += ((int32_t)filled_buf[i] - dc_estimate) >> 8; /* v5.9: was >>5 */
            filled_buf[i] = (int16_t)((int32_t)filled_buf[i] - dc_estimate);
        }
#endif /* ENABLE_DC_REMOVAL */
#endif /* DIAGNOSTIC_TONE */

        /* ── Optional Fix #6: Software bandpass (20–1000 Hz biquad) ──────
         * Uncomment and implement biquad_apply() with scipy-generated Q15
         * coefficients when moving to chest auscultation recordings.
         * Apply AFTER the DC removal block above.
         *
         * for (uint32_t i = 0; i < HALF_BUF_SAMPLES; i++) {
         *     int16_t s = filled_buf[i];
         *     s = biquad_apply(s, 0);
         *     s = biquad_apply(s, 1);
         *     filled_buf[i] = s;
         * }
         * ─────────────────────────────────────────────────────────────── */

        if (k_sem_count_get(&half_ready_sem) > 0) {
            saadc_dma_overruns++;
            LOG_WRN("DMA overrun #%u — main thread too slow",
                    saadc_dma_overruns);
        }

        ready_half = filled_half;
        k_sem_give(&half_ready_sem);
        break;
    }

    default:
        break;
    }
}

/* ══════════════════════════════════════════════════════════════
 * saadc_init()
 * ══════════════════════════════════════════════════════════════ */
static int saadc_init(void)
{
    nrfx_err_t err;

    nrfx_saadc_uninit();

    IRQ_CONNECT(SAADC_IRQn, SAADC_IRQ_PRIORITY, nrfx_saadc_irq_handler, NULL, 0);
    irq_enable(SAADC_IRQn);
    LOG_INF("IRQ connected");

    err = nrfx_saadc_init(SAADC_IRQ_PRIORITY);
    LOG_INF("nrfx_saadc_init returned: 0x%08X (NRFX_SUCCESS=0x%08X)",
            err, NRFX_SUCCESS);
    if (err != 0) {
        LOG_ERR("STEP 1 FAILED (nrfx_saadc_init)");
        return -EIO;
    }

    err = nrfx_saadc_channel_config(&saadc_channel_cfg);
    LOG_INF("nrfx_saadc_channel_config returned: 0x%08X", err);
    if (err != 0) {
        LOG_ERR("STEP 2 FAILED (channel_config)");
        return -EIO;
    }

    nrfx_saadc_adv_config_t adv_cfg = NRFX_SAADC_DEFAULT_ADV_CONFIG;
    adv_cfg.oversampling      = NRF_SAADC_OVERSAMPLE_DISABLED; /* v5.9 FIX 1: was OVERSAMPLE_4X */
    adv_cfg.burst             = NRF_SAADC_BURST_DISABLED;      /* v5.9 FIX 1: was BURST_ENABLED */
    adv_cfg.internal_timer_cc = SAADC_CC_VALUE;                /* v5.9 FIX 1: 2000 (was 500) */
    adv_cfg.start_on_end      = true;

    uint32_t channel_mask = BIT(saadc_channel_cfg.channel_index);
    err = nrfx_saadc_advanced_mode_set(channel_mask,
                                        NRF_SAADC_RESOLUTION_12BIT,
                                        &adv_cfg,
                                        saadc_event_handler);
    LOG_INF("nrfx_saadc_advanced_mode_set returned: 0x%08X", err);
    if (err != 0) {
        LOG_ERR("STEP 3 FAILED (advanced_mode_set)");
        return -EIO;
    }

    LOG_INF("nrfx_saadc ready");
    return 0;
}

/* ══════════════════════════════════════════════════════════════
 * saadc_start_streaming()
 * ══════════════════════════════════════════════════════════════ */
static int saadc_start_streaming(void)
{
    nrfx_err_t err;

    saadc_dma_overruns = 0;
    dc_estimate        = 2048;
#ifdef DIAGNOSTIC_TONE
    tone_half_count    = 0;
#endif

    /* v5.9 FIX 6 stub: reset bandpass filter state if using Fix #6.
     * Uncomment when biquad filter is implemented:
     * memset(bp_x1, 0, sizeof(bp_x1));
     * memset(bp_x2, 0, sizeof(bp_x2));
     * memset(bp_y1, 0, sizeof(bp_y1));
     * memset(bp_y2, 0, sizeof(bp_y2)); */

    if (saadc_init() != 0) {
        LOG_ERR("saadc_start_streaming: init failed");
        return -EIO;
    }

    next_dma_buf = 1;

    err = nrfx_saadc_buffer_set(ping_pong[0], HALF_BUF_SAMPLES);
    if (err != 0) {
        LOG_ERR("nrfx_saadc_buffer_set[0] failed: 0x%08X", err);
        return -EIO;
    }

    err = nrfx_saadc_mode_trigger();
    if (err != 0) {
        LOG_ERR("nrfx_saadc_mode_trigger failed: 0x%08X", err);
        return -EIO;
    }

    LOG_INF("SAADC DMA streaming started");
    return 0;
}

static void saadc_stop_streaming(void)
{
    nrfx_saadc_uninit();
    LOG_INF("SAADC DMA streaming stopped (overruns: %u)", saadc_dma_overruns);
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

/* ══════════════════════════════════════════════════════════════
 * FLAGS
 * ══════════════════════════════════════════════════════════════ */
static volatile bool start_recording = false;
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
        .interval_min = 12, .interval_max = 24, .latency = 0, .timeout = 400,
    };
    bt_conn_le_param_update(conn, &conn_params);

    int mtu_err = bt_gatt_exchange_mtu(conn, &exchange_params);
    if (mtu_err) {
        LOG_WRN("MTU exchange failed: %d, using default %d", mtu_err, nus_chunk_size);
        mtu_exchanged = true;
    }
    LOG_INF("Connected");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
    /* FIX (v5.4): Guard — uninit on idle peripheral corrupts nrfx state. */
    if (analog_recording) {
        saadc_stop_streaming();
    }
    analog_recording = false;
    is_connected     = false;
    start_recording  = false;
    mtu_exchanged    = false;
    nus_chunk_size   = 244;
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
    if (len == 3 && memcmp(data, "REC", 3) == 0) {
        start_recording = true;
    }
}

static struct bt_nus_cb nus_listener = { .received = received };

/* ══════════════════════════════════════════════════════════════
 * SEND ONE HALF-BUFFER OVER BLE
 * ══════════════════════════════════════════════════════════════ */
static bool send_half(uint8_t half_idx)
{
    const uint8_t *ptr    = (const uint8_t *)ping_pong[half_idx];
    uint32_t       remain = HALF_BUF_BYTES;
    uint32_t       offset = 0;

    while (remain > 0) {
        uint16_t to_send = (uint16_t)MIN(remain, (uint32_t)nus_chunk_size);
        int err;
        do {
            err = bt_nus_send(NULL, ptr + offset, to_send);
            if (err == -EAGAIN) k_sleep(K_MSEC(1));
        } while (err == -EAGAIN);

        if (err < 0) {
            LOG_ERR("bt_nus_send failed: %d (offset %u)", err, offset);
            return false;
        }
        offset += to_send;
        remain -= to_send;
    }
    return true;
}

/* ══════════════════════════════════════════════════════════════
 * RECORD + STREAM
 * ══════════════════════════════════════════════════════════════ */
static void record_and_stream(void)
{
    const uint32_t total_halves = TOTAL_AUDIO_SAMPLES / HALF_BUF_SAMPLES;

    char hdr[32];
    snprintf(hdr, sizeof(hdr), "START:%u\n", (uint32_t)TOTAL_AUDIO_BYTES);
    bt_nus_send(NULL, hdr, strlen(hdr));
    k_sleep(K_MSEC(10));

    led_set_cyan();
    k_sem_reset(&half_ready_sem);

    analog_recording = true;
    if (saadc_start_streaming() != 0) {
        LOG_ERR("Failed to start SAADC streaming");
        analog_recording = false;
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:SAADC", 9);
        return;
    }
    /* v5.9: log reflects no oversampling */
    LOG_INF("SAADC DMA streaming started @ %d Hz (no oversample, CC=%u)",
            SAMPLING_RATE, SAADC_CC_VALUE);

    bool ok = true;
    for (uint32_t h = 0; h < total_halves && ok; h++) {
        if (k_sem_take(&half_ready_sem, K_MSEC(500)) != 0) {
            LOG_ERR("Timeout waiting for half-buffer %u", h);
            ok = false;
            break;
        }
        log_buffer_stats(ping_pong[ready_half], ready_half);
        ok = send_half(ready_half);
    }

    saadc_stop_streaming();
    analog_recording = false;

    if (!ok) {
        LOG_ERR("Stream aborted (DMA overruns: %u)", saadc_dma_overruns);
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:ABORT", 9);
    } else {
        bt_nus_send(NULL, "finished\n", 9);
        LOG_INF("Stream complete — %u half-buffers, %u bytes, %u DMA overruns",
                total_halves, (uint32_t)TOTAL_AUDIO_BYTES, saadc_dma_overruns);
        led_set_green();
    }
}

/* ══════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════ */
int main(void)
{
    int err;

    gpio_pin_configure_dt(&red_led,   GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&green_led, GPIO_OUTPUT_ACTIVE);
    gpio_pin_configure_dt(&blue_led,  GPIO_OUTPUT_ACTIVE);
    led_set_white();

    k_work_init_delayable(&adv_restart_work, adv_restart_work_handler);

    err = saadc_init();
    if (err < 0) {
        LOG_ERR("SAADC init failed: %d", err);
        led_error_flash(led_set_yellow);
        return err;
    }

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

    led_set_red();
    LOG_INF("AcoustEEEcare ready — waiting for BLE connection");

    while (true) {
        k_sleep(K_MSEC(100));
        if (is_connected && mtu_exchanged && start_recording && !analog_recording) {
            start_recording = false;
            LOG_INF("REC received — starting record+stream (%d s @ %d Hz)",
                    DURATION_S, SAMPLING_RATE);
            record_and_stream();
        }
    }

    return 0;
}

/*
 * ══════════════════════════════════════════════════════════════
 * CHANGELOG
 * ══════════════════════════════════════════════════════════════
 *
 * v5.9 — Heart/Lung auscultation fixes
 *
 *   Fix 1 (CRITICAL): Sample rate corrected.
 *     SAADC_CC_VALUE: 500 → 2000
 *     adv_cfg.oversampling: NRF_SAADC_OVERSAMPLE_4X → OVERSAMPLE_DISABLED
 *     adv_cfg.burst: NRF_SAADC_BURST_ENABLED → BURST_DISABLED
 *     channel .burst: NRF_SAADC_BURST_ENABLED → BURST_DISABLED
 *     Root cause: burst+oversample combination doubled effective sample
 *     period; 1 kHz tone detected as 500 Hz, voice pitch-shifted down.
 *     Math: 16,000,000 / 2000 = 8000 Hz exactly, no oversampling.
 *
 *   Fix 2: DC removal filter shift >>5 → >>8.
 *     Moves high-pass corner from ~40 Hz to ~5 Hz.
 *     Preserves heart sound fundamentals at 20–150 Hz.
 *
 *   Fix 3: HALF_BUF_SAMPLES 256 → 512.
 *     Doubles DMA drain deadline (32 ms → 64 ms) to absorb BLE-induced
 *     main-thread stalls. Requires CONFIG_MAIN_THREAD_PRIORITY=5 in prj.conf.
 *
 * v5.8 — Clipping fix + tone phase fix
 *   Fix 1: Gain GAIN1 → GAIN1_2. External amp caused hard clipping.
 *   Fix 2: tone_offset → tone_half_count; deterministic phase computation.
 *
 * v5.7 — Diagnostic tone and DC filter fixes
 *   Fix 1: DC removal excluded from DIAGNOSTIC_TONE path.
 *   Fix 2: tone_offset to file scope, reset per recording.
 *   Fix 3: DC removal gated by #define ENABLE_DC_REMOVAL.
 *
 * v5.6 — Audio quality fixes
 *   Fix 1: Gain GAIN1_4 → GAIN1.
 *   Fix 2: Reference INTERNAL → VDD4.
 *   Fix 3: DC removal IIR filter added.
 *
 * v5.5 — Correct continuous-streaming pattern (BUF_REQ handling)
 * v5.4 — Double-buffer priming attempt + several critical fixes
 * v5.3 — Header + error-check fixes
 * v5.2 — SAADC re-init fix
 * v5.1 — IRQ wiring fix
 * v5.0 — nrfx_saadc hardware-timed DMA streaming
 * v4.1 — Priority inversion + semaphore overflow fix
 * v4.0 — Hardware-timer-driven sampling
 * v3.0 — Ping-pong stream
 * v2.0 — BLE-only / no SD
 * v1.3 — IMU removed
 * v1.2 — RAM overflow fix
 * v1.1 — Auscultawear mic/SD integration
 */