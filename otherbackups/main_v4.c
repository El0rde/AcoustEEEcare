/*
 * AcoustEEEcare — SAADC BLE-only, Ping-Pong Stream Edition
 * ============================================================
 * v4.1 — Fix sample_sem overflow + mic_thread priority inversion
 *
 * BUGS FIXED FROM v4.0:
 *
 * Bug 1 — sample_sem max_count was 1 (K_SEM_DEFINE(sample_sem, 0, 1)).
 *   If the BLE stack or any higher-priority work preempts mic_thread
 *   for longer than 125 µs, the timer fires multiple ticks before
 *   mic_thread wakes.  A semaphore with max_count=1 silently drops
 *   all ticks beyond the first.  mic_thread then reads one sample
 *   and immediately blocks on k_sem_take() — which now has nothing
 *   queued — and waits up to 1 second (the timeout), producing a
 *   1-second hole in the audio stream.
 *   FIX: raise max_count to HALF_BUF_SAMPLES (256) so burst ticks
 *   during preemption are absorbed rather than dropped.
 *
 * Bug 2 — mic_thread priority (5) was below BLE cooperative threads.
 *   Zephyr BLE stack runs cooperative threads at priorities 0-4.
 *   mic_thread at priority 5 was routinely preempted mid-half-buffer,
 *   causing the semaphore overflow above.
 *   FIX: raise mic_thread to priority 2 (below BLE radio ISR at 0-1,
 *   above most BLE host cooperative work at 3-4).
 *   Adjust if your BLE stack version uses different priorities.
 *
 * Bug 3 — k_sem_take timeout of K_MSEC(1000) in mic_thread.
 *   With the timer always running during recording, a timeout should
 *   never fire.  When it did (due to Bug 1), it stalled mic_thread
 *   for a full second, compounding the dropout.
 *   FIX: use K_FOREVER during recording — the timer guarantees a
 *   tick will arrive.  A watchdog log is added via a separate
 *   dropped-tick counter instead.
 *
 * Bug 4 — analog_recording cleared too early.
 *   In record_and_stream(), after the send loop, the code set
 *   analog_recording=false immediately.  If mic_thread hadn't
 *   finished its last half-buffer the counter reset, corrupting
 *   fill_count/fill_half state on next REC.
 *   FIX: let mic_thread own the analog_recording=false transition
 *   (already done when fill_count >= TOTAL_AUDIO_SAMPLES); main
 *   only clears it as an abort fallback.
 *
 * Everything else (ping-pong buffers, BLE framing) unchanged.
 * ============================================================
 */

#include <stdint.h>
#include <zephyr/bluetooth/services/nus.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/adc.h>
#include <zephyr/logging/log.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include "zephyr/dt-bindings/adc/adc.h"
#include "zephyr/kernel/thread_stack.h"
#include "zephyr/sys/time_units.h"

#include <string.h>
#include <stdio.h>

LOG_MODULE_REGISTER(AcoustEEEcare);

/* ══════════════════════════════════════════════════════════════
 * SAADC
 * ══════════════════════════════════════════════════════════════ */
#define ADC_RESOLUTION          12
#define ADC_GAIN                ADC_GAIN_4
#define ADC_REFERENCE           ADC_REF_INTERNAL
#define ADC_ACQUISITION_TIME    ADC_ACQ_TIME_DEFAULT
#define ADC_CHANNEL_MIC         0       /* AIN0 - P0.02 */

static int16_t adc_raw[1];

static struct adc_sequence adc_sequence = {
    .channels    = BIT(ADC_CHANNEL_MIC),
    .buffer      = adc_raw,
    .buffer_size = sizeof(adc_raw),
    .resolution  = ADC_RESOLUTION,
};

static const struct device *adc_dev = DEVICE_DT_GET(DT_NODELABEL(adc));

static const struct adc_channel_cfg adc_channel_cfg_mic = {
    .gain             = ADC_GAIN,
    .reference        = ADC_REFERENCE,
    .acquisition_time = ADC_ACQUISITION_TIME,
    .channel_id       = ADC_CHANNEL_MIC,
    .input_positive   = SAADC_CH_PSELP_PSELP_AnalogInput0,
};

/* ══════════════════════════════════════════════════════════════
 * AUDIO PARAMS
 * ══════════════════════════════════════════════════════════════ */
#define SAMPLING_RATE        8000
#define SAMPLE_PERIOD_US     125
#define DURATION_S           10
#define TOTAL_AUDIO_SAMPLES  (SAMPLING_RATE * DURATION_S)             /* 80 000 */
#define TOTAL_AUDIO_BYTES    (TOTAL_AUDIO_SAMPLES * sizeof(int16_t))  /* 160 000 */

#define HALF_BUF_SAMPLES   256
#define HALF_BUF_BYTES     (HALF_BUF_SAMPLES * sizeof(int16_t))       /* 512 B  */

static int16_t ping_pong[2][HALF_BUF_SAMPLES];   /* 1 024 B total */

/* main waits on this for each completed half-buffer */
static K_SEM_DEFINE(half_ready_sem, 0, 1);

/* index (0 or 1) of the half that is ready to send */
static volatile uint8_t ready_half = 0;

static volatile bool analog_recording = false;
static uint16_t      nus_chunk_size   = 244;

/* ══════════════════════════════════════════════════════════════
 * HARDWARE SAMPLE TIMER
 *
 * Fires sample_timer_expiry ISR every SAMPLE_PERIOD_US.
 * ISR gives sample_sem — mic_thread reads ADC on each token.
 *
 * FIX (Bug 1): max_count raised from 1 to HALF_BUF_SAMPLES.
 *   Allows up to 256 pending ticks to queue without loss if
 *   mic_thread is briefly preempted by BLE work.  mic_thread
 *   will drain the backlog rapidly (one ADC read per token)
 *   and catch up before the next half-buffer deadline.
 * ══════════════════════════════════════════════════════════════ */
static K_SEM_DEFINE(sample_sem, 0, HALF_BUF_SAMPLES);   /* was (0, 1) — BUG FIXED */

/* Diagnostic: counts ticks given by ISR while mic_thread was preempted */
static volatile uint32_t dropped_ticks = 0;

static void sample_timer_expiry(struct k_timer *timer)
{
    ARG_UNUSED(timer);
    if (!analog_recording) return;

    /*
     * k_sem_give() returns void — it silently clamps at max_count.
     * Check count beforehand to detect overflow.
     */
    if (k_sem_count_get(&sample_sem) >= HALF_BUF_SAMPLES) {
        dropped_ticks++;
    } else {
        k_sem_give(&sample_sem);
    }
}

static void sample_timer_stop_cb(struct k_timer *timer)
{
    ARG_UNUSED(timer);
}

K_TIMER_DEFINE(sample_timer, sample_timer_expiry, sample_timer_stop_cb);

/* ══════════════════════════════════════════════════════════════
 * MIC THREAD
 *
 * FIX (Bug 2): priority raised from 5 → 2.
 *   BLE radio runs at IRQ priority 0-1.
 *   BLE host cooperative threads run at Zephyr priority 0-4.
 *   mic_thread at priority 5 was preempted by all of them.
 *   Priority 2 keeps us above most host work while still below
 *   the radio ISR.  Tune if your BLE version differs.
 * ══════════════════════════════════════════════════════════════ */
#define MIC_THREAD_STACK_SIZE  2048
#define MIC_THREAD_PRIORITY    2        /* was 5 — BUG FIXED */

static struct k_thread mic_thread_data;
K_THREAD_STACK_DEFINE(mic_thread_stack, MIC_THREAD_STACK_SIZE);

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
        .interval_min = 6, .interval_max = 12, .latency = 0, .timeout = 400,
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
    k_timer_stop(&sample_timer);
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
 * MIC THREAD
 *
 * Blocks on sample_sem (given by hardware timer ISR every 125 µs).
 * Reads ADC immediately on wake, writes to ping_pong[fill_half].
 * When a half-buffer is full, signals main via half_ready_sem.
 *
 * FIX (Bug 3): k_sem_take now uses K_FOREVER during recording.
 *   The timer guarantees a tick every 125 µs — no timeout needed.
 *   The old K_MSEC(1000) timeout caused 1-second audio dropouts
 *   whenever a tick was missed due to the semaphore overflow bug.
 *
 * FIX (Bug 4): mic_thread is the sole owner of analog_recording=false.
 *   It sets it when fill_count reaches TOTAL_AUDIO_SAMPLES, then
 *   stops the timer itself.  main only clears it as an abort path.
 * ══════════════════════════════════════════════════════════════ */
static void mic_thread(void *a, void *b, void *c)
{
    uint32_t fill_count = 0;
    uint8_t  fill_half  = 0;

    while (1) {
        /*
         * K_FOREVER: safe because the timer is always running while
         * analog_recording is true, and we check the flag after wake.
         * If recording stops (abort/disconnect), analog_recording is
         * cleared and the timer stopped — we'll spin once through the
         * !analog_recording check then block here harmlessly until
         * the next recording starts (next timer start + sem give).
         */
        k_sem_take(&sample_sem, K_FOREVER);

        if (!analog_recording) {
            continue;
        }

        /* Read ADC — sample timing is locked to the hardware tick */
        int16_t sample;
        if (adc_read(adc_dev, &adc_sequence) == 0) {
            sample = adc_raw[0];
        } else {
            sample = 0;
            LOG_WRN("ADC read failed at fill_count %u", fill_count);
        }

        uint32_t pos = fill_count % HALF_BUF_SAMPLES;
        ping_pong[fill_half][pos] = sample;
        fill_count++;

        /* Half-buffer complete — hand off to main */
        if (pos == HALF_BUF_SAMPLES - 1) {
            ready_half = fill_half;
            fill_half  = 1 - fill_half;
            k_sem_give(&half_ready_sem);
        }

        /* Full recording done */
        if (fill_count >= TOTAL_AUDIO_SAMPLES) {
            analog_recording = false;    /* mic_thread owns this transition */
            k_timer_stop(&sample_timer);
            if (dropped_ticks > 0) {
                LOG_WRN("Recording done — %u timer ticks were dropped (sem overflow)",
                        dropped_ticks);
            }
            LOG_INF("mic_thread: all %u samples captured", TOTAL_AUDIO_SAMPLES);
            fill_count = 0;
            fill_half  = 0;
        }
    }
}

static void start_mic_thread(void)
{
    k_thread_create(&mic_thread_data, mic_thread_stack,
                    K_THREAD_STACK_SIZEOF(mic_thread_stack),
                    mic_thread, NULL, NULL, NULL,
                    MIC_THREAD_PRIORITY, 0, K_NO_WAIT);
    k_thread_name_set(&mic_thread_data, "mic_thread");
}

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

    /* Announce total byte count to receiver */
    char hdr[32];
    snprintf(hdr, sizeof(hdr), "START:%u\n", (uint32_t)TOTAL_AUDIO_BYTES);
    bt_nus_send(NULL, hdr, strlen(hdr));
    k_sleep(K_MSEC(10));

    led_set_cyan();

    /* Reset all semaphores before arming */
    k_sem_reset(&half_ready_sem);
    k_sem_reset(&sample_sem);
    dropped_ticks = 0;

    /*
     * Set analog_recording BEFORE starting the timer so the ISR
     * never fires while the flag is still false.
     */
    analog_recording = true;
    k_timer_start(&sample_timer, K_USEC(SAMPLE_PERIOD_US), K_USEC(SAMPLE_PERIOD_US));
    LOG_INF("Sample timer started @ %d Hz (period %d µs)", SAMPLING_RATE, SAMPLE_PERIOD_US);

    bool ok = true;
    for (uint32_t h = 0; h < total_halves && ok; h++) {
        /*
         * 256 samples at 8 kHz = 32 ms per half-buffer.
         * 500 ms timeout is ~15x margin — plenty for BLE congestion.
         */
        if (k_sem_take(&half_ready_sem, K_MSEC(500)) != 0) {
            LOG_ERR("Timeout waiting for half-buffer %u", h);
            ok = false;
            break;
        }
        ok = send_half(ready_half);
    }

    /* Abort path: stop timer and clear flag if mic_thread hasn't already */
    if (!ok) {
        k_timer_stop(&sample_timer);
        analog_recording = false;
    }

    /* Brief drain in case mic_thread is mid-sample */
    for (int i = 0; analog_recording && i < 100; i++) k_sleep(K_MSEC(1));

    if (!ok) {
        LOG_ERR("Stream aborted (dropped_ticks so far: %u)", dropped_ticks);
        led_error_flash(led_set_yellow);
        bt_nus_send(NULL, "ERR:ABORT", 9);
    } else {
        bt_nus_send(NULL, "finished\n", 9);
        LOG_INF("Stream complete — %u half-buffers, %u bytes, %u dropped ticks",
                total_halves, (uint32_t)TOTAL_AUDIO_BYTES, dropped_ticks);
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

    if (!device_is_ready(adc_dev)) {
        LOG_ERR("ADC device not ready");
        led_error_flash(led_set_yellow);
        return -ENODEV;
    }

    err = adc_channel_setup(adc_dev, &adc_channel_cfg_mic);
    if (err < 0) {
        LOG_ERR("Failed to setup ADC channel: %d", err);
        led_error_flash(led_set_yellow);
        return err;
    }
    LOG_INF("SAADC ready on AIN0 @ %d Hz", SAMPLING_RATE);

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

    start_mic_thread();

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
 * v4.1 — Priority inversion + semaphore overflow fix (this file)
 *
 *   Bug 1 (CRITICAL): K_SEM_DEFINE(sample_sem, 0, 1) — max_count of
 *     1 caused all timer ticks beyond the first to be silently
 *     dropped whenever mic_thread was preempted by BLE work.
 *     Fixed: max_count raised to HALF_BUF_SAMPLES (256).
 *
 *   Bug 2 (CRITICAL): MIC_THREAD_PRIORITY = 5 placed mic_thread
 *     below BLE cooperative host threads (Zephyr priority 0-4),
 *     causing frequent preemption and the semaphore overflow above.
 *     Fixed: priority raised to 2.
 *
 *   Bug 3: k_sem_take K_MSEC(1000) timeout in mic_thread caused
 *     1-second audio silence on each missed tick.
 *     Fixed: K_FOREVER — timer guarantees ticks while recording.
 *
 *   Bug 4: record_and_stream() cleared analog_recording=false
 *     immediately after the send loop, racing with mic_thread's
 *     fill_count/fill_half state reset.
 *     Fixed: mic_thread owns analog_recording=false at completion;
 *     main only touches it on the abort path.
 *
 *   Added: dropped_ticks counter — reported in logs so overflow
 *     can be detected even if it's infrequent.
 *
 *   Added: k_thread_name_set() for mic_thread — improves Zephyr
 *     kernel-aware debugger output.
 *
 * v4.0 — Hardware-timer-driven sampling
 *   Replaced k_busy_wait() with K_TIMER_DEFINE + sample_sem.
 *
 * v3.0 — Ping-pong stream
 *   Replaced mic_buf[80000] (160 KB) with ping_pong[2][256] (1 KB).
 *
 * v2.0 — BLE-only / no SD (overflowed RAM)
 * v1.3 — IMU removed
 * v1.2 — RAM overflow fix
 * v1.1 — Auscultawear mic/SD integration
 */