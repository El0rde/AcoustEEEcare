/*
 * AcoustEEEcare — BLE + one-capture DUAL inference (v8.6)
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
 *   FIX A  gain = GAIN1_4 (full-scale ≈ VDD so the 1.65 V bias sits at mid-scale;
 *          amplifying gains such as GAIN1/GAIN2/GAIN4 saturate the bias).
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
 * Host protocol (v8.4 — one capture, both results):
 *   "READY\n"                         (once, after MTU; gate the Record button)
 *   "START:<bytes>:DUAL\n"
 *   audio packets : [u16 seq LE][u16 len LE][pcm int16 LE ...]
 *   "finished\n"
 *   "INF:START\n" → "HR:<bpm>\n" → "RR:<bpm>\n" → "INF:DONE\n"
 *   (on error, "ERR:HEART_INF\n" / "ERR:LUNG_INF\n" replaces the value)
 *   "PING\n" keepalives may arrive any time while connected.
 * The host builds the .wav from the streamed PCM and shows HR + RR.
 *
 * v8.6 fix — 250 Hz BLE radio noise:
 *   WAV analysis showed a broadband noise hump at ~249 Hz that was 1.4×
 *   stronger than the heart signal band (20–200 Hz). Root cause: the BLE
 *   radio fires a current burst every connection event (15–30 ms interval)
 *   AND every PING (250 ms = 4 Hz, but radio bursts repeat at the
 *   connection interval = ~250 Hz). These current spikes couple into the
 *   analog supply rail and through the op-amp PSRR into the SAADC input.
 *   Fix A (firmware): suppress PING keepalives during analog_recording.
 *     The connection interval is 15–30 ms so BLE data still flows fine
 *     for audio streaming; the supervision timer (6 s) covers the full
 *     10 s capture + inference window without keepalives.
 *   Fix B (firmware): widen connection interval to 100–200 ms during
 *     capture so the radio fires less often (~5–10 Hz instead of ~50 Hz),
 *     then restore fast interval for audio streaming.
 *     NOTE: wider interval = lower BLE throughput. At 160 KB / 10 s =
 *     16 KB/s, even a 200 ms interval (5 packets/s × ~240 B = 1200 B/s)
 *     is too slow. So Fix B is NOT applied here — audio throughput wins.
 *     The real solution is hardware shielding / supply decoupling (see
 *     hardware note below).
 *   Fix C (firmware): extend supervision timeout to cover capture +
 *     inference (10 s + ~1 s) without any keepalive: set to 1500 (15 s).
 *     This replaces the keepalive-during-capture pattern entirely.
 *   Hardware note: add 100 nF + 10 µF decoupling caps on the op-amp VCC
 *     pin as close to the package as possible. A ferrite bead between the
 *     nRF52840 VDD and op-amp VCC is the most effective fix. This cannot
 *     be solved in firmware alone if the PCB has no supply filtering.
 *
 * v8.5 fix — audio speed-up:
 *   Root cause: send_audio_block() was non-blocking. On BLE congestion
 *   chunks were silently dropped but tx_seq still advanced. If the host
 *   plays back only received packets without zero-filling the gaps, audio
 *   plays faster than real-time (e.g. 30% drops → 70% playback speed).
 *   Fix: bounded retry (up to BLE_SEND_RETRIES × 1 ms) before dropping.
 *   This keeps the staging ring safe — the ring holds ~1 s of slack, and
 *   a short retry window (≤4 ms per chunk) is far less than that budget.
 *   In practice the BLE stack clears within 1–2 ms; retries are rare.
 *   If a chunk still can't be sent after all retries it is dropped and
 *   tx_seq advances so the host can zero-fill that exact gap position.
 *
 * Audio quality improvements (on top of warmup-discard fix):
 *   IMP 1  OVERSAMPLE_8X instead of 4X: halves quantization noise floor
 *          (~3 dB SNR gain) at no timing cost at 8 kHz.
 *   IMP 2  ACQTIME_5US (revised from 20US): 20 µs was incompatible with
 *          OVERSAMPLE_8X at CC=2000 — 8×(320+26)=2768 cycles exceeds the
 *          2000-cycle CC period, causing the SAADC to stretch timing and
 *          produce sped-up, higher-pitch audio. 5 µs gives a safe burst
 *          of only 848 cycles. The 8x averaging itself provides the SNR
 *          improvement that 20 µs was intended to achieve.
 *   IMP 3  DC IIR tau slowed from >>8 (~32 ms) to >>10 (~128 ms): prevents
 *          the filter from attenuating low-frequency heart sounds (S1/S2
 *          are 20–100 Hz). DC is still removed; it just tracks slower.
 *   IMP 4  dc_estimate seeded from the average of the last 16 warmup
 *          samples instead of a single tail sample: far stabler initial
 *          value, eliminates any residual step at capture start.
 *   IMP 5  12-bit SAADC output left-shifted by 4 into the int16 MSBs
 *          before BLE streaming: uses the full int16 dynamic range so
 *          the host WAV is properly scaled (no signal loss; MFCC path
 *          feeds the unshifted raw samples, unchanged).
 *
 * Warmup-discard fix (non-rail-to-rail op-amp):
 *   A non-RRIO op-amp cannot swing to the supply rails, so its output
 *   sits at an undefined voltage until the bias network settles. Without
 *   a discard phase the ramp contaminates the first ~200-500 ms of the
 *   WAV (listening path) and — less critically — the earliest MFCC frames.
 *   Fix: after saadc_start() we drain WARMUP_SAMPLES into /dev/null
 *   (no MFCC feed, no BLE send). Only then is dc_estimate seeded from
 *   the first real sample and normal capture begins.
 *   WARMUP_SAMPLES = 4000 = 500 ms @ 8 kHz — tune down if the op-amp
 *   settles faster (measure the WAV ramp and halve until it disappears).
 */

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

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
#include <zephyr/sys/atomic.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>

#include <nrfx.h>
#include <drivers/nrfx_errors.h>
#include <nrfx_saadc.h>

#include "dsp_mfcc.h"
#include "heart_mfcc_config.h"
#include "lung_mfcc_config.h"
#include "cnn_norm_stats.h"
#include "tflm_inference.h"

LOG_MODULE_REGISTER(AcoustEEEcare);

#define DEBUG_INJECT_BUFFER  0     /* 0 = disabled for production; set to 1 to re-enable self-test */
 
#if DEBUG_INJECT_BUFFER
#include "debug_audio.h"           /* const int16_t debug_audio_pcm[80000] @ 8 kHz */
#endif

/* ══════════════════════════════════════════════════════════════════
 * ORGAN SELECTION
 * ══════════════════════════════════════════════════════════════════ */
/* v8.4: TWO persistent pipelines fed from ONE capture.
 * Each chunk is fed to heart then lung SEQUENTIALLY (never concurrently),
 * so the shared FFT scratch inside dsp_mfcc.c is safe — each feed_chunk
 * finishes its transform before the next begins. Each pipeline owns its
 * own circular window (heart=60, lung=1200). No organ selection: every
 * recording produces BOTH HR and RR. */
static dsp_mfcc_pipeline_t heart_pipeline;
static dsp_mfcc_pipeline_t lung_pipeline;
static int16_t heart_window[60]   __aligned(4);
static int16_t lung_window[1200]  __aligned(4);

/* ══════════════════════════════════════════════════════════════════
 * SAADC CONFIG  (FIX A: GAIN1_4)
 * ══════════════════════════════════════════════════════════════════ */
/* CC timing constraint with OVERSAMPLE_8X + ACQTIME_5US:
 *   Each burst = 8 × (ACQTIME + CONVTIME) = 8 × (80 + 26) ≈ 848 cycles
 *   CC must be ≥ burst cycles AND = 16 MHz / output_rate.
 *   16 MHz / 8000 Hz = 2000 cycles.  848 << 2000, so 5 µs acq is safe.
 *   ACQTIME_20US was NOT safe: 8 × (320+26) = 2768 > 2000 → timing break.
 *   ACQTIME_10US is marginal: 8 × (160+26) = 1488 < 2000 but tight.
 *   ACQTIME_5US gives plenty of headroom and is fine for a driven op-amp. */
#define SAADC_CC_VALUE      2000U   /* 16 MHz / 2000 = 8 kHz output rate */
#define SAADC_IRQ_PRIORITY  6

static const nrfx_saadc_channel_t saadc_channel_cfg = {
    .channel_config = {
        .resistor_p = NRF_SAADC_RESISTOR_DISABLED,
        .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
        .gain       = NRF_SAADC_GAIN1_4,            /* GAIN1_4: full-scale ≈ VDD so the 1.65 V bias sits at mid-scale; amplifying gains saturate the bias */
        .reference  = NRF_SAADC_REFERENCE_VDD4,
        .acq_time   = NRF_SAADC_ACQTIME_5US,        /* 5 µs safe with OVERSAMPLE_8X @ CC=2000: burst=8×(80+26)=848 cycles << 2000 */
        .mode       = NRF_SAADC_MODE_SINGLE_ENDED,
        .burst      = NRF_SAADC_BURST_ENABLED,
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

/* ── Warmup discard (non-RRIO op-amp ramp-up fix) ──────────────────
 * How long to let the op-amp output settle before we start capturing.
 * 500 ms (4000 samples @ 8 kHz) is conservative; halve it if your
 * scope / WAV shows the ramp has already gone by 250 ms.
 * These samples are drained from the SAADC ring and discarded — no
 * MFCC feed, no BLE audio send. Only TOTAL_AUDIO_SAMPLES are captured
 * and streamed to the host after the warmup phase completes.       */
#define WARMUP_SAMPLES       (SAMPLING_RATE / 2)     /* 4 000 = 500 ms */

#define HALF_BUF_SAMPLES     512
#define HALF_BUF_BYTES       (HALF_BUF_SAMPLES * sizeof(int16_t))

static int16_t ping_pong[2][HALF_BUF_SAMPLES];
static volatile uint8_t  next_dma_buf     = 1;
static volatile bool     analog_recording = false;
static uint16_t          nus_chunk_size   = 244;

/* Supervision timeout covers the full capture (10 s) + both inferences
 * (~1 s) + margin, with NO keepalive pings during capture (v8.6 fix).
 * 1500 = 15 s. Must be > (capture_duration + inference_time) × 100. */
#define BLE_SUPERVISION_TIMEOUT   1500
#define PING_INTERVAL_MS          250     /* B7: keep link warm; prevent central from relaxing interval */

/* ══════════════════════════════════════════════════════════════════
 * ISR -> MAIN STAGING RING
 *   The ISR copies each completed half-buffer into one of STAGE_SLOTS
 *   staging buffers and gives stage_sem. The MAIN thread drains them.
 *   8 slots = ~512 ms of slack, plenty to cover BLE send + FFT time.
 * ══════════════════════════════════════════════════════════════════ */
#define STAGE_SLOTS   16      /* ~1 s of slack (was 8 = 512 ms) */
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
/* v8.4: int8 feature stores, one per organ. Frames are normalized
 * (cnn_norm_stats.h) and quantized to int8 IN the callbacks, so we never
 * hold a large float matrix — heart 16,625 B + lung 8,424 B ≈ 25 KB total,
 * which fits comfortably alongside a 72 KB arena. */
#define HEART_FRAMES  665
#define HEART_NMFCC   25
#define LUNG_FRAMES   324
#define LUNG_NMFCC    26
#define HEART_FEAT_N  (HEART_FRAMES * HEART_NMFCC)   /* 16625 */
#define LUNG_FEAT_N   (LUNG_FRAMES  * LUNG_NMFCC)    /*  8424 */

static int8_t  heart_features[HEART_FEAT_N] __aligned(4);
static int8_t  lung_features [LUNG_FEAT_N]  __aligned(4);
static volatile uint32_t heart_feat_count;   /* int8 values written */
static volatile uint32_t lung_feat_count;
static volatile int      heart_frame_count;  /* frames seen */
static volatile int      lung_frame_count;

/* Per-organ int8 input quantization params.
 * Discovered at boot by probing each model's input tensor (tflm_get_input_quant),
 * so they always match the loaded model — no hardcoded-scale drift. The verified
 * values are kept only as a fallback if the probe fails. */
static float   heart_q_scale = 0.39602566f;
static int32_t heart_q_zp    = 38;
static float   lung_q_scale  = 0.33742353f;
static int32_t lung_q_zp     = 28;

/* TFLite Micro shared tensor arena (heart and lung run sequentially on it).
 * Start at 72 KB (proven on the previous build); each inference logs
 * "arena used = N" — shrink to N + ~4 KB once measured. Heart is worst case. */
#define TENSOR_ARENA_BYTES  (72u * 1024u)
static uint8_t tensor_arena[TENSOR_ARENA_BYTES] __aligned(16);

/* Exported high-water for logging; written by tflm_inference.cc. */
volatile uint32_t g_tflm_arena_used_bytes;

#define CHUNK_HEADER_BYTES  4
/* Max 1 ms retries per BLE chunk before dropping.  Keeps audio near
 * real-time without blocking long enough to overflow the staging ring.
 * Each retry = 1 ms sleep; 4 retries = 4 ms worst-case per chunk.
 * The staging ring holds ~1 s of slack so this is well within budget. */
#define BLE_SEND_RETRIES    4
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
    adv.oversampling      = NRF_SAADC_OVERSAMPLE_8X;  /* IMP1: 8x vs 4x → ~3 dB lower quantization noise floor */
    adv.burst             = NRF_SAADC_BURST_ENABLED;
    adv.internal_timer_cc = SAADC_CC_VALUE;
    adv.start_on_end      = true;

    err = nrfx_saadc_advanced_mode_set(BIT(saadc_channel_cfg.channel_index),
                                        NRF_SAADC_RESOLUTION_12BIT,
                                        &adv, saadc_event_handler);
    if (err != 0) { LOG_ERR("advanced_mode_set: 0x%08X", err); return -EIO; }

    LOG_INF("SAADC configured once @ %d Hz (GAIN1_4)", SAMPLING_RATE);

    /* Offset calibration: run once while SAADC is enabled but not yet
     * sampling.  Removes temperature/supply-dependent offset drift.
     * Must be called BEFORE saadc_start(). */
    err = nrfx_saadc_offset_calibrate(NULL);   /* blocking form */
    if (err != 0) {
        LOG_WRN("saadc offset_calibrate: 0x%08X (non-fatal, continuing)", err);
    } else {
        LOG_INF("SAADC offset calibration done");
    }

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
static volatile bool start_test = false;

/* B1: connection handle + current interval for fast-link re-assertion */
static struct bt_conn   *current_conn;
static volatile uint16_t cur_interval;


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
    /* Handshake: the host app waits for READY before enabling Record. */
    bt_nus_send(NULL, "READY\n", 6);
    LOG_INF("sent READY");
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
    current_conn = bt_conn_ref(conn);   /* B2: keep ref for fast-link re-assertion */
    led_green();

    static const struct bt_conn_le_phy_param phy = {
        .options     = BT_CONN_LE_PHY_OPT_NONE,
        .pref_tx_phy = BT_GAP_LE_PHY_2M,
        .pref_rx_phy = BT_GAP_LE_PHY_2M,
    };
    bt_conn_le_phy_update(conn, &phy);

    static const struct bt_le_conn_param cp = {
        .interval_min = 12, .interval_max = 24, .latency = 0,
        .timeout = BLE_SUPERVISION_TIMEOUT,
    };
    bt_conn_le_param_update(conn, &cp);
    bt_gatt_exchange_mtu(conn, &exchange_params);
    LOG_INF("Connected");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
    if (current_conn) { bt_conn_unref(current_conn); current_conn = NULL; }  /* B3 */
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

/* B4: log whenever the central changes the connection interval */
static void le_param_updated(struct bt_conn *conn, uint16_t interval,
                              uint16_t latency, uint16_t timeout)
{
    ARG_UNUSED(conn); ARG_UNUSED(latency);
    cur_interval = interval;
    LOG_INF("conn interval=%u (%u.%02u ms) timeout=%u",
            interval, (interval*5)/4, ((interval*5)%4)*25, timeout);
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
    .connected        = connected,
    .disconnected     = disconnected,
    .le_param_updated = le_param_updated,   /* B4 */
};

static void received(struct bt_conn *conn, const void *data, uint16_t len, void *ctx)
{
    ARG_UNUSED(conn); ARG_UNUSED(ctx);
    /* v8.4: every recording produces BOTH HR and RR, so only REC is needed. */
    if (len == 3 && memcmp(data, "REC", 3) == 0) {
        start_recording = true;
    }

#if DEBUG_INJECT_BUFFER
    else if (len == 4 && memcmp(data, "TEST", 4) == 0) {
        start_test = true;
    }
#endif

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
/* Both callbacks: per-coefficient z-score normalization (matching CNN
 * training, heart_cnn/data.py: v = (mfcc - mean[i])/(std[i] + CNN_NORM_EPS)),
 * then int8 quantization with that organ's input scale/zp, appended into
 * the organ's int8 feature buffer. Stats from cnn_norm_stats.h. */
static void heart_frame_cb(int idx, const float *coeffs, void *user)
{
    ARG_UNUSED(user); ARG_UNUSED(idx);
    if (heart_frame_count >= HEART_FRAMES) return;
    heart_frame_count++;
    const int nc = heart_pipeline.cfg->n_mfcc;
    for (int c = 0; c < nc; c++) {
        float v = (coeffs[c] - heart_mfcc_mean[c]) / (heart_mfcc_std[c] + CNN_NORM_EPS);
        int32_t q = (int32_t)lroundf(v / heart_q_scale) + heart_q_zp;
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        if (heart_feat_count < HEART_FEAT_N)
            heart_features[heart_feat_count++] = (int8_t)q;
    }
}

static void lung_frame_cb(int idx, const float *coeffs, void *user)
{
    ARG_UNUSED(user); ARG_UNUSED(idx);
    if (lung_frame_count >= LUNG_FRAMES) return;
    lung_frame_count++;
    const int nc = lung_pipeline.cfg->n_mfcc;
    for (int c = 0; c < nc; c++) {
        float v = (coeffs[c] - lung_mfcc_mean[c]) / (lung_mfcc_std[c] + CNN_NORM_EPS);
        int32_t q = (int32_t)lroundf(v / lung_q_scale) + lung_q_zp;
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        if (lung_feat_count < LUNG_FEAT_N)
            lung_features[lung_feat_count++] = (int8_t)q;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * Send one block of PCM as audio packets (DC-removed for listening)
 * ══════════════════════════════════════════════════════════════════ */
static int16_t dc_copy[HALF_BUF_SAMPLES];

static void send_audio_block(const int16_t *raw, uint32_t n)
{
    /* DC-remove a copy (listening path only).
     * IMP3: >>10 (~128 ms τ) instead of >>8 (~32 ms): prevents the IIR
     *       from attenuating low-frequency heart sounds (S1/S2: 20–100 Hz).
     * IMP5: left-shift the 12-bit SAADC value into the int16 MSBs so the
     *       host WAV uses the full int16 dynamic range.  The MFCC path
     *       receives unshifted raw[] — this shift is ONLY in dc_copy[]. */
    for (uint32_t i = 0; i < n; i++) {
        dc_estimate += ((int32_t)raw[i] - dc_estimate) >> 10;   /* IMP3 */
        int32_t centered = (int32_t)raw[i] - dc_estimate;
        dc_copy[i] = (int16_t)(centered << 4);                  /* IMP5: 12→16 bit scaling */
    }
 
    const uint8_t *p = (const uint8_t *)dc_copy;
    uint32_t bytes   = n * sizeof(int16_t);
    uint32_t offset  = 0;
    uint8_t  chunk[251];
 
    while (bytes > 0) {
        uint16_t payload = nus_chunk_size - CHUNK_HEADER_BYTES;
        uint16_t send    = (bytes < payload) ? (uint16_t)bytes : payload;
        sys_put_le16(tx_seq, &chunk[0]);
        sys_put_le16(send,   &chunk[2]);
        memcpy(&chunk[4], p + offset, send);
 
        /* Bounded-retry send: attempt up to BLE_SEND_RETRIES times with a
         * 1 ms sleep between attempts.  This gives the BLE stack time to
         * drain its TX queue without blocking so long that the SAADC
         * staging ring overflows.  If all retries fail the chunk is dropped
         * and tx_seq still advances so the host can zero-fill that gap
         * (silence at that position) rather than compressing the timeline. */
        {
            int err = -ENOMEM;
            for (int r = 0; r < BLE_SEND_RETRIES && (err == -ENOMEM || err == -EAGAIN); r++) {
                err = bt_nus_send(NULL, chunk, CHUNK_HEADER_BYTES + send);
                if (err == -ENOMEM || err == -EAGAIN) k_sleep(K_MSEC(1));
            }
            (void)err;   /* intentional drop after retries exhausted */
        }
 
        tx_seq  = (tx_seq + 1) & 0xFFFF;    /* advance even on drop */
        offset += send;
        bytes  -= send;
    }
}

/* ══════════════════════════════════════════════════════════════════
 * Stream all stored MFCC frames
 * ══════════════════════════════════════════════════════════════════ */
/* Run on-device inference on the normalized feature store for the active
 * organ and report the result over BLE.
 *
 *   heart -> "INFER:HEART:<HR>\n"   (HR clipped [40,180])
 *   lung  -> "INFER:LUNG:<RR>\n"    (RR clipped [6,50])
 *   error -> "INFER:ERR:<rc>\n"
 *
 * The normalized features were filled by mfcc_frame_cb during capture.
 * TFLM quantizes them with the model's own int8 input scale/zp. */
/* [STAB FIX] True only while the two Invoke() calls run, so the main-loop
 * keepalive keeps PINGing through the ~20-35 s inference window. */
static atomic_t in_inference = ATOMIC_INIT(0);


/* Run BOTH inferences on the shared arena (heart then lung) and report:
 *   "INF:START\n" → "HR:<bpm>\n" → "RR:<bpm>\n" → "INF:DONE\n"
 * Errors send "ERR:HEART_INF\n" / "ERR:LUNG_INF\n" in place of the value. */
static void run_inference_and_report(void)
{
    if (heart_frame_count < HEART_FRAMES || lung_frame_count < LUNG_FRAMES) {
        LOG_WRN("frames: heart=%d/%d lung=%d/%d (inference may be degraded)",
                heart_frame_count, HEART_FRAMES, lung_frame_count, LUNG_FRAMES);
    }
    LOG_INF("feat counts: heart=%u/%u lung=%u/%u",
            (unsigned)heart_feat_count, (unsigned)HEART_FEAT_N,
            (unsigned)lung_feat_count,  (unsigned)LUNG_FEAT_N);

    heart_result_t hr; lung_result_t rr;
    char msg[48];

    ble_send_blocking("INF:START\n", 10);
    atomic_set(&in_inference, 1);

    led_purple();
    run_heart_inference_ram(tensor_arena, sizeof(tensor_arena),
                            heart_features, HEART_FRAMES, HEART_NMFCC, &hr);

    k_yield();   /* let the BLE stack breathe between the two Invokes */

    led_cyan();
    run_lung_inference_ram(tensor_arena, sizeof(tensor_arena),
                           lung_features, LUNG_FRAMES, LUNG_NMFCC, &rr);

    atomic_set(&in_inference, 0);

    if (hr.rc == 0) {
        int n = snprintf(msg, sizeof(msg), "HR:%.1f\n", (double)hr.value);
        ble_send_blocking(msg, n);
        LOG_INF("HR = %.1f BPM", (double)hr.value);
    } else {
        ble_send_blocking("ERR:HEART_INF\n", 14);
        LOG_ERR("heart inference failed: %d", hr.rc);
    }
    k_sleep(K_MSEC(20));
    if (rr.rc == 0) {
        int n = snprintf(msg, sizeof(msg), "RR:%.1f\n", (double)rr.value);
        ble_send_blocking(msg, n);
        LOG_INF("RR = %.1f BPM", (double)rr.value);
    } else {
        ble_send_blocking("ERR:LUNG_INF\n", 13);
        LOG_ERR("lung inference failed: %d", rr.rc);
    }

    ble_send_blocking("INF:DONE\n", 9);
    LOG_INF("arena high-water = %u B", (unsigned)g_tflm_arena_used_bytes);
}

/* ══════════════════════════════════════════════════════════════════
 * RECORD AND STREAM (everything in the MAIN thread)
 * ══════════════════════════════════════════════════════════════════ */

/* B5: re-request the fast connection interval (12–24 = 15–30 ms).
 * Called at the start of each recording; the central may have widened
 * the interval while idle to save power. */
static void request_fast_link(void)
{
    if (!current_conn) return;
    static const struct bt_le_conn_param fast = {
        .interval_min = 12, .interval_max = 24, .latency = 0,
        .timeout = BLE_SUPERVISION_TIMEOUT,
    };
    int e = bt_conn_le_param_update(current_conn, &fast);
    if (e) LOG_WRN("fast-link request: %d", e);
}

#if DEBUG_INJECT_BUFFER
/* Inject a known heartbeat buffer through the SAME pipelines the live
 * capture uses (no SAADC). Confirms MFCC + normalize + quantize + TFLM
 * end-to-end on hardware. Expected HR ≈ 85–90 BPM for the bundled clip. */
static void run_injected_test(void)
{
    heart_feat_count = lung_feat_count = 0;
    heart_frame_count = lung_frame_count = 0;
    memset(heart_features, 0, sizeof(heart_features));
    memset(lung_features,  0, sizeof(lung_features));
    dsp_mfcc_reset(&heart_pipeline);
    dsp_mfcc_reset(&lung_pipeline);
    dc_estimate = 2048;
    tx_seq = 0;
 
    char hdr[40];
    snprintf(hdr, sizeof(hdr), "START:%u:DEBUG\n", (uint32_t)(DEBUG_AUDIO_LEN * 2));
    ble_send_blocking(hdr, strlen(hdr));
    k_sleep(K_MSEC(10));
    LOG_INF("INJECT TEST: feeding %u known samples (expect HR~85-90)",
            (unsigned)DEBUG_AUDIO_LEN);
 
    led_cyan();
    uint32_t done = 0;
    while (done < DEBUG_AUDIO_LEN) {
        uint32_t use = (DEBUG_AUDIO_LEN - done < HALF_BUF_SAMPLES)
                       ? (DEBUG_AUDIO_LEN - done) : HALF_BUF_SAMPLES;
        int16_t *chunk = (int16_t *)&debug_audio_pcm[done];   /* flash, read-only */
        dsp_mfcc_feed_chunk(&heart_pipeline, chunk, (int)use);
        dsp_mfcc_feed_chunk(&lung_pipeline,  chunk, (int)use);
        send_audio_block(chunk, use);   /* stream so the host can verify the clip */
        done += use;
    }
 
    ble_send_blocking("finished\n", 9);
    LOG_INF("INJECT TEST: frames h=%d l=%d", heart_frame_count, lung_frame_count);
    run_inference_and_report();
    led_green();
}
#endif


static void record_and_stream(void)
{
    /* B6: re-assert fast interval + push a warm-up burst so the link is
     * fast before audio streaming starts (~400 ms). Must run BEFORE the
     * resets below so nothing is clobbered. "PING" is ignored by the host. */
    request_fast_link();
    for (int i = 0; i < 8; i++) {
        bt_nus_send(NULL, "PING\n", 5);
        k_sleep(K_MSEC(50));
    }

    /* Reset all per-recording state (FIX C). */
    dc_estimate      = 2048;
    tx_seq           = 0;
    stage_wr = stage_rd = 0;
    stage_overruns   = 0;
    heart_feat_count = lung_feat_count = 0;
    heart_frame_count = lung_frame_count = 0;
    memset(heart_features, 0, sizeof(heart_features));
    memset(lung_features,  0, sizeof(lung_features));
    k_sem_reset(&stage_sem);

    /* Reset both persistent pipelines (clears bandpass state, window, phase). */
    dsp_mfcc_reset(&heart_pipeline);
    dsp_mfcc_reset(&lung_pipeline);

    char hdr[40];
    snprintf(hdr, sizeof(hdr), "START:%u:DUAL\n", (uint32_t)TOTAL_AUDIO_BYTES);
    ble_send_blocking(hdr, strlen(hdr));
    k_sleep(K_MSEC(10));

    led_cyan();
    analog_recording = true;
    if (saadc_start() != 0) {
        analog_recording = false;
        ble_send_blocking("ERR:SAADC", 9);
        return;
    }

    LOG_INF("Recording %d s @ %d Hz (heart + lung)", DURATION_S, SAMPLING_RATE);

    /* ── Warmup discard phase ──────────────────────────────────────────
     * Drain WARMUP_SAMPLES from the SAADC staging ring and throw them
     * away.  No MFCC feed, no BLE audio send.  This lets the non-RRIO
     * op-amp output settle to its mid-supply operating point before we
     * start capturing meaningful audio.
     *
     * The staging ring is sized for 16 slots (~1 s), so the warmup period
     * (500 ms = 8 half-buffers) does not overflow it.
     *
     * After the loop we seed dc_estimate from the first sample of the
     * first real buffer, so the DC IIR starts at the true operating point
     * instead of the hardcoded 2048.  This eliminates the residual
     * exponential tail that would otherwise appear even after the op-amp
     * settles.                                                           */
    {
        uint32_t warmup_done = 0;
        bool     warmup_ok   = true;

        LOG_INF("Warmup discard: draining %u samples (%u ms) ...",
                (unsigned)WARMUP_SAMPLES,
                (unsigned)(WARMUP_SAMPLES * 1000u / SAMPLING_RATE));

        while (warmup_done < WARMUP_SAMPLES) {
            if (k_sem_take(&stage_sem, K_MSEC(500)) != 0) {
                LOG_ERR("warmup staging timeout at %u/%u samples",
                        warmup_done, (unsigned)WARMUP_SAMPLES);
                warmup_ok = false;
                break;
            }
            uint8_t  rd    = stage_rd;
            uint32_t avail = stage_len[rd];
            uint32_t rem   = WARMUP_SAMPLES - warmup_done;
            uint32_t use   = (avail < rem) ? avail : rem;

            /* IMP4: seed dc_estimate from the average of the last 16
             * samples of the final warmup buffer — far stabler than a
             * single tail sample, eliminates any residual step at the
             * start of real capture.                                    */
            if (warmup_done + use >= WARMUP_SAMPLES) {
                /* Average the last min(16, use) samples of this buffer. */
                uint32_t avg_n = (use < 16u) ? use : 16u;
                int32_t  sum   = 0;
                for (uint32_t s = use - avg_n; s < use; s++)
                    sum += (int32_t)stage_buf[rd][s];
                dc_estimate = sum / (int32_t)avg_n;
                LOG_INF("dc_estimate seeded (avg of last %u warmup samples): %d",
                        (unsigned)avg_n, (int)dc_estimate);
            }

            warmup_done += use;
            stage_rd = (rd + 1) % STAGE_SLOTS;
        }

        if (!warmup_ok) {
            saadc_stop();
            analog_recording = false;
            ble_send_blocking("ERR:WARMUP\n", 11);
            led_red();
            return;
        }

        LOG_INF("Warmup done. overruns so far=%u. Starting real capture.", stage_overruns);
    }
    /* ── End warmup discard ────────────────────────────────────────── */

    /* Drain staged buffers in the main thread until exactly
     * TOTAL_AUDIO_SAMPLES are handled. Each chunk is fed to BOTH pipelines
     * sequentially (shared FFT scratch is safe), then the DC-removed copy
     * is streamed for the host WAV. The last partial chunk feeds only its
     * remaining samples so each pipeline emits exactly its n_frames. */
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

        /* MFCC paths — RAW samples (matches training); heart then lung. */
        dsp_mfcc_feed_chunk(&heart_pipeline, raw, (int)use);
        dsp_mfcc_feed_chunk(&lung_pipeline,  raw, (int)use);

        /* Listening path — DC-removed audio over BLE (host builds the WAV). */
        send_audio_block(raw, use);

        samples_done += use;
        stage_rd = (rd + 1) % STAGE_SLOTS;
    }

    saadc_stop();
    analog_recording = false;

    ble_send_blocking("finished\n", 9);
    LOG_INF("Audio done: %u samples, overruns=%u, frames h=%d l=%d",
            samples_done, stage_overruns, heart_frame_count, lung_frame_count);

    if (!ok) {
        led_red();
        return;
    }

    /* One capture → both inferences on the shared arena. */
    run_inference_and_report();

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

    /* Build both persistent pipelines once and bind their windows + callbacks. */
    if (dsp_mfcc_init(&heart_pipeline, &heart_mfcc_config) != 0) {
        LOG_ERR("heart dsp_mfcc_init failed"); return -1;
    }
    heart_pipeline.window = heart_window;
    dsp_mfcc_set_callback(&heart_pipeline, heart_frame_cb, NULL);

    if (dsp_mfcc_init(&lung_pipeline, &lung_mfcc_config) != 0) {
        LOG_ERR("lung dsp_mfcc_init failed"); return -1;
    }
    lung_pipeline.window = lung_window;
    dsp_mfcc_set_callback(&lung_pipeline, lung_frame_cb, NULL);

    /* Probe each model's int8 input scale/zp once, so the in-callback
     * quantization always matches the loaded model (no hardcoded drift).
     * Verified values remain as the fallback if a probe fails. */
    {
        float sc; int32_t zp;
        if (tflm_get_input_quant(0 /*heart*/, tensor_arena, sizeof(tensor_arena),
                                 &sc, &zp) == 0) {
            heart_q_scale = sc; heart_q_zp = zp;
        } else {
            LOG_WRN("heart quant probe failed — using fallback %.6f/%d",
                    (double)heart_q_scale, (int)heart_q_zp);
        }
        if (tflm_get_input_quant(1 /*lung*/, tensor_arena, sizeof(tensor_arena),
                                 &sc, &zp) == 0) {
            lung_q_scale = sc; lung_q_zp = zp;
        } else {
            LOG_WRN("lung quant probe failed — using fallback %.6f/%d",
                    (double)lung_q_scale, (int)lung_q_zp);
        }
        LOG_INF("quant: heart %.6f/%d  lung %.6f/%d",
                (double)heart_q_scale, (int)heart_q_zp,
                (double)lung_q_scale,  (int)lung_q_zp);
    }

    led_red();
    LOG_INF("AcoustEEEcare v8.4 ready — one-capture dual inference");

    uint32_t last_ping = 0;
    while (true) {
        k_sleep(K_MSEC(100));

        if (is_connected && mtu_exchanged && start_recording && !analog_recording) {
            start_recording = false;
            record_and_stream();
        }

#if DEBUG_INJECT_BUFFER
        if (is_connected && mtu_exchanged && start_test && !analog_recording) {
            start_test = false;
            run_injected_test();
        }
#endif

        /* Keepalive: PING every PING_INTERVAL_MS when connected and IDLE.
         * v8.6: NEVER ping during analog_recording — the radio burst every
         * connection event couples ~250 Hz noise into the op-amp supply and
         * overwhelms the heart signal band. The supervision timeout (15 s)
         * covers the full capture + inference window without any keepalive.
         * Resume pinging during inference (in_inference) since the SAADC
         * is stopped by then and radio noise no longer contaminates audio. */
        if (is_connected && (!analog_recording || atomic_get(&in_inference))) {
            uint32_t now = k_uptime_get_32();
            if (now - last_ping > PING_INTERVAL_MS) {
                bt_nus_send(NULL, "PING\n", 5);
                last_ping = now;
            }
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
 *   Input layout frame-major: features[frame*n_mfcc + coeff], int8.
 *
 * BOTH HR AND RR FROM ONE CAPTURE — why it is not done here:
 *   Heart and lung use DIFFERENT MFCC front-ends (heart: /4 -> 2 kHz,
 *   10-200 Hz bandpass, 665x25; lung: /2 -> 4 kHz, 100-1000 Hz, 324x26),
 *   and dsp_mfcc.c uses ONE shared FFT scratch, so the two pipelines
 *   cannot run concurrently. Doing both from a single capture would mean
 *   buffering all 160 KB of 8 kHz audio and processing it twice; that
 *   160 KB plus the ~80 KB arena does not fit alongside the BLE stack on
 *   the nRF52840 (256 KB). The previous SD-card build (v7.5) buffered
 *   audio.pcm to SD precisely to enable both. BLE-only -> one organ per
 *   recording (HEART then LUNG). If both-in-one is required, add SD back.
 * ==================================================================
 */