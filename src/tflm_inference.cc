/*
 * tflm_inference.cc — TFLite Micro heart + lung inference
 * =========================================================
 * v2.2 — lung shape validation + arena-used export.
 *
 * Changes from v2.1:
 *   - LUNG_EXPECTED_FRAMES / LUNG_EXPECTED_N_MFCC defined so the
 *     existing shape validation in run_inference() actually
 *     compares against the model's expected shape (was passing
 *     n_frames/n_mfcc as their own expected values — no-op).
 *   - g_tflm_arena_used_bytes (defined in main.c) is updated after
 *     each successful AllocateTensors() so report_ram_usage() shows
 *     the real arena high-water.
 *
 * HEART MODEL (best_mcu_int8.tflite — trial 59, MAE 3.30 BPM)
 * ─────────────────────────────────────────────────────────
 *   Input:  [1, 665, 25, 1]  int8
 *   Output: [1, 1]            int8  (regression: BPM scalar)
 *
 * LUNG MODEL (best_mcu_lung_int8.tflite, ~28.8 KB)
 * ─────────────────────────────────────────────────
 *   Input:  [1, 324, 26, 1]  int8
 *           (324 = 1 + (40000 - 1200) / 120 from lung_winning_config)
 *   Output: regression scalar (RR in BPM)
 *
 *   NOTE: if AllocateTensors() rejects the input shape, run
 *   `flatc -t schema.fbs -- best_mcu_lung_int8.tflite | jq` and
 *   update LUNG_EXPECTED_FRAMES below to whatever the model has in
 *   its subgraphs[0].tensors[0].shape.
 */

/* ── Standard headers ──
 * Use C-style headers (string.h, stdio.h, math.h) rather than their
 * C++ <cstring>/<cstdio>/<cmath> wrappers, because Zephyr's minimal
 * C++ library does not provide the <c*> headers. */
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>

/* ── Zephyr / POSIX headers ── */
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/fs/fs.h>

/* ── TFLite Micro headers ── */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* ── Our own header (C-linkage declarations) ── */
#include "tflm_inference.h"

/* ── Compiled-in model data ── */
#include "heart_model.h"

#if defined(ENABLE_LUNG_MODEL) && ENABLE_LUNG_MODEL
#include "lung_model.h"
#endif

LOG_MODULE_REGISTER(tflm_inference, LOG_LEVEL_INF);

/* ══════════════════════════════════════════════════════════════════
 * Heart model expected input shape -- validated at runtime
 * (best_mcu_int8 model: 665 frames × 25 MFCC)
 * ══════════════════════════════════════════════════════════════════ */
#define HEART_EXPECTED_FRAMES   665
#define HEART_EXPECTED_N_MFCC   25

/* ══════════════════════════════════════════════════════════════════
 * Lung model expected input shape -- v2.2
 * Derived from lung_winning_config (frame=300 ms / hop=10% / fft=2048):
 *   n_frames = 1 + (40000 - 1200) / 120 = 324
 *   n_mfcc   = 26
 * If the model itself disagrees, AllocateTensors will fail with a
 * clear dims mismatch message and you can adjust either the config
 * or these macros to match the trained model.
 * ══════════════════════════════════════════════════════════════════ */
#define LUNG_EXPECTED_FRAMES    324
#define LUNG_EXPECTED_N_MFCC    26

/* SD read chunk size, in FLOATS. */
#define MFCC_READ_CHUNK_FLOATS  256

/* v2.2: arena high-water exported so main.c can report it.
 * Defined in main.c, declared here as volatile (writer is C++, reader
 * is C, so the volatile is paranoia against LTO reordering). */
extern "C" volatile uint32_t g_tflm_arena_used_bytes;

/* ══════════════════════════════════════════════════════════════════
 * FORWARD DECLARATIONS (so we can keep public API at the bottom).
 * ══════════════════════════════════════════════════════════════════ */
static int   load_and_quantize_mfcc(const char *path, TfLiteTensor *input,
                                    int n_frames, int n_mfcc, const char *tag);
static float dequantize_output_scalar(TfLiteTensor *output);
static int   write_result_file(const char *path, const char *text);

/* ══════════════════════════════════════════════════════════════════
 * Helper: load + quantize MFCC float data from SD directly into the
 * model's int8 input tensor.
 *
 * Layout invariants:
 *   - Input tensor shape is [1, n_frames, n_mfcc, 1] NHWC int8.
 *   - With C=1, memory layout is identical to row-major
 *     [frame][mfcc], so we write contiguously without any
 *     transpose/stride work.
 *   - On-disk float MFCC data is also row-major [frame][mfcc].
 *
 * Returns 0 on success, negative errno on failure.
 * ══════════════════════════════════════════════════════════════════ */
static int load_and_quantize_mfcc(const char   *path,
                                  TfLiteTensor *input,
                                  int           n_frames,
                                  int           n_mfcc,
                                  const char   *tag)
{
    /* Sanity: required input element count. */
    const size_t expected_elements = (size_t)n_frames * (size_t)n_mfcc;
    const size_t input_elements    = input->bytes;  /* int8 => 1 byte each */

    if (input_elements != expected_elements) {
        LOG_ERR("%s: input tensor has %u elements, expected %u",
                tag, (unsigned)input_elements, (unsigned)expected_elements);
        return -EINVAL;
    }
    if (input->type != kTfLiteInt8) {
        LOG_ERR("%s: input tensor is not int8 (type=%d)", tag, input->type);
        return -EINVAL;
    }

    const float   scale = input->params.scale;
    const int32_t zp    = input->params.zero_point;
    int8_t       *dst   = input->data.int8;

    LOG_INF("%s: quantizing MFCC into input (scale=%.6f zp=%d)",
            tag, (double)scale, (int)zp);

    /* Open the .f32 file on SD. */
    struct fs_file_t f;
    fs_file_t_init(&f);
    int rc = fs_open(&f, path, FS_O_READ);
    if (rc < 0) {
        LOG_ERR("%s: fs_open(%s) failed: %d", tag, path, rc);
        return rc;
    }

    /* ── 4-byte aligned stack scratch buffer ── */
    float    chunk[MFCC_READ_CHUNK_FLOATS] __aligned(4);
    size_t   elements_done = 0;

    while (elements_done < expected_elements) {
        size_t want_elements = expected_elements - elements_done;
        if (want_elements > MFCC_READ_CHUNK_FLOATS) {
            want_elements = MFCC_READ_CHUNK_FLOATS;
        }

        size_t want_bytes = want_elements * sizeof(float);
        ssize_t got_bytes = fs_read(&f, chunk, want_bytes);
        if (got_bytes <= 0) {
            LOG_ERR("%s: short read at element %u (got=%d, want=%u)",
                    tag, (unsigned)elements_done, (int)got_bytes,
                    (unsigned)want_bytes);
            /* Quantized-zero fill the rest so the input tensor is at
             * least valid (model output will be wrong, but no crash). */
            int8_t qzero = (int8_t)((-zp) > 127 ? 127 :
                                    ((-zp) < -128 ? -128 : -zp));
            memset(dst + elements_done, qzero,
                   expected_elements - elements_done);
            fs_close(&f);
            return -EIO;
        }

        size_t got_elements = (size_t)got_bytes / sizeof(float);

        for (size_t i = 0; i < got_elements; i++) {
            int32_t q = (int32_t)lroundf(chunk[i] / scale) + zp;
            if (q < -128) q = -128;
            if (q >  127) q =  127;
            dst[elements_done + i] = (int8_t)q;
        }
        elements_done += got_elements;

        /* Yield occasionally so BLE stack keeps running. */
        if ((elements_done & 0x3FF) == 0) {
            k_yield();
        }
    }

    fs_close(&f);
    LOG_INF("%s: loaded + quantized %u elements", tag, (unsigned)elements_done);

    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Output dequantization helper for the regression case.
 * Output tensor is int8 scalar -> float dequantized value.
 * ══════════════════════════════════════════════════════════════════ */
static float dequantize_output_scalar(TfLiteTensor *output)
{
    if (output->type == kTfLiteInt8) {
        int8_t   q     = output->data.int8[0];
        float    scale = output->params.scale;
        int32_t  zp    = output->params.zero_point;
        return ((float)((int32_t)q - zp)) * scale;
    }
    if (output->type == kTfLiteFloat32) {
        return output->data.f[0];
    }
    LOG_ERR("Unsupported output tensor type: %d", output->type);
    return NAN;
}

/* ══════════════════════════════════════════════════════════════════
 * Write a small text result file to SD.
 * ══════════════════════════════════════════════════════════════════ */
static int write_result_file(const char *path, const char *text)
{
    struct fs_file_t f;
    fs_file_t_init(&f);

    int rc = fs_open(&f, path, FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc < 0) {
        LOG_ERR("write_result_file: fs_open(%s) failed: %d", path, rc);
        return rc;
    }

    ssize_t w = fs_write(&f, text, strlen(text));
    fs_close(&f);

    if (w < 0) {
        LOG_ERR("write_result_file: fs_write(%s) failed: %d", path, (int)w);
        return (int)w;
    }

    LOG_INF("write_result_file: wrote \"%s\" to %s", text, path);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Generic inference runner — heart and lung share this body.
 *
 *   model_data       compiled-in .tflite flatbuffer (in flash)
 *   model_name       "heart" or "lung" for logs
 *   arena/arena_bytes  shared tensor arena
 *   mfcc_path        SD path to .f32 file
 *   n_frames/n_mfcc  shape produced by the offline MFCC pass
 *   exp_frames/exp_mfcc  shape the model was trained on
 *   out_value        filled with dequantized output scalar
 *
 * Returns 0 on success, negative errno on failure.
 * ══════════════════════════════════════════════════════════════════ */
static int run_inference(const uint8_t *model_data,
                         const char    *model_name,
                         uint8_t       *arena,
                         size_t         arena_bytes,
                         const char    *mfcc_path,
                         int            n_frames,
                         int            n_mfcc,
                         int            exp_frames,
                         int            exp_mfcc,
                         float         *out_value)
{
    /* ── 1. Validate firmware MFCC shape matches model ── */
    if (n_mfcc != exp_mfcc) {
        LOG_ERR("%s: n_mfcc mismatch -- firmware=%d, model=%d. "
                "Regenerate %s MFCC tables with num_mfcc=%d.",
                model_name, n_mfcc, exp_mfcc, model_name, exp_mfcc);
        return -EINVAL;
    }
    if (n_frames != exp_frames) {
        LOG_ERR("%s: n_frames mismatch -- firmware=%d, model=%d. "
                "MFCC config has wrong hop/frame settings.",
                model_name, n_frames, exp_frames);
        return -EINVAL;
    }

    /* ── 2. Load and verify model flatbuffer ── */
    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        LOG_ERR("%s: schema version %u (firmware expects %d)",
                model_name, (unsigned)model->version(),
                (int)TFLITE_SCHEMA_VERSION);
        return -EINVAL;
    }

    /* ── 3. Build resolver INLINE; some TFLM versions hold internal
     * pointers into the resolver that get invalidated on move/copy. ── */
    tflite::MicroMutableOpResolver<8> resolver;
    if (resolver.AddConv2D()         != kTfLiteOk ||
        resolver.AddMaxPool2D()      != kTfLiteOk ||
        resolver.AddMean()           != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk) {
        LOG_ERR("%s: op resolver setup failed", model_name);
        return -ENOTSUP;
    }
    /* Spare slots (resolver size = 8) reserved for future ops
     * the lung model might need (e.g. AddReshape, AddSoftmax). */

    /* ── 4. Build interpreter on the shared arena ── */
    tflite::MicroInterpreter interpreter(model, resolver, arena, arena_bytes);

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        LOG_ERR("%s: AllocateTensors failed -- arena too small? "
                "(arena=%u B). Increase TENSOR_ARENA_BYTES.",
                model_name, (unsigned)arena_bytes);
        return -ENOMEM;
    }

    const size_t arena_used = interpreter.arena_used_bytes();
    g_tflm_arena_used_bytes = (uint32_t)arena_used;   /* v2.2: export */
    LOG_INF("%s: AllocateTensors OK -- arena used = %u / %u bytes",
            model_name, (unsigned)arena_used, (unsigned)arena_bytes);

    /* ── 5. Get input tensor; validate it matches what we expect ── */
    TfLiteTensor *input = interpreter.input(0);
    if (input == nullptr) {
        LOG_ERR("%s: null input tensor", model_name);
        return -EINVAL;
    }

    /* Verify tensor dims line up with what the firmware will provide.
     * Expected: [1, n_frames, n_mfcc, 1]. */
    if (input->dims->size != 4 ||
        input->dims->data[0] != 1 ||
        input->dims->data[1] != exp_frames ||
        input->dims->data[2] != exp_mfcc ||
        input->dims->data[3] != 1) {
        LOG_ERR("%s: input dims [%d,%d,%d,%d] do not match expected "
                "[1,%d,%d,1]",
                model_name,
                input->dims->size > 0 ? input->dims->data[0] : -1,
                input->dims->size > 1 ? input->dims->data[1] : -1,
                input->dims->size > 2 ? input->dims->data[2] : -1,
                input->dims->size > 3 ? input->dims->data[3] : -1,
                exp_frames, exp_mfcc);
        return -EINVAL;
    }

    /* ── 6. Stream-quantize MFCC from SD straight into input tensor ── */
    int rc = load_and_quantize_mfcc(mfcc_path, input,
                                    n_frames, n_mfcc, model_name);
    if (rc < 0) {
        return rc;
    }

    /* ── 7. Run inference ── */
    int64_t t0 = k_uptime_get();
    TfLiteStatus invoke_status = interpreter.Invoke();
    int64_t elapsed_ms = k_uptime_delta(&t0);

    if (invoke_status != kTfLiteOk) {
        LOG_ERR("%s: Invoke() failed (status=%d)", model_name,
                (int)invoke_status);
        return -EIO;
    }
    LOG_INF("%s: inference done in %lld ms", model_name, elapsed_ms);

    /* ── 8. Extract regression output ── */
    TfLiteTensor *output = interpreter.output(0);
    if (output == nullptr) {
        LOG_ERR("%s: null output tensor", model_name);
        return -EINVAL;
    }

    *out_value = dequantize_output_scalar(output);
    LOG_INF("%s: output dequantized = %.3f", model_name, (double)*out_value);

    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Public API — heart
 * ══════════════════════════════════════════════════════════════════ */
extern "C"
void run_heart_inference(uint8_t       *arena,
                         size_t         arena_bytes,
                         const char    *mfcc_path,
                         int            n_frames,
                         int            n_mfcc,
                         heart_result_t *result)
{
    memset(result, 0, sizeof(*result));
    result->class_idx  = -1;          /* regression: no class */
    result->confidence = 1.0f;        /* regression has no confidence */

    result->rc = run_inference(g_heart_model_data,
                               "heart",
                               arena, arena_bytes,
                               mfcc_path,
                               n_frames, n_mfcc,
                               HEART_EXPECTED_FRAMES, HEART_EXPECTED_N_MFCC,
                               &result->value);

    if (result->rc == 0) {
        char txt[32];
        snprintf(txt, sizeof(txt), "HR:%.0f\n", (double)result->value);
        int wrc = write_result_file("/SD:/hr.txt", txt);
        if (wrc < 0) {
            LOG_WRN("heart: failed to write hr.txt (%d)", wrc);
            /* Non-fatal -- result struct is still valid. */
        }
    }
}

/* ══════════════════════════════════════════════════════════════════
 * Public API — lung
 *
 * Stub for now -- returns -ENOTSUP until ENABLE_LUNG_MODEL is set
 * and the lung model is integrated.
 * ══════════════════════════════════════════════════════════════════ */
extern "C"
void run_lung_inference(uint8_t      *arena,
                        size_t        arena_bytes,
                        const char   *mfcc_path,
                        int           n_frames,
                        int           n_mfcc,
                        lung_result_t *result)
{
    memset(result, 0, sizeof(*result));
    result->class_idx  = -1;
    result->confidence = 1.0f;

#if defined(ENABLE_LUNG_MODEL) && ENABLE_LUNG_MODEL
    /* v2.2: pass real expected dims so the shape validation in
     * run_inference() actually catches mismatches. */
    result->rc = run_inference(g_lung_model_data,
                               "lung",
                               arena, arena_bytes,
                               mfcc_path,
                               n_frames, n_mfcc,
                               LUNG_EXPECTED_FRAMES, LUNG_EXPECTED_N_MFCC,
                               &result->value);

    if (result->rc == 0) {
        char txt[32];
        snprintf(txt, sizeof(txt), "RR:%.0f\n", (double)result->value);
        int wrc = write_result_file("/SD:/rr.txt", txt);
        if (wrc < 0) {
            LOG_WRN("lung: failed to write rr.txt (%d)", wrc);
        }
    }
#else
    /* Lung model not yet integrated. */
    (void)arena; (void)arena_bytes; (void)mfcc_path;
    (void)n_frames; (void)n_mfcc;
    result->rc = -ENOTSUP;
    LOG_INF("lung inference skipped (ENABLE_LUNG_MODEL=0)");
#endif
}