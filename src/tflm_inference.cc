/*
 * tflm_inference.cc — TFLite Micro heart + lung inference
 * =========================================================
 * v3.1 — merged v2.2 (SD path) + v3.0 (BLE/SD-free additions).
 *
 * Added over v2.2:
 *   - get_heart_quant_params() / get_lung_quant_params()
 *     Required by main.c v8.1: called once at boot so the host
 *     knows the model's input scale/zero_point and can correctly
 *     quantize float32 MFCC → int8 before sending over BLE.
 *   - run_heart_inference() / run_lung_inference() signatures
 *     updated to accept pre-quantized int8 pointer (BLE path)
 *     OR the SD mfcc_path string (legacy path).
 *     Both are kept so this file compiles against either main.c.
 *
 * NOTE: get_*_quant_params() instantiates a MicroInterpreter on the
 * shared arena just long enough to call AllocateTensors() and read
 * the input tensor's params.  It tears down before returning.
 * Call it ONCE per connection, before any recording starts.
 */

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <errno.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/fs/fs.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tflm_inference.h"
#include "heart_model.h"

#if defined(ENABLE_LUNG_MODEL) && ENABLE_LUNG_MODEL
#include "lung_model.h"
#endif

LOG_MODULE_REGISTER(tflm_inference, LOG_LEVEL_INF);

#define HEART_EXPECTED_FRAMES   665
#define HEART_EXPECTED_N_MFCC    25
#define LUNG_EXPECTED_FRAMES    324
#define LUNG_EXPECTED_N_MFCC     26

#define MFCC_READ_CHUNK_FLOATS  256

extern "C" volatile uint32_t g_tflm_arena_used_bytes;

/* ══════════════════════════════════════════════════════════════════
 * build_resolver — shared op set for both models
 * ══════════════════════════════════════════════════════════════════ */
static int build_resolver(tflite::MicroMutableOpResolver<8> &resolver,
                          const char *tag)
{
    if (resolver.AddConv2D()         != kTfLiteOk ||
        resolver.AddMaxPool2D()      != kTfLiteOk ||
        resolver.AddMean()           != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk) {
        LOG_ERR("%s: op resolver setup failed", tag);
        return -ENOTSUP;
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * get_quant_params — internal helper
 *
 * Allocates tensors on the arena, reads input->params, tears down.
 * ══════════════════════════════════════════════════════════════════ */
static int get_quant_params(const uint8_t *model_data,
                            const char    *tag,
                            uint8_t       *arena,
                            size_t         arena_bytes,
                            float         *out_scale,
                            int32_t       *out_zp)
{
    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        LOG_ERR("%s: schema version mismatch", tag);
        return -EINVAL;
    }

    tflite::MicroMutableOpResolver<8> resolver;
    if (build_resolver(resolver, tag) != 0) return -ENOTSUP;

    tflite::MicroInterpreter interpreter(model, resolver, arena, arena_bytes);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        LOG_ERR("%s: AllocateTensors failed in get_quant_params", tag);
        return -ENOMEM;
    }

    TfLiteTensor *input = interpreter.input(0);
    if (!input) { LOG_ERR("%s: null input tensor", tag); return -EINVAL; }

    *out_scale = input->params.scale;
    *out_zp    = input->params.zero_point;
    LOG_INF("%s quant params: scale=%.6f zp=%d",
            tag, (double)*out_scale, (int)*out_zp);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Public: get_heart_quant_params / get_lung_quant_params
 * Called by main.c once after BLE connect to send scale+zp to host.
 * ══════════════════════════════════════════════════════════════════ */
extern "C"
int get_heart_quant_params(uint8_t *arena, size_t arena_bytes,
                           float *out_scale, int32_t *out_zp)
{
    return get_quant_params(g_heart_model_data, "heart",
                            arena, arena_bytes, out_scale, out_zp);
}

extern "C"
int get_lung_quant_params(uint8_t *arena, size_t arena_bytes,
                          float *out_scale, int32_t *out_zp)
{
#if defined(ENABLE_LUNG_MODEL) && ENABLE_LUNG_MODEL
    return get_quant_params(g_lung_model_data, "lung",
                            arena, arena_bytes, out_scale, out_zp);
#else
    (void)arena; (void)arena_bytes; (void)out_scale; (void)out_zp;
    return -ENOTSUP;
#endif
}

/* ══════════════════════════════════════════════════════════════════
 * SD helper: load + quantize .f32 file into input tensor (v2.2 path)
 * ══════════════════════════════════════════════════════════════════ */
static int load_and_quantize_mfcc(const char   *path,
                                  TfLiteTensor *input,
                                  int           n_frames,
                                  int           n_mfcc,
                                  const char   *tag)
{
    const size_t expected_elements = (size_t)n_frames * (size_t)n_mfcc;
    const size_t input_elements    = input->bytes;

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

    LOG_INF("%s: quantizing MFCC from SD (scale=%.6f zp=%d)",
            tag, (double)scale, (int)zp);

    struct fs_file_t f;
    fs_file_t_init(&f);
    int rc = fs_open(&f, path, FS_O_READ);
    if (rc < 0) {
        LOG_ERR("%s: fs_open(%s) failed: %d", tag, path, rc);
        return rc;
    }

    float  chunk[MFCC_READ_CHUNK_FLOATS] __aligned(4);
    size_t elements_done = 0;

    while (elements_done < expected_elements) {
        size_t want_elements = expected_elements - elements_done;
        if (want_elements > MFCC_READ_CHUNK_FLOATS) {
            want_elements = MFCC_READ_CHUNK_FLOATS;
        }
        size_t  want_bytes = want_elements * sizeof(float);
        ssize_t got_bytes  = fs_read(&f, chunk, want_bytes);
        if (got_bytes <= 0) {
            LOG_ERR("%s: short read at element %u", tag, (unsigned)elements_done);
            int8_t qzero = (int8_t)((-zp) > 127 ? 127 : ((-zp) < -128 ? -128 : -zp));
            memset(dst + elements_done, qzero, expected_elements - elements_done);
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
        if ((elements_done & 0x3FF) == 0) k_yield();
    }

    fs_close(&f);
    LOG_INF("%s: loaded + quantized %u elements from SD", tag, (unsigned)elements_done);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Output dequantization
 * ══════════════════════════════════════════════════════════════════ */
static float dequantize_output_scalar(TfLiteTensor *output)
{
    if (output->type == kTfLiteInt8) {
        return ((float)((int32_t)output->data.int8[0] -
                        output->params.zero_point)) * output->params.scale;
    }
    if (output->type == kTfLiteFloat32) {
        return output->data.f[0];
    }
    LOG_ERR("Unsupported output type: %d", output->type);
    return NAN;
}

/* ══════════════════════════════════════════════════════════════════
 * write_result_file — SD result persistence (v2.2 path)
 * ══════════════════════════════════════════════════════════════════ */
static int __attribute__((unused)) write_result_file(const char *path, const char *text)
{
    struct fs_file_t f;
    fs_file_t_init(&f);
    int rc = fs_open(&f, path, FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (rc < 0) { LOG_ERR("write_result_file: fs_open(%s) failed: %d", path, rc); return rc; }
    ssize_t w = fs_write(&f, text, strlen(text));
    fs_close(&f);
    if (w < 0) { LOG_ERR("write_result_file: fs_write failed: %d", (int)w); return (int)w; }
    LOG_INF("write_result_file: wrote \"%s\" -> %s", text, path);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Core inference runner
 *
 * Two input modes:
 *   mfcc_int8 != NULL  →  BLE path: copy pre-quantized int8 directly
 *   mfcc_int8 == NULL  →  SD path:  load+quantize from mfcc_path
 *
 * main.c v8.1 always uses the BLE path (mfcc_int8 pointer).
 * ══════════════════════════════════════════════════════════════════ */
static int run_inference(const uint8_t *model_data,
                         const char    *model_name,
                         uint8_t       *arena,
                         size_t         arena_bytes,
                         const int8_t  *mfcc_int8,   /* BLE path, or NULL */
                         const char    *mfcc_path,   /* SD  path, or NULL */
                         int            n_frames,
                         int            n_mfcc,
                         int            exp_frames,
                         int            exp_mfcc,
                         float         *out_value)
{
    if (n_mfcc != exp_mfcc || n_frames != exp_frames) {
        LOG_ERR("%s: shape mismatch got [%d,%d] expected [%d,%d]",
                model_name, n_frames, n_mfcc, exp_frames, exp_mfcc);
        return -EINVAL;
    }

    const tflite::Model *model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        LOG_ERR("%s: schema version mismatch", model_name);
        return -EINVAL;
    }

    tflite::MicroMutableOpResolver<8> resolver;
    if (build_resolver(resolver, model_name) != 0) return -ENOTSUP;

    tflite::MicroInterpreter interpreter(model, resolver, arena, arena_bytes);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        LOG_ERR("%s: AllocateTensors failed (arena=%u B)", model_name,
                (unsigned)arena_bytes);
        return -ENOMEM;
    }

    g_tflm_arena_used_bytes = (uint32_t)interpreter.arena_used_bytes();
    LOG_INF("%s: arena used = %u / %u B", model_name,
            (unsigned)interpreter.arena_used_bytes(), (unsigned)arena_bytes);

    TfLiteTensor *input = interpreter.input(0);
    if (!input) { LOG_ERR("%s: null input tensor", model_name); return -EINVAL; }

    if (input->dims->size != 4            ||
        input->dims->data[0] != 1         ||
        input->dims->data[1] != exp_frames||
        input->dims->data[2] != exp_mfcc  ||
        input->dims->data[3] != 1) {
        LOG_ERR("%s: tensor dims [%d,%d,%d,%d] != expected [1,%d,%d,1]",
                model_name,
                input->dims->data[0], input->dims->data[1],
                input->dims->data[2], input->dims->data[3],
                exp_frames, exp_mfcc);
        return -EINVAL;
    }

    /* ── Fill input tensor ── */
    int rc = 0;
    if (mfcc_int8 != NULL) {
        /* BLE path: pre-quantized int8 from host, copy directly */
        if (input->type != kTfLiteInt8) {
            LOG_ERR("%s: expected int8 input tensor (type=%d)",
                    model_name, input->type);
            return -EINVAL;
        }
        const size_t expected_bytes = (size_t)exp_frames * (size_t)exp_mfcc;
        if (input->bytes != expected_bytes) {
            LOG_ERR("%s: tensor bytes %u != expected %u",
                    model_name, (unsigned)input->bytes, (unsigned)expected_bytes);
            return -EINVAL;
        }
        memcpy(input->data.int8, mfcc_int8, expected_bytes);
    } else if (mfcc_path != NULL) {
        /* SD path: load + quantize float32 file */
        rc = load_and_quantize_mfcc(mfcc_path, input,
                                    n_frames, n_mfcc, model_name);
        if (rc < 0) return rc;
    } else {
        LOG_ERR("%s: both mfcc_int8 and mfcc_path are NULL", model_name);
        return -EINVAL;
    }

    int64_t t0 = k_uptime_get();
    if (interpreter.Invoke() != kTfLiteOk) {
        LOG_ERR("%s: Invoke() failed", model_name);
        return -EIO;
    }
    LOG_INF("%s: inference done in %lld ms", model_name, k_uptime_delta(&t0));

    TfLiteTensor *output = interpreter.output(0);
    if (!output) { LOG_ERR("%s: null output tensor", model_name); return -EINVAL; }
    *out_value = dequantize_output_scalar(output);
    LOG_INF("%s: result = %.3f", model_name, (double)*out_value);

    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * Public API — heart (BLE path, pre-quantized int8)
 * ══════════════════════════════════════════════════════════════════ */
extern "C"
void run_heart_inference(uint8_t        *arena,
                         size_t          arena_bytes,
                         const int8_t   *mfcc_int8,
                         int             n_frames,
                         int             n_mfcc,
                         heart_result_t *result)
{
    memset(result, 0, sizeof(*result));
    result->class_idx  = -1;
    result->confidence = 1.0f;
    result->rc = run_inference(g_heart_model_data, "heart",
                               arena, arena_bytes,
                               mfcc_int8, NULL,
                               n_frames, n_mfcc,
                               HEART_EXPECTED_FRAMES, HEART_EXPECTED_N_MFCC,
                               &result->value);
}

/* ══════════════════════════════════════════════════════════════════
 * Public API — lung (BLE path, pre-quantized int8)
 * ══════════════════════════════════════════════════════════════════ */
extern "C"
void run_lung_inference(uint8_t       *arena,
                        size_t         arena_bytes,
                        const int8_t  *mfcc_int8,
                        int            n_frames,
                        int            n_mfcc,
                        lung_result_t *result)
{
    memset(result, 0, sizeof(*result));
    result->class_idx  = -1;
    result->confidence = 1.0f;

#if defined(ENABLE_LUNG_MODEL) && ENABLE_LUNG_MODEL
    result->rc = run_inference(g_lung_model_data, "lung",
                               arena, arena_bytes,
                               mfcc_int8, NULL,
                               n_frames, n_mfcc,
                               LUNG_EXPECTED_FRAMES, LUNG_EXPECTED_N_MFCC,
                               &result->value);
#else
    (void)arena; (void)arena_bytes; (void)mfcc_int8;
    (void)n_frames; (void)n_mfcc;
    result->rc = -ENOTSUP;
    LOG_INF("lung inference skipped (ENABLE_LUNG_MODEL=0)");
#endif
}