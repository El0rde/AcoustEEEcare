/*
 * tflm_inference.cc — TFLite Micro heart + lung inference (v3.2, BLE-only)
 * ========================================================================
 * Pre-quantized int8 features in, dequantized + clipped scalar out.
 * No SD / no filesystem. See tflm_inference.h for the contract.
 *
 * Models (verified): best_mcu_heart_int8.tflite / best_mcu_lung_int8.tflite
 *   HEART input [1,665,25,1] int8  output [1,1] int8  → HR, clip [40,180]
 *   LUNG  input [1,324,26,1] int8  output [1,1] int8  → RR, clip [6,50]
 *   Ops (both): CONV_2D, MAX_POOL_2D, MEAN, FULLY_CONNECTED.
 */

#include <string.h>
#include <errno.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tflm_inference.h"
#include "heart_model.h"
#include "lung_model.h"

LOG_MODULE_REGISTER(tflm_inference, LOG_LEVEL_INF);

#define HEART_FRAMES 665
#define HEART_NMFCC   25
#define LUNG_FRAMES  324
#define LUNG_NMFCC    26

#define HEART_BPM_MIN  40.0f
#define HEART_BPM_MAX 180.0f
#define LUNG_RR_MIN     6.0f
#define LUNG_RR_MAX    50.0f

extern "C" volatile uint32_t g_tflm_arena_used_bytes;

/* Resolver: exactly the four ops both models use. */
using OpResolver = tflite::MicroMutableOpResolver<4>;
static int build_resolver(OpResolver &r)
{
    if (r.AddConv2D()         != kTfLiteOk) return -1;
    if (r.AddMaxPool2D()      != kTfLiteOk) return -1;
    if (r.AddMean()           != kTfLiteOk) return -1;
    if (r.AddFullyConnected() != kTfLiteOk) return -1;
    return 0;
}

static const uint8_t *model_for(int organ)
{
    return (organ == 1) ? g_lung_model_data : g_heart_model_data;
}

/* ── Probe input quantization (boot-time) ────────────────────────────── */
extern "C"
int tflm_get_input_quant(int organ, uint8_t *arena, size_t arena_bytes,
                         float *scale, int32_t *zero_point)
{
    const tflite::Model *model = tflite::GetModel(model_for(organ));
    if (model->version() != TFLITE_SCHEMA_VERSION) return -EINVAL;

    OpResolver resolver;
    if (build_resolver(resolver) != 0) return -ENOTSUP;

    tflite::MicroInterpreter interpreter(model, resolver, arena, arena_bytes);
    if (interpreter.AllocateTensors() != kTfLiteOk) return -ENOMEM;

    TfLiteTensor *input = interpreter.input(0);
    if (input == nullptr || input->type != kTfLiteInt8) return -EINVAL;

    *scale      = input->params.scale;
    *zero_point = input->params.zero_point;
    return 0;
}

/* ── Core int8 inference ─────────────────────────────────────────────── */
static int run_ram(int organ, const char *name,
                   uint8_t *arena, size_t arena_bytes,
                   const int8_t *features, int n_frames, int n_mfcc,
                   int exp_frames, int exp_mfcc, float *out_value)
{
    if (n_frames != exp_frames || n_mfcc != exp_mfcc) {
        LOG_ERR("%s: shape fw=[%d,%d] model=[%d,%d]",
                name, n_frames, n_mfcc, exp_frames, exp_mfcc);
        return -EINVAL;
    }

    const tflite::Model *model = tflite::GetModel(model_for(organ));
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        LOG_ERR("%s: bad schema version", name);
        return -EINVAL;
    }

    OpResolver resolver;
    if (build_resolver(resolver) != 0) { LOG_ERR("%s: resolver", name); return -ENOTSUP; }

    tflite::MicroInterpreter interpreter(model, resolver, arena, arena_bytes);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        LOG_ERR("%s: AllocateTensors failed — arena too small? (%u B)",
                name, (unsigned)arena_bytes);
        return -ENOMEM;
    }

    const size_t used = interpreter.arena_used_bytes();
    g_tflm_arena_used_bytes = (uint32_t)used;
    LOG_INF("%s: arena used = %u / %u B", name, (unsigned)used, (unsigned)arena_bytes);

    TfLiteTensor *input = interpreter.input(0);
    if (input == nullptr || input->type != kTfLiteInt8) {
        LOG_ERR("%s: bad input tensor", name);
        return -EINVAL;
    }
    const size_t need = (size_t)n_frames * (size_t)n_mfcc;
    if (input->bytes != need) {
        LOG_ERR("%s: input bytes %u != %u", name, (unsigned)input->bytes, (unsigned)need);
        return -EINVAL;
    }

    /* Features are already quantized with this tensor's scale/zp. */
    memcpy(input->data.int8, features, need);

    int64_t t0 = k_uptime_get();
    if (interpreter.Invoke() != kTfLiteOk) { LOG_ERR("%s: Invoke failed", name); return -EIO; }
    LOG_INF("%s: inference %lld ms", name, k_uptime_delta(&t0));

    TfLiteTensor *output = interpreter.output(0);
    if (output == nullptr) return -EINVAL;

    float v;
    if (output->type == kTfLiteInt8) {
        v = ((float)((int32_t)output->data.int8[0] - output->params.zero_point))
            * output->params.scale;
    } else if (output->type == kTfLiteFloat32) {
        v = output->data.f[0];
    } else {
        LOG_ERR("%s: bad output type %d", name, output->type);
        return -EINVAL;
    }
    *out_value = v;
    LOG_INF("%s: raw output = %.3f", name, (double)v);
    return 0;
}

/* ── Public API ──────────────────────────────────────────────────────── */
extern "C"
void run_heart_inference_ram(uint8_t *arena, size_t arena_bytes,
                             const int8_t *features, int n_frames, int n_mfcc,
                             heart_result_t *result)
{
    memset(result, 0, sizeof(*result));
    result->class_idx = -1; result->confidence = 1.0f;
    float v = 0.0f;
    result->rc = run_ram(0, "heart", arena, arena_bytes, features,
                         n_frames, n_mfcc, HEART_FRAMES, HEART_NMFCC, &v);
    if (result->rc == 0) {
        if (v < HEART_BPM_MIN) v = HEART_BPM_MIN;
        if (v > HEART_BPM_MAX) v = HEART_BPM_MAX;
        result->value = v;
        LOG_INF("heart: HR = %.1f BPM", (double)v);
    }
}

extern "C"
void run_lung_inference_ram(uint8_t *arena, size_t arena_bytes,
                            const int8_t *features, int n_frames, int n_mfcc,
                            lung_result_t *result)
{
    memset(result, 0, sizeof(*result));
    result->class_idx = -1; result->confidence = 1.0f;
    float v = 0.0f;
    result->rc = run_ram(1, "lung", arena, arena_bytes, features,
                         n_frames, n_mfcc, LUNG_FRAMES, LUNG_NMFCC, &v);
    if (result->rc == 0) {
        if (v < LUNG_RR_MIN) v = LUNG_RR_MIN;
        if (v > LUNG_RR_MAX) v = LUNG_RR_MAX;
        result->value = v;
        LOG_INF("lung: RR = %.1f BPM", (double)v);
    }
}
