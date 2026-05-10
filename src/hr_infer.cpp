/*
 * hr_infer.cpp — Heart-Rate TFLite Micro inference
 * =================================================
 * Integrates HR_trial_144_int8.tflite into the AcoustEEEcare firmware.
 *
 * This file is C++ because TFLite Micro's public API is C++.
 * Your main.c calls it through the plain-C hr_infer.h interface.
 *
 * BUILD NOTE:
 *   Add to CMakeLists.txt:
 *     target_sources(app PRIVATE src/hr_infer.cpp)
 *   Zephyr's TFLM module handles include paths when
 *   CONFIG_TENSORFLOW_LITE_MICRO=y is set in prj.conf.
 */

#include "hr_infer.h"
#include "hr_model_data.h"   /* alignas(8) uint8_t hr_model_tflite[] */

/* TFLite Micro headers — provided by Zephyr's tflite-micro module */
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <zephyr/logging/log.h>
#include <string.h>

LOG_MODULE_REGISTER(hr_infer);

/* ── Tensor arena ──────────────────────────────────────────────────
 * Start at 80 KB. After the first successful inference, read the
 * "Arena used: XXXXX bytes" log line printed by hr_infer_arena_used()
 * in main.c, then set this to that value + 4096 and rebuild.
 *
 * nRF52840 has 256 KB RAM total. With the rest of the firmware using
 * ~155 KB (kernel + BLE + DSP + MFCC buffer + rings), 80 KB fits
 * at ~235 KB total — just inside the limit. After trimming the arena
 * to the real minimum you will recover 20–35 KB of headroom.
 * ────────────────────────────────────────────────────────────────── */
#define TENSOR_ARENA_SIZE  (90 * 1024)

/* Static storage — no heap needed */
static uint8_t tensor_arena[TENSOR_ARENA_SIZE] __attribute__((aligned(16)));

/* Resolver: only the 4 op types used by this model */
using HRResolver = tflite::MicroMutableOpResolver<4>;
static HRResolver resolver;

/* Interpreter in static buffer — avoids heap/new */
alignas(tflite::MicroInterpreter)
static uint8_t interpreter_buf[sizeof(tflite::MicroInterpreter)];
static tflite::MicroInterpreter *interpreter = nullptr;

static const tflite::Model *model_ptr = nullptr;
static bool initialised = false;

/* ── Quantization helpers ──────────────────────────────────────── */

/*
 * float_to_int8_input() — quantize one float MFCC coefficient to INT8.
 * Formula: q = round(f / scale) + zero_point, clamped to [-128, 127].
 */
static inline int8_t float_to_int8_input(float f)
{
    float q = (f / HR_INPUT_SCALE) + (float)HR_INPUT_ZP;
    int32_t qi = (int32_t)(q >= 0.0f ? (q + 0.5f) : (q - 0.5f));
    if (qi < -128) qi = -128;
    if (qi >  127) qi =  127;
    return (int8_t)qi;
}

/*
 * int8_output_to_bpm() — dequantize the single INT8 output to BPM.
 * Params from model output quantization: scale=0.474156, zp=-128.
 * Range: 0.0 BPM (raw=-128) to ~120.9 BPM (raw=+127).
 */
static inline float int8_output_to_bpm(int8_t raw)
{
    return (float)((int32_t)raw - HR_OUTPUT_ZP) * HR_OUTPUT_SCALE;
}

/* ══════════════════════════════════════════════════════════════════
 * hr_infer_init()
 * ══════════════════════════════════════════════════════════════════ */
int hr_infer_init(void)
{
    tflite::InitializeTarget();

    model_ptr = tflite::GetModel(hr_model_tflite);
    if (model_ptr->version() != TFLITE_SCHEMA_VERSION) {
        LOG_ERR("HR model schema mismatch: model=%u runtime=%u",
                model_ptr->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    /*
     * Register ONLY the 4 ops this model uses.
     * Ops confirmed by flatbuffers inspection of HR_trial_144_int8.tflite:
     *   builtin_code 3  → CONV_2D
     *   builtin_code 17 → MAX_POOL_2D
     *   builtin_code 40 → MEAN  (used as GlobalAveragePooling)
     *   builtin_code 9  → FULLY_CONNECTED
     */
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddFullyConnected();

    /* Placement-new into static buffer — no heap allocation */
    interpreter = new (interpreter_buf)
        tflite::MicroInterpreter(model_ptr, resolver,
                                 tensor_arena, TENSOR_ARENA_SIZE);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG_ERR("HR model: AllocateTensors() failed — "
                "increase TENSOR_ARENA_SIZE (currently %d KB)",
                TENSOR_ARENA_SIZE / 1024);
        return -1;
    }

    /* Verify input/output shapes match our expectations */
    TfLiteTensor *in  = interpreter->input(0);
    TfLiteTensor *out = interpreter->output(0);

    bool shape_ok = (in->dims->size == 4)
                 && (in->dims->data[1] == HR_MODEL_FRAMES)
                 && (in->dims->data[2] == HR_MODEL_MFCC)
                 && (in->dims->data[3] == 1)
                 && (in->type == kTfLiteInt8)
                 && (out->dims->data[1] == 1)
                 && (out->type == kTfLiteInt8);

    if (!shape_ok) {
        LOG_ERR("HR model: unexpected tensor shape or type");
        LOG_ERR("  input  dims=%d [%d,%d,%d,%d] type=%d",
                in->dims->size,
                in->dims->data[0], in->dims->data[1],
                in->dims->data[2], in->dims->data[3],
                in->type);
        LOG_ERR("  output dims=%d [%d,%d] type=%d",
                out->dims->size,
                out->dims->data[0], out->dims->data[1],
                out->type);
        return -1;
    }

    initialised = true;
    LOG_INF("HR TFLite model ready | arena=%d KB | "
            "input=[1,%d,%d,1] INT8 | output=[1,1] INT8 | "
            "BPM range: 0.0–%.1f",
            TENSOR_ARENA_SIZE / 1024,
            HR_MODEL_FRAMES, HR_MODEL_MFCC,
            (double)int8_output_to_bpm(127));

    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * hr_infer_arena_used()
 *
 * Returns the number of tensor arena bytes actually used after
 * AllocateTensors() + Invoke(). Call this from main.c after a
 * successful hr_infer_run_int8() to find the real minimum arena size.
 * ══════════════════════════════════════════════════════════════════ */
uint32_t hr_infer_arena_used(void)
{
    if (!initialised || !interpreter) {
        return 0;
    }
    return (uint32_t)interpreter->arena_used_bytes();
}

/* ══════════════════════════════════════════════════════════════════
 * hr_infer_run()  — float input path
 * ══════════════════════════════════════════════════════════════════ */
int hr_infer_run(const float *mfcc_flat, int n_frames, hr_result_t *out)
{
    if (!initialised || !interpreter || !out) {
        LOG_ERR("hr_infer_run: not initialised");
        return -1;
    }

    TfLiteTensor *input_tensor = interpreter->input(0);
    int8_t *input_data = input_tensor->data.int8;

    int frames_to_copy = n_frames < HR_MODEL_FRAMES ? n_frames : HR_MODEL_FRAMES;

    for (int f = 0; f < frames_to_copy; f++) {
        for (int c = 0; c < HR_MODEL_MFCC; c++) {
            input_data[f * HR_MODEL_MFCC + c] =
                float_to_int8_input(mfcc_flat[f * HR_MODEL_MFCC + c]);
        }
    }

    /* Zero-pad remaining frames (0.0 float → HR_INPUT_ZP in INT8) */
    if (frames_to_copy < HR_MODEL_FRAMES) {
        int pad_start = frames_to_copy * HR_MODEL_MFCC;
        int pad_count = (HR_MODEL_FRAMES - frames_to_copy) * HR_MODEL_MFCC;
        memset(&input_data[pad_start], HR_INPUT_ZP, pad_count);
        LOG_WRN("hr_infer_run: only %d/%d frames — padded %d with silence",
                frames_to_copy, HR_MODEL_FRAMES,
                HR_MODEL_FRAMES - frames_to_copy);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        LOG_ERR("hr_infer_run: Invoke() failed");
        return -1;
    }

    TfLiteTensor *output_tensor = interpreter->output(0);
    int8_t raw = output_tensor->data.int8[0];

    out->raw_output = raw;
    out->bpm        = (double)int8_output_to_bpm(raw);

    LOG_INF("HR inference: raw_int8=%d → %.1f BPM", (int)raw, (double)out->bpm);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════
 * hr_infer_run_int8()  — pre-quantized INT8 input path (preferred)
 *
 * main.c fills s_mfcc_flat_q8[] frame-by-frame during on_mfcc_frame(),
 * then passes the complete buffer here. We memcpy it straight into the
 * input tensor — no float conversion needed.
 * ══════════════════════════════════════════════════════════════════ */
int hr_infer_run_int8(const int8_t *mfcc_q8, int n_frames, hr_result_t *out)
{
    if (!initialised || !interpreter || !out) {
        LOG_ERR("hr_infer_run_int8: not initialised");
        return -1;
    }

    TfLiteTensor *input_tensor = interpreter->input(0);
    int8_t *input_data = input_tensor->data.int8;

    int frames_to_copy = n_frames < HR_MODEL_FRAMES ? n_frames : HR_MODEL_FRAMES;

    /* Copy pre-quantized data directly — no float conversion needed */
    memcpy(input_data,
           mfcc_q8,
           (size_t)frames_to_copy * HR_MODEL_MFCC * sizeof(int8_t));

    /* Zero-pad remaining frames using zero_point to represent 0.0 float */
    if (frames_to_copy < HR_MODEL_FRAMES) {
        int pad_start = frames_to_copy * HR_MODEL_MFCC;
        int pad_count = (HR_MODEL_FRAMES - frames_to_copy) * HR_MODEL_MFCC;
        memset(&input_data[pad_start], (uint8_t)HR_INPUT_ZP, pad_count);
        LOG_WRN("hr_infer_run_int8: padded %d frames with silence",
                HR_MODEL_FRAMES - frames_to_copy);
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        LOG_ERR("hr_infer_run_int8: Invoke() failed");
        return -1;
    }

    TfLiteTensor *output_tensor = interpreter->output(0);
    int8_t raw = output_tensor->data.int8[0];

    out->raw_output = raw;
    out->bpm        = (double)int8_output_to_bpm(raw);

    LOG_INF("HR inference (int8): raw=%d → %.1f BPM", (int)raw, (double)out->bpm);
    return 0;
}