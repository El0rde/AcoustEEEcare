/*
 * tflm_inference.h — TFLite Micro heart + lung inference
 * =========================================================
 * v2.0 — corrected for the actual heart model architecture.
 *
 * Sequential inference on a shared tensor arena:
 *
 *   run_heart_inference(arena, arena_bytes, mfcc_path, n_frames, n_mfcc, &result);
 *   // arena is now logically free
 *   run_lung_inference (arena, arena_bytes, mfcc_path, n_frames, n_mfcc, &result);
 *
 * Both functions:
 *   1. Validate n_frames / n_mfcc match the model's expected input shape.
 *   2. Build a MicroInterpreter on the provided arena.
 *   3. Stream-quantize the MFCC .f32 file from SD directly into the
 *      int8 input tensor (no intermediate float buffer).
 *   4. Invoke the model.
 *   5. Dequantize the scalar output.
 *   6. Write a small text result file to SD.
 *
 * Heart model (trial_144_int8.tflite):
 *   Input  shape  [1, 1331, 20, 1]   int8 (scale ≈ 0.0656, zp = 23)
 *   Output shape  [1, 1]              int8 (scale ≈ 0.4742, zp = -128)
 *   Regression: output[0] is HR in BPM after dequantization.
 *
 *   IMPORTANT: this requires the firmware heart MFCC config to be
 *   regenerated with num_mfcc=20 (not 25).  Run:
 *       python tools/gen_mfcc_tables.py --num-mfcc 20 ...
 *   to produce heart_mfcc_tables.c, and update heart_mfcc_config.h
 *   so n_mfcc=20 and the mel_fb / dct shapes match.
 *
 * Lung model: not integrated.  When ready, define ENABLE_LUNG_MODEL=1
 * in main.c and follow the TODO in tflm_inference.cc.
 *
 * Output files written to SD:
 *   /SD:/hr.txt    e.g. "HR:72\n"
 *   /SD:/rr.txt    e.g. "RR:15\n"   (when lung model is integrated)
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Result structs ──────────────────────────────────────────────── */
typedef struct {
    float    value;        /* HR in BPM (dequantized regression output) */
    float    confidence;   /* 1.0 for regression (no probability)       */
    int      class_idx;    /* -1 for regression                         */
    int      rc;           /* 0 = ok, negative errno on failure         */
} heart_result_t;

typedef struct {
    float    value;        /* RR in BPM (when lung model is integrated) */
    float    confidence;   /* 1.0 for regression                        */
    int      class_idx;    /* -1 for regression                         */
    int      rc;
} lung_result_t;

/* ── Public API ──────────────────────────────────────────────────── */

/**
 * Run heart inference.
 *
 * @param arena        Shared tensor arena buffer.
 * @param arena_bytes  Size of the arena in bytes.  Heart model needs ~104 KB.
 * @param mfcc_path    SD path to heart_mfcc.f32 (e.g. "/SD:/heart_mfcc.f32").
 * @param n_frames     Frames written by the offline MFCC pass.  Must == 1331.
 * @param n_mfcc       Coefficients per frame.  Must == 20.
 * @param result       Output struct.  result->rc < 0 on failure.
 *
 * On success: writes "/SD:/hr.txt" with "HR:<value>\n".
 * Leaves the arena in an indeterminate state — safe to reuse for
 * run_lung_inference() because that call re-initialises everything.
 */
void run_heart_inference(uint8_t       *arena,
                         size_t         arena_bytes,
                         const char    *mfcc_path,
                         int            n_frames,
                         int            n_mfcc,
                         heart_result_t *result);

/**
 * Run lung inference.  Stub until the lung model is integrated.
 *
 * Same contract as run_heart_inference().
 * On success: writes "/SD:/rr.txt" with "RR:<value>\n".
 * When ENABLE_LUNG_MODEL is 0 or undefined, returns -ENOTSUP without
 * doing anything.
 */
void run_lung_inference(uint8_t      *arena,
                        size_t        arena_bytes,
                        const char   *mfcc_path,
                        int           n_frames,
                        int           n_mfcc,
                        lung_result_t *result);

#ifdef __cplusplus
}
#endif