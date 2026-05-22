/*
 * tflm_inference.h — TFLite Micro heart + lung inference
 * =========================================================
 * v2.2 — synced with tflm_inference.cc v2.2 (lung integrated).
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
 * Heart model (best_mcu_int8.tflite):
 *   Input  shape  [1, 665, 25, 1]   int8
 *   Output shape  [1, 1]             int8
 *   Regression: output[0] is HR in BPM after dequantization.
 *   MFCC config must match heart_winning_config (665 frames × 25 mfcc).
 *
 * Lung model (best_mcu_lung_int8.tflite):
 *   Input  shape  [1, 324, 26, 1]   int8   (derived from
 *                 lung_winning_config: frame=300 ms, hop=10%)
 *   Output: regression scalar (RR in BPM).
 *
 * Output files written to SD:
 *   /SD:/hr.txt    e.g. "HR:72\n"
 *   /SD:/rr.txt    e.g. "RR:15\n"
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
    float    value;        /* RR in BPM (regression output)             */
    float    confidence;   /* 1.0 for regression                        */
    int      class_idx;    /* -1 for regression                         */
    int      rc;
} lung_result_t;

/* ── Public API ──────────────────────────────────────────────────── */

/**
 * Run heart inference.
 *
 * @param arena        Shared tensor arena buffer.
 * @param arena_bytes  Size of the arena in bytes.  Heart needs ~92 KB; 100 KB recommended.
 * @param mfcc_path    SD path to heart_mfcc.f32 (e.g. "/SD:/heart_mfcc.f32").
 * @param n_frames     Frames written by the offline MFCC pass.  Must == 665.
 * @param n_mfcc       Coefficients per frame.  Must == 25.
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
 * Run lung inference.
 *
 * Same contract as run_heart_inference().
 *   n_frames must == 324, n_mfcc must == 26 (per lung_winning_config).
 *
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

/* Inference from an in-RAM, already-int8-quantized MFCC buffer (no SD).
 * features = n_frames*n_mfcc int8 values, row-major [frame][mfcc],
 * quantized with the model's input scale/zero_point. */
void run_heart_inference_ram(uint8_t *arena, size_t arena_bytes,
                             const int8_t *features, int n_frames, int n_mfcc,
                             heart_result_t *result);
void run_lung_inference_ram(uint8_t *arena, size_t arena_bytes,
                            const int8_t *features, int n_frames, int n_mfcc,
                            lung_result_t *result);

#ifdef __cplusplus
}
#endif