/*
 * tflm_inference.h — TFLite Micro heart + lung inference
 * =========================================================
 * v3.1 — merged v2.2 + v3.0 (BLE SD-free additions).
 *
 * run_heart_inference / run_lung_inference now take a pre-quantized
 * int8 pointer (BLE path).  The SD mfcc_path string argument is
 * removed from the public API; SD quantization is handled internally
 * in tflm_inference.cc if needed.
 *
 * get_heart_quant_params / get_lung_quant_params added so main.c
 * can send scale+zp to the host over BLE at startup.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Result structs ──────────────────────────────────────────────── */
typedef struct {
    float value;        /* HR in BPM (dequantized regression output) */
    float confidence;   /* 1.0 for regression models                  */
    int   class_idx;    /* -1 for regression                          */
    int   rc;           /* 0 = ok, negative errno on failure          */
} heart_result_t;

typedef struct {
    float value;        /* RR in breaths/min                          */
    float confidence;
    int   class_idx;
    int   rc;
} lung_result_t;

/**
 * run_heart_inference()
 *
 * @param arena        Shared tensor arena (72 KB recommended).
 * @param arena_bytes  Size of arena in bytes.
 * @param mfcc_int8    Pre-quantized int8 MFCC, row-major [n_frames][n_mfcc].
 *                     Must be exactly n_frames * n_mfcc bytes.
 *                     Quantized by host using QUANT_HEART scale/zp.
 * @param n_frames     Must == 665.
 * @param n_mfcc       Must == 25.
 * @param result       Output. result->rc < 0 on failure.
 */
void run_heart_inference(uint8_t        *arena,
                         size_t          arena_bytes,
                         const int8_t   *mfcc_int8,
                         int             n_frames,
                         int             n_mfcc,
                         heart_result_t *result);

/**
 * run_lung_inference()
 *
 * @param mfcc_int8  Pre-quantized int8, row-major [n_frames][n_mfcc].
 * @param n_frames   Must == 324.
 * @param n_mfcc     Must == 26.
 */
void run_lung_inference(uint8_t       *arena,
                        size_t         arena_bytes,
                        const int8_t  *mfcc_int8,
                        int            n_frames,
                        int            n_mfcc,
                        lung_result_t *result);

/**
 * get_heart_quant_params() / get_lung_quant_params()
 *
 * Retrieve the input tensor's quantization parameters by briefly
 * instantiating the model on the arena.  Call once after BLE connect,
 * before any recording.  The host uses scale/zp to quantize MFCC
 * float32 → int8 before sending over BLE.
 *
 * @return 0 on success, negative errno on failure.
 */
int get_heart_quant_params(uint8_t *arena, size_t arena_bytes,
                           float *out_scale, int32_t *out_zp);

int get_lung_quant_params(uint8_t *arena, size_t arena_bytes,
                          float *out_scale, int32_t *out_zp);

#ifdef __cplusplus
}
#endif