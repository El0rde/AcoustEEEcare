/*
 * tflm_inference.h — TFLite Micro heart + lung inference (v3.2, BLE-only)
 * =======================================================================
 * v3.2 (one-capture dual inference):
 *   - run_*_inference_ram(): caller passes PRE-QUANTIZED int8 features
 *     (normalized + quantized on-device in the MFCC callbacks). This file
 *     loads them into the model input tensor, invokes, dequantizes, clips.
 *   - tflm_get_input_quant(): probe a model's int8 input scale/zero_point
 *     at boot so the on-device quantizer matches the loaded model exactly
 *     (no hardcoded-scale drift).
 *   - No SD card / no filesystem.
 *
 * Heart: input [1,665,25,1] int8 → HR, clipped [40,180]
 * Lung:  input [1,324,26,1] int8 → RR, clipped [6,50]
 * Ops (both): CONV_2D, MAX_POOL_2D, MEAN, FULLY_CONNECTED.
 */

#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { float value; float confidence; int class_idx; int rc; } heart_result_t;
typedef struct { float value; float confidence; int class_idx; int rc; } lung_result_t;

/* organ: 0 = heart, 1 = lung. Probes the model input tensor's quantization.
 * Returns 0 and fills *scale/*zero_point on success; negative on failure
 * (caller should keep its fallback constants). Uses the provided arena. */
int tflm_get_input_quant(int organ, uint8_t *arena, size_t arena_bytes,
                         float *scale, int32_t *zero_point);

/* features: n_frames*n_mfcc int8, row-major [frame][coeff], already
 * normalized + quantized with the model's input scale/zp. */
void run_heart_inference_ram(uint8_t *arena, size_t arena_bytes,
                             const int8_t *features, int n_frames, int n_mfcc,
                             heart_result_t *result);

void run_lung_inference_ram(uint8_t *arena, size_t arena_bytes,
                            const int8_t *features, int n_frames, int n_mfcc,
                            lung_result_t *result);

#ifdef __cplusplus
}
#endif
