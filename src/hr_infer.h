#pragma once
#include <stdint.h>

typedef struct {
    float    bpm;
    int8_t   raw_output;
} hr_result_t;

#define HR_MODEL_FRAMES   1331
#define HR_MODEL_MFCC     20
#define HR_MODEL_ELEMENTS (HR_MODEL_FRAMES * HR_MODEL_MFCC)

#define HR_INPUT_SCALE      0.06558491f
#define HR_INPUT_ZP         23
#define HR_OUTPUT_SCALE     0.47415614f
#define HR_OUTPUT_ZP        (-128)

#ifdef __cplusplus
extern "C" {
#endif

int hr_infer_init(void);
int hr_infer_run(const float *mfcc_flat, int n_frames, hr_result_t *out);
int hr_infer_run_int8(const int8_t *mfcc_q8, int n_frames, hr_result_t *out);

#ifdef __cplusplus
}
#endif