#ifndef BANDPASS_RESAMPLE_H
#define BANDPASS_RESAMPLE_H

#include <stdint.h>
#include <stddef.h>

/* Bandpass mode selection */
typedef enum {
    BP_MODE_10_200HZ  = 0,   /* 10 Hz – 200 Hz  (e.g. low-freq / vibration speech) */
    BP_MODE_100_1000HZ = 1,  /* 100 Hz – 1000 Hz (e.g. standard narrowband speech)  */
} bp_mode_t;

/**
 * @brief Initialize bandpass + anti-alias filter states.
 *        Call once at startup (or whenever mode changes).
 *
 * @param mode  BP_MODE_10_200HZ or BP_MODE_100_1000HZ
 */
void bandpass_resample_init(bp_mode_t mode);

/**
 * @brief Process one frame: bandpass → anti-alias LP → decimate by 2.
 *
 * @param in        Input PCM at 16 kHz, length = in_len  (int16_t)
 * @param out       Output PCM at 8 kHz, length = in_len/2 (int16_t)
 * @param in_len    Number of input samples (must be even)
 */
void bandpass_resample(const int16_t *in, int16_t *out, size_t in_len);

#endif /* BANDPASS_RESAMPLE_H */