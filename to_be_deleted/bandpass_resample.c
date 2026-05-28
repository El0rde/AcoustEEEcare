#include "bandpass_resample.h"
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * Biquad IIR — Direct Form I
 *
 * Each section stores:  b0, b1, b2, a1, a2  (a0 normalised to 1)
 * State per section:    x1, x2, y1, y2
 * ───────────────────────────────────────────────────────────────────────────*/
typedef struct {
    float b0, b1, b2;
    float a1, a2;
    float x1, x2;
    float y1, y2;
} biquad_t;

static float biquad_process(biquad_t *s, float x)
{
    float y = s->b0 * x
            + s->b1 * s->x1
            + s->b2 * s->x2
            - s->a1 * s->y1
            - s->a2 * s->y2;

    s->x2 = s->x1;  s->x1 = x;
    s->y2 = s->y1;  s->y1 = y;
    return y;
}

static void biquad_reset(biquad_t *s)
{
    s->x1 = s->x2 = s->y1 = s->y2 = 0.0f;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Coefficient design — all for Fs = 16 000 Hz
 *
 * Method: 2nd-order Butterworth sections (bilinear transform).
 * A 4th-order bandpass = 2 biquad sections in series.
 * Anti-alias LP = 4th-order Butterworth at 3800 Hz (< Nyquist of 8 kHz).
 *
 * Coefficients computed offline with:
 *   [b,a] = butter(2, [Wl Wh])          % MATLAB, Wl/Wh normalised 0-1
 *   [sos] = tf2sos(b, a)
 *
 * Section layout: {b0, b1, b2, a1, a2}
 * ───────────────────────────────────────────────────────────────────────────*/

/* ── Mode 0: Bandpass 10 – 200 Hz @ 16 kHz ──────────────────────────────────
 *   butter(2, [10 200]/8000)  → 4th order → 2 biquad sections
 *   MATLAB: [sos,g] = butter(2,[10 200]/8000,'bandpass'); sos scaled by g   */
static const biquad_t BP_10_200_COEFFS[2] = {
    /* Section 1 */
    { .b0 =  1.0f,        .b1 =  0.0f,        .b2 = -1.0f,
      .a1 = -1.99380f,    .a2 =  0.99382f },
    /* Section 2 */
    { .b0 =  1.0f,        .b1 =  0.0f,        .b2 = -1.0f,
      .a1 = -1.84597f,    .a2 =  0.85650f },
};
/* Overall gain for 10-200 Hz bandpass (product of section gains from MATLAB) */
#define BP_10_200_GAIN  7.617e-4f

/* ── Mode 1: Bandpass 100 – 1000 Hz @ 16 kHz ────────────────────────────────
 *   butter(2, [100 1000]/8000)  → 4th order → 2 biquad sections             */
static const biquad_t BP_100_1000_COEFFS[2] = {
    /* Section 1 */
    { .b0 =  1.0f,        .b1 =  0.0f,        .b2 = -1.0f,
      .a1 = -1.92562f,    .a2 =  0.92684f },
    /* Section 2 */
    { .b0 =  1.0f,        .b1 =  0.0f,        .b2 = -1.0f,
      .a1 = -1.56720f,    .a2 =  0.64359f },
};
#define BP_100_1000_GAIN  3.036e-2f

/* ── Anti-alias LP: 4th-order Butterworth at 3800 Hz @ 16 kHz ───────────────
 *   butter(2, 3800/8000)  → 2nd order; cascade two identical sections
 *   for 4th-order roll-off before decimation                                  */
static const biquad_t AA_LP_COEFFS[2] = {
    /* Section 1 */
    { .b0 =  0.25574f,    .b1 =  0.51147f,    .b2 =  0.25574f,
      .a1 = -0.17834f,    .a2 =  0.20128f },
    /* Section 2 */
    { .b0 =  0.25574f,    .b1 =  0.51147f,    .b2 =  0.25574f,
      .a1 = -0.17834f,    .a2 =  0.20128f },
};

/* ─────────────────────────────────────────────────────────────────────────────
 * Runtime state
 * ───────────────────────────────────────────────────────────────────────────*/
static biquad_t bp_stage[2];   /* active bandpass sections (copied + cleared) */
static biquad_t aa_stage[2];   /* anti-alias LP sections                      */
static float    bp_gain;       /* scalar gain for the active bandpass          */

void bandpass_resample_init(bp_mode_t mode)
{
    /* Copy coefficient templates into mutable state structs and zero delays */
    const biquad_t *bp_src;

    if (mode == BP_MODE_10_200HZ) {
        bp_src  = BP_10_200_COEFFS;
        bp_gain = BP_10_200_GAIN;
    } else {
        bp_src  = BP_100_1000_COEFFS;
        bp_gain = BP_100_1000_GAIN;
    }

    for (int i = 0; i < 2; i++) {
        bp_stage[i] = bp_src[i];
        biquad_reset(&bp_stage[i]);

        aa_stage[i] = AA_LP_COEFFS[i];
        biquad_reset(&aa_stage[i]);
    }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * bandpass_resample()
 *
 * Pipeline per input sample:
 *   int16 → float → [BP stage 1] → [BP stage 2] → gain scale
 *                 → [AA LP stage 1] → [AA LP stage 2]
 *                 → decimate (keep every 2nd sample) → int16
 * ───────────────────────────────────────────────────────────────────────────*/
void bandpass_resample(const int16_t *in, int16_t *out, size_t in_len)
{
    size_t out_idx = 0;

    for (size_t n = 0; n < in_len; n++) {
        /* 1. Convert to float (keep full int16 scale — no divide) */
        float x = (float)in[n];

        /* 2. Bandpass — 2 biquad sections in series + gain */
        x = biquad_process(&bp_stage[0], x);
        x = biquad_process(&bp_stage[1], x);
        x *= bp_gain;

        /* 3. Anti-alias low-pass — 2 biquad sections in series */
        x = biquad_process(&aa_stage[0], x);
        x = biquad_process(&aa_stage[1], x);

        /* 4. Decimate by 2 — keep even-indexed samples */
        if ((n & 1u) == 0u) {
            /* Saturating cast back to int16 */
            float clipped = x < -32768.0f ? -32768.0f
                          : x >  32767.0f ?  32767.0f : x;
            out[out_idx++] = (int16_t)clipped;
        }
    }
}