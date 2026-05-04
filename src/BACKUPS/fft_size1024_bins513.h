/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * fft_entry.h
 *
 * Code generation for function 'fft_entry'
 *
 */

#ifndef FFT_ENTRY_H
#define FFT_ENTRY_H

/* Include files */
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
extern void fft_entry(const short x[1024], creal_T y[513]);

extern void fft_entry_initialize(void);

extern void fft_entry_terminate(void);

#ifdef __cplusplus
}
#endif

#endif
/* End of code generation (fft_entry.h) */
