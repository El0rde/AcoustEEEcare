/* cnn_norm_stats.h — per-coefficient z-score stats from CNN training.
 *
 * heart: cnn_results/cnn_norm_stats.npz   (25 coefficients)
 * lung:  lung_cnn_search/.../cnn_norm_stats.npz  (26 coefficients)
 *
 * Apply:  v = (coeff - mean[i]) / (std[i] + CNN_NORM_EPS)
 * BEFORE int8 quantization in heart_frame_cb and lung_frame_cb.
 *
 * The int8 input quantization scales (heart 0.396/zp38, lung 0.337/zp28)
 * were calibrated on NORMALIZED features.  Feeding raw MFCC values
 * causes the quantized inputs to land in the wrong range and the model
 * output to saturate at ~321 BPM.  This header fixes that.
 */
#ifndef CNN_NORM_STATS_H
#define CNN_NORM_STATS_H

#define CNN_NORM_EPS 1e-8f

/* ── Heart (25 coefficients) ─────────────────────────────────────── */
static const float heart_mfcc_mean[25] = {
    -39.99887f,  16.44387f,   7.44548f,   1.87824f,  -0.25391f,
     -0.43583f,   0.03216f,   0.28094f,   0.22963f,   0.01929f,
     -0.10588f,  -0.11651f,  -0.05044f,  -0.00097f,   0.01591f,
     -0.00197f,  -0.02068f,  -0.03133f,  -0.02724f,  -0.01735f,
     -0.00687f,  -0.00742f,  -0.00608f,  -0.00815f,  -0.00424f
};

static const float heart_mfcc_std[25] = {
      8.41331f,   3.01032f,   1.52634f,   1.75015f,   1.48996f,
      1.08274f,   0.91992f,   0.84160f,   0.75321f,   0.68056f,
      0.61896f,   0.56503f,   0.51448f,   0.46915f,   0.42906f,
      0.39057f,   0.35400f,   0.32480f,   0.30080f,   0.27495f,
      0.25527f,   0.24982f,   0.22503f,   0.22196f,   0.24233f
};

/* ── Lung (26 coefficients) ──────────────────────────────────────── */
static const float lung_mfcc_mean[26] = {
    -19.21439f,  19.46190f,  -7.15684f,   4.00321f,   0.00528f,
      0.16393f,   0.83603f,   0.15148f,   0.32986f,  -0.12560f,
      0.19978f,  -0.14356f,   0.06371f,  -0.05238f,   0.00675f,
      0.04725f,  -0.01417f,   0.02209f,   0.00533f,  -0.01905f,
     -0.04135f,  -0.03226f,   0.00532f,   0.04169f,  -0.00170f,
      0.03349f
};

static const float lung_mfcc_std[26] = {
      4.69837f,   2.33547f,   2.16953f,   1.40876f,   1.13252f,
      0.83434f,   0.69917f,   0.62863f,   0.59668f,   0.55600f,
      0.52916f,   0.50479f,   0.48703f,   0.45823f,   0.43214f,
      0.40609f,   0.37811f,   0.34997f,   0.32744f,   0.30237f,
      0.28253f,   0.26862f,   0.25542f,   0.25113f,   0.24119f,
      0.23850f
};

#endif /* CNN_NORM_STATS_H */
