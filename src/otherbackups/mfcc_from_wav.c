/*
 * mfcc_from_wav.c
 *
 * Reads input.wav (16 kHz, 16-bit, mono PCM),
 * decimates to 8 kHz, computes MFCC and writes mfcc_c.csv.
 *
 * Designed to match MATLAB's mfcc() with BandEdges exactly:
 *   - No pre-emphasis
 *   - Periodic Hamming window
 *   - Float FFT (no int16 quantization)
 *   - Power spectrum |X|^2
 *   - Mel filterbank with round() bin mapping  <- matches MATLAB
 *   - log10 rectification
 *   - Orthonormal DCT-II, 13 coefficients
 *
 * Build:
 *   gcc -O2 -o mfcc_from_wav mfcc_from_wav.c -lm
 *
 * Usage:
 *   mfcc_from_wav.exe                        reads input.wav, writes mfcc_c.csv
 *   mfcc_from_wav.exe myfile.wav             custom input
 *   mfcc_from_wav.exe myfile.wav out.csv     custom input + output
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ══════════════════════════════════════════════════════════════
 * PARAMETERS
 * ══════════════════════════════════════════════════════════════ */
#define SAMPLE_RATE       4000   /* input WAV rate                  */
#define TARGET_RATE       2000   /* MFCC processing rate            */
#define DECIMATE_FACTOR      2   /* SAMPLE_RATE / TARGET_RATE       */

#define FRAME_SAMPLES       60   /* 30 ms @ 2 kHz                   */
#define OVERLAP_SAMPLES     36   /* 12 ms @ 2 kHz                   */
#define HOP_SIZE            24   /* 18 ms @ 2 kHz                   */
#define FFT_SIZE           512
#define NUM_BINS           257   /* FFT_SIZE/2 + 1                  */
#define NUM_MEL_FILTERS     13
#define NUM_MFCC            13
#define LOW_FREQ_HZ       80.0f  /* mel filterbank low edge         */

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#define MIN(a,b) ((a)<(b)?(a):(b))

/* ══════════════════════════════════════════════════════════════
 * FFT — naive DFT, float input, float output
 * ══════════════════════════════════════════════════════════════ */
static float fft_re[FFT_SIZE];
static float fft_im[FFT_SIZE];

static void fft_shim(const float *in)
{
    for (int k = 0; k < FFT_SIZE; k++) {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < FFT_SIZE; n++) {
            double angle = 2.0 * M_PI * k * n / FFT_SIZE;
            re += in[n] * cos(angle);
            im -= in[n] * sin(angle);
        }
        fft_re[k] = (float)re;
        fft_im[k] = (float)im;
    }
}

/* ══════════════════════════════════════════════════════════════
 * MEL FILTERBANK
 * Uses round() to match MATLAB's internal bin mapping.
 * MATLAB's mfcc() with BandEdges uses round(), not floor().
 * ══════════════════════════════════════════════════════════════ */
static float mel_fb[NUM_MEL_FILTERS][FFT_SIZE];

static inline float hz_to_mel(float hz)  { return 1127.0f * logf(1.0f + hz / 700.0f); }
static inline float mel_to_hz(float mel) { return 700.0f  * (expf(mel / 1127.0f) - 1.0f); }

static void build_mel_filterbank(void)
{
    /* Replicates audio.internal.designMelFilterBank() exactly:
     *
     * 1. Bin frequency: linFq[k] = k/FFT_SIZE * TARGET_RATE  (Hz)
     * 2. Inflection point p[e] = first bin where linFq[bin] > bandEdge[e]
     *    (strict greater-than, same as MATLAB's inner loop)
     * 3. Rising slope  (bins p[e]   .. p[e+1]-1):
     *      weight = (linFq[bin] - edge[e]) / (edge[e+1] - edge[e])
     * 4. Falling slope (bins p[e+1] .. p[e+2]-1):
     *      weight = (edge[e+2] - linFq[bin]) / (edge[e+2] - edge[e+1])
     * 5. Bandwidth normalisation: divide by (edge[e+2] - edge[e]) / 2
     *    (MATLAB uses filterBandWidth/2, not filterBandWidth)
     */
    float mel_low  = hz_to_mel(LOW_FREQ_HZ);
    float mel_high = hz_to_mel((float)TARGET_RATE / 2.0f);

    /* Compute the 22 band edge frequencies in Hz */
    float edge_hz[NUM_MEL_FILTERS + 2];
    for (int pt = 0; pt < NUM_MEL_FILTERS + 2; pt++) {
        float fraction = (float)pt / (float)(NUM_MEL_FILTERS + 1);
        float mel      = mel_low + fraction * (mel_high - mel_low);
        edge_hz[pt]    = mel_to_hz(mel);
    }

    /* Inflection points: p[e] = first bin index where linFq[bin] > edge_hz[e]
     * linFq[bin] = bin * TARGET_RATE / FFT_SIZE
     * Strict >, matching MATLAB's: if linFq(index) > bandEdgesCast(edgeNumber) */
    int p[NUM_MEL_FILTERS + 2];
    for (int e = 0; e < NUM_MEL_FILTERS + 2; e++) {
        p[e] = FFT_SIZE - 1;   /* default: last bin */
        for (int b = 0; b < FFT_SIZE; b++) {
            float linFq = (float)b / (float)FFT_SIZE * (float)TARGET_RATE;
            if (linFq > edge_hz[e]) {
                p[e] = b;
                break;
            }
        }
    }

    memset(mel_fb, 0, sizeof(mel_fb));
    for (int f = 0; f < NUM_MEL_FILTERS; f++) {
        float bw_rise = edge_hz[f + 1] - edge_hz[f];
        float bw_fall = edge_hz[f + 2] - edge_hz[f + 1];
        if (bw_rise < 1e-10f) bw_rise = 1e-10f;
        if (bw_fall < 1e-10f) bw_fall = 1e-10f;

        /* Bandwidth normalisation weight = (edge[f+2] - edge[f]) / 2 */
        float weight = (edge_hz[f + 2] - edge_hz[f]) / 2.0f;
        if (weight < 1e-10f) weight = 1e-10f;

        /* Rising slope: bins p[f] .. p[f+1]-1 */
        for (int b = p[f]; b < p[f + 1] && b < FFT_SIZE; b++) {
            float linFq      = (float)b / (float)FFT_SIZE * (float)TARGET_RATE;
            mel_fb[f][b]     = ((linFq - edge_hz[f]) / bw_rise) / weight;
        }

        /* Falling slope: bins p[f+1] .. p[f+2]-1 */
        for (int b = p[f + 1]; b < p[f + 2] && b < FFT_SIZE; b++) {
            float linFq      = (float)b / (float)FFT_SIZE * (float)TARGET_RATE;
            mel_fb[f][b]     = ((edge_hz[f + 2] - linFq) / bw_fall) / weight;
        }
    }
}

/* ══════════════════════════════════════════════════════════════
 * MFCC
 * ══════════════════════════════════════════════════════════════ */
static float windowed_f[FFT_SIZE];
static float mag_buf[FFT_SIZE];
static float log_energy[NUM_MEL_FILTERS];
static float mfcc_out[NUM_MFCC];

static void compute_mfcc(const int16_t *pcm)
{
    /* Periodic Hamming window, normalised to [-1, 1] — no pre-emphasis */
    for (int n = 0; n < FRAME_SAMPLES; n++) {
        float signal    = (float)pcm[n] / 32768.0f;
        float win       = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * n
                                                / (float)FRAME_SAMPLES);
        windowed_f[n]   = signal * win;
    }
    for (int n = FRAME_SAMPLES; n < FFT_SIZE; n++)
        windowed_f[n] = 0.0f;

    /* FFT — float input, no int16 quantization */
    fft_shim(windowed_f);

    /* Magnitude spectrum — matches MATLAB's Z = abs(fft(...))
     * Full FFT_SIZE bins, symmetric second half already filled by fft_shim */
    for (int k = 0; k < FFT_SIZE; k++) {
        float re = fft_re[k], im = fft_im[k];
        mag_buf[k] = sqrtf(re * re + im * im);
    }

    /* Mel filterbank + log10 — filterbank spans full FFT_SIZE */
    for (int m = 0; m < NUM_MEL_FILTERS; m++) {
        float e = 0.0f;
        for (int k = 0; k < FFT_SIZE; k++)
            e += mel_fb[m][k] * mag_buf[k];
        log_energy[m] = log10f(e < 1e-10f ? 1e-10f : e);
    }

    /* Orthonormal DCT-II */
    float scale_dc   = sqrtf(1.0f / (float)NUM_MEL_FILTERS);
    float scale_rest = sqrtf(2.0f / (float)NUM_MEL_FILTERS);
    for (int n = 0; n < NUM_MFCC; n++) {
        float sum = 0.0f;
        for (int m = 0; m < NUM_MEL_FILTERS; m++)
            sum += log_energy[m]
                 * cosf((float)M_PI * (float)n * (m + 0.5f)
                        / (float)NUM_MEL_FILTERS);
        mfcc_out[n] = (n == 0 ? scale_dc : scale_rest) * sum;
    }
}

/* ══════════════════════════════════════════════════════════════
 * WAV PARSER  (uncompressed PCM, 16-bit only)
 * ══════════════════════════════════════════════════════════════ */
typedef struct {
    uint32_t sample_rate;
    uint16_t num_channels;
    uint16_t bits_per_sample;
    uint32_t num_samples;
    long     data_offset;
} WavInfo;

static int parse_wav(FILE *fp, WavInfo *info)
{
    uint8_t buf[12];
    if (fread(buf, 1, 12, fp) != 12)       return -1;
    if (memcmp(buf,     "RIFF", 4) != 0)   return -1;
    if (memcmp(buf + 8, "WAVE", 4) != 0)   return -1;

    uint32_t sr = 0, data_len = 0;
    uint16_t nc = 0, bps = 0;
    long data_off = 0;
    int got_fmt = 0, got_data = 0;

    while (!got_data) {
        uint8_t tag[4]; uint32_t chunk_size;
        if (fread(tag,         1, 4, fp) != 4) break;
        if (fread(&chunk_size, 1, 4, fp) != 4) break;

        if (memcmp(tag, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (fread(fmt, 1, 16, fp) != 16) return -1;
            uint16_t afmt; memcpy(&afmt, fmt, 2);
            memcpy(&nc,  fmt +  2, 2);
            memcpy(&sr,  fmt +  4, 4);
            memcpy(&bps, fmt + 14, 2);
            if (afmt != 1) { fprintf(stderr,"Only PCM WAV supported\n"); return -1; }
            if (chunk_size > 16) fseek(fp, chunk_size - 16, SEEK_CUR);
            got_fmt = 1;
        } else if (memcmp(tag, "data", 4) == 0) {
            data_off = ftell(fp);
            data_len = chunk_size;
            got_data = 1;
        } else {
            fseek(fp, chunk_size, SEEK_CUR);
        }
    }
    if (!got_fmt || !got_data) return -1;
    info->sample_rate     = sr;
    info->num_channels    = nc;
    info->bits_per_sample = bps;
    info->num_samples     = data_len / (nc * (bps / 8));
    info->data_offset     = data_off;
    return 0;
}

/* ══════════════════════════════════════════════════════════════
 * MAIN
 * ══════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    const char *wav_path = (argc >= 2) ? argv[1] : "input.wav";
    const char *csv_path = (argc >= 3) ? argv[2] : "mfcc_c.csv";

    /* 1. Open WAV */
    FILE *fp = fopen(wav_path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open '%s'\n", wav_path); return 1; }

    WavInfo wav;
    if (parse_wav(fp, &wav) != 0) {
        fprintf(stderr, "Failed to parse WAV — must be uncompressed 16-bit PCM\n");
        fclose(fp); return 1;
    }
    if (wav.bits_per_sample != 16) {
        fprintf(stderr, "Only 16-bit WAV supported\n"); fclose(fp); return 1;
    }
    if ((int)wav.sample_rate != SAMPLE_RATE)
        fprintf(stderr, "Warning: WAV is %u Hz, expected %d Hz\n",
                wav.sample_rate, SAMPLE_RATE);

    printf("WAV: %u samples @ %u Hz, %u ch, %u-bit\n",
           wav.num_samples, wav.sample_rate,
           wav.num_channels, wav.bits_per_sample);

    /* 2. Read samples — left channel only */
    int      n_16k = (int)wav.num_samples;
    int16_t *s_16k = (int16_t *)malloc(n_16k * sizeof(int16_t));
    if (!s_16k) { perror("malloc"); fclose(fp); return 1; }

    fseek(fp, wav.data_offset, SEEK_SET);
    for (int i = 0; i < n_16k; i++) {
        int16_t s;
        if (fread(&s, 2, 1, fp) != 1) { n_16k = i; break; }
        s_16k[i] = s;
        for (int ch = 1; ch < (int)wav.num_channels; ch++) {
            int16_t dummy; fread(&dummy, 2, 1, fp);
        }
    }
    fclose(fp);
    printf("Read %d samples @ %d Hz\n", n_16k, SAMPLE_RATE);

    /* 3. Decimate 16 kHz -> 8 kHz: take every 2nd sample */
    int      n_8k = n_16k / DECIMATE_FACTOR;
    int16_t *s_8k = (int16_t *)malloc(n_8k * sizeof(int16_t));
    if (!s_8k) { fprintf(stderr, "malloc failed\n"); free(s_16k); return 1; }
    for (int i = 0; i < n_8k; i++)
        s_8k[i] = s_16k[i * DECIMATE_FACTOR];
    free(s_16k);
    printf("After decimation: %d samples @ %d Hz\n", n_8k, TARGET_RATE);

    /* 4. Init */
    build_mel_filterbank();

    /* 5. Frame loop */
    int n_frames = (n_8k >= FRAME_SAMPLES)
                 ? 1 + (n_8k - FRAME_SAMPLES) / HOP_SIZE
                 : 0;
    printf("Frames: %d\n", n_frames);

    /* 6. Open CSV */
    FILE *csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "Cannot open '%s' for writing\n", csv_path);
        free(s_8k); return 1;
    }
    for (int i = 0; i < NUM_MFCC; i++)
        fprintf(csv, i < NUM_MFCC - 1 ? "c%d," : "c%d\n", i);

    /* 7. Process */
    for (int frame = 0; frame < n_frames; frame++) {
        compute_mfcc(s_8k + frame * HOP_SIZE);
        for (int i = 0; i < NUM_MFCC; i++)
            fprintf(csv, i < NUM_MFCC - 1 ? "%.6f," : "%.6f\n",
                    (double)mfcc_out[i]);
    }

    fclose(csv);
    free(s_8k);
    printf("Saved -> %s  (%d frames x %d coefficients)\n",
           csv_path, n_frames, NUM_MFCC);
    return 0;
}