#!/usr/bin/env python3
"""
gen_debug_audio.py — embed a known WAV as an 8 kHz int16 C array for the
firmware injection self-test. Upsamples (linear) to the device's 8 kHz ADC
rate so the on-device pipeline processes it exactly like a live capture.

Usage:  python gen_debug_audio.py input.wav debug_audio.h
"""
import sys, wave, numpy as np

def main():
    src, out = sys.argv[1], sys.argv[2]
    with wave.open(src,'rb') as w:
        sr=w.getframerate(); x=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2')
    TARGET=8000; DURS=10
    # linear upsample/resample to 8 kHz, then pad/trim to exactly 10 s
    n_out=TARGET*DURS
    t_src=np.arange(len(x))/sr
    t_out=np.arange(n_out)/TARGET
    y=np.interp(t_out, t_src, x.astype(np.float64))
    y=np.clip(np.round(y),-32768,32767).astype(np.int16)
    with open(out,'w') as f:
        f.write("/* AUTO-GENERATED debug heartbeat buffer (8 kHz int16, 10 s).\n")
        f.write(f" * Source: {src}  resampled {sr}->{TARGET} Hz.\n")
        f.write(" * For DEBUG_INJECT_BUFFER firmware self-test only. */\n")
        f.write("#ifndef DEBUG_AUDIO_H\n#define DEBUG_AUDIO_H\n#include <stdint.h>\n")
        f.write(f"#define DEBUG_AUDIO_LEN {n_out}u\n")
        f.write("static const int16_t debug_audio_pcm[DEBUG_AUDIO_LEN] = {\n")
        for i in range(0,n_out,20):
            f.write(" "+",".join(str(int(v)) for v in y[i:i+20])+",\n")
        f.write("};\n#endif /* DEBUG_AUDIO_H */\n")
    print(f"wrote {out}: {n_out} samples @ {TARGET} Hz")

if __name__=="__main__": main()
