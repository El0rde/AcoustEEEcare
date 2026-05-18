#!/usr/bin/env python3
"""
AcoustEEEcare v7.5 — Host-side BLE receiver

Updated for firmware v7.5 + winning configs:
  - Heart MFCC: 665 frames × 25 coeffs   (was 1331 × 25)
  - Lung  MFCC: 324 frames × 26 coeffs   (was 7981 × 13)
  - _all_done() now has a grace period so results aren't skipped
    when the receiver polls between MFCC_LUNG_END and RESULT_*_START.

Stream order (firmware v7.5):
  1. Audio PCM          START:<n>\\n … finished\\n
  2. SD:OK\\n
  3. Heart MFCC         MFCC_HEART_START:<n>\\n … MFCC_HEART_END\\n
  4. Lung MFCC          MFCC_LUNG_START:<n>\\n  … MFCC_LUNG_END\\n
  5. Heart result       RESULT_HEART_START:<n>\\n … RESULT_HEART_END\\n  (if model enabled)
  6. Lung result        RESULT_LUNG_START:<n>\\n  … RESULT_LUNG_END\\n   (if model enabled)

Files saved:
  recordings/rec_NNN_TIMESTAMP.wav
  recordings/heart_mfcc_NNN_TIMESTAMP.npy
  recordings/lung_mfcc_NNN_TIMESTAMP.npy
  recordings/hr_NNN_TIMESTAMP.txt          (e.g. "HR:72")
  recordings/rr_NNN_TIMESTAMP.txt          (e.g. "RR:15")
"""

import asyncio
import struct
import wave
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from bleak import BleakScanner, BleakClient

NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

DEVICE_NAME    = "AcoustEEEcare"
SAMPLE_RATE    = 8000
OUTPUT_DIR     = Path("recordings")
SCAN_TIMEOUT   = 20.0
REC_TIMEOUT    = 120.0   # v7.5: was 360 s. Realistic upper bound is ~50 s
                         # (10 s capture + 5 s MFCC + 5 s inference + 30 s BLE).
RESULT_GRACE_S = 3.0     # v7.5: time to wait after MFCC_LUNG_END for the
                         # RESULT_*_START packets before assuming the firmware
                         # has results disabled.

# v7.5: match firmware-side winning configs.
HEART_N_FRAMES = 665     # was 1331
HEART_N_MFCC   = 25      # unchanged
LUNG_N_FRAMES  = 324     # was 7981
LUNG_N_MFCC    = 26      # was 13


# ── State machine ─────────────────────────────────────────────────────────────

class StreamMode:
    IDLE         = "idle"
    AUDIO        = "audio"
    MFCC_HEART   = "mfcc_heart"
    MFCC_LUNG    = "mfcc_lung"
    RESULT_HEART = "result_heart"
    RESULT_LUNG  = "result_lung"


class State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode             = StreamMode.IDLE
        # Audio
        self.expected_bytes   = 0
        self.audio_done       = False
        self.audio_samples    = bytearray()
        self.audio_seq        = 0
        self.audio_gaps       = 0
        self.audio_chunks     = 0
        # Heart MFCC
        self.heart_expected_b = 0
        self.heart_done       = False
        self.heart_bytes      = bytearray()
        self.heart_seq        = 0
        self.heart_gaps       = 0
        self.heart_chunks     = 0
        # Lung MFCC
        self.lung_expected_b  = 0
        self.lung_done        = False
        self.lung_bytes       = bytearray()
        self.lung_seq         = 0
        self.lung_gaps        = 0
        self.lung_chunks      = 0
        # Heart result
        self.hr_expected_b    = 0
        self.hr_done          = False
        self.hr_bytes         = bytearray()
        self.hr_seq           = 0
        # Lung result
        self.rr_expected_b    = 0
        self.rr_done          = False
        self.rr_bytes         = bytearray()
        self.rr_seq           = 0
        # Progress
        self.last_progress    = 0.0
        # v7.5: latch the moment all base streams (audio+heart+lung MFCC)
        # are done, so _all_done() can apply a grace period before
        # declaring "no results coming".
        self.base_done_at     = None


state = State()


# ── Packet classifier ─────────────────────────────────────────────────────────

def _is_data_chunk(data: bytearray) -> bool:
    if state.mode == StreamMode.IDLE or len(data) < 5:
        return False
    ln = struct.unpack_from("<H", data, 2)[0]
    return ln > 0 and ln == len(data) - 4


def handle_notification(sender, data: bytearray):
    if _is_data_chunk(data):
        if state.mode == StreamMode.AUDIO:
            _handle_audio_chunk(data)
        elif state.mode == StreamMode.MFCC_HEART:
            _handle_mfcc_chunk(data, "heart")
        elif state.mode == StreamMode.MFCC_LUNG:
            _handle_mfcc_chunk(data, "lung")
        elif state.mode == StreamMode.RESULT_HEART:
            _handle_result_chunk(data, "heart")
        elif state.mode == StreamMode.RESULT_LUNG:
            _handle_result_chunk(data, "lung")
    else:
        _handle_text(data)


# ── Text / control handler ────────────────────────────────────────────────────

def _handle_text(data: bytearray):
    try:
        text = data.decode("ascii", errors="replace").strip()
    except Exception:
        return

    if text.startswith("START:"):
        try:
            state.reset()
            state.expected_bytes = int(text.split(":")[1])
            state.mode           = StreamMode.AUDIO
            print(f"\n  [AUDIO ] expecting {state.expected_bytes} B "
                  f"({state.expected_bytes / 2 / SAMPLE_RATE:.1f} s)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed START: {text!r}")

    elif text.startswith("finished"):
        state.audio_done = True
        state.mode       = StreamMode.IDLE
        print(f"\n  [AUDIO ] done — {len(state.audio_samples)}/{state.expected_bytes} B")

    elif text.startswith("MFCC_HEART_START:"):
        try:
            state.heart_expected_b = int(text.split(":")[1])
            state.mode             = StreamMode.MFCC_HEART
            n_frames = state.heart_expected_b // (HEART_N_MFCC * 4)
            print(f"\n  [HEART ] expecting {state.heart_expected_b} B "
                  f"({n_frames} frames × {HEART_N_MFCC} coeffs)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed MFCC_HEART_START: {text!r}")

    elif text.startswith("MFCC_HEART_END"):
        state.heart_done = True
        state.mode       = StreamMode.IDLE
        print(f"\n  [HEART ] done — {len(state.heart_bytes)}/{state.heart_expected_b} B")

    elif text.startswith("MFCC_LUNG_START:"):
        try:
            state.lung_expected_b = int(text.split(":")[1])
            state.mode            = StreamMode.MFCC_LUNG
            n_frames = state.lung_expected_b // (LUNG_N_MFCC * 4)
            print(f"\n  [LUNG  ] expecting {state.lung_expected_b} B "
                  f"({n_frames} frames × {LUNG_N_MFCC} coeffs)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed MFCC_LUNG_START: {text!r}")

    elif text.startswith("MFCC_LUNG_END"):
        state.lung_done = True
        state.mode      = StreamMode.IDLE
        print(f"\n  [LUNG  ] done — {len(state.lung_bytes)}/{state.lung_expected_b} B")

    elif text.startswith("RESULT_HEART_START:"):
        try:
            state.hr_expected_b = int(text.split(":")[1])
            state.mode          = StreamMode.RESULT_HEART
            print(f"\n  [HR    ] expecting {state.hr_expected_b} B (heart result)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed RESULT_HEART_START: {text!r}")

    elif text.startswith("RESULT_HEART_END"):
        state.hr_done = True
        state.mode    = StreamMode.IDLE
        result_str    = state.hr_bytes.decode("ascii", errors="replace").strip()
        print(f"\n  [HR    ] done — result: '{result_str}'")

    elif text.startswith("RESULT_LUNG_START:"):
        try:
            state.rr_expected_b = int(text.split(":")[1])
            state.mode          = StreamMode.RESULT_LUNG
            print(f"\n  [RR    ] expecting {state.rr_expected_b} B (lung result)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed RESULT_LUNG_START: {text!r}")

    elif text.startswith("RESULT_LUNG_END"):
        state.rr_done = True
        state.mode    = StreamMode.IDLE
        result_str    = state.rr_bytes.decode("ascii", errors="replace").strip()
        print(f"\n  [RR    ] done — result: '{result_str}'")

    elif text.startswith("SD:OK"):
        print(f"  [SD    ] SD files written OK")

    elif text.startswith("ERR:"):
        print(f"\n  [ERROR ] Firmware: {text}")
        # Mark everything done so we don't hang waiting
        state.audio_done = True
        state.heart_done = True
        state.lung_done  = True
        state.hr_done    = True
        state.rr_done    = True


# ── Chunk handlers ────────────────────────────────────────────────────────────

def _handle_audio_chunk(data: bytearray):
    seq     = struct.unpack_from("<H", data, 0)[0]
    ln      = struct.unpack_from("<H", data, 2)[0]
    payload = data[4: 4 + ln]

    if seq != state.audio_seq:
        gap_pkts  = (seq - state.audio_seq) & 0xFFFF
        gap_bytes = gap_pkts * ln
        print(f"\n  [GAP   ] audio seq {state.audio_seq}→{seq}")
        # Maintain int16 alignment when filling gaps.
        state.audio_samples += b"\x00\x00" * (gap_bytes // 2)
        state.audio_gaps    += gap_pkts

    state.audio_samples += payload
    state.audio_seq      = (seq + 1) & 0xFFFF
    state.audio_chunks  += 1
    _print_progress("AUDIO", len(state.audio_samples), state.expected_bytes)


def _handle_mfcc_chunk(data: bytearray, pipeline: str):
    seq     = struct.unpack_from("<H", data, 0)[0]
    ln      = struct.unpack_from("<H", data, 2)[0]
    payload = data[4: 4 + ln]

    if pipeline == "heart":
        if seq != state.heart_seq:
            gap_pkts = (seq - state.heart_seq) & 0xFFFF
            print(f"\n  [GAP   ] heart MFCC seq {state.heart_seq}→{seq}")
            state.heart_bytes += b"\x00" * (gap_pkts * ln)
            state.heart_gaps  += gap_pkts
        state.heart_bytes  += payload
        state.heart_seq     = (seq + 1) & 0xFFFF
        state.heart_chunks += 1
        _print_progress("HEART", len(state.heart_bytes), state.heart_expected_b)
    else:
        if seq != state.lung_seq:
            gap_pkts = (seq - state.lung_seq) & 0xFFFF
            print(f"\n  [GAP   ] lung MFCC seq {state.lung_seq}→{seq}")
            state.lung_bytes += b"\x00" * (gap_pkts * ln)
            state.lung_gaps  += gap_pkts
        state.lung_bytes  += payload
        state.lung_seq     = (seq + 1) & 0xFFFF
        state.lung_chunks += 1
        _print_progress("LUNG ", len(state.lung_bytes), state.lung_expected_b)


def _handle_result_chunk(data: bytearray, pipeline: str):
    # Result files are tiny text (e.g. "HR:72\n") — no gap detection needed
    ln      = struct.unpack_from("<H", data, 2)[0]
    payload = data[4: 4 + ln]
    if pipeline == "heart":
        state.hr_bytes += payload
    else:
        state.rr_bytes += payload


def _print_progress(label: str, got: int, expected: int):
    now = time.monotonic()
    if now - state.last_progress < 0.25:
        return
    state.last_progress = now
    pct = min(got * 100 // max(expected, 1), 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{label}][{bar}] {pct:3d}%  {got:>8}/{expected} B",
          end="", flush=True)


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_wav(samples: bytearray, filename: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = state.expected_bytes
    if len(samples) > target:
        samples = samples[:target]
    elif len(samples) < target:
        samples += b"\x00" * (target - len(samples))
    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples)
    kb  = filename.stat().st_size / 1024
    dur = len(samples) / 2 / SAMPLE_RATE
    print(f"  [SAVE  ] {filename}  ({kb:.1f} kB, {dur:.2f} s)")


def save_mfcc_npy(raw_bytes: bytearray, filename: Path,
                  n_frames: int, n_mfcc: int, label: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    expected = n_frames * n_mfcc * 4
    if len(raw_bytes) < expected:
        print(f"  [PAD   ] {label}: zero-padding {expected - len(raw_bytes)} bytes")
        raw_bytes += b"\x00" * (expected - len(raw_bytes))
    elif len(raw_bytes) > expected:
        raw_bytes = raw_bytes[:expected]

    arr = np.frombuffer(bytes(raw_bytes), dtype="<f4").reshape(n_frames, n_mfcc)
    np.save(str(filename), arr)

    kb = filename.stat().st_size / 1024
    print(f"  [SAVE  ] {filename}  ({kb:.1f} kB, shape={arr.shape}, dtype=float32)")

    means = arr.mean(axis=0)
    print(f"  [CHECK ] {label} mean c0..c4: "
          f"{means[0]:+.3f} {means[1]:+.3f} {means[2]:+.3f} "
          f"{means[3]:+.3f} {means[4]:+.3f}")
    print(f"  [CHECK ] {label} range: [{arr.min():.3f}, {arr.max():.3f}]")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        print(f"  [WARN  ] {label} contains NaN/Inf — check firmware DSP!")
    elif np.all(arr == 0):
        print(f"  [WARN  ] {label} is all zeros — ring overflow or pipeline error?")
    else:
        print(f"  [CHECK ] No NaN/Inf/all-zero ✓")


def save_result_txt(raw_bytes: bytearray, filename: Path, label: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = raw_bytes.decode("ascii", errors="replace").strip()
    filename.write_text(text + "\n")
    print(f"  [SAVE  ] {filename}  → '{text}'")


# ── Completion check ──────────────────────────────────────────────────────────

def _all_done() -> bool:
    """
    Wait for audio + both MFCCs, then give the result streams up to
    `RESULT_GRACE_S` to start. If they haven't arrived in that window,
    assume the firmware has results disabled (ENABLE_*_MODEL=0) and
    proceed.

    Why the grace period: firmware Phase 5 sleeps only ~20 ms between
    MFCC_LUNG_END and RESULT_HEART_START. The receiver polls every
    50 ms — without the grace period, ~1 in 3 RECs sees both
    *_done=True with *_expected_b=0 momentarily and bails out of the
    wait loop before the result streams ever start.
    """
    base_done = state.audio_done and state.heart_done and state.lung_done
    if not base_done:
        return False

    # First time we hit base_done, latch the timestamp.
    if state.base_done_at is None:
        state.base_done_at = time.monotonic()

    grace_elapsed = (time.monotonic() - state.base_done_at) > RESULT_GRACE_S
    hr_settled = state.hr_done or (state.hr_expected_b == 0 and grace_elapsed)
    rr_settled = state.rr_done or (state.rr_expected_b == 0 and grace_elapsed)
    return hr_settled and rr_settled


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run():
    print(f"Scanning for '{DEVICE_NAME}' …")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=SCAN_TIMEOUT)
    if device is None:
        print(f"ERROR: '{DEVICE_NAME}' not found.")
        sys.exit(1)

    print(f"  Found: {DEVICE_NAME} [{device.address}]")

    async with BleakClient(device) as client:
        mtu = getattr(client, "mtu_size", "?")
        print(f"Connected!  MTU = {mtu}")
        await client.start_notify(NUS_TX_CHAR_UUID, handle_notification)
        print("Subscribed to NUS notifications.\nPress Ctrl+C to stop.\n")

        rec_num = 1
        try:
            while True:
                inp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: input(
                        f"{'─'*50}\n"
                        f"[Ready] ENTER = start recording #{rec_num}  "
                        f"(q+ENTER = quit): "
                    )
                )
                if inp.strip().lower() == "q":
                    break

                state.reset()
                print(f"\nRecording #{rec_num} — sending REC …")
                await client.write_gatt_char(NUS_RX_CHAR_UUID, b"REC", response=False)
                print(f"  Sequence: audio → SD flush → heart MFCC → lung MFCC "
                      f"→ inference → results")
                print(f"  Timeout:  {REC_TIMEOUT:.0f} s total "
                      f"(result grace: {RESULT_GRACE_S:.0f} s)")

                deadline = time.monotonic() + REC_TIMEOUT
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                    if _all_done():
                        print(f"\n  [DONE  ] All streams complete.")
                        break
                else:
                    print(f"\n  [WARN  ] Timeout — saving whatever was received.")

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")

                # ── Save audio ──
                if state.audio_samples:
                    wav_fn = OUTPUT_DIR / f"rec_{rec_num:03d}_{ts}.wav"
                    print(f"\n  Audio: chunks={state.audio_chunks} "
                          f"bytes={len(state.audio_samples)}/{state.expected_bytes} "
                          f"gaps={state.audio_gaps}")
                    save_wav(bytearray(state.audio_samples), wav_fn)
                else:
                    print("  [WARN  ] No audio received — WAV not saved.")

                # ── Save heart MFCC ──
                if state.heart_bytes:
                    heart_fn = OUTPUT_DIR / f"heart_mfcc_{rec_num:03d}_{ts}.npy"
                    print(f"  Heart:  chunks={state.heart_chunks} "
                          f"bytes={len(state.heart_bytes)}/{state.heart_expected_b} "
                          f"gaps={state.heart_gaps}")
                    save_mfcc_npy(bytearray(state.heart_bytes), heart_fn,
                                  HEART_N_FRAMES, HEART_N_MFCC, "heart")
                else:
                    print("  [WARN  ] No heart MFCC data received.")

                # ── Save lung MFCC ──
                if state.lung_bytes:
                    lung_fn = OUTPUT_DIR / f"lung_mfcc_{rec_num:03d}_{ts}.npy"
                    print(f"  Lung:   chunks={state.lung_chunks} "
                          f"bytes={len(state.lung_bytes)}/{state.lung_expected_b} "
                          f"gaps={state.lung_gaps}")
                    save_mfcc_npy(bytearray(state.lung_bytes), lung_fn,
                                  LUNG_N_FRAMES, LUNG_N_MFCC, "lung")
                else:
                    print("  [WARN  ] No lung MFCC data received.")

                # ── Save heart result ──
                if state.hr_bytes:
                    hr_fn = OUTPUT_DIR / f"hr_{rec_num:03d}_{ts}.txt"
                    save_result_txt(bytearray(state.hr_bytes), hr_fn, "HR")
                else:
                    print("  [INFO  ] No heart result received "
                          "(model disabled or inference failed).")

                # ── Save lung result ──
                if state.rr_bytes:
                    rr_fn = OUTPUT_DIR / f"rr_{rec_num:03d}_{ts}.txt"
                    save_result_txt(bytearray(state.rr_bytes), rr_fn, "RR")
                else:
                    print("  [INFO  ] No lung result received "
                          "(model disabled or inference failed).")

                rec_num += 1

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            await client.stop_notify(NUS_TX_CHAR_UUID)
            print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(run())