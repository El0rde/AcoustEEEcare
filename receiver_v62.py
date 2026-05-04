#!/usr/bin/env python3
"""
AcoustEEEcare v6.2 — Host-side BLE receiver
Handles: audio WAV stream + on-device MFCC matrix stream.

After each recording the device runs the full DSP pipeline and streams:
  1. Audio PCM  →  saved as .wav
  2. MFCC matrix (float32, shape [n_frames, n_coeffs])  →  saved as .npy

MFCC wire protocol:
    "MFCC_START:<n_frames>:<n_coeffs>\\n"  — control message
    [seq u16 LE][len u16 LE][float32 bytes] — same chunked format as audio
    "MFCC_END\\n"                           — control message

Usage:
    pip install bleak numpy
    python receiver_v62.py
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

DEVICE_NAME  = "AcoustEEEcare"
SAMPLE_RATE  = 8000
OUTPUT_DIR   = Path("recordings_mfcc")
SCAN_TIMEOUT = 20.0
REC_TIMEOUT  = 100.0   # 10 s audio + ~1 s DSP + MFCC stream time


# ── State machine ────────────────────────────────────────────────────────────

class StreamMode:
    IDLE  = "idle"
    AUDIO = "audio"
    MFCC  = "mfcc"


class State:
    def __init__(self):
        self.reset()

    def reset(self):
        # Audio stream
        self.mode            = StreamMode.IDLE
        self.expected_bytes  = 0
        self.audio_done      = False
        self.audio_samples   = bytearray()
        self.audio_seq       = 0
        self.audio_gaps      = 0
        self.audio_chunks    = 0
        # MFCC stream
        self.mfcc_n_frames   = 0
        self.mfcc_n_coeffs   = 0
        self.mfcc_expected_b = 0
        self.mfcc_done       = False
        self.mfcc_bytes      = bytearray()
        self.mfcc_seq        = 0
        self.mfcc_gaps       = 0
        self.mfcc_chunks     = 0
        # Progress
        self.last_progress   = 0.0


state = State()


# ── Packet classifier ────────────────────────────────────────────────────────

def _is_data_chunk(data: bytearray) -> bool:
    """
    True if this looks like a sequenced data chunk: 4-byte header where
    bytes[2:4] == len(data) - 4 and > 0.
    Only valid when we are in AUDIO or MFCC streaming mode.
    """
    if state.mode == StreamMode.IDLE or len(data) < 5:
        return False
    ln = struct.unpack_from("<H", data, 2)[0]
    return ln > 0 and ln == len(data) - 4


def handle_notification(sender, data: bytearray):
    if _is_data_chunk(data):
        if state.mode == StreamMode.AUDIO:
            _handle_audio_chunk(data)
        elif state.mode == StreamMode.MFCC:
            _handle_mfcc_chunk(data)
    else:
        _handle_text(data)


# ── Text / control message handler ───────────────────────────────────────────

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
                  f"({state.expected_bytes // 2} samples, "
                  f"{state.expected_bytes / 2 / SAMPLE_RATE:.1f} s)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed START: {text!r}")

    elif text.startswith("finished"):
        state.audio_done = True
        got = len(state.audio_samples)
        exp = state.expected_bytes
        print(f"\n  [AUDIO ] 'finished' — {got}/{exp} B")

    elif text.startswith("MFCC_START:"):
        # "MFCC_START:<n_frames>:<n_coeffs>"
        try:
            parts              = text.split(":")
            state.mfcc_n_frames = int(parts[1])
            state.mfcc_n_coeffs = int(parts[2])
            state.mfcc_expected_b = (state.mfcc_n_frames
                                     * state.mfcc_n_coeffs * 4)  # float32
            state.mode         = StreamMode.MFCC
            print(f"\n  [MFCC  ] {state.mfcc_n_frames} frames × "
                  f"{state.mfcc_n_coeffs} coeffs = "
                  f"{state.mfcc_expected_b} B")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed MFCC_START: {text!r}")

    elif text.startswith("MFCC_END"):
        state.mfcc_done = True
        got = len(state.mfcc_bytes)
        exp = state.mfcc_expected_b
        print(f"\n  [MFCC  ] 'MFCC_END' — {got}/{exp} B")

    elif text.startswith("ERR:"):
        print(f"\n  [ERROR ] Firmware: {text}")
        state.audio_done = True
        state.mfcc_done  = True


# ── Audio chunk handler ───────────────────────────────────────────────────────

def _handle_audio_chunk(data: bytearray):
    seq     = struct.unpack_from("<H", data, 0)[0]
    ln      = struct.unpack_from("<H", data, 2)[0]
    payload = data[4: 4 + ln]

    if seq != state.audio_seq:
        gap_pkts  = (seq - state.audio_seq) & 0xFFFF
        gap_bytes = gap_pkts * ln
        print(f"\n  [GAP   ] audio seq {state.audio_seq}→{seq}: {gap_pkts} pkt(s)")
        state.audio_samples += b"\x00\x00" * (gap_bytes // 2)
        state.audio_gaps    += gap_pkts

    state.audio_samples += payload
    state.audio_seq      = (seq + 1) & 0xFFFF
    state.audio_chunks  += 1

    _print_audio_progress()


def _print_audio_progress():
    now = time.monotonic()
    if now - state.last_progress < 0.25:
        return
    state.last_progress = now
    got = len(state.audio_samples)
    exp = state.expected_bytes
    pct = min(got * 100 // max(exp, 1), 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%  {got:>7}/{exp} B  gaps={state.audio_gaps}",
          end="", flush=True)


# ── MFCC chunk handler ────────────────────────────────────────────────────────

def _handle_mfcc_chunk(data: bytearray):
    seq     = struct.unpack_from("<H", data, 0)[0]
    ln      = struct.unpack_from("<H", data, 2)[0]
    payload = data[4: 4 + ln]

    if seq != state.mfcc_seq:
        gap_pkts  = (seq - state.mfcc_seq) & 0xFFFF
        gap_bytes = gap_pkts * ln
        print(f"\n  [GAP   ] MFCC seq {state.mfcc_seq}→{seq}: {gap_pkts} pkt(s)")
        state.mfcc_bytes += b"\x00" * gap_bytes
        state.mfcc_gaps  += gap_pkts

    state.mfcc_bytes += payload
    state.mfcc_seq    = (seq + 1) & 0xFFFF
    state.mfcc_chunks += 1

    now = time.monotonic()
    if now - state.last_progress >= 0.25:
        state.last_progress = now
        got = len(state.mfcc_bytes)
        exp = state.mfcc_expected_b
        pct = min(got * 100 // max(exp, 1), 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [MFCC  ][{bar}] {pct:3d}%  {got:>7}/{exp} B",
              end="", flush=True)


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_wav(samples: bytearray, filename: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
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


def save_mfcc(mfcc_bytes: bytearray, filename: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    exp = state.mfcc_expected_b

    if len(mfcc_bytes) < exp:
        print(f"  [PAD   ] MFCC: {exp - len(mfcc_bytes)} bytes zero-padded")
        mfcc_bytes += b"\x00" * (exp - len(mfcc_bytes))
    elif len(mfcc_bytes) > exp:
        mfcc_bytes = mfcc_bytes[:exp]

    # Interpret as float32 and reshape to [n_frames, n_coeffs]
    arr = np.frombuffer(bytes(mfcc_bytes), dtype="<f4")
    arr = arr.reshape(state.mfcc_n_frames, state.mfcc_n_coeffs)
    np.save(str(filename), arr)

    kb = filename.stat().st_size / 1024
    print(f"  [SAVE  ] {filename}  ({kb:.1f} kB, shape={arr.shape})")

    # Quick sanity check: print the mean of each coefficient
    means = arr.mean(axis=0)
    print(f"  [CHECK ] MFCC mean per coeff (c0…c4): "
          f"{means[0]:+.3f} {means[1]:+.3f} {means[2]:+.3f} "
          f"{means[3]:+.3f} {means[4]:+.3f}")
    print(f"  [CHECK ] MFCC value range: [{arr.min():.3f}, {arr.max():.3f}]")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        print("  [WARN  ] MFCC contains NaN or Inf — check firmware DSP!")
    else:
        print("  [CHECK ] No NaN/Inf detected ✓")


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
                        f"{'─'*41}\n"
                        f"[Ready] ENTER = start recording #{rec_num}  "
                        f"(q+ENTER = quit): "
                    )
                )
                if inp.strip().lower() == "q":
                    break

                state.reset()
                print(f"\nRecording #{rec_num} — sending REC …")
                await client.write_gatt_char(NUS_RX_CHAR_UUID, b"REC", response=False)
                print(f"  Timeout: {REC_TIMEOUT:.0f} s  "
                      f"(10 s audio + ~1 s DSP + MFCC stream)")

                deadline = time.monotonic() + REC_TIMEOUT
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                    # Both streams must complete before we declare done
                    if state.audio_done and state.mfcc_done:
                        print(f"\n  [DONE  ] Audio + MFCC both complete.")
                        break
                else:
                    print(f"\n  [WARN  ] Timeout — saving partial data.")

                # ── Save audio ──
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                if state.audio_samples:
                    wav_fn = OUTPUT_DIR / f"rec_{rec_num:03d}_{ts}.wav"
                    got = len(state.audio_samples)
                    exp = state.expected_bytes
                    print(f"\n  Audio chunks={state.audio_chunks}  "
                          f"bytes={got}/{exp}  gaps={state.audio_gaps}")
                    save_wav(bytearray(state.audio_samples), wav_fn)
                else:
                    print("  [WARN  ] No audio — WAV not saved.")

                # ── Save MFCC ──
                if state.mfcc_bytes and state.mfcc_n_frames > 0:
                    npy_fn = OUTPUT_DIR / f"mfcc_{rec_num:03d}_{ts}.npy"
                    got = len(state.mfcc_bytes)
                    exp = state.mfcc_expected_b
                    print(f"  MFCC  chunks={state.mfcc_chunks}  "
                          f"bytes={got}/{exp}  gaps={state.mfcc_gaps}")
                    save_mfcc(bytearray(state.mfcc_bytes), npy_fn)
                else:
                    print("  [WARN  ] No MFCC data received.")

                rec_num += 1

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            await client.stop_notify(NUS_TX_CHAR_UUID)
            print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(run())
