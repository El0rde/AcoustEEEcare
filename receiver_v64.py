#!/usr/bin/env python3
"""
AcoustEEEcare v6.4 — Host-side BLE receiver
============================================================
Supports both firmware compile modes:

  USE_SD true  (SD-first, default in v6.4):
    Device sends: "REC:OK\\n" → MFCC stream → audio stream
    Commands:     "REC"  — record + MFCC + audio pipeline
                  "SEND" — retransmit last SD recording (no re-record)

  USE_SD false (BLE-only, v6.3 behaviour):
    Device sends: "START:<n>\\n" → audio stream → MFCC stream
    Command:      "REC"  — record + stream pipeline

MFCC wire protocol (both modes):
    "MFCC_START:<n_frames>:<n_coeffs>\\n"   — header
    [seq u16 LE][len u16 LE][float32 bytes]  — one packet per frame
    "MFCC_END\\n"                            — trailer

Audio wire protocol:
    USE_SD true:  "REC:OK\\n" (end of recording phase)
                  then MFCC stream (see above)
                  then "START:<audio_bytes>\\n" → chunks → "finished\\n"
    USE_SD false: "START:<audio_bytes>\\n" → chunks → "finished\\n"
                  then MFCC stream (see above)

Error messages from firmware:
    "ERR:<CODE>\\n"  — e.g. ERR:NOSD, ERR:CORRUPT, ERR:TIMEOUT

Usage:
    pip install bleak numpy
    python receiver_v64.py
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

# ── BLE / device constants ────────────────────────────────────────────────────

NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

DEVICE_NAME  = "AcoustEEEcare"
SAMPLE_RATE  = 8000
OUTPUT_DIR   = Path("recordings_v64")
SCAN_TIMEOUT = 20.0

# Generous timeout: 10 s record + MFCC stream + audio stream + margin
REC_TIMEOUT  = 120.0
# Timeout used for SEND command (no recording phase)
SEND_TIMEOUT = 60.0


# ── Stream state machine ──────────────────────────────────────────────────────

class StreamMode:
    IDLE  = "idle"
    AUDIO = "audio"
    MFCC  = "mfcc"


class State:
    """Holds all receiver state for one recording session."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Session flags
        self.rec_ok          = False   # "REC:OK" received  (USE_SD true only)
        self.error           = None    # last ERR: string from firmware

        # Stream routing
        self.mode            = StreamMode.IDLE

        # Audio stream
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

        # Progress display throttle
        self.last_progress   = 0.0


state = State()


# ── Packet classifier ─────────────────────────────────────────────────────────

def _is_data_chunk(data: bytearray) -> bool:
    """
    True when the packet looks like a sequenced data chunk:
      bytes[0:2] = seq (u16 LE)
      bytes[2:4] = payload_len (u16 LE)
      len(data)  = 4 + payload_len
    Only checked while we are in AUDIO or MFCC mode to avoid false
    positives on short control strings.
    """
    if state.mode == StreamMode.IDLE or len(data) < 5:
        return False
    payload_len = struct.unpack_from("<H", data, 2)[0]
    return payload_len > 0 and payload_len == len(data) - 4


def handle_notification(_sender, data: bytearray):
    if _is_data_chunk(data):
        if state.mode == StreamMode.AUDIO:
            _handle_audio_chunk(data)
        elif state.mode == StreamMode.MFCC:
            _handle_mfcc_chunk(data)
    else:
        _handle_text(data)


# ── Control / text message handler ───────────────────────────────────────────

def _handle_text(data: bytearray):
    try:
        text = data.decode("ascii", errors="replace").strip()
    except Exception:
        return

    # ── USE_SD true: recording phase complete ──
    if text == "REC:OK":
        state.rec_ok = True
        print("\n  [PHASE1] Recording to SD complete — MFCC stream incoming")

    # ── Audio stream header ──
    elif text.startswith("START:"):
        try:
            n = int(text.split(":")[1])
            state.expected_bytes = n
            state.audio_samples  = bytearray()
            state.audio_seq      = 0
            state.audio_gaps     = 0
            state.audio_chunks   = 0
            state.audio_done     = False
            state.mode           = StreamMode.AUDIO
            dur = n / 2 / SAMPLE_RATE
            print(f"\n  [AUDIO ] START — expecting {n} B "
                  f"({n // 2} samples, {dur:.1f} s)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed START: {text!r}")

    # ── Audio stream footer ──
    elif text.startswith("finished"):
        state.audio_done = True
        got = len(state.audio_samples)
        exp = state.expected_bytes
        pct = min(got * 100 // max(exp, 1), 100)
        print(f"\n  [AUDIO ] finished — {got}/{exp} B ({pct}%)"
              f"  chunks={state.audio_chunks}  gaps={state.audio_gaps}")
        state.mode = StreamMode.IDLE

    # ── MFCC stream header ──
    elif text.startswith("MFCC_START:"):
        try:
            parts = text.split(":")
            nf    = int(parts[1])
            nc    = int(parts[2])
            state.mfcc_n_frames   = nf
            state.mfcc_n_coeffs   = nc
            state.mfcc_expected_b = nf * nc * 4   # float32
            state.mfcc_bytes      = bytearray()
            state.mfcc_seq        = 0
            state.mfcc_gaps       = 0
            state.mfcc_chunks     = 0
            state.mfcc_done       = False
            state.mode            = StreamMode.MFCC
            print(f"\n  [MFCC  ] START — {nf} frames × {nc} coeffs "
                  f"= {state.mfcc_expected_b} B")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed MFCC_START: {text!r}")

    # ── MFCC stream footer ──
    elif text.startswith("MFCC_END"):
        state.mfcc_done = True
        got = len(state.mfcc_bytes)
        exp = state.mfcc_expected_b
        pct = min(got * 100 // max(exp, 1), 100)
        print(f"\n  [MFCC  ] MFCC_END — {got}/{exp} B ({pct}%)"
              f"  chunks={state.mfcc_chunks}  gaps={state.mfcc_gaps}")
        state.mode = StreamMode.IDLE

    # ── Firmware error ──
    elif text.startswith("ERR:"):
        state.error      = text
        state.audio_done = True   # unblock any wait loop
        state.mfcc_done  = True
        state.mode       = StreamMode.IDLE
        print(f"\n  [ERROR ] Firmware reported: {text}")

    else:
        # Unknown / debug string from firmware
        print(f"\n  [FW    ] {text!r}")


# ── Audio chunk handler ───────────────────────────────────────────────────────

def _handle_audio_chunk(data: bytearray):
    seq     = struct.unpack_from("<H", data, 0)[0]
    ln      = struct.unpack_from("<H", data, 2)[0]
    payload = data[4: 4 + ln]

    if seq != state.audio_seq:
        gap_pkts  = (seq - state.audio_seq) & 0xFFFF
        gap_bytes = gap_pkts * ln
        print(f"\n  [GAP   ] audio seq {state.audio_seq}→{seq} "
              f"({gap_pkts} missing, ~{gap_bytes} B zero-filled)")
        # Zero-fill the gap so the WAV timestamp stays correct
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
    print(f"\r  [{bar}] {pct:3d}%  {got:>7}/{exp} B  "
          f"chunks={state.audio_chunks}  gaps={state.audio_gaps}",
          end="", flush=True)


# ── MFCC chunk handler ────────────────────────────────────────────────────────

def _handle_mfcc_chunk(data: bytearray):
    seq     = struct.unpack_from("<H", data, 0)[0]
    ln      = struct.unpack_from("<H", data, 2)[0]
    payload = data[4: 4 + ln]

    if seq != state.mfcc_seq:
        gap_pkts  = (seq - state.mfcc_seq) & 0xFFFF
        gap_bytes = gap_pkts * ln
        print(f"\n  [GAP   ] MFCC seq {state.mfcc_seq}→{seq} "
              f"({gap_pkts} missing, ~{gap_bytes} B zero-filled)")
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
        print(f"\r  [MFCC  ][{bar}] {pct:3d}%  {got:>7}/{exp} B  "
              f"chunks={state.mfcc_chunks}  gaps={state.mfcc_gaps}",
              end="", flush=True)


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_wav(samples: bytearray, filename: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    target = state.expected_bytes

    # Trim or pad to exact expected size
    if len(samples) > target:
        samples = samples[:target]
    elif len(samples) < target:
        pad = target - len(samples)
        print(f"  [PAD   ] Audio: {pad} bytes zero-padded")
        samples += b"\x00" * pad

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
        pad = exp - len(mfcc_bytes)
        print(f"  [PAD   ] MFCC: {pad} bytes zero-padded")
        mfcc_bytes += b"\x00" * pad
    elif len(mfcc_bytes) > exp:
        mfcc_bytes = mfcc_bytes[:exp]

    arr = np.frombuffer(bytes(mfcc_bytes), dtype="<f4")
    arr = arr.reshape(state.mfcc_n_frames, state.mfcc_n_coeffs)
    np.save(str(filename), arr)

    kb = filename.stat().st_size / 1024
    print(f"  [SAVE  ] {filename}  ({kb:.1f} kB, shape={arr.shape})")

    means = arr.mean(axis=0)
    print(f"  [CHECK ] MFCC mean per coeff (c0…c4): "
          f"{means[0]:+.3f} {means[1]:+.3f} {means[2]:+.3f} "
          f"{means[3]:+.3f} {means[4]:+.3f}")
    print(f"  [CHECK ] MFCC value range: [{arr.min():.3f}, {arr.max():.3f}]")
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        print("  [WARN  ] MFCC contains NaN or Inf — check firmware DSP!")
    else:
        print("  [CHECK ] No NaN/Inf detected ✓")


# ── Session completion detector ───────────────────────────────────────────────

def _session_complete(use_sd_mode: bool) -> bool:
    """
    Return True when a full session has been received.

    USE_SD true  pipeline order: REC:OK → MFCC_END → finished
    USE_SD false pipeline order: finished → MFCC_END

    Either way we need both audio_done and mfcc_done (or an error).
    """
    if state.error:
        return True
    return state.audio_done and state.mfcc_done


# ── Wait loop ─────────────────────────────────────────────────────────────────

async def _wait_for_completion(timeout: float, use_sd_mode: bool) -> bool:
    """
    Poll until _session_complete() or timeout.
    Returns True if completed cleanly, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        if _session_complete(use_sd_mode):
            return True
    return False


# ── Save session outputs ──────────────────────────────────────────────────────

def _save_session(rec_num: int) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if state.audio_samples and state.expected_bytes > 0:
        wav_fn = OUTPUT_DIR / f"rec_{rec_num:03d}_{ts}.wav"
        save_wav(bytearray(state.audio_samples), wav_fn)
    else:
        print("  [WARN  ] No audio received — WAV not saved.")

    if state.mfcc_bytes and state.mfcc_n_frames > 0:
        npy_fn = OUTPUT_DIR / f"mfcc_{rec_num:03d}_{ts}.npy"
        save_mfcc(bytearray(state.mfcc_bytes), npy_fn)
    else:
        print("  [WARN  ] No MFCC data received — .npy not saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

async def run():
    print(f"Scanning for '{DEVICE_NAME}' (timeout={SCAN_TIMEOUT:.0f} s)…")
    device = await BleakScanner.find_device_by_name(
        DEVICE_NAME, timeout=SCAN_TIMEOUT
    )
    if device is None:
        print(f"ERROR: '{DEVICE_NAME}' not found within {SCAN_TIMEOUT:.0f} s.")
        sys.exit(1)

    print(f"  Found: {DEVICE_NAME}  [{device.address}]")

    async with BleakClient(device) as client:
        mtu = getattr(client, "mtu_size", "?")
        print(f"Connected!  MTU = {mtu}")
        await client.start_notify(NUS_TX_CHAR_UUID, handle_notification)
        print("Subscribed to NUS notifications.")
        print()
        print("Commands:")
        print("  ENTER      — send REC (record + stream)")
        print("  s + ENTER  — send SEND (retransmit last SD recording, USE_SD=true only)")
        print("  q + ENTER  — quit")
        print()

        rec_num = 1
        try:
            while True:
                sep = "─" * 55
                inp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: input(
                        f"{sep}\n"
                        f"[Ready #{rec_num}]  ENTER=REC  s=SEND  q=quit: "
                    )
                )

                cmd = inp.strip().lower()

                if cmd == "q":
                    print("Quitting.")
                    break

                elif cmd == "s":
                    # ── SEND command (USE_SD true only) ──
                    state.reset()
                    print(f"\nSending SEND command (retransmit from SD)…")
                    await client.write_gatt_char(
                        NUS_RX_CHAR_UUID, b"SEND", response=False
                    )
                    print(f"  Waiting up to {SEND_TIMEOUT:.0f} s for streams…")
                    completed = await _wait_for_completion(
                        SEND_TIMEOUT, use_sd_mode=True
                    )
                    if not completed:
                        print("\n  [WARN  ] Timeout — saving partial data.")
                    elif state.error:
                        print(f"\n  [ERROR ] Firmware error: {state.error}")
                    else:
                        print(f"\n  [DONE  ] SEND complete.")
                    _save_session(rec_num)
                    rec_num += 1

                else:
                    # ── REC command ──
                    state.reset()
                    print(f"\nSending REC command…")
                    await client.write_gatt_char(
                        NUS_RX_CHAR_UUID, b"REC", response=False
                    )
                    print(
                        f"  Waiting up to {REC_TIMEOUT:.0f} s "
                        f"(10 s record + MFCC + audio stream)…"
                    )

                    # Detect firmware mode from first message:
                    #   USE_SD true  → first message is "REC:OK"
                    #   USE_SD false → first message is "START:<n>"
                    # We wait for either audio_done+mfcc_done or error.
                    completed = await _wait_for_completion(
                        REC_TIMEOUT, use_sd_mode=True   # permissive; handles both
                    )
                    if not completed:
                        print("\n  [WARN  ] Timeout — saving partial data.")
                    elif state.error:
                        print(f"\n  [ERROR ] Firmware error: {state.error}")
                    else:
                        print(f"\n  [DONE  ] REC complete.")
                    _save_session(rec_num)
                    rec_num += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            await client.stop_notify(NUS_TX_CHAR_UUID)
            print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(run())
    