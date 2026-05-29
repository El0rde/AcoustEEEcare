#!/usr/bin/env python3
"""
AcoustEEEcare — Host-side BLE receiver matched to firmware v5.9
============================================================
Firmware v5.9 wire protocol (BLE-only, no SD card, no MFCC):

  Command:  "REC"  — triggers record + stream on the device

  Device sends:
    "START:<audio_bytes>\\n"   — stream header (text)
    [raw int16 LE audio bytes] — chunks up to MTU-3 bytes, NO seq/len framing
    "finished\\n"              — stream footer (text)

  Error messages:
    "ERR:<CODE>\\n"            — e.g. ERR:SAADC, ERR:ABORT

Removed vs v6.4:
  - Packet seq/len framing expectation (_is_data_chunk removed)
  - MFCC stream handling (firmware v5.9 has none)
  - SD-card mode / REC:OK / SEND command (firmware v5.9 has none)
  - Gap detection (no sequence numbers to track)

Usage:
    pip install bleak numpy
    python receiver_v59.py
"""

import asyncio
import wave
import time
import sys
from datetime import datetime
from pathlib import Path

from bleak import BleakScanner, BleakClient

# ── BLE / device constants ─────────────────────────────────────────────────────

NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

DEVICE_NAME  = "AcoustEEEcare"
SAMPLE_RATE  = 8000          # must match SAMPLING_RATE in firmware
OUTPUT_DIR   = Path("recording_29_05_2026_flexible_pcb")
SCAN_TIMEOUT = 20.0
REC_TIMEOUT  = 60.0          # 10 s record + audio stream + margin


# ── Packet stats ───────────────────────────────────────────────────────────────

class PacketStats:
    """
    Tracks per-session BLE transfer statistics.
    No gap/loss counting — firmware sends raw bytes without sequence numbers.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.audio_pkts_rx       = 0   # BLE notifications counted as audio
        self.audio_bytes_rx      = 0   # payload bytes actually received
        self.audio_bytes_exp     = 0   # from START:<n>
        self.total_notifications = 0   # every handle_notification call
        self.total_raw_bytes     = 0   # sum of len(data) for all notifications
        self.session_start_time  = None
        self.session_end_time    = None

    def start_session(self):
        self.session_start_time = time.monotonic()

    def end_session(self):
        self.session_end_time = time.monotonic()

    @property
    def session_duration(self):
        if self.session_start_time and self.session_end_time:
            return self.session_end_time - self.session_start_time
        return 0.0

    @property
    def audio_rx_pct(self):
        return self.audio_bytes_rx * 100 // max(self.audio_bytes_exp, 1)

    @property
    def audio_throughput_kbps(self):
        dur = self.session_duration
        return (self.audio_bytes_rx * 8) / dur / 1000.0 if dur > 0 else 0.0

    def print_report(self):
        dur = self.session_duration
        sep = "━" * 58
        print(f"\n{sep}")
        print("  BLE SESSION STATISTICS")
        print(sep)
        print(f"  Duration              : {dur:.2f} s")
        print(f"  Total notifications   : {self.total_notifications}")
        print(f"  Total raw BLE bytes   : {self.total_raw_bytes:,} B "
              f"({self.total_raw_bytes / 1024:.1f} kB)")
        print()
        print("  ── Audio stream ──────────────────────────────────")
        print(f"  Packets received      : {self.audio_pkts_rx}")
        print(f"  Payload bytes received: {self.audio_bytes_rx:,} B "
              f"({self.audio_bytes_rx / 1024:.1f} kB)")
        print(f"  Payload bytes expected: {self.audio_bytes_exp:,} B "
              f"({self.audio_bytes_exp / 1024:.1f} kB)  "
              f"→ {self.audio_rx_pct}% received")
        print(f"  Throughput (payload)  : {self.audio_throughput_kbps:.1f} kbps")
        print(sep)


pstats = PacketStats()


# ── Stream state machine ───────────────────────────────────────────────────────

class StreamMode:
    IDLE  = "idle"
    AUDIO = "audio"


class State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode           = StreamMode.IDLE
        self.expected_bytes = 0
        self.audio_done     = False
        self.audio_samples  = bytearray()
        self.audio_chunks   = 0
        self.error          = None
        self.last_progress  = 0.0

        pstats.reset()
        pstats.start_session()


state = State()


# ── Notification router ────────────────────────────────────────────────────────

# Known text prefixes sent by the firmware.
# Audio sample bytes are raw int16 LE and will not cleanly decode as ASCII
# strings beginning with any of these prefixes.
_CONTROL_PREFIXES = ("START:", "finished", "ERR:")


def _is_control_message(data: bytearray) -> bool:
    """
    Returns True if data looks like a firmware text control message.
    Decoding must succeed as pure ASCII and start with a known prefix.
    Raw int16 audio bytes very rarely satisfy both conditions simultaneously.
    """
    try:
        text = data.decode("ascii").strip()
        return any(text.startswith(p) for p in _CONTROL_PREFIXES)
    except (UnicodeDecodeError, ValueError):
        return False


def handle_notification(_sender, data: bytearray):
    pstats.total_notifications += 1
    pstats.total_raw_bytes     += len(data)

    # While streaming audio, route to audio handler unless this is a
    # control message (finished / ERR:) arriving at the end of the stream.
    if state.mode == StreamMode.AUDIO and not _is_control_message(data):
        _handle_audio_chunk(data)
    else:
        _handle_text(data)


# ── Control / text message handler ────────────────────────────────────────────

def _handle_text(data: bytearray):
    try:
        text = data.decode("ascii", errors="replace").strip()
    except Exception:
        return

    # ── Audio stream header ──
    if text.startswith("START:"):
        try:
            n = int(text.split(":")[1])
            state.expected_bytes   = n
            state.audio_samples    = bytearray()
            state.audio_chunks     = 0
            state.audio_done       = False
            state.mode             = StreamMode.AUDIO
            pstats.audio_bytes_exp = n

            dur = n / 2 / SAMPLE_RATE
            print(f"\n  [AUDIO ] START — expecting {n} B "
                  f"({n // 2} samples, {dur:.1f} s @ {SAMPLE_RATE} Hz)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed START: {text!r}")

    # ── Audio stream footer ──
    elif text.startswith("finished"):
        state.audio_done = True
        got = len(state.audio_samples)
        exp = state.expected_bytes
        pct = min(got * 100 // max(exp, 1), 100)
        print(f"\n  [AUDIO ] finished — {got}/{exp} B ({pct}%)"
              f"  pkts_rx={pstats.audio_pkts_rx}")
        state.mode = StreamMode.IDLE

    # ── Firmware error ──
    elif text.startswith("ERR:"):
        state.error      = text
        state.audio_done = True          # unblock the wait loop
        state.mode       = StreamMode.IDLE
        print(f"\n  [ERROR ] Firmware reported: {text}")

    # ── Unknown / debug string ──
    else:
        print(f"\n  [FW    ] {text!r}")


# ── Audio chunk handler ────────────────────────────────────────────────────────

def _handle_audio_chunk(data: bytearray):
    """
    Firmware v5.9 sends raw int16 LE audio bytes with no seq/len header.
    Append the entire notification payload directly to the sample buffer.
    """
    pstats.audio_pkts_rx  += 1
    pstats.audio_bytes_rx += len(data)
    state.audio_samples   += data
    state.audio_chunks    += 1
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
    print(f"\r  [{bar}] {pct:3d}%  {got:>7,}/{exp:,} B  "
          f"pkts={pstats.audio_pkts_rx}",
          end="", flush=True)


# ── Save helpers ───────────────────────────────────────────────────────────────

def save_wav(samples: bytearray, filename: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    target = state.expected_bytes

    # Trim or zero-pad to exact expected length
    if len(samples) > target:
        samples = samples[:target]
    elif len(samples) < target:
        pad = target - len(samples)
        print(f"  [PAD   ] {pad} bytes zero-padded to reach expected length")
        samples += b"\x00" * pad

    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # int16 → 2 bytes per sample
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples)

    kb  = filename.stat().st_size / 1024
    dur = len(samples) / 2 / SAMPLE_RATE
    print(f"  [SAVE  ] {filename}  ({kb:.1f} kB, {dur:.2f} s)")


# ── Session completion / wait ──────────────────────────────────────────────────

def _session_complete() -> bool:
    """True once the audio stream ends cleanly or the firmware signals an error."""
    return bool(state.error) or state.audio_done


async def _wait_for_completion(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        if _session_complete():
            return True
    return False


# ── Save session outputs ───────────────────────────────────────────────────────

def _save_session(rec_num: int) -> None:
    pstats.end_session()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if state.audio_samples and state.expected_bytes > 0:
        wav_fn = OUTPUT_DIR / f"rec_{rec_num:03d}_{ts}.wav"
        save_wav(bytearray(state.audio_samples), wav_fn)
    else:
        print("  [WARN  ] No audio received — WAV not saved.")

    pstats.print_report()


# ── Main ───────────────────────────────────────────────────────────────────────

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
        print("  q + ENTER  — quit")
        print()

        rec_num = 1
        try:
            while True:
                sep = "─" * 55
                inp = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: input(
                        f"{sep}\n"
                        f"[Ready #{rec_num}]  ENTER=REC  q=quit: "
                    )
                )

                if inp.strip().lower() == "q":
                    print("Quitting.")
                    break

                # ── REC ──
                state.reset()
                print("\nSending REC command…")
                await client.write_gatt_char(
                    NUS_RX_CHAR_UUID, b"REC", response=False
                )
                print(f"  Waiting up to {REC_TIMEOUT:.0f} s "
                      f"(10 s record + audio stream)…")

                completed = await _wait_for_completion(REC_TIMEOUT)

                if not completed:
                    print("\n  [WARN  ] Timeout — saving partial data.")
                elif state.error:
                    print(f"\n  [ERROR ] Firmware error: {state.error}")
                else:
                    print("\n  [DONE  ] REC complete.")

                _save_session(rec_num)
                rec_num += 1

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            await client.stop_notify(NUS_TX_CHAR_UUID)
            print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(run())