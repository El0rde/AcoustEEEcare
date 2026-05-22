#!/usr/bin/env python3
"""
AcoustEEEcare — BLE Receiver (new firmware: stream-to-phone + on-device inference)

Firmware stream order (new architecture, no SD):
  1. START:<n>\n          — n = total raw audio bytes
  2. [binary audio chunks] — framed as [seq:u16][len:u16][payload]
  3. finished\n            — audio stream complete
  4. HR:<value>\n          — heart rate result (or ERR:HEART_INF\n)
  5. RR:<value>\n          — respiration rate result (or ERR:LUNG_INF\n)

Other control messages from firmware:
  ERR:FRAMES\n   — wrong frame count (DSP issue)
  ERR:SAADC\n    — ADC failed to start
  ERR:TIMEOUT\n  — SAADC timed out mid-capture

Files saved to ./recordings/:
  rec_NNN_TIMESTAMP.wav     — raw audio (8 kHz, mono, int16)
  results_NNN_TIMESTAMP.txt — HR and RR values
"""

import asyncio
import struct
import wave
import time
import sys
from datetime import datetime
from pathlib import Path

from bleak import BleakScanner, BleakClient

# ── BLE UUIDs (Nordic UART Service) ──────────────────────────────────────────
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE_NAME   = "AcoustEEEcare"
SAMPLE_RATE   = 8000
OUTPUT_DIR    = Path("final_device_recording_22_05_26")
SCAN_TIMEOUT  = 20.0
REC_TIMEOUT   = 60.0   # 10 s capture + ~15 s inference + margin

CHUNK_HEADER  = 4      # [seq:u16][len:u16]


# ── State ─────────────────────────────────────────────────────────────────────

class StreamMode:
    IDLE  = "idle"
    AUDIO = "audio"


class State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode           = StreamMode.IDLE
        self.expected_bytes = 0
        self.audio_samples  = bytearray()
        self.audio_seq      = 0
        self.audio_gaps     = 0
        self.audio_chunks   = 0
        self.audio_done     = False
        self.hr_value       = None   # float or None
        self.rr_value       = None   # float or None
        self.hr_done        = False
        self.rr_done        = False
        self.error          = None   # last ERR: string if any
        self.last_progress  = 0.0


state = State()


# ── Notification handler ──────────────────────────────────────────────────────

def _is_audio_chunk(data: bytearray) -> bool:
    """Binary audio chunk: [seq:u16][len:u16][payload]. len must match."""
    if state.mode != StreamMode.AUDIO or len(data) < CHUNK_HEADER + 1:
        return False
    ln = struct.unpack_from("<H", data, 2)[0]
    return ln > 0 and ln == len(data) - CHUNK_HEADER


def handle_notification(sender, data: bytearray):
    if _is_audio_chunk(data):
        _handle_audio_chunk(data)
    else:
        _handle_text(data)


def _handle_text(data: bytearray):
    try:
        text = data.decode("ascii", errors="replace").strip()
    except Exception:
        return

    if not text:
        return

    # ── Audio stream start ────────────────────────────────────────
    if text.startswith("START:"):
        try:
            state.reset()
            state.expected_bytes = int(text.split(":")[1])
            state.mode           = StreamMode.AUDIO
            dur = state.expected_bytes / 2 / SAMPLE_RATE
            print(f"\n  [AUDIO ] START — expecting {state.expected_bytes} B "
                  f"({dur:.1f} s @ {SAMPLE_RATE} Hz)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed START: {text!r}")

    # ── Audio stream end ──────────────────────────────────────────
    elif text.startswith("finished"):
        state.audio_done = True
        state.mode       = StreamMode.IDLE
        print(f"\n  [AUDIO ] finished — "
              f"{len(state.audio_samples)}/{state.expected_bytes} B  "
              f"chunks={state.audio_chunks}  gaps={state.audio_gaps}")

    # ── Inference results ─────────────────────────────────────────
    elif text.startswith("HR:"):
        try:
            state.hr_value = float(text.split(":")[1])
            state.hr_done  = True
            print(f"\n  [HR    ] Heart Rate  = {state.hr_value:.0f} BPM")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed HR: {text!r}")
            state.hr_done = True

    elif text.startswith("RR:"):
        try:
            state.rr_value = float(text.split(":")[1])
            state.rr_done  = True
            print(f"\n  [RR    ] Resp Rate   = {state.rr_value:.0f} BrPM")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed RR: {text!r}")
            state.rr_done = True

    # ── Errors ────────────────────────────────────────────────────
    elif text.startswith("ERR:"):
        state.error = text
        print(f"\n  [ERROR ] Firmware: {text}")
        # Mark all pending streams done so we don't hang
        state.audio_done = True
        state.hr_done    = True
        state.rr_done    = True

    else:
        print(f"\n  [MSG   ] {text!r}")


def _handle_audio_chunk(data: bytearray):
    seq     = struct.unpack_from("<H", data, 0)[0]
    ln      = struct.unpack_from("<H", data, 2)[0]
    payload = data[CHUNK_HEADER: CHUNK_HEADER + ln]

    if seq != state.audio_seq:
        gap_pkts  = (seq - state.audio_seq) & 0xFFFF
        gap_bytes = gap_pkts * ln
        print(f"\n  [GAP   ] audio seq {state.audio_seq}→{seq} "
              f"({gap_pkts} pkts, ~{gap_bytes} B)")
        state.audio_samples += b"\x00\x00" * (gap_bytes // 2)
        state.audio_gaps    += gap_pkts

    state.audio_samples += payload
    state.audio_seq      = (seq + 1) & 0xFFFF
    state.audio_chunks  += 1
    _print_progress(len(state.audio_samples), state.expected_bytes)


def _print_progress(got: int, expected: int):
    now = time.monotonic()
    if now - state.last_progress < 0.25:
        return
    state.last_progress = now
    pct = min(got * 100 // max(expected, 1), 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [AUDIO ][{bar}] {pct:3d}%  {got:>8}/{expected} B",
          end="", flush=True)


# ── Completion check ──────────────────────────────────────────────────────────

def _all_done() -> bool:
    return state.audio_done and state.hr_done and state.rr_done


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


def save_results(hr, rr, filename: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    if hr is not None:
        lines.append(f"HR:{hr:.0f}")
    if rr is not None:
        lines.append(f"RR:{rr:.0f}")
    text = "\n".join(lines) + "\n" if lines else "(no results)\n"
    filename.write_text(text)
    print(f"  [SAVE  ] {filename}  -> {text.strip()!r}")


# ── Device discovery ──────────────────────────────────────────────────────────

async def find_device():
    """
    Scan and return the BLEDevice for AcoustEEEcare.
    Matches by name first; falls back to NUS service UUID when Windows
    returns name=None on a first passive scan (stale cache).
    """
    print(f"Scanning for '{DEVICE_NAME}' (timeout={SCAN_TIMEOUT:.0f} s) …")
    results = await BleakScanner.discover(timeout=SCAN_TIMEOUT, return_adv=True)

    # Pass 1: exact name match
    for device, adv in results.values():
        if device.name == DEVICE_NAME:
            print(f"  Found: {device.name} [{device.address}]  RSSI={adv.rssi}")
            return device

    # Pass 2: NUS service UUID match (Windows often caches name=None on first scan)
    for device, adv in results.values():
        service_uuids = [u.lower() for u in (adv.service_uuids or [])]
        if NUS_SERVICE_UUID in service_uuids:
            print(f"  Found via NUS UUID: [{device.address}]  "
                  f"name={device.name!r}  RSSI={adv.rssi}")
            print(f"  (name was not resolved on this scan — connecting anyway)")
            return device

    # Not found — show what was seen to help debug
    print(f"\nERROR: '{DEVICE_NAME}' not found in scan.")
    print("Nearby devices were:")
    for device, adv in sorted(results.values(),
                               key=lambda x: x[1].rssi or -999,
                               reverse=True):
        print(f"  RSSI={adv.rssi:4d}  {device.address}  name={device.name!r}")
    print("\nTips:")
    print("  • Power-cycle the MCU and try again")
    print("  • Toggle Bluetooth off/on in Windows Settings")
    return None


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run():
    device = await find_device()
    if device is None:
        sys.exit(1)

    print(f"Connecting to {device.address} …")
    async with BleakClient(device.address) as client:
        mtu = getattr(client, "mtu_size", "?")
        print(f"Connected!  MTU = {mtu}")
        await client.start_notify(NUS_TX_CHAR_UUID, handle_notification)
        print("Subscribed to NUS notifications.")
        print("Firmware stream: audio -> finished -> HR:<n> -> RR:<n>\n")

        rec_num = 1
        try:
            while True:
                inp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: input(
                        f"{'─' * 52}\n"
                        f"[Ready] ENTER = start recording #{rec_num}  "
                        f"(q+ENTER = quit): "
                    )
                )
                if inp.strip().lower() == "q":
                    break

                state.reset()
                print(f"\nRecording #{rec_num} — sending REC …")
                await client.write_gatt_char(NUS_RX_CHAR_UUID,
                                             b"REC", response=False)
                print(f"  Sequence: START -> audio chunks -> finished -> HR -> RR")
                print(f"  Timeout: {REC_TIMEOUT:.0f} s")

                deadline = time.monotonic() + REC_TIMEOUT
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                    if _all_done():
                        print(f"\n  [DONE  ] All streams complete.")
                        break
                else:
                    print(f"\n  [WARN  ] Timeout — saving whatever was received.")

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")

                # ── Save audio WAV ────────────────────────────────
                if state.audio_samples:
                    wav_fn = OUTPUT_DIR / f"rec_{rec_num:03d}_{ts}.wav"
                    save_wav(bytearray(state.audio_samples), wav_fn)
                else:
                    print("  [WARN  ] No audio received — WAV not saved.")

                # ── Save results ──────────────────────────────────
                res_fn = OUTPUT_DIR / f"results_{rec_num:03d}_{ts}.txt"
                save_results(state.hr_value, state.rr_value, res_fn)

                # ── Summary ───────────────────────────────────────
                print(f"\n  ── Summary #{rec_num} ──────────────────────────────")
                print(f"     Audio:      {len(state.audio_samples)} / "
                      f"{state.expected_bytes} B  "
                      f"({state.audio_chunks} chunks, {state.audio_gaps} gaps)")
                hr_str = f"{state.hr_value:.0f} BPM" if state.hr_value is not None else "(not received)"
                rr_str = f"{state.rr_value:.0f} BrPM" if state.rr_value is not None else "(not received)"
                print(f"     Heart Rate: {hr_str}")
                print(f"     Resp Rate:  {rr_str}")
                if state.error:
                    print(f"     Firmware error: {state.error}")
                print()

                rec_num += 1

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            await client.stop_notify(NUS_TX_CHAR_UUID)
            print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(run())