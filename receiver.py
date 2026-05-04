#!/usr/bin/env python3
"""
AcoustEEEcare v6.1 — Host-side BLE receiver

Fixes vs v6.0:
  - Recording completion is now driven by TWO independent signals:
      1. "finished\n" from firmware (primary — firmware sends this only after
         all audio chunks have been bt_nus_send()'d by the BLE TX thread).
      2. All expected bytes received (safety cross-check).
    The 0.3 s grace period is removed — it was masking the real bug and
    was still too short to catch in-flight BLE retransmits. With v6.1
    firmware the "finished" signal is already serialised after all audio.
  - Added configurable SCAN_TIMEOUT (15 → 20 s) for slower hosts.

Wire format per BLE notification (firmware v6.1):
    [0..1]  seq  (uint16 LE)
    [2..3]  len  (uint16 LE) — payload byte count
    [4..]   PCM  (int16 LE, 8 kHz mono)

Control messages are plain ASCII with no 4-byte header:
    START:<total_bytes>\n
    finished\n
    ERR:<reason>
"""

import asyncio
import struct
import wave
import time
import sys
from datetime import datetime
from pathlib import Path
from bleak import BleakScanner, BleakClient

NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

DEVICE_NAME  = "AcoustEEEcare"
SAMPLE_RATE  = 8000
OUTPUT_DIR   = Path("recordings")
SCAN_TIMEOUT = 20.0   # seconds
REC_TIMEOUT  = 40.0   # seconds — firmware takes ~10 s + BLE drain


class State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.expected_bytes  = 0
        self.recording       = False
        self.fw_finished     = False
        self.samples         = bytearray()
        self.expected_seq    = 0
        self.total_gaps      = 0
        self.total_gap_bytes = 0
        self.chunks          = 0
        self.last_progress   = 0.0


state = State()


def _is_audio_chunk(data: bytearray) -> bool:
    """
    Audio chunks: 4-byte header where bytes[2:4] == len(data) - 4 and > 0.
    Text control messages never satisfy this because their byte[2:4]
    values don't accidentally equal (total_len - 4) in practice.
    """
    if not state.recording or len(data) < 5:   # need header + ≥1 payload byte
        return False
    ln = struct.unpack_from("<H", data, 2)[0]
    return ln > 0 and ln == len(data) - 4


def handle_notification(sender, data: bytearray):
    if _is_audio_chunk(data):
        _handle_audio(data)
    else:
        _handle_text(data)


def _handle_text(data: bytearray):
    try:
        text = data.decode("ascii", errors="replace").strip()
    except Exception:
        return

    if text.startswith("START:"):
        try:
            state.reset()
            state.expected_bytes = int(text.split(":")[1])
            state.recording      = True
            print(f"\n  [START] expecting {state.expected_bytes} bytes "
                  f"({state.expected_bytes // 2} samples, "
                  f"{state.expected_bytes / 2 / SAMPLE_RATE:.1f} s)")
        except (ValueError, IndexError):
            print(f"  [WARN ] Malformed START: {text!r}")

    elif text.startswith("finished"):
        state.fw_finished = True
        got = len(state.samples)
        exp = state.expected_bytes
        pct = min(got * 100 // max(exp, 1), 100)
        print(f"\n  [FW   ] 'finished' received — {got}/{exp} B ({pct}%)")

    elif text.startswith("ERR:"):
        print(f"\n  [ERROR] Firmware: {text}")
        state.fw_finished = True


def _handle_audio(data: bytearray):
    seq = struct.unpack_from("<H", data, 0)[0]
    ln  = struct.unpack_from("<H", data, 2)[0]
    payload = data[4: 4 + ln]

    # Gap detection + zero-fill
    if seq != state.expected_seq:
        gap_pkts   = (seq - state.expected_seq) & 0xFFFF
        # Use the current packet's length as the estimate for missing ones
        gap_bytes  = gap_pkts * ln
        gap_ms     = gap_bytes // 2 * 1000 // SAMPLE_RATE
        print(f"\n  [GAP  ] seq {state.expected_seq}→{seq}: "
              f"{gap_pkts} pkt(s), ~{gap_ms} ms at sample {len(state.samples) // 2}")
        state.samples         += b"\x00\x00" * (gap_bytes // 2)
        state.total_gaps      += gap_pkts
        state.total_gap_bytes += gap_bytes

    state.samples      += payload
    state.expected_seq  = (seq + 1) & 0xFFFF
    state.chunks       += 1

    # Progress bar throttled to ~4 Hz
    now = time.monotonic()
    if now - state.last_progress >= 0.25:
        state.last_progress = now
        got = len(state.samples)
        exp = state.expected_bytes
        pct = min(got * 100 // max(exp, 1), 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {got:>7}/{exp} B  gaps={state.total_gaps}",
              end="", flush=True)


def save_wav(samples: bytearray, filename: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    target = state.expected_bytes

    if len(samples) > target:
        print(f"\n  [TRIM ] {len(samples) - target} extra bytes discarded")
        samples = samples[:target]
    elif len(samples) < target:
        pad = target - len(samples)
        print(f"\n  [PAD  ] {pad} bytes zero-padded to reach {target}")
        samples += b"\x00" * pad

    with wave.open(str(filename), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples)

    kb  = filename.stat().st_size / 1024
    dur = len(samples) / 2 / SAMPLE_RATE
    print(f"  [SAVE ] {filename}  ({kb:.1f} kB, {dur:.2f} s)")


async def run():
    print(f"Scanning for '{DEVICE_NAME}' …")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=SCAN_TIMEOUT)
    if device is None:
        print(f"ERROR: '{DEVICE_NAME}' not found. Is it advertising?")
        sys.exit(1)

    print(f"  Found: {DEVICE_NAME} [{device.address}]")
    print(f"Connecting to {device.address} …")

    async with BleakClient(device) as client:
        mtu = getattr(client, "mtu_size", "?")
        print(f"Connected!  MTU = {mtu}")
        await client.start_notify(NUS_TX_CHAR_UUID, handle_notification)
        print("Subscribed to NUS notifications.")
        print("Press Ctrl+C at any time to stop.\n")

        rec_num = 1
        try:
            while True:
                inp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: input(
                        f"{'─' * 41}\n"
                        f"[Ready] Press ENTER to start recording #{rec_num} "
                        f"(or 'q' + ENTER to quit): "
                    )
                )
                if inp.strip().lower() == "q":
                    break

                state.reset()
                print(f"\nRecording #{rec_num}  —  sending REC …")
                await client.write_gatt_char(NUS_RX_CHAR_UUID, b"REC", response=False)
                print(f"  Streaming … (timeout {REC_TIMEOUT:.0f} s)")

                deadline = time.monotonic() + REC_TIMEOUT
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.05)

                    # Primary: firmware signalled done (v6.1 firmware sends this
                    # only after all audio chunks have been bt_nus_send()'d).
                    # No grace period needed — the signal is already serialised.
                    if state.fw_finished:
                        print(f"\n  [DONE ] Firmware 'finished' signal received.")
                        break

                    # Safety: if somehow we got all bytes but no "finished"
                    if (state.expected_bytes > 0
                            and len(state.samples) >= state.expected_bytes):
                        print(f"\n  [DONE ] All {state.expected_bytes} bytes received.")
                        break
                else:
                    print(f"\n  [WARN ] {REC_TIMEOUT:.0f} s timeout — saving what we have.")

                got = len(state.samples)
                exp = state.expected_bytes
                print(f"  Chunks received : {state.chunks}")
                print(f"  Bytes collected : {got} / {exp}  "
                      f"({'OK' if got == exp else f'MISMATCH delta={got - exp:+d}'})")
                if state.total_gaps:
                    print(f"  Gaps detected   : {state.total_gaps} pkt(s), "
                          f"{state.total_gap_bytes} B "
                          f"({state.total_gap_bytes // 2 * 1000 // SAMPLE_RATE} ms)")

                if state.samples:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn = OUTPUT_DIR / f"recording_{rec_num:03d}_{ts}.wav"
                    save_wav(bytearray(state.samples), fn)
                    rec_num += 1
                else:
                    print("  [WARN ] No audio data — WAV not saved.")

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            await client.stop_notify(NUS_TX_CHAR_UUID)
            print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(run())