#!/usr/bin/env python3
"""
AcoustEEEcare — BLE Receiver  (firmware v8.1 compatible)

Firmware stream order:
  1. START:<n>\n           — n = total raw audio bytes
  2. [binary audio chunks] — [seq:u16][len:u16][crc16:u16][payload]  (6-byte header)
  3. finished\n            — audio stream complete
  4. WARN:DROPS:<n>\n      — optional; n ring-buffer drops (audio degraded)
  5. HR:<value>\n          — heart rate result (or ERR:HEART_INF\n)
  6. RR:<value>\n          — respiration rate result (or ERR:LUNG_INF\n)

NACK protocol (v8.1):
  • Receiver validates CRC-16/CCITT of each chunk payload.
  • On mismatch, sends "NACK:<seq>\n" to firmware, which retransmits.
  • On sequence gap, zero-fills missing samples (best-effort recovery).

Files saved to OUTPUT_DIR:
  rec_NNN_TIMESTAMP[_DEGRADED].wav  — raw audio (8 kHz mono int16)
  results_NNN_TIMESTAMP.txt         — HR and RR values
"""

import asyncio
import struct
import wave
import time
import sys
from datetime import datetime
from pathlib import Path

# aioconsole keeps the event loop unblocked during input() waits.
# Install with: pip install aioconsole
try:
    import aioconsole
    _HAS_AIOCONSOLE = True
except ImportError:
    _HAS_AIOCONSOLE = False

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
REC_TIMEOUT   = 60.0    # 10 s capture + ~15 s inference + margin

# v8.1: header is [seq:u16][len:u16][crc16:u16] = 6 bytes
CHUNK_HEADER  = 6


# ── CRC-16/CCITT (matches Zephyr crc16_ccitt, init=0xFFFF) ──────────────────

def _crc16_firmware(data: bytes | bytearray) -> int:
    """Matches Zephyr crc16_ccitt(0xFFFF, data, len).
    Reflected polynomial 0x8408 (= 0x1021 bit-reversed), init=0xFFFF, no final XOR."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
    return crc


# ── State ─────────────────────────────────────────────────────────────────────

class StreamMode:
    IDLE  = "idle"
    AUDIO = "audio"


class RecState:
    """All mutable state for one recording session."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.mode            = StreamMode.IDLE
        self.expected_bytes  = 0
        self.audio_samples   = bytearray()
        self.audio_seq       = 0       # next expected sequence number
        self.audio_gaps      = 0       # gap events (missing pkts)
        self.audio_chunks    = 0       # successfully processed chunks
        self.audio_crc_errs  = 0       # CRC mismatches → NACKed
        self.audio_done      = False
        self.hr_value        = None    # float or None
        self.rr_value        = None
        self.hr_done         = False
        self.rr_done         = False
        self.audio_degraded  = False   # set when WARN:DROPS received
        self.error           = None    # last ERR: string
        self.last_progress   = 0.0

        # NACK: filled by _handle_text; read + cleared by send-loop
        self.nack_seq        = None    # int or None
        self.nack_event      = asyncio.Event()


# Module-level state object; reset() called before each recording.
rec = RecState()

# Reference to the BleakClient, set in run() so the notification handler
# can send NACKs without a callback parameter.
_ble_client: BleakClient | None = None


# ── Notification handler ──────────────────────────────────────────────────────

def _looks_like_audio_chunk(data: bytearray) -> bool:
    """
    Heuristic: binary audio chunk starts with a valid 6-byte header where
    the embedded payload length matches the actual packet length.
    """
    if rec.mode != StreamMode.AUDIO or len(data) < CHUNK_HEADER + 1:
        return False
    ln = struct.unpack_from("<H", data, 2)[0]
    return ln > 0 and ln == len(data) - CHUNK_HEADER


def handle_notification(_sender, data: bytearray):
    # Always dispatch to the event loop first — never read rec.mode here.
    # The Bleak notification thread may deliver audio chunks before the
    # call_soon_threadsafe for "START:..." has executed and set rec.mode=AUDIO.
    # Classifying here would therefore misroute early chunks as text.
    # _dispatch() runs on the event loop where ordering is guaranteed.
    asyncio.get_event_loop().call_soon_threadsafe(_dispatch, bytes(data))


def _dispatch(data: bytes):
    """Called on the asyncio event loop — rec.mode is safe to read here."""
    ba = bytearray(data)
    if _looks_like_audio_chunk(ba):
        _handle_audio_chunk(ba)
    else:
        _handle_text(ba)


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
            rec.reset()
            rec.expected_bytes = int(text.split(":")[1])
            rec.mode           = StreamMode.AUDIO
            dur = rec.expected_bytes / 2 / SAMPLE_RATE
            print(f"\n  [AUDIO ] START — expecting {rec.expected_bytes} B "
                  f"({dur:.1f} s @ {SAMPLE_RATE} Hz)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed START: {text!r}")

    # ── Audio stream end ──────────────────────────────────────────
    elif text.startswith("finished"):
        rec.audio_done = True
        rec.mode       = StreamMode.IDLE
        print(f"\n  [AUDIO ] finished — "
              f"{len(rec.audio_samples)}/{rec.expected_bytes} B  "
              f"chunks={rec.audio_chunks}  gaps={rec.audio_gaps}  "
              f"crc_errs={rec.audio_crc_errs}")

    # ── Drop warning (v8.1) ───────────────────────────────────────
    elif text.startswith("WARN:DROPS:"):
        try:
            n = int(text[11:])
            rec.audio_degraded = True
            print(f"\n  [WARN  ] Firmware: {n} audio ring-buffer drop(s) — "
                  f"WAV will be marked DEGRADED.")
        except ValueError:
            pass

    # ── Inference results ─────────────────────────────────────────
    elif text.startswith("HR:"):
        try:
            rec.hr_value = float(text.split(":")[1])
            rec.hr_done  = True
            print(f"\n  [HR    ] Heart Rate  = {rec.hr_value:.0f} BPM")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed HR: {text!r}")
            rec.hr_done = True

    elif text.startswith("RR:"):
        try:
            rec.rr_value = float(text.split(":")[1])
            rec.rr_done  = True
            print(f"\n  [RR    ] Resp Rate   = {rec.rr_value:.0f} BrPM")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed RR: {text!r}")
            rec.rr_done = True

    # ── Errors ────────────────────────────────────────────────────
    elif text.startswith("ERR:"):
        rec.error      = text
        rec.audio_done = True
        rec.hr_done    = True
        rec.rr_done    = True
        print(f"\n  [ERROR ] Firmware: {text}")

    else:
        print(f"\n  [MSG   ] {text!r}")


def _handle_audio_chunk(data: bytearray):
    """
    Parse a v8.1 audio chunk: [seq:u16][len:u16][crc16:u16][payload].
    Validates CRC; sends NACK on mismatch.  Zero-fills sequence gaps.
    """
    seq     = struct.unpack_from("<H", data, 0)[0]
    ln      = struct.unpack_from("<H", data, 2)[0]
    rx_crc  = struct.unpack_from("<H", data, 4)[0]
    payload = data[CHUNK_HEADER: CHUNK_HEADER + ln]

    # ── CRC validation ────────────────────────────────────────────
    calc_crc = _crc16_firmware(payload)
    if calc_crc != rx_crc:
        rec.audio_crc_errs += 1
        print(f"\n  [CRC   ] seq={seq} calc=0x{calc_crc:04X} rx=0x{rx_crc:04X} "
              f"— sending NACK")
        # Signal the send-loop to transmit NACK (avoids calling async from sync)
        rec.nack_seq = seq
        rec.nack_event.set()
        return   # do NOT add corrupt payload to audio buffer

    # ── Sequence-gap detection ────────────────────────────────────
    if seq != rec.audio_seq:
        gap_pkts  = (seq - rec.audio_seq) & 0xFFFF
        gap_bytes = gap_pkts * ln          # best-effort; ln is current chunk size
        print(f"\n  [GAP   ] audio seq {rec.audio_seq}→{seq} "
              f"({gap_pkts} pkt(s), ~{gap_bytes} B zeroed)")
        rec.audio_samples += b"\x00\x00" * (gap_bytes // 2)
        rec.audio_gaps    += gap_pkts


    
    print(f"  [DEBUG ] seq={seq} ln={ln} rx_crc=0x{rx_crc:04X} "
      f"calc_0000=0x{_crc16_firmware(payload):04X}")
    

    rec.audio_samples += payload
    rec.audio_seq      = (seq + 1) & 0xFFFF
    rec.audio_chunks  += 1
    _print_progress(len(rec.audio_samples), rec.expected_bytes)


def _print_progress(got: int, expected: int):
    now = time.monotonic()
    if now - rec.last_progress < 0.25:
        return
    rec.last_progress = now
    pct = min(got * 100 // max(expected, 1), 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [AUDIO ][{bar}] {pct:3d}%  {got:>8}/{expected} B",
          end="", flush=True)


# ── Completion check ──────────────────────────────────────────────────────────

def _all_done() -> bool:
    return rec.audio_done and rec.hr_done and rec.rr_done


# ── NACK sender ───────────────────────────────────────────────────────────────

async def _nack_sender_loop():
    """
    Background task: waits for rec.nack_event, then sends NACK:<seq>\n to
    the firmware.  Runs for the lifetime of one recording session.
    """
    while True:
        await rec.nack_event.wait()
        rec.nack_event.clear()
        seq = rec.nack_seq
        if seq is not None and _ble_client is not None:
            nack_msg = f"NACK:{seq}\n".encode()
            try:
                await _ble_client.write_gatt_char(NUS_RX_CHAR_UUID,
                                                  nack_msg,
                                                  response=False)
                print(f"\n  [NACK  ] Sent NACK for seq={seq}")
            except Exception as e:
                print(f"\n  [WARN  ] Failed to send NACK: {e}")


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_wav(samples: bytearray, filename: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = rec.expected_bytes
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
    print(f"Scanning for '{DEVICE_NAME}' (timeout={SCAN_TIMEOUT:.0f} s) …")
    results = await BleakScanner.discover(timeout=SCAN_TIMEOUT, return_adv=True)

    # Pass 1: exact name match
    for device, adv in results.values():
        if device.name == DEVICE_NAME:
            print(f"  Found: {device.name} [{device.address}]  RSSI={adv.rssi}")
            return device

    # Pass 2: NUS service UUID (Windows may not resolve name on first scan)
    for device, adv in results.values():
        service_uuids = [u.lower() for u in (adv.service_uuids or [])]
        if NUS_SERVICE_UUID in service_uuids:
            print(f"  Found via NUS UUID: [{device.address}]  "
                  f"name={device.name!r}  RSSI={adv.rssi}")
            print("  (name not resolved on this scan — connecting anyway)")
            return device

    print(f"\nERROR: '{DEVICE_NAME}' not found in scan.")
    print("Nearby devices:")
    for device, adv in sorted(results.values(),
                               key=lambda x: x[1].rssi or -999,
                               reverse=True):
        print(f"  RSSI={adv.rssi:4d}  {device.address}  name={device.name!r}")
    print("\nTips:")
    print("  • Power-cycle the MCU and try again")
    print("  • Toggle Bluetooth off/on in Windows Settings")
    return None


# ── Async input helper ────────────────────────────────────────────────────────

async def _async_input(prompt: str) -> str:
    """Non-blocking input: uses aioconsole if available, executor fallback."""
    if _HAS_AIOCONSOLE:
        return await aioconsole.ainput(prompt)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


# ── Main loop ─────────────────────────────────────────────────────────────────

async def run():
    global _ble_client

    device = await find_device()
    if device is None:
        sys.exit(1)

    disconnect_event = asyncio.Event()

    def on_disconnect(_client):
        print("\n  [WARN  ] Device disconnected unexpectedly.")
        # Mark everything done so the wait-loop exits cleanly
        rec.audio_done = True
        rec.hr_done    = True
        rec.rr_done    = True
        disconnect_event.set()

    print(f"Connecting to {device.address} …")
    async with BleakClient(device.address,
                           disconnected_callback=on_disconnect) as client:
        _ble_client = client
        mtu = getattr(client, "mtu_size", "?")
        print(f"Connected!  MTU = {mtu}")
        await client.start_notify(NUS_TX_CHAR_UUID, handle_notification)
        print("Subscribed to NUS notifications.")
        print(f"Chunk header: {CHUNK_HEADER} bytes  "
              f"[seq:u16][len:u16][crc16:u16]\n")

        rec_num = 1
        try:
            while True:
                prompt = (
                    f"{'─' * 52}\n"
                    f"[Ready] ENTER = start recording #{rec_num}  "
                    f"(q+ENTER = quit): "
                )

                # Race between user input and unexpected disconnect
                inp_task  = asyncio.create_task(_async_input(prompt))
                disc_task = asyncio.create_task(disconnect_event.wait())
                done, _   = await asyncio.wait(
                    [inp_task, disc_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disc_task in done:
                    inp_task.cancel()
                    print("  Connection lost — exiting.")
                    break
                disc_task.cancel()
                inp = inp_task.result()

                if inp.strip().lower() == "q":
                    break

                rec.reset()
                disconnect_event.clear()

                print(f"\nRecording #{rec_num} — sending REC …")
                await client.write_gatt_char(NUS_RX_CHAR_UUID,
                                             b"REC", response=False)
                print(f"  Sequence: START -> audio chunks -> finished "
                      f"-> [WARN:DROPS] -> HR -> RR")
                print(f"  Timeout: {REC_TIMEOUT:.0f} s")

                # Start background NACK sender for this session
                nack_task = asyncio.create_task(_nack_sender_loop())

                deadline = time.monotonic() + REC_TIMEOUT
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.05)
                    if _all_done() or disconnect_event.is_set():
                        if _all_done():
                            print("\n  [DONE  ] All streams complete.")
                        break
                else:
                    print("\n  [WARN  ] Timeout — saving whatever was received.")

                nack_task.cancel()

                ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
                suffix  = "_DEGRADED" if rec.audio_degraded else ""

                # ── Save audio WAV ────────────────────────────────
                if rec.audio_samples:
                    wav_fn = OUTPUT_DIR / f"rec_{rec_num:03d}_{ts}{suffix}.wav"
                    save_wav(bytearray(rec.audio_samples), wav_fn)
                else:
                    print("  [WARN  ] No audio received — WAV not saved.")

                # ── Save results ──────────────────────────────────
                res_fn = OUTPUT_DIR / f"results_{rec_num:03d}_{ts}.txt"
                save_results(rec.hr_value, rec.rr_value, res_fn)

                # ── Summary ───────────────────────────────────────
                print(f"\n  ── Summary #{rec_num} ──────────────────────────────")
                print(f"     Audio:      {len(rec.audio_samples)} / "
                      f"{rec.expected_bytes} B  "
                      f"({rec.audio_chunks} chunks, "
                      f"{rec.audio_gaps} gaps, "
                      f"{rec.audio_crc_errs} CRC errors)")
                hr_str = (f"{rec.hr_value:.0f} BPM"
                          if rec.hr_value is not None else "(not received)")
                rr_str = (f"{rec.rr_value:.0f} BrPM"
                          if rec.rr_value is not None else "(not received)")
                print(f"     Heart Rate: {hr_str}")
                print(f"     Resp Rate:  {rr_str}")
                if rec.audio_degraded:
                    print("     ⚠  Audio degraded (firmware ring-buffer drops)")
                if rec.error:
                    print(f"     Firmware error: {rec.error}")
                print()

                rec_num += 1

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            _ble_client = None
            try:
                await client.stop_notify(NUS_TX_CHAR_UUID)
            except Exception:
                pass
            print("Disconnected.")


if __name__ == "__main__":
    if not _HAS_AIOCONSOLE:
        print("NOTE: 'aioconsole' not found. Install with: pip install aioconsole")
        print("      Falling back to blocking input() — disconnect events may lag.\n")
    asyncio.run(run())