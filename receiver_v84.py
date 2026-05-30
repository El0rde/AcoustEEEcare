#!/usr/bin/env python3
"""
AcoustEEEcare v8.4 — Host-side BLE receiver
============================================================
Matches firmware v8.4 (BLE-only, ONE capture → BOTH HR and RR computed
on-device, dynamic MTU, no SD card, no MFCC stream).

Protocol (v8.4)
───────────────
    Device → host, once after MTU exchange:
        "READY\\n"                         (gate the Record action)

    Host → device:
        "REC"                              (no organ select — always dual)

    Device → host, per recording:
        "START:<audio_bytes>:DUAL\\n"
        [seq u16 LE][len u16 LE][int16 PCM]   — audio chunks (for the WAV)
        "finished\\n"
        "INF:START\\n"                       (on-device inference begins)
        "HR:<bpm>\\n"                        (heart result, or ERR:HEART_INF)
        "RR:<bpm>\\n"                        (lung result,  or ERR:LUNG_INF)
        "INF:DONE\\n"                        (session complete)

    Any time while connected:
        "PING\\n"                            (keepalive — ignored here)

    Fatal capture error:
        "ERR:SAADC\\n" / "ERR:DSP\\n" / etc.

Notes
─────
• The MFCC stream from older firmware is gone: inference now runs on the
  device, so only audio (for the WAV) plus the HR/RR text lines arrive.
• Inference of both models takes ~20–35 s, so REC_TIMEOUT is generous.
• Each recording saves:  rec_<n>_<ts>.wav  and  rec_<n>_<ts>.txt (HR/RR).

Usage
─────
    pip install bleak
    python receiver_v84.py
"""

import asyncio
import struct
import wave
import time
import sys
from datetime import datetime
from pathlib import Path

from bleak import BleakScanner, BleakClient

# ── BLE / device constants ────────────────────────────────────────────────────

NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"

DEVICE_NAME  = "AcoustEEEcare"
SAMPLE_RATE  = 8000
OUTPUT_DIR   = Path("recording_v84_oversampled_gain1_4_10us_battery")
SCAN_TIMEOUT = 20.0

# 10 s record + audio drain + ~20-35 s dual inference + margin.
REC_TIMEOUT  = 90.0

# How long to wait for the firmware's READY handshake after connecting.
READY_TIMEOUT = 5.0

idle_keepalive = True        # True when at the prompt (link idle)


# ── Audio packet / loss tracker ───────────────────────────────────────────────

class PacketStats:
    """Per-session audio packet and byte counts for a loss report."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.audio_pkts_rx    = 0
        self.audio_bytes_rx   = 0
        self.audio_bytes_exp  = 0
        self.audio_pkts_lost  = 0
        self.audio_bytes_lost = 0
        self.audio_overhead   = 0

        self.total_notifications = 0
        self.total_raw_bytes     = 0
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
    def audio_loss_pct(self):
        total = self.audio_pkts_rx + self.audio_pkts_lost
        return self.audio_pkts_lost * 100.0 / max(total, 1)

    @property
    def audio_throughput_kbps(self):
        return (self.audio_bytes_rx * 8) / max(self.session_duration, 0.001) / 1000.0

    def print_report(self):
        dur = self.session_duration
        sep = "━" * 58
        pct = self.audio_bytes_rx * 100 // max(self.audio_bytes_exp, 1)

        print(f"\n{sep}")
        print("  BLE SESSION STATISTICS")
        print(sep)
        print(f"  Duration              : {dur:.2f} s")
        print(f"  Total notifications   : {self.total_notifications} pkts")
        print(f"  Total raw BLE bytes   : {self.total_raw_bytes:,} B "
              f"({self.total_raw_bytes / 1024:.1f} kB)")
        print()
        print("  ── Audio stream ──────────────────────────────────")
        print(f"  Packets received      : {self.audio_pkts_rx}")
        print(f"  Packets lost (gaps)   : {self.audio_pkts_lost}"
              f"  ({self.audio_loss_pct:.2f}% loss)")
        print(f"  Payload bytes received: {self.audio_bytes_rx:,} B "
              f"({self.audio_bytes_rx / 1024:.1f} kB)")
        print(f"  Payload bytes expected: {self.audio_bytes_exp:,} B "
              f"({self.audio_bytes_exp / 1024:.1f} kB)  → {pct}% received")
        print(f"  Bytes lost / zero-pad : {self.audio_bytes_lost:,} B")
        print(f"  Header overhead       : {self.audio_overhead:,} B "
              f"(4 B × {self.audio_pkts_rx} pkts)")
        print(f"  Throughput (payload)  : {self.audio_throughput_kbps:.1f} kbps")
        print(sep)


pstats = PacketStats()


# ── Receiver state ────────────────────────────────────────────────────────────

class StreamMode:
    IDLE  = "idle"
    AUDIO = "audio"


class State:
    """Holds all receiver state for one recording session."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.error          = None        # fatal capture error (e.g. ERR:SAADC)
        self.mode           = StreamMode.IDLE

        # Audio
        self.expected_bytes = 0
        self.audio_done     = False
        self.audio_samples  = bytearray()
        self.audio_seq      = 0
        self.audio_gaps     = 0
        self.audio_chunks   = 0

        # Inference results
        self.inf_started    = False
        self.inf_done       = False
        self.hr             = None        # float BPM, or "ERR"
        self.rr             = None        # float BPM, or "ERR"

        self.last_progress  = 0.0

        pstats.reset()
        pstats.start_session()


state = State()
ready_event = asyncio.Event()   # set when firmware sends READY


# ── Packet classifier ─────────────────────────────────────────────────────────

def _is_data_chunk(data: bytearray) -> bool:
    """
    True when the packet is a sequenced audio chunk:
      bytes[0:2] = seq (u16 LE), bytes[2:4] = payload_len (u16 LE),
      len(data)  = 4 + payload_len.
    Only meaningful while in AUDIO mode.
    """
    if state.mode != StreamMode.AUDIO or len(data) < 5:
        return False
    payload_len = struct.unpack_from("<H", data, 2)[0]
    return payload_len > 0 and payload_len == len(data) - 4


def handle_notification(_sender, data: bytearray):
    pstats.total_notifications += 1
    pstats.total_raw_bytes     += len(data)

    if _is_data_chunk(data):
        _handle_audio_chunk(data)
    else:
        _handle_text(data)


# ── Control / text message handler ────────────────────────────────────────────

def _handle_text(data: bytearray):
    try:
        text = data.decode("ascii", errors="replace").strip()
    except Exception:
        return

    # Keepalive — silently ignore (uncomment to debug link health).
    if text == "PING":
        # print("  [PING ]")
        return

    # Firmware ready handshake (sent once after MTU exchange).
    if text.startswith("READY"):
        ready_event.set()
        print("  [READY ] Firmware ready.")
        return

    # Audio stream header: "START:<bytes>:DUAL"
    if text.startswith("START:"):
        try:
            parts = text.split(":")
            n     = int(parts[1])
            state.expected_bytes   = n
            state.audio_samples    = bytearray()
            state.audio_seq        = 0
            state.audio_gaps       = 0
            state.audio_chunks     = 0
            state.audio_done       = False
            state.mode             = StreamMode.AUDIO
            pstats.audio_bytes_exp = n
            dur = n / 2 / SAMPLE_RATE
            print(f"\n  [AUDIO ] START — expecting {n} B "
                  f"({n // 2} samples, {dur:.1f} s)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed START: {text!r}")
        return

    # Audio stream footer.
    if text.startswith("finished"):
        state.audio_done = True
        got = len(state.audio_samples)
        exp = state.expected_bytes
        pct = min(got * 100 // max(exp, 1), 100)
        print(f"\n  [AUDIO ] finished — {got}/{exp} B ({pct}%)"
              f"  pkts_rx={pstats.audio_pkts_rx}"
              f"  lost={pstats.audio_pkts_lost}"
              f"  ({pstats.audio_loss_pct:.2f}% loss)")
        state.mode = StreamMode.IDLE
        return

    # Inference lifecycle.
    if text.startswith("INF:START"):
        state.inf_started = True
        print("\n  [INFER ] Running heart + lung on-device "
              "(this can take ~20-35 s)…")
        return

    if text.startswith("INF:DONE"):
        state.inf_done = True
        print("  [INFER ] Inference complete.")
        return

    # Results.
    if text.startswith("HR:"):
        try:
            state.hr = float(text.split(":")[1])
            print(f"  [RESULT] HR = {state.hr:.1f} BPM")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed HR: {text!r}")
        return

    if text.startswith("RR:"):
        try:
            state.rr = float(text.split(":")[1])
            print(f"  [RESULT] RR = {state.rr:.1f} breaths/min")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed RR: {text!r}")
        return

    # Per-organ inference errors are NOT fatal — the session still ends
    # on INF:DONE; the other organ's result may still be valid.
    if text.startswith("ERR:HEART_INF"):
        state.hr = "ERR"
        print("  [ERROR ] Heart inference failed on device.")
        return
    if text.startswith("ERR:LUNG_INF"):
        state.rr = "ERR"
        print("  [ERROR ] Lung inference failed on device.")
        return

    # Any other ERR: is a fatal capture error.
    if text.startswith("ERR:"):
        state.error     = text
        state.audio_done = True
        state.inf_done   = True
        state.mode       = StreamMode.IDLE
        print(f"\n  [ERROR ] Firmware reported: {text}")
        return

    print(f"\n  [FW    ] {text!r}")


# ── Audio chunk handler ───────────────────────────────────────────────────────

def _handle_audio_chunk(data: bytearray):
    seq     = struct.unpack_from("<H", data, 0)[0]
    ln      = struct.unpack_from("<H", data, 2)[0]
    payload = data[4: 4 + ln]

    pstats.audio_pkts_rx  += 1
    pstats.audio_bytes_rx += ln
    pstats.audio_overhead += 4

    if seq != state.audio_seq:
        gap_pkts  = (seq - state.audio_seq) & 0xFFFF
        gap_bytes = gap_pkts * ln   # dynamic chunk size → estimate from this one
        print(f"\n  [GAP   ] audio seq {state.audio_seq}→{seq} "
              f"({gap_pkts} pkt(s) missing, ~{gap_bytes} B zero-filled)")
        state.audio_samples     += b"\x00\x00" * (gap_bytes // 2)
        state.audio_gaps        += gap_pkts
        pstats.audio_pkts_lost  += gap_pkts
        pstats.audio_bytes_lost += gap_bytes

    state.audio_samples += payload
    state.audio_seq      = (seq + 1) & 0xFFFF
    state.audio_chunks  += 1
    _print_audio_progress()


def _print_audio_progress():
    now = time.monotonic()
    if now - state.last_progress < 0.25:
        return
    state.last_progress = now
    got  = len(state.audio_samples)
    exp  = state.expected_bytes
    pct  = min(got * 100 // max(exp, 1), 100)
    bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%  {got:>7}/{exp} B  "
          f"pkts={pstats.audio_pkts_rx}  lost={pstats.audio_pkts_lost}"
          f"  ({pstats.audio_loss_pct:.1f}% loss)",
          end="", flush=True)


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_wav(samples: bytearray, filename: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    target = state.expected_bytes

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


def _fmt_result(v):
    if v is None:
        return "n/a"
    if v == "ERR":
        return "ERROR"
    return f"{v:.1f}"


def save_results(filename: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(filename, "w") as f:
        f.write(f"timestamp : {datetime.now().isoformat()}\n")
        f.write(f"HR_bpm    : {_fmt_result(state.hr)}\n")
        f.write(f"RR_bpm    : {_fmt_result(state.rr)}\n")
    print(f"  [SAVE  ] {filename}")


# ── Completion + wait loop ────────────────────────────────────────────────────

def _session_complete() -> bool:
    if state.error:
        return True
    return state.inf_done


async def _wait_for_completion(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        if _session_complete():
            return True
    return False


def _save_session(rec_num: int) -> None:
    pstats.end_session()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print()
    print("  ── Results ───────────────────────────────────────")
    print(f"  HR : {_fmt_result(state.hr)} BPM")
    print(f"  RR : {_fmt_result(state.rr)} breaths/min")

    if state.audio_samples and state.expected_bytes > 0:
        wav_fn = OUTPUT_DIR / f"rec_{rec_num:03d}_{ts}.wav"
        save_wav(bytearray(state.audio_samples), wav_fn)
    else:
        print("  [WARN  ] No audio received — WAV not saved.")

    txt_fn = OUTPUT_DIR / f"rec_{rec_num:03d}_{ts}.txt"
    save_results(txt_fn)

    pstats.print_report()


# ── Main ──────────────────────────────────────────────────────────────────────
async def keepalive_loop(client):
       while True:
           try:
               if idle_keepalive:
                   await client.write_gatt_char(NUS_RX_CHAR_UUID, b"KA",
                                                response=False)
           except Exception:
               pass
           await asyncio.sleep(0.25)     # 250 ms keeps Windows from relaxing

async def run():
    print(f"Scanning for '{DEVICE_NAME}' (timeout={SCAN_TIMEOUT:.0f} s)…")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=SCAN_TIMEOUT)
    if device is None:
        print(f"ERROR: '{DEVICE_NAME}' not found within {SCAN_TIMEOUT:.0f} s.")
        sys.exit(1)

    print(f"  Found: {DEVICE_NAME}  [{device.address}]")

    async with BleakClient(device) as client:
        mtu = getattr(client, "mtu_size", "?")
        print(f"Connected!  MTU = {mtu}")
        await client.start_notify(NUS_TX_CHAR_UUID, handle_notification)
        ka_task = asyncio.create_task(keepalive_loop(client))
        print("Subscribed to NUS notifications.")

        # Wait for the firmware's READY handshake (sent after MTU exchange).
        try:
            await asyncio.wait_for(ready_event.wait(), timeout=READY_TIMEOUT)
        except asyncio.TimeoutError:
            print("  [WARN  ] No READY within "
                  f"{READY_TIMEOUT:.0f} s — proceeding anyway.")

        print()
        print("Commands:")
        print("  ENTER  — send REC (captures once, returns HR + RR)")
        print("  q      — quit")
        print()

        rec_num = 1
        try:
            while True:
                sep = "─" * 55
                inp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: input(f"{sep}\n[Ready #{rec_num}]  ENTER=REC  q=quit: ")
                )
                cmd = inp.strip().lower()

                if cmd == "q":
                    print("Quitting.")
                    break

                elif cmd == "t":
                    await client.write_gatt_char(NUS_RX_CHAR_UUID, b"TEST", response=False)

                # ── REC ──
                globals()['idle_keepalive'] = False     # before write REC
                state.reset()
                print("\nSending REC…")
                await client.write_gatt_char(NUS_RX_CHAR_UUID, b"REC", response=False)
                print(f"  Waiting up to {REC_TIMEOUT:.0f} s "
                      f"(10 s record + audio + dual inference)…")

                completed = await _wait_for_completion(REC_TIMEOUT)
                
                if not completed:
                    print("\n  [WARN  ] Timeout — saving whatever arrived.")
                elif state.error:
                    print(f"\n  [ERROR ] Firmware error: {state.error}")
                else:
                    print("\n  [DONE  ] REC complete.")

                _save_session(rec_num)
                rec_num += 1

                globals()['idle_keepalive'] = True       # after _save_session

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            await client.stop_notify(NUS_TX_CHAR_UUID)
            print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(run())