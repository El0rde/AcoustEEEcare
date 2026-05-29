#!/usr/bin/env python3
"""
AcoustEEEcare v6.7 — Host-side BLE receiver
============================================================
Matches firmware v6.7 (BLE-only, no SD card, no SEND command).

Pipeline (single mode):
    Host sends: "REC"
    Device sends:
        "START:<audio_bytes>\\n"  — audio stream header
        [seq u16 LE][len u16 LE][int16 payload]  — audio chunks
        "finished\\n"             — audio stream footer
        "MFCC_START:<n_frames>:<n_coeffs>\\n"  — MFCC header
        [seq u16 LE][len u16 LE][float32 bytes] — one packet per frame
        "MFCC_END\\n"             — MFCC footer

Error messages from firmware:
    "ERR:<CODE>\\n"  — e.g. ERR:SAADC, ERR:TIMEOUT, ERR:DSP

Usage:
    pip install bleak numpy
    python receiver_v67.py
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
OUTPUT_DIR   = Path("recording_29_05_2026_flex_updated_main")
SCAN_TIMEOUT = 20.0

# 10 s record + audio stream + MFCC stream + margin
REC_TIMEOUT  = 120.0


# ── Packet / data-loss tracker ────────────────────────────────────────────────

class PacketStats:
    """
    Accumulates per-stream packet and byte counts for a loss report
    at the end of each session.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # ── Audio ──────────────────────────────────────────────────
        self.audio_pkts_rx    = 0
        self.audio_bytes_rx   = 0
        self.audio_bytes_exp  = 0   # from START:<n>
        self.audio_pkts_lost  = 0
        self.audio_bytes_lost = 0
        self.audio_overhead   = 0   # 4 B per packet (seq u16 + len u16)

        # ── MFCC ───────────────────────────────────────────────────
        self.mfcc_pkts_rx     = 0
        self.mfcc_bytes_rx    = 0
        self.mfcc_pkts_exp    = 0   # = n_frames (one packet per frame)
        self.mfcc_bytes_exp   = 0   # n_frames * n_coeffs * 4
        self.mfcc_pkts_lost   = 0
        self.mfcc_bytes_lost  = 0
        self.mfcc_overhead    = 0

        # ── Session-level BLE totals ───────────────────────────────
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
    def mfcc_loss_pct(self):
        total = self.mfcc_pkts_rx + self.mfcc_pkts_lost
        return self.mfcc_pkts_lost * 100.0 / max(total, 1)

    @property
    def audio_throughput_kbps(self):
        dur = self.session_duration
        return (self.audio_bytes_rx * 8) / max(dur, 0.001) / 1000.0

    @property
    def mfcc_throughput_kbps(self):
        dur = self.session_duration
        return (self.mfcc_bytes_rx * 8) / max(dur, 0.001) / 1000.0

    def print_report(self):
        dur = self.session_duration
        sep = "━" * 58

        print(f"\n{sep}")
        print("  BLE SESSION STATISTICS")
        print(sep)
        print(f"  Duration              : {dur:.2f} s")
        print(f"  Total notifications   : {self.total_notifications} pkts")
        print(f"  Total raw BLE bytes   : {self.total_raw_bytes:,} B "
              f"({self.total_raw_bytes / 1024:.1f} kB)")
        print()

        # ── Audio stream ──────────────────────────────────────────
        print("  ── Audio stream ──────────────────────────────────")
        _pct_rx = self.audio_bytes_rx * 100 // max(self.audio_bytes_exp, 1)
        print(f"  Packets received      : {self.audio_pkts_rx}")
        print(f"  Packets lost (gaps)   : {self.audio_pkts_lost}"
              f"  ({self.audio_loss_pct:.2f}% loss)")
        print(f"  Payload bytes received: {self.audio_bytes_rx:,} B "
              f"({self.audio_bytes_rx / 1024:.1f} kB)")
        print(f"  Payload bytes expected: {self.audio_bytes_exp:,} B "
              f"({self.audio_bytes_exp / 1024:.1f} kB)  → {_pct_rx}% received")
        print(f"  Bytes lost / zero-pad : {self.audio_bytes_lost:,} B")
        print(f"  Header overhead       : {self.audio_overhead:,} B "
              f"(4 B × {self.audio_pkts_rx} pkts)")
        print(f"  Throughput (payload)  : {self.audio_throughput_kbps:.1f} kbps")
        print()

        # ── MFCC stream ───────────────────────────────────────────
        print("  ── MFCC stream ───────────────────────────────────")
        _mpct_rx = self.mfcc_bytes_rx * 100 // max(self.mfcc_bytes_exp, 1)
        print(f"  Packets received      : {self.mfcc_pkts_rx}")
        print(f"  Packets lost (gaps)   : {self.mfcc_pkts_lost}"
              f"  ({self.mfcc_loss_pct:.2f}% loss)")
        print(f"  Payload bytes received: {self.mfcc_bytes_rx:,} B "
              f"({self.mfcc_bytes_rx / 1024:.1f} kB)")
        print(f"  Payload bytes expected: {self.mfcc_bytes_exp:,} B "
              f"({self.mfcc_bytes_exp / 1024:.1f} kB)  → {_mpct_rx}% received")
        print(f"  Bytes lost / zero-pad : {self.mfcc_bytes_lost:,} B")
        print(f"  Header overhead       : {self.mfcc_overhead:,} B "
              f"(4 B × {self.mfcc_pkts_rx} pkts)")
        print(f"  Throughput (payload)  : {self.mfcc_throughput_kbps:.1f} kbps")
        print()

        # ── Combined ──────────────────────────────────────────────
        total_payload_rx  = self.audio_bytes_rx  + self.mfcc_bytes_rx
        total_payload_exp = self.audio_bytes_exp + self.mfcc_bytes_exp
        total_pkts_rx     = self.audio_pkts_rx   + self.mfcc_pkts_rx
        total_pkts_lost   = self.audio_pkts_lost  + self.mfcc_pkts_lost
        total_pkts_all    = total_pkts_rx + total_pkts_lost
        total_overhead    = self.audio_overhead   + self.mfcc_overhead
        combined_loss_pct = total_pkts_lost * 100.0 / max(total_pkts_all, 1)
        combined_pct_rx   = total_payload_rx * 100 // max(total_payload_exp, 1)
        combined_kbps     = (total_payload_rx * 8) / max(dur, 0.001) / 1000.0

        print("  ── Combined ──────────────────────────────────────")
        print(f"  Total packets rx      : {total_pkts_rx}")
        print(f"  Total packets lost    : {total_pkts_lost}"
              f"  ({combined_loss_pct:.2f}% loss)")
        print(f"  Total payload rx      : {total_payload_rx:,} B "
              f"({total_payload_rx / 1024:.1f} kB)")
        print(f"  Total payload expected: {total_payload_exp:,} B "
              f"({total_payload_exp / 1024:.1f} kB)  → {combined_pct_rx}% received")
        print(f"  Total overhead        : {total_overhead:,} B")
        print(f"  Effective throughput  : {combined_kbps:.1f} kbps")
        print(sep)


pstats = PacketStats()


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

        pstats.reset()
        pstats.start_session()


state = State()


# ── Packet classifier ─────────────────────────────────────────────────────────

def _is_data_chunk(data: bytearray) -> bool:
    """
    True when the packet looks like a sequenced data chunk:
      bytes[0:2] = seq (u16 LE)
      bytes[2:4] = payload_len (u16 LE)
      len(data)  = 4 + payload_len
    Only checked while in AUDIO or MFCC mode.
    """
    if state.mode == StreamMode.IDLE or len(data) < 5:
        return False
    payload_len = struct.unpack_from("<H", data, 2)[0]
    return payload_len > 0 and payload_len == len(data) - 4


def handle_notification(_sender, data: bytearray):
    pstats.total_notifications += 1
    pstats.total_raw_bytes     += len(data)

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

    # ── Audio stream header ──
    if text.startswith("START:"):
        try:
            n = int(text.split(":")[1])
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

    # ── Audio stream footer ──
    elif text.startswith("finished"):
        state.audio_done = True
        got = len(state.audio_samples)
        exp = state.expected_bytes
        pct = min(got * 100 // max(exp, 1), 100)
        print(f"\n  [AUDIO ] finished — {got}/{exp} B ({pct}%)"
              f"  pkts_rx={pstats.audio_pkts_rx}"
              f"  lost={pstats.audio_pkts_lost}"
              f"  ({pstats.audio_loss_pct:.2f}% loss)")
        state.mode = StreamMode.IDLE

    # ── MFCC stream header ──
    elif text.startswith("MFCC_START:"):
        try:
            parts = text.split(":")
            nf    = int(parts[1])
            nc    = int(parts[2])
            state.mfcc_n_frames    = nf
            state.mfcc_n_coeffs    = nc
            state.mfcc_expected_b  = nf * nc * 4   # float32
            state.mfcc_bytes       = bytearray()
            state.mfcc_seq         = 0
            state.mfcc_gaps        = 0
            state.mfcc_chunks      = 0
            state.mfcc_done        = False
            state.mode             = StreamMode.MFCC
            pstats.mfcc_pkts_exp   = nf
            pstats.mfcc_bytes_exp  = nf * nc * 4

            print(f"\n  [MFCC  ] START — {nf} frames × {nc} coeffs "
                  f"= {state.mfcc_expected_b} B  "
                  f"(expecting {nf} pkts)")
        except (ValueError, IndexError):
            print(f"  [WARN  ] Malformed MFCC_START: {text!r}")

    # ── MFCC stream footer ──
    elif text.startswith("MFCC_END"):
        state.mfcc_done = True
        got = len(state.mfcc_bytes)
        exp = state.mfcc_expected_b
        pct = min(got * 100 // max(exp, 1), 100)
        print(f"\n  [MFCC  ] MFCC_END — {got}/{exp} B ({pct}%)"
              f"  pkts_rx={pstats.mfcc_pkts_rx}"
              f"  lost={pstats.mfcc_pkts_lost}"
              f"  ({pstats.mfcc_loss_pct:.2f}% loss)")
        state.mode = StreamMode.IDLE

    # ── Firmware error ──
    elif text.startswith("ERR:"):
        state.error      = text
        state.audio_done = True   # unblock wait loop
        state.mfcc_done  = True
        state.mode       = StreamMode.IDLE
        print(f"\n  [ERROR ] Firmware reported: {text}")

    else:
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
        gap_bytes = gap_pkts * ln
        print(f"\n  [GAP   ] audio seq {state.audio_seq}→{seq} "
              f"({gap_pkts} pkt(s) missing, ~{gap_bytes} B zero-filled)")
        state.audio_samples    += b"\x00\x00" * (gap_bytes // 2)
        state.audio_gaps       += gap_pkts
        pstats.audio_pkts_lost += gap_pkts
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
    loss = pstats.audio_loss_pct
    print(f"\r  [{bar}] {pct:3d}%  {got:>7}/{exp} B  "
          f"pkts={pstats.audio_pkts_rx}  lost={pstats.audio_pkts_lost}"
          f"  ({loss:.1f}% loss)",
          end="", flush=True)


# ── MFCC chunk handler ────────────────────────────────────────────────────────

def _handle_mfcc_chunk(data: bytearray):
    seq     = struct.unpack_from("<H", data, 0)[0]
    ln      = struct.unpack_from("<H", data, 2)[0]
    payload = data[4: 4 + ln]

    pstats.mfcc_pkts_rx  += 1
    pstats.mfcc_bytes_rx += ln
    pstats.mfcc_overhead += 4

    if seq != state.mfcc_seq:
        gap_pkts  = (seq - state.mfcc_seq) & 0xFFFF
        gap_bytes = gap_pkts * ln
        print(f"\n  [GAP   ] MFCC seq {state.mfcc_seq}→{seq} "
              f"({gap_pkts} pkt(s) missing, ~{gap_bytes} B zero-filled)")
        state.mfcc_bytes        += b"\x00" * gap_bytes
        state.mfcc_gaps         += gap_pkts
        pstats.mfcc_pkts_lost   += gap_pkts
        pstats.mfcc_bytes_lost  += gap_bytes

    state.mfcc_bytes += payload
    state.mfcc_seq    = (seq + 1) & 0xFFFF
    state.mfcc_chunks += 1

    now = time.monotonic()
    if now - state.last_progress >= 0.25:
        state.last_progress = now
        got  = len(state.mfcc_bytes)
        exp  = state.mfcc_expected_b
        pct  = min(got * 100 // max(exp, 1), 100)
        bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
        loss = pstats.mfcc_loss_pct
        print(f"\r  [MFCC  ][{bar}] {pct:3d}%  {got:>7}/{exp} B  "
              f"pkts={pstats.mfcc_pkts_rx}  lost={pstats.mfcc_pkts_lost}"
              f"  ({loss:.1f}% loss)",
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

def _session_complete() -> bool:
    """True when both audio and MFCC streams have finished, or on error."""
    if state.error:
        return True
    return state.audio_done and state.mfcc_done


# ── Wait loop ─────────────────────────────────────────────────────────────────

async def _wait_for_completion(timeout: float) -> bool:
    """Poll until _session_complete() or timeout. Returns True if clean."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(0.05)
        if _session_complete():
            return True
    return False


# ── Save session outputs ──────────────────────────────────────────────────────

def _save_session(rec_num: int) -> None:
    pstats.end_session()

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

    pstats.print_report()


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
                        f"[Ready #{rec_num}]  ENTER=REC  q=quit: "
                    )
                )

                cmd = inp.strip().lower()

                if cmd == "q":
                    print("Quitting.")
                    break

                else:
                    # ── REC command ──
                    state.reset()
                    print(f"\nSending REC command…")
                    await client.write_gatt_char(
                        NUS_RX_CHAR_UUID, b"REC", response=False
                    )
                    print(
                        f"  Waiting up to {REC_TIMEOUT:.0f} s "
                        f"(10 s record + audio stream + MFCC stream)…"
                    )

                    completed = await _wait_for_completion(REC_TIMEOUT)
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