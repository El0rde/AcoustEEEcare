#!/usr/bin/env python3
"""
AcoustEEEcare v8.1 — Host-side BLE receiver
=============================================
Protocol: BLE-streaming, no SD card.

Adopted from v7.5:
  - Stays connected between recordings (interactive loop).
  - Numbered sessions with timestamps.
  - Progress bar per stream.
  - Gap detection and zero-fill for audio chunks.
  - Saves: rec_NNN_TS.wav, hr_NNN_TS.txt, rr_NNN_TS.txt

v8.1 differences from v7.5:
  - No SD card. Host computes MFCC from PCM, quantizes to int8,
    sends raw int8 chunks back to nRF (no .npy header).
  - nRF sends HR/RR as plain BLE strings ("HR:<val>\\n", "RR:<val>\\n")
    instead of chunked binary result streams.
  - Receives QUANT_HEART / QUANT_LUNG (or LUNG_DISABLED) at startup.
  - Sends MFCC_HEART_START / chunks / MFCC_HEART_END, then waits for
    HEART_DONE before sending lung MFCC.

Stream order (firmware v8.1):
  nRF → host:  QUANT_HEART:<scale_hex>:<zp>\\n
               QUANT_LUNG:<scale_hex>:<zp>\\n   (or LUNG_DISABLED\\n)
  host → nRF:  REC
  nRF → host:  START:<n>\\n
               [seq16][len16][pcm_bytes] ...
               finished\\n
  host → nRF:  MFCC_HEART_START:<n>\\n
               [seq16][len16][int8_bytes] ...
               MFCC_HEART_END\\n
  nRF → host:  HR:<value>\\n
               HEART_DONE\\n
  host → nRF:  MFCC_LUNG_START:<n>\\n           (skipped if LUNG_DISABLED)
               [seq16][len16][int8_bytes] ...
               MFCC_LUNG_END\\n
  nRF → host:  RR:<value>\\n

Files saved per recording:
  recordings_no_sd/rec_NNN_TIMESTAMP.wav
  recordings/hr_NNN_TIMESTAMP.txt
  recordings/rr_NNN_TIMESTAMP.txt   (omitted if lung disabled)

Dependencies:
  pip install bleak numpy librosa
"""

import asyncio
import datetime
import logging
import struct
import time
import sys
import wave
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE_NAME    = "AcoustEEEcare"
OUTPUT_DIR     = Path("recordings_no_sd")
SCAN_TIMEOUT   = 20.0
REC_TIMEOUT    = 120.0   # total per-recording timeout (s)
RESULT_GRACE_S = 5.0     # wait after HEART_DONE before assuming lung skipped

NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # nRF → host
NUS_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # host → nRF

CHUNK_HEADER_BYTES = 4    # [seq16 LE][len16 LE]
WRITE_CHUNK_SIZE   = 240  # conservative MTU-safe payload limit

SAMPLE_RATE = 8000

# MFCC params — must match nRF dsp_mfcc configs exactly
HEART_CFG = dict(
    target_rate  = 2000,
    frame_ms     = 30,
    hop_pct      = 0.50,
    fft_size     = 512,
    n_mel        = 25,
    n_mfcc       = 25,
    n_frames     = 665,
    f_low        = 10,
    f_high       = 1000,
)
LUNG_CFG = dict(
    target_rate  = 4000,
    frame_ms     = 300,
    hop_pct      = 0.10,
    fft_size     = 2048,
    n_mel        = 26,
    n_mfcc       = 26,
    n_frames     = 324,
    f_low        = 100,
    f_high       = 2000,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,   # suppress bleak noise; use print() for UX
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("receiver")


# ---------------------------------------------------------------------------
# Recording state
# ---------------------------------------------------------------------------
class RecState:
    """Holds all mutable state for one recording session."""

    def __init__(self):
        self.reset()

    def reset(self):
        # Audio
        self.pcm_expected    = 0
        self.pcm_chunks      = {}      # seq → bytes
        self.pcm_next_seq    = 0
        self.pcm_total_bytes = 0
        self.pcm_gaps        = 0
        self.pcm_done        = asyncio.Event()

        # HR / RR results
        self.hr_value        = None
        self.rr_value        = None
        self.hr_evt          = asyncio.Event()
        self.heart_done_evt  = asyncio.Event()
        self.rr_evt          = asyncio.Event()

        # Error
        self.error           = None
        self.error_evt       = asyncio.Event()

        # Progress display
        self._last_progress  = 0.0
        self._line_buf       = ""

    # ------------------------------------------------------------------
    # BLE notify callback
    # ------------------------------------------------------------------
    def on_notify(self, _handle, data: bytearray):
        # Binary audio chunk heuristic: 4-byte header where len16 == len(data)-4
        if (len(data) >= CHUNK_HEADER_BYTES and
                self.pcm_expected > 0 and
                not self.pcm_done.is_set()):
            seq, payload_len = struct.unpack_from("<HH", data, 0)
            if payload_len == len(data) - CHUNK_HEADER_BYTES and payload_len > 0:
                self._handle_pcm_chunk(seq, bytes(data[CHUNK_HEADER_BYTES:]))
                return

        # ASCII line path
        try:
            self._line_buf += data.decode("ascii")
        except UnicodeDecodeError:
            return

        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            line = line.strip()
            if line:
                self._handle_line(line)

    def _handle_pcm_chunk(self, seq: int, payload: bytes):
        if seq != self.pcm_next_seq:
            gap = (seq - self.pcm_next_seq) & 0xFFFF
            print(f"\n  [GAP   ] audio seq {self.pcm_next_seq}→{seq} "
                  f"({gap} missing)")
            # Zero-fill to maintain int16 alignment
            self.pcm_chunks[self.pcm_next_seq] = b"\x00" * (gap * len(payload))
            self.pcm_gaps += gap

        self.pcm_chunks[seq]  = payload
        self.pcm_next_seq     = (seq + 1) & 0xFFFF
        self.pcm_total_bytes += len(payload)
        _print_progress("AUDIO", self.pcm_total_bytes, self.pcm_expected)

    def _handle_line(self, line: str):
        log.debug("nRF→host: %r", line)

        if line.startswith("START:"):
            try:
                self.pcm_expected = int(line[6:])
                print(f"\n  [AUDIO ] expecting {self.pcm_expected} B "
                      f"({self.pcm_expected / 2 / SAMPLE_RATE:.1f} s)")
            except ValueError:
                print(f"  [WARN  ] Malformed START: {line!r}")
            return

        if line == "finished":
            self.pcm_done.set()
            print(f"\n  [AUDIO ] done — "
                  f"{self.pcm_total_bytes}/{self.pcm_expected} B  "
                  f"gaps={self.pcm_gaps}")
            return

        if line.startswith("HR:"):
            try:
                self.hr_value = float(line[3:])
                print(f"\n  [HR    ] {self.hr_value:.0f} BPM")
                self.hr_evt.set()
            except ValueError:
                print(f"  [WARN  ] Bad HR: {line!r}")
            return

        if line == "HEART_DONE":
            self.heart_done_evt.set()
            return

        if line.startswith("RR:"):
            try:
                self.rr_value = float(line[3:])
                print(f"\n  [RR    ] {self.rr_value:.0f} breaths/min")
                self.rr_evt.set()
            except ValueError:
                print(f"  [WARN  ] Bad RR: {line!r}")
            return

        if line.startswith("ERR:"):
            self.error = line[4:]
            print(f"\n  [ERROR ] Device: {self.error}")
            self.error_evt.set()
            # Unblock any waiting event so the pipeline doesn't hang
            self.pcm_done.set()
            self.hr_evt.set()
            self.heart_done_evt.set()
            self.rr_evt.set()
            return

        log.debug("Unhandled: %r", line)


# ---------------------------------------------------------------------------
# Connection-level state (persists across recordings)
# ---------------------------------------------------------------------------
class ConnState:
    def __init__(self):
        self.quant_heart_scale = 1.0
        self.quant_heart_zp    = 0
        self.quant_lung_scale  = 1.0
        self.quant_lung_zp     = 0
        self.lung_disabled     = False
        self._line_buf         = ""

    def on_notify_conn(self, _handle, data: bytearray):
        """Handles connection-level messages (QUANT_*, LUNG_DISABLED)."""
        try:
            self._line_buf += data.decode("ascii")
        except UnicodeDecodeError:
            return
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            line = line.strip()
            if line:
                self._handle_conn_line(line)

    def _handle_conn_line(self, line: str):
        if line.startswith("QUANT_HEART:"):
            parts = line[12:].split(":")
            if len(parts) == 2:
                self.quant_heart_scale = float.fromhex(parts[0])
                self.quant_heart_zp    = int(parts[1])
                print(f"  [QUANT ] heart scale={self.quant_heart_scale:.6f} "
                      f"zp={self.quant_heart_zp}")
            return

        if line.startswith("QUANT_LUNG:"):
            parts = line[11:].split(":")
            if len(parts) == 2:
                self.quant_lung_scale = float.fromhex(parts[0])
                self.quant_lung_zp    = int(parts[1])
                print(f"  [QUANT ] lung  scale={self.quant_lung_scale:.6f} "
                      f"zp={self.quant_lung_zp}")
            return

        if line == "LUNG_DISABLED":
            self.lung_disabled = True
            print("  [INFO  ] Lung model disabled on device.")
            return


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------
_last_progress_t = 0.0

def _print_progress(label: str, got: int, expected: int):
    global _last_progress_t
    now = time.monotonic()
    if now - _last_progress_t < 0.25:
        return
    _last_progress_t = now
    pct = min(got * 100 // max(expected, 1), 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{label:<5}][{bar}] {pct:3d}%  {got:>8}/{expected} B",
          end="", flush=True)


# ---------------------------------------------------------------------------
# PCM reassembly
# ---------------------------------------------------------------------------
def reassemble_pcm(rec: RecState) -> np.ndarray:
    ordered = b"".join(rec.pcm_chunks[k] for k in sorted(rec.pcm_chunks))
    # Trim or pad to expected length
    target = rec.pcm_expected
    if len(ordered) > target:
        ordered = ordered[:target]
    elif len(ordered) < target:
        ordered += b"\x00" * (target - len(ordered))
    pcm = np.frombuffer(ordered, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


# ---------------------------------------------------------------------------
# WAV saving
# ---------------------------------------------------------------------------
def save_wav(pcm: np.ndarray, path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16.tobytes())
    kb  = path.stat().st_size / 1024
    dur = len(pcm_int16) / SAMPLE_RATE
    print(f"  [SAVE  ] {path}  ({kb:.1f} kB, {dur:.2f} s, 8 kHz mono 16-bit)")


# ---------------------------------------------------------------------------
# MFCC computation
# ---------------------------------------------------------------------------
def compute_mfcc(pcm: np.ndarray, cfg: dict) -> np.ndarray:
    import librosa

    pcm_rs = librosa.resample(pcm, orig_sr=SAMPLE_RATE,
                              target_sr=cfg["target_rate"])

    frame_samples = int(cfg["frame_ms"] / 1000.0 * cfg["target_rate"])
    hop_samples   = int(frame_samples * cfg["hop_pct"])

    mfcc = librosa.feature.mfcc(
        y          = pcm_rs,
        sr         = cfg["target_rate"],
        n_mfcc     = cfg["n_mfcc"],
        n_fft      = cfg["fft_size"],
        hop_length = hop_samples,
        win_length = frame_samples,
        n_mels     = cfg["n_mel"],
        fmin       = cfg["f_low"],
        fmax       = cfg["f_high"],
    )
    # librosa → [n_mfcc, n_frames]; transpose to [n_frames, n_mfcc]
    mfcc = mfcc.T.astype(np.float32)

    n_exp = cfg["n_frames"]
    if mfcc.shape[0] > n_exp:
        mfcc = mfcc[:n_exp]
    elif mfcc.shape[0] < n_exp:
        pad  = np.zeros((n_exp - mfcc.shape[0], cfg["n_mfcc"]), dtype=np.float32)
        mfcc = np.concatenate([mfcc, pad], axis=0)

    return mfcc


def mfcc_to_raw_int8(mfcc_f32: np.ndarray, scale: float, zp: int) -> bytes:
    """
    Quantize float32 MFCC → int8 raw bytes (no .npy header).
    Formula: q = clip(round(f / scale) + zp, -128, 127)
    """
    q = np.round(mfcc_f32 / scale).astype(np.int32) + zp
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.tobytes()


# ---------------------------------------------------------------------------
# BLE MFCC sender
# ---------------------------------------------------------------------------
async def send_mfcc_chunks(client: BleakClient,
                           raw_int8: bytes,
                           label: str) -> None:
    n_bytes    = len(raw_int8)
    payload_sz = WRITE_CHUNK_SIZE - CHUNK_HEADER_BYTES
    offset     = 0
    seq        = 0

    await client.write_gatt_char(
        NUS_RX_CHAR_UUID,
        f"{label}_START:{n_bytes}\n".encode(),
        response=False,
    )
    await asyncio.sleep(0.02)

    print(f"\n  [{label.split('_')[1]:<5}] sending {n_bytes} B to nRF...")
    while offset < n_bytes:
        payload = raw_int8[offset: offset + payload_sz]
        header  = struct.pack("<HH", seq & 0xFFFF, len(payload))
        await client.write_gatt_char(
            NUS_RX_CHAR_UUID, header + payload, response=False
        )
        offset += len(payload)
        seq    += 1
        _print_progress(label.split("_")[1], offset, n_bytes)
        await asyncio.sleep(0.002)

    await client.write_gatt_char(
        NUS_RX_CHAR_UUID,
        f"{label}_END\n".encode(),
        response=False,
    )
    print(f"\r  [{label.split('_')[1]:<5}] sent {n_bytes} B in {seq} chunks"
          + " " * 20)


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------
def save_result_txt(value: float, path: Path, label: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = f"{label}:{value:.0f}\n"
    path.write_text(text, encoding="utf-8")
    print(f"  [SAVE  ] {path}  → '{text.strip()}'")


# ---------------------------------------------------------------------------
# Single recording pipeline
# ---------------------------------------------------------------------------
async def do_recording(client: BleakClient,
                       conn: ConnState,
                       rec_num: int) -> None:
    rec = RecState()

    # Switch notify handler to this recording's state
    await client.stop_notify(NUS_TX_CHAR_UUID)
    await client.start_notify(NUS_TX_CHAR_UUID, rec.on_notify)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nRecording #{rec_num} — sending REC …")
    await client.write_gatt_char(NUS_RX_CHAR_UUID, b"REC", response=False)

    # ── Phase 1: receive PCM ────────────────────────────────────────────
    print(f"  [Phase 1] Receiving PCM audio…")
    try:
        await asyncio.wait_for(rec.pcm_done.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        print("  [WARN  ] Timeout waiting for audio — saving partial.")

    if rec.error:
        print(f"  [ABORT ] Device error in Phase 1: {rec.error}")
        return

    pcm = reassemble_pcm(rec)

    # Save WAV immediately
    wav_path = OUTPUT_DIR / f"rec_{rec_num:03d}_{ts}.wav"
    save_wav(pcm, wav_path)

    # ── Phase 2: compute + send heart MFCC ─────────────────────────────
    print(f"  [Phase 2] Computing heart MFCC…")
    heart_f32  = compute_mfcc(pcm, HEART_CFG)
    heart_int8 = mfcc_to_raw_int8(
        heart_f32, conn.quant_heart_scale, conn.quant_heart_zp
    )
    await send_mfcc_chunks(client, heart_int8, "MFCC_HEART")

    # ── Phase 3: receive HR ─────────────────────────────────────────────
    print(f"  [Phase 3] Waiting for HR result…")
    try:
        await asyncio.wait_for(rec.hr_evt.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        print("  [WARN  ] Timeout waiting for HR result.")

    # ── Phase 4+5: lung (skip if disabled) ─────────────────────────────
    if conn.lung_disabled:
        print(f"  [Phase 4] Skipped — LUNG_DISABLED.")
    else:
        print(f"  [Phase 4] Waiting for HEART_DONE…")
        try:
            await asyncio.wait_for(rec.heart_done_evt.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            print("  [WARN  ] Timeout waiting for HEART_DONE.")

        print(f"  [Phase 4] Computing lung MFCC…")
        lung_f32  = compute_mfcc(pcm, LUNG_CFG)
        lung_int8 = mfcc_to_raw_int8(
            lung_f32, conn.quant_lung_scale, conn.quant_lung_zp
        )
        await send_mfcc_chunks(client, lung_int8, "MFCC_LUNG")

        print(f"  [Phase 5] Waiting for RR result…")
        try:
            await asyncio.wait_for(rec.rr_evt.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            print("  [WARN  ] Timeout waiting for RR result.")

    # ── Save results ────────────────────────────────────────────────────
    if rec.hr_value is not None:
        save_result_txt(rec.hr_value, OUTPUT_DIR / f"hr_{rec_num:03d}_{ts}.txt", "HR")
    else:
        print("  [INFO  ] No HR result received (model disabled or failed).")

    if rec.rr_value is not None:
        save_result_txt(rec.rr_value, OUTPUT_DIR / f"rr_{rec_num:03d}_{ts}.txt", "RR")
    elif not conn.lung_disabled:
        print("  [INFO  ] No RR result received (model disabled or failed).")

    print(f"\n  {'─'*46}")
    print(f"  Recording #{rec_num} complete.")
    if rec.hr_value is not None:
        print(f"    HR : {rec.hr_value:.0f} BPM")
    if rec.rr_value is not None:
        print(f"    RR : {rec.rr_value:.0f} breaths/min")
    print(f"    WAV: {wav_path}")
    print(f"  {'─'*46}\n")

    # Restore connection-level notify handler for next QUANT_* (reconnect)
    await client.stop_notify(NUS_TX_CHAR_UUID)
    await client.start_notify(NUS_TX_CHAR_UUID, conn.on_notify_conn)


# ---------------------------------------------------------------------------
# Main — stays connected across recordings
# ---------------------------------------------------------------------------
async def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for '{DEVICE_NAME}' …")
    device = await BleakScanner.find_device_by_name(
        DEVICE_NAME, timeout=SCAN_TIMEOUT
    )
    if device is None:
        print(f"ERROR: '{DEVICE_NAME}' not found within {SCAN_TIMEOUT:.0f} s.")
        sys.exit(1)

    print(f"  Found: {DEVICE_NAME}  [{device.address}]")

    async with BleakClient(device, timeout=30.0) as client:
        print(f"Connected!  MTU={client.mtu_size}")

        conn = ConnState()
        await client.start_notify(NUS_TX_CHAR_UUID, conn.on_notify_conn)

        # Give nRF time to send QUANT_* after MTU exchange
        print("Waiting for quantization parameters…")
        await asyncio.sleep(1.5)

        print("\nPress ENTER to start a recording, q+ENTER to quit.\n")

        rec_num = 1
        loop    = asyncio.get_event_loop()

        try:
            while True:
                prompt = (
                    f"{'─'*50}\n"
                    f"[Ready] ENTER = recording #{rec_num}  "
                    f"(q+ENTER = quit): "
                )
                inp = await loop.run_in_executor(None, lambda: input(prompt))

                if inp.strip().lower() == "q":
                    break

                await do_recording(client, conn, rec_num)
                rec_num += 1

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            try:
                await client.stop_notify(NUS_TX_CHAR_UUID)
            except Exception:
                pass
            print("Disconnected.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run())