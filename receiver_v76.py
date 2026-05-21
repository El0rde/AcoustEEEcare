#!/usr/bin/env python3
"""
receiver.py — AcoustEEEcare host-side BLE receiver
====================================================
v8.1 — Aligned with main.c v8.1 protocol fixes.

Pipeline:
    1. Connect to "AcoustEEEcare" over BLE NUS.
    2. Receive QUANT_HEART / QUANT_LUNG (or LUNG_DISABLED) params.
    3. Send "REC" to trigger recording.
    4. Receive chunked PCM audio  [seq16][len16][pcm_bytes]  until "finished\n".
    5. Compute heart MFCC (float32 -> int8 using QUANT_HEART params).
       Send raw int8 chunks to nRF (NO .npy header).
    6. Receive "HR:<val>\n".
    7. Receive "HEART_DONE\n", compute + send lung MFCC (or skip if LUNG_DISABLED).
    8. Receive "RR:<val>\n".
    9. Save results to results.txt.

Issue fixes implemented here:
    #1  .npy header stripped: mfcc_to_raw_int8() returns raw bytes only.
    #2  Results saved to results.txt (timestamped).
    #9  LUNG_DISABLED message handled; lung phases skipped gracefully.
    #10 seq16 gap detection on received audio chunks.

Dependencies:
    pip install bleak numpy librosa

Usage:
    python receiver.py [--output results.txt] [--device "AcoustEEEcare"] [--verbose]
"""

import argparse
import asyncio
import logging
import struct
import datetime
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("receiver")

# ---------------------------------------------------------------------------
# NUS UUIDs (Nordic UART Service)
# ---------------------------------------------------------------------------
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # host -> nRF (write)
NUS_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # nRF -> host (notify)

CHUNK_HEADER_BYTES = 4    # [seq16 LE][len16 LE]
WRITE_CHUNK_SIZE   = 240  # conservative MTU-safe payload limit

# ---------------------------------------------------------------------------
# MFCC parameters — must match nRF dsp_mfcc configs exactly
# ---------------------------------------------------------------------------
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

SAMPLING_RATE = 8000

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
class Session:
    def __init__(self):
        self.quant_heart_scale = 1.0
        self.quant_heart_zp    = 0
        self.quant_lung_scale  = 1.0
        self.quant_lung_zp     = 0
        self.lung_disabled     = False

        self.pcm_expected  = 0
        self.pcm_chunks    = {}       # seq(int) -> bytes
        self.pcm_next_seq  = 0
        self.pcm_done      = asyncio.Event()

        self.hr_value      = None
        self.rr_value      = None

        self.hr_evt         = asyncio.Event()
        self.heart_done_evt = asyncio.Event()
        self.rr_evt         = asyncio.Event()

        self.error     = None
        self.error_evt = asyncio.Event()

        self._line_buf = ""

    # ------------------------------------------------------------------
    def on_notify(self, _handle, data: bytearray):
        """BLE notify callback — runs in bleak event loop."""
        try:
            text = data.decode("ascii")
        except UnicodeDecodeError:
            self._handle_pcm_chunk(bytes(data))
            return

        self._line_buf += text
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            line = line.strip()
            if line:
                self._handle_line(line)

    def _handle_line(self, line: str):
        log.debug("nRF->host: %r", line)

        if line.startswith("QUANT_HEART:"):
            parts = line[12:].split(":")
            if len(parts) == 2:
                self.quant_heart_scale = float.fromhex(parts[0])
                self.quant_heart_zp    = int(parts[1])
                log.info("QUANT_HEART  scale=%g  zp=%d",
                         self.quant_heart_scale, self.quant_heart_zp)
            return

        if line.startswith("QUANT_LUNG:"):
            parts = line[11:].split(":")
            if len(parts) == 2:
                self.quant_lung_scale = float.fromhex(parts[0])
                self.quant_lung_zp    = int(parts[1])
                log.info("QUANT_LUNG  scale=%g  zp=%d",
                         self.quant_lung_scale, self.quant_lung_zp)
            return

        if line == "LUNG_DISABLED":
            # Issue #9: skip lung phases entirely
            self.lung_disabled = True
            log.info("Lung model disabled on device — skipping lung pipeline")
            return

        if line.startswith("START:"):
            try:
                self.pcm_expected = int(line[6:])
                log.info("Audio stream starting — expecting %d PCM bytes",
                         self.pcm_expected)
            except ValueError:
                log.warning("Bad START line: %r", line)
            return

        if line == "finished":
            log.info("Audio stream finished")
            self.pcm_done.set()
            return

        if line.startswith("HR:"):
            try:
                self.hr_value = float(line[3:])
                log.info("HR = %.0f BPM", self.hr_value)
                self.hr_evt.set()
            except ValueError:
                log.warning("Bad HR line: %r", line)
            return

        if line == "HEART_DONE":
            log.info("HEART_DONE received")
            self.heart_done_evt.set()
            return

        if line.startswith("RR:"):
            try:
                self.rr_value = float(line[3:])
                log.info("RR = %.0f breaths/min", self.rr_value)
                self.rr_evt.set()
            except ValueError:
                log.warning("Bad RR line: %r", line)
            return

        if line.startswith("ERR:"):
            self.error = line[4:]
            log.error("Device error: %s", self.error)
            self.error_evt.set()
            return

        log.debug("Unhandled line: %r", line)

    def _handle_pcm_chunk(self, data: bytes):
        """Binary audio chunk: [seq16 LE][len16 LE][pcm_payload]."""
        if len(data) < CHUNK_HEADER_BYTES:
            log.warning("Short binary chunk (%d B) — discarding", len(data))
            return

        seq, payload_len = struct.unpack_from("<HH", data, 0)
        payload = data[CHUNK_HEADER_BYTES: CHUNK_HEADER_BYTES + payload_len]

        # Issue #10: warn on gaps (BLE stack should be ordered, but log anyway)
        if seq != self.pcm_next_seq:
            log.warning("PCM SEQ GAP: expected %d got %d",
                        self.pcm_next_seq, seq)

        self.pcm_chunks[seq] = payload
        self.pcm_next_seq = seq + 1
        log.debug("PCM chunk seq=%d  %d B  total_seqs=%d",
                  seq, len(payload), len(self.pcm_chunks))


# ---------------------------------------------------------------------------
# PCM reassembly
# ---------------------------------------------------------------------------
def reassemble_pcm(session: Session) -> np.ndarray:
    ordered = b"".join(session.pcm_chunks[k]
                       for k in sorted(session.pcm_chunks))
    pcm = np.frombuffer(ordered, dtype=np.int16).astype(np.float32) / 32768.0
    log.info("Reassembled PCM: %d samples (%.2f s)",
             len(pcm), len(pcm) / SAMPLING_RATE)
    return pcm


# ---------------------------------------------------------------------------
# MFCC computation
# ---------------------------------------------------------------------------
def compute_mfcc(pcm: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Compute MFCC [n_frames, n_mfcc] float32.
    Matches nRF dsp_mfcc.c pipeline parameters.
    """
    import librosa

    pcm_rs = librosa.resample(pcm, orig_sr=SAMPLING_RATE,
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
    # librosa: [n_mfcc, n_frames] -> [n_frames, n_mfcc]
    mfcc = mfcc.T.astype(np.float32)

    n_exp = cfg["n_frames"]
    if mfcc.shape[0] > n_exp:
        mfcc = mfcc[:n_exp]
    elif mfcc.shape[0] < n_exp:
        pad = np.zeros((n_exp - mfcc.shape[0], cfg["n_mfcc"]), dtype=np.float32)
        mfcc = np.concatenate([mfcc, pad], axis=0)

    log.info("MFCC shape: %s  (expected [%d, %d])",
             mfcc.shape, n_exp, cfg["n_mfcc"])
    return mfcc


def mfcc_to_raw_int8(mfcc_f32: np.ndarray, scale: float, zp: int) -> bytes:
    """
    Quantize float32 MFCC to int8 and return raw bytes.

    Issue #1: Returns raw int8 payload ONLY — no .npy magic bytes,
    no header, no dtype descriptor. The nRF firmware expects bare
    int8 data; adding a .npy header would corrupt the first tensor
    rows with 128 bytes of header garbage.

    Quantization formula:
        q = clip(round(f / scale) + zp, -128, 127)
    """
    q = np.round(mfcc_f32 / scale).astype(np.int32) + zp
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.tobytes()   # row-major C order — matches nRF memcpy layout


# ---------------------------------------------------------------------------
# BLE chunk sender
# ---------------------------------------------------------------------------
async def send_mfcc_chunks(client: BleakClient,
                           raw_int8: bytes,
                           label: str) -> None:
    """
    Send raw int8 MFCC to nRF framed as [seq16 LE][len16 LE][payload].
    Sends LABEL_START:<n>\n first, then chunks, then LABEL_END\n.
    """
    n_bytes    = len(raw_int8)
    payload_sz = WRITE_CHUNK_SIZE - CHUNK_HEADER_BYTES
    offset     = 0
    seq        = 0

    start_msg = f"{label}_START:{n_bytes}\n".encode()
    await client.write_gatt_char(NUS_RX_CHAR_UUID, start_msg, response=False)
    await asyncio.sleep(0.02)

    while offset < n_bytes:
        payload = raw_int8[offset: offset + payload_sz]
        header  = struct.pack("<HH", seq & 0xFFFF, len(payload))
        await client.write_gatt_char(
            NUS_RX_CHAR_UUID, header + payload, response=False
        )
        offset += len(payload)
        seq    += 1
        await asyncio.sleep(0.002)   # yield to avoid flooding BLE TX queue

    end_msg = f"{label}_END\n".encode()
    await client.write_gatt_char(NUS_RX_CHAR_UUID, end_msg, response=False)
    log.info("Sent %s: %d bytes in %d chunks", label, n_bytes, seq)


# ---------------------------------------------------------------------------
# Result saving (Issue #2)
# ---------------------------------------------------------------------------
def save_results(output_path: Path,
                 hr,
                 rr,
                 lung_disabled: bool) -> None:
    """
    Save HR and RR results to a plain-text file.

    File format:
        timestamp: 2025-06-01T14:32:00
        HR_bpm: 72
        RR_breaths_per_min: 16
        lung_model: enabled
    """
    ts         = datetime.datetime.now().isoformat(timespec="seconds")
    hr_str     = f"{hr:.0f}" if hr is not None else "N/A"
    rr_str     = f"{rr:.0f}" if rr is not None else "N/A"
    lung_state = "disabled" if lung_disabled else "enabled"

    content = (
        f"timestamp: {ts}\n"
        f"HR_bpm: {hr_str}\n"
        f"RR_breaths_per_min: {rr_str}\n"
        f"lung_model: {lung_state}\n"
    )

    output_path.write_text(content, encoding="utf-8")
    log.info("Results saved -> %s", output_path)
    print("\n" + "=" * 40)
    print(content.strip())
    print("=" * 40 + "\n")


# ---------------------------------------------------------------------------
# Main async pipeline
# ---------------------------------------------------------------------------
async def run(device_name: str, output_path: Path) -> None:
    log.info("Scanning for '%s'...", device_name)
    device = await BleakScanner.find_device_by_name(device_name, timeout=20.0)
    if device is None:
        raise RuntimeError(f"Device '{device_name}' not found within scan timeout")

    log.info("Found: %s  (%s)", device.name, device.address)
    session = Session()

    async with BleakClient(device, timeout=30.0) as client:
        log.info("Connected  MTU=%d", client.mtu_size)
        await client.start_notify(NUS_TX_CHAR_UUID, session.on_notify)

        # nRF sends QUANT_* right after MTU exchange — give it a moment
        log.info("Waiting for quantization parameters...")
        await asyncio.sleep(1.5)

        # -- Send REC -------------------------------------------------
        log.info("Sending REC command...")
        await client.write_gatt_char(NUS_RX_CHAR_UUID, b"REC", response=False)

        # -- Phase 1: receive PCM audio -------------------------------
        log.info("Phase 1: receiving PCM audio...")
        try:
            await asyncio.wait_for(session.pcm_done.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout waiting for PCM audio stream")

        pcm = reassemble_pcm(session)

        # -- Phase 2: compute + send heart MFCC -----------------------
        log.info("Phase 2: computing heart MFCC...")
        heart_f32  = compute_mfcc(pcm, HEART_CFG)
        heart_int8 = mfcc_to_raw_int8(
            heart_f32, session.quant_heart_scale, session.quant_heart_zp
        )
        log.info("Heart MFCC: %d raw int8 bytes (no .npy header)", len(heart_int8))
        await send_mfcc_chunks(client, heart_int8, "MFCC_HEART")

        # -- Phase 3: receive HR -------------------------------------
        log.info("Phase 3: waiting for HR result...")
        try:
            await asyncio.wait_for(session.hr_evt.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout waiting for HR result")

        # -- Phase 4 + 5: lung ---------------------------------------
        if session.lung_disabled:
            log.info("Phase 4+5: skipped — device sent LUNG_DISABLED")
        else:
            log.info("Phase 4: waiting for HEART_DONE signal...")
            try:
                await asyncio.wait_for(session.heart_done_evt.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                raise RuntimeError("Timeout waiting for HEART_DONE")

            log.info("Phase 4: computing lung MFCC...")
            lung_f32  = compute_mfcc(pcm, LUNG_CFG)
            lung_int8 = mfcc_to_raw_int8(
                lung_f32, session.quant_lung_scale, session.quant_lung_zp
            )
            log.info("Lung MFCC: %d raw int8 bytes (no .npy header)", len(lung_int8))
            await send_mfcc_chunks(client, lung_int8, "MFCC_LUNG")

            log.info("Phase 5: waiting for RR result...")
            try:
                await asyncio.wait_for(session.rr_evt.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                raise RuntimeError("Timeout waiting for RR result")

        # -- Save results --------------------------------------------
        save_results(output_path, session.hr_value, session.rr_value,
                     session.lung_disabled)

        await client.stop_notify(NUS_TX_CHAR_UUID)
        log.info("Session complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AcoustEEEcare BLE receiver v8.1")
    parser.add_argument("--device",  default="AcoustEEEcare",
                        help="BLE advertised device name (default: AcoustEEEcare)")
    parser.add_argument("--output",  default="results.txt",
                        help="Path to save HR/RR results (default: results.txt)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        asyncio.run(run(args.device, Path(args.output)))
    except RuntimeError as exc:
        log.error("Fatal: %s", exc)
        raise SystemExit(1)
    except KeyboardInterrupt:
        log.info("Interrupted.")


if __name__ == "__main__":
    main()