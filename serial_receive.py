"""
AcoustEEEcare BLE Receiver — Windows
=====================================
Connects to the "AcoustEEEcare" device, sends REC then SEND,
receives raw int16 PCM over Nordic UART Service (NUS), and saves
each recording as both .wav and .pcm.  Loops indefinitely.

Requirements:
    pip install bleak

Usage:
    python acousteeecare_receiver.py

Optional flags:
    --device  <MAC or name>   override auto-scan (e.g. "AA:BB:CC:DD:EE:FF")
    --rate    <hz>            sample rate (default 8000)
    --dur     <s>             expected recording duration in seconds (default 10)
    --out     <folder>        output folder (default ./recordings)
"""

import asyncio
import argparse
import struct
import time
import wave
import os
import sys
from datetime import datetime
from bleak import BleakScanner, BleakClient

# ── Nordic UART Service UUIDs ──────────────────────────────────────────────────
NUS_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
NUS_TX_UUID      = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # device → PC (notify)
NUS_RX_UUID      = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # PC → device (write)

DEVICE_NAME      = "AcoustEEEcare"
SAMPLE_WIDTH     = 2        # int16 = 2 bytes
CHANNELS         = 1        # mono


# ── Session state ──────────────────────────────────────────────────────────────
class Session:
    def __init__(self, sample_rate: int, out_dir: str):
        self.sample_rate   = sample_rate
        self.out_dir       = out_dir
        self.reset()

    def reset(self):
        self.expected_bytes  = None   # set when START:<n> arrives
        self.audio_buf       = bytearray()
        self.receiving       = False
        self.done_event      = asyncio.Event()
        self.transfer_ok     = False
        self.rec_index       = 0      # incremented each completed recording


# ── BLE notification handler ───────────────────────────────────────────────────
def make_notify_handler(session: Session):
    """Returns the callback that processes incoming NUS notifications."""

    leftover = bytearray()   # partial text tokens that span two packets

    def handler(_sender, data: bytearray):
        nonlocal leftover

        # ── While not yet in a binary transfer, look for control tokens ────
        if not session.receiving:
            # Prepend any leftover text from the last packet
            text_data = leftover + data
            leftover  = bytearray()

            # A packet might contain a control token followed immediately by
            # binary data (unlikely at 8 kHz but handle it defensively).
            try:
                text = text_data.decode("utf-8", errors="replace")
            except Exception:
                text = ""

            if text.startswith("START:"):
                # Extract byte count — rest of packet (if any) is audio data
                nl = text.find("\n")
                if nl != -1:
                    token  = text[: nl]
                    rest   = text_data[nl + 1 :]
                else:
                    token  = text.strip()
                    rest   = bytearray()

                try:
                    session.expected_bytes = int(token.split(":")[1])
                    session.audio_buf      = bytearray()
                    session.receiving      = True
                    print(f"\n  [START] expecting {session.expected_bytes} bytes "
                          f"({session.expected_bytes // SAMPLE_WIDTH} samples)")
                    if rest:
                        _append_audio(session, bytearray(rest.encode()
                                      if isinstance(rest, str) else rest))
                except ValueError:
                    print(f"  [WARN] Malformed START token: {token!r}")

            elif "REC:OK" in text:
                print("  [REC:OK] Device confirms recording finished — requesting transfer …")

            elif "ERR:" in text:
                err = text.strip()
                print(f"  [ERROR from device] {err}")
                session.transfer_ok = False
                session.done_event.set()

            else:
                # Might be a partial token; save for next packet
                leftover = text_data[-32:]   # keep last 32 bytes as context

        else:
            # ── Binary audio data ────────────────────────────────────────────
            # Check if this packet contains the "finished\n" footer
            footer = b"finished\n"
            idx    = data.find(footer)

            if idx != -1:
                # Audio is everything before the footer
                _append_audio(session, data[:idx])
                session.receiving   = False
                session.transfer_ok = True
                leftover            = bytearray()
                print(f"\r  [DATA] {len(session.audio_buf):>9} / "
                      f"{session.expected_bytes} bytes received … done    ")
                session.done_event.set()
            else:
                _append_audio(session, data)
                if session.expected_bytes:
                    pct = 100 * len(session.audio_buf) / session.expected_bytes
                    print(f"\r  [DATA] {len(session.audio_buf):>9} / "
                          f"{session.expected_bytes} bytes  ({pct:5.1f}%)", end="", flush=True)

    return handler


def _append_audio(session: Session, data: bytearray):
    session.audio_buf.extend(data)


# ── Save helpers ───────────────────────────────────────────────────────────────
def save_recording(session: Session) -> str:
    """Save audio_buf as a .wav file.  Returns wav_path."""
    os.makedirs(session.out_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx = session.rec_index

    wav_path = os.path.join(session.out_dir, f"recording_{idx:03d}_{ts}.wav")

    print(f"  Writing WAV at {session.sample_rate} Hz, {len(session.audio_buf)} bytes, {len(session.audio_buf) / 2 / session.sample_rate:.2f} s")


    # WAV (16-bit signed, mono, 8000 Hz)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(session.sample_rate)
        wf.writeframes(bytes(session.audio_buf))

    return wav_path


# ── Device discovery ───────────────────────────────────────────────────────────
async def find_device(target: str | None) -> str:
    """Scan and return the BLE address of the AcoustEEEcare device."""
    print(f"Scanning for '{target or DEVICE_NAME}' …")
    devices = await BleakScanner.discover(timeout=10.0)
    for d in devices:
        name = d.name or ""
        if target:
            if target.upper() == d.address.upper() or target.lower() in name.lower():
                print(f"  Found: {d.name} [{d.address}]")
                return d.address
        else:
            if DEVICE_NAME.lower() in name.lower():
                print(f"  Found: {d.name} [{d.address}]")
                return d.address
    raise RuntimeError(f"Device '{target or DEVICE_NAME}' not found in scan")


# ── Single recording cycle ─────────────────────────────────────────────────────
async def do_recording(client: BleakClient, session: Session, args):
    """Send REC — device records and streams simultaneously.  Collect audio and save."""

    # Reset state for this cycle
    session.reset()
    session.done_event.clear()
    session.transfer_ok = False

    print("\n─────────────────────────────────────────")
    print(f"Recording #{session.rec_index + 1}  —  sending REC …")

    await client.write_gatt_char(NUS_RX_UUID, b"REC", response=False)

    # Device streams audio immediately after REC.
    # Timeout = recording duration + generous BLE transfer margin.
    timeout_s = args.dur + 30
    print(f"  Streaming … (timeout {timeout_s} s)")
    try:
        await asyncio.wait_for(session.done_event.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        print("\n  [TIMEOUT] Transfer did not complete in time")
        return False

    if not session.transfer_ok:
        print("  [FAIL] Transfer flagged as failed")
        return False

    # Validate size
    got = len(session.audio_buf)
    exp = session.expected_bytes or 0
    if exp and abs(got - exp) > SAMPLE_WIDTH * 10:
        print(f"\n  [WARN] Size mismatch — expected {exp} B, got {got} B")

    # Save
    session.rec_index += 1
    wav_path = save_recording(session)
    samples  = got // SAMPLE_WIDTH
    duration = samples / session.sample_rate
    print(f"\n  Saved {samples} samples ({duration:.2f} s)")
    print(f"    WAV → {wav_path}")
    return True


# ── Main loop ──────────────────────────────────────────────────────────────────
async def main(args):
    address = await find_device(args.device)
    session = Session(sample_rate=args.rate, out_dir=args.out)

    print(f"\nConnecting to {address} …")
    async with BleakClient(address, timeout=20.0) as client:
        print(f"Connected!  MTU = {client.mtu_size}")

        # Subscribe to NUS TX notifications
        handler = make_notify_handler(session)
        await client.start_notify(NUS_TX_UUID, handler)
        print("Subscribed to NUS notifications.\n")
        print("Press Ctrl+C to stop after the current recording.\n")

        try:
            while True:
                ok = await do_recording(client, session, args)
                if not ok:
                    print("  Retrying in 3 s …")
                    await asyncio.sleep(3)
                else:
                    # Brief pause before next cycle so device can settle
                    print("\nWaiting 2 s before next recording …")
                    await asyncio.sleep(2)

        except KeyboardInterrupt:
            print("\n\nStopped by user.")

        await client.stop_notify(NUS_TX_UUID)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AcoustEEEcare BLE receiver")
    parser.add_argument("--device", default=None,
                        help="BLE MAC address or device name substring (auto-scan if omitted)")
    parser.add_argument("--rate",   type=int, default=8000,
                        help="Sample rate in Hz (default: 8000)")
    parser.add_argument("--dur",    type=int, default=10,
                        help="Recording duration in seconds (default: 10)")
    parser.add_argument("--out",    default="recordings",
                        help="Output folder (default: ./recordings)")
    args = parser.parse_args()

    asyncio.run(main(args))