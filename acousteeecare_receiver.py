"""
AcoustEEEcare BLE Receiver — Windows
=====================================
Connects to the "AcoustEEEcare" device, sends REC,
receives sequenced int16 PCM over Nordic UART Service (NUS), and saves
each recording as a .wav file.  Waits for user keypress before each cycle.

Wire format (firmware v6.1):
    Each BLE notification is ONE of:
      a) Control message  — plain ASCII, no binary header
            "START:<total_bytes>\\n"
            "finished\\n"
            "ERR:<reason>"
      b) Audio chunk      — 4-byte header + PCM payload
            [0..1]  seq  uint16 LE   (wraps at 0xFFFF)
            [2..3]  len  uint16 LE   (payload byte count)
            [4..]   int16 PCM samples, little-endian

    "finished\\n" is sent by the firmware AFTER the ring buffer fully drains,
    so all audio chunks are guaranteed to arrive before it.

Requirements:
    pip install bleak

Usage:
    python acousteeecare_receiver.py

Optional flags:
    --device  <MAC or name>   override auto-scan
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

DEVICE_NAME  = "AcoustEEEcare"
SAMPLE_WIDTH = 2   # int16
CHANNELS     = 1   # mono


# ── Keypress helper ────────────────────────────────────────────────────────────
def wait_for_keypress(prompt: str) -> bool:
    """Blocks until ENTER.  Returns False if user typed 'q'."""
    try:
        return input(prompt).strip().lower() != "q"
    except EOFError:
        return True


# ── Packet classifier ──────────────────────────────────────────────────────────
def _is_audio_chunk(data: bytearray) -> bool:
    """
    Audio chunks satisfy: bytes[2:4] as uint16 LE == len(data) - 4.
    Text control messages never satisfy this because their byte[2:4]
    values are ASCII characters that don't equal (packet_length - 4).
    Examples:
      "START:160000\\n"  len=14, bytes[2:4]="AR"=0x5241=21057  ≠ 10  → text
      "finished\\n"      len=9,  bytes[2:4]="ni"=0x696E=26990  ≠ 5   → text
      audio chunk        len=508, bytes[2:4]=504 LE             == 504 → audio
    """
    if len(data) < 4:
        return False
    ln = struct.unpack_from("<H", data, 2)[0]
    return ln == len(data) - 4 and ln > 0


# ── Session state ──────────────────────────────────────────────────────────────
class Session:
    def __init__(self, sample_rate: int, out_dir: str):
        self.sample_rate = sample_rate
        self.out_dir     = out_dir
        self.rec_index   = 0
        self.reset()

    def reset(self):
        self.expected_bytes  = 0
        self.audio_buf       = bytearray()
        self.receiving       = False
        self.fw_finished     = False
        self.transfer_ok     = False
        self.done_event      = asyncio.Event()
        self.expected_seq    = 0
        self.total_gaps      = 0
        self.total_gap_bytes = 0
        self.chunks          = 0
        self.last_progress   = 0.0


# ── BLE notification handler ───────────────────────────────────────────────────
def make_notify_handler(session: Session):

    def handler(_sender, data: bytearray):
        if _is_audio_chunk(data) and session.receiving:
            _handle_audio(session, data)
        else:
            _handle_text(session, data)

    return handler


def _handle_text(session: Session, data: bytearray):
    try:
        text = data.decode("ascii", errors="replace").strip()
    except Exception:
        return

    if text.startswith("START:"):
        try:
            session.reset()
            session.expected_bytes = int(text.split(":")[1])
            session.receiving      = True
            print(f"\n  [START] expecting {session.expected_bytes} bytes "
                  f"({session.expected_bytes // SAMPLE_WIDTH} samples, "
                  f"{session.expected_bytes / SAMPLE_WIDTH / session.sample_rate:.1f} s)")
        except (ValueError, IndexError):
            print(f"  [WARN ] Malformed START: {text!r}")

    elif text.startswith("finished"):
        # Firmware sends this only after the ring buffer has fully drained,
        # so all audio chunks are already in session.audio_buf by now.
        session.fw_finished = True
        session.receiving   = False
        session.transfer_ok = True
        print(f"\r  [DATA] {len(session.audio_buf):>9} / "
              f"{session.expected_bytes} bytes received … done    ")
        session.done_event.set()

    elif text.startswith("ERR:"):
        print(f"\n  [ERROR] Firmware: {text}")
        session.receiving   = False
        session.transfer_ok = False
        session.done_event.set()


def _handle_audio(session: Session, data: bytearray):
    seq = struct.unpack_from("<H", data, 0)[0]
    ln  = struct.unpack_from("<H", data, 2)[0]
    payload = data[4 : 4 + ln]

    # Gap detection + zero-fill (Layer C)
    if seq != session.expected_seq:
        gap_pkts  = (seq - session.expected_seq) & 0xFFFF
        gap_bytes = gap_pkts * ln
        gap_ms    = gap_bytes // SAMPLE_WIDTH * 1000 // session.sample_rate
        print(f"\n  [GAP  ] seq {session.expected_seq}→{seq}: "
              f"{gap_pkts} pkt(s), ~{gap_ms} ms at sample {len(session.audio_buf)//SAMPLE_WIDTH}")
        session.audio_buf       += b"\x00\x00" * (gap_bytes // SAMPLE_WIDTH)
        session.total_gaps      += gap_pkts
        session.total_gap_bytes += gap_bytes

    session.audio_buf    += payload
    session.expected_seq  = (seq + 1) & 0xFFFF
    session.chunks       += 1

    # Progress bar — throttled to ~4 Hz
    now = time.monotonic()
    if now - session.last_progress >= 0.25:
        session.last_progress = now
        got = len(session.audio_buf)
        exp = session.expected_bytes
        pct = min(got * 100 // max(exp, 1), 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {got:>7}/{exp} B  gaps={session.total_gaps}",
              end="", flush=True)

    # Also complete on byte count in case firmware "finished" gets lost
    if session.expected_bytes > 0 and len(session.audio_buf) >= session.expected_bytes:
        session.receiving   = False
        session.transfer_ok = True
        print(f"\r  [DATA] {len(session.audio_buf):>9} / "
              f"{session.expected_bytes} bytes received … done    ")
        session.done_event.set()


# ── Save helpers ───────────────────────────────────────────────────────────────
def save_recording(session: Session) -> str:
    os.makedirs(session.out_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx = session.rec_index + 1

    buf = bytearray(session.audio_buf)
    target = session.expected_bytes

    if len(buf) > target:
        print(f"\n  [TRIM ] {len(buf) - target} extra bytes discarded")
        buf = buf[:target]
    elif len(buf) < target:
        pad = target - len(buf)
        print(f"\n  [PAD  ] {pad} bytes zero-padded")
        buf += b"\x00" * pad

    wav_path = os.path.join(session.out_dir, f"recording_{idx:03d}_{ts}.wav")
    dur_s    = len(buf) / SAMPLE_WIDTH / session.sample_rate
    print(f"  Writing WAV at {session.sample_rate} Hz, {len(buf)} bytes, {dur_s:.2f} s")

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(session.sample_rate)
        wf.writeframes(bytes(buf))

    return wav_path


# ── Device discovery ───────────────────────────────────────────────────────────
async def find_device(target: str | None) -> str:
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
    session.reset()
    session.done_event.clear()

    print("\n─────────────────────────────────────────")
    print(f"Recording #{session.rec_index + 1}  —  sending REC …")
    await client.write_gatt_char(NUS_RX_UUID, b"REC", response=False)

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

    got = len(session.audio_buf)
    exp = session.expected_bytes
    if exp and abs(got - exp) > SAMPLE_WIDTH * 10:
        print(f"\n  [WARN ] Size mismatch — expected {exp} B, got {got} B")

    if session.total_gaps:
        print(f"  Gaps detected   : {session.total_gaps} pkt(s), "
              f"{session.total_gap_bytes} B "
              f"({session.total_gap_bytes // SAMPLE_WIDTH * 1000 // session.sample_rate} ms)")

    wav_path = save_recording(session)
    samples  = exp // SAMPLE_WIDTH
    duration = samples / session.sample_rate
    print(f"\n  Saved {samples} samples ({duration:.2f} s)")
    print(f"    WAV → {wav_path}")
    session.rec_index += 1
    return True


# ── Main loop ──────────────────────────────────────────────────────────────────
async def main(args):
    address = await find_device(args.device)
    session = Session(sample_rate=args.rate, out_dir=args.out)

    print(f"\nConnecting to {address} …")
    async with BleakClient(address, timeout=20.0) as client:
        print(f"Connected!  MTU = {client.mtu_size}")
        handler = make_notify_handler(session)
        await client.start_notify(NUS_TX_UUID, handler)
        print("Subscribed to NUS notifications.")
        print("Press Ctrl+C at any time to stop.\n")

        try:
            while True:
                should_continue = await asyncio.get_event_loop().run_in_executor(
                    None,
                    wait_for_keypress,
                    f"\n[Ready] Press ENTER to start recording #{session.rec_index + 1} "
                    f"(or type 'q' + ENTER to quit): "
                )
                if not should_continue:
                    print("\nQuitting — user requested exit.")
                    break

                ok = await do_recording(client, session, args)
                if not ok:
                    print("  Recording failed — you can try again.")

        except KeyboardInterrupt:
            print("\n\nStopped by user.")

        await client.stop_notify(NUS_TX_UUID)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AcoustEEEcare BLE receiver")
    parser.add_argument("--device", default=None,
                        help="BLE MAC or device name substring (auto-scan if omitted)")
    parser.add_argument("--rate",   type=int, default=8000,
                        help="Sample rate in Hz (default: 8000)")
    parser.add_argument("--dur",    type=int, default=10,
                        help="Recording duration in seconds (default: 10)")
    parser.add_argument("--out",    default="recordings",
                        help="Output folder (default: ./recordings)")
    args = parser.parse_args()
    asyncio.run(main(args))