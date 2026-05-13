#!/usr/bin/env python3
"""
tflite_to_header.py
===================
Convert a .tflite model file into a C header containing the model
as a uint8_t array.  Cross-platform replacement for `xxd -i`.

Usage:
    python tflite_to_header.py trial_144_int8.tflite heart_model.h g_heart_model_data
    python tflite_to_header.py lung_model.tflite     lung_model.h  g_lung_model_data

Arguments:
    1. Path to input .tflite file
    2. Path to output .h file
    3. Name of the C array (e.g. g_heart_model_data)

The output header defines:
    alignas(8) const unsigned char <name>[]      = { ... };
    const unsigned int             <name>_len    = <length>;

The `alignas(8)` is important: TFLite Micro requires the flatbuffer
to be at least 4-byte aligned, and 8-byte alignment is safer across
all targets.
"""

import sys
import os
from pathlib import Path


def convert(in_path: str, out_path: str, array_name: str) -> None:
    in_path  = Path(in_path)
    out_path = Path(out_path)

    if not in_path.is_file():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    data = in_path.read_bytes()
    n    = len(data)

    # Derive a header-guard symbol from the array name.
    guard = array_name.upper().replace(' ', '_') + "_H"

    lines = []
    lines.append("/*")
    lines.append(f" * {out_path.name}")
    lines.append(f" * Auto-generated from {in_path.name} ({n} bytes)")
    lines.append(" * DO NOT EDIT BY HAND.")
    lines.append(" */")
    lines.append("")
    lines.append(f"#ifndef {guard}")
    lines.append(f"#define {guard}")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("extern \"C\" {")
    lines.append("#endif")
    lines.append("")
    lines.append("/* 8-byte alignment is required by TFLite Micro for the flatbuffer. */")
    lines.append("#if defined(__GNUC__) || defined(__clang__)")
    lines.append(f"const unsigned char {array_name}[] __attribute__((aligned(8))) = {{")
    lines.append("#else")
    lines.append(f"alignas(8) const unsigned char {array_name}[] = {{")
    lines.append("#endif")

    # 12 bytes per line, hex with 0x prefix, comma-separated.
    BYTES_PER_LINE = 12
    for i in range(0, n, BYTES_PER_LINE):
        chunk = data[i:i + BYTES_PER_LINE]
        hex_bytes = ", ".join(f"0x{b:02x}" for b in chunk)
        # Trailing comma on every line except the last.
        suffix = "," if i + BYTES_PER_LINE < n else ""
        lines.append(f"  {hex_bytes}{suffix}")

    lines.append("};")
    lines.append("")
    lines.append(f"const unsigned int {array_name}_len = {n};")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")
    lines.append(f"#endif /* {guard} */")

    out_path.write_text("\n".join(lines))
    print(f"OK: wrote {out_path}  ({n} bytes -> array '{array_name}')")


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)

    in_path    = sys.argv[1]
    out_path   = sys.argv[2]
    array_name = sys.argv[3]
    convert(in_path, out_path, array_name)


if __name__ == "__main__":
    main()
