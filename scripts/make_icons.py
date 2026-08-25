#!/usr/bin/env python3
"""Generate the application icon set Tauri needs to bundle.

The repository carries no binary artwork, so the icon is drawn from code: a
record with three coloured segments, one per crate, which is what the app is
about. Run once and commit the results, or let CI regenerate them.

    python scripts/make_icons.py
"""

import argparse
import math
import struct
import zlib
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parent.parent
ICONS = ROOT / "src-tauri" / "icons"

# One colour per crate, echoing the three-groups idea the UI is built around.
SEGMENTS = [
    (0.0, 120.0, (99, 102, 241)),  # indigo
    (120.0, 240.0, (16, 185, 129)),  # emerald
    (240.0, 360.0, (244, 114, 182)),  # pink
]
BACKGROUND = (17, 17, 27)
LABEL = (250, 250, 250)

PNG_SIZES = [16, 32, 48, 64, 128, 256, 512, 1024]
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
ICNS_SIZES = [
    ("ic07", 128),
    ("ic08", 256),
    ("ic09", 512),
    ("ic10", 1024),
    ("ic11", 32),
    ("ic12", 64),
    ("ic13", 256),
    ("ic14", 512),
]

Pixel = Tuple[int, int, int, int]


def render(size: int) -> List[Pixel]:
    """Draw the icon at `size`x`size` as RGBA pixels, supersampled 2x."""
    pixels: List[Pixel] = []
    centre = size / 2
    outer = size * 0.46
    inner = size * 0.13
    hole = size * 0.045
    samples = 2

    for y in range(size):
        for x in range(size):
            accumulated = [0.0, 0.0, 0.0, 0.0]
            for sub_y in range(samples):
                for sub_x in range(samples):
                    px = x + (sub_x + 0.5) / samples
                    py = y + (sub_y + 0.5) / samples
                    colour = sample(px, py, centre, outer, inner, hole)
                    for channel in range(4):
                        accumulated[channel] += colour[channel]
            count = samples * samples
            pixels.append(tuple(int(value / count) for value in accumulated))  # type: ignore[arg-type]
    return pixels


def sample(px: float, py: float, centre: float, outer: float, inner: float, hole: float) -> Pixel:
    dx, dy = px - centre, py - centre
    distance = math.hypot(dx, dy)

    if distance > outer:
        return (0, 0, 0, 0)
    if distance <= hole:
        return (*BACKGROUND, 255)
    if distance <= inner:
        return (*LABEL, 255)

    angle = (math.degrees(math.atan2(dy, dx)) + 90.0) % 360.0
    for start, end, colour in SEGMENTS:
        if start <= angle < end:
            # A groove every few pixels, so it reads as a record rather than a pie.
            groove = math.sin(distance * math.pi / max(outer * 0.06, 1.0))
            shade = 1.0 - 0.12 * max(groove, 0.0)
            return (*[int(channel * shade) for channel in colour], 255)  # type: ignore[misc]
    return (*BACKGROUND, 255)


def png_bytes(pixels: List[Pixel], size: int) -> bytes:
    """Encode RGBA pixels as a PNG, without depending on an imaging library."""
    raw = bytearray()
    for y in range(size):
        raw.append(0)  # filter type: none
        for x in range(size):
            raw.extend(pixels[y * size + x])

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def ico_bytes(rendered: dict) -> bytes:
    """Pack PNG-compressed images into a Windows .ico."""
    entries, payloads, offset = b"", b"", 6 + 16 * len(ICO_SIZES)
    for size in ICO_SIZES:
        data = rendered[size]
        entries += struct.pack(
            "<BBBBHHII",
            0 if size >= 256 else size,
            0 if size >= 256 else size,
            0,
            0,
            1,
            32,
            len(data),
            offset,
        )
        payloads += data
        offset += len(data)
    return struct.pack("<HHH", 0, 1, len(ICO_SIZES)) + entries + payloads


def icns_bytes(rendered: dict) -> bytes:
    """Pack PNG images into a macOS .icns."""
    body = b""
    for tag, size in ICNS_SIZES:
        data = rendered[size]
        body += tag.encode("ascii") + struct.pack(">I", len(data) + 8) + data
    return b"icns" + struct.pack(">I", len(body) + 8) + body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ICONS, help="Where to write the icons")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    needed = sorted({*PNG_SIZES, *ICO_SIZES, *(size for _, size in ICNS_SIZES)})

    rendered = {}
    for size in needed:
        rendered[size] = png_bytes(render(size), size)

    for size in PNG_SIZES:
        (args.out / f"{size}x{size}.png").write_bytes(rendered[size])
    # Tauri looks for these exact names.
    (args.out / "128x128@2x.png").write_bytes(rendered[256])
    (args.out / "icon.png").write_bytes(rendered[512])
    (args.out / "icon.ico").write_bytes(ico_bytes(rendered))
    (args.out / "icon.icns").write_bytes(icns_bytes(rendered))

    # Windows Store / MSIX logos, referenced by some bundlers.
    for name, size in (("Square30x30Logo.png", 32), ("Square44x44Logo.png", 48)):
        (args.out / name).write_bytes(rendered[size])

    print(f"Wrote {len(list(args.out.iterdir()))} icon files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
