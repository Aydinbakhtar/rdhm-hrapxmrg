"""XMRG reader/writer.

The writer supports the conservative format needed for the user's current
RDHM forcing workflow:

- gzip-compressed binary
- Fortran-unformatted record markers
- primary header with either int32 or float32 convention
- row-by-row int16 or float32 payloads
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gzip
import struct

import numpy as np


@dataclass(frozen=True)
class XMRGMeta:
    path: str
    endian: str
    header_type: str
    dtype: str
    xor: float
    yor: float
    maxx: int
    maxy: int
    secondary_headers_hex: tuple[str, ...]


def _wrap_record(payload: bytes, endian: str = "<") -> bytes:
    n = len(payload)
    return struct.pack(f"{endian}i", n) + payload + struct.pack(f"{endian}i", n)

def _missing_mask(arr: np.ndarray, missing: float) -> np.ndarray:
    """Return True where arr should be written as the missing-value sentinel."""
    if np.isnan(missing):
        return np.isnan(arr)
    return np.isnan(arr) | np.isclose(arr, missing)


def read_xmrg(path: str | Path) -> tuple[np.ndarray, XMRGMeta]:
    """Read an XMRG .gz file and return (grid, metadata).

    The grid is returned as stored numeric values. For scaled int16 temperature,
    divide by the variable-specific RDHM divisor outside this function.
    """
    path = Path(path)
    with gzip.open(path, "rb") as f:
        b = f.read()

    pos = 0
    endian = None
    n = None
    hdr = None

    for candidate in ("<", ">"):
        try:
            n0 = struct.unpack(f"{candidate}i", b[pos : pos + 4])[0]
            if 0 < n0 < 4096:
                payload = b[pos + 4 : pos + 4 + n0]
                end = struct.unpack(f"{candidate}i", b[pos + 4 + n0 : pos + 8 + n0])[0]
                if end == n0:
                    endian = candidate
                    n = n0
                    hdr = payload
                    break
        except Exception:
            continue

    if endian is None or n is None or hdr is None:
        raise ValueError(f"{path}: could not detect XMRG Fortran record structure")

    pos += 8 + n

    if len(hdr) < 16:
        raise ValueError(f"{path}: primary header too short: {len(hdr)}")

    i_xor, i_yor, i_mx, i_my = struct.unpack(f"{endian}iiii", hdr[:16])
    f_xor, f_yor, f_mx, f_my = struct.unpack(f"{endian}ffii", hdr[:16])

    if (
        0 < i_mx < 100000
        and 0 < i_my < 100000
        and -10000 < i_xor < 10000
        and -10000 < i_yor < 10000
    ):
        xor, yor, mx, my, htype = float(i_xor), float(i_yor), int(i_mx), int(i_my), "int32"
    elif (
        0 < f_mx < 100000
        and 0 < f_my < 100000
        and -10000 < f_xor < 10000
        and -10000 < f_yor < 10000
    ):
        xor, yor, mx, my, htype = float(f_xor), float(f_yor), int(f_mx), int(f_my), "float32"
    else:
        raise ValueError(f"{path}: implausible XMRG primary header")

    secondary_headers: list[bytes] = []

    while True:
        if pos + 4 > len(b):
            raise ValueError(f"{path}: reached EOF before data rows")
        nrow_or_sec = struct.unpack(f"{endian}i", b[pos : pos + 4])[0]
        if nrow_or_sec in (2 * mx, 4 * mx):
            break
        payload = b[pos + 4 : pos + 4 + nrow_or_sec]
        end = struct.unpack(f"{endian}i", b[pos + 4 + nrow_or_sec : pos + 8 + nrow_or_sec])[0]
        if end != nrow_or_sec:
            raise ValueError(f"{path}: bad secondary-header record marker")
        secondary_headers.append(payload)
        pos += 8 + nrow_or_sec

    dtype = np.dtype(f"{endian}i2") if nrow_or_sec == 2 * mx else np.dtype(f"{endian}f4")
    dtype_name = "int16" if nrow_or_sec == 2 * mx else "float32"

    grid = np.empty((my, mx), dtype=np.float64)
    for j in range(my):
        nrow = struct.unpack(f"{endian}i", b[pos : pos + 4])[0]
        if nrow not in (2 * mx, 4 * mx):
            raise ValueError(f"{path}: row {j} has unexpected payload length {nrow}")
        payload = b[pos + 4 : pos + 4 + nrow]
        end = struct.unpack(f"{endian}i", b[pos + 4 + nrow : pos + 8 + nrow])[0]
        if end != nrow:
            raise ValueError(f"{path}: row {j} has mismatched record marker")
        grid[j, :] = np.frombuffer(payload, dtype=dtype, count=mx)
        pos += 8 + nrow

    return grid, XMRGMeta(
        path=str(path),
        endian="little" if endian == "<" else "big",
        header_type=htype,
        dtype=dtype_name,
        xor=xor,
        yor=yor,
        maxx=mx,
        maxy=my,
        secondary_headers_hex=tuple(x.hex() for x in secondary_headers),
    )


def write_xmrg(
    path: str | Path,
    grid: np.ndarray,
    xor: int | float,
    yor: int | float,
    *,
    header_type: str = "int32",
    dtype: str = "int16",
    scale: float = 1.0,
    missing: float = -999.0,
    secondary_header: bool = False,
    orientation: str = "as-is",
) -> None:
    """Write an XMRG .gz file.

    Parameters
    ----------
    grid:
        2D array in physical units unless already scaled and scale=1.
    scale:
        Multiplier applied when dtype='int16'. For current project temperature,
        use scale=100 when grid is Fahrenheit.
    orientation:
        'as-is' keeps row order. 'flipud' flips north/south before writing.
    """
    path = Path(path)

    if grid.ndim != 2:
        raise ValueError("grid must be 2D")

    arr = np.array(grid, copy=True)

    if orientation == "flipud":
        arr = np.flipud(arr)
    elif orientation != "as-is":
        raise ValueError("orientation must be 'as-is' or 'flipud'")

    my, mx = arr.shape

    if header_type == "int32":
        hdr = struct.pack("<iiii", int(round(xor)), int(round(yor)), mx, my)
    elif header_type == "float32":
        hdr = struct.pack("<ffii", float(xor), float(yor), mx, my)
    else:
        raise ValueError("header_type must be 'int32' or 'float32'")

    buf = bytearray()
    buf += _wrap_record(hdr, "<")

    if secondary_header:
        flag = 2 if dtype == "int16" else 4
        sec = struct.pack("<10siff", b"POSTGRIDS ", flag, 1.0, float(missing))
        buf += _wrap_record(sec, "<")

    if dtype == "int16":
        # scaled = np.where(arr <= -900, missing, np.rint(arr * scale))
        scaled = np.where(_missing_mask(arr, missing), missing, np.rint(arr * scale))
        if np.nanmin(scaled) < np.iinfo(np.int16).min or np.nanmax(scaled) > np.iinfo(np.int16).max:
            raise OverflowError("scaled grid exceeds int16 range")
        stored = scaled.astype("<i2")
        for j in range(my):
            buf += _wrap_record(stored[j].tobytes(), "<")
    elif dtype == "float32":
        stored = arr.astype("<f4")
        for j in range(my):
            buf += _wrap_record(stored[j].tobytes(), "<")
    else:
        raise ValueError("dtype must be 'int16' or 'float32'")

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(bytes(buf))
