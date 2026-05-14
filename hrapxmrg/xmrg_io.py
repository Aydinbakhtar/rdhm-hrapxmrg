"""XMRG binary file reader and writer.

XMRG is the binary forcing-file format used by RDHM/HL-RDHM.  Files are
gzip-compressed Fortran-unformatted binary streams.

Record structure
----------------
Fortran unformatted sequential records have a 4-byte integer record-length
marker **before** and **after** each data payload.  The marker value is the
number of bytes in the payload.

Record 1 – header (always present):
    Payload: XOR (int16), YOR (int16), MAXX (int16), MAXY (int16)
    followed by an 8-byte operating-system / user ID field (often zeros).
    Total payload: 16 bytes.  Record markers: ``00 00 00 10`` (= 16).

Record 2 – extended header (optional, version-dependent):
    If present the marker will be non-zero.  This implementation detects
    whether record 2 is a date-time header or the first data row by
    inspecting the record length.

Data records – one per row, south-to-north:
    Each record contains ``MAXX`` values in either int16 or float32
    format.  The record-length marker determines the element count and
    inferred dtype.

Row orientation
---------------
XMRG stores rows from the *southernmost* row (index 0) to the
*northernmost* row (index MAXY-1).  ESRI ASCII grids store the
*northernmost* row first.  The writer therefore flips the input array
unless explicitly instructed otherwise.
"""

from __future__ import annotations

import gzip
import io
import struct
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import HRAPDomain

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

#: Bytes in a Fortran record-length marker.
_MARKER_SIZE: int = 4

#: Header payload size: 4 × int16 (XOR, YOR, MAXX, MAXY) + 8-byte user ID.
_HEADER1_PAYLOAD: int = 16

#: Extended header payload size (date/time record used in some versions).
_HEADER2_PAYLOAD: int = 66

#: Number of int32 geographic values in header 1 (XOR, YOR, MAXX, MAXY).
_HEADER1_INT16_COUNT: int = 4


# ------------------------------------------------------------------ #
# Low-level I/O helpers
# ------------------------------------------------------------------ #

def _read_fortran_record(f: BinaryIO) -> bytes:
    """Read one Fortran unformatted sequential record.

    Returns the raw payload bytes (without the surrounding markers).
    Raises :class:`EOFError` if the stream is exhausted.
    Raises :class:`ValueError` if the trailing marker does not match.
    """
    marker_bytes = f.read(_MARKER_SIZE)
    if not marker_bytes:
        raise EOFError("End of stream reached while reading record marker.")
    if len(marker_bytes) < _MARKER_SIZE:
        raise ValueError(
            f"Truncated record marker: expected {_MARKER_SIZE} bytes, "
            f"got {len(marker_bytes)}."
        )
    (length,) = struct.unpack("<I", marker_bytes)

    payload = f.read(length)
    if len(payload) < length:
        raise ValueError(
            f"Truncated record payload: expected {length} bytes, "
            f"got {len(payload)}."
        )

    trailer_bytes = f.read(_MARKER_SIZE)
    if len(trailer_bytes) < _MARKER_SIZE:
        raise ValueError("Missing record trailer marker.")
    (trailer,) = struct.unpack("<I", trailer_bytes)
    if trailer != length:
        raise ValueError(
            f"Record marker mismatch: leading={length}, trailing={trailer}."
        )

    return payload


def _write_fortran_record(f: BinaryIO, payload: bytes) -> None:
    """Write *payload* as a Fortran unformatted sequential record."""
    length = len(payload)
    marker = struct.pack("<I", length)
    f.write(marker)
    f.write(payload)
    f.write(marker)


# ------------------------------------------------------------------ #
# Public reader
# ------------------------------------------------------------------ #

class XMRGReader:
    """Read a gzip-compressed XMRG file.

    Parameters
    ----------
    path:
        Path to the ``.gz`` XMRG file.

    Attributes
    ----------
    domain:
        Parsed :class:`~hrapxmrg.config.HRAPDomain` from the header.
    data:
        2-D NumPy array of stored values.  Shape ``(nrows, ncols)`` where
        row 0 is the *southernmost* row (as stored in the file).
    dtype:
        NumPy dtype of :attr:`data` (``numpy.int16`` or ``numpy.float32``).
    has_extended_header:
        Whether a second (date/time) header record was detected.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self.domain: Optional[HRAPDomain] = None
        self.data: Optional[NDArray] = None
        self.dtype: Optional[np.dtype] = None
        self.has_extended_header: bool = False
        self._read()

    # -------------------------------------------------------------- #

    def _read(self) -> None:
        with gzip.open(self._path, "rb") as gz:
            raw = gz.read()

        buf = io.BytesIO(raw)

        # --- Record 1: header ---
        hdr = _read_fortran_record(buf)
        if len(hdr) < 8:
            raise ValueError(
                f"Header record too short: {len(hdr)} bytes (expected ≥ 8)."
            )
        xor, yor, maxx, maxy = struct.unpack_from("<4h", hdr, 0)
        self.domain = HRAPDomain(xor=int(xor), yor=int(yor),
                                 maxx=int(maxx), maxy=int(maxy))

        # --- Optional record 2: extended header ---
        # Peek at the next marker to see how many bytes are in the next record.
        peek = buf.read(_MARKER_SIZE)
        if len(peek) < _MARKER_SIZE:
            raise ValueError("File ended before any data records.")
        (next_len,) = struct.unpack("<I", peek)
        buf.seek(-_MARKER_SIZE, 1)  # rewind the peek

        expected_row_int16 = maxx * 2      # bytes in one int16 row
        expected_row_float32 = maxx * 4    # bytes in one float32 row

        if next_len == _HEADER2_PAYLOAD:
            # Extended date/time header present; consume it.
            _read_fortran_record(buf)
            self.has_extended_header = True

        # --- Data records: one per row ---
        rows: list[NDArray] = []
        inferred_dtype: Optional[np.dtype] = None

        for row_idx in range(maxy):
            try:
                row_payload = _read_fortran_record(buf)
            except EOFError:
                raise ValueError(
                    f"File ended after {row_idx} rows; expected {maxy}."
                )

            n_bytes = len(row_payload)
            if n_bytes == expected_row_int16:
                row_dtype = np.dtype("<i2")  # int16 little-endian
            elif n_bytes == expected_row_float32:
                row_dtype = np.dtype("<f4")  # float32 little-endian
            else:
                raise ValueError(
                    f"Row {row_idx}: unexpected payload length {n_bytes} bytes "
                    f"(expected {expected_row_int16} for int16 or "
                    f"{expected_row_float32} for float32, maxx={maxx})."
                )

            if inferred_dtype is None:
                inferred_dtype = row_dtype
            elif inferred_dtype != row_dtype:
                raise ValueError(
                    f"Row {row_idx} dtype changed from {inferred_dtype} to "
                    f"{row_dtype}; mixed-type XMRG not supported."
                )

            row_arr = np.frombuffer(row_payload, dtype=row_dtype).copy()
            rows.append(row_arr)

        self.data = np.stack(rows, axis=0)  # shape: (maxy, maxx)
        self.dtype = inferred_dtype

    # -------------------------------------------------------------- #

    @property
    def ncols(self) -> int:
        """Number of columns."""
        return self.domain.maxx

    @property
    def nrows(self) -> int:
        """Number of rows."""
        return self.domain.maxy

    def data_north_first(self) -> NDArray:
        """Return :attr:`data` with the northernmost row first.

        XMRG stores south-first; this flips the array so that index 0
        corresponds to the northernmost row (matching ESRI ASCII convention).
        """
        return self.data[::-1, :]

    def stats(self, nodata: Optional[float] = None) -> dict:
        """Return basic statistics of the stored values.

        Parameters
        ----------
        nodata:
            If given, cells equal to *nodata* are excluded from statistics.

        Returns
        -------
        dict with keys ``min``, ``max``, ``mean``, ``nodata_count``, ``count``.
        """
        arr = self.data.astype(np.float64)
        mask = np.zeros_like(arr, dtype=bool)
        if nodata is not None:
            mask = arr == nodata

        valid = arr[~mask]
        return {
            "min": float(valid.min()) if valid.size else float("nan"),
            "max": float(valid.max()) if valid.size else float("nan"),
            "mean": float(valid.mean()) if valid.size else float("nan"),
            "nodata_count": int(mask.sum()),
            "count": int(arr.size),
        }


# ------------------------------------------------------------------ #
# Public writer
# ------------------------------------------------------------------ #

class XMRGWriter:
    """Write a gzip-compressed XMRG file.

    Parameters
    ----------
    domain:
        :class:`~hrapxmrg.config.HRAPDomain` describing the output grid.
    storage_dtype:
        NumPy dtype for data values.  Either ``numpy.int16`` (default,
        for temperature and scaled precipitation) or ``numpy.float32``.
    rows_north_first:
        If ``True`` (default), the input array passed to :meth:`write` is
        assumed to have the *northernmost* row at index 0 (ESRI ASCII
        convention) and will be flipped before writing.  Set to ``False``
        if the input is already in south-first XMRG order.
    compress_level:
        gzip compression level (0–9; default 6).

    Examples
    --------
    Write a small test XMRG::

        import numpy as np
        from hrapxmrg import XMRGWriter, DEFAULT_DOMAIN

        data = np.zeros((DEFAULT_DOMAIN.maxy, DEFAULT_DOMAIN.maxx), dtype=np.int16)
        writer = XMRGWriter(DEFAULT_DOMAIN)
        writer.write(data, "output.gz")
    """

    def __init__(
        self,
        domain: HRAPDomain,
        storage_dtype: np.dtype | type = np.int16,
        rows_north_first: bool = True,
        compress_level: int = 6,
    ) -> None:
        self.domain = domain
        self.storage_dtype = np.dtype(storage_dtype)
        if self.storage_dtype not in (np.dtype(np.int16), np.dtype(np.float32)):
            raise ValueError(
                f"storage_dtype must be int16 or float32, got {self.storage_dtype}."
            )
        self.rows_north_first = rows_north_first
        self.compress_level = compress_level

    # -------------------------------------------------------------- #

    def _build_header1(self) -> bytes:
        """Build the 16-byte header record 1 payload."""
        d = self.domain
        header_ints = struct.pack(
            "<4h",
            d.xor, d.yor, d.maxx, d.maxy,
        )
        user_id = b"\x00" * 8  # 8-byte OS/user ID field (zeros)
        return header_ints + user_id  # 8 + 8 = 16 bytes

    def write(
        self,
        data: NDArray,
        path: str | Path,
    ) -> None:
        """Write *data* as a gzip-compressed XMRG file.

        Parameters
        ----------
        data:
            2-D NumPy array of shape ``(nrows, ncols)`` where ``nrows``
            and ``ncols`` match :attr:`domain`.  If :attr:`rows_north_first`
            is ``True`` the array is expected in north-first order and will
            be flipped internally.
        path:
            Output file path.  Conventionally ends in ``.gz``.

        Raises
        ------
        ValueError
            If *data* shape does not match the domain dimensions.
        """
        d = self.domain
        expected_shape = (d.maxy, d.maxx)
        if data.shape != expected_shape:
            raise ValueError(
                f"data.shape {data.shape} does not match domain shape "
                f"{expected_shape} (nrows={d.maxy}, ncols={d.maxx})."
            )

        # Ensure correct dtype
        arr = np.asarray(data, dtype=self.storage_dtype)

        # Flip to south-first XMRG order if input is north-first
        if self.rows_north_first:
            arr = arr[::-1, :]

        buf = io.BytesIO()

        # --- Record 1 ---
        _write_fortran_record(buf, self._build_header1())

        # --- Data records (one per row, south to north) ---
        for row_idx in range(d.maxy):
            row_bytes = arr[row_idx, :].tobytes()
            _write_fortran_record(buf, row_bytes)

        raw = buf.getvalue()

        # --- gzip compress ---
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb", compresslevel=self.compress_level) as gz:
            gz.write(raw)
