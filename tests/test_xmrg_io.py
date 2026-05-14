"""Tests for hrapxmrg.xmrg_io – XMRGReader and XMRGWriter."""

import gzip
import io
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hrapxmrg.config import HRAPDomain, DEFAULT_DOMAIN
from hrapxmrg.xmrg_io import XMRGReader, XMRGWriter


# ------------------------------------------------------------------ #
# Helpers to build synthetic XMRG files in memory
# ------------------------------------------------------------------ #

def _write_fortran_record(buf: io.BytesIO, payload: bytes) -> None:
    length = len(payload)
    marker = struct.pack("<I", length)
    buf.write(marker)
    buf.write(payload)
    buf.write(marker)


def make_xmrg_bytes(
    xor: int, yor: int, maxx: int, maxy: int,
    data: np.ndarray,  # shape (maxy, maxx), south-first
    dtype=np.int16,
) -> bytes:
    """Build raw (uncompressed) XMRG bytes."""
    buf = io.BytesIO()
    # Header record: 4×int16 + 8 null bytes
    hdr = struct.pack("<4h", xor, yor, maxx, maxy) + b"\x00" * 8
    _write_fortran_record(buf, hdr)
    # Data records (south-first)
    arr = np.asarray(data, dtype=dtype)
    for row_idx in range(maxy):
        _write_fortran_record(buf, arr[row_idx, :].tobytes())
    return buf.getvalue()


def make_xmrg_gz(
    xor: int, yor: int, maxx: int, maxy: int,
    data: np.ndarray,
    dtype=np.int16,
    compress_level: int = 6,
) -> bytes:
    """Build gzip-compressed XMRG bytes."""
    raw = make_xmrg_bytes(xor, yor, maxx, maxy, data, dtype)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=compress_level) as gz:
        gz.write(raw)
    return out.getvalue()


# ------------------------------------------------------------------ #
# XMRGWriter tests
# ------------------------------------------------------------------ #

class TestXMRGWriter:
    def _make_domain(self, maxx=4, maxy=3, xor=341, yor=313):
        return HRAPDomain(xor=xor, yor=yor, maxx=maxx, maxy=maxy)

    def test_write_creates_gz_file(self, tmp_path):
        domain = self._make_domain()
        data = np.zeros((domain.maxy, domain.maxx), dtype=np.int16)
        writer = XMRGWriter(domain)
        out = tmp_path / "test.gz"
        writer.write(data, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_write_read_roundtrip_int16(self, tmp_path):
        """Write and read back int16 data; values should be identical."""
        domain = self._make_domain(maxx=8, maxy=5)
        rng = np.random.default_rng(42)
        data = rng.integers(-1000, 10000, size=(domain.maxy, domain.maxx),
                            dtype=np.int16)
        writer = XMRGWriter(domain, storage_dtype=np.int16, rows_north_first=False)
        out = tmp_path / "roundtrip.gz"
        writer.write(data, out)

        reader = XMRGReader(out)
        assert reader.domain.xor == domain.xor
        assert reader.domain.yor == domain.yor
        assert reader.domain.maxx == domain.maxx
        assert reader.domain.maxy == domain.maxy
        np.testing.assert_array_equal(reader.data, data)

    def test_write_read_roundtrip_float32(self, tmp_path):
        """Write and read back float32 data."""
        domain = self._make_domain(maxx=6, maxy=4)
        data = np.arange(24, dtype=np.float32).reshape(4, 6) * 0.5
        writer = XMRGWriter(domain, storage_dtype=np.float32, rows_north_first=False)
        out = tmp_path / "roundtrip_f32.gz"
        writer.write(data, out)

        reader = XMRGReader(out)
        np.testing.assert_array_equal(reader.data, data)
        assert reader.dtype == np.dtype("<f4")

    def test_north_first_flip(self, tmp_path):
        """Writing north-first should flip rows to south-first in the file."""
        domain = self._make_domain(maxx=3, maxy=2)
        # Row 0 = north = [1, 2, 3], Row 1 = south = [4, 5, 6]
        data_north_first = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
        writer = XMRGWriter(domain, rows_north_first=True)
        out = tmp_path / "flip_test.gz"
        writer.write(data_north_first, out)

        reader = XMRGReader(out)
        # In the file, row 0 = southernmost = [4, 5, 6]
        np.testing.assert_array_equal(reader.data[0], [4, 5, 6])
        np.testing.assert_array_equal(reader.data[1], [1, 2, 3])
        # data_north_first() should restore the original order
        restored = reader.data_north_first()
        np.testing.assert_array_equal(restored, data_north_first)

    def test_wrong_shape_raises(self, tmp_path):
        """Passing wrong-shape data should raise ValueError."""
        domain = self._make_domain(maxx=4, maxy=3)
        bad_data = np.zeros((5, 4), dtype=np.int16)
        writer = XMRGWriter(domain)
        out = tmp_path / "bad.gz"
        with pytest.raises(ValueError, match="shape"):
            writer.write(bad_data, out)

    def test_invalid_dtype_raises(self):
        """Passing an unsupported dtype should raise ValueError."""
        domain = self._make_domain()
        with pytest.raises(ValueError, match="storage_dtype"):
            XMRGWriter(domain, storage_dtype=np.float64)

    def test_domain_stored_in_header(self, tmp_path):
        """XOR/YOR/MAXX/MAXY in the file header must match the domain."""
        domain = HRAPDomain(xor=341, yor=313, maxx=64, maxy=87)
        data = np.zeros((87, 64), dtype=np.int16)
        writer = XMRGWriter(domain)
        out = tmp_path / "hdr_test.gz"
        writer.write(data, out)
        reader = XMRGReader(out)
        assert reader.domain.xor == 341
        assert reader.domain.yor == 313
        assert reader.domain.maxx == 64
        assert reader.domain.maxy == 87


# ------------------------------------------------------------------ #
# XMRGReader tests
# ------------------------------------------------------------------ #

class TestXMRGReader:
    def _write_gz(self, tmp_path: Path, content: bytes) -> Path:
        p = tmp_path / "test.gz"
        p.write_bytes(content)
        return p

    def test_read_synthetic_int16(self, tmp_path):
        """Reader correctly parses a synthetic int16 XMRG."""
        maxx, maxy = 4, 3
        # South-first: row 0 = [0,1,2,3], row 1 = [4,5,6,7], row 2 = [8,9,10,11]
        data = np.arange(12, dtype=np.int16).reshape(maxy, maxx)
        gz_bytes = make_xmrg_gz(341, 313, maxx, maxy, data)
        path = self._write_gz(tmp_path, gz_bytes)

        reader = XMRGReader(path)
        assert reader.domain.xor == 341
        assert reader.domain.yor == 313
        assert reader.domain.maxx == maxx
        assert reader.domain.maxy == maxy
        np.testing.assert_array_equal(reader.data, data)
        assert reader.dtype == np.dtype("<i2")

    def test_read_synthetic_float32(self, tmp_path):
        """Reader correctly parses a synthetic float32 XMRG."""
        maxx, maxy = 5, 2
        data = np.linspace(0.0, 10.0, maxx * maxy, dtype=np.float32).reshape(maxy, maxx)
        gz_bytes = make_xmrg_gz(100, 200, maxx, maxy, data, dtype=np.float32)
        path = self._write_gz(tmp_path, gz_bytes)

        reader = XMRGReader(path)
        np.testing.assert_allclose(reader.data, data)
        assert reader.dtype == np.dtype("<f4")

    def test_data_north_first(self, tmp_path):
        """data_north_first() should reverse the row order."""
        maxx, maxy = 3, 2
        data = np.array([[10, 11, 12], [20, 21, 22]], dtype=np.int16)
        gz_bytes = make_xmrg_gz(0, 0, maxx, maxy, data)
        path = self._write_gz(tmp_path, gz_bytes)

        reader = XMRGReader(path)
        north_first = reader.data_north_first()
        # Original row 0 (south) → north_first[-1]; row 1 (north) → north_first[0]
        np.testing.assert_array_equal(north_first[0], [20, 21, 22])
        np.testing.assert_array_equal(north_first[1], [10, 11, 12])

    def test_stats_with_nodata(self, tmp_path):
        """stats() correctly excludes nodata cells."""
        maxx, maxy = 4, 1
        data = np.array([[-9999, 100, 200, 300]], dtype=np.int16)
        gz_bytes = make_xmrg_gz(0, 0, maxx, maxy, data)
        path = self._write_gz(tmp_path, gz_bytes)

        reader = XMRGReader(path)
        stats = reader.stats(nodata=-9999)
        assert stats["nodata_count"] == 1
        assert stats["min"] == pytest.approx(100.0)
        assert stats["max"] == pytest.approx(300.0)
        assert stats["mean"] == pytest.approx(200.0)

    def test_stats_no_nodata(self, tmp_path):
        """stats() with no nodata=None includes all cells."""
        maxx, maxy = 2, 2
        data = np.array([[1, 2], [3, 4]], dtype=np.int16)
        gz_bytes = make_xmrg_gz(0, 0, maxx, maxy, data)
        path = self._write_gz(tmp_path, gz_bytes)

        reader = XMRGReader(path)
        stats = reader.stats()
        assert stats["count"] == 4
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(4.0)
        assert stats["mean"] == pytest.approx(2.5)

    def test_truncated_file_raises(self, tmp_path):
        """Truncated XMRG content should raise ValueError."""
        truncated = b"\x10\x00\x00\x00"  # partial header marker, no payload
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=out, mode="wb") as gz:
            gz.write(truncated)
        path = tmp_path / "bad.gz"
        path.write_bytes(out.getvalue())

        with pytest.raises((ValueError, EOFError, Exception)):
            XMRGReader(path)
