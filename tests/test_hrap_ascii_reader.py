"""Tests for hrapxmrg.readers.hrap_ascii – HRAPAsciiReader."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hrapxmrg.readers.hrap_ascii import HRAPAsciiReader


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def write_ascii(path: Path, ncols: int, nrows: int,
                xll: float, yll: float, cellsize: float,
                nodata: float, rows: list[list[float]]) -> None:
    """Write a synthetic HRAP ASCII file."""
    with open(path, "w") as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {xll}\n")
        f.write(f"yllcorner     {yll}\n")
        f.write(f"cellsize      {cellsize}\n")
        f.write(f"NODATA_value  {nodata}\n")
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")


SAMPLE_ROWS = [
    [0.0, 0.016, 0.5, 1.2],   # north (row 0 in ASCII)
    [0.0, 0.0,   2.3, 3.1],
    [-999.0, 7.5, 8.9, 0.0],  # south (row 2 in ASCII)
]


# ------------------------------------------------------------------ #
# HRAPAsciiReader tests
# ------------------------------------------------------------------ #

class TestHRAPAsciiReader:
    def test_basic_read(self, tmp_path):
        """Read a small synthetic ASCII file without error."""
        asc = tmp_path / "test.asc"
        write_ascii(asc, ncols=4, nrows=3, xll=341.0, yll=313.0,
                    cellsize=1.0, nodata=-999.0, rows=SAMPLE_ROWS)
        reader = HRAPAsciiReader(asc)
        assert reader.domain.maxx == 4
        assert reader.domain.maxy == 3
        assert reader.domain.xor == 341
        assert reader.domain.yor == 313
        assert reader.nodata == -999.0

    def test_data_shape(self, tmp_path):
        asc = tmp_path / "shape.asc"
        write_ascii(asc, 4, 3, 341, 313, 1, -999, SAMPLE_ROWS)
        reader = HRAPAsciiReader(asc)
        assert reader.data.shape == (3, 4)

    def test_data_north_first_values(self, tmp_path):
        """Row 0 of .data should be the northernmost row from the file."""
        asc = tmp_path / "north.asc"
        write_ascii(asc, 4, 3, 341, 313, 1, -999, SAMPLE_ROWS)
        reader = HRAPAsciiReader(asc)
        # First row of data = first row in file = northernmost
        np.testing.assert_array_equal(reader.data[0], SAMPLE_ROWS[0])

    def test_data_south_first(self, tmp_path):
        """data_south_first() should return rows in reverse order."""
        asc = tmp_path / "south.asc"
        write_ascii(asc, 4, 3, 341, 313, 1, -999, SAMPLE_ROWS)
        reader = HRAPAsciiReader(asc)
        south = reader.data_south_first()
        # Row 0 of south_first should be the last ASCII row (southernmost)
        np.testing.assert_array_equal(south[0], SAMPLE_ROWS[-1])
        np.testing.assert_array_equal(south[-1], SAMPLE_ROWS[0])

    def test_nodata_mask(self, tmp_path):
        asc = tmp_path / "nd.asc"
        write_ascii(asc, 4, 3, 341, 313, 1, -999, SAMPLE_ROWS)
        reader = HRAPAsciiReader(asc)
        mask = reader.nodata_mask()
        # Only SAMPLE_ROWS[2][0] = -999 is nodata
        assert mask.sum() == 1
        assert mask[2, 0]  # last row (south), first column

    def test_stats_excludes_nodata(self, tmp_path):
        asc = tmp_path / "stats.asc"
        write_ascii(asc, 4, 3, 341, 313, 1, -999, SAMPLE_ROWS)
        reader = HRAPAsciiReader(asc)
        stats = reader.stats()
        assert stats["nodata_count"] == 1
        # Valid min should be 0.0 (there are zeros in the data)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(8.9)

    def test_missing_required_header_key_raises(self, tmp_path):
        """Missing ncols in header should raise ValueError."""
        asc = tmp_path / "bad_hdr.asc"
        with open(asc, "w") as f:
            f.write("nrows 3\n")
            f.write("xllcorner 341\n")
            f.write("yllcorner 313\n")
            f.write("cellsize 1\n")
            f.write("1 2 3 4\n1 2 3 4\n1 2 3 4\n")
        with pytest.raises(ValueError, match="ncols"):
            HRAPAsciiReader(asc)

    def test_wrong_row_count_raises(self, tmp_path):
        """Too few data rows should raise ValueError."""
        asc = tmp_path / "bad_rows.asc"
        write_ascii(asc, 4, 3, 341, 313, 1, -999,
                    [[1, 2, 3, 4], [5, 6, 7, 8]])  # only 2 rows, expected 3
        with pytest.raises(ValueError, match="data rows"):
            HRAPAsciiReader(asc)

    def test_wrong_col_count_raises(self, tmp_path):
        """Rows with wrong column count should raise ValueError."""
        asc = tmp_path / "bad_cols.asc"
        write_ascii(asc, 4, 3, 341, 313, 1, -999,
                    [[1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4]])  # row 1 has 3 cols
        with pytest.raises(ValueError, match="values"):
            HRAPAsciiReader(asc)

    def test_default_domain_dimensions(self, tmp_path):
        """Default Upper Rio Grande domain: 64 cols × 87 rows."""
        # Build a 64×87 grid filled with zeros plus one nodata cell
        rows = [[0.0] * 64 for _ in range(87)]
        asc = tmp_path / "default_domain.asc"
        write_ascii(asc, 64, 87, 341, 313, 1, -999, rows)
        reader = HRAPAsciiReader(asc)
        from hrapxmrg.config import DEFAULT_DOMAIN
        assert reader.domain.matches(DEFAULT_DOMAIN)

    def test_ncols_nrows_properties(self, tmp_path):
        asc = tmp_path / "props.asc"
        write_ascii(asc, 4, 3, 341, 313, 1, -999, SAMPLE_ROWS)
        reader = HRAPAsciiReader(asc)
        assert reader.ncols == 4
        assert reader.nrows == 3

    def test_case_insensitive_header(self, tmp_path):
        """Header keys should be parsed case-insensitively."""
        asc = tmp_path / "case.asc"
        with open(asc, "w") as f:
            f.write("NCOLS 4\nNROWS 3\n")
            f.write("XLLCORNER 341\nYLLCORNER 313\nCELLSIZE 1\n")
            f.write("NODATA_VALUE -999\n")
            for row in SAMPLE_ROWS:
                f.write(" ".join(str(v) for v in row) + "\n")
        reader = HRAPAsciiReader(asc)
        assert reader.ncols == 4
