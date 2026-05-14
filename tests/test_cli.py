"""Tests for hrapxmrg CLI commands."""

import gzip
import io
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from hrapxmrg.cli import main
from hrapxmrg.config import DEFAULT_DOMAIN, HRAPDomain
from hrapxmrg.xmrg_io import XMRGReader, XMRGWriter


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def write_hrap_ascii(path: Path, ncols: int = 4, nrows: int = 3,
                     xll: float = 341, yll: float = 313,
                     cellsize: float = 1, nodata: float = -999,
                     rows=None) -> None:
    if rows is None:
        rows = [[float(i * ncols + j) for j in range(ncols)] for i in range(nrows)]
    with open(path, "w") as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {xll}\n")
        f.write(f"yllcorner     {yll}\n")
        f.write(f"cellsize      {cellsize}\n")
        f.write(f"NODATA_value  {nodata}\n")
        for row in rows:
            f.write(" ".join(str(v) for v in row) + "\n")


def make_xmrg_gz(path: Path, domain: HRAPDomain, fill: int = 5000) -> None:
    data = np.full((domain.maxy, domain.maxx), fill_value=fill, dtype=np.int16)
    writer = XMRGWriter(domain, rows_north_first=False)
    writer.write(data, path)


@pytest.fixture
def runner():
    return CliRunner()


# ------------------------------------------------------------------ #
# inspect command
# ------------------------------------------------------------------ #

class TestInspectCommand:
    def test_inspect_basic(self, runner, tmp_path):
        xmrg = tmp_path / "test.gz"
        domain = HRAPDomain(xor=341, yor=313, maxx=4, maxy=3)
        make_xmrg_gz(xmrg, domain, fill=5000)
        result = runner.invoke(main, ["inspect", str(xmrg)])
        assert result.exit_code == 0
        assert "341" in result.output
        assert "313" in result.output

    def test_inspect_with_nodata(self, runner, tmp_path):
        domain = HRAPDomain(xor=341, yor=313, maxx=4, maxy=3)
        data = np.full((3, 4), 5000, dtype=np.int16)
        data[0, 0] = -9999
        xmrg = tmp_path / "nd.gz"
        writer = XMRGWriter(domain, rows_north_first=False)
        writer.write(data, xmrg)
        result = runner.invoke(main, ["inspect", str(xmrg), "--nodata", "-9999"])
        assert result.exit_code == 0
        assert "1" in result.output  # nodata count

    def test_inspect_variable_reporting(self, runner, tmp_path):
        """--variable tair should print physical °F values."""
        domain = HRAPDomain(xor=341, yor=313, maxx=4, maxy=3)
        make_xmrg_gz(tmp_path / "tair.gz", domain, fill=3200)
        result = runner.invoke(main, [
            "inspect", str(tmp_path / "tair.gz"), "--variable", "tair"
        ])
        assert result.exit_code == 0
        assert "°F" in result.output or "F" in result.output

    def test_inspect_missing_file(self, runner, tmp_path):
        result = runner.invoke(main, ["inspect", str(tmp_path / "ghost.gz")])
        assert result.exit_code != 0

    def test_inspect_shows_dtype(self, runner, tmp_path):
        domain = HRAPDomain(xor=0, yor=0, maxx=2, maxy=2)
        make_xmrg_gz(tmp_path / "dtype.gz", domain)
        result = runner.invoke(main, ["inspect", str(tmp_path / "dtype.gz")])
        assert result.exit_code == 0
        assert "int16" in result.output


# ------------------------------------------------------------------ #
# ascii-to-xmrg command
# ------------------------------------------------------------------ #

class TestAsciiToXmrgCommand:
    def test_basic_conversion(self, runner, tmp_path):
        asc = tmp_path / "input.asc"
        out = tmp_path / "output.gz"
        write_hrap_ascii(asc)
        result = runner.invoke(main, [
            "ascii-to-xmrg", str(asc), str(out)
        ])
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_output_is_readable_xmrg(self, runner, tmp_path):
        asc = tmp_path / "readable.asc"
        out = tmp_path / "readable.gz"
        rows = [[10.0, 20.0, 30.0, 40.0]] * 3
        write_hrap_ascii(asc, rows=rows)
        runner.invoke(main, ["ascii-to-xmrg", str(asc), str(out)])
        reader = XMRGReader(out)
        assert reader.domain.xor == 341
        assert reader.domain.maxx == 4

    def test_temperature_conversion_f100(self, runner, tmp_path):
        """tair conversion: 0 °C should become 3200 stored."""
        asc = tmp_path / "celsius.asc"
        out = tmp_path / "f100.gz"
        rows = [[0.0, 0.0, 0.0, 0.0]] * 3
        write_hrap_ascii(asc, rows=rows)
        runner.invoke(main, [
            "ascii-to-xmrg", str(asc), str(out), "--variable", "tair"
        ])
        reader = XMRGReader(out)
        # All cells should be 3200 (0°C → 32°F → 3200 stored)
        valid = reader.data[reader.data != -9999]
        assert np.all(valid == 3200), f"Expected 3200, got unique={np.unique(valid)}"

    def test_not_fahrenheit_times_10(self, runner, tmp_path):
        """Ensure F×10 is NOT used: 0°C stored should be 3200, not 320."""
        asc = tmp_path / "not_f10.asc"
        out = tmp_path / "not_f10.gz"
        rows = [[0.0] * 4] * 3
        write_hrap_ascii(asc, rows=rows)
        runner.invoke(main, [
            "ascii-to-xmrg", str(asc), str(out), "--variable", "tair"
        ])
        reader = XMRGReader(out)
        valid = reader.data[reader.data != -9999]
        assert not np.any(valid == 320), "Must NOT use F×10 convention"
        assert np.all(valid == 3200), "Must use F×100 convention"

    def test_passthrough_no_variable(self, runner, tmp_path):
        """Without --variable, values are passed through (rounded to int16)."""
        asc = tmp_path / "raw.asc"
        out = tmp_path / "raw.gz"
        rows = [[5.6, 10.3, 0.0, -999.0]] + [[0.0] * 4] * 2
        write_hrap_ascii(asc, rows=rows)
        runner.invoke(main, ["ascii-to-xmrg", str(asc), str(out)])
        reader = XMRGReader(out)
        assert reader.domain.maxx == 4


# ------------------------------------------------------------------ #
# validate command
# ------------------------------------------------------------------ #

class TestValidateCommand:
    def test_validate_valid_file(self, runner, tmp_path):
        xmrg = tmp_path / "valid.gz"
        make_xmrg_gz(xmrg, DEFAULT_DOMAIN, fill=5000)
        result = runner.invoke(main, [
            "validate", str(xmrg),
            "--variable", "tair",
            "--xor", "341", "--yor", "313",
            "--maxx", "64", "--maxy", "87",
        ])
        assert result.exit_code == 0, result.output
        assert "PASSED" in result.output

    def test_validate_wrong_domain(self, runner, tmp_path):
        wrong = HRAPDomain(xor=100, yor=200, maxx=10, maxy=10)
        xmrg = tmp_path / "wrong.gz"
        make_xmrg_gz(xmrg, wrong, fill=5000)
        result = runner.invoke(main, [
            "validate", str(xmrg),
            "--xor", "341", "--yor", "313", "--maxx", "64", "--maxy", "87",
        ])
        assert result.exit_code != 0
        assert "FAILED" in result.output

    def test_validate_missing_file(self, runner, tmp_path):
        result = runner.invoke(main, [
            "validate", str(tmp_path / "ghost.gz")
        ])
        assert result.exit_code != 0

    def test_validate_check_filename(self, runner, tmp_path):
        """--check-filename should verify the filename naming convention."""
        # Wrong filename (no prefix) for tair
        xmrg = tmp_path / "0101201512z.gz"
        make_xmrg_gz(xmrg, DEFAULT_DOMAIN, fill=5000)
        result = runner.invoke(main, [
            "validate", str(xmrg),
            "--variable", "tair",
            "--check-filename",
        ])
        assert result.exit_code != 0


# ------------------------------------------------------------------ #
# scan-log command
# ------------------------------------------------------------------ #

class TestScanLogCommand:
    def _write_log(self, path: Path, lines: list[str]) -> None:
        with open(path, "w") as f:
            for line in lines:
                f.write(line + "\n")

    def test_scan_log_clean(self, runner, tmp_path):
        log = tmp_path / "clean.log"
        self._write_log(log, ["Run started", "All OK", "Run complete"])
        result = runner.invoke(main, ["scan-log", str(log)])
        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_scan_log_missing_tair(self, runner, tmp_path):
        log = tmp_path / "missing.log"
        self._write_log(log, [
            "missing tair : 2015-Jan-01 01:00:00",
            "missing tair : 2015-Jan-01 02:00:00",
        ])
        result = runner.invoke(main, ["scan-log", str(log)])
        assert result.exit_code != 0
        assert "FAILED" in result.output

    def test_scan_log_no_fail_flag(self, runner, tmp_path):
        """--no-fail-on-missing should exit 0 even with missing messages."""
        log = tmp_path / "nofail.log"
        self._write_log(log, ["missing tair 01:00"])
        result = runner.invoke(main, [
            "scan-log", str(log), "--no-fail-on-missing"
        ])
        assert result.exit_code == 0

    def test_scan_log_missing_file(self, runner, tmp_path):
        result = runner.invoke(main, ["scan-log", str(tmp_path / "ghost.log")])
        assert result.exit_code != 0
