"""Tests for hrapxmrg.validator – validation logic."""

import gzip
import io
import re
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hrapxmrg.config import DEFAULT_DOMAIN, HRAPDomain
from hrapxmrg.validator import (
    ValidationError,
    ValidationResult,
    scan_rdhm_log,
    summarize_log_scan,
    validate_data_range,
    validate_filename,
    validate_xmrg_file,
)
from hrapxmrg.xmrg_io import XMRGWriter


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def make_simple_xmrg(path: Path, domain: HRAPDomain,
                     fill: int = 5000, dtype=np.int16) -> None:
    """Write a simple XMRG file filled with *fill* using XMRGWriter."""
    data = np.full((domain.maxy, domain.maxx), fill_value=fill, dtype=dtype)
    writer = XMRGWriter(domain, storage_dtype=dtype, rows_north_first=False)
    writer.write(data, path)


# ------------------------------------------------------------------ #
# ValidationResult
# ------------------------------------------------------------------ #

class TestValidationResult:
    def test_passed_when_empty(self):
        r = ValidationResult()
        assert r.passed

    def test_failed_when_errors_added(self):
        r = ValidationResult()
        r.add_error("something is wrong")
        assert not r.passed

    def test_raise_if_failed(self):
        r = ValidationResult()
        r.add_error("oops")
        with pytest.raises(ValidationError):
            r.raise_if_failed()

    def test_warnings_do_not_cause_failure(self):
        r = ValidationResult()
        r.add_warning("just a warning")
        assert r.passed

    def test_str_passed(self):
        r = ValidationResult()
        assert "PASSED" in str(r)

    def test_str_failed_contains_error(self):
        r = ValidationResult()
        r.add_error("bad file")
        s = str(r)
        assert "FAILED" in s
        assert "bad file" in s


# ------------------------------------------------------------------ #
# validate_filename
# ------------------------------------------------------------------ #

class TestValidateFilename:
    @pytest.mark.parametrize("fname,variable", [
        ("tair0101201512z.gz", "tair"),
        ("tmax0101201512z.gz", "tmax"),
        ("tmin0101201512z.gz", "tmin"),
    ])
    def test_valid_daily_temperature(self, fname, variable):
        result = validate_filename(fname, variable=variable)
        assert result.passed, str(result)

    @pytest.mark.parametrize("fname,variable", [
        ("tair0101201500z.gz", "tair"),
        ("tair0101201523z.gz", "tair"),
        ("tmax0101201506z.gz", "tmax"),
    ])
    def test_valid_hourly_temperature(self, fname, variable):
        result = validate_filename(fname, variable=variable)
        assert result.passed, str(result)

    def test_invalid_missing_prefix(self):
        """A filename without the tair prefix should fail for tair."""
        result = validate_filename("0101201512z.gz", variable="tair")
        assert not result.passed

    def test_wrong_prefix_for_variable(self):
        """tmax prefix for tair variable should fail."""
        result = validate_filename("tmax0101201512z.gz", variable="tair")
        assert not result.passed

    def test_precip_no_prefix_warning(self):
        """Precipitation filename (no prefix) gets a warning, not error."""
        result = validate_filename("0101200012z.gz", variable="prep")
        # Should not have errors (warnings are OK for prep)
        assert result.passed

    def test_unknown_variable_errors(self):
        result = validate_filename("tair0101201512z.gz", variable="unknown_var")
        assert not result.passed

    def test_generic_no_variable(self):
        """Without variable, recognised patterns should not produce warnings."""
        result = validate_filename("tair0101201512z.gz")
        assert result.passed


# ------------------------------------------------------------------ #
# validate_data_range
# ------------------------------------------------------------------ #

class TestValidateDataRange:
    def test_tair_valid_range(self):
        """F100 int16 values in plausible tair range should pass."""
        data = np.array([[3200, 4000, 5000]], dtype=np.int16)
        result = validate_data_range(data, variable="tair")
        assert result.passed

    def test_tair_too_low(self):
        """Values below TAIR valid_stored_min (-4000) should fail."""
        data = np.array([[-5000]], dtype=np.int16)
        result = validate_data_range(data, variable="tair")
        assert not result.passed
        assert any("min" in e for e in result.errors)

    def test_tair_all_nodata_warns(self):
        """All nodata cells should produce a warning but not an error."""
        data = np.full((2, 2), fill_value=-9999, dtype=np.int16)
        result = validate_data_range(data, variable="tair", nodata=-9999)
        assert result.passed
        assert any("nodata" in w.lower() for w in result.warnings)

    def test_prep_valid(self):
        data = np.array([[0, 100, 500]], dtype=np.int16)
        result = validate_data_range(data, variable="prep")
        assert result.passed

    def test_unknown_variable_errors(self):
        data = np.zeros((2, 2))
        result = validate_data_range(data, variable="badvar")
        assert not result.passed


# ------------------------------------------------------------------ #
# validate_xmrg_file
# ------------------------------------------------------------------ #

class TestValidateXmrgFile:
    def test_valid_file_passes(self, tmp_path):
        """A correctly written XMRG file with tair data should pass."""
        out = tmp_path / "tair0101201512z.gz"
        make_simple_xmrg(out, DEFAULT_DOMAIN, fill=5000)
        result = validate_xmrg_file(out, variable="tair",
                                    expected_domain=DEFAULT_DOMAIN,
                                    nodata=-9999)
        assert result.passed, str(result)

    def test_missing_file_fails(self, tmp_path):
        result = validate_xmrg_file(tmp_path / "nonexistent.gz")
        assert not result.passed

    def test_wrong_domain_fails(self, tmp_path):
        out = tmp_path / "wrong_domain.gz"
        wrong_domain = HRAPDomain(xor=100, yor=200, maxx=32, maxy=40)
        make_simple_xmrg(out, wrong_domain, fill=5000)
        result = validate_xmrg_file(out, expected_domain=DEFAULT_DOMAIN)
        assert not result.passed
        assert any("mismatch" in e.lower() for e in result.errors)

    def test_data_range_failure_propagated(self, tmp_path):
        """Out-of-range stored values should fail validation."""
        out = tmp_path / "bad_range.gz"
        # Fill with -5000 which is below TAIR valid_stored_min (-4000)
        small_domain = HRAPDomain(xor=341, yor=313, maxx=4, maxy=3)
        data = np.full((3, 4), fill_value=-5000, dtype=np.int16)
        writer = XMRGWriter(small_domain, rows_north_first=False)
        writer.write(data, out)
        result = validate_xmrg_file(out, variable="tair",
                                    expected_domain=small_domain,
                                    nodata=-9999)
        assert not result.passed

    def test_no_variable_no_range_check(self, tmp_path):
        """Without variable, no range check is done; structural check only."""
        out = tmp_path / "novar.gz"
        make_simple_xmrg(out, DEFAULT_DOMAIN, fill=-5000)
        result = validate_xmrg_file(out, expected_domain=DEFAULT_DOMAIN)
        assert result.passed  # no range check → passes


# ------------------------------------------------------------------ #
# scan_rdhm_log
# ------------------------------------------------------------------ #

class TestScanRdhmLog:
    def _write_log(self, path: Path, lines: list[str]) -> None:
        with open(path, "w") as f:
            for line in lines:
                f.write(line + "\n")

    def test_no_missing_messages(self, tmp_path):
        log = tmp_path / "clean.log"
        self._write_log(log, [
            "RDHM run started",
            "Processing 2015-01-01",
            "Run complete OK",
        ])
        matches = scan_rdhm_log(log)
        assert all(len(v) == 0 for v in matches.values())

    def test_detects_missing_tair(self, tmp_path):
        log = tmp_path / "missing.log"
        self._write_log(log, [
            "missing tair : 2015-Jan-01 13:00:00",
            "missing tair : 2015-Jan-01 14:00:00",
            "Processing OK",
        ])
        matches = scan_rdhm_log(log, variables=["tair"])
        assert len(matches["tair"]) == 2

    def test_detects_missing_tmax(self, tmp_path):
        log = tmp_path / "tmax_missing.log"
        self._write_log(log, ["MISSING TMAX 2015-01-01"])
        matches = scan_rdhm_log(log, variables=["tmax"])
        assert len(matches["tmax"]) == 1

    def test_missing_log_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_rdhm_log(tmp_path / "nonexistent.log")

    def test_summarize_passed(self, tmp_path):
        log = tmp_path / "ok.log"
        self._write_log(log, ["all good"])
        matches = scan_rdhm_log(log)
        summary = summarize_log_scan(matches)
        assert "PASSED" in summary

    def test_summarize_failed(self, tmp_path):
        log = tmp_path / "fail.log"
        self._write_log(log, ["missing tair 01:00"])
        matches = scan_rdhm_log(log, variables=["tair"])
        summary = summarize_log_scan(matches)
        assert "FAILED" in summary
        assert "tair" in summary
