"""XMRG and RDHM forcing validation functions.

Validation is central to the project: every generated forcing file should
pass structural, dimensional, and physical-range checks before being used
in an RDHM calibration run.

This module provides:

- :func:`validate_xmrg_file`: Full file-level validation.
- :func:`validate_data_range`: Physical-range checks for specific variables.
- :func:`validate_filename`: File-naming convention checks.
- :func:`scan_rdhm_log`: Scan an RDHM log file for missing-forcing messages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import HRAPDomain, DEFAULT_DOMAIN
from .variable_registry import get_variable, REGISTRY
from .xmrg_io import XMRGReader

# ------------------------------------------------------------------ #
# Result types
# ------------------------------------------------------------------ #

class ValidationResult:
    """Accumulated validation result.

    Attributes
    ----------
    passed:
        ``True`` if all checks passed (no errors).
    errors:
        List of error messages (non-empty means failed).
    warnings:
        List of warning messages (do not cause failure).
    """

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:
        lines = []
        if self.passed:
            lines.append("PASSED")
        else:
            lines.append("FAILED")
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        """Raise :class:`ValidationError` if there are any errors."""
        if not self.passed:
            raise ValidationError(str(self))


class ValidationError(Exception):
    """Raised when XMRG validation fails."""


# ------------------------------------------------------------------ #
# Filename validation
# ------------------------------------------------------------------ #

#: Regex patterns for valid XMRG filenames by variable.
#:
#: Daily temperature:  ``tair0101201512z.gz`` (prefix + MMDDYYYY + HH + z.gz)
#: Hourly temperature: ``tair0101201500z.gz`` through ``tair0101201523z.gz``
#: Precipitation (no prefix): ``0101200012z.gz``
_TEMPERATURE_DAILY_RE = re.compile(
    r"^(tair|tmax|tmin)(\d{8})12z\.gz$"
)
_TEMPERATURE_HOURLY_RE = re.compile(
    r"^(tair|tmax|tmin)(\d{10})z\.gz$"
)
_PRECIP_RE = re.compile(
    r"^(\d{10})z\.gz$"
)


def validate_filename(
    filename: str,
    variable: Optional[str] = None,
) -> ValidationResult:
    """Validate an XMRG filename against RDHM naming conventions.

    Parameters
    ----------
    filename:
        Bare filename (not full path) to validate.
    variable:
        Optional variable name (``"tair"``, ``"tmax"``, ``"tmin"``,
        ``"prep"``).  If provided, the prefix is checked against the
        variable registry.

    Returns
    -------
    :class:`ValidationResult`
    """
    result = ValidationResult()
    fname = Path(filename).name

    if variable is not None:
        try:
            spec = get_variable(variable)
        except KeyError as exc:
            result.add_error(str(exc))
            return result

        if variable in ("tair", "tmax", "tmin"):
            # Must match daily or hourly temperature pattern with correct prefix
            if not (
                _TEMPERATURE_DAILY_RE.match(fname) or
                _TEMPERATURE_HOURLY_RE.match(fname)
            ):
                result.add_error(
                    f"Filename {fname!r} does not match expected temperature "
                    f"pattern '{spec.file_prefix}MMDDYYYYHH[HH]z.gz'."
                )
            else:
                prefix = fname[:4]
                if prefix != spec.file_prefix:
                    result.add_error(
                        f"Filename prefix {prefix!r} does not match "
                        f"expected prefix {spec.file_prefix!r} for "
                        f"variable {variable!r}."
                    )
        elif variable == "prep":
            if not _PRECIP_RE.match(fname):
                result.add_warning(
                    f"Filename {fname!r} does not match expected precipitation "
                    "pattern MMDDYYYYHH z.gz.  Verify against existing "
                    "working RDHM files."
                )
    else:
        # Generic check: any recognised pattern
        if not (
            _TEMPERATURE_DAILY_RE.match(fname) or
            _TEMPERATURE_HOURLY_RE.match(fname) or
            _PRECIP_RE.match(fname)
        ):
            result.add_warning(
                f"Filename {fname!r} does not match any known RDHM XMRG "
                "naming pattern."
            )

    return result


# ------------------------------------------------------------------ #
# Data-range validation
# ------------------------------------------------------------------ #

def validate_data_range(
    data: np.ndarray,
    variable: str,
    nodata: Optional[float] = None,
) -> ValidationResult:
    """Check that stored XMRG values are within physically plausible bounds.

    For ``F100_int16`` variables (tair, tmax, tmin) the stored values are
    Fahrenheit × 100 and the physical values after dividing by 100 are
    reported in the error messages.

    Parameters
    ----------
    data:
        2-D array of stored values (as read from the XMRG binary).
    variable:
        Variable name from the registry.
    nodata:
        Stored nodata sentinel; cells equal to *nodata* are excluded.

    Returns
    -------
    :class:`ValidationResult`
    """
    result = ValidationResult()

    try:
        spec = get_variable(variable)
    except KeyError as exc:
        result.add_error(str(exc))
        return result

    arr = data.astype(np.float64)
    mask = np.zeros_like(arr, dtype=bool)
    if nodata is not None:
        mask = arr == float(nodata)

    valid = arr[~mask]
    if valid.size == 0:
        result.add_warning("All cells are nodata; cannot check physical range.")
        return result

    vmin = float(valid.min())
    vmax = float(valid.max())

    # Check stored bounds
    if spec.valid_stored_min is not None and vmin < spec.valid_stored_min:
        phys_min = spec.physical_from_stored(vmin)
        phys_bound = spec.physical_from_stored(spec.valid_stored_min)
        result.add_error(
            f"Variable {variable!r}: stored min {vmin:.0f} is below "
            f"expected minimum {spec.valid_stored_min:.0f} "
            f"(physical: {phys_min:.2f} vs bound {phys_bound:.2f} "
            f"in {_physical_units(spec.scale_convention)})."
        )

    if spec.valid_stored_max is not None and vmax > spec.valid_stored_max:
        phys_max = spec.physical_from_stored(vmax)
        phys_bound = spec.physical_from_stored(spec.valid_stored_max)
        result.add_error(
            f"Variable {variable!r}: stored max {vmax:.0f} exceeds "
            f"expected maximum {spec.valid_stored_max:.0f} "
            f"(physical: {phys_max:.2f} vs bound {phys_bound:.2f} "
            f"in {_physical_units(spec.scale_convention)})."
        )

    return result


def _physical_units(scale: str) -> str:
    if scale == "F100_int16":
        return "°F"
    if scale == "mm_raw":
        return "mm"
    return "raw units"


# ------------------------------------------------------------------ #
# Structural / file-level validation
# ------------------------------------------------------------------ #

def validate_xmrg_file(
    path: str | Path,
    variable: Optional[str] = None,
    expected_domain: Optional[HRAPDomain] = None,
    nodata: Optional[float] = None,
) -> ValidationResult:
    """Validate an XMRG file structurally and (optionally) by content.

    Checks performed:

    1. File exists and is readable.
    2. gzip decompression succeeds.
    3. Fortran record structure is valid.
    4. Header XOR/YOR/MAXX/MAXY match *expected_domain* (if provided).
    5. Data row count and cell count are correct.
    6. If *variable* is given, physical-range check via
       :func:`validate_data_range`.

    Parameters
    ----------
    path:
        Path to the XMRG ``.gz`` file.
    variable:
        Optional variable name for range validation.
    expected_domain:
        Expected HRAP domain.  Uses :data:`~hrapxmrg.config.DEFAULT_DOMAIN`
        if not provided and *variable* is given.
    nodata:
        Stored nodata sentinel; defaults to the domain's ``nodata`` value.

    Returns
    -------
    :class:`ValidationResult`
    """
    result = ValidationResult()
    path = Path(path)

    # 1. File existence
    if not path.exists():
        result.add_error(f"File not found: {path}")
        return result

    # 2-4. Read the file (catches gzip, record, and header errors)
    try:
        reader = XMRGReader(path)
    except Exception as exc:
        result.add_error(f"Failed to read XMRG file: {exc}")
        return result

    # 4. Domain check
    domain_to_check = expected_domain
    if domain_to_check is None and variable is not None:
        domain_to_check = DEFAULT_DOMAIN

    if domain_to_check is not None:
        if not reader.domain.matches(domain_to_check):
            result.add_error(
                f"Domain mismatch: file has "
                f"xor={reader.domain.xor}, yor={reader.domain.yor}, "
                f"maxx={reader.domain.maxx}, maxy={reader.domain.maxy}; "
                f"expected "
                f"xor={domain_to_check.xor}, yor={domain_to_check.yor}, "
                f"maxx={domain_to_check.maxx}, maxy={domain_to_check.maxy}."
            )

    # 5. Data shape sanity (already enforced by reader, but double-check)
    if reader.data.shape != (reader.domain.maxy, reader.domain.maxx):
        result.add_error(
            f"Data shape {reader.data.shape} does not match domain "
            f"({reader.domain.maxy}, {reader.domain.maxx})."
        )

    # 6. Physical range check
    if variable is not None:
        nodata_val = nodata
        if nodata_val is None and domain_to_check is not None:
            nodata_val = domain_to_check.nodata
        range_result = validate_data_range(reader.data, variable, nodata=nodata_val)
        result.errors.extend(range_result.errors)
        result.warnings.extend(range_result.warnings)

    return result


# ------------------------------------------------------------------ #
# RDHM log scanner
# ------------------------------------------------------------------ #

#: Variables whose absence in RDHM logs we watch for.
_MISSING_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("tair", re.compile(r"missing\s+tair", re.IGNORECASE)),
    ("tmax", re.compile(r"missing\s+tmax", re.IGNORECASE)),
    ("tmin", re.compile(r"missing\s+tmin", re.IGNORECASE)),
    ("precip", re.compile(r"missing\s+precip", re.IGNORECASE)),
    ("general", re.compile(r"missing\s+\w+", re.IGNORECASE)),
]


def scan_rdhm_log(
    log_path: str | Path,
    variables: Optional[Sequence[str]] = None,
) -> dict[str, list[str]]:
    """Scan an RDHM log file for missing-forcing messages.

    Parameters
    ----------
    log_path:
        Path to the RDHM log file (plain text).
    variables:
        Sequence of variable names to check.  Defaults to
        ``["tair", "tmax", "tmin"]``.

    Returns
    -------
    dict
        Mapping from variable name to list of matching log lines.
        If a variable has no matches, its list is empty.

    Example
    -------
    >>> matches = scan_rdhm_log("rdhm_run.log")
    >>> for var, lines in matches.items():
    ...     print(f"{var}: {len(lines)} missing messages")
    """
    if variables is None:
        variables = ["tair", "tmax", "tmin"]

    patterns: list[tuple[str, re.Pattern]] = [
        (v, re.compile(rf"missing\s+{re.escape(v)}", re.IGNORECASE))
        for v in variables
    ]

    results: dict[str, list[str]] = {v: [] for v in variables}

    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"RDHM log not found: {log_path}")

    with open(log_path, "r", errors="replace") as fh:
        for line in fh:
            stripped = line.rstrip()
            for var, pattern in patterns:
                if pattern.search(stripped):
                    results[var].append(stripped)

    return results


def summarize_log_scan(matches: dict[str, list[str]]) -> str:
    """Format a human-readable summary of :func:`scan_rdhm_log` results.

    Parameters
    ----------
    matches:
        Output from :func:`scan_rdhm_log`.

    Returns
    -------
    str
        Multi-line summary.
    """
    lines = []
    any_missing = False
    for var, hits in matches.items():
        count = len(hits)
        if count:
            any_missing = True
            lines.append(f"  {var}: {count} missing-forcing message(s)")
            for hit in hits[:5]:
                lines.append(f"    {hit}")
            if count > 5:
                lines.append(f"    ... ({count - 5} more)")
        else:
            lines.append(f"  {var}: OK (no missing messages)")

    header = "RDHM log scan – FAILED (missing forcing detected):" if any_missing else \
             "RDHM log scan – PASSED (no missing forcing messages):"
    return header + "\n" + "\n".join(lines)
