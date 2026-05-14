"""Validation helpers for RDHM HRAP/XMRG forcing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import calendar
import re

import numpy as np

from .hrap import TargetGrid
from .variables import get_variable_spec
from .xmrg import read_xmrg


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str
    details: dict[str, object]


def validate_xmrg_file(
    path: str | Path,
    variable: str,
    target_grid: TargetGrid | None = None,
) -> ValidationResult:
    spec = get_variable_spec(variable)
    grid, meta = read_xmrg(path)

    details: dict[str, object] = {
        "path": str(path),
        "header_type": meta.header_type,
        "dtype": meta.dtype,
        "xor": meta.xor,
        "yor": meta.yor,
        "maxx": meta.maxx,
        "maxy": meta.maxy,
    }

    if target_grid is not None:
        expected = (target_grid.maxy, target_grid.maxx)
        actual = (meta.maxy, meta.maxx)
        if expected != actual:
            return ValidationResult(False, f"shape mismatch: expected {expected}, got {actual}", details)
        if int(meta.xor) != target_grid.xor or int(meta.yor) != target_grid.yor:
            return ValidationResult(False, "HRAP origin mismatch", details)

    valid = grid[(grid > -900) & np.isfinite(grid)]
    if valid.size == 0:
        return ValidationResult(False, "no valid cells", details)

    if variable in ("tair", "tmax", "tmin") and meta.dtype == "int16":
        physical = valid / spec.rdhm_read_divisor
    else:
        physical = valid

    details.update(
        min=float(np.nanmin(physical)),
        mean=float(np.nanmean(physical)),
        max=float(np.nanmax(physical)),
        n_valid=int(valid.size),
    )

    if float(np.nanmin(physical)) < spec.valid_min or float(np.nanmax(physical)) > spec.valid_max:
        return ValidationResult(False, f"{variable} outside physical range", details)

    return ValidationResult(True, "ok", details)


def validate_temperature_filename(filename: str) -> bool:
    daily = r"(tair|tmax|tmin)[0-9]{8}12z\.gz"
    hourly = r"(tair|tmax|tmin)[0-9]{10}z\.gz"
    return bool(re.fullmatch(f"({daily})|({hourly})", filename))


def expected_daily_count(year: int) -> int:
    return 366 if calendar.isleap(year) else 365


def expected_hourly_count(year: int) -> int:
    return 24 * expected_daily_count(year)


def scan_rdhm_log_for_missing_forcing(log_path: str | Path) -> dict[str, int]:
    text = Path(log_path).read_text(errors="ignore")
    return {
        "missing_tair": len(re.findall(r"missing\s+tair", text, flags=re.I)),
        "missing_tmax": len(re.findall(r"missing\s+tmax", text, flags=re.I)),
        "missing_tmin": len(re.findall(r"missing\s+tmin", text, flags=re.I)),
        "missing_precip": len(re.findall(r"missing\s+(xmrg|prep|precip)", text, flags=re.I)),
    }
