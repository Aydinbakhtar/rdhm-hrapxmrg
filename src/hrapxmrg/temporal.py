"""Daily-to-hourly RDHM forcing generation."""

from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .filenames import parse_date, precip_filename_for_end_time, rdhm_filename
from .hrap import TargetGrid
from .pipeline import raster_to_xmrg_manifest, write_raster_to_xmrg_manifest
from .prism import extract_prism_zip, is_zip
from .regrid import convert_units, reproject_raster_to_hrap
from .validate import validate_xmrg_file, ValidationResult
from .variables import get_variable_spec
from .xmrg import write_xmrg


PPT_SUMMARY_COLUMNS = [
    "date",
    "hour",
    "end_time",
    "output",
    "ok",
    "min",
    "mean",
    "max",
    "n_valid",
    "report",
]

TEMP_SUMMARY_COLUMNS = [
    "date",
    "hour",
    "valid_time",
    "output",
    "ok",
    "min",
    "mean",
    "max",
    "n_valid",
    "report",
]


def _as_date(value: str | date | datetime) -> date:
    parsed = parse_date(value)
    return parsed.date() if isinstance(parsed, datetime) else parsed


def _valid_mask(array: np.ndarray, nodata: float) -> np.ndarray:
    return np.isfinite(array) & (array > -900) & ~np.isclose(array, nodata)


def daily_ppt_to_hourly_arrays(
    daily_array: np.ndarray,
    method: str = "uniform",
    nodata: float = -999.0,
) -> list[np.ndarray]:
    """Split a daily precipitation array into 24 hourly accumulation arrays."""
    if method != "uniform":
        raise ValueError("daily precipitation method must be 'uniform'")

    daily = np.asarray(daily_array, dtype=np.float32)
    valid = _valid_mask(daily, nodata)
    hourly = []
    for _hour in range(24):
        arr = np.full(daily.shape, nodata, dtype=np.float32)
        arr[valid] = daily[valid] / 24.0
        hourly.append(arr)

    summed = np.sum(np.stack(hourly), axis=0)
    if not np.allclose(summed[valid], daily[valid], atol=1e-4):
        raise ValueError("hourly precipitation sum check failed")
    return hourly


def hourly_precip_end_times(physical_date: str | date | datetime) -> list[datetime]:
    """Return 24 hourly precipitation accumulation end times for a physical day."""
    base = _as_date(physical_date)
    start = datetime.combine(base, time(hour=0))
    return [start + timedelta(hours=hour) for hour in range(1, 25)]


def hourly_temperature_valid_times(physical_date: str | date | datetime) -> list[datetime]:
    """Return 24 hourly temperature valid times for a physical day."""
    base = _as_date(physical_date)
    start = datetime.combine(base, time(hour=0))
    return [start + timedelta(hours=hour) for hour in range(24)]


def hourly_temperature_from_tmin_tmax(
    tmin_array: np.ndarray,
    tmax_array: np.ndarray,
    method: str = "sinusoidal",
    tmean_array: np.ndarray | None = None,
    tmin_hour: int = 6,
    tmax_hour: int = 15,
    nodata: float = -999.0,
) -> list[np.ndarray]:
    """Generate 24 hourly Celsius temperature arrays from daily tmin/tmax."""
    if method != "sinusoidal":
        raise ValueError("daily temperature method must be 'sinusoidal'")
    if type(tmin_hour) is not int or not 0 <= tmin_hour <= 23:
        raise ValueError("tmin_hour must be 0 through 23")
    if type(tmax_hour) is not int or not 0 <= tmax_hour <= 23:
        raise ValueError("tmax_hour must be 0 through 23")

    tmin = np.asarray(tmin_array, dtype=np.float32)
    tmax = np.asarray(tmax_array, dtype=np.float32)
    valid = _valid_mask(tmin, nodata) & _valid_mask(tmax, nodata) & (tmax >= tmin)

    mean = (tmin + tmax) / 2.0
    amp = (tmax - tmin) / 2.0
    hourly: list[np.ndarray] = []
    for hour in range(24):
        arr = np.full(tmin.shape, nodata, dtype=np.float32)
        values = mean + amp * np.cos(2.0 * np.pi * (hour - tmax_hour) / 24.0)
        arr[valid] = np.clip(values[valid], tmin[valid], tmax[valid])
        hourly.append(arr)

    # First version intentionally does not force tmean matching; the parameter
    # is accepted so callers can pass it without changing the interface later.
    _ = tmean_array
    return hourly


def _raster_source(path: str | Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    path = Path(path)
    if not is_zip(path):
        return path, None
    temp = tempfile.TemporaryDirectory()
    return extract_prism_zip(path, temp.name), temp


def _write_array_xmrg(
    *,
    array: np.ndarray,
    output: Path,
    variable: str,
    target_grid: TargetGrid,
) -> ValidationResult:
    spec = get_variable_spec(variable)
    write_xmrg(
        output,
        array,
        xor=target_grid.xor,
        yor=target_grid.yor,
        header_type=spec.header_type,
        dtype=spec.dtype,
        scale=spec.storage_scale,
        missing=spec.missing_value,
        orientation="flipud",
    )
    return validate_xmrg_file(output, variable=variable, target_grid=target_grid)


def _report_path(report_dir: str | Path | None, filename: str) -> Path | None:
    if report_dir is None:
        return None
    stem = filename[:-3] if filename.endswith(".gz") else filename
    return Path(report_dir) / f"{stem}.report.json"


def _summary_value(details: dict[str, object], key: str) -> object:
    value = details.get(key)
    return "" if value is None else value


def _write_summary(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _daily_stats(array: np.ndarray, nodata: float = -999.0) -> dict[str, float | None]:
    valid = array[_valid_mask(array, nodata)]
    if valid.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": float(np.nanmin(valid)),
        "mean": float(np.nanmean(valid)),
        "max": float(np.nanmax(valid)),
    }


def daily_raster_ppt_to_hourly_xmrg(
    *,
    input_raster: str | Path,
    output_dir: str | Path,
    target_grid: TargetGrid,
    date_value: str | date | datetime,
    method: str = "uniform",
    band: int = 1,
    resampling: str = "bilinear",
    report_dir: str | Path | None = None,
    summary: str | Path | None = None,
    target_domain_source: str | None = None,
) -> dict[str, object]:
    """Convert daily physical-mm precipitation raster to 24 hourly prep XMRGs."""
    physical_date = _as_date(date_value)
    output_dir = Path(output_dir)
    raster_path, temp = _raster_source(input_raster)
    try:
        daily = reproject_raster_to_hrap(
            raster_path,
            target_grid,
            band=band,
            resampling=resampling,
            dst_nodata=get_variable_spec("prep").missing_value,
        )
    finally:
        if temp is not None:
            temp.cleanup()

    hourly_arrays = daily_ppt_to_hourly_arrays(daily, method=method, nodata=-999.0)
    end_times = hourly_precip_end_times(physical_date)
    rows: list[dict[str, object]] = []

    for arr, end_time in zip(hourly_arrays, end_times):
        filename = precip_filename_for_end_time(end_time)
        output = output_dir / filename
        result = _write_array_xmrg(array=arr, output=output, variable="prep", target_grid=target_grid)
        report = _report_path(report_dir, filename)
        if report is not None:
            manifest = raster_to_xmrg_manifest(
                input_raster=input_raster,
                output_xmrg=output,
                variable="prep",
                target_grid=target_grid,
                validation=result,
                date=physical_date.isoformat(),
                hour=end_time.hour,
                daily_precip=False,
                source_units="mm",
                target_units="mm",
                target_domain_source=target_domain_source,
            )
            manifest["method"] = method
            manifest["end_time"] = end_time.isoformat()
            write_raster_to_xmrg_manifest(report, manifest)

        rows.append(
            {
                "date": physical_date.isoformat(),
                "hour": end_time.hour,
                "end_time": end_time.isoformat(),
                "output": str(output),
                "ok": result.ok,
                "min": _summary_value(result.details, "min"),
                "mean": _summary_value(result.details, "mean"),
                "max": _summary_value(result.details, "max"),
                "n_valid": _summary_value(result.details, "n_valid"),
                "report": "" if report is None else str(report),
            }
        )

    if summary is not None:
        _write_summary(summary, rows, PPT_SUMMARY_COLUMNS)

    daily_stats = _daily_stats(daily)
    ok = all(bool(row["ok"]) for row in rows)
    hourly_sum = np.sum(np.stack(hourly_arrays), axis=0)
    valid = _valid_mask(daily, -999.0)
    sum_ok = bool(np.allclose(hourly_sum[valid], daily[valid], atol=0.05))
    return {
        "ok": ok and sum_ok,
        "variable": "prep",
        "date": physical_date.isoformat(),
        "method": method,
        "n_files": len(rows),
        "output_dir": str(output_dir),
        "daily_min": daily_stats["min"],
        "daily_mean": daily_stats["mean"],
        "daily_max": daily_stats["max"],
        "hourly_sum_check": "PASS" if sum_ok else "FAIL",
        "rows": rows,
    }


def daily_temp_to_hourly_tair_xmrg(
    *,
    tmin_raster: str | Path,
    tmax_raster: str | Path,
    output_dir: str | Path,
    target_grid: TargetGrid,
    date_value: str | date | datetime,
    tmean_raster: str | Path | None = None,
    method: str = "sinusoidal",
    tmin_hour: int = 6,
    tmax_hour: int = 15,
    band: int = 1,
    resampling: str = "bilinear",
    report_dir: str | Path | None = None,
    summary: str | Path | None = None,
    target_domain_source: str | None = None,
) -> dict[str, object]:
    """Convert daily Celsius tmin/tmax rasters to 24 hourly tair XMRGs."""
    physical_date = _as_date(date_value)
    output_dir = Path(output_dir)

    def load(path: str | Path) -> np.ndarray:
        raster_path, temp = _raster_source(path)
        try:
            return reproject_raster_to_hrap(
                raster_path,
                target_grid,
                band=band,
                resampling=resampling,
                dst_nodata=get_variable_spec("tair").missing_value,
            )
        finally:
            if temp is not None:
                temp.cleanup()

    tmin = load(tmin_raster)
    tmax = load(tmax_raster)
    tmean = load(tmean_raster) if tmean_raster is not None else None
    hourly_c = hourly_temperature_from_tmin_tmax(
        tmin,
        tmax,
        method=method,
        tmean_array=tmean,
        tmin_hour=tmin_hour,
        tmax_hour=tmax_hour,
        nodata=-999.0,
    )
    valid_times = hourly_temperature_valid_times(physical_date)
    rows: list[dict[str, object]] = []

    for arr_c, valid_time in zip(hourly_c, valid_times):
        arr_f = convert_units(arr_c, source_units="C", target_units="F")
        filename = rdhm_filename("tair", valid_time.date(), hour=valid_time.hour)
        output = output_dir / filename
        result = _write_array_xmrg(array=arr_f, output=output, variable="tair", target_grid=target_grid)
        report = _report_path(report_dir, filename)
        if report is not None:
            manifest = raster_to_xmrg_manifest(
                input_raster=str(tmin_raster),
                output_xmrg=output,
                variable="tair",
                target_grid=target_grid,
                validation=result,
                date=physical_date.isoformat(),
                hour=valid_time.hour,
                daily_precip=False,
                source_units="C",
                target_units="F",
                target_domain_source=target_domain_source,
            )
            manifest["input_files"] = {
                "tmin": str(tmin_raster),
                "tmax": str(tmax_raster),
                "tmean": None if tmean_raster is None else str(tmean_raster),
            }
            manifest["method"] = method
            manifest["valid_time"] = valid_time.isoformat()
            write_raster_to_xmrg_manifest(report, manifest)

        rows.append(
            {
                "date": physical_date.isoformat(),
                "hour": valid_time.hour,
                "valid_time": valid_time.isoformat(),
                "output": str(output),
                "ok": result.ok,
                "min": _summary_value(result.details, "min"),
                "mean": _summary_value(result.details, "mean"),
                "max": _summary_value(result.details, "max"),
                "n_valid": _summary_value(result.details, "n_valid"),
                "report": "" if report is None else str(report),
            }
        )

    if summary is not None:
        _write_summary(summary, rows, TEMP_SUMMARY_COLUMNS)

    return {
        "ok": all(bool(row["ok"]) for row in rows),
        "variable": "tair",
        "date": physical_date.isoformat(),
        "method": method,
        "n_files": len(rows),
        "output_dir": str(output_dir),
        "physical_units": "degF",
        "rows": rows,
    }
