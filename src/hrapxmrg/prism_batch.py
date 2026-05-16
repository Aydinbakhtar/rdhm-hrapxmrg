"""Batch PRISM daily ZIP/BIL to hourly RDHM XMRG generation."""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .filenames import parse_date
from .hrap import TargetGrid
from .prism import prism_date_from_filename
from .temporal import daily_raster_ppt_to_hourly_xmrg, daily_temp_to_hourly_tair_xmrg


PREP_SUMMARY_COLUMNS = [
    "date",
    "source_ppt",
    "hour",
    "end_time",
    "output",
    "ok",
    "min",
    "mean",
    "max",
    "n_valid",
    "message",
    "report",
]

TAIR_SUMMARY_COLUMNS = [
    "date",
    "source_tmin",
    "source_tmax",
    "source_tmean",
    "tmean_missing",
    "hour",
    "valid_time",
    "output",
    "ok",
    "min",
    "mean",
    "max",
    "n_valid",
    "message",
    "report",
]


def _as_date(value: str | date | datetime | None) -> date | None:
    if value is None:
        return None
    parsed = parse_date(value)
    return parsed.date() if isinstance(parsed, datetime) else parsed


def discover_prism_files(input_dir: str | Path, pattern: str) -> list[Path]:
    return sorted(Path(input_dir).glob(pattern))


def prism_files_by_date(input_dir: str | Path, pattern: str) -> dict[date, Path]:
    files: dict[date, Path] = {}
    for path in discover_prism_files(input_dir, pattern):
        parsed_date = prism_date_from_filename(path)
        if parsed_date in files:
            raise ValueError(f"multiple PRISM files for {parsed_date}: {files[parsed_date]} and {path}")
        files[parsed_date] = path
    return files


def _filter_dates(dates: list[date], start_date: str | date | None, end_date: str | date | None) -> list[date]:
    start = _as_date(start_date)
    end = _as_date(end_date)
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be before or equal to end_date")
    return [
        d
        for d in sorted(dates)
        if (start is None or d >= start) and (end is None or d <= end)
    ]


def _write_summary(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _prep_failure_row(parsed_date: date | None, source: Path | None, message: str) -> dict[str, object]:
    return {
        "date": "" if parsed_date is None else parsed_date.isoformat(),
        "source_ppt": "" if source is None else str(source),
        "hour": "",
        "end_time": "",
        "output": "",
        "ok": False,
        "min": "",
        "mean": "",
        "max": "",
        "n_valid": "",
        "message": message,
        "report": "",
    }


def _tair_failure_row(
    parsed_date: date,
    *,
    source_tmin: Path | None,
    source_tmax: Path | None,
    source_tmean: Path | None,
    tmean_missing: bool,
    message: str,
) -> dict[str, object]:
    return {
        "date": parsed_date.isoformat(),
        "source_tmin": "" if source_tmin is None else str(source_tmin),
        "source_tmax": "" if source_tmax is None else str(source_tmax),
        "source_tmean": "" if source_tmean is None else str(source_tmean),
        "tmean_missing": tmean_missing,
        "hour": "",
        "valid_time": "",
        "output": "",
        "ok": False,
        "min": "",
        "mean": "",
        "max": "",
        "n_valid": "",
        "message": message,
        "report": "",
    }


def batch_prism_hourly_prep(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    target_grid: TargetGrid,
    pattern: str = "prism_ppt_*.zip",
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    method: str = "uniform",
    band: int = 1,
    resampling: str = "bilinear",
    summary: str | Path | None = None,
    report_dir: str | Path | None = None,
    continue_on_error: bool = False,
    target_domain_source: str | None = None,
) -> dict[str, object]:
    files_by_date = prism_files_by_date(input_dir, pattern)
    dates = _filter_dates(list(files_by_date), start_date, end_date)
    if not dates:
        raise ValueError("no PRISM ppt files matched the requested date range")

    rows: list[dict[str, object]] = []
    failed_days = 0

    for parsed_date in dates:
        source = files_by_date[parsed_date]
        try:
            result = daily_raster_ppt_to_hourly_xmrg(
                input_raster=source,
                output_dir=output_dir,
                target_grid=target_grid,
                date_value=parsed_date,
                method=method,
                band=band,
                resampling=resampling,
                report_dir=report_dir,
                target_domain_source=target_domain_source,
            )
            for row in result["rows"]:
                rows.append(
                    {
                        "date": row["date"],
                        "source_ppt": str(source),
                        "hour": row["hour"],
                        "end_time": row["end_time"],
                        "output": row["output"],
                        "ok": row["ok"],
                        "min": row["min"],
                        "mean": row["mean"],
                        "max": row["max"],
                        "n_valid": row["n_valid"],
                        "message": "ok" if row["ok"] else "validation failed",
                        "report": row["report"],
                    }
                )
            if not result["ok"]:
                failed_days += 1
                if not continue_on_error:
                    raise ValueError(f"{source}: hourly prep generation failed")
        except Exception as exc:
            failed_days += 1
            if not continue_on_error:
                raise
            rows.append(_prep_failure_row(parsed_date, source, str(exc)))

    if summary is not None:
        _write_summary(summary, rows, PREP_SUMMARY_COLUMNS)

    n_files = sum(1 for row in rows if row.get("output"))
    return {
        "ok": failed_days == 0 or continue_on_error,
        "variable": "prep",
        "n_days": len(dates),
        "n_files": n_files,
        "n_failed_days": failed_days,
        "output_dir": str(output_dir),
        "summary": None if summary is None else str(summary),
        "rows": rows,
    }


def batch_prism_hourly_tair(
    *,
    tmin_dir: str | Path,
    tmax_dir: str | Path,
    output_dir: str | Path,
    target_grid: TargetGrid,
    tmean_dir: str | Path | None = None,
    tmin_pattern: str = "prism_tmin_*.zip",
    tmax_pattern: str = "prism_tmax_*.zip",
    tmean_pattern: str = "prism_tmean_*.zip",
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    method: str = "sinusoidal",
    tmin_hour: int = 6,
    tmax_hour: int = 15,
    band: int = 1,
    resampling: str = "bilinear",
    summary: str | Path | None = None,
    report_dir: str | Path | None = None,
    continue_on_error: bool = False,
    target_domain_source: str | None = None,
) -> dict[str, object]:
    tmin_by_date = prism_files_by_date(tmin_dir, tmin_pattern)
    tmax_by_date = prism_files_by_date(tmax_dir, tmax_pattern)
    tmean_by_date = prism_files_by_date(tmean_dir, tmean_pattern) if tmean_dir is not None else {}

    all_dates = sorted(set(tmin_by_date) | set(tmax_by_date))
    dates = _filter_dates(all_dates, start_date, end_date)
    if not dates:
        raise ValueError("no PRISM tmin/tmax files matched the requested date range")

    rows: list[dict[str, object]] = []
    failed_days = 0
    processed_days = 0

    for parsed_date in dates:
        source_tmin = tmin_by_date.get(parsed_date)
        source_tmax = tmax_by_date.get(parsed_date)
        source_tmean = tmean_by_date.get(parsed_date)
        tmean_missing = tmean_dir is not None and source_tmean is None

        if source_tmin is None or source_tmax is None:
            failed_days += 1
            missing = []
            if source_tmin is None:
                missing.append("tmin")
            if source_tmax is None:
                missing.append("tmax")
            message = f"missing required PRISM files: {', '.join(missing)}"
            if not continue_on_error:
                raise ValueError(f"{parsed_date}: {message}")
            rows.append(
                _tair_failure_row(
                    parsed_date,
                    source_tmin=source_tmin,
                    source_tmax=source_tmax,
                    source_tmean=source_tmean,
                    tmean_missing=tmean_missing,
                    message=message,
                )
            )
            continue

        try:
            result = daily_temp_to_hourly_tair_xmrg(
                tmin_raster=source_tmin,
                tmax_raster=source_tmax,
                tmean_raster=source_tmean,
                output_dir=output_dir,
                target_grid=target_grid,
                date_value=parsed_date,
                method=method,
                tmin_hour=tmin_hour,
                tmax_hour=tmax_hour,
                band=band,
                resampling=resampling,
                report_dir=report_dir,
                target_domain_source=target_domain_source,
            )
            processed_days += 1
            for row in result["rows"]:
                rows.append(
                    {
                        "date": row["date"],
                        "source_tmin": str(source_tmin),
                        "source_tmax": str(source_tmax),
                        "source_tmean": "" if source_tmean is None else str(source_tmean),
                        "tmean_missing": tmean_missing,
                        "hour": row["hour"],
                        "valid_time": row["valid_time"],
                        "output": row["output"],
                        "ok": row["ok"],
                        "min": row["min"],
                        "mean": row["mean"],
                        "max": row["max"],
                        "n_valid": row["n_valid"],
                        "message": "ok" if row["ok"] else "validation failed",
                        "report": row["report"],
                    }
                )
            if not result["ok"]:
                failed_days += 1
                if not continue_on_error:
                    raise ValueError(f"{parsed_date}: hourly tair generation failed")
        except Exception as exc:
            failed_days += 1
            if not continue_on_error:
                raise
            rows.append(
                _tair_failure_row(
                    parsed_date,
                    source_tmin=source_tmin,
                    source_tmax=source_tmax,
                    source_tmean=source_tmean,
                    tmean_missing=tmean_missing,
                    message=str(exc),
                )
            )

    if summary is not None:
        _write_summary(summary, rows, TAIR_SUMMARY_COLUMNS)

    n_files = sum(1 for row in rows if row.get("output"))
    return {
        "ok": failed_days == 0 or continue_on_error,
        "variable": "tair",
        "n_days": processed_days,
        "n_files": n_files,
        "n_failed_days": failed_days,
        "output_dir": str(output_dir),
        "summary": None if summary is None else str(summary),
        "rows": rows,
    }
