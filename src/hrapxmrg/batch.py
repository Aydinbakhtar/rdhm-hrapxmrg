"""Batch raster-to-XMRG processing."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
import re
from typing import Any

from .filenames import rdhm_filename
from .hrap import TargetGrid
from .pipeline import raster_to_xmrg, raster_to_xmrg_manifest, write_raster_to_xmrg_manifest


SUMMARY_COLUMNS = [
    "input",
    "output",
    "variable",
    "date",
    "hour",
    "daily_precip",
    "dry_run",
    "ok",
    "message",
    "min",
    "mean",
    "max",
    "n_valid",
    "target_xor",
    "target_yor",
    "target_maxx",
    "target_maxy",
    "report",
]


def discover_input_files(input_dir: str | Path, pattern: str) -> list[Path]:
    """Return sorted input files matching pattern below input_dir."""
    return sorted(Path(input_dir).glob(pattern))


def parse_date_from_filename(path: str | Path, date_regex: str) -> date:
    """Parse a date from a filename using year/month/day named regex groups."""
    filename = Path(path).name
    match = re.search(date_regex, filename)
    if match is None:
        raise ValueError(f"{filename}: date regex did not match")

    groups = match.groupdict()
    missing = [name for name in ("year", "month", "day") if name not in groups]
    if missing:
        raise ValueError(f"date regex must include named groups year, month, day; missing {missing}")

    try:
        return date(int(groups["year"]), int(groups["month"]), int(groups["day"]))
    except ValueError as exc:
        raise ValueError(f"{filename}: invalid date parsed from filename") from exc


def _report_path(report_dir: str | Path | None, generated_filename: str) -> Path | None:
    if report_dir is None:
        return None
    stem = generated_filename[:-3] if generated_filename.endswith(".gz") else generated_filename
    return Path(report_dir) / f"{stem}.report.json"


def _detail_value(details: dict[str, object], key: str) -> object:
    value = details.get(key)
    return "" if value is None else value


def _summary_row(
    *,
    input_path: Path,
    output_path: Path | None,
    variable: str,
    parsed_date: date | None,
    hour: int | None,
    daily_precip: bool,
    dry_run: bool,
    ok: bool,
    message: str,
    target_grid: TargetGrid,
    report_path: Path | None = None,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    details = details or {}
    return {
        "input": str(input_path),
        "output": "" if output_path is None else str(output_path),
        "variable": variable,
        "date": "" if parsed_date is None else parsed_date.isoformat(),
        "hour": "" if hour is None else hour,
        "daily_precip": daily_precip,
        "dry_run": dry_run,
        "ok": ok,
        "message": message,
        "min": _detail_value(details, "min"),
        "mean": _detail_value(details, "mean"),
        "max": _detail_value(details, "max"),
        "n_valid": _detail_value(details, "n_valid"),
        "target_xor": target_grid.xor,
        "target_yor": target_grid.yor,
        "target_maxx": target_grid.maxx,
        "target_maxy": target_grid.maxy,
        "report": "" if report_path is None else str(report_path),
    }


def _validate_batch_options(variable: str, hour: int | None, daily_precip: bool) -> None:
    if daily_precip and variable != "prep":
        raise ValueError("daily_precip is only valid with variable='prep'")
    if variable == "prep":
        if daily_precip and hour is not None:
            raise ValueError("prep cannot use both daily_precip and hour")
        if not daily_precip and hour is None:
            raise ValueError("prep requires daily_precip=True or hour")
        return
    if variable in {"tair", "tmax", "tmin"}:
        if hour is None:
            raise ValueError(f"{variable} requires hour")
        return
    raise ValueError(f"unsupported batch variable: {variable}")


def batch_raster_to_xmrg(
    *,
    input_dir: str | Path,
    pattern: str,
    date_regex: str,
    variable: str,
    output_dir: str | Path,
    target_grid: TargetGrid,
    target_domain_source: str | None = None,
    hour: int | None = None,
    daily_precip: bool = False,
    source_units: str | None = None,
    target_units: str | None = None,
    band: int = 1,
    resampling: str = "bilinear",
    summary: str | Path | None = None,
    report_dir: str | Path | None = None,
    continue_on_error: bool = False,
    dry_run: bool = False,
    suffix_gz: bool = True,
) -> list[dict[str, object]]:
    """Convert many rasters to RDHM-ready XMRG files and return summary rows."""
    variable = variable.lower()
    _validate_batch_options(variable, hour, daily_precip)

    input_files = discover_input_files(input_dir, pattern)
    if not input_files:
        raise ValueError(f"no input files matched {Path(input_dir) / pattern}")

    rows: list[dict[str, object]] = []
    output_dir = Path(output_dir)

    for input_path in input_files:
        parsed_date: date | None = None
        output_path: Path | None = None
        report_path: Path | None = None

        try:
            parsed_date = parse_date_from_filename(input_path, date_regex)
            generated_filename = rdhm_filename(
                variable,
                parsed_date,
                hour=hour,
                daily=daily_precip,
                suffix_gz=suffix_gz,
            )
            output_path = output_dir / generated_filename
            report_path = _report_path(report_dir, generated_filename)

            if dry_run:
                rows.append(
                    _summary_row(
                        input_path=input_path,
                        output_path=output_path,
                        variable=variable,
                        parsed_date=parsed_date,
                        hour=hour,
                        daily_precip=daily_precip,
                        dry_run=True,
                        ok=True,
                        message="planned",
                        target_grid=target_grid,
                        report_path=report_path,
                    )
                )
                continue

            result = raster_to_xmrg(
                input_raster=input_path,
                output_xmrg=output_path,
                variable=variable,
                target_grid=target_grid,
                band=band,
                resampling=resampling,
                source_units=source_units,
                target_units=target_units,
            )

            if report_path is not None:
                manifest = raster_to_xmrg_manifest(
                    input_raster=input_path,
                    output_xmrg=output_path,
                    variable=variable,
                    target_grid=target_grid,
                    validation=result,
                    date=parsed_date.isoformat(),
                    hour=hour,
                    daily_precip=daily_precip,
                    source_units=source_units,
                    target_units=target_units,
                    target_domain_source=target_domain_source,
                )
                write_raster_to_xmrg_manifest(report_path, manifest)

            rows.append(
                _summary_row(
                    input_path=input_path,
                    output_path=output_path,
                    variable=variable,
                    parsed_date=parsed_date,
                    hour=hour,
                    daily_precip=daily_precip,
                    dry_run=False,
                    ok=result.ok,
                    message=result.message,
                    target_grid=target_grid,
                    report_path=report_path,
                    details=result.details,
                )
            )

            if not result.ok and not continue_on_error:
                raise ValueError(f"{input_path}: {result.message}")

        except Exception as exc:
            if not continue_on_error:
                raise
            rows.append(
                _summary_row(
                    input_path=input_path,
                    output_path=output_path,
                    variable=variable,
                    parsed_date=parsed_date,
                    hour=hour,
                    daily_precip=daily_precip,
                    dry_run=dry_run,
                    ok=False,
                    message=str(exc),
                    target_grid=target_grid,
                    report_path=report_path,
                )
            )

    if summary is not None:
        write_summary_csv(summary, rows)

    return rows


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write batch summary rows as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})
