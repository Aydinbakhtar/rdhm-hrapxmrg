"""PRISM BIL/ZIP helpers and conversion pipeline."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re
import tempfile
import zipfile

import numpy as np

from .filenames import parse_date, rdhm_filename
from .hrap import TargetGrid
from .pipeline import raster_to_xmrg, raster_to_xmrg_manifest, write_raster_to_xmrg_manifest
from .validate import ValidationResult

PRISM_VARIABLES = {"ppt", "tmean", "tmin", "tmax"}
PRISM_TO_RDHM = {
    "ppt": "prep",
    "tmean": "tair",
    "tmin": "tmin",
    "tmax": "tmax",
}


def is_zip(path: str | Path) -> bool:
    return Path(path).suffix.lower() == ".zip"


def list_prism_zip_members(path: str | Path) -> list[str]:
    with zipfile.ZipFile(path) as zf:
        return zf.namelist()


def find_prism_bil_in_zip(path: str | Path) -> str:
    bil_members = [
        name
        for name in list_prism_zip_members(path)
        if not name.endswith("/") and Path(name).suffix.lower() == ".bil"
    ]
    if not bil_members:
        raise ValueError(f"{path}: ZIP contains no .bil file")
    if len(bil_members) == 1:
        return bil_members[0]

    basenames = [Path(name).name for name in bil_members]
    unique_basenames = [name for name in bil_members if basenames.count(Path(name).name) == 1]
    if len(unique_basenames) == 1:
        return unique_basenames[0]
    raise ValueError(f"{path}: ZIP contains multiple .bil files: {bil_members}")


def extract_prism_zip(path: str | Path, temp_dir: str | Path) -> Path:
    """Extract a PRISM ZIP and return the extracted .bil path."""
    bil_member = find_prism_bil_in_zip(path)
    temp_dir = Path(temp_dir)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(temp_dir)
    return temp_dir / bil_member


def prism_variable_from_filename(path: str | Path) -> str:
    name = Path(path).name.lower()
    match = re.search(r"(?:^|[_-])(ppt|tmean|tmin|tmax)(?:[_-]|$)", name)
    if match is None:
        raise ValueError(f"{path}: could not parse PRISM variable from filename")
    return match.group(1)


def prism_date_from_filename(path: str | Path) -> date:
    name = Path(path).name
    match = re.search(r"(19|20)\d{6}", name)
    if match is None:
        raise ValueError(f"{path}: could not parse YYYYMMDD date from filename")
    value = match.group(0)
    return date(int(value[0:4]), int(value[4:6]), int(value[6:8]))


def prism_units(prism_variable: str) -> str:
    prism_variable = prism_variable.lower()
    if prism_variable == "ppt":
        return "mm"
    if prism_variable in {"tmean", "tmin", "tmax"}:
        return "C"
    raise ValueError(f"unsupported PRISM variable: {prism_variable}")


def default_rdhm_variable(prism_variable: str) -> str:
    prism_variable = prism_variable.lower()
    try:
        return PRISM_TO_RDHM[prism_variable]
    except KeyError as exc:
        raise ValueError(f"unsupported PRISM variable: {prism_variable}") from exc


def target_units_for_rdhm_variable(rdhm_variable: str) -> str:
    rdhm_variable = rdhm_variable.lower()
    if rdhm_variable == "prep":
        return "mm"
    if rdhm_variable in {"tair", "tmax", "tmin"}:
        return "F"
    raise ValueError(f"unsupported RDHM variable: {rdhm_variable}")


def _resolve_date(value: str | date | None, input_path: str | Path) -> date:
    if value is None:
        return prism_date_from_filename(input_path)
    parsed = parse_date(value)
    if not isinstance(parsed, date):
        raise ValueError("date must resolve to a date")
    return parsed.date() if hasattr(parsed, "date") else parsed


def _resolve_output_path(
    *,
    output: str | Path | None,
    output_dir: str | Path | None,
    rdhm_variable: str,
    date_value: date,
    hour: int | None,
    daily_precip: bool,
) -> Path:
    if output and output_dir:
        raise ValueError("provide either output or output_dir, not both")
    if not output and not output_dir:
        raise ValueError("provide either output or output_dir")
    if output:
        return Path(output)
    filename = rdhm_filename(
        rdhm_variable,
        date_value,
        hour=hour,
        daily=daily_precip,
        suffix_gz=True,
    )
    return Path(output_dir) / filename


def _validate_timing_options(rdhm_variable: str, hour: int | None, daily_precip: bool) -> None:
    if rdhm_variable == "prep":
        if daily_precip and hour is not None:
            raise ValueError("prep cannot use both daily_precip and hour")
        if not daily_precip and hour is None:
            raise ValueError("prep requires daily_precip=True or hour")
        return
    if daily_precip:
        raise ValueError("daily_precip is only valid for prep")
    if hour is None:
        raise ValueError(f"{rdhm_variable} requires hour")


def _raster_path_for_input(path: str | Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    path = Path(path)
    if not is_zip(path):
        return path, None
    temp = tempfile.TemporaryDirectory()
    return extract_prism_zip(path, temp.name), temp


def prism_to_xmrg(
    *,
    input_path: str | Path,
    target_grid: TargetGrid,
    prism_variable: str | None = None,
    rdhm_variable: str | None = None,
    date_value: str | date | None = None,
    hour: int | None = None,
    daily_precip: bool = False,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
    report: str | Path | None = None,
    resampling: str = "bilinear",
    band: int = 1,
    target_domain_source: str | None = None,
) -> ValidationResult:
    """Convert a PRISM BIL/ZIP/raster to RDHM-ready XMRG."""
    input_path = Path(input_path)
    prism_variable = (prism_variable or prism_variable_from_filename(input_path)).lower()
    if prism_variable not in PRISM_VARIABLES:
        raise ValueError(f"unsupported PRISM variable: {prism_variable}")

    rdhm_variable = (rdhm_variable or default_rdhm_variable(prism_variable)).lower()
    source_units = prism_units(prism_variable)
    target_units = target_units_for_rdhm_variable(rdhm_variable)
    parsed_date = _resolve_date(date_value, input_path)
    _validate_timing_options(rdhm_variable, hour, daily_precip)
    output_path = _resolve_output_path(
        output=output,
        output_dir=output_dir,
        rdhm_variable=rdhm_variable,
        date_value=parsed_date,
        hour=hour,
        daily_precip=daily_precip,
    )

    raster_path, temp = _raster_path_for_input(input_path)
    try:
        result = raster_to_xmrg(
            input_raster=raster_path,
            output_xmrg=output_path,
            variable=rdhm_variable,
            target_grid=target_grid,
            band=band,
            resampling=resampling,
            source_units=source_units,
            target_units=target_units,
        )

        if report is not None:
            manifest = raster_to_xmrg_manifest(
                input_raster=input_path,
                output_xmrg=output_path,
                variable=rdhm_variable,
                target_grid=target_grid,
                validation=result,
                date=parsed_date.isoformat(),
                hour=hour,
                daily_precip=daily_precip,
                source_units=source_units,
                target_units=target_units,
                target_domain_source=target_domain_source,
            )
            write_raster_to_xmrg_manifest(report, manifest)

        return result
    finally:
        if temp is not None:
            temp.cleanup()


def prism_info(path: str | Path, *, sample_stats: bool = False) -> dict[str, object]:
    """Return JSON-ready information about a PRISM raster or ZIP."""
    path = Path(path)
    info: dict[str, object] = {
        "input": str(path),
        "is_zip": is_zip(path),
    }

    if is_zip(path):
        members = list_prism_zip_members(path)
        info["zip_members"] = members
        try:
            info["bil_path"] = find_prism_bil_in_zip(path)
        except ValueError as exc:
            info["bil_error"] = str(exc)

    try:
        variable = prism_variable_from_filename(path)
        info["parsed_prism_variable"] = variable
        info["source_units"] = prism_units(variable)
        info["default_rdhm_variable"] = default_rdhm_variable(variable)
    except ValueError as exc:
        info["parsed_prism_variable_error"] = str(exc)

    try:
        info["parsed_date"] = prism_date_from_filename(path).isoformat()
    except ValueError as exc:
        info["parsed_date_error"] = str(exc)

    raster_path: Path | None = None
    temp: tempfile.TemporaryDirectory[str] | None = None
    try:
        raster_path, temp = _raster_path_for_input(path)
        info["resolved_raster_path"] = str(raster_path)
        info.update(_raster_metadata(raster_path, sample_stats=sample_stats))
    except Exception as exc:
        info["raster_error"] = str(exc)
    finally:
        if temp is not None:
            temp.cleanup()

    return info


def _raster_metadata(path: str | Path, *, sample_stats: bool = False) -> dict[str, object]:
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "PRISM raster inspection requires rasterio. Install with: python -m pip install rasterio"
        ) from exc

    with rasterio.open(path) as src:
        metadata: dict[str, object] = {
            "driver": src.driver,
            "crs": None if src.crs is None else src.crs.to_string(),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "nodata": src.nodata,
            "dtype": src.dtypes[0] if src.dtypes else None,
        }
        if sample_stats:
            arr = src.read(1).astype("float64")
            if src.nodata is not None:
                arr = arr[~np.isclose(arr, src.nodata)]
            arr = arr[np.isfinite(arr)]
            metadata["sample_stats"] = {
                "min": float(np.nanmin(arr)) if arr.size else None,
                "mean": float(np.nanmean(arr)) if arr.size else None,
                "max": float(np.nanmax(arr)) if arr.size else None,
            }
        return metadata
