"""Batch forecast NetCDF to RDHM XMRG generation.

This module loops over explicitly requested forecast leads and optional
ensemble members, delegating each selected 2D slice to ``nc_to_xmrg``. It does
not duplicate NetCDF slicing, regridding, unit conversion, or XMRG writing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .filenames import rdhm_filename
from .hrap import TargetGrid
from .netcdf import _parse_datetime, _xarray, identify_dimension_roles, nc_to_xmrg, normalize_lead_to_hours


SUMMARY_COLUMNS = [
    "input",
    "nc_variable",
    "rdhm_variable",
    "init_time",
    "init_index",
    "lead_hour",
    "lead_index",
    "valid_time",
    "valid_time_source",
    "member_index",
    "member_name",
    "output",
    "ok",
    "message",
    "min",
    "mean",
    "max",
    "n_valid",
    "source_units",
    "target_units",
    "accumulation_mode",
    "accumulation_hours",
    "report",
]


@dataclass(frozen=True)
class LeadSelection:
    hour: float
    index: int | None = None


@dataclass(frozen=True)
class MemberSelection:
    index: int | None = None
    name: str | None = None
    label: str | None = None


def _fmt_hour(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def parse_lead_hours(
    *,
    lead_hours: str | None = None,
    lead_start: float | None = None,
    lead_end: float | None = None,
    lead_step: float | None = None,
) -> list[float]:
    """Parse explicit lead-hour arguments."""
    if lead_hours:
        values = [float(part.strip()) for part in lead_hours.split(",") if part.strip()]
        if not values:
            raise ValueError("--lead-hours did not contain any values")
        return values

    provided_range = [lead_start is not None, lead_end is not None, lead_step is not None]
    if any(provided_range):
        if not all(provided_range):
            raise ValueError("--lead-start, --lead-end, and --lead-step must be provided together")
        if lead_step == 0:
            raise ValueError("--lead-step cannot be zero")
        values: list[float] = []
        current = float(lead_start)
        end = float(lead_end)
        step = float(lead_step)
        if step > 0:
            while current <= end + 1e-9:
                values.append(current)
                current += step
        else:
            while current >= end - 1e-9:
                values.append(current)
                current += step
        return values

    raise ValueError("Provide --lead-hours, --lead-start/--lead-end/--lead-step, or --all-leads")


def _first_dim(names: list[str], dims: tuple[str, ...], label: str) -> str | None:
    for name in names:
        if name in dims:
            return name
    return None


def _coord_values(ds: Any, dim: str) -> np.ndarray:
    if dim in ds.coords:
        return np.asarray(ds.coords[dim].values)
    return np.arange(ds.sizes[dim])


def discover_leads_from_netcdf(input_path: str | Path, nc_variable: str) -> list[LeadSelection]:
    """Discover all lead hours available for a forecast variable."""
    xr = _xarray()
    with xr.open_dataset(input_path, decode_times=True) as ds:
        if nc_variable not in ds:
            raise ValueError(f"NetCDF variable not found: {nc_variable}")
        da = ds[nc_variable]
        roles = identify_dimension_roles(ds)
        dim = _first_dim(roles["lead"], da.dims, "lead")
        if dim is None:
            raise ValueError("No lead dimension was found; specify --lead-hours for non-forecast time axes")
        values = _coord_values(ds, dim)
        return [LeadSelection(hour=normalize_lead_to_hours(value), index=int(i)) for i, value in enumerate(values)]


def discover_members_from_netcdf(input_path: str | Path, nc_variable: str) -> list[MemberSelection]:
    """Discover member coordinate values for a forecast variable."""
    xr = _xarray()
    with xr.open_dataset(input_path, decode_times=True) as ds:
        if nc_variable not in ds:
            raise ValueError(f"NetCDF variable not found: {nc_variable}")
        da = ds[nc_variable]
        roles = identify_dimension_roles(ds)
        dim = _first_dim(roles["member"], da.dims, "member")
        if dim is None:
            return [MemberSelection(label=None)]
        values = _coord_values(ds, dim)
        members = []
        for i, value in enumerate(values):
            if isinstance(value, np.generic):
                value = value.item()
            name = str(value) if not isinstance(value, (int, np.integer)) else None
            members.append(MemberSelection(index=int(i), name=name, label=f"member_{i:03d}"))
        return members


def _discover_init_values(input_path: str | Path, nc_variable: str) -> list[str]:
    xr = _xarray()
    with xr.open_dataset(input_path, decode_times=True) as ds:
        if nc_variable not in ds:
            raise ValueError(f"NetCDF variable not found: {nc_variable}")
        da = ds[nc_variable]
        roles = identify_dimension_roles(ds)
        dim = _first_dim(roles["init_time"], da.dims, "init_time")
        if dim is None:
            return []
        values = _coord_values(ds, dim)
        return [_parse_datetime(value).isoformat() for value in values]


def _resolve_init(
    *,
    input_path: str | Path,
    nc_variable: str,
    init_time: str | None,
    init_index: int | None,
) -> tuple[str | None, int | None]:
    if init_time is not None and init_index is not None:
        raise ValueError("--init-time and --init-index cannot be combined")
    values = _discover_init_values(input_path, nc_variable)
    if init_time is not None:
        return _parse_datetime(init_time).isoformat(), None
    if init_index is not None:
        if values and (init_index < 0 or init_index >= len(values)):
            raise ValueError(f"--init-index {init_index} is outside available init times")
        return values[init_index] if values else None, init_index
    if len(values) > 1:
        raise ValueError("Multiple init times exist; specify --init-time or --init-index")
    if len(values) == 1:
        return values[0], 0
    raise ValueError("No init time was found; specify --init-time for forecast valid-time derivation")


def build_member_output_dir(
    output_dir: str | Path,
    member: MemberSelection,
    *,
    member_output_mode: str = "subdirs",
    member_prefix: str = "member",
) -> Path:
    """Return the output directory for a member in subdirectory mode."""
    base = Path(output_dir)
    if member_output_mode == "flat" or member.label is None:
        return base
    if member.index is not None:
        return base / f"{member_prefix}_{member.index:03d}"
    safe = str(member.name).replace("/", "_").replace("\\", "_").replace(" ", "_")
    return base / f"{member_prefix}_{safe}"


def _member_prefix(member: MemberSelection, member_prefix: str) -> str | None:
    if member.label is None:
        return None
    if member.index is not None:
        return f"{member_prefix}_{member.index:03d}"
    safe = str(member.name).replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"{member_prefix}_{safe}"


def _filename_for_valid_time(variable: str, valid_time: datetime) -> str:
    return rdhm_filename(variable, valid_time.date(), hour=valid_time.hour)


def _planned_output(
    *,
    output_dir: str | Path,
    variable: str,
    valid_time: datetime,
    member: MemberSelection,
    member_output_mode: str,
    member_prefix: str,
) -> Path:
    filename = _filename_for_valid_time(variable, valid_time)
    if member.label is None:
        return Path(output_dir) / filename
    if member_output_mode == "subdirs":
        return build_member_output_dir(
            output_dir,
            member,
            member_output_mode=member_output_mode,
            member_prefix=member_prefix,
        ) / filename
    prefix = _member_prefix(member, member_prefix)
    return Path(output_dir) / f"{prefix}_{filename}"


def _report_path(
    *,
    report_dir: str | Path | None,
    output_path: Path,
    member: MemberSelection,
    member_output_mode: str,
    member_prefix: str,
) -> Path | None:
    if report_dir is None:
        return None
    name = output_path.name
    stem = name[:-3] if name.endswith(".gz") else name
    report_name = f"{stem}.report.json"
    if member.label is not None and member_output_mode == "subdirs":
        return build_member_output_dir(
            report_dir,
            member,
            member_output_mode=member_output_mode,
            member_prefix=member_prefix,
        ) / report_name
    return Path(report_dir) / report_name


def _select_leads(
    *,
    input_path: str | Path,
    nc_variable: str,
    lead_hours: str | None,
    lead_indices: str | None,
    lead_start: float | None,
    lead_end: float | None,
    lead_step: float | None,
    all_leads: bool,
) -> list[LeadSelection]:
    explicit_count = sum(
        [
            bool(lead_hours),
            bool(lead_indices),
            any(value is not None for value in (lead_start, lead_end, lead_step)),
            all_leads,
        ]
    )
    if explicit_count != 1:
        raise ValueError("Specify exactly one lead selection mode")
    discovered = discover_leads_from_netcdf(input_path, nc_variable)
    if all_leads:
        return discovered
    if lead_indices:
        values = [int(part.strip()) for part in lead_indices.split(",") if part.strip()]
        by_index = {lead.index: lead for lead in discovered}
        missing = [value for value in values if value not in by_index]
        if missing:
            raise ValueError(f"lead indices not found: {missing}")
        return [by_index[value] for value in values]
    requested = parse_lead_hours(lead_hours=lead_hours, lead_start=lead_start, lead_end=lead_end, lead_step=lead_step)
    by_hour = {round(lead.hour, 6): lead for lead in discovered}
    selected = []
    for hour in requested:
        key = round(float(hour), 6)
        if key not in by_hour:
            raise ValueError(f"lead hour {hour!r} was not found in the NetCDF lead coordinate")
        selected.append(by_hour[key])
    return selected


def _select_members(
    *,
    input_path: str | Path,
    nc_variable: str,
    member_index: int | None,
    member_name: str | None,
    all_members: bool,
) -> list[MemberSelection]:
    if sum([member_index is not None, member_name is not None, all_members]) > 1:
        raise ValueError("Specify only one member selection mode")
    members = discover_members_from_netcdf(input_path, nc_variable)
    if len(members) == 1 and members[0].label is None:
        return members
    if all_members:
        return members
    if member_index is not None:
        matches = [member for member in members if member.index == member_index]
        if not matches:
            raise ValueError(f"member index {member_index} was not found")
        return matches
    if member_name is not None:
        matches = [member for member in members if str(member.name) == str(member_name)]
        if not matches:
            raise ValueError(f"member name {member_name!r} was not found")
        return matches
    if len(members) > 1:
        raise ValueError("Multiple members exist; specify --member-index, --member-name, or --all-members")
    return members


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def _row_from_result(
    *,
    input_path: str | Path,
    nc_variable: str,
    variable: str,
    init_time: str | None,
    init_index: int | None,
    lead: LeadSelection,
    valid_time: datetime,
    valid_time_source: str,
    member: MemberSelection,
    output: Path,
    source_units: str,
    target_units: str,
    accumulation_mode: str,
    accumulation_hours: float | None,
    report: Path | None,
    ok: bool,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details = details or {}
    return {
        "input": str(input_path),
        "nc_variable": nc_variable,
        "rdhm_variable": variable,
        "init_time": init_time,
        "init_index": init_index,
        "lead_hour": lead.hour,
        "lead_index": lead.index,
        "valid_time": valid_time.isoformat(),
        "valid_time_source": valid_time_source,
        "member_index": member.index,
        "member_name": member.name,
        "output": str(output),
        "ok": ok,
        "message": message,
        "min": details.get("min"),
        "mean": details.get("mean"),
        "max": details.get("max"),
        "n_valid": details.get("n_valid"),
        "source_units": source_units,
        "target_units": target_units,
        "accumulation_mode": accumulation_mode,
        "accumulation_hours": accumulation_hours,
        "report": str(report) if report else None,
    }


def batch_forecast_nc_to_xmrg(
    *,
    input_path: str | Path,
    nc_variable: str,
    variable: str,
    source_units: str,
    target_units: str,
    output_dir: str | Path,
    target_grid: TargetGrid,
    target_domain_source: str | None = None,
    init_time: str | None = None,
    init_index: int | None = None,
    lead_hours: str | None = None,
    lead_indices: str | None = None,
    lead_start: float | None = None,
    lead_end: float | None = None,
    lead_step: float | None = None,
    all_leads: bool = False,
    member_index: int | None = None,
    member_name: str | None = None,
    all_members: bool = False,
    member_output_mode: str = "subdirs",
    member_prefix: str = "member",
    accumulation_mode: str = "step",
    accumulation_hours: float | None = None,
    resampling: str = "bilinear",
    summary: str | Path | None = None,
    report_dir: str | Path | None = None,
    continue_on_error: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate a forecast folder by looping over NetCDF leads and members."""
    if accumulation_mode == "total-since-init":
        raise ValueError("total-since-init precipitation requires adjacent-lead differencing; not supported yet.")

    resolved_init_time, resolved_init_index = _resolve_init(
        input_path=input_path,
        nc_variable=nc_variable,
        init_time=init_time,
        init_index=init_index,
    )
    if resolved_init_time is None:
        raise ValueError("init_time is required to derive forecast valid times")
    init_dt = _parse_datetime(resolved_init_time)
    leads = _select_leads(
        input_path=input_path,
        nc_variable=nc_variable,
        lead_hours=lead_hours,
        lead_indices=lead_indices,
        lead_start=lead_start,
        lead_end=lead_end,
        lead_step=lead_step,
        all_leads=all_leads,
    )
    members = _select_members(
        input_path=input_path,
        nc_variable=nc_variable,
        member_index=member_index,
        member_name=member_name,
        all_members=all_members,
    )

    rows: list[dict[str, Any]] = []
    for member in members:
        for lead in leads:
            valid_time = init_dt + timedelta(hours=float(lead.hour))
            output = _planned_output(
                output_dir=output_dir,
                variable=variable,
                valid_time=valid_time,
                member=member,
                member_output_mode=member_output_mode,
                member_prefix=member_prefix,
            )
            report = _report_path(
                report_dir=report_dir,
                output_path=output,
                member=member,
                member_output_mode=member_output_mode,
                member_prefix=member_prefix,
            )
            if dry_run:
                rows.append(
                    _row_from_result(
                        input_path=input_path,
                        nc_variable=nc_variable,
                        variable=variable,
                        init_time=resolved_init_time,
                        init_index=resolved_init_index,
                        lead=lead,
                        valid_time=valid_time,
                        valid_time_source="derived_from_init_plus_lead",
                        member=member,
                        output=output,
                        source_units=source_units,
                        target_units=target_units,
                        accumulation_mode=accumulation_mode,
                        accumulation_hours=accumulation_hours,
                        report=report,
                        ok=True,
                        message="planned",
                    )
                )
                continue

            try:
                if member.label is not None and member_output_mode == "flat":
                    conversion_output = output
                    conversion_output_dir = None
                else:
                    conversion_output = None
                    conversion_output_dir = output.parent

                result = nc_to_xmrg(
                    input_path=input_path,
                    nc_variable=nc_variable,
                    output_xmrg=conversion_output,
                    output_dir=conversion_output_dir,
                    variable=variable,
                    init_index=resolved_init_index if init_time is None else None,
                    init_time=resolved_init_time if init_time is not None else None,
                    lead_hour=lead.hour,
                    member_index=member.index,
                    member_name=member.name if member.index is None else None,
                    source_units=source_units,
                    target_units=target_units,
                    target_grid=target_grid,
                    target_domain_source=target_domain_source,
                    resampling=resampling,
                    report=report,
                    accumulation_mode=accumulation_mode,
                    accumulation_hours=accumulation_hours,
                )
                rows.append(
                    _row_from_result(
                        input_path=input_path,
                        nc_variable=nc_variable,
                        variable=variable,
                        init_time=result.details.get("init_time") or resolved_init_time,
                        init_index=result.details.get("init_index") if result.details.get("init_index") is not None else resolved_init_index,
                        lead=lead,
                        valid_time=valid_time,
                        valid_time_source=str(result.details.get("valid_time_source") or "derived_from_init_plus_lead"),
                        member=member,
                        output=output,
                        source_units=source_units,
                        target_units=target_units,
                        accumulation_mode=accumulation_mode,
                        accumulation_hours=accumulation_hours,
                        report=report,
                        ok=result.ok,
                        message=result.message,
                        details=result.details,
                    )
                )
                if not result.ok and not continue_on_error:
                    break
            except Exception as exc:
                rows.append(
                    _row_from_result(
                        input_path=input_path,
                        nc_variable=nc_variable,
                        variable=variable,
                        init_time=resolved_init_time,
                        init_index=resolved_init_index,
                        lead=lead,
                        valid_time=valid_time,
                        valid_time_source="derived_from_init_plus_lead",
                        member=member,
                        output=output,
                        source_units=source_units,
                        target_units=target_units,
                        accumulation_mode=accumulation_mode,
                        accumulation_hours=accumulation_hours,
                        report=report,
                        ok=False,
                        message=str(exc),
                    )
                )
                if not continue_on_error:
                    if summary:
                        write_summary_csv(summary, rows)
                    raise

    if summary:
        write_summary_csv(summary, rows)

    n_ok = sum(1 for row in rows if row["ok"])
    n_failed = len(rows) - n_ok
    return {
        "ok": n_failed == 0 or continue_on_error,
        "input": str(input_path),
        "nc_variable": nc_variable,
        "variable": variable,
        "n_requested": len(rows),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "n_members": len(members),
        "output_dir": str(output_dir),
        "summary": str(summary) if summary else None,
        "rows": rows,
    }
