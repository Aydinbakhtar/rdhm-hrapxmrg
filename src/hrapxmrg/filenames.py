"""RDHM filename helpers."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

PRECIP_VARIABLES = {"prep"}
TEMPERATURE_VARIABLES = {"tair", "tmax", "tmin"}


def parse_date(value: str | date | datetime) -> date | datetime:
    """Parse a YYYY-MM-DD value while preserving date/datetime inputs."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"date must be YYYY-MM-DD: {value!r}") from exc
    raise ValueError("date value must be a YYYY-MM-DD string, date, or datetime")


def _require_hour(hour: int | None) -> int:
    if type(hour) is not int or hour < 0 or hour > 23:
        raise ValueError("hour must be an integer from 0 through 23")
    return hour


def _datetime_for_hour(date_value: str | date | datetime, hour: int) -> datetime:
    hour = _require_hour(hour)
    parsed = parse_date(date_value)
    base_date = parsed.date() if isinstance(parsed, datetime) else parsed
    return datetime.combine(base_date, time(hour=hour))


def format_mmddyyyyhh(dt: datetime, prefix: str = "", *, suffix_gz: bool = True) -> str:
    """Return prefix + MMDDYYYYHH + z, optionally with .gz."""
    name = f"{prefix}{dt.month:02d}{dt.day:02d}{dt.year:04d}{dt.hour:02d}z"
    if suffix_gz:
        name += ".gz"
    return name


def precip_filename_for_end_time(end_time: datetime, *, suffix_gz: bool = True) -> str:
    """Return an RDHM precipitation filename for an accumulation end time."""
    if not isinstance(end_time, datetime):
        raise ValueError("end_time must be a datetime")
    return format_mmddyyyyhh(end_time, "xmrg", suffix_gz=suffix_gz)


def daily_precip_filename(physical_date: str | date | datetime, *, suffix_gz: bool = True) -> str:
    """Return the RDHM filename for a physical daily precipitation date."""
    parsed = parse_date(physical_date)
    base_date = parsed.date() if isinstance(parsed, datetime) else parsed
    end_time = datetime.combine(base_date + timedelta(days=1), time(hour=0))
    return precip_filename_for_end_time(end_time, suffix_gz=suffix_gz)


def temperature_filename(variable: str, valid_time: datetime, *, suffix_gz: bool = True) -> str:
    """Return an RDHM temperature filename for a valid time."""
    variable = variable.lower()
    if variable not in TEMPERATURE_VARIABLES:
        raise ValueError("temperature filename only supports tair, tmax, and tmin")
    if not isinstance(valid_time, datetime):
        raise ValueError("valid_time must be a datetime")
    return format_mmddyyyyhh(valid_time, variable, suffix_gz=suffix_gz)


def rdhm_filename(
    variable: str,
    date_value: str | date | datetime,
    *,
    hour: int | None = None,
    daily: bool = False,
    suffix_gz: bool = True,
) -> str:
    """Return an RDHM forcing filename for precipitation or temperature."""
    variable = variable.lower()

    if variable == "prep":
        if daily and hour is not None:
            raise ValueError("prep cannot use both daily=True and hour")
        if daily:
            return daily_precip_filename(date_value, suffix_gz=suffix_gz)
        if hour is None:
            raise ValueError("prep requires daily=True or an hour end time")
        return precip_filename_for_end_time(
            _datetime_for_hour(date_value, hour),
            suffix_gz=suffix_gz,
        )

    if variable in TEMPERATURE_VARIABLES:
        if daily:
            raise ValueError("daily=True is only valid for prep")
        if hour is None:
            raise ValueError(f"{variable} requires hour")
        return temperature_filename(
            variable,
            _datetime_for_hour(date_value, hour),
            suffix_gz=suffix_gz,
        )

    raise ValueError(f"unsupported RDHM filename variable: {variable}")


def mmddyyyy(d: date | datetime) -> str:
    return f"{d.month:02d}{d.day:02d}{d.year:04d}"


def daily_filename(variable: str, d: date | datetime, precip_prefix: str | None = None) -> str:
    """Return daily 12z filename.

    Temperature always uses tair/tmax/tmin prefix.
    Precipitation prefix should be confirmed from the working RDHM folders.
    """
    if variable in {"tair", "tmax", "tmin"}:
        return f"{variable}{mmddyyyy(d)}12z.gz"
    if variable in {"prep", "ppt", "precip"}:
        prefix = "" if precip_prefix is None else precip_prefix
        return f"{prefix}{mmddyyyy(d)}12z.gz"
    raise ValueError(f"unknown variable for filename: {variable}")


def hourly_temperature_filename(variable: str, dt: datetime) -> str:
    return temperature_filename(variable, dt)


def safe_symlink_or_copy(src: str | Path, dst: str | Path, copy: bool = False) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        import shutil
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)
