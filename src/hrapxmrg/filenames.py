"""RDHM filename helpers."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path


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
    if variable not in {"tair", "tmax", "tmin"}:
        raise ValueError("hourly filename helper only supports temperature variables")
    return f"{variable}{dt.month:02d}{dt.day:02d}{dt.year:04d}{dt.hour:02d}z.gz"


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
