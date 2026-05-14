"""High-level conversion pipelines."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from .ascii_grid import read_ascii_grid
from .ascii_writer import write_ascii_grid
from .hrap import TargetGrid
from .variables import get_variable_spec
from .xmrg import read_xmrg, write_xmrg
from .validate import validate_xmrg_file, ValidationResult



def ascii_to_xmrg(
    input_ascii: str | Path,
    output_xmrg: str | Path,
    variable: str,
    *,
    header_type: str | None = None,
    dtype: str | None = None,
    scale: float | None = None,
    orientation: str = "as-is",
    secondary_header: bool = False,
) -> ValidationResult:
    """Convert HRAP ASCII to XMRG and validate readback."""
    spec = get_variable_spec(variable)
    asc = read_ascii_grid(input_ascii)
    target = asc.to_target_grid()

    header_type = header_type or spec.header_type
    dtype = dtype or spec.dtype
    scale = spec.storage_scale if scale is None else scale

    arr = np.where(asc.array == asc.nodata, spec.missing_value, asc.array)

    write_xmrg(
        output_xmrg,
        arr,
        xor=target.xor,
        yor=target.yor,
        header_type=header_type,
        dtype=dtype,
        scale=scale,
        missing=spec.missing_value,
        secondary_header=secondary_header,
        orientation=orientation,
    )

    return validate_xmrg_file(output_xmrg, variable=variable, target_grid=target)


def xmrg_to_ascii(
    input_xmrg: str | Path,
    output_ascii: str | Path,
    variable: str,
    *,
    orientation: str = "flipud",
    physical: bool = True,
) -> None:
    spec = get_variable_spec(variable)
    grid, meta = read_xmrg(input_xmrg)

    arr = grid.copy()

    if physical and meta.dtype == "int16" and spec.rdhm_read_divisor != 1.0:
        valid = arr > -900
        arr[valid] = arr[valid] / spec.rdhm_read_divisor

    if orientation == "flipud":
        arr = np.flipud(arr)
    elif orientation != "as-is":
        raise ValueError("orientation must be 'as-is' or 'flipud'")

    target = TargetGrid(
        xor=int(meta.xor),
        yor=int(meta.yor),
        maxx=meta.maxx,
        maxy=meta.maxy,
        cellsize=1.0,
        nodata=spec.missing_value,
    )

    write_ascii_grid(output_ascii, arr, target, nodata=spec.missing_value)


def format_xmrg_report(
    input_xmrg: str | Path,
    variable: str | None = None,
    preview_rows: int = 5,
    preview_cols: int = 8,
) -> str:
    grid, meta = read_xmrg(input_xmrg)

    lines = []
    lines.append("XMRG REPORT")
    lines.append("=" * 60)
    lines.append(f"path: {input_xmrg}")
    lines.append(f"endian: {meta.endian}")
    lines.append(f"header_type: {meta.header_type}")
    lines.append(f"dtype: {meta.dtype}")
    lines.append(f"xor: {meta.xor}")
    lines.append(f"yor: {meta.yor}")
    lines.append(f"maxx: {meta.maxx}")
    lines.append(f"maxy: {meta.maxy}")
    lines.append(f"secondary_headers: {len(meta.secondary_headers_hex)}")

    valid = grid[(grid > -900) & np.isfinite(grid)]
    lines.append("")
    lines.append("Stored values")
    lines.append(f"min: {valid.min()}")
    lines.append(f"mean: {valid.mean()}")
    lines.append(f"max: {valid.max()}")

    if variable is not None:
        spec = get_variable_spec(variable)
        if meta.dtype == "int16" and spec.rdhm_read_divisor != 1.0:
            physical = valid / spec.rdhm_read_divisor
        else:
            physical = valid
        lines.append("")
        lines.append(f"Physical values ({spec.physical_units})")
        lines.append(f"min: {physical.min()}")
        lines.append(f"mean: {physical.mean()}")
        lines.append(f"max: {physical.max()}")

    lines.append("")
    lines.append("Corners stored TL/TR/BL/BR")
    lines.append(f"{grid[0,0]}, {grid[0,-1]}, {grid[-1,0]}, {grid[-1,-1]}")

    lines.append("")
    lines.append(f"Preview stored values [{preview_rows} x {preview_cols}]")
    preview = grid[:preview_rows, :preview_cols]
    for row in preview:
        lines.append(" ".join(f"{v:10.2f}" for v in row))

    return "\n".join(lines)




