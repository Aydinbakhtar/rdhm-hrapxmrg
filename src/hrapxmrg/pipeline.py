"""High-level conversion pipelines."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ascii_grid import read_ascii_grid
from .variables import get_variable_spec
from .xmrg import write_xmrg
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
