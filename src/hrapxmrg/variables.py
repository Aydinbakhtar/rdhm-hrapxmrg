"""Variable conventions for RDHM forcing and static grids.

Important project rule:
For the current Upper Rio Grande / Rio Grande Headwaters RDHM setup,
tair/tmax/tmin are stored as int16 Fahrenheit x 100.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableSpec:
    name: str
    filename_prefix: str | None
    header_type: str
    dtype: str
    storage_scale: float
    rdhm_read_divisor: float
    missing_value: float
    physical_units: str
    valid_min: float
    valid_max: float
    description: str


VARIABLES: dict[str, VariableSpec] = {
    "tair": VariableSpec(
        name="tair",
        filename_prefix="tair",
        header_type="int32",
        dtype="int16",
        storage_scale=100.0,
        rdhm_read_divisor=100.0,
        missing_value=-999.0,
        physical_units="degF",
        valid_min=-80.0,
        valid_max=130.0,
        description="Mean air temperature, Fahrenheit x 100 int16 for this project",
    ),
    "tmax": VariableSpec(
        name="tmax",
        filename_prefix="tmax",
        header_type="int32",
        dtype="int16",
        storage_scale=100.0,
        rdhm_read_divisor=100.0,
        missing_value=-999.0,
        physical_units="degF",
        valid_min=-80.0,
        valid_max=140.0,
        description="Maximum air temperature, Fahrenheit x 100 int16 for this project",
    ),
    "tmin": VariableSpec(
        name="tmin",
        filename_prefix="tmin",
        header_type="int32",
        dtype="int16",
        storage_scale=100.0,
        rdhm_read_divisor=100.0,
        missing_value=-999.0,
        physical_units="degF",
        valid_min=-100.0,
        valid_max=120.0,
        description="Minimum air temperature, Fahrenheit x 100 int16 for this project",
    ),
    "prep": VariableSpec(
        name="prep",
        filename_prefix=None,
        header_type="int32",
        dtype="int16",
        storage_scale=1.0,
        rdhm_read_divisor=1.0,
        missing_value=-999.0,
        physical_units="mm/day",
        valid_min=0.0,
        valid_max=500.0,
        description="Daily precipitation; scale should be verified against working RDHM files",
    ),
    "snow_ALAT": VariableSpec(
        name="snow_ALAT",
        filename_prefix="snow_ALAT",
        header_type="int32",
        dtype="int16",
        storage_scale=100.0,
        rdhm_read_divisor=100.0,
        missing_value=-999.0,
        physical_units="degrees",
        valid_min=20.0,
        valid_max=60.0,
        description="Latitude per HRAP cell, latitude x 100",
    ),
}


def get_variable_spec(name: str) -> VariableSpec:
    key = name.lower()
    if key not in VARIABLES:
        valid = ", ".join(sorted(VARIABLES))
        raise KeyError(f"Unknown variable {name!r}. Valid variables: {valid}")
    return VARIABLES[key]
