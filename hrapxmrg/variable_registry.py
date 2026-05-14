"""Variable registry for RDHM forcing variables.

Defines the authoritative rules for each variable:
- Storage data type
- Scale convention
- File-name prefix
- Physical range (for validation)
- Unit conversion to apply before writing

Temperature rule (authoritative for this project):
    PRISM temperature is raw Celsius float32.
    Convert: F = C * 9/5 + 32
    Store:   int16 = round(F * 100)
    Prefix:  tair, tmax, tmin

    Do NOT use Fahrenheit × 10 for this project unless a new RDHM smoke
    test proves that convention for another setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


StorageType = Literal["int16", "float32"]
ScaleConvention = Literal["F100_int16", "mm_raw", "none"]


@dataclass(frozen=True)
class VariableSpec:
    """Specification for one RDHM forcing variable.

    Attributes
    ----------
    name:
        Canonical variable name used internally (e.g. ``"tair"``).
    file_prefix:
        Required filename prefix for RDHM recognition (e.g. ``"tair"``).
        Empty string means no prefix required (precipitation).
    storage_type:
        NumPy dtype stored in the XMRG binary (``"int16"`` or ``"float32"``).
    scale_convention:
        Human-readable description of how physical values are encoded.
        ``"F100_int16"``: Fahrenheit × 100, stored as int16.
        ``"mm_raw"``:     Millimetres, no scaling (float or integer).
        ``"none"``:       No prescribed scaling.
    source_units:
        Physical units of the raw source data (e.g. ``"C"``, ``"mm/day"``).
    valid_stored_min:
        Minimum plausible *stored* value (as written in XMRG).
        Used for validation; ``None`` disables the lower-bound check.
    valid_stored_max:
        Maximum plausible *stored* value (as written in XMRG).
        Used for validation; ``None`` disables the upper-bound check.
    description:
        Free-form human-readable description.
    """

    name: str
    file_prefix: str
    storage_type: StorageType
    scale_convention: ScaleConvention
    source_units: str
    valid_stored_min: Optional[float]
    valid_stored_max: Optional[float]
    description: str = ""

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #

    def physical_from_stored(self, stored_value: float) -> float:
        """Convert a stored XMRG value back to a physical value.

        For ``F100_int16`` the stored value is Fahrenheit × 100; dividing by
        100 gives Fahrenheit.

        For ``mm_raw`` the stored value is already in mm.
        """
        if self.scale_convention == "F100_int16":
            return stored_value / 100.0
        return float(stored_value)

    @property
    def valid_physical_min(self) -> Optional[float]:
        """Minimum plausible *physical* value (in output units)."""
        if self.valid_stored_min is None:
            return None
        return self.physical_from_stored(self.valid_stored_min)

    @property
    def valid_physical_max(self) -> Optional[float]:
        """Maximum plausible *physical* value (in output units)."""
        if self.valid_stored_max is None:
            return None
        return self.physical_from_stored(self.valid_stored_max)


# ------------------------------------------------------------------ #
# Registry entries
# ------------------------------------------------------------------ #

#: Mean daily air temperature.
#:
#: Authoritative conversion rule:
#:   C (PRISM float32) → F = C * 9/5 + 32 → store round(F * 100) as int16
#:
#: File-naming rule: ``tairMMDDYYYY12z.gz`` (daily) or
#: ``tairMMDDYYYYHHz.gz`` (hourly).
TAIR = VariableSpec(
    name="tair",
    file_prefix="tair",
    storage_type="int16",
    scale_convention="F100_int16",
    source_units="C",
    # Stored as Fahrenheit × 100; plausible range for Upper Rio Grande:
    # roughly −40 °F (−4000) to +120 °F (+12000).
    valid_stored_min=-4000,
    valid_stored_max=12000,
    description=(
        "Mean daily air temperature. "
        "Stored as int16 Fahrenheit × 100 (F100). "
        "Source: PRISM tmean Celsius → F = C*9/5+32 → int16(round(F*100))."
    ),
)

#: Maximum daily air temperature.
TMAX = VariableSpec(
    name="tmax",
    file_prefix="tmax",
    storage_type="int16",
    scale_convention="F100_int16",
    source_units="C",
    valid_stored_min=-4000,
    valid_stored_max=15000,
    description=(
        "Maximum daily air temperature. "
        "Stored as int16 Fahrenheit × 100 (F100). "
        "Source: PRISM tmax Celsius → F = C*9/5+32 → int16(round(F*100))."
    ),
)

#: Minimum daily air temperature.
TMIN = VariableSpec(
    name="tmin",
    file_prefix="tmin",
    storage_type="int16",
    scale_convention="F100_int16",
    source_units="C",
    valid_stored_min=-6000,
    valid_stored_max=12000,
    description=(
        "Minimum daily air temperature. "
        "Stored as int16 Fahrenheit × 100 (F100). "
        "Source: PRISM tmin Celsius → F = C*9/5+32 → int16(round(F*100))."
    ),
)

#: Daily precipitation.
#:
#: Precipitation filename prefix and scale should be verified against
#: existing working RDHM files before full production.  The common
#: RDHM XMRG precipitation file name does **not** use a variable prefix
#: (e.g. ``0101200012z.gz``).  Set ``file_prefix=""`` to indicate no prefix.
PREP = VariableSpec(
    name="prep",
    file_prefix="",
    storage_type="int16",
    scale_convention="mm_raw",
    source_units="mm/day",
    # Precipitation should be ≥ 0; realistic daily totals < 1000 mm.
    valid_stored_min=0,
    valid_stored_max=100000,
    description=(
        "Daily precipitation in mm. "
        "Prefix and scale must be verified against existing working RDHM "
        "files before production use. "
        "Source: PRISM ppt (mm/day, float32)."
    ),
)


# ------------------------------------------------------------------ #
# Registry lookup
# ------------------------------------------------------------------ #

#: Map from canonical variable name to :class:`VariableSpec`.
REGISTRY: dict[str, VariableSpec] = {
    "tair": TAIR,
    "tmax": TMAX,
    "tmin": TMIN,
    "prep": PREP,
}


def get_variable(name: str) -> VariableSpec:
    """Return the :class:`VariableSpec` for *name*.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    try:
        return REGISTRY[name.lower()]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(
            f"Unknown variable {name!r}. Known variables: {known}"
        ) from None
