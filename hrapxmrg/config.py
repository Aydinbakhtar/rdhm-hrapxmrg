"""HRAP domain configuration for RDHM preprocessing.

The default domain covers the Upper Rio Grande Headwaters project.
Additional domains can be defined using the :class:`HRAPDomain` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HRAPDomain:
    """Represents an HRAP grid domain.

    Parameters
    ----------
    xor:
        HRAP x-coordinate of the south-west (lower-left) corner column.
    yor:
        HRAP y-coordinate of the south-west (lower-left) corner row.
    maxx:
        Number of columns (width) in the HRAP grid.
    maxy:
        Number of rows (height) in the HRAP grid.
    cellsize:
        Cell size in HRAP units (almost always 1).
    nodata:
        Missing-data sentinel value stored in XMRG files.
    name:
        Optional human-readable name for this domain.
    """

    xor: int
    yor: int
    maxx: int
    maxy: int
    cellsize: float = 1.0
    nodata: float = -999.0
    name: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #

    @property
    def ncols(self) -> int:
        """Alias for :attr:`maxx` (number of columns)."""
        return self.maxx

    @property
    def nrows(self) -> int:
        """Alias for :attr:`maxy` (number of rows)."""
        return self.maxy

    @property
    def xllcorner(self) -> float:
        """Lower-left x in HRAP units (same as :attr:`xor`)."""
        return float(self.xor)

    @property
    def yllcorner(self) -> float:
        """Lower-left y in HRAP units (same as :attr:`yor`)."""
        return float(self.yor)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def matches(self, other: "HRAPDomain") -> bool:
        """Return True if both domains share the same grid geometry."""
        return (
            self.xor == other.xor
            and self.yor == other.yor
            and self.maxx == other.maxx
            and self.maxy == other.maxy
        )

    def __str__(self) -> str:
        label = f" ({self.name})" if self.name else ""
        return (
            f"HRAPDomain{label}: "
            f"xor={self.xor}, yor={self.yor}, "
            f"maxx={self.maxx}, maxy={self.maxy}, "
            f"cellsize={self.cellsize}, nodata={self.nodata}"
        )


# ------------------------------------------------------------------ #
# Default domain – Upper Rio Grande Headwaters
# ------------------------------------------------------------------ #

#: Default HRAP domain used for the Upper Rio Grande Headwaters project.
#:
#: Grid parameters match the existing HRAP ASCII files at::
#:
#:   /data/aydin/project/RDHM/rdhm-prep_05/forcing/obs/hrap_asc/ppt/0101200012z.asc
#:
#: which contain:
#:   ncols 64, nrows 87, xllcorner 341, yllcorner 313, cellsize 1, NODATA -999
DEFAULT_DOMAIN = HRAPDomain(
    xor=341,
    yor=313,
    maxx=64,
    maxy=87,
    cellsize=1.0,
    nodata=-999.0,
    name="Upper Rio Grande Headwaters",
)
