"""HRAP target grid definitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, degrees, radians, sin

HRAP_CRS_PROJ4 = (
    "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 "
    "+x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs"
)

HRAP_MESH_M = 4762.5
HRAP_MESH_KM = 4.7625
XPOL = 401.0
YPOL = 1601.0
RE_KM = 6371.2
LAT_TRUE = 60.0


@dataclass(frozen=True)
class TargetGrid:
    """An RDHM HRAP target grid."""

    xor: int
    yor: int
    maxx: int
    maxy: int
    cellsize: float = 1.0
    nodata: float = -999.0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.maxy, self.maxx)

    @property
    def x_max_hrap(self) -> int:
        return self.xor + self.maxx - 1

    @property
    def y_max_hrap(self) -> int:
        return self.yor + self.maxy - 1

    @classmethod
    def rio_grande_headwaters(cls) -> "TargetGrid":
        return cls(xor=341, yor=313, maxx=64, maxy=87, cellsize=1.0, nodata=-999.0)


def hrap_to_lat(i: float, j: float) -> float:
    """Latitude from HRAP i/j index."""
    x = i - XPOL
    y = j - YPOL
    rsq = x * x + y * y
    gi = RE_KM * (1.0 + sin(radians(LAT_TRUE))) / HRAP_MESH_KM
    return degrees(asin((gi * gi - rsq) / (gi * gi + rsq)))


def hrap_stereo_m_to_index(xster: float, yster: float) -> tuple[float, float]:
    """Convert HRAP polar-stereographic meters to HRAP index coordinates."""
    hrapx = (xster + XPOL * HRAP_MESH_M) / HRAP_MESH_M
    hrapy = (yster + YPOL * HRAP_MESH_M) / HRAP_MESH_M
    return hrapx, hrapy
