"""HRAP grid and projection utilities.

The HRAP (Hydrologic Rainfall Analysis Project) coordinate system uses a
polar-stereographic projection true at 60 °N with a standard meridian of
105 °W (central longitude of the contiguous USA).

Key constants (from NWS/OHD documentation):
    - Earth radius  : 6371.2 km
    - Standard lat  : 60 °N
    - Central lon   : 105 °W
    - Grid spacing  : 4762.5 m at 60 °N (1 HRAP unit ≈ 4762.5 m)
    - Mesh length   : 4762.5 m (ξ in OHD notation)
    - Origin offset : The HRAP coordinate (0, 0) corresponds to a specific
                      point in the polar-stereographic plane; the offset
                      constants below encode that.

This module provides:
    - :func:`hrap_to_latlon`: Convert HRAP (x, y) to geographic (lon, lat).
    - :func:`latlon_to_hrap`: Convert geographic (lon, lat) to HRAP (x, y).
    - :func:`domain_extent_latlon`: Return geographic bounds of an HRAPDomain.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .config import HRAPDomain

# ------------------------------------------------------------------ #
# HRAP polar-stereographic constants
# ------------------------------------------------------------------ #

#: Earth radius used in HRAP coordinate system (km).
EARTH_RADIUS_KM: float = 6371.2

#: Grid spacing at 60 °N (km), i.e. one HRAP unit.
HRAP_MESH_KM: float = 4.7625  # 4762.5 m

#: Standard (true-scale) latitude (degrees).
HRAP_TRUE_LAT_DEG: float = 60.0

#: Standard meridian (degrees, positive east).
HRAP_STDLON_DEG: float = -105.0

# Pre-computed: HRAP origin in polar-stereographic plane (km from pole)
# The HRAP grid origin (0, 0) is defined by OHD such that:
#   polster_x(0, 0) = -401 * HRAP_MESH_KM
#   polster_y(0, 0) = -1601 * HRAP_MESH_KM
# See: hrapx = (xster + 401 * 4762.5) / 4762.5
_ORIGIN_OFFSET_X: float = 401.0  # HRAP units
_ORIGIN_OFFSET_Y: float = 1601.0  # HRAP units


# ------------------------------------------------------------------ #
# Core coordinate transforms
# ------------------------------------------------------------------ #

def _to_radians(deg: float) -> float:
    return deg * math.pi / 180.0


def latlon_to_hrap(
    lat: float | NDArray[np.floating],
    lon: float | NDArray[np.floating],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert geographic coordinates to HRAP grid coordinates.

    Parameters
    ----------
    lat:
        Latitude in decimal degrees (positive north).
    lon:
        Longitude in decimal degrees (positive east, so western longitudes
        are negative).

    Returns
    -------
    hrap_x, hrap_y:
        HRAP column and row coordinates (fractional).

    Notes
    -----
    Uses the polar-stereographic projection described in OHD documentation.
    The formula is::

        r = Earth_radius * (1 + sin(60°)) / (1 + sin(lat))
        x_ps = r * cos(lat) * sin(lon - stdlon)
        y_ps = r * cos(lat) * cos(lon - stdlon)     [pointing toward 105°W]
        hrap_x = x_ps / mesh + origin_x_offset
        hrap_y = y_ps / mesh + origin_y_offset
    """
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    lat_r = np.deg2rad(lat)
    dlon = np.deg2rad(lon - HRAP_STDLON_DEG)

    true_lat_r = _to_radians(HRAP_TRUE_LAT_DEG)
    scale = EARTH_RADIUS_KM * (1.0 + math.sin(true_lat_r))  # km

    r = scale / (1.0 + np.sin(lat_r))  # km per unit at this latitude
    cos_lat = np.cos(lat_r)

    # Polar-stereographic metre offsets from the pole
    x_ps = r * cos_lat * np.sin(dlon)   # east component (km)
    y_ps = -r * cos_lat * np.cos(dlon)  # north component toward 105°W (km)

    hrap_x = x_ps / HRAP_MESH_KM + _ORIGIN_OFFSET_X
    hrap_y = y_ps / HRAP_MESH_KM + _ORIGIN_OFFSET_Y

    return hrap_x, hrap_y


def hrap_to_latlon(
    hrap_x: float | NDArray[np.floating],
    hrap_y: float | NDArray[np.floating],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert HRAP grid coordinates to geographic coordinates.

    Parameters
    ----------
    hrap_x:
        HRAP column coordinate (fractional).
    hrap_y:
        HRAP row coordinate (fractional).

    Returns
    -------
    lat, lon:
        Latitude and longitude in decimal degrees.
    """
    hrap_x = np.asarray(hrap_x, dtype=np.float64)
    hrap_y = np.asarray(hrap_y, dtype=np.float64)

    # Convert back to polar-stereographic km offsets from origin
    x_ps = (hrap_x - _ORIGIN_OFFSET_X) * HRAP_MESH_KM
    y_ps = (hrap_y - _ORIGIN_OFFSET_Y) * HRAP_MESH_KM

    true_lat_r = _to_radians(HRAP_TRUE_LAT_DEG)
    scale = EARTH_RADIUS_KM * (1.0 + math.sin(true_lat_r))

    rho = np.sqrt(x_ps**2 + y_ps**2)  # km from pole (in plane)

    lat_r = np.arcsin((scale**2 - rho**2) / (scale**2 + rho**2))
    lon_r = np.arctan2(x_ps, -y_ps) + _to_radians(HRAP_STDLON_DEG)

    return np.rad2deg(lat_r), np.rad2deg(lon_r)


def domain_extent_latlon(domain: HRAPDomain) -> dict[str, float]:
    """Return the approximate geographic bounding box of an HRAP domain.

    The returned dictionary contains keys ``lat_min``, ``lat_max``,
    ``lon_min``, and ``lon_max`` (all in decimal degrees).

    Parameters
    ----------
    domain:
        HRAP domain to evaluate.

    Returns
    -------
    dict
        Geographic bounding box of the domain corners.
    """
    # Sample all four corners (HRAP cell centres at corners of the domain).
    corners_x = [domain.xor, domain.xor + domain.maxx - 1]
    corners_y = [domain.yor, domain.yor + domain.maxy - 1]

    lats = []
    lons = []
    for cx in corners_x:
        for cy in corners_y:
            lat, lon = hrap_to_latlon(float(cx), float(cy))
            lats.append(float(lat))
            lons.append(float(lon))

    return {
        "lat_min": min(lats),
        "lat_max": max(lats),
        "lon_min": min(lons),
        "lon_max": max(lons),
    }


def hrap_filename_timestamp(date_str: str, hour: int = 12) -> str:
    """Build the RDHM XMRG timestamp portion of a filename.

    Parameters
    ----------
    date_str:
        Date string in ``YYYY-MM-DD`` format (or ``YYYYMMDD``).
    hour:
        UTC hour (0–23).  Use 12 for daily files (``12z``).

    Returns
    -------
    str
        Timestamp portion such as ``0101201512`` (``MMDDYYYYHH``).
    """
    date_str = date_str.replace("-", "")
    if len(date_str) != 8:
        raise ValueError(
            f"Expected date in YYYYMMDD or YYYY-MM-DD format, got {date_str!r}"
        )
    yyyy = date_str[:4]
    mm = date_str[4:6]
    dd = date_str[6:8]
    return f"{mm}{dd}{yyyy}{hour:02d}"


def build_xmrg_filename(
    variable: str,
    date_str: str,
    hour: int = 12,
    suffix: str = ".gz",
) -> str:
    """Build an XMRG filename following RDHM naming conventions.

    Parameters
    ----------
    variable:
        Variable name from the registry (e.g. ``"tair"``).  If the variable
        has an empty ``file_prefix`` (e.g. precipitation), no prefix is added.
    date_str:
        Date string in ``YYYY-MM-DD`` or ``YYYYMMDD`` format.
    hour:
        UTC hour (0–23).  Use 12 for the standard daily 12z file.
    suffix:
        File extension (default ``.gz``).

    Returns
    -------
    str
        XMRG filename such as ``tair0101201512z.gz``.

    Examples
    --------
    >>> build_xmrg_filename("tair", "2015-01-01", hour=12)
    'tair0101201512z.gz'
    >>> build_xmrg_filename("tmin", "2015-01-01", hour=0)
    'tmin0101201500z.gz'
    >>> build_xmrg_filename("prep", "2000-01-01", hour=12)
    '0101200012z.gz'
    """
    from .variable_registry import get_variable

    spec = get_variable(variable)
    ts = hrap_filename_timestamp(date_str, hour)
    prefix = spec.file_prefix  # empty string for precipitation
    return f"{prefix}{ts}z{suffix}"
