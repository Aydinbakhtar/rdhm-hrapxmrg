"""Unit conversion utilities."""

from __future__ import annotations

import numpy as np


def celsius_to_fahrenheit(c: np.ndarray) -> np.ndarray:
    return c * 9.0 / 5.0 + 32.0


def kelvin_to_fahrenheit(k: np.ndarray) -> np.ndarray:
    return (k - 273.15) * 9.0 / 5.0 + 32.0


def fahrenheit_to_i16_scaled(
    f: np.ndarray,
    scale: float = 100.0,
    missing: float = -999.0,
) -> np.ndarray:
    """Convert Fahrenheit grid to scaled int16.

    For this RDHM project, scale=100 for tair/tmax/tmin.
    """
    out = np.where(f <= -900, missing, np.rint(f * scale))
    if np.nanmin(out) < np.iinfo(np.int16).min or np.nanmax(out) > np.iinfo(np.int16).max:
        raise OverflowError("Scaled temperature exceeds int16 range")
    return out.astype(np.int16)


def celsius_to_f100_i16(c: np.ndarray, missing: float = -999.0) -> np.ndarray:
    f = celsius_to_fahrenheit(c)
    return fahrenheit_to_i16_scaled(f, scale=100.0, missing=missing)
