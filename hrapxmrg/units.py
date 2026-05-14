"""Unit conversion functions for RDHM XMRG forcing files.

Authoritative temperature conversion rule for this project:
    1. Start with raw Celsius (float32 from PRISM or other source).
    2. Convert to Fahrenheit: F = C * 9/5 + 32
    3. Multiply by 100 and round to nearest integer.
    4. Clamp and store as int16.

    Do NOT use Fahrenheit × 10 unless a new RDHM smoke test proves that
    convention is required for a different model setup.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #

INT16_MIN: int = np.iinfo(np.int16).min  # −32768
INT16_MAX: int = np.iinfo(np.int16).max  # +32767


def _to_f100_int16(
    fahrenheit: NDArray[np.floating],
    nodata_mask: NDArray[np.bool_] | None = None,
    nodata_stored: int = -9999,
) -> NDArray[np.int16]:
    """Convert a Fahrenheit array to int16 Fahrenheit × 100.

    Parameters
    ----------
    fahrenheit:
        Array of temperature values in Fahrenheit.
    nodata_mask:
        Boolean array; ``True`` where the cell is missing/nodata.
        Missing cells are filled with *nodata_stored*.
    nodata_stored:
        Sentinel int16 value written for nodata cells.
        Must be representable as int16 (−32768 to +32767).

    Returns
    -------
    NDArray[np.int16]
        Fahrenheit × 100 values clamped to int16 range.
    """
    f100 = np.round(fahrenheit * 100.0).astype(np.int64)
    # Clamp to int16 range before cast
    f100 = np.clip(f100, INT16_MIN, INT16_MAX)
    result = f100.astype(np.int16)

    if nodata_mask is not None:
        if nodata_stored < INT16_MIN or nodata_stored > INT16_MAX:
            raise ValueError(
                f"nodata_stored={nodata_stored} is outside int16 range "
                f"[{INT16_MIN}, {INT16_MAX}]."
            )
        result[nodata_mask] = np.int16(nodata_stored)

    return result


# ------------------------------------------------------------------ #
# Public conversion functions
# ------------------------------------------------------------------ #


def celsius_to_f100_int16(
    celsius: NDArray[np.floating],
    nodata_mask: NDArray[np.bool_] | None = None,
    nodata_stored: int = -9999,
) -> NDArray[np.int16]:
    """Convert Celsius to RDHM int16 Fahrenheit × 100 (F100).

    Conversion: ``F = C * 9/5 + 32``  then  ``int16(round(F * 100))``.

    Parameters
    ----------
    celsius:
        Input temperature in degrees Celsius (float array).
    nodata_mask:
        Boolean array; ``True`` where cells are missing/nodata.
    nodata_stored:
        int16 sentinel value used for nodata cells.

    Returns
    -------
    NDArray[np.int16]
        Fahrenheit × 100, stored as int16.

    Examples
    --------
    >>> import numpy as np
    >>> from hrapxmrg.units import celsius_to_f100_int16
    >>> celsius_to_f100_int16(np.array([0.0]))
    array([3200], dtype=int16)
    >>> celsius_to_f100_int16(np.array([-17.777778]))
    array([0], dtype=int16)   # ≈ 0 °F
    """
    c = np.asarray(celsius, dtype=np.float64)
    fahrenheit = c * (9.0 / 5.0) + 32.0
    return _to_f100_int16(fahrenheit, nodata_mask=nodata_mask, nodata_stored=nodata_stored)


def kelvin_to_f100_int16(
    kelvin: NDArray[np.floating],
    nodata_mask: NDArray[np.bool_] | None = None,
    nodata_stored: int = -9999,
) -> NDArray[np.int16]:
    """Convert Kelvin to RDHM int16 Fahrenheit × 100 (F100).

    Conversion: ``C = K − 273.15``, then ``F = C * 9/5 + 32``,
    then ``int16(round(F * 100))``.

    Parameters
    ----------
    kelvin:
        Input temperature in Kelvin (float array).
    nodata_mask:
        Boolean array; ``True`` where cells are missing/nodata.
    nodata_stored:
        int16 sentinel value used for nodata cells.

    Returns
    -------
    NDArray[np.int16]
        Fahrenheit × 100, stored as int16.
    """
    k = np.asarray(kelvin, dtype=np.float64)
    celsius = k - 273.15
    return celsius_to_f100_int16(celsius, nodata_mask=nodata_mask, nodata_stored=nodata_stored)


def fahrenheit_to_f100_int16(
    fahrenheit: NDArray[np.floating],
    nodata_mask: NDArray[np.bool_] | None = None,
    nodata_stored: int = -9999,
) -> NDArray[np.int16]:
    """Convert Fahrenheit to RDHM int16 Fahrenheit × 100 (F100).

    Conversion: ``int16(round(F * 100))``.

    Parameters
    ----------
    fahrenheit:
        Input temperature in Fahrenheit (float array).
    nodata_mask:
        Boolean array; ``True`` where cells are missing/nodata.
    nodata_stored:
        int16 sentinel value used for nodata cells.

    Returns
    -------
    NDArray[np.int16]
        Fahrenheit × 100, stored as int16.
    """
    f = np.asarray(fahrenheit, dtype=np.float64)
    return _to_f100_int16(f, nodata_mask=nodata_mask, nodata_stored=nodata_stored)


def f100_int16_to_fahrenheit(
    f100: NDArray[np.int16],
    nodata_stored: int = -9999,
) -> NDArray[np.float64]:
    """Convert RDHM int16 F100 stored values back to Fahrenheit.

    Parameters
    ----------
    f100:
        int16 array of Fahrenheit × 100 values as stored in XMRG.
    nodata_stored:
        Sentinel value indicating missing data; converted to NaN.

    Returns
    -------
    NDArray[np.float64]
        Fahrenheit values; nodata cells become ``NaN``.
    """
    arr = np.asarray(f100, dtype=np.float64)
    result = arr / 100.0
    result[np.asarray(f100) == nodata_stored] = np.nan
    return result


def f100_int16_to_celsius(
    f100: NDArray[np.int16],
    nodata_stored: int = -9999,
) -> NDArray[np.float64]:
    """Convert RDHM int16 F100 stored values back to Celsius.

    Parameters
    ----------
    f100:
        int16 array of Fahrenheit × 100 values as stored in XMRG.
    nodata_stored:
        Sentinel value indicating missing data; converted to NaN.

    Returns
    -------
    NDArray[np.float64]
        Celsius values; nodata cells become ``NaN``.
    """
    fahrenheit = f100_int16_to_fahrenheit(f100, nodata_stored=nodata_stored)
    return (fahrenheit - 32.0) * (5.0 / 9.0)
