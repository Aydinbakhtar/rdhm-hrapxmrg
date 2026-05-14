"""HRAP ASCII grid reader.

Reads ESRI ASCII raster files that use HRAP grid coordinates, as produced
by tools such as ``asctoxmrg`` (the inverse direction) or directly from
RDHM preprocessing scripts.

Expected header format::

    ncols         64
    nrows         87
    xllcorner     341.0
    yllcorner     313.0
    cellsize      1.0
    NODATA_value  -999.0

Followed by *nrows* lines of *ncols* space-separated values.  The first
data line corresponds to the **northernmost** row (standard ESRI ASCII
convention).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..config import HRAPDomain


class HRAPAsciiReader:
    """Read an HRAP ASCII raster file.

    Parameters
    ----------
    path:
        Path to the ``.asc`` file.

    Attributes
    ----------
    domain:
        :class:`~hrapxmrg.config.HRAPDomain` parsed from the header.
    data:
        2-D float64 array of shape ``(nrows, ncols)`` with the northernmost
        row at index 0 (ESRI ASCII convention).
    nodata:
        Nodata sentinel value from the header.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self.domain: Optional[HRAPDomain] = None
        self.data: Optional[NDArray[np.float64]] = None
        self.nodata: float = -999.0
        self._read()

    # ------------------------------------------------------------------ #

    def _read(self) -> None:
        header: dict[str, str] = {}
        data_lines: list[str] = []

        # Keys recognised in the header (case-insensitive)
        _HEADER_KEYS = {"ncols", "nrows", "xllcorner", "yllcorner",
                        "cellsize", "nodata_value"}
        _REQUIRED = {"ncols", "nrows", "xllcorner", "yllcorner", "cellsize"}

        with open(self._path, "r") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                # Try to parse as a header key-value pair
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[0].lower() in _HEADER_KEYS:
                    header[parts[0].lower()] = parts[1].strip()
                else:
                    # First non-header line starts the data block
                    data_lines.append(line)
                    break

            # Read remaining data lines
            for raw_line in fh:
                stripped = raw_line.strip()
                if stripped:
                    data_lines.append(stripped)

        missing = _REQUIRED - header.keys()
        if missing:
            raise ValueError(
                f"HRAP ASCII header missing required keys: {sorted(missing)}"
            )

        ncols = int(header["ncols"])
        nrows = int(header["nrows"])
        xllcorner = float(header["xllcorner"])
        yllcorner = float(header["yllcorner"])
        cellsize = float(header["cellsize"])
        nodata = float(header.get("nodata_value", "-999"))

        self.nodata = nodata
        self.domain = HRAPDomain(
            xor=int(round(xllcorner)),
            yor=int(round(yllcorner)),
            maxx=ncols,
            maxy=nrows,
            cellsize=cellsize,
            nodata=nodata,
        )

        # Parse data rows
        if len(data_lines) != nrows:
            raise ValueError(
                f"Expected {nrows} data rows, found {len(data_lines)} "
                f"in {self._path}."
            )

        rows: list[NDArray[np.float64]] = []
        for row_idx, line in enumerate(data_lines):
            values = np.fromstring(line, dtype=np.float64, sep=" ")
            if values.size != ncols:
                raise ValueError(
                    f"Row {row_idx}: expected {ncols} values, "
                    f"found {values.size} in {self._path}."
                )
            rows.append(values)

        # Stack into (nrows, ncols) – northernmost row at index 0
        self.data = np.stack(rows, axis=0)

    # ------------------------------------------------------------------ #

    @property
    def ncols(self) -> int:
        """Number of columns."""
        return self.domain.maxx

    @property
    def nrows(self) -> int:
        """Number of rows."""
        return self.domain.maxy

    def data_south_first(self) -> NDArray[np.float64]:
        """Return :attr:`data` with the southernmost row at index 0.

        This matches the XMRG storage convention.
        """
        return self.data[::-1, :]

    def nodata_mask(self) -> NDArray[np.bool_]:
        """Return a boolean mask; ``True`` where cells equal :attr:`nodata`."""
        return self.data == self.nodata

    def stats(self) -> dict:
        """Return basic statistics of the valid (non-nodata) cells."""
        valid = self.data[~self.nodata_mask()]
        return {
            "min": float(valid.min()) if valid.size else float("nan"),
            "max": float(valid.max()) if valid.size else float("nan"),
            "mean": float(valid.mean()) if valid.size else float("nan"),
            "nodata_count": int(self.nodata_mask().sum()),
            "count": int(self.data.size),
        }
