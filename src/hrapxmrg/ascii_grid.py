"""ESRI/HRAP ASCII grid reader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hrap import TargetGrid


@dataclass(frozen=True)
class AsciiGrid:
    array: np.ndarray
    ncols: int
    nrows: int
    xllcorner: float
    yllcorner: float
    cellsize: float
    nodata: float

    def to_target_grid(self) -> TargetGrid:
        return TargetGrid(
            xor=int(round(self.xllcorner)),
            yor=int(round(self.yllcorner)),
            maxx=self.ncols,
            maxy=self.nrows,
            cellsize=self.cellsize,
            nodata=self.nodata,
        )


def read_ascii_grid(path: str | Path) -> AsciiGrid:
    """Read ESRI ASCII grid with a 6-line header."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        header_lines = [next(f).strip() for _ in range(6)]
        data = np.loadtxt(f, dtype=float)

    header = {}
    for line in header_lines:
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed ASCII header line in {path}: {line!r}")
        header[parts[0].lower()] = float(parts[1])

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])

    if data.shape != (nrows, ncols):
        raise ValueError(f"{path}: data shape {data.shape} does not match header {(nrows, ncols)}")

    nodata = header.get("nodata_value", header.get("nodata", -999.0))

    return AsciiGrid(
        array=data,
        ncols=ncols,
        nrows=nrows,
        xllcorner=header["xllcorner"],
        yllcorner=header["yllcorner"],
        cellsize=header["cellsize"],
        nodata=nodata,
    )
