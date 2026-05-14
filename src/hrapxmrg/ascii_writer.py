from __future__ import annotations

from pathlib import Path
import numpy as np

from .hrap import TargetGrid


def write_ascii_grid(
    path: str | Path,
    array: np.ndarray,
    target_grid: TargetGrid,
    *,
    nodata: float = -999.0,
    fmt: str = "%.2f",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"ncols         {target_grid.maxx}\n")
        f.write(f"nrows         {target_grid.maxy}\n")
        f.write(f"xllcorner     {target_grid.xor}\n")
        f.write(f"yllcorner     {target_grid.yor}\n")
        f.write(f"cellsize      {target_grid.cellsize:g}\n")
        f.write(f"NODATA_value  {nodata:g}\n")

        for row in array:
            f.write(" ".join(fmt % v for v in row) + "\n")