from pathlib import Path

import numpy as np
import pytest

from hrapxmrg.hrap import TargetGrid
from hrapxmrg.pipeline import raster_to_xmrg
from hrapxmrg.xmrg import read_xmrg


def test_raster_to_xmrg_from_aligned_geotiff(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS

    from hrapxmrg.hrap import HRAP_CRS_PROJ4
    from hrapxmrg.regrid import target_transform_from_hrap_grid

    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)

    src_path = tmp_path / "source.tif"
    out_path = tmp_path / "out.gz"

    transform = target_transform_from_hrap_grid(grid)
    crs = CRS.from_proj4(HRAP_CRS_PROJ4)

    arr = np.zeros((grid.maxy, grid.maxx), dtype=np.float32)
    for r in range(grid.maxy):
        for c in range(grid.maxx):
            arr[r, c] = 0.23 + r * 0.10 + c * 0.10

    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        height=grid.maxy,
        width=grid.maxx,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-999.0,
    ) as dst:
        dst.write(arr, 1)

    result = raster_to_xmrg(
        src_path,
        out_path,
        "prep",
        grid,
        band=1,
        resampling="nearest",
    )

    assert result.ok

    stored, meta = read_xmrg(out_path)

    assert stored.shape == (87, 64)
    assert meta.xor == 341
    assert meta.yor == 313
    assert meta.maxx == 64
    assert meta.maxy == 87

    # XMRG is written flipud by default, so flip back before physical comparison.
    physical_back = np.flipud(stored) / 100.0

    assert np.allclose(physical_back, arr, atol=0.01)


def test_raster_to_xmrg_celsius_to_fahrenheit(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS

    from hrapxmrg.hrap import HRAP_CRS_PROJ4
    from hrapxmrg.regrid import target_transform_from_hrap_grid

    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)

    src_path = tmp_path / "temp_c.tif"
    out_path = tmp_path / "tair.gz"

    transform = target_transform_from_hrap_grid(grid)
    crs = CRS.from_proj4(HRAP_CRS_PROJ4)

    arr_c = np.full((grid.maxy, grid.maxx), 0.0, dtype=np.float32)

    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        height=grid.maxy,
        width=grid.maxx,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-999.0,
    ) as dst:
        dst.write(arr_c, 1)

    result = raster_to_xmrg(
        src_path,
        out_path,
        "tair",
        grid,
        band=1,
        resampling="nearest",
        source_units="C",
        target_units="F",
    )

    assert result.ok

    stored, meta = read_xmrg(out_path)
    physical_back_f = np.flipud(stored) / 100.0

    assert np.allclose(physical_back_f, 32.0, atol=0.01)
