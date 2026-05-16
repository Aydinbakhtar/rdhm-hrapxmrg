from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from hrapxmrg.hrap import TargetGrid
from hrapxmrg.temporal import (
    daily_ppt_to_hourly_arrays,
    daily_raster_ppt_to_hourly_xmrg,
    daily_temp_to_hourly_tair_xmrg,
    hourly_precip_end_times,
    hourly_temperature_from_tmin_tmax,
    hourly_temperature_valid_times,
)
from hrapxmrg.xmrg import read_xmrg


def _write_hrap_geotiff(path: Path, grid: TargetGrid, array: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS

    from hrapxmrg.hrap import HRAP_CRS_PROJ4
    from hrapxmrg.regrid import target_transform_from_hrap_grid

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.maxy,
        width=grid.maxx,
        count=1,
        dtype="float32",
        crs=CRS.from_proj4(HRAP_CRS_PROJ4),
        transform=target_transform_from_hrap_grid(grid),
        nodata=-999.0,
    ) as dst:
        dst.write(array.astype("float32"), 1)


def test_hourly_precip_end_times():
    times = hourly_precip_end_times("2025-01-01")

    assert len(times) == 24
    assert times[0] == datetime(2025, 1, 1, 1)
    assert times[-1] == datetime(2025, 1, 2, 0)


def test_daily_ppt_to_hourly_arrays_uniform_split():
    daily = np.full((2, 3), 24.0, dtype=np.float32)
    hourly = daily_ppt_to_hourly_arrays(daily)

    assert len(hourly) == 24
    assert all(np.allclose(arr, 1.0) for arr in hourly)
    assert np.allclose(np.sum(np.stack(hourly), axis=0), daily)


def test_daily_ppt_to_hourly_pipeline(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    daily = np.full((grid.maxy, grid.maxx), 24.0, dtype=np.float32)
    src = tmp_path / "daily_ppt.tif"
    output_dir = tmp_path / "out"
    summary = tmp_path / "summary.csv"
    _write_hrap_geotiff(src, grid, daily)

    result = daily_raster_ppt_to_hourly_xmrg(
        input_raster=src,
        output_dir=output_dir,
        target_grid=grid,
        date_value="2025-01-01",
        method="uniform",
        resampling="nearest",
        summary=summary,
    )

    assert result["ok"] is True
    assert result["n_files"] == 24
    assert (output_dir / "xmrg0101202501z.gz").exists()
    assert (output_dir / "xmrg0102202500z.gz").exists()
    assert summary.exists()

    stored, _meta = read_xmrg(output_dir / "xmrg0101202501z.gz")
    assert np.allclose(np.flipud(stored) / 100.0, 1.0, atol=0.01)

    total = np.zeros((grid.maxy, grid.maxx), dtype=np.float64)
    for output in sorted(output_dir.glob("xmrg*.gz")):
        stored, _meta = read_xmrg(output)
        total += np.flipud(stored) / 100.0
    assert np.allclose(total, 24.0, atol=0.05)


def test_hourly_temperature_valid_times():
    times = hourly_temperature_valid_times("2025-01-01")

    assert len(times) == 24
    assert times[0] == datetime(2025, 1, 1, 0)
    assert times[-1] == datetime(2025, 1, 1, 23)


def test_hourly_temperature_from_tmin_tmax_range():
    tmin = np.full((2, 2), 0.0, dtype=np.float32)
    tmax = np.full((2, 2), 10.0, dtype=np.float32)

    hourly = hourly_temperature_from_tmin_tmax(tmin, tmax)
    stacked = np.stack(hourly)

    assert len(hourly) == 24
    assert float(np.nanmin(stacked)) == pytest.approx(0.0, abs=0.01)
    assert float(np.nanmax(stacked)) == pytest.approx(10.0, abs=0.01)
    assert np.all(stacked >= 0.0)
    assert np.all(stacked <= 10.0)


def test_daily_temp_to_hourly_tair_pipeline(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    tmin = np.full((grid.maxy, grid.maxx), 0.0, dtype=np.float32)
    tmax = np.full((grid.maxy, grid.maxx), 10.0, dtype=np.float32)
    tmin_path = tmp_path / "tmin.tif"
    tmax_path = tmp_path / "tmax.tif"
    output_dir = tmp_path / "out"
    _write_hrap_geotiff(tmin_path, grid, tmin)
    _write_hrap_geotiff(tmax_path, grid, tmax)

    result = daily_temp_to_hourly_tair_xmrg(
        tmin_raster=tmin_path,
        tmax_raster=tmax_path,
        output_dir=output_dir,
        target_grid=grid,
        date_value="2025-01-01",
        method="sinusoidal",
        resampling="nearest",
    )

    assert result["ok"] is True
    assert result["n_files"] == 24
    assert (output_dir / "tair0101202500z.gz").exists()
    assert (output_dir / "tair0101202523z.gz").exists()

    mins = []
    maxes = []
    for output in sorted(output_dir.glob("tair*.gz")):
        stored, _meta = read_xmrg(output)
        physical = np.flipud(stored) / 100.0
        mins.append(float(np.nanmin(physical)))
        maxes.append(float(np.nanmax(physical)))

    assert min(mins) == pytest.approx(32.0, abs=0.01)
    assert max(maxes) == pytest.approx(50.0, abs=0.01)

    stored, _meta = read_xmrg(output_dir / "tair0101202515z.gz")
    assert np.nanmax(np.flipud(stored)) == pytest.approx(5000.0, abs=1.0)


def test_temporal_nodata_behavior():
    daily = np.array([[24.0, -999.0]], dtype=np.float32)
    hourly_ppt = daily_ppt_to_hourly_arrays(daily)
    assert hourly_ppt[0][0, 0] == pytest.approx(1.0)
    assert hourly_ppt[0][0, 1] == pytest.approx(-999.0)

    tmin = np.array([[0.0, -999.0]], dtype=np.float32)
    tmax = np.array([[10.0, 10.0]], dtype=np.float32)
    hourly_temp = hourly_temperature_from_tmin_tmax(tmin, tmax)
    assert hourly_temp[0][0, 0] >= 0.0
    assert hourly_temp[0][0, 1] == pytest.approx(-999.0)
