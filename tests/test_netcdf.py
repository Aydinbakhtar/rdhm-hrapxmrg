from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hrapxmrg.hrap import TargetGrid
from hrapxmrg.netcdf import inspect_netcdf, nc_to_xmrg
from hrapxmrg.xmrg import read_xmrg


def _xarray():
    return pytest.importorskip("xarray")


def _lat_lon_covering_grid(grid: TargetGrid, *, nlat: int = 24, nlon: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    from rasterio.warp import transform

    from hrapxmrg.hrap import HRAP_CRS_PROJ4
    from hrapxmrg.regrid import target_transform_from_hrap_grid

    tr = target_transform_from_hrap_grid(grid)
    xs = [tr.c, tr.c + tr.a * grid.maxx]
    ys = [tr.f, tr.f + tr.e * grid.maxy]
    lons, lats = transform(
        CRS.from_proj4(HRAP_CRS_PROJ4),
        "EPSG:4326",
        [xs[0], xs[1], xs[0], xs[1]],
        [ys[0], ys[0], ys[1], ys[1]],
    )
    lon_min, lon_max = min(lons) - 0.2, max(lons) + 0.2
    lat_min, lat_max = min(lats) - 0.2, max(lats) + 0.2
    lon = np.linspace(lon_min, lon_max, nlon, dtype=np.float64)
    lat = np.linspace(lat_max, lat_min, nlat, dtype=np.float64)
    return lat, lon


def _write_netcdf(path: Path, ds) -> None:
    try:
        ds.to_netcdf(path)
    except Exception as exc:
        pytest.skip(f"xarray NetCDF writing is unavailable: {exc}")


def _simple_dataset(path: Path, variable: str, values: np.ndarray) -> tuple[TargetGrid, Path]:
    xr = _xarray()
    grid = TargetGrid(xor=341, yor=313, maxx=8, maxy=6)
    lat, lon = _lat_lon_covering_grid(grid)
    ds = xr.Dataset(
        {variable: (("time", "lat", "lon"), values.astype(np.float32))},
        coords={
            "time": np.array(["2025-01-01T01:00:00", "2025-01-01T02:00:00"], dtype="datetime64[ns]"),
            "lat": lat,
            "lon": lon,
        },
    )
    _write_netcdf(path, ds)
    return grid, path


def _physical(path: Path) -> np.ndarray:
    stored, _meta = read_xmrg(path)
    return np.flipud(stored) / 100.0


def test_nc_info_runs_on_simple_synthetic_netcdf(tmp_path: Path):
    xr = _xarray()
    lat = np.array([40.0, 39.9])
    lon = np.array([-105.1, -105.0, -104.9])
    ds = xr.Dataset(
        {"APCP": (("time", "lat", "lon"), np.ones((1, 2, 3), dtype=np.float32))},
        coords={"time": np.array(["2025-01-01T01:00:00"], dtype="datetime64[ns]"), "lat": lat, "lon": lon},
    )
    path = tmp_path / "simple.nc"
    _write_netcdf(path, ds)

    info = inspect_netcdf(path)

    assert info["input"] == str(path)
    assert info["xarray_open_ok"] is True
    assert "APCP" in info["data_vars"]
    assert info["dims"]["time"] == 1
    assert "lat" in info["coords"]


def test_nc_to_xmrg_precip_mm_time_index(tmp_path: Path):
    values = np.ones((2, 24, 24), dtype=np.float32)
    grid, path = _simple_dataset(tmp_path / "apcp.nc", "APCP", values)
    output_dir = tmp_path / "out"

    result = nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        time_index=0,
        variable="prep",
        source_units="mm",
        target_units="mm",
        target_grid=grid,
        output_dir=output_dir,
        date_value="2025-01-01",
        hour=1,
        resampling="nearest",
    )

    output = output_dir / "xmrg0101202501z.gz"
    assert result.ok
    assert output.exists()
    assert np.nanmean(_physical(output)) == pytest.approx(1.0, abs=0.01)


def test_nc_to_xmrg_temperature_kelvin_to_f(tmp_path: Path):
    values = np.full((2, 24, 24), 273.15, dtype=np.float32)
    grid, path = _simple_dataset(tmp_path / "t2m.nc", "T2M", values)
    output_dir = tmp_path / "out"

    result = nc_to_xmrg(
        input_path=path,
        nc_variable="T2M",
        time_index=0,
        variable="tair",
        source_units="K",
        target_units="F",
        target_grid=grid,
        output_dir=output_dir,
        valid_time="2025-01-01T12:00:00",
        resampling="nearest",
    )

    output = output_dir / "tair0101202512z.gz"
    assert result.ok
    assert output.exists()
    assert np.nanmean(_physical(output)) == pytest.approx(32.0, abs=0.02)


def _forecast_dataset(path: Path, *, member_names: bool = False) -> tuple[TargetGrid, Path]:
    xr = _xarray()
    grid = TargetGrid(xor=341, yor=313, maxx=8, maxy=6)
    lat, lon = _lat_lon_covering_grid(grid)
    leads = np.array([1, 6, 12], dtype=np.int32)
    members = np.array(["m01", "m02"]) if member_names else np.array([0, 1], dtype=np.int32)
    data = np.zeros((1, 3, 2, len(lat), len(lon)), dtype=np.float32)
    for lead_i, lead in enumerate(leads):
        for member_i in range(len(members)):
            data[0, lead_i, member_i, :, :] = float(lead * 10 + member_i)
    ds = xr.Dataset(
        {"APCP": (("init_time", "lead", "member", "lat", "lon"), data)},
        coords={
            "init_time": np.array(["2025-01-01T00:00:00"], dtype="datetime64[ns]"),
            "lead": leads,
            "member": members,
            "lat": lat,
            "lon": lon,
        },
    )
    _write_netcdf(path, ds)
    return grid, path


def test_forecast_member_lead_selection(tmp_path: Path):
    grid, path = _forecast_dataset(tmp_path / "forecast.nc")
    output_dir = tmp_path / "out"

    result = nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        init_time="2025-01-01T00:00:00",
        lead_hour=6,
        member_index=1,
        variable="prep",
        source_units="mm",
        target_units="mm",
        target_grid=grid,
        output_dir=output_dir,
        valid_time="2025-01-01T06:00:00",
        resampling="nearest",
    )

    assert result.ok
    assert result.details["lead_hour"] == 6
    assert result.details["member_index"] == 1
    assert np.nanmean(_physical(output_dir / "xmrg0101202506z.gz")) == pytest.approx(61.0, abs=0.02)


def test_unselected_forecast_dims_raise(tmp_path: Path):
    grid, path = _forecast_dataset(tmp_path / "forecast.nc")

    with pytest.raises(ValueError, match="remaining dimensions"):
        nc_to_xmrg(
            input_path=path,
            nc_variable="APCP",
            variable="prep",
            source_units="mm",
            target_units="mm",
            target_grid=grid,
            output_dir=tmp_path / "out",
            valid_time="2025-01-01T06:00:00",
        )


def test_member_name_selection(tmp_path: Path):
    grid, path = _forecast_dataset(tmp_path / "forecast_named.nc", member_names=True)
    output_dir = tmp_path / "out"

    result = nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        init_time="2025-01-01T00:00:00",
        lead_hour=6,
        member_name="m02",
        variable="prep",
        source_units="mm",
        target_units="mm",
        target_grid=grid,
        output_dir=output_dir,
        valid_time="2025-01-01T06:00:00",
        resampling="nearest",
    )

    assert result.ok
    assert result.details["member_name"] == "m02"
    assert np.nanmean(_physical(output_dir / "xmrg0101202506z.gz")) == pytest.approx(61.0, abs=0.02)


def test_valid_time_filename(tmp_path: Path):
    values = np.ones((2, 24, 24), dtype=np.float32)
    grid, path = _simple_dataset(tmp_path / "apcp.nc", "APCP", values)
    prep_dir = tmp_path / "prep"
    tair_dir = tmp_path / "tair"

    nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        time_index=0,
        variable="prep",
        source_units="mm",
        target_units="mm",
        target_grid=grid,
        output_dir=prep_dir,
        valid_time="2025-01-01T06:00:00",
        resampling="nearest",
    )
    nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        time_index=0,
        variable="tair",
        source_units="C",
        target_units="F",
        target_grid=grid,
        output_dir=tair_dir,
        valid_time="2025-01-01T12:00:00",
        resampling="nearest",
    )

    assert (prep_dir / "xmrg0101202506z.gz").exists()
    assert (tair_dir / "tair0101202512z.gz").exists()


def test_invalid_rate_units_raise(tmp_path: Path):
    values = np.ones((2, 24, 24), dtype=np.float32)
    grid, path = _simple_dataset(tmp_path / "apcp.nc", "APCP", values)

    with pytest.raises(ValueError, match="Rate units require explicit accumulation duration"):
        nc_to_xmrg(
            input_path=path,
            nc_variable="APCP",
            time_index=0,
            variable="prep",
            source_units="kg m-2 s-1",
            target_units="mm",
            target_grid=grid,
            output_dir=tmp_path / "out",
            valid_time="2025-01-01T06:00:00",
        )


def test_valid_time_derived_from_init_and_lead(tmp_path: Path):
    grid, path = _forecast_dataset(tmp_path / "forecast.nc")
    output_dir = tmp_path / "out"

    result = nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        init_time="2025-01-01T00:00:00",
        lead_hour=6,
        member_index=0,
        variable="prep",
        source_units="mm",
        target_units="mm",
        target_grid=grid,
        output_dir=output_dir,
        resampling="nearest",
    )

    assert result.ok
    assert result.details["valid_time_source"] == "derived_from_init_plus_lead"
    assert (output_dir / "xmrg0101202506z.gz").exists()


def test_precip_rate_conversion_with_accumulation_hours(tmp_path: Path):
    values = np.full((2, 24, 24), 1.0 / 3600.0, dtype=np.float32)
    grid, path = _simple_dataset(tmp_path / "rate.nc", "APCP", values)
    output_dir = tmp_path / "out"

    result = nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        time_index=0,
        variable="prep",
        source_units="kg m-2 s-1",
        target_units="mm",
        target_grid=grid,
        output_dir=output_dir,
        valid_time="2025-01-01T01:00:00",
        accumulation_mode="rate",
        accumulation_hours=1.0,
        resampling="nearest",
    )

    assert result.ok
    assert np.nanmean(_physical(output_dir / "xmrg0101202501z.gz")) == pytest.approx(1.0, abs=0.01)


def test_total_since_init_mode_raises(tmp_path: Path):
    values = np.ones((2, 24, 24), dtype=np.float32)
    grid, path = _simple_dataset(tmp_path / "apcp.nc", "APCP", values)

    with pytest.raises(ValueError, match="total-since-init precipitation requires adjacent lead differencing"):
        nc_to_xmrg(
            input_path=path,
            nc_variable="APCP",
            time_index=0,
            variable="prep",
            source_units="mm",
            target_units="mm",
            target_grid=grid,
            output_dir=tmp_path / "out",
            valid_time="2025-01-01T01:00:00",
            accumulation_mode="total-since-init",
        )
