from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from hrapxmrg.hrap import TargetGrid
from hrapxmrg.netcdf_batch import batch_forecast_nc_to_xmrg
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
    lon = np.linspace(min(lons) - 0.2, max(lons) + 0.2, nlon, dtype=np.float64)
    lat = np.linspace(max(lats) + 0.2, min(lats) - 0.2, nlat, dtype=np.float64)
    return lat, lon


def _write_netcdf(path: Path, ds) -> None:
    try:
        ds.to_netcdf(path)
    except Exception as exc:
        pytest.skip(f"xarray NetCDF writing is unavailable: {exc}")


def _forecast_dataset(
    path: Path,
    *,
    variable: str = "APCP",
    leads: tuple[int, ...] = (1, 2, 3),
    members: tuple[int, ...] = (0,),
    base_value: float = 0.0,
) -> tuple[TargetGrid, Path]:
    xr = _xarray()
    grid = TargetGrid(xor=341, yor=313, maxx=8, maxy=6)
    lat, lon = _lat_lon_covering_grid(grid)
    init = np.array(["2026-05-18T00:00:00"], dtype="datetime64[ns]")
    lead_values = np.array(leads, dtype=np.int32)
    member_values = np.array(members, dtype=np.int32)
    data = np.zeros((1, len(leads), len(members), len(lat), len(lon)), dtype=np.float32)
    for lead_i, lead in enumerate(leads):
        for member_i, member in enumerate(members):
            data[0, lead_i, member_i, :, :] = base_value + float(lead) + float(member * 10)
    ds = xr.Dataset(
        {variable: (("init_time", "lead", "member", "lat", "lon"), data)},
        coords={
            "init_time": init,
            "lead": lead_values,
            "member": member_values,
            "lat": lat,
            "lon": lon,
        },
    )
    _write_netcdf(path, ds)
    return grid, path


def _cumulative_dataset(
    path: Path,
    *,
    leads: tuple[int, ...],
    cumulative_by_member: dict[int, list[float]],
) -> tuple[TargetGrid, Path]:
    xr = _xarray()
    grid = TargetGrid(xor=341, yor=313, maxx=8, maxy=6)
    lat, lon = _lat_lon_covering_grid(grid)
    members = tuple(cumulative_by_member)
    data = np.zeros((1, len(leads), len(members), len(lat), len(lon)), dtype=np.float32)
    for member_i, member in enumerate(members):
        values = cumulative_by_member[member]
        assert len(values) == len(leads)
        for lead_i, value in enumerate(values):
            data[0, lead_i, member_i, :, :] = float(value)
    ds = xr.Dataset(
        {"APCP": (("init_time", "lead", "member", "lat", "lon"), data)},
        coords={
            "init_time": np.array(["2026-05-18T00:00:00"], dtype="datetime64[ns]"),
            "lead": np.array(leads, dtype=np.int32),
            "member": np.array(members, dtype=np.int32),
            "lat": lat,
            "lon": lon,
        },
    )
    _write_netcdf(path, ds)
    return grid, path


def _physical(path: Path) -> np.ndarray:
    stored, _meta = read_xmrg(path)
    return np.flipud(stored) / 100.0


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_batch_forecast_apcp_all_leads(tmp_path: Path):
    grid, path = _forecast_dataset(tmp_path / "forecast.nc", leads=(1, 2, 3), members=(0,))
    output_dir = tmp_path / "out"
    summary = tmp_path / "summary.csv"

    result = batch_forecast_nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        variable="prep",
        source_units="mm",
        target_units="mm",
        output_dir=output_dir,
        init_time="2026-05-18T00:00:00",
        all_leads=True,
        target_grid=grid,
        summary=summary,
        resampling="nearest",
    )

    assert result["ok"]
    assert result["n_requested"] == 3
    assert (output_dir / "xmrg0518202601z.gz").exists()
    assert (output_dir / "xmrg0518202602z.gz").exists()
    assert (output_dir / "xmrg0518202603z.gz").exists()
    assert not (output_dir / "member_000").exists()
    rows = _read_summary(summary)
    assert len(rows) == 3
    assert rows[0]["member_index"] == "0"
    assert rows[0]["output"].endswith("out/xmrg0518202601z.gz")


def test_batch_forecast_all_members_subdirs(tmp_path: Path):
    grid, path = _forecast_dataset(tmp_path / "forecast.nc", leads=(1, 2), members=(0, 1))
    output_dir = tmp_path / "out"

    result = batch_forecast_nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        variable="prep",
        source_units="mm",
        target_units="mm",
        output_dir=output_dir,
        init_time="2026-05-18T00:00:00",
        all_leads=True,
        all_members=True,
        target_grid=grid,
        resampling="nearest",
    )

    assert result["ok"]
    assert result["n_requested"] == 4
    assert (output_dir / "member_000" / "xmrg0518202601z.gz").exists()
    assert (output_dir / "member_000" / "xmrg0518202602z.gz").exists()
    assert (output_dir / "member_001" / "xmrg0518202601z.gz").exists()
    assert (output_dir / "member_001" / "xmrg0518202602z.gz").exists()


def test_batch_forecast_missing_member_selector_raises(tmp_path: Path):
    grid, path = _forecast_dataset(tmp_path / "forecast.nc", leads=(1, 2), members=(0, 1))

    with pytest.raises(ValueError, match="Multiple members exist"):
        batch_forecast_nc_to_xmrg(
            input_path=path,
            nc_variable="APCP",
            variable="prep",
            source_units="mm",
            target_units="mm",
            output_dir=tmp_path / "out",
            init_time="2026-05-18T00:00:00",
            all_leads=True,
            target_grid=grid,
        )


def test_batch_forecast_temperature_kelvin_to_f(tmp_path: Path):
    grid, path = _forecast_dataset(tmp_path / "t2m.nc", variable="T2M", leads=(1, 2), members=(0,), base_value=273.15)
    output_dir = tmp_path / "out"

    result = batch_forecast_nc_to_xmrg(
        input_path=path,
        nc_variable="T2M",
        variable="tair",
        source_units="K",
        target_units="F",
        output_dir=output_dir,
        init_time="2026-05-18T00:00:00",
        all_leads=True,
        target_grid=grid,
        resampling="nearest",
    )

    assert result["ok"]
    assert (output_dir / "tair0518202601z.gz").exists()
    assert not (output_dir / "member_000").exists()
    assert np.nanmean(_physical(output_dir / "tair0518202601z.gz")) == pytest.approx(33.8, abs=0.05)


def test_batch_forecast_rate_precipitation(tmp_path: Path):
    grid, path = _forecast_dataset(
        tmp_path / "rate.nc",
        variable="APCP",
        leads=(1,),
        members=(0,),
        base_value=(1.0 / 3600.0) - 1.0,
    )
    output_dir = tmp_path / "out"

    result = batch_forecast_nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        variable="prep",
        source_units="kg m-2 s-1",
        target_units="mm",
        output_dir=output_dir,
        init_time="2026-05-18T00:00:00",
        all_leads=True,
        target_grid=grid,
        accumulation_mode="rate",
        accumulation_hours=1.0,
        resampling="nearest",
    )

    assert result["ok"]
    assert not (output_dir / "member_000").exists()
    assert np.nanmean(_physical(output_dir / "xmrg0518202601z.gz")) == pytest.approx(1.0, abs=0.02)


def test_batch_forecast_total_since_init_rejects_temperature(tmp_path: Path):
    grid, path = _forecast_dataset(tmp_path / "tair.nc", variable="T2M", leads=(1,), members=(0,))

    with pytest.raises(ValueError, match="total-since-init accumulation mode is only valid"):
        batch_forecast_nc_to_xmrg(
            input_path=path,
            nc_variable="T2M",
            variable="tair",
            source_units="K",
            target_units="F",
            output_dir=tmp_path / "out",
            init_time="2026-05-18T00:00:00",
            all_leads=True,
            target_grid=grid,
            accumulation_mode="total-since-init",
        )


def test_total_since_init_basic_hourly_differencing(tmp_path: Path):
    grid, path = _cumulative_dataset(tmp_path / "cumulative.nc", leads=(1, 2, 3), cumulative_by_member={0: [1, 3, 6]})
    output_dir = tmp_path / "out"
    summary = tmp_path / "summary.csv"

    result = batch_forecast_nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        variable="prep",
        source_units="mm",
        target_units="mm",
        output_dir=output_dir,
        init_time="2026-05-18T00:00:00",
        all_leads=True,
        target_grid=grid,
        accumulation_mode="total-since-init",
        summary=summary,
        resampling="nearest",
    )

    assert result["ok"]
    assert np.nanmean(_physical(output_dir / "xmrg0518202601z.gz")) == pytest.approx(1.0, abs=0.02)
    assert np.nanmean(_physical(output_dir / "xmrg0518202602z.gz")) == pytest.approx(2.0, abs=0.02)
    assert np.nanmean(_physical(output_dir / "xmrg0518202603z.gz")) == pytest.approx(3.0, abs=0.02)
    rows = _read_summary(summary)
    assert rows[1]["previous_lead_hour"] == "1.0"
    assert rows[1]["diff_method"] == "current_minus_previous_lead"


def test_total_since_init_uses_lead_zero_for_first_selected_lead(tmp_path: Path):
    grid, path = _cumulative_dataset(tmp_path / "cumulative.nc", leads=(0, 1, 2), cumulative_by_member={0: [0, 5, 8]})
    output_dir = tmp_path / "out"

    result = batch_forecast_nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        variable="prep",
        source_units="mm",
        target_units="mm",
        output_dir=output_dir,
        init_time="2026-05-18T00:00:00",
        lead_hours="1,2",
        target_grid=grid,
        accumulation_mode="total-since-init",
        resampling="nearest",
    )

    assert result["ok"]
    assert np.nanmean(_physical(output_dir / "xmrg0518202601z.gz")) == pytest.approx(5.0, abs=0.02)
    assert np.nanmean(_physical(output_dir / "xmrg0518202602z.gz")) == pytest.approx(3.0, abs=0.02)


def test_total_since_init_negative_diff_raises(tmp_path: Path):
    grid, path = _cumulative_dataset(tmp_path / "negative.nc", leads=(1, 2), cumulative_by_member={0: [5, 4]})

    with pytest.raises(ValueError, match="difference is negative beyond tolerance"):
        batch_forecast_nc_to_xmrg(
            input_path=path,
            nc_variable="APCP",
            variable="prep",
            source_units="mm",
            target_units="mm",
            output_dir=tmp_path / "out",
            init_time="2026-05-18T00:00:00",
            all_leads=True,
            target_grid=grid,
            accumulation_mode="total-since-init",
            resampling="nearest",
        )


def test_total_since_init_negative_diff_with_allow_clamps_to_zero(tmp_path: Path):
    grid, path = _cumulative_dataset(tmp_path / "tiny_negative.nc", leads=(1, 2), cumulative_by_member={0: [5, 4.999999]})
    output_dir = tmp_path / "out"

    result = batch_forecast_nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        variable="prep",
        source_units="mm",
        target_units="mm",
        output_dir=output_dir,
        init_time="2026-05-18T00:00:00",
        all_leads=True,
        target_grid=grid,
        accumulation_mode="total-since-init",
        allow_negative_diff=True,
        negative_diff_tolerance=1e-6,
        resampling="nearest",
    )

    assert result["ok"]
    assert np.nanmin(_physical(output_dir / "xmrg0518202602z.gz")) >= 0.0


def test_total_since_init_all_members_differenced_independently(tmp_path: Path):
    grid, path = _cumulative_dataset(
        tmp_path / "members.nc",
        leads=(1, 2),
        cumulative_by_member={0: [1, 3], 1: [10, 13]},
    )
    output_dir = tmp_path / "out"

    result = batch_forecast_nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        variable="prep",
        source_units="mm",
        target_units="mm",
        output_dir=output_dir,
        init_time="2026-05-18T00:00:00",
        all_leads=True,
        all_members=True,
        target_grid=grid,
        accumulation_mode="total-since-init",
        resampling="nearest",
    )

    assert result["ok"]
    assert (output_dir / "member_000" / "xmrg0518202601z.gz").exists()
    assert (output_dir / "member_001" / "xmrg0518202602z.gz").exists()
    assert np.nanmean(_physical(output_dir / "member_000" / "xmrg0518202602z.gz")) == pytest.approx(2.0, abs=0.02)
    assert np.nanmean(_physical(output_dir / "member_001" / "xmrg0518202602z.gz")) == pytest.approx(3.0, abs=0.02)


def test_batch_forecast_dry_run_writes_planned_summary_without_outputs(tmp_path: Path):
    grid, path = _forecast_dataset(tmp_path / "forecast.nc", leads=(1, 2, 3), members=(0,))
    output_dir = tmp_path / "out"
    summary = tmp_path / "summary.csv"

    result = batch_forecast_nc_to_xmrg(
        input_path=path,
        nc_variable="APCP",
        variable="prep",
        source_units="mm",
        target_units="mm",
        output_dir=output_dir,
        init_time="2026-05-18T00:00:00",
        all_leads=True,
        target_grid=grid,
        summary=summary,
        dry_run=True,
    )

    assert result["ok"]
    assert result["n_requested"] == 3
    assert not (output_dir / "xmrg0518202601z.gz").exists()
    assert not (output_dir / "member_000").exists()
    rows = _read_summary(summary)
    assert len(rows) == 3
    assert {row["message"] for row in rows} == {"planned"}
    assert rows[0]["output"].endswith("xmrg0518202601z.gz")
    assert "member_000" not in rows[0]["output"]
