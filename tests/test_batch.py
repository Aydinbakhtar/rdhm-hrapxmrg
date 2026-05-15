from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from hrapxmrg.batch import batch_raster_to_xmrg, parse_date_from_filename
from hrapxmrg.hrap import TargetGrid
from hrapxmrg.xmrg import read_xmrg


DATE_REGEX = r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})"


def _write_domain_config(path: Path, grid: TargetGrid) -> None:
    path.write_text(
        f"""
target_grid:
  xor: {grid.xor}
  yor: {grid.yor}
  maxx: {grid.maxx}
  maxy: {grid.maxy}
  cellsize: {grid.cellsize}
  nodata: {grid.nodata}
""".lstrip(),
        encoding="utf-8",
    )


def _write_hrap_geotiff(path: Path, grid: TargetGrid, value: float) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS

    from hrapxmrg.hrap import HRAP_CRS_PROJ4
    from hrapxmrg.regrid import target_transform_from_hrap_grid

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((grid.maxy, grid.maxx), value, dtype=np.float32)
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
        dst.write(arr, 1)


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_parse_date_from_filename():
    parsed = parse_date_from_filename("prep_19990501.tif", DATE_REGEX)
    assert parsed == date(1999, 5, 1)


def test_batch_daily_prep_writes_outputs_and_summary(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    summary = tmp_path / "summary.csv"
    _write_domain_config(tmp_path / "domain.yaml", grid)
    _write_hrap_geotiff(input_dir / "prep_19990501.tif", grid, 1.25)
    _write_hrap_geotiff(input_dir / "prep_19990502.tif", grid, 2.50)

    rows = batch_raster_to_xmrg(
        input_dir=input_dir,
        pattern="*.tif",
        date_regex=DATE_REGEX,
        variable="prep",
        daily_precip=True,
        output_dir=output_dir,
        target_grid=grid,
        target_domain_source=f"config={tmp_path / 'domain.yaml'}",
        resampling="nearest",
        summary=summary,
    )

    assert len(rows) == 2
    assert (output_dir / "xmrg0502199900z.gz").exists()
    assert (output_dir / "xmrg0503199900z.gz").exists()

    summary_rows = _read_summary(summary)
    assert len(summary_rows) == 2
    assert {row["ok"] for row in summary_rows} == {"True"}

    stored, meta = read_xmrg(output_dir / "xmrg0502199900z.gz")
    assert stored.shape == (87, 64)
    assert int(meta.xor) == 341
    assert int(meta.yor) == 313
    assert np.allclose(np.flipud(stored) / 100.0, 1.25, atol=0.01)


def test_batch_dry_run_writes_planned_summary_without_outputs(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    summary = tmp_path / "summary.csv"
    _write_hrap_geotiff(input_dir / "prep_19990501.tif", grid, 1.25)
    _write_hrap_geotiff(input_dir / "prep_19990502.tif", grid, 2.50)

    rows = batch_raster_to_xmrg(
        input_dir=input_dir,
        pattern="*.tif",
        date_regex=DATE_REGEX,
        variable="prep",
        daily_precip=True,
        output_dir=output_dir,
        target_grid=grid,
        resampling="nearest",
        summary=summary,
        dry_run=True,
    )

    assert len(rows) == 2
    assert not (output_dir / "xmrg0502199900z.gz").exists()
    assert not (output_dir / "xmrg0503199900z.gz").exists()
    summary_rows = _read_summary(summary)
    assert [row["message"] for row in summary_rows] == ["planned", "planned"]
    assert all(row["dry_run"] == "True" for row in summary_rows)


def test_batch_temperature_converts_celsius_to_fahrenheit(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_hrap_geotiff(input_dir / "tair_19990501.tif", grid, 0.0)
    _write_hrap_geotiff(input_dir / "tair_19990502.tif", grid, 0.0)

    rows = batch_raster_to_xmrg(
        input_dir=input_dir,
        pattern="*.tif",
        date_regex=DATE_REGEX,
        variable="tair",
        hour=12,
        output_dir=output_dir,
        target_grid=grid,
        source_units="C",
        target_units="F",
        resampling="nearest",
    )

    assert len(rows) == 2
    assert (output_dir / "tair0501199912z.gz").exists()
    assert (output_dir / "tair0502199912z.gz").exists()

    stored, _meta = read_xmrg(output_dir / "tair0501199912z.gz")
    physical = np.flipud(stored) / 100.0
    assert np.nanmin(physical) == pytest.approx(32.0, abs=0.01)
    assert np.nanmax(physical) == pytest.approx(32.0, abs=0.01)


def test_batch_bad_filename_raises_or_records_failure(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    bad_file = input_dir / "prep_no_date.tif"
    bad_file.write_text("not a raster", encoding="utf-8")

    with pytest.raises(ValueError, match="date regex did not match"):
        batch_raster_to_xmrg(
            input_dir=input_dir,
            pattern="*.tif",
            date_regex=DATE_REGEX,
            variable="prep",
            daily_precip=True,
            output_dir=tmp_path / "output",
            target_grid=grid,
        )

    rows = batch_raster_to_xmrg(
        input_dir=input_dir,
        pattern="*.tif",
        date_regex=DATE_REGEX,
        variable="prep",
        daily_precip=True,
        output_dir=tmp_path / "output",
        target_grid=grid,
        continue_on_error=True,
    )

    assert len(rows) == 1
    assert rows[0]["ok"] is False
    assert "date regex did not match" in str(rows[0]["message"])
