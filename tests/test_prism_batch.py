from __future__ import annotations

import csv
from pathlib import Path
import zipfile

import numpy as np
import pytest

from hrapxmrg.hrap import TargetGrid
from hrapxmrg.prism_batch import batch_prism_hourly_prep, batch_prism_hourly_tair
from hrapxmrg.xmrg import read_xmrg


def _write_prism_bil_zip(zip_path: Path, grid: TargetGrid, arr: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS

    from hrapxmrg.hrap import HRAP_CRS_PROJ4
    from hrapxmrg.regrid import target_transform_from_hrap_grid

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    bil_path = zip_path.with_suffix(".bil")
    try:
        with rasterio.open(
            bil_path,
            "w",
            driver="ENVI",
            height=grid.maxy,
            width=grid.maxx,
            count=1,
            dtype="float32",
            crs=CRS.from_proj4(HRAP_CRS_PROJ4),
            transform=target_transform_from_hrap_grid(grid),
            nodata=-999.0,
            interleave="bil",
        ) as dst:
            dst.write(arr.astype("float32"), 1)
    except Exception as exc:
        pytest.skip(f"rasterio ENVI BIL writing is unavailable: {exc}")

    sidecars = [path for path in zip_path.parent.iterdir() if path.stem == bil_path.stem]
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sidecars:
            zf.write(path, arcname=path.name)


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_batch_prism_hourly_prep(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    input_dir = tmp_path / "ppt"
    output_dir = tmp_path / "out"
    summary = tmp_path / "prep_summary.csv"
    _write_prism_bil_zip(
        input_dir / "prism_ppt_20250101.zip",
        grid,
        np.full(grid.shape, 24.0, dtype=np.float32),
    )
    _write_prism_bil_zip(
        input_dir / "prism_ppt_20250102.zip",
        grid,
        np.full(grid.shape, 48.0, dtype=np.float32),
    )

    result = batch_prism_hourly_prep(
        input_dir=input_dir,
        output_dir=output_dir,
        target_grid=grid,
        resampling="nearest",
        summary=summary,
    )

    assert result["ok"] is True
    assert result["n_days"] == 2
    assert result["n_files"] == 48
    assert len(_read_summary(summary)) == 48
    assert (output_dir / "xmrg0101202501z.gz").exists()
    assert (output_dir / "xmrg0102202500z.gz").exists()
    assert (output_dir / "xmrg0102202501z.gz").exists()
    assert (output_dir / "xmrg0103202500z.gz").exists()

    stored, _meta = read_xmrg(output_dir / "xmrg0101202501z.gz")
    assert np.allclose(np.flipud(stored) / 100.0, 1.0, atol=0.01)
    stored, _meta = read_xmrg(output_dir / "xmrg0102202501z.gz")
    assert np.allclose(np.flipud(stored) / 100.0, 2.0, atol=0.01)


def test_batch_prism_hourly_tair(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    tmin_dir = tmp_path / "tmin"
    tmax_dir = tmp_path / "tmax"
    output_dir = tmp_path / "out"
    summary = tmp_path / "tair_summary.csv"
    _write_prism_bil_zip(tmin_dir / "prism_tmin_20250101.zip", grid, np.full(grid.shape, 0.0))
    _write_prism_bil_zip(tmax_dir / "prism_tmax_20250101.zip", grid, np.full(grid.shape, 10.0))
    _write_prism_bil_zip(tmin_dir / "prism_tmin_20250102.zip", grid, np.full(grid.shape, 5.0))
    _write_prism_bil_zip(tmax_dir / "prism_tmax_20250102.zip", grid, np.full(grid.shape, 15.0))

    result = batch_prism_hourly_tair(
        tmin_dir=tmin_dir,
        tmax_dir=tmax_dir,
        output_dir=output_dir,
        target_grid=grid,
        resampling="nearest",
        summary=summary,
    )

    assert result["ok"] is True
    assert result["n_days"] == 2
    assert result["n_files"] == 48
    assert len(_read_summary(summary)) == 48
    assert (output_dir / "tair0101202500z.gz").exists()
    assert (output_dir / "tair0101202523z.gz").exists()
    assert (output_dir / "tair0102202500z.gz").exists()
    assert (output_dir / "tair0102202523z.gz").exists()

    stored, _meta = read_xmrg(output_dir / "tair0101202506z.gz")
    assert np.nanmean(np.flipud(stored) / 100.0) == pytest.approx(32.0, abs=0.01)
    stored, _meta = read_xmrg(output_dir / "tair0101202515z.gz")
    assert np.nanmean(np.flipud(stored) / 100.0) == pytest.approx(50.0, abs=0.01)


def test_batch_prism_hourly_tair_optional_tmean(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    tmin_dir = tmp_path / "tmin"
    tmax_dir = tmp_path / "tmax"
    tmean_dir = tmp_path / "tmean"
    output_dir = tmp_path / "out"
    summary = tmp_path / "tair_summary.csv"
    _write_prism_bil_zip(tmin_dir / "prism_tmin_20250101.zip", grid, np.full(grid.shape, 0.0))
    _write_prism_bil_zip(tmax_dir / "prism_tmax_20250101.zip", grid, np.full(grid.shape, 10.0))
    _write_prism_bil_zip(tmean_dir / "prism_tmean_20250101.zip", grid, np.full(grid.shape, 5.0))
    _write_prism_bil_zip(tmin_dir / "prism_tmin_20250102.zip", grid, np.full(grid.shape, 5.0))
    _write_prism_bil_zip(tmax_dir / "prism_tmax_20250102.zip", grid, np.full(grid.shape, 15.0))

    result = batch_prism_hourly_tair(
        tmin_dir=tmin_dir,
        tmax_dir=tmax_dir,
        tmean_dir=tmean_dir,
        output_dir=output_dir,
        target_grid=grid,
        resampling="nearest",
        summary=summary,
    )

    assert result["ok"] is True
    rows = _read_summary(summary)
    jan1 = [row for row in rows if row["date"] == "2025-01-01"]
    jan2 = [row for row in rows if row["date"] == "2025-01-02"]
    assert jan1[0]["source_tmean"].endswith("prism_tmean_20250101.zip")
    assert jan1[0]["tmean_missing"] == "False"
    assert jan2[0]["source_tmean"] == ""
    assert jan2[0]["tmean_missing"] == "True"


def test_batch_prism_hourly_prep_date_filtering(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    input_dir = tmp_path / "ppt"
    output_dir = tmp_path / "out"
    summary = tmp_path / "prep_summary.csv"
    for day in (1, 2, 3):
        _write_prism_bil_zip(
            input_dir / f"prism_ppt_2025010{day}.zip",
            grid,
            np.full(grid.shape, 24.0, dtype=np.float32),
        )

    result = batch_prism_hourly_prep(
        input_dir=input_dir,
        output_dir=output_dir,
        target_grid=grid,
        start_date="2025-01-02",
        end_date="2025-01-02",
        resampling="nearest",
        summary=summary,
    )

    assert result["n_days"] == 1
    assert result["n_files"] == 24
    rows = _read_summary(summary)
    assert len(rows) == 24
    assert {row["date"] for row in rows} == {"2025-01-02"}
    assert not (output_dir / "xmrg0101202501z.gz").exists()
    assert (output_dir / "xmrg0102202501z.gz").exists()


def test_batch_prism_hourly_tair_missing_required_pair(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    tmin_dir = tmp_path / "tmin"
    tmax_dir = tmp_path / "tmax"
    tmin_dir.mkdir()
    tmax_dir.mkdir()
    (tmin_dir / "prism_tmin_20250101.zip").write_text("placeholder", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required PRISM files"):
        batch_prism_hourly_tair(
            tmin_dir=tmin_dir,
            tmax_dir=tmax_dir,
            output_dir=tmp_path / "out",
            target_grid=grid,
        )

    result = batch_prism_hourly_tair(
        tmin_dir=tmin_dir,
        tmax_dir=tmax_dir,
        output_dir=tmp_path / "out",
        target_grid=grid,
        continue_on_error=True,
    )

    assert result["ok"] is True
    assert result["n_failed_days"] == 1
    assert result["rows"][0]["ok"] is False
    assert "missing required PRISM files" in result["rows"][0]["message"]
