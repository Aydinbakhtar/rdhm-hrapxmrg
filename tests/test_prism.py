from __future__ import annotations

from datetime import date
from pathlib import Path
import zipfile

import numpy as np
import pytest

from hrapxmrg.hrap import TargetGrid
from hrapxmrg.prism import (
    default_rdhm_variable,
    extract_prism_zip,
    find_prism_bil_in_zip,
    prism_date_from_filename,
    prism_info,
    prism_to_xmrg,
    prism_units,
    prism_variable_from_filename,
)
from hrapxmrg.xmrg import read_xmrg


def test_prism_variable_from_filename():
    assert prism_variable_from_filename("prism_ppt_20250101.zip") == "ppt"
    assert prism_variable_from_filename("prism_tmean_20250101.zip") == "tmean"
    assert prism_variable_from_filename("prism_tmin_20250101.zip") == "tmin"
    assert prism_variable_from_filename("prism_tmax_20250101.zip") == "tmax"


def test_prism_date_from_filename():
    assert prism_date_from_filename("prism_ppt_20250101.zip") == date(2025, 1, 1)


def test_default_rdhm_variable():
    assert default_rdhm_variable("ppt") == "prep"
    assert default_rdhm_variable("tmean") == "tair"
    assert default_rdhm_variable("tmin") == "tmin"
    assert default_rdhm_variable("tmax") == "tmax"


def test_prism_units():
    assert prism_units("ppt") == "mm"
    assert prism_units("tmean") == "C"
    assert prism_units("tmin") == "C"
    assert prism_units("tmax") == "C"


def test_prism_zip_member_discovery_and_extract(tmp_path: Path):
    zip_path = tmp_path / "prism_ppt_20250101.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("nested/prism_ppt_20250101.bil", b"bil")
        zf.writestr("nested/prism_ppt_20250101.hdr", b"hdr")

    assert find_prism_bil_in_zip(zip_path) == "nested/prism_ppt_20250101.bil"
    extracted = extract_prism_zip(zip_path, tmp_path / "extract")
    assert extracted.name == "prism_ppt_20250101.bil"
    assert extracted.exists()
    assert extracted.with_suffix(".hdr").exists()


def _write_prism_bil_zip(zip_path: Path, grid: TargetGrid, arr: np.ndarray) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS

    from hrapxmrg.hrap import HRAP_CRS_PROJ4
    from hrapxmrg.regrid import target_transform_from_hrap_grid

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


def test_prism_ppt_zip_to_xmrg(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    zip_path = tmp_path / "prism_ppt_20250101.zip"
    output_dir = tmp_path / "out"
    report = tmp_path / "xmrg0102202500z.report.json"
    arr = np.full((grid.maxy, grid.maxx), 1.0, dtype=np.float32)
    _write_prism_bil_zip(zip_path, grid, arr)

    result = prism_to_xmrg(
        input_path=zip_path,
        target_grid=grid,
        output_dir=output_dir,
        daily_precip=True,
        resampling="nearest",
        report=report,
    )

    output = output_dir / "xmrg0102202500z.gz"
    assert result.ok
    assert output.exists()
    assert report.exists()

    stored, meta = read_xmrg(output)
    assert stored.shape == (87, 64)
    assert int(meta.xor) == 341
    assert int(meta.yor) == 313
    assert np.allclose(np.flipud(stored), 100.0, atol=1.0)
    assert np.allclose(np.flipud(stored) / 100.0, 1.0, atol=0.01)


def test_prism_tmean_zip_to_xmrg_converts_to_fahrenheit(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    zip_path = tmp_path / "prism_tmean_20250101.zip"
    output_dir = tmp_path / "out"
    arr = np.zeros((grid.maxy, grid.maxx), dtype=np.float32)
    arr[:, grid.maxx // 2 :] = 10.0
    _write_prism_bil_zip(zip_path, grid, arr)

    result = prism_to_xmrg(
        input_path=zip_path,
        target_grid=grid,
        output_dir=output_dir,
        hour=12,
        resampling="nearest",
    )

    output = output_dir / "tair0101202512z.gz"
    assert result.ok
    assert output.exists()

    stored, _meta = read_xmrg(output)
    physical = np.flipud(stored) / 100.0
    assert float(np.nanmin(physical)) == pytest.approx(32.0, abs=0.01)
    assert float(np.nanmax(physical)) == pytest.approx(50.0, abs=0.01)


def test_prism_info_reads_synthetic_zip_metadata(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    zip_path = tmp_path / "prism_ppt_20250101.zip"
    arr = np.full((grid.maxy, grid.maxx), 1.0, dtype=np.float32)
    _write_prism_bil_zip(zip_path, grid, arr)

    info = prism_info(zip_path, sample_stats=True)

    assert info["parsed_prism_variable"] == "ppt"
    assert info["parsed_date"] == "2025-01-01"
    assert info["source_units"] == "mm"
    assert info["default_rdhm_variable"] == "prep"
    assert info["width"] == 64
    assert info["height"] == 87
    assert info["sample_stats"]["min"] == pytest.approx(1.0)
