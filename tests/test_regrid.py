import numpy as np
import pytest

from hrapxmrg.hrap import TargetGrid
from hrapxmrg.regrid import convert_units, target_transform_from_hrap_grid


def test_target_transform_from_hrap_grid_origin():
    rasterio = pytest.importorskip("rasterio")

    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    transform = target_transform_from_hrap_grid(grid)

    # Known HRAP meter bounds from current RDHM domain.
    assert transform.c == pytest.approx(-285750.0)
    assert transform.f == pytest.approx(-5719762.5)
    assert transform.a == pytest.approx(4762.5)
    assert transform.e == pytest.approx(-4762.5)


def test_convert_c_to_f_preserves_nodata():
    arr = np.array([[0.0, 10.0], [-999.0, np.nan]], dtype=np.float32)
    out = convert_units(arr, source_units="C", target_units="F")

    assert out[0, 0] == pytest.approx(32.0)
    assert out[0, 1] == pytest.approx(50.0)
    assert out[1, 0] == pytest.approx(-999.0)
    assert np.isnan(out[1, 1])


def test_convert_k_to_f():
    arr = np.array([[273.15, 283.15]], dtype=np.float32)
    out = convert_units(arr, source_units="K", target_units="F")

    assert out[0, 0] == pytest.approx(32.0)
    assert out[0, 1] == pytest.approx(50.0)
