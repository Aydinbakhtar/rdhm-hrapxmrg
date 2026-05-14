from pathlib import Path

import numpy as np

from hrapxmrg.xmrg import read_xmrg, write_xmrg
from hrapxmrg.validate import validate_xmrg_file


def test_temperature_f100_roundtrip(tmp_path: Path):
    grid_f = np.array([[8.58, 10.12], [32.00, -5.25]], dtype=float)
    out = tmp_path / "tair0101201512z.gz"

    write_xmrg(
        out,
        grid_f,
        xor=341,
        yor=313,
        header_type="int32",
        dtype="int16",
        scale=100,
        missing=-999,
    )

    stored, meta = read_xmrg(out)
    assert meta.dtype == "int16"
    assert meta.maxx == 2
    assert meta.maxy == 2

    recovered_f = stored / 100.0
    np.testing.assert_allclose(recovered_f, grid_f, atol=0.01)

    result = validate_xmrg_file(out, "tair")
    assert result.ok


def test_float32_roundtrip(tmp_path: Path):
    grid = np.array([[0.0, 1.25], [2.5, 9.0]], dtype=float)
    out = tmp_path / "0101200012z.gz"

    write_xmrg(
        out,
        grid,
        xor=341,
        yor=313,
        header_type="int32",
        dtype="float32",
        scale=1,
        missing=-999,
    )

    recovered, meta = read_xmrg(out)
    assert meta.dtype == "float32"
    np.testing.assert_allclose(recovered, grid, atol=1e-6)
