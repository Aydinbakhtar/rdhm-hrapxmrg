from pathlib import Path

from hrapxmrg.ascii_grid import read_ascii_grid


def test_read_ascii_grid(tmp_path: Path):
    p = tmp_path / "sample.asc"
    p.write_text(
        "\n".join(
            [
                "ncols 2",
                "nrows 2",
                "xllcorner 341",
                "yllcorner 313",
                "cellsize 1",
                "NODATA_value -999",
                "1 2",
                "3 4",
            ]
        )
    )
    asc = read_ascii_grid(p)
    assert asc.ncols == 2
    assert asc.nrows == 2
    assert asc.to_target_grid().xor == 341
    assert asc.array[1, 1] == 4
