from pathlib import Path

from hrapxmrg.domain import (
    grid_covers,
    hrap_bounds_from_grid,
    target_grid_from_con_file,
    target_grid_from_yaml,
    write_target_grid_yaml,
)
from hrapxmrg.hrap import TargetGrid


def test_write_and_read_domain_yaml(tmp_path: Path):
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    p = tmp_path / "domain.yaml"

    write_target_grid_yaml(p, grid, source="test")
    loaded = target_grid_from_yaml(p)

    assert loaded.xor == 341
    assert loaded.yor == 313
    assert loaded.maxx == 64
    assert loaded.maxy == 87


def test_hrap_bounds_from_grid():
    grid = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    bounds = hrap_bounds_from_grid(grid)

    assert bounds["xor"] == 341
    assert bounds["yor"] == 313
    assert bounds["urx"] == 404
    assert bounds["ury"] == 399


def test_grid_covers():
    outer = TargetGrid(xor=341, yor=313, maxx=64, maxy=87)
    inner = TargetGrid(xor=346, yor=346, maxx=52, maxy=52)

    assert grid_covers(inner, outer)
    assert not grid_covers(outer, inner)


def test_target_grid_from_con_coordinate_fallback(tmp_path: Path):
    con = tmp_path / "fake.con"
    con.write_text(
        "\n".join(
            [
                "341 313 342 313",
                "342 313 343 313",
                "404 399 -1 -1",
                "350 350 351 350",
                "351 350 352 350",
                "352 350 353 350",
                "353 350 354 350",
                "354 350 355 350",
                "355 350 356 350",
                "356 350 357 350",
            ]
        )
    )

    grid = target_grid_from_con_file(con)

    assert grid.xor == 341
    assert grid.yor == 313
    assert grid.x_max_hrap == 404
    assert grid.y_max_hrap == 399

def test_target_grid_from_con_prefers_col_row_over_inconsistent_ury(tmp_path: Path):
    con = tmp_path / "inconsistent_header.con"
    con.write_text(
        "\n".join(
            [
                "TEXT_SEQ",
                "NUM_HEADER_REC       14    5555",
                "COL         64",
                "ROW         87",
                "LLX  341.0000",
                "LLY  313.0000",
                "URX  404.0000",
                "URY  401.0000",
                "DXY    1.0000",
                "DATA_HRAP",
                "0 2 Rv 1 61 86 6.2615 402.0000 315.0000",
            ]
        )
    )

    grid = target_grid_from_con_file(con)

    assert grid.xor == 341
    assert grid.yor == 313
    assert grid.maxx == 64
    assert grid.maxy == 87
    assert grid.x_max_hrap == 404
    assert grid.y_max_hrap == 399
