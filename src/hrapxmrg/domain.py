"""Domain/grid helpers for RDHM HRAP workflows.

This module defines TargetGrid objects from:
- HRAP ASCII templates
- XMRG templates
- YAML configs
- RDHM .con files
- polygon shapefiles

Important design rule:
The RDHM target grid must come from a trusted domain source. Do not infer the
RDHM output grid from a climate source raster extent.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math
import re
from typing import Iterable

import yaml

from .ascii_grid import read_ascii_grid
from .hrap import HRAP_CRS_PROJ4, HRAP_MESH_M, XPOL, YPOL, TargetGrid
from .xmrg import read_xmrg


def target_grid_from_ascii_template(path: str | Path) -> TargetGrid:
    """Create TargetGrid from an HRAP/ESRI ASCII template."""
    return read_ascii_grid(path).to_target_grid()


def target_grid_from_xmrg_template(path: str | Path) -> TargetGrid:
    """Create TargetGrid from an XMRG template."""
    _, meta = read_xmrg(path)
    return TargetGrid(
        xor=int(round(meta.xor)),
        yor=int(round(meta.yor)),
        maxx=int(meta.maxx),
        maxy=int(meta.maxy),
        cellsize=1.0,
        nodata=-999.0,
    )


def target_grid_from_yaml(path: str | Path) -> TargetGrid:
    """Create TargetGrid from YAML config.

    Supports either:
      target_grid:
        xor: ...
        yor: ...
        maxx: ...
        maxy: ...

    or direct top-level keys.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}

    grid_data = data.get("target_grid", data)

    required = ["xor", "yor", "maxx", "maxy"]
    missing = [k for k in required if k not in grid_data]
    if missing:
        raise ValueError(f"{path}: missing required target grid keys: {missing}")

    return TargetGrid(
        xor=int(grid_data["xor"]),
        yor=int(grid_data["yor"]),
        maxx=int(grid_data["maxx"]),
        maxy=int(grid_data["maxy"]),
        cellsize=float(grid_data.get("cellsize", 1.0)),
        nodata=float(grid_data.get("nodata", -999.0)),
    )


def write_target_grid_yaml(path: str | Path, grid: TargetGrid, *, source: str | None = None) -> None:
    """Write TargetGrid to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "target_grid": asdict(grid),
    }
    if source is not None:
        payload["source"] = source

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def hrap_bounds_from_grid(grid: TargetGrid) -> dict[str, int]:
    """Return inclusive HRAP cell-index bounds."""
    return {
        "xor": grid.xor,
        "yor": grid.yor,
        "urx": grid.xor + grid.maxx - 1,
        "ury": grid.yor + grid.maxy - 1,
        "maxx": grid.maxx,
        "maxy": grid.maxy,
    }


def projected_bounds_from_grid(grid: TargetGrid) -> dict[str, float]:
    """Return outer grid bounds in HRAP stereographic meters."""
    xmin = (grid.xor - XPOL) * HRAP_MESH_M
    ymin = (grid.yor - YPOL) * HRAP_MESH_M
    xmax = xmin + grid.maxx * HRAP_MESH_M
    ymax = ymin + grid.maxy * HRAP_MESH_M
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def _projected_m_to_hrap_index(x: float, y: float) -> tuple[float, float]:
    """Convert HRAP stereographic meters to fractional HRAP indices."""
    hrapx = (x + XPOL * HRAP_MESH_M) / HRAP_MESH_M
    hrapy = (y + YPOL * HRAP_MESH_M) / HRAP_MESH_M
    return hrapx, hrapy


def target_grid_from_shapefile(path: str | Path, *, buffer_cells: int = 0) -> TargetGrid:
    """Create snapped HRAP TargetGrid from polygon shapefile bounds.

    Requires geopandas. This is useful for new basin/domain creation.
    For established RDHM workflows, compare with .con or existing templates.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "Shapefile support requires geopandas. Install with: "
            "python -m pip install geopandas shapely pyproj"
        ) from exc

    path = Path(path)
    gdf = gpd.read_file(path)

    if gdf.empty:
        raise ValueError(f"{path}: shapefile has no features")
    if gdf.crs is None:
        raise ValueError(f"{path}: shapefile CRS is missing. Define CRS before using it.")

    hrap_gdf = gdf.to_crs(HRAP_CRS_PROJ4)
    xmin, ymin, xmax, ymax = hrap_gdf.total_bounds

    hrap_min_x, hrap_min_y = _projected_m_to_hrap_index(xmin, ymin)
    hrap_max_x, hrap_max_y = _projected_m_to_hrap_index(xmax, ymax)

    xor = math.floor(min(hrap_min_x, hrap_max_x)) - buffer_cells
    yor = math.floor(min(hrap_min_y, hrap_max_y)) - buffer_cells
    urx = math.ceil(max(hrap_min_x, hrap_max_x)) + buffer_cells
    ury = math.ceil(max(hrap_min_y, hrap_max_y)) + buffer_cells

    return TargetGrid(
        xor=int(xor),
        yor=int(yor),
        maxx=int(urx - xor + 1),
        maxy=int(ury - yor + 1),
        cellsize=1.0,
        nodata=-999.0,
    )




def _parse_explicit_con_bounds(text: str) -> TargetGrid | None:
    """Try to parse explicit grid information from an RDHM .con-like text file.

    Preferred RDHM .con behavior:
    - Use COL and ROW as authoritative grid dimensions.
    - Use LLX and LLY as the lower-left HRAP origin.
    - Treat URX/URY as consistency-check fields only.

    This is important because some .con files can contain inconsistent
    ROW/URY information. For forcing generation, RDHM/XMRG needs the actual
    array size and lower-left origin.
    """
    lower = text.lower()

    def find_number(label: str) -> float | None:
        match = re.search(rf"^\s*{label.lower()}\s+(-?\d+(?:\.\d+)?)", lower, flags=re.M)
        if match:
            return float(match.group(1))
        return None

    col = find_number("col")
    row = find_number("row")
    llx = find_number("llx")
    lly = find_number("lly")
    urx = find_number("urx")
    ury = find_number("ury")

    if col is not None and row is not None and llx is not None and lly is not None:
        return TargetGrid(
            xor=int(round(llx)),
            yor=int(round(lly)),
            maxx=int(round(col)),
            maxy=int(round(row)),
            cellsize=1.0,
            nodata=-999.0,
        )

    # Fallback for other .con-like files with only explicit corner labels.
    patterns = {
        "left": r"(?:left|llx|xmin|xll|xor)\D+(-?\d+)",
        "right": r"(?:right|urx|xmax)\D+(-?\d+)",
        "bottom": r"(?:bottom|lly|ymin|yll|yor)\D+(-?\d+)",
        "top": r"(?:top|ury|ymax)\D+(-?\d+)",
    }

    found: dict[str, int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, lower)
        if match:
            found[key] = int(match.group(1))

    if {"left", "right", "bottom", "top"}.issubset(found):
        xor = found["left"]
        yor = found["bottom"]
        urx_i = found["right"]
        ury_i = found["top"]
        return TargetGrid(xor=xor, yor=yor, maxx=urx_i - xor + 1, maxy=ury_i - yor + 1)

    return None


def _iter_first_two_hrap_ints(lines: Iterable[str]) -> Iterable[tuple[int, int]]:
    """Yield plausible HRAP x/y integer pairs from lines.

    Fallback parser: if a line starts with two plausible HRAP indices, use them.
    This avoids station IDs because those are usually much larger than 2000.
    """
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        nums = re.findall(r"[-+]?\d+", stripped)
        if len(nums) < 2:
            continue

        x = int(nums[0])
        y = int(nums[1])

        if 0 <= x <= 2000 and 0 <= y <= 2000:
            yield x, y


def target_grid_from_con_file(path: str | Path, *, buffer_cells: int = 0) -> TargetGrid:
    """Create TargetGrid from RDHM .con file.

    First tries explicit bounds. If those are unavailable, it parses plausible
    first-column HRAP x/y cell records and derives min/max bounds.

    If this fails for a specific .con format, inspect the file and add a parser
    for that format rather than guessing silently.
    """
    path = Path(path)
    text = path.read_text(errors="ignore")

    explicit = _parse_explicit_con_bounds(text)
    if explicit is not None:
        if buffer_cells:
            return TargetGrid(
                xor=explicit.xor - buffer_cells,
                yor=explicit.yor - buffer_cells,
                maxx=explicit.maxx + 2 * buffer_cells,
                maxy=explicit.maxy + 2 * buffer_cells,
                cellsize=explicit.cellsize,
                nodata=explicit.nodata,
            )
        return explicit

    pairs = list(_iter_first_two_hrap_ints(text.splitlines()))

    if len(pairs) < 10:
        raise ValueError(
            f"{path}: could not infer HRAP bounds from .con file. "
            "Use an ASCII/XMRG template or YAML config, or add a parser for this .con format."
        )

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    xor = min(xs) - buffer_cells
    yor = min(ys) - buffer_cells
    urx = max(xs) + buffer_cells
    ury = max(ys) + buffer_cells

    return TargetGrid(
        xor=int(xor),
        yor=int(yor),
        maxx=int(urx - xor + 1),
        maxy=int(ury - yor + 1),
        cellsize=1.0,
        nodata=-999.0,
    )


def grid_covers(inner: TargetGrid, outer: TargetGrid) -> bool:
    """Return True if outer grid fully covers inner grid."""
    ib = hrap_bounds_from_grid(inner)
    ob = hrap_bounds_from_grid(outer)
    return (
        ob["xor"] <= ib["xor"]
        and ob["yor"] <= ib["yor"]
        and ob["urx"] >= ib["urx"]
        and ob["ury"] >= ib["ury"]
    )


def _corner_latlon_lines(grid: TargetGrid) -> list[str]:
    """Return corner lon/lat lines if pyproj is available."""
    try:
        from pyproj import CRS, Transformer
    except ImportError:
        return ["corner lon/lat: unavailable because pyproj is not installed"]

    bounds = projected_bounds_from_grid(grid)
    transformer = Transformer.from_crs(CRS.from_proj4(HRAP_CRS_PROJ4), CRS.from_epsg(4326), always_xy=True)

    corners = {
        "lower_left": (bounds["xmin"], bounds["ymin"]),
        "lower_right": (bounds["xmax"], bounds["ymin"]),
        "upper_left": (bounds["xmin"], bounds["ymax"]),
        "upper_right": (bounds["xmax"], bounds["ymax"]),
    }

    lines = []
    for name, (x, y) in corners.items():
        lon, lat = transformer.transform(x, y)
        lines.append(f"  {name}: lon={lon:.6f}, lat={lat:.6f}")
    return lines


def format_domain_report(grid: TargetGrid, *, source: str = "unknown") -> str:
    """Format a human-readable HRAP domain report."""
    hb = hrap_bounds_from_grid(grid)
    pb = projected_bounds_from_grid(grid)

    lines = []
    lines.append("HRAP DOMAIN REPORT")
    lines.append("=" * 60)
    lines.append(f"source: {source}")
    lines.append("")
    lines.append("HRAP grid:")
    lines.append(f"  xor: {hb['xor']}")
    lines.append(f"  yor: {hb['yor']}")
    lines.append(f"  urx: {hb['urx']}")
    lines.append(f"  ury: {hb['ury']}")
    lines.append(f"  maxx: {hb['maxx']}")
    lines.append(f"  maxy: {hb['maxy']}")
    lines.append(f"  cellsize: {grid.cellsize}")
    lines.append(f"  nodata: {grid.nodata}")
    lines.append("")
    lines.append("Projected HRAP bounds, meters:")
    lines.append(f"  xmin: {pb['xmin']:.3f}")
    lines.append(f"  ymin: {pb['ymin']:.3f}")
    lines.append(f"  xmax: {pb['xmax']:.3f}")
    lines.append(f"  ymax: {pb['ymax']:.3f}")
    lines.append("")
    lines.append("Corner lon/lat:")
    lines.extend(_corner_latlon_lines(grid))

    return "\n".join(lines)


def format_domain_comparison_report(
    *,
    con_grid: TargetGrid | None = None,
    shp_grid: TargetGrid | None = None,
    template_grid: TargetGrid | None = None,
) -> str:
    """Format a comparison report among available domain sources."""
    lines = []
    lines.append("HRAP DOMAIN COMPARISON")
    lines.append("=" * 60)

    if con_grid is not None:
        lines.append("")
        lines.append(format_domain_report(con_grid, source=".con"))

    if shp_grid is not None:
        lines.append("")
        lines.append(format_domain_report(shp_grid, source="shapefile"))

    if template_grid is not None:
        lines.append("")
        lines.append(format_domain_report(template_grid, source="template/config"))

    lines.append("")
    lines.append("Checks:")

    if con_grid is not None and shp_grid is not None:
        lines.append(f"  shapefile-derived grid covers .con grid: {'PASS' if grid_covers(con_grid, shp_grid) else 'FAIL'}")
        lines.append(f"  .con grid covers shapefile-derived grid: {'PASS' if grid_covers(shp_grid, con_grid) else 'FAIL'}")
        lines.append("  authority recommendation: use .con for RDHM grid; use shapefile as geographic check")

    if con_grid is not None and template_grid is not None:
        lines.append(f"  template/config grid covers .con grid: {'PASS' if grid_covers(con_grid, template_grid) else 'FAIL'}")

    if shp_grid is not None and template_grid is not None:
        lines.append(f"  template/config grid covers shapefile-derived grid: {'PASS' if grid_covers(shp_grid, template_grid) else 'FAIL'}")

    return "\n".join(lines)

def con_header_consistency_warnings(path: str | Path) -> list[str]:
    """Return warnings for inconsistent .con header dimensions."""
    path = Path(path)
    text = path.read_text(errors="ignore").lower()

    def find_number(label: str) -> float | None:
        match = re.search(rf"^\s*{label.lower()}\s+(-?\d+(?:\.\d+)?)", text, flags=re.M)
        if match:
            return float(match.group(1))
        return None

    col = find_number("col")
    row = find_number("row")
    llx = find_number("llx")
    lly = find_number("lly")
    urx = find_number("urx")
    ury = find_number("ury")

    warnings = []

    if col is not None and llx is not None and urx is not None:
        implied_col = int(round(urx - llx + 1))
        if implied_col != int(round(col)):
            warnings.append(
                f"COL={int(col)} but URX-LLX+1={implied_col}. "
                "Using COL as authoritative dimension."
            )

    if row is not None and lly is not None and ury is not None:
        implied_row = int(round(ury - lly + 1))
        if implied_row != int(round(row)):
            warnings.append(
                f"ROW={int(row)} but URY-LLY+1={implied_row}. "
                "Using ROW as authoritative dimension."
            )

    return warnings
