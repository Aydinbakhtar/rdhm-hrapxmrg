"""Raster-to-HRAP regridding utilities.

This module reprojects a source raster onto a trusted RDHM HRAP TargetGrid.

Important design rule:
The source raster extent never defines the RDHM output grid. The output grid
must come from a TargetGrid built from .con, ASCII/XMRG template, YAML config,
or shapefile-derived HRAP domain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .hrap import HRAP_CRS_PROJ4, HRAP_MESH_M, XPOL, YPOL, TargetGrid


def target_transform_from_hrap_grid(grid: TargetGrid):
    """Return rasterio Affine transform for an HRAP TargetGrid.

    HRAP index to stereographic meters:
        x_m = (hrapx - XPOL) * HRAP_MESH_M
        y_m = (hrapy - YPOL) * HRAP_MESH_M

    Rasterio transform uses upper-left outer corner. Target array shape is
    (grid.maxy, grid.maxx).
    """
    try:
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise ImportError(
            "Raster regridding requires rasterio. Install with: "
            "python -m pip install rasterio"
        ) from exc

    xmin = (grid.xor - XPOL) * HRAP_MESH_M
    ymin = (grid.yor - YPOL) * HRAP_MESH_M
    ymax = ymin + grid.maxy * HRAP_MESH_M

    return from_origin(xmin, ymax, HRAP_MESH_M, HRAP_MESH_M)


def reproject_raster_to_hrap(
    input_raster: str | Path,
    target_grid: TargetGrid,
    *,
    band: int = 1,
    resampling: str = "bilinear",
    dst_nodata: float | None = None,
) -> np.ndarray:
    """Reproject a raster band onto an exact HRAP target grid.

    Parameters
    ----------
    input_raster:
        GeoTIFF or any raster readable by rasterio.
    target_grid:
        Trusted RDHM HRAP target grid.
    band:
        Source raster band number, 1-based.
    resampling:
        One of nearest, bilinear, cubic, average.
    dst_nodata:
        Destination nodata. Defaults to target_grid.nodata.

    Returns
    -------
    np.ndarray
        2D array with shape (target_grid.maxy, target_grid.maxx), in source
        physical units. No XMRG scaling is applied here.
    """
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError as exc:
        raise ImportError(
            "Raster regridding requires rasterio. Install with: "
            "python -m pip install rasterio"
        ) from exc

    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    if resampling not in resampling_map:
        raise ValueError(f"resampling must be one of {sorted(resampling_map)}")

    input_raster = Path(input_raster)
    dst_nodata = target_grid.nodata if dst_nodata is None else dst_nodata

    dst = np.full(
        (target_grid.maxy, target_grid.maxx),
        dst_nodata,
        dtype=np.float32,
    )

    dst_transform = target_transform_from_hrap_grid(target_grid)
    dst_crs = CRS.from_proj4(HRAP_CRS_PROJ4)

    with rasterio.open(input_raster) as src:
        if src.crs is None:
            raise ValueError(f"{input_raster}: source raster CRS is missing")

        src_data = src.read(band)

        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=dst_nodata,
            resampling=resampling_map[resampling],
        )

    return dst


def convert_units(array: np.ndarray, *, source_units: str, target_units: str) -> np.ndarray:
    """Convert common climate units.

    This operates on physical values before XMRG scaling.
    """
    src = source_units.lower()
    dst = target_units.lower()

    if src == dst:
        return array

    out = array.astype(np.float32, copy=True)

    valid = np.isfinite(out) & (out > -900)

    if src in ("c", "celsius", "degc") and dst in ("f", "fahrenheit", "degf"):
        out[valid] = out[valid] * 9.0 / 5.0 + 32.0
        return out

    if src in ("k", "kelvin") and dst in ("f", "fahrenheit", "degf"):
        out[valid] = (out[valid] - 273.15) * 9.0 / 5.0 + 32.0
        return out

    if src in ("k", "kelvin") and dst in ("c", "celsius", "degc"):
        out[valid] = out[valid] - 273.15
        return out

    raise ValueError(f"unsupported unit conversion: {source_units} -> {target_units}")
