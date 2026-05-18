"""Forecast-aware NetCDF inspection and one-slice XMRG conversion.

This first NetCDF implementation is intentionally conservative:

* one selected 2D slice is converted at a time;
* forecast and ensemble dimensions require explicit selectors when ambiguous;
* rectilinear 1D lat/lon grids are supported;
* curvilinear 2D lat/lon grids are not supported yet;
* rate precipitation units require an explicit accumulation duration;
* total-since-initialization precipitation differencing is not implemented yet.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import math
import re
import tempfile
from typing import Any

import numpy as np

from .filenames import parse_date, rdhm_filename
from .hrap import TargetGrid
from .pipeline import (
    raster_to_xmrg,
    raster_to_xmrg_manifest,
    write_raster_to_xmrg_manifest,
)
from .validate import ValidationResult


TIME_NAMES = {"time", "forecast_time", "date", "datetime"}
VALID_TIME_NAMES = {"valid_time"}
INIT_TIME_NAMES = {
    "init",
    "init_time",
    "forecast_reference_time",
    "reference_time",
    "model_initialization_time",
}
LEAD_NAMES = {"lead", "lead_time", "step", "forecast_hour", "fhour", "horizon"}
MEMBER_NAMES = {"member", "ensemble", "ens", "realization", "number"}
Y_NAMES = {"y", "hrapy"}
X_NAMES = {"x", "hrapx"}
LAT_NAMES = {"lat", "latitude"}
LON_NAMES = {"lon", "longitude"}


def _xarray():
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError("xarray is required for dimension-based NetCDF selection.") from exc
    return xr


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, np.timedelta64):
        return str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    return str(value)


def _normalize_name(name: str) -> str:
    return name.lower().replace("-", "_")


def _role_names(names: set[str], available: list[str]) -> list[str]:
    return [name for name in available if _normalize_name(name) in names]


def list_netcdf_subdatasets(path: str | Path) -> list[str]:
    """Return rasterio NetCDF subdataset names, if rasterio can inspect them."""
    try:
        import rasterio
    except ImportError:
        return []

    try:
        with rasterio.open(path) as src:
            return list(src.subdatasets or [])
    except Exception:
        return []


def netcdf_variable_path(input_path: str | Path, nc_variable: str) -> str:
    """Return a rasterio-readable NetCDF variable path."""
    input_path = Path(input_path)
    needle = nc_variable.lower()
    for subdataset in list_netcdf_subdatasets(input_path):
        tail = subdataset.rsplit(":", 1)[-1].strip('"').lower()
        if tail == needle or subdataset.lower().endswith(f":{needle}"):
            return subdataset
    return f'NETCDF:"{input_path}":{nc_variable}'


def identify_dimension_roles(ds: Any) -> dict[str, list[str]]:
    """Identify likely NetCDF dimension/coordinate roles for reporting."""
    names = list(dict.fromkeys(list(ds.dims) + list(ds.coords)))
    roles = {
        "time": _role_names(TIME_NAMES, names),
        "valid_time": _role_names(VALID_TIME_NAMES, names),
        "init_time": _role_names(INIT_TIME_NAMES, names),
        "lead": _role_names(LEAD_NAMES, names),
        "member": _role_names(MEMBER_NAMES, names),
        "y": _role_names(Y_NAMES, names),
        "x": _role_names(X_NAMES, names),
        "lat": _role_names(LAT_NAMES, names),
        "lon": _role_names(LON_NAMES, names),
    }
    for name in names:
        attrs = getattr(ds.coords.get(name), "attrs", {}) if name in ds.coords else {}
        standard_name = _normalize_name(str(attrs.get("standard_name", "")))
        axis = str(attrs.get("axis", "")).upper()
        if "forecast_reference_time" in standard_name and name not in roles["init_time"]:
            roles["init_time"].append(name)
        if "time" in standard_name and name not in roles["time"]:
            roles["time"].append(name)
        if "longitude" in standard_name or axis == "X":
            if name not in roles["lon"] and name not in roles["x"]:
                roles["lon"].append(name)
        if "latitude" in standard_name or axis == "Y":
            if name not in roles["lat"] and name not in roles["y"]:
                roles["lat"].append(name)
    return roles


def _sample_band_stats(dataset: Any, band: int = 1) -> dict[str, Any]:
    arr = dataset.read(band, masked=True)
    valid = np.asarray(arr.compressed() if np.ma.isMaskedArray(arr) else arr[np.isfinite(arr)])
    if valid.size == 0:
        return {"min": None, "mean": None, "max": None, "n_valid": 0}
    return {
        "min": float(np.nanmin(valid)),
        "mean": float(np.nanmean(valid)),
        "max": float(np.nanmax(valid)),
        "n_valid": int(valid.size),
    }


def _rasterio_metadata(src: Any, *, sample_stats: bool = False) -> dict[str, Any]:
    out = {
        "driver": src.driver,
        "width": src.width,
        "height": src.height,
        "count": src.count,
        "crs": str(src.crs) if src.crs else None,
        "transform": list(src.transform) if src.transform else None,
        "bounds": _json_value(src.bounds),
        "nodata": _json_value(src.nodata),
        "dtypes": list(src.dtypes),
        "descriptions": list(src.descriptions),
        "tags": _json_value(src.tags()),
    }
    if sample_stats and src.count:
        out["sample_stats"] = _sample_band_stats(src, 1)
    return out


def _coord_summary(ds: Any, name: str) -> dict[str, Any]:
    coord = ds.coords[name]
    values = coord.values
    flat = np.asarray(values).reshape(-1)
    return {
        "name": name,
        "dimension": coord.dims[0] if coord.dims else None,
        "dtype": str(coord.dtype),
        "count": int(flat.size),
        "first": _json_value(flat[0]) if flat.size else None,
        "last": _json_value(flat[-1]) if flat.size else None,
        "units": coord.attrs.get("units"),
        "calendar": coord.attrs.get("calendar"),
        "attributes": _json_value(coord.attrs),
    }


def inspect_netcdf(
    path: str | Path,
    *,
    sample_stats: bool = False,
    max_subdatasets: int = 20,
) -> dict[str, Any]:
    """Inspect NetCDF structure with rasterio and xarray where available."""
    path = Path(path)
    out: dict[str, Any] = {
        "input": str(path),
        "rasterio_open_ok": False,
        "xarray_open_ok": False,
        "subdatasets": [],
        "variables_from_subdatasets": [],
        "possible_dimension_roles": {},
    }

    try:
        import rasterio

        try:
            with rasterio.open(path) as src:
                out["rasterio_open_ok"] = True
                out["rasterio"] = _rasterio_metadata(src, sample_stats=sample_stats)
                subdatasets = list(src.subdatasets or [])
        except Exception as exc:
            out["rasterio_error"] = str(exc)
            subdatasets = list_netcdf_subdatasets(path)

        out["subdatasets"] = subdatasets
        out["variables_from_subdatasets"] = [s.rsplit(":", 1)[-1].strip('"') for s in subdatasets]
        sub_infos = []
        for subdataset in subdatasets[:max_subdatasets]:
            item: dict[str, Any] = {
                "name": subdataset,
                "variable": subdataset.rsplit(":", 1)[-1].strip('"'),
                "open_ok": False,
            }
            try:
                with rasterio.open(subdataset) as src:
                    item["open_ok"] = True
                    item.update(_rasterio_metadata(src, sample_stats=sample_stats))
            except Exception as exc:
                item["error"] = str(exc)
            sub_infos.append(item)
        out["subdataset_metadata"] = sub_infos
    except ImportError:
        out["rasterio_error"] = "rasterio is not installed"

    try:
        xr = _xarray()
        with xr.open_dataset(path, decode_times=True) as ds:
            out["xarray_open_ok"] = True
            roles = identify_dimension_roles(ds)
            out["possible_dimension_roles"] = roles
            out["global_attributes"] = _json_value(ds.attrs)
            out["dims"] = {str(k): int(v) for k, v in ds.sizes.items()}
            out["coords"] = {
                name: {
                    "dims": list(coord.dims),
                    "shape": list(coord.shape),
                    "dtype": str(coord.dtype),
                    "attrs": _json_value(coord.attrs),
                }
                for name, coord in ds.coords.items()
            }
            out["data_vars"] = {
                name: {
                    "dims": list(var.dims),
                    "shape": list(var.shape),
                    "attrs": _json_value(var.attrs),
                    "units": var.attrs.get("units"),
                }
                for name, var in ds.data_vars.items()
            }
            coord_names = list(ds.coords)
            time_like = sorted(
                set(roles["time"] + roles["valid_time"] + roles["init_time"] + roles["lead"] + roles["member"])
                & set(coord_names)
            )
            out["time_like_coordinates"] = [_coord_summary(ds, name) for name in time_like]
            out["forecast_like_coordinates"] = {
                role: [_coord_summary(ds, name) for name in roles[role] if name in ds.coords]
                for role in ("init_time", "lead", "valid_time", "member")
            }
    except Exception as exc:
        out["xarray_error"] = str(exc)

    return out


def _parse_datetime(value: str | datetime | np.datetime64) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, np.datetime64):
        text = np.datetime_as_string(value, unit="s")
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text).replace(tzinfo=None)


def _datetime64(value: str | datetime | np.datetime64) -> np.datetime64:
    return np.datetime64(_parse_datetime(value))


def normalize_lead_to_hours(value: Any) -> float:
    """Convert common lead/step coordinate values to hours."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.timedelta64):
        return float(value / np.timedelta64(1, "h"))
    if isinstance(value, timedelta):
        return value.total_seconds() / 3600.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().lower()
    try:
        return float(text)
    except ValueError:
        pass
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*(hour|hours|hr|hrs|h)", text)
    if match:
        return float(match.group(1))
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*(day|days|d)", text)
    if match:
        return float(match.group(1)) * 24.0
    raise ValueError(f"cannot interpret lead value as hours: {value!r}")


def parse_time_selector(
    *,
    band: int | None = None,
    time_index: int | None = None,
    time_value: str | None = None,
    valid_time: str | None = None,
    init_index: int | None = None,
    init_time: str | None = None,
    lead_index: int | None = None,
    lead_hour: float | None = None,
    member_index: int | None = None,
    member_name: str | None = None,
) -> str:
    """Validate selector combinations and return selection mode."""
    if time_index is not None and time_value is not None:
        raise ValueError("--time-index and --time-value cannot be combined")
    if valid_time is not None and time_value is not None:
        raise ValueError("--valid-time and --time-value cannot be combined")
    if init_time is not None and init_index is not None:
        raise ValueError("--init-time and --init-index cannot be combined")
    if lead_hour is not None and lead_index is not None:
        raise ValueError("--lead-hour and --lead-index cannot be combined")
    if member_index is not None and member_name is not None:
        raise ValueError("--member-index and --member-name cannot be combined")
    if band is not None:
        xarray_selectors = [
            time_value,
            valid_time,
            init_index,
            init_time,
            lead_index,
            lead_hour,
            member_index,
            member_name,
        ]
        if any(value is not None for value in xarray_selectors):
            raise ValueError("--band cannot be combined with dimension selectors")
        return "rasterio"
    return "xarray"


def _first_matching_dim(da: Any, names: list[str], label: str) -> str:
    for name in names:
        if name in da.dims:
            return name
    raise ValueError(f"{label} selector was provided, but no matching dimension exists")


def _select_exact_coord(da: Any, dim: str, value: Any, label: str) -> Any:
    if dim in da.coords:
        coord_values = np.asarray(da.coords[dim].values)
        if np.issubdtype(coord_values.dtype, np.datetime64):
            target = _datetime64(value)
            matches = np.where(coord_values == target)[0]
        else:
            matches = np.where(coord_values == value)[0]
        if matches.size == 0:
            raise ValueError(f"{label} value {value!r} was not found on dimension {dim!r}")
        return da.isel({dim: int(matches[0])}), coord_values[int(matches[0])]
    return da.sel({dim: value}), value


def _coord_value(da: Any, dim: str) -> Any:
    if dim in da.coords:
        value = np.asarray(da.coords[dim].values).reshape(-1)[0]
        return _json_value(value)
    return None


def _spatial_dims(ds: Any, da: Any, roles: dict[str, list[str]]) -> set[str]:
    spatial = set(roles["x"] + roles["y"] + roles["lat"] + roles["lon"])
    for coord_name in roles["lat"] + roles["lon"]:
        if coord_name in ds.coords:
            spatial.update(ds.coords[coord_name].dims)
    return spatial & set(da.dims)


def select_xarray_slice(
    ds: Any,
    nc_variable: str,
    *,
    time_index: int | None = None,
    time_value: str | None = None,
    valid_time: str | None = None,
    init_index: int | None = None,
    init_time: str | None = None,
    lead_index: int | None = None,
    lead_hour: float | None = None,
    member_index: int | None = None,
    member_name: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Select one NetCDF variable slice without guessing ambiguous dimensions."""
    if nc_variable not in ds:
        raise ValueError(f"NetCDF variable not found: {nc_variable}")

    da = ds[nc_variable]
    roles = identify_dimension_roles(ds)
    metadata: dict[str, Any] = {
        "nc_variable": nc_variable,
        "selected_dimensions": {},
        "selected_coordinate_values": {},
        "init_time_auto_selected": False,
        "valid_time_source": None,
    }

    if time_index is not None:
        dim = _first_matching_dim(da, roles["time"] + roles["valid_time"], "time-index")
        da = da.isel({dim: time_index})
        metadata["selected_dimensions"][dim] = time_index
        metadata["selected_coordinate_values"][dim] = _coord_value(da, dim)
    if time_value is not None:
        dim = _first_matching_dim(da, roles["time"], "time-value")
        da, value = _select_exact_coord(da, dim, time_value, "time")
        metadata["selected_dimensions"][dim] = time_value
        metadata["selected_coordinate_values"][dim] = _json_value(value)
    if valid_time is not None:
        names = roles["valid_time"] + roles["time"]
        dim = next((name for name in names if name in da.dims), None)
        if dim is not None:
            da, value = _select_exact_coord(da, dim, valid_time, "valid-time")
            metadata["selected_dimensions"][dim] = valid_time
            metadata["selected_coordinate_values"][dim] = _json_value(value)
            metadata["valid_time"] = _parse_datetime(value).isoformat()
            metadata["valid_time_source"] = "netcdf"
        else:
            metadata["valid_time"] = _parse_datetime(valid_time).isoformat()
            metadata["valid_time_source"] = "cli"

    init_dim = None
    if init_index is not None:
        init_dim = _first_matching_dim(da, roles["init_time"], "init-index")
        da = da.isel({init_dim: init_index})
        metadata["selected_dimensions"][init_dim] = init_index
        metadata["selected_coordinate_values"][init_dim] = _coord_value(da, init_dim)
    elif init_time is not None:
        init_dim = _first_matching_dim(da, roles["init_time"], "init-time")
        da, value = _select_exact_coord(da, init_dim, init_time, "init-time")
        metadata["selected_dimensions"][init_dim] = init_time
        metadata["selected_coordinate_values"][init_dim] = _json_value(value)
    else:
        for name in roles["init_time"]:
            if name in da.dims and int(da.sizes[name]) == 1:
                init_dim = name
                da = da.isel({name: 0})
                metadata["selected_dimensions"][name] = 0
                metadata["selected_coordinate_values"][name] = _coord_value(da, name)
                metadata["init_time_auto_selected"] = True
                break

    init_value = None
    for name in roles["init_time"]:
        if name in metadata["selected_coordinate_values"]:
            init_value = metadata["selected_coordinate_values"][name]
            metadata["init_time"] = init_value
            break

    if lead_index is not None:
        dim = _first_matching_dim(da, roles["lead"], "lead-index")
        da = da.isel({dim: lead_index})
        metadata["selected_dimensions"][dim] = lead_index
        value = _coord_value(da, dim)
        metadata["selected_coordinate_values"][dim] = value
        if value is not None:
            metadata["lead_hour"] = normalize_lead_to_hours(value)
    elif lead_hour is not None:
        dim = _first_matching_dim(da, roles["lead"], "lead-hour")
        values = np.asarray(da.coords[dim].values if dim in da.coords else np.arange(da.sizes[dim]))
        hours = np.array([normalize_lead_to_hours(v) for v in values], dtype=float)
        matches = np.where(np.isclose(hours, float(lead_hour), rtol=0.0, atol=1e-6))[0]
        if matches.size == 0:
            raise ValueError(f"lead-hour {lead_hour!r} was not found on dimension {dim!r}")
        index = int(matches[0])
        da = da.isel({dim: index})
        metadata["selected_dimensions"][dim] = lead_hour
        metadata["selected_coordinate_values"][dim] = _json_value(values[index])
        metadata["lead_hour"] = float(lead_hour)

    if member_index is not None:
        dim = _first_matching_dim(da, roles["member"], "member-index")
        da = da.isel({dim: member_index})
        metadata["selected_dimensions"][dim] = member_index
        metadata["selected_coordinate_values"][dim] = _coord_value(da, dim)
        metadata["member_index"] = member_index
    elif member_name is not None:
        dim = _first_matching_dim(da, roles["member"], "member-name")
        values = np.asarray(da.coords[dim].values if dim in da.coords else np.arange(da.sizes[dim]))
        matches = np.where(values.astype(str) == str(member_name))[0]
        if matches.size == 0:
            raise ValueError(f"member-name {member_name!r} was not found on dimension {dim!r}")
        index = int(matches[0])
        da = da.isel({dim: index})
        metadata["selected_dimensions"][dim] = member_name
        metadata["selected_coordinate_values"][dim] = _json_value(values[index])
        metadata["member_name"] = member_name

    da = da.squeeze(drop=True)
    spatial = _spatial_dims(ds, da, roles)
    remaining = [dim for dim in da.dims if dim not in spatial]
    if remaining:
        raise ValueError(
            "Selected NetCDF slice is not 2D; remaining dimensions: "
            + ", ".join(remaining)
            + ". Add explicit time/init/lead/member selectors."
        )
    if len(da.dims) != 2:
        raise ValueError(f"Selected NetCDF slice must be 2D; got dimensions {list(da.dims)}")

    if metadata.get("valid_time") is None and init_value is not None and metadata.get("lead_hour") is not None:
        derived = _parse_datetime(init_value) + timedelta(hours=float(metadata["lead_hour"]))
        metadata["valid_time"] = derived.isoformat()
        metadata["valid_time_source"] = "derived_from_init_plus_lead"

    return da, metadata


def _regular_spacing(values: np.ndarray, label: str) -> float:
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{label} coordinate must be a 1D array with at least two values")
    diffs = np.diff(values.astype(float))
    if np.any(np.isclose(diffs, 0.0)):
        raise ValueError(f"{label} coordinate spacing cannot be zero")
    if not np.allclose(diffs, diffs[0], rtol=1e-5, atol=1e-8):
        raise ValueError(f"{label} coordinate is not regularly spaced")
    return float(abs(diffs[0]))


def write_xarray_slice_to_temp_geotiff(data_array: Any, temp_tif: str | Path, *, nodata: float = -999.0) -> None:
    """Write a selected rectilinear 1D lat/lon DataArray slice to GeoTIFF."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise ImportError("rasterio is required to write selected NetCDF slices.") from exc

    lat_name = next((name for name in LAT_NAMES if name in data_array.coords), None)
    lon_name = next((name for name in LON_NAMES if name in data_array.coords), None)
    if lat_name is None or lon_name is None:
        raise ValueError(
            "NetCDF variable has no readable CRS/geotransform. "
            "Use a CF-compliant gridded file or convert externally."
        )

    lat_coord = data_array.coords[lat_name]
    lon_coord = data_array.coords[lon_name]
    if lat_coord.ndim != 1 or lon_coord.ndim != 1:
        raise ValueError("Curvilinear NetCDF grids are not supported in this first version.")

    lat_dim = lat_coord.dims[0]
    lon_dim = lon_coord.dims[0]
    if lat_dim not in data_array.dims or lon_dim not in data_array.dims:
        raise ValueError("Latitude/longitude coordinates do not align with the selected 2D slice.")

    da = data_array.transpose(lat_dim, lon_dim)
    arr = np.asarray(da.values, dtype=np.float32)
    lat = np.asarray(lat_coord.values, dtype=float)
    lon = np.asarray(lon_coord.values, dtype=float)

    dx = _regular_spacing(lon, "longitude")
    dy = _regular_spacing(lat, "latitude")

    if lon[0] > lon[-1]:
        lon = lon[::-1]
        arr = arr[:, ::-1]
    if lat[0] < lat[-1]:
        lat = lat[::-1]
        arr = arr[::-1, :]

    transform = from_origin(float(lon.min() - dx / 2.0), float(lat.max() + dy / 2.0), dx, dy)
    arr = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)

    temp_tif = Path(temp_tif)
    temp_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        temp_tif,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)


def _unit_key(units: str) -> str:
    return units.strip().lower().replace("_", " ").replace("**", "^")


def _is_rate_unit(units: str) -> bool:
    key = _unit_key(units)
    rate_units = {
        "kg m-2 s-1",
        "kg/m2/s",
        "kg m^-2 s^-1",
        "mm/s",
        "mm s-1",
        "mm hr-1",
        "mm/hr",
        "mm/hour",
        "m/s",
    }
    return key in rate_units


def convert_netcdf_units(
    array: np.ndarray,
    *,
    variable: str,
    source_units: str,
    target_units: str,
    accumulation_mode: str = "step",
    accumulation_hours: float | None = None,
    nodata: float = -999.0,
) -> np.ndarray:
    """Convert selected NetCDF physical values before XMRG scaling."""
    if accumulation_mode == "total-since-init":
        raise ValueError(
            "total-since-init precipitation requires adjacent lead differencing; "
            "not supported in this first version."
        )
    if accumulation_mode not in {"step", "rate"}:
        raise ValueError("accumulation-mode must be step, rate, or total-since-init")

    out = np.asarray(array, dtype=np.float32).copy()
    valid = np.isfinite(out) & (out > -900)
    src = _unit_key(source_units)
    dst = _unit_key(target_units)

    if variable == "prep":
        if _is_rate_unit(source_units) or accumulation_mode == "rate":
            if accumulation_hours is None:
                raise ValueError(
                    "Rate units require explicit accumulation duration; "
                    "not supported in nc-to-xmrg first version."
                )
            seconds = float(accumulation_hours) * 3600.0
            if src in {"kg m-2 s-1", "kg/m2/s", "kg m^-2 s^-1", "mm/s", "mm s-1"}:
                out[valid] = out[valid] * seconds
            elif src in {"mm hr-1", "mm/hr", "mm/hour"}:
                out[valid] = out[valid] * float(accumulation_hours)
            elif src == "m/s":
                out[valid] = out[valid] * seconds * 1000.0
            else:
                raise ValueError(f"unsupported precipitation rate units: {source_units}")
        elif src in {"mm", "millimeter", "millimeters", "kg m-2", "kg/m2", "kg m^-2"}:
            pass
        else:
            raise ValueError(f"unsupported precipitation unit conversion: {source_units} -> {target_units}")

        if dst not in {"mm", "millimeter", "millimeters"}:
            raise ValueError(f"unsupported precipitation target units: {target_units}")
        out[~valid] = nodata
        return out

    if variable in {"tair", "tmax", "tmin"}:
        if dst not in {"f", "fahrenheit", "degf"}:
            raise ValueError(f"unsupported temperature target units: {target_units}")
        if src in {"f", "fahrenheit", "degf"}:
            pass
        elif src in {"c", "celsius", "degc"}:
            out[valid] = out[valid] * 9.0 / 5.0 + 32.0
        elif src in {"k", "kelvin"}:
            out[valid] = (out[valid] - 273.15) * 9.0 / 5.0 + 32.0
        else:
            raise ValueError(f"unsupported temperature unit conversion: {source_units} -> {target_units}")
        out[~valid] = nodata
        return out

    raise ValueError(f"unsupported RDHM variable: {variable}")


def _date_hour_from_valid_time(valid_time: str | datetime | None) -> tuple[date | None, int | None]:
    if valid_time is None:
        return None, None
    dt = _parse_datetime(valid_time)
    return dt.date(), dt.hour


def _resolve_output_path(
    *,
    output_xmrg: str | Path | None,
    output_dir: str | Path | None,
    variable: str,
    date_value: str | date | datetime | None,
    hour: int | None,
    valid_time: str | datetime | None,
    daily_precip: bool,
) -> tuple[Path, str | None, int | None]:
    if output_xmrg and output_dir:
        raise ValueError("Provide either output_xmrg or output_dir, not both")
    if not output_xmrg and not output_dir:
        raise ValueError("Provide either output_xmrg or output_dir")

    if valid_time is not None and (date_value is None or hour is None):
        vt_date, vt_hour = _date_hour_from_valid_time(valid_time)
        date_value = date_value or vt_date
        hour = hour if hour is not None else vt_hour

    if output_xmrg:
        return Path(output_xmrg), str(date_value) if date_value is not None else None, hour

    if variable == "prep" and daily_precip:
        if date_value is None:
            raise ValueError("--daily-precip requires --date")
        if hour is not None:
            raise ValueError("--daily-precip cannot be combined with --hour")
        filename = rdhm_filename("prep", date_value, daily=True)
    elif variable == "prep":
        if date_value is None or hour is None:
            raise ValueError("hourly prep output-dir requires --date + --hour or --valid-time")
        filename = rdhm_filename("prep", date_value, hour=hour)
    elif variable in {"tair", "tmax", "tmin"}:
        if daily_precip:
            raise ValueError("--daily-precip is invalid for temperature variables")
        if date_value is None or hour is None:
            raise ValueError(f"{variable} output-dir requires --date + --hour or --valid-time")
        filename = rdhm_filename(variable, date_value, hour=hour)
    else:
        raise ValueError(f"unsupported RDHM variable: {variable}")
    return Path(output_dir) / filename, str(date_value), hour


def nc_to_xmrg(
    *,
    input_path: str | Path,
    nc_variable: str,
    output_xmrg: str | Path | None = None,
    output_dir: str | Path | None = None,
    variable: str,
    date_value: str | date | datetime | None = None,
    hour: int | None = None,
    valid_time: str | None = None,
    daily_precip: bool = False,
    band: int | None = None,
    time_index: int | None = None,
    time_value: str | None = None,
    init_index: int | None = None,
    init_time: str | None = None,
    lead_index: int | None = None,
    lead_hour: float | None = None,
    member_index: int | None = None,
    member_name: str | None = None,
    source_units: str,
    target_units: str,
    target_grid: TargetGrid,
    target_domain_source: str | None = None,
    resampling: str = "bilinear",
    report: str | Path | None = None,
    accumulation_mode: str = "step",
    accumulation_hours: float | None = None,
) -> ValidationResult:
    """Convert one explicitly selected NetCDF 2D slice to RDHM XMRG."""
    variable = variable.lower()
    parse_time_selector(
        band=band,
        time_index=time_index,
        time_value=time_value,
        valid_time=valid_time,
        init_index=init_index,
        init_time=init_time,
        lead_index=lead_index,
        lead_hour=lead_hour,
        member_index=member_index,
        member_name=member_name,
    )

    if variable == "prep" and daily_precip and hour is not None:
        raise ValueError("--daily-precip cannot be combined with --hour")
    if variable in {"tair", "tmax", "tmin"} and daily_precip:
        raise ValueError("--daily-precip is invalid for temperature variables")

    selected: dict[str, Any] = {
        "nc_variable": nc_variable,
        "selected_dimensions": {},
        "selected_coordinate_values": {},
        "init_time_auto_selected": False,
        "valid_time_source": "cli" if valid_time else None,
        "valid_time": _parse_datetime(valid_time).isoformat() if valid_time else None,
    }
    with tempfile.TemporaryDirectory(prefix="hrapxmrg-nc-") as tmp:
        tmp_path = Path(tmp)
        temp_tif = tmp_path / "selected_slice.tif"
        if band is not None:
            try:
                import rasterio
            except ImportError as exc:
                raise ImportError("rasterio is required for NetCDF band selection.") from exc

            source = netcdf_variable_path(input_path, nc_variable)
            with rasterio.open(source) as src:
                arr = src.read(band).astype(np.float32)
                converted = convert_netcdf_units(
                    arr,
                    variable=variable,
                    source_units=source_units,
                    target_units=target_units,
                    accumulation_mode=accumulation_mode,
                    accumulation_hours=accumulation_hours,
                    nodata=src.nodata if src.nodata is not None else -999.0,
                )
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    count=1,
                    dtype="float32",
                    nodata=src.nodata if src.nodata is not None else -999.0,
                )
                with rasterio.open(temp_tif, "w", **profile) as dst:
                    dst.write(converted.astype(np.float32), 1)
            selected["band"] = band
        else:
            xr = _xarray()
            with xr.open_dataset(input_path, decode_times=True) as ds:
                da, selected = select_xarray_slice(
                    ds,
                    nc_variable,
                    time_index=time_index,
                    time_value=time_value,
                    valid_time=valid_time,
                    init_index=init_index,
                    init_time=init_time,
                    lead_index=lead_index,
                    lead_hour=lead_hour,
                    member_index=member_index,
                    member_name=member_name,
                )
                if valid_time is None and selected.get("valid_time"):
                    valid_time = selected["valid_time"]
                elif valid_time is not None:
                    selected["valid_time"] = _parse_datetime(valid_time).isoformat()
                    selected["valid_time_source"] = selected.get("valid_time_source") or "cli"

                converted = convert_netcdf_units(
                    np.asarray(da.values, dtype=np.float32),
                    variable=variable,
                    source_units=source_units,
                    target_units=target_units,
                    accumulation_mode=accumulation_mode,
                    accumulation_hours=accumulation_hours,
                )
                da = da.copy(data=converted)
                write_xarray_slice_to_temp_geotiff(da, temp_tif)

        output_path, report_date, report_hour = _resolve_output_path(
            output_xmrg=output_xmrg,
            output_dir=output_dir,
            variable=variable,
            date_value=date_value,
            hour=hour,
            valid_time=valid_time,
            daily_precip=daily_precip,
        )

        result = raster_to_xmrg(
            input_raster=temp_tif,
            output_xmrg=output_path,
            variable=variable,
            target_grid=target_grid,
            band=1,
            resampling=resampling,
            source_units=target_units,
            target_units=target_units,
        )

    nc_details = {
        "input_netcdf": str(input_path),
        "nc_variable": nc_variable,
        "variable": variable,
        "band": band,
        "time_index": time_index,
        "time_value": time_value,
        "init_index": init_index,
        "init_time": selected.get("init_time") or init_time,
        "init_time_auto_selected": selected.get("init_time_auto_selected", False),
        "lead_index": lead_index,
        "lead_hour": selected.get("lead_hour", lead_hour),
        "valid_time": selected.get("valid_time") or valid_time,
        "valid_time_source": selected.get("valid_time_source"),
        "member_index": selected.get("member_index", member_index),
        "member_name": selected.get("member_name", member_name),
        "source_units": source_units,
        "target_units": target_units,
        "accumulation_mode": accumulation_mode,
        "accumulation_hours": accumulation_hours,
        "selected_dimensions": selected.get("selected_dimensions", {}),
        "selected_coordinate_values": selected.get("selected_coordinate_values", {}),
    }
    details = dict(result.details)
    details.update(nc_details)
    updated = ValidationResult(result.ok, result.message, details)

    if report:
        manifest = raster_to_xmrg_manifest(
            input_raster=str(input_path),
            output_xmrg=output_path,
            variable=variable,
            target_grid=target_grid,
            validation=updated,
            date=report_date,
            hour=report_hour,
            daily_precip=daily_precip,
            source_units=source_units,
            target_units=target_units,
            target_domain_source=target_domain_source,
        )
        manifest["netcdf"] = nc_details
        write_raster_to_xmrg_manifest(report, manifest)

    return updated
