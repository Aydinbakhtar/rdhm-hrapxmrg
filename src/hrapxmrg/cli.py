"""Command-line interface for hrapxmrg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .batch import batch_raster_to_xmrg
from .filenames import rdhm_filename
from .netcdf import inspect_netcdf, nc_to_xmrg
from .netcdf_batch import batch_forecast_nc_to_xmrg
from .pipeline import (
    ascii_to_xmrg,
    format_xmrg_report,
    raster_to_xmrg,
    raster_to_xmrg_manifest,
    write_raster_to_xmrg_manifest,
    xmrg_to_ascii,
)
from .prism import prism_info, prism_to_xmrg
from .prism_batch import batch_prism_hourly_prep, batch_prism_hourly_tair
from .temporal import daily_raster_ppt_to_hourly_xmrg, daily_temp_to_hourly_tair_xmrg
from .validate import validate_xmrg_file, scan_rdhm_log_for_missing_forcing
from .xmrg import read_xmrg

from .ascii_writer import write_ascii_grid
from .regrid import convert_units, reproject_raster_to_hrap

from .domain import (
    con_header_consistency_warnings,
    format_domain_comparison_report,
    format_domain_report,
    target_grid_from_ascii_template,
    target_grid_from_con_file,
    target_grid_from_shapefile,
    target_grid_from_xmrg_template,
    target_grid_from_yaml,
    write_target_grid_yaml,
)



def cmd_inspect(args: argparse.Namespace) -> int:
    grid, meta = read_xmrg(args.path)
    valid = grid[grid > -900]
    out = {
        "path": str(args.path),
        "meta": meta.__dict__,
        "stored_min": float(valid.min()) if valid.size else None,
        "stored_mean": float(valid.mean()) if valid.size else None,
        "stored_max": float(valid.max()) if valid.size else None,
    }
    print(json.dumps(out, indent=2))
    return 0


def cmd_ascii_to_xmrg(args: argparse.Namespace) -> int:
    result = ascii_to_xmrg(
        input_ascii=args.input,
        output_xmrg=args.output,
        variable=args.variable,
        header_type=args.header_type,
        dtype=args.dtype,
        scale=args.scale,
        orientation=args.orientation,
        secondary_header=args.secondary_header,
    )
    print(json.dumps({"ok": result.ok, "message": result.message, "details": result.details}, indent=2))
    return 0 if result.ok else 2

def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_xmrg_file(args.path, variable=args.variable)
    print(json.dumps({"ok": result.ok, "message": result.message, "details": result.details}, indent=2))
    return 0 if result.ok else 2

def cmd_xmrg_to_ascii(args: argparse.Namespace) -> int:
    xmrg_to_ascii(
        args.input,
        args.output,
        args.variable,
        orientation=args.orientation,
        physical=not args.stored,
    )
    print(f"Wrote ASCII grid: {args.output}")
    return 0


def cmd_print_xmrg(args: argparse.Namespace) -> int:
    report = format_xmrg_report(args.input, variable=args.variable)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote report: {args.output}")
    else:
        print(report)

    return 0


def cmd_scan_log(args: argparse.Namespace) -> int:
    print(json.dumps(scan_rdhm_log_for_missing_forcing(args.path), indent=2))
    return 0


def cmd_filename(args: argparse.Namespace) -> int:
    print(
        rdhm_filename(
            args.variable,
            args.date,
            hour=args.hour,
            daily=args.daily,
            suffix_gz=not args.no_gzip,
        )
    )
    return 0



def _grid_from_domain_args(args: argparse.Namespace):
    provided = [
        args.ascii_template is not None,
        args.xmrg_template is not None,
        args.config is not None,
        args.con is not None,
        args.shp is not None,
    ]

    if sum(provided) != 1:
        raise SystemExit(
            "Provide exactly one domain source: "
            "--ascii-template, --xmrg-template, --config, --con, or --shp"
        )

    if args.ascii_template:
        return target_grid_from_ascii_template(args.ascii_template), f"ascii_template={args.ascii_template}"
    if args.xmrg_template:
        return target_grid_from_xmrg_template(args.xmrg_template), f"xmrg_template={args.xmrg_template}"
    if args.config:
        return target_grid_from_yaml(args.config), f"config={args.config}"
    if args.con:
        return target_grid_from_con_file(args.con, buffer_cells=args.buffer_cells), f"con={args.con}"
    if args.shp:
        return target_grid_from_shapefile(args.shp, buffer_cells=args.buffer_cells), f"shp={args.shp}"

    raise AssertionError("unreachable")


def cmd_describe_domain(args: argparse.Namespace) -> int:
    grid, source = _grid_from_domain_args(args)
    report = format_domain_report(grid, source=source)

    if args.con:
        warnings = con_header_consistency_warnings(args.con)
        if warnings:
            report += "\n\nCON header warnings:\n"
            report += "\n".join(f"  WARNING: {w}" for w in warnings)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote domain report: {args.output}")
    else:
        print(report)

    return 0


def cmd_domain_from_shp(args: argparse.Namespace) -> int:
    grid = target_grid_from_shapefile(args.shp, buffer_cells=args.buffer_cells)
    write_target_grid_yaml(args.output, grid, source=f"shapefile={args.shp}")
    print(format_domain_report(grid, source=f"shapefile={args.shp}"))
    print(f"\nWrote domain config: {args.output}")
    return 0


def cmd_domain_from_con(args: argparse.Namespace) -> int:
    grid = target_grid_from_con_file(args.con, buffer_cells=args.buffer_cells)
    write_target_grid_yaml(args.output, grid, source=f"con={args.con}")
    print(format_domain_report(grid, source=f"con={args.con}"))
    warnings = con_header_consistency_warnings(args.con)
    if warnings:
        print("\nCON header warnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
    print(f"\nWrote domain config: {args.output}")
    return 0


def cmd_check_domain(args: argparse.Namespace) -> int:
    con_grid = target_grid_from_con_file(args.con, buffer_cells=0) if args.con else None
    shp_grid = target_grid_from_shapefile(args.shp, buffer_cells=args.buffer_cells) if args.shp else None

    template_grid = None
    if args.ascii_template:
        template_grid = target_grid_from_ascii_template(args.ascii_template)
    elif args.xmrg_template:
        template_grid = target_grid_from_xmrg_template(args.xmrg_template)
    elif args.config:
        template_grid = target_grid_from_yaml(args.config)

    report = format_domain_comparison_report(
        con_grid=con_grid,
        shp_grid=shp_grid,
        template_grid=template_grid,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote domain comparison report: {args.output}")
    else:
        print(report)

    return 0


def _target_grid_and_source_from_target_args(args: argparse.Namespace):
    provided = [
        args.target_ascii_template is not None,
        args.target_xmrg_template is not None,
        args.target_config is not None,
        args.target_con is not None,
        args.target_shp is not None,
    ]

    if sum(provided) != 1:
        raise SystemExit(
            "Provide exactly one target domain source: "
            "--target-ascii-template, --target-xmrg-template, "
            "--target-config, --target-con, or --target-shp"
    )

    if args.target_ascii_template:
        return target_grid_from_ascii_template(args.target_ascii_template), f"ascii_template={args.target_ascii_template}"
    if args.target_xmrg_template:
        return target_grid_from_xmrg_template(args.target_xmrg_template), f"xmrg_template={args.target_xmrg_template}"
    if args.target_config:
        return target_grid_from_yaml(args.target_config), f"config={args.target_config}"
    if args.target_con:
        return target_grid_from_con_file(args.target_con, buffer_cells=args.buffer_cells), f"con={args.target_con}"
    if args.target_shp:
        return target_grid_from_shapefile(args.target_shp, buffer_cells=args.buffer_cells), f"shp={args.target_shp}"

    raise AssertionError("unreachable")


def _target_grid_from_target_args(args: argparse.Namespace):
    target_grid, _source = _target_grid_and_source_from_target_args(args)
    return target_grid


def cmd_raster_to_ascii(args: argparse.Namespace) -> int:
    target_grid = _target_grid_from_target_args(args)

    arr = reproject_raster_to_hrap(
        args.input,
        target_grid,
        band=args.band,
        resampling=args.resampling,
        dst_nodata=args.nodata,
    )

    if args.source_units and args.target_units:
        arr = convert_units(arr, source_units=args.source_units, target_units=args.target_units)

    write_ascii_grid(
        args.output,
        arr,
        target_grid,
        nodata=args.nodata,
        fmt=args.fmt,
    )

    print(f"Wrote HRAP ASCII grid: {args.output}")
    print(f"shape: {arr.shape}")
    print(f"target: xor={target_grid.xor}, yor={target_grid.yor}, maxx={target_grid.maxx}, maxy={target_grid.maxy}")

    valid = arr[np.isfinite(arr) & (arr > -900)]
    if valid.size:
        print(f"valid min/mean/max: {float(valid.min()):.4f}, {float(valid.mean()):.4f}, {float(valid.max()):.4f}")
    else:
        print("valid min/mean/max: no valid cells")

    return 0


def cmd_raster_to_xmrg(args: argparse.Namespace) -> int:
    if args.output and args.output_dir:
        raise SystemExit("Provide either --output or --output-dir, not both")
    if not args.output and not args.output_dir:
        raise SystemExit("Provide either --output or --output-dir")
    if args.daily_precip and args.variable != "prep":
        raise SystemExit("--daily-precip is only valid with --variable prep")

    output_xmrg = args.output
    if args.output_dir:
        filename = rdhm_filename(
            args.variable,
            args.date,
            hour=args.hour,
            daily=args.daily_precip,
            suffix_gz=not args.no_gzip_name,
        )
        output_xmrg = args.output_dir / filename

    target_grid, target_domain_source = _target_grid_and_source_from_target_args(args)

    result = raster_to_xmrg(
        input_raster=args.input,
        output_xmrg=output_xmrg,
        variable=args.variable,
        target_grid=target_grid,
        band=args.band,
        resampling=args.resampling,
        source_units=args.source_units,
        target_units=args.target_units,
        header_type=args.header_type,
        dtype=args.dtype,
        scale=args.scale,
        orientation=args.orientation,
        secondary_header=args.secondary_header,
    )

    if args.report:
        manifest = raster_to_xmrg_manifest(
            input_raster=args.input,
            output_xmrg=output_xmrg,
            variable=args.variable,
            target_grid=target_grid,
            validation=result,
            date=args.date,
            hour=args.hour,
            daily_precip=args.daily_precip,
            source_units=args.source_units,
            target_units=args.target_units,
            target_domain_source=target_domain_source,
            header_type=args.header_type,
            dtype=args.dtype,
            scale=args.scale,
            orientation=args.orientation,
            secondary_header=args.secondary_header,
        )
        write_raster_to_xmrg_manifest(args.report, manifest)

    print(json.dumps({"ok": result.ok, "message": result.message, "details": result.details}, indent=2))
    return 0 if result.ok else 2


def cmd_batch_raster_to_xmrg(args: argparse.Namespace) -> int:
    target_grid, target_domain_source = _target_grid_and_source_from_target_args(args)
    try:
        rows = batch_raster_to_xmrg(
            input_dir=args.input_dir,
            pattern=args.pattern,
            date_regex=args.date_regex,
            variable=args.variable,
            output_dir=args.output_dir,
            target_grid=target_grid,
            target_domain_source=target_domain_source,
            hour=args.hour,
            daily_precip=args.daily_precip,
            source_units=args.source_units,
            target_units=args.target_units,
            band=args.band,
            resampling=args.resampling,
            summary=args.summary,
            report_dir=args.report_dir,
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run,
            suffix_gz=not args.no_gzip_name,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "n_total": 0,
                    "n_ok": 0,
                    "n_failed": 1,
                    "summary": str(args.summary) if args.summary else None,
                    "message": str(exc),
                },
                indent=2,
            )
        )
        return 2

    n_total = len(rows)
    n_ok = sum(1 for row in rows if row["ok"])
    n_failed = n_total - n_ok
    ok = n_failed == 0 or args.continue_on_error
    print(
        json.dumps(
            {
                "ok": ok,
                "n_total": n_total,
                "n_ok": n_ok,
                "n_failed": n_failed,
                "summary": str(args.summary) if args.summary else None,
            },
            indent=2,
        )
    )
    return 0 if ok else 2


def cmd_prism_info(args: argparse.Namespace) -> int:
    print(json.dumps(prism_info(args.input, sample_stats=args.sample_stats), indent=2))
    return 0


def cmd_prism_to_xmrg(args: argparse.Namespace) -> int:
    target_grid, target_domain_source = _target_grid_and_source_from_target_args(args)
    try:
        result = prism_to_xmrg(
            input_path=args.input,
            target_grid=target_grid,
            prism_variable=args.prism_variable,
            rdhm_variable=args.rdhm_variable,
            date_value=args.date,
            hour=args.hour,
            daily_precip=args.daily_precip,
            output=args.output,
            output_dir=args.output_dir,
            report=args.report,
            resampling=args.resampling,
            target_domain_source=target_domain_source,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "message": str(exc), "details": {}}, indent=2))
        return 2

    print(json.dumps({"ok": result.ok, "message": result.message, "details": result.details}, indent=2))
    return 0 if result.ok else 2


def cmd_daily_ppt_to_hourly_xmrg(args: argparse.Namespace) -> int:
    target_grid, target_domain_source = _target_grid_and_source_from_target_args(args)
    result = daily_raster_ppt_to_hourly_xmrg(
        input_raster=args.input,
        output_dir=args.output_dir,
        target_grid=target_grid,
        date_value=args.date,
        method=args.method,
        band=args.band,
        resampling=args.resampling,
        report_dir=args.report_dir,
        summary=args.summary,
        target_domain_source=target_domain_source,
    )
    result.pop("rows", None)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def cmd_daily_temp_to_hourly_tair(args: argparse.Namespace) -> int:
    target_grid, target_domain_source = _target_grid_and_source_from_target_args(args)
    result = daily_temp_to_hourly_tair_xmrg(
        tmin_raster=args.tmin,
        tmax_raster=args.tmax,
        tmean_raster=args.tmean,
        output_dir=args.output_dir,
        target_grid=target_grid,
        date_value=args.date,
        method=args.method,
        tmin_hour=args.tmin_hour,
        tmax_hour=args.tmax_hour,
        band=args.band,
        resampling=args.resampling,
        report_dir=args.report_dir,
        summary=args.summary,
        target_domain_source=target_domain_source,
    )
    result.pop("rows", None)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def cmd_batch_prism_hourly_prep(args: argparse.Namespace) -> int:
    target_grid, target_domain_source = _target_grid_and_source_from_target_args(args)
    try:
        result = batch_prism_hourly_prep(
            input_dir=args.input_dir,
            pattern=args.pattern,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            method=args.method,
            band=args.band,
            resampling=args.resampling,
            summary=args.summary,
            report_dir=args.report_dir,
            continue_on_error=args.continue_on_error,
            target_grid=target_grid,
            target_domain_source=target_domain_source,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "variable": "prep", "message": str(exc)}, indent=2))
        return 2
    result.pop("rows", None)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def cmd_batch_prism_hourly_tair(args: argparse.Namespace) -> int:
    target_grid, target_domain_source = _target_grid_and_source_from_target_args(args)
    try:
        result = batch_prism_hourly_tair(
            tmin_dir=args.tmin_dir,
            tmax_dir=args.tmax_dir,
            tmean_dir=args.tmean_dir,
            tmin_pattern=args.tmin_pattern,
            tmax_pattern=args.tmax_pattern,
            tmean_pattern=args.tmean_pattern,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            method=args.method,
            tmin_hour=args.tmin_hour,
            tmax_hour=args.tmax_hour,
            band=args.band,
            resampling=args.resampling,
            summary=args.summary,
            report_dir=args.report_dir,
            continue_on_error=args.continue_on_error,
            target_grid=target_grid,
            target_domain_source=target_domain_source,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "variable": "tair", "message": str(exc)}, indent=2))
        return 2
    result.pop("rows", None)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2


def cmd_nc_info(args: argparse.Namespace) -> int:
    print(
        json.dumps(
            inspect_netcdf(
                args.input,
                sample_stats=args.sample_stats,
                max_subdatasets=args.max_subdatasets,
            ),
            indent=2,
        )
    )
    return 0


def cmd_nc_to_xmrg(args: argparse.Namespace) -> int:
    target_grid, target_domain_source = _target_grid_and_source_from_target_args(args)
    try:
        result = nc_to_xmrg(
            input_path=args.input,
            nc_variable=args.nc_variable,
            output_xmrg=args.output,
            output_dir=args.output_dir,
            variable=args.variable,
            date_value=args.date,
            hour=args.hour,
            valid_time=args.valid_time,
            daily_precip=args.daily_precip,
            band=args.band,
            time_index=args.time_index,
            time_value=args.time_value,
            init_index=args.init_index,
            init_time=args.init_time,
            lead_index=args.lead_index,
            lead_hour=args.lead_hour,
            member_index=args.member_index,
            member_name=args.member_name,
            source_units=args.source_units,
            target_units=args.target_units,
            target_grid=target_grid,
            target_domain_source=target_domain_source,
            resampling=args.resampling,
            report=args.report,
            accumulation_mode=args.accumulation_mode,
            accumulation_hours=args.accumulation_hours,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "message": str(exc), "details": {}}, indent=2))
        return 2

    print(json.dumps({"ok": result.ok, "message": result.message, "details": result.details}, indent=2))
    return 0 if result.ok else 2


def cmd_batch_forecast_nc_to_xmrg(args: argparse.Namespace) -> int:
    target_grid, target_domain_source = _target_grid_and_source_from_target_args(args)
    try:
        result = batch_forecast_nc_to_xmrg(
            input_path=args.input,
            nc_variable=args.nc_variable,
            variable=args.variable,
            source_units=args.source_units,
            target_units=args.target_units,
            output_dir=args.output_dir,
            target_grid=target_grid,
            target_domain_source=target_domain_source,
            init_time=args.init_time,
            init_index=args.init_index,
            lead_hours=args.lead_hours,
            lead_indices=args.lead_indices,
            lead_start=args.lead_start,
            lead_end=args.lead_end,
            lead_step=args.lead_step,
            all_leads=args.all_leads,
            member_index=args.member_index,
            member_name=args.member_name,
            all_members=args.all_members,
            member_output_mode=args.member_output_mode,
            member_prefix=args.member_prefix,
            accumulation_mode=args.accumulation_mode,
            accumulation_hours=args.accumulation_hours,
            allow_negative_diff=args.allow_negative_diff,
            negative_diff_tolerance=args.negative_diff_tolerance,
            resampling=args.resampling,
            summary=args.summary,
            report_dir=args.report_dir,
            continue_on_error=args.continue_on_error,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "message": str(exc)}, indent=2))
        return 2

    result.pop("rows", None)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 2



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrapxmrg",
        description="HRAP/XMRG tools for RDHM forcing generation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="Read an XMRG file and print metadata")
    p.add_argument("path", type=Path)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("ascii-to-xmrg", help="Convert HRAP ASCII grid to XMRG")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--variable", required=True, choices=["prep", "tair", "tmax", "tmin", "snow_ALAT"])
    p.add_argument("--header-type", choices=["int32", "float32"], default=None)
    p.add_argument("--dtype", choices=["int16", "float32"], default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--orientation", choices=["as-is", "flipud"], default="as-is")
    p.add_argument("--secondary-header", action="store_true")
    p.set_defaults(func=cmd_ascii_to_xmrg)

    p = sub.add_parser("validate", help="Validate one XMRG file")
    p.add_argument("path", type=Path)
    p.add_argument("--variable", required=True, choices=["prep", "tair", "tmax", "tmin", "snow_ALAT"])
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("xmrg-to-ascii", help="Convert XMRG to HRAP/ESRI ASCII grid")
    p.add_argument("input", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--variable", required=True, choices=["prep", "tair", "tmax", "tmin", "snow_ALAT"])
    p.add_argument("--orientation", choices=["as-is", "flipud"], default="flipud")
    p.add_argument("--stored", action="store_true", help="Write stored values instead of physical values")
    p.set_defaults(func=cmd_xmrg_to_ascii)

    p = sub.add_parser("print-xmrg", help="Print or save XMRG metadata and value summary")
    p.add_argument("input", type=Path)
    p.add_argument("--variable", choices=["prep", "tair", "tmax", "tmin", "snow_ALAT"], default=None)
    p.add_argument("--output", type=Path, default=None)
    p.set_defaults(func=cmd_print_xmrg)

    p = sub.add_parser("scan-log", help="Scan RDHM log for missing forcing messages")
    p.add_argument("path", type=Path)
    p.set_defaults(func=cmd_scan_log)

    p = sub.add_parser("filename", help="Print an RDHM forcing filename")
    p.add_argument("--variable", required=True, choices=["prep", "tair", "tmax", "tmin"])
    p.add_argument("--date", required=True)
    p.add_argument("--hour", type=int, default=None)
    p.add_argument("--daily", action="store_true")
    p.add_argument("--no-gzip", action="store_true")
    p.set_defaults(func=cmd_filename)

    p = sub.add_parser("describe-domain", help="Describe an HRAP target domain")
    p.add_argument("--ascii-template", type=Path, default=None)
    p.add_argument("--xmrg-template", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--con", type=Path, default=None)
    p.add_argument("--shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    p.set_defaults(func=cmd_describe_domain)

    p = sub.add_parser("domain-from-shp", help="Create HRAP domain YAML from shapefile")
    p.add_argument("--shp", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--buffer-cells", type=int, default=0)
    p.set_defaults(func=cmd_domain_from_shp)

    p = sub.add_parser("domain-from-con", help="Create HRAP domain YAML from RDHM .con file")
    p.add_argument("--con", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--buffer-cells", type=int, default=0)
    p.set_defaults(func=cmd_domain_from_con)

    p = sub.add_parser("check-domain", help="Compare HRAP domains from .con, shapefile, template, or config")
    p.add_argument("--con", type=Path, default=None)
    p.add_argument("--shp", type=Path, default=None)
    p.add_argument("--ascii-template", type=Path, default=None)
    p.add_argument("--xmrg-template", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    p.set_defaults(func=cmd_check_domain)

    p = sub.add_parser("raster-to-ascii", help="Reproject source raster to exact HRAP ASCII grid")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--band", type=int, default=1)
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "average"], default="bilinear")
    p.add_argument("--nodata", type=float, default=-999.0)
    p.add_argument("--fmt", default="%.2f")
    p.add_argument("--source-units", default=None, help="Optional source units, e.g. C, K, F, mm")
    p.add_argument("--target-units", default=None, help="Optional target units, e.g. F, C, mm")

    p.add_argument("--target-ascii-template", type=Path, default=None)
    p.add_argument("--target-xmrg-template", type=Path, default=None)
    p.add_argument("--target-config", type=Path, default=None)
    p.add_argument("--target-con", type=Path, default=None)
    p.add_argument("--target-shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)

    p.set_defaults(func=cmd_raster_to_ascii)

    p = sub.add_parser("raster-to-xmrg", help="Reproject source raster to exact HRAP grid and write XMRG")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--hour", type=int, default=None)
    p.add_argument("--daily-precip", action="store_true")
    p.add_argument("--no-gzip-name", action="store_true")
    p.add_argument("--variable", required=True, choices=["prep", "tair", "tmax", "tmin", "snow_ALAT"])
    p.add_argument("--band", type=int, default=1)
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "average"], default="bilinear")
    p.add_argument("--source-units", default=None, help="Optional source units, e.g. C, K, F, mm")
    p.add_argument("--target-units", default=None, help="Optional target units, e.g. F, C, mm")
    p.add_argument("--header-type", choices=["int32", "float32"], default=None)
    p.add_argument("--dtype", choices=["int16", "float32"], default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--orientation", choices=["as-is", "flipud"], default="flipud")
    p.add_argument("--secondary-header", action="store_true")

    p.add_argument("--target-ascii-template", type=Path, default=None)
    p.add_argument("--target-xmrg-template", type=Path, default=None)
    p.add_argument("--target-config", type=Path, default=None)
    p.add_argument("--target-con", type=Path, default=None)
    p.add_argument("--target-shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)

    p.set_defaults(func=cmd_raster_to_xmrg)

    p = sub.add_parser("batch-raster-to-xmrg", help="Batch convert rasters to RDHM XMRG")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--pattern", required=True)
    p.add_argument("--date-regex", required=True)
    p.add_argument("--variable", required=True, choices=["prep", "tair", "tmax", "tmin"])
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--hour", type=int, default=None)
    p.add_argument("--daily-precip", action="store_true")
    p.add_argument("--source-units", default=None, help="Optional source units, e.g. C, K, F, mm")
    p.add_argument("--target-units", default=None, help="Optional target units, e.g. F, C, mm")
    p.add_argument("--band", type=int, default=1)
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "average"], default="bilinear")
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-gzip-name", action="store_true")

    p.add_argument("--target-ascii-template", type=Path, default=None)
    p.add_argument("--target-xmrg-template", type=Path, default=None)
    p.add_argument("--target-config", type=Path, default=None)
    p.add_argument("--target-con", type=Path, default=None)
    p.add_argument("--target-shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)

    p.set_defaults(func=cmd_batch_raster_to_xmrg)

    p = sub.add_parser("prism-info", help="Inspect a PRISM BIL/ZIP/raster")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--sample-stats", action="store_true")
    p.set_defaults(func=cmd_prism_info)

    p = sub.add_parser("prism-to-xmrg", help="Convert PRISM BIL/ZIP/raster to RDHM XMRG")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--prism-variable", choices=["ppt", "tmean", "tmin", "tmax"], default=None)
    p.add_argument("--rdhm-variable", choices=["prep", "tair", "tmax", "tmin"], default=None)
    p.add_argument("--date", default=None)
    p.add_argument("--hour", type=int, default=None)
    p.add_argument("--daily-precip", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "average"], default="bilinear")

    p.add_argument("--target-ascii-template", type=Path, default=None)
    p.add_argument("--target-xmrg-template", type=Path, default=None)
    p.add_argument("--target-config", type=Path, default=None)
    p.add_argument("--target-con", type=Path, default=None)
    p.add_argument("--target-shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)

    p.set_defaults(func=cmd_prism_to_xmrg)

    p = sub.add_parser("daily-ppt-to-hourly-xmrg", help="Split daily precipitation raster to hourly prep XMRG")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--date", required=True)
    p.add_argument("--method", choices=["uniform"], default="uniform")
    p.add_argument("--band", type=int, default=1)
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "average"], default="bilinear")
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=None)

    p.add_argument("--target-ascii-template", type=Path, default=None)
    p.add_argument("--target-xmrg-template", type=Path, default=None)
    p.add_argument("--target-config", type=Path, default=None)
    p.add_argument("--target-con", type=Path, default=None)
    p.add_argument("--target-shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)

    p.set_defaults(func=cmd_daily_ppt_to_hourly_xmrg)

    p = sub.add_parser("daily-temp-to-hourly-tair", help="Generate hourly tair XMRG from daily tmin/tmax rasters")
    p.add_argument("--tmin", required=True, type=Path)
    p.add_argument("--tmax", required=True, type=Path)
    p.add_argument("--tmean", type=Path, default=None)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--date", required=True)
    p.add_argument("--method", choices=["sinusoidal"], default="sinusoidal")
    p.add_argument("--tmin-hour", type=int, default=6)
    p.add_argument("--tmax-hour", type=int, default=15)
    p.add_argument("--band", type=int, default=1)
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "average"], default="bilinear")
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=None)

    p.add_argument("--target-ascii-template", type=Path, default=None)
    p.add_argument("--target-xmrg-template", type=Path, default=None)
    p.add_argument("--target-config", type=Path, default=None)
    p.add_argument("--target-con", type=Path, default=None)
    p.add_argument("--target-shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)

    p.set_defaults(func=cmd_daily_temp_to_hourly_tair)

    p = sub.add_parser("batch-prism-hourly-prep", help="Batch convert daily PRISM ppt to hourly prep XMRG")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--pattern", default="prism_ppt_*.zip")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--method", choices=["uniform"], default="uniform")
    p.add_argument("--band", type=int, default=1)
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "average"], default="bilinear")
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--continue-on-error", action="store_true")

    p.add_argument("--target-ascii-template", type=Path, default=None)
    p.add_argument("--target-xmrg-template", type=Path, default=None)
    p.add_argument("--target-config", type=Path, default=None)
    p.add_argument("--target-con", type=Path, default=None)
    p.add_argument("--target-shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)

    p.set_defaults(func=cmd_batch_prism_hourly_prep)

    p = sub.add_parser("batch-prism-hourly-tair", help="Batch convert daily PRISM tmin/tmax to hourly tair XMRG")
    p.add_argument("--tmin-dir", required=True, type=Path)
    p.add_argument("--tmax-dir", required=True, type=Path)
    p.add_argument("--tmean-dir", type=Path, default=None)
    p.add_argument("--tmin-pattern", default="prism_tmin_*.zip")
    p.add_argument("--tmax-pattern", default="prism_tmax_*.zip")
    p.add_argument("--tmean-pattern", default="prism_tmean_*.zip")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--method", choices=["sinusoidal"], default="sinusoidal")
    p.add_argument("--tmin-hour", type=int, default=6)
    p.add_argument("--tmax-hour", type=int, default=15)
    p.add_argument("--band", type=int, default=1)
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "average"], default="bilinear")
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--continue-on-error", action="store_true")

    p.add_argument("--target-ascii-template", type=Path, default=None)
    p.add_argument("--target-xmrg-template", type=Path, default=None)
    p.add_argument("--target-config", type=Path, default=None)
    p.add_argument("--target-con", type=Path, default=None)
    p.add_argument("--target-shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)

    p.set_defaults(func=cmd_batch_prism_hourly_tair)

    p = sub.add_parser(
        "nc-info",
        help="Inspect NetCDF variables, coordinates, forecast dimensions, and raster subdatasets",
    )
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--sample-stats", action="store_true")
    p.add_argument("--max-subdatasets", type=int, default=20)
    p.set_defaults(func=cmd_nc_info)

    p = sub.add_parser("nc-to-xmrg", help="Convert one selected NetCDF 2D slice to RDHM XMRG")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--nc-variable", required=True)
    p.add_argument("--variable", required=True, choices=["prep", "tair", "tmax", "tmin"])
    p.add_argument("--source-units", required=True)
    p.add_argument("--target-units", required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)

    p.add_argument("--band", type=int, default=None)
    p.add_argument("--time-index", type=int, default=None)
    p.add_argument("--time-value", default=None)
    p.add_argument("--valid-time", default=None)
    p.add_argument("--init-index", type=int, default=None)
    p.add_argument("--init-time", default=None)
    p.add_argument("--lead-index", type=int, default=None)
    p.add_argument("--lead-hour", type=float, default=None)
    p.add_argument("--member-index", type=int, default=None)
    p.add_argument("--member-name", default=None)

    p.add_argument("--date", default=None)
    p.add_argument("--hour", type=int, default=None)
    p.add_argument("--daily-precip", action="store_true")
    p.add_argument("--accumulation-hours", type=float, default=None)
    p.add_argument("--accumulation-mode", choices=["step", "total-since-init", "rate"], default="step")
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "average"], default="bilinear")

    p.add_argument("--target-ascii-template", type=Path, default=None)
    p.add_argument("--target-xmrg-template", type=Path, default=None)
    p.add_argument("--target-config", type=Path, default=None)
    p.add_argument("--target-con", type=Path, default=None)
    p.add_argument("--target-shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)

    p.set_defaults(func=cmd_nc_to_xmrg)

    p = sub.add_parser("batch-forecast-nc-to-xmrg", help="Batch convert forecast NetCDF leads to RDHM XMRG")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--nc-variable", required=True)
    p.add_argument("--variable", required=True, choices=["prep", "tair", "tmax", "tmin"])
    p.add_argument("--source-units", required=True)
    p.add_argument("--target-units", required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    init_group = p.add_mutually_exclusive_group(required=True)
    init_group.add_argument("--init-time", default=None)
    init_group.add_argument("--init-index", type=int, default=None)
    p.add_argument("--lead-hours", default=None, help="Comma-separated lead hours, e.g. 1,2,3,6")
    p.add_argument("--lead-indices", default=None, help="Comma-separated lead coordinate indices")
    p.add_argument("--lead-start", type=float, default=None)
    p.add_argument("--lead-end", type=float, default=None)
    p.add_argument("--lead-step", type=float, default=None)
    p.add_argument("--all-leads", action="store_true")
    p.add_argument("--member-index", type=int, default=None)
    p.add_argument("--member-name", default=None)
    p.add_argument("--all-members", action="store_true")
    p.add_argument("--member-output-mode", choices=["flat", "subdirs"], default="subdirs")
    p.add_argument("--member-prefix", default="member")
    p.add_argument("--accumulation-mode", choices=["step", "rate", "total-since-init"], default="step")
    p.add_argument("--accumulation-hours", type=float, default=None)
    p.add_argument("--allow-negative-diff", action="store_true")
    p.add_argument("--negative-diff-tolerance", type=float, default=1e-6)
    p.add_argument("--resampling", choices=["nearest", "bilinear", "cubic", "average"], default="bilinear")
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--target-ascii-template", type=Path, default=None)
    p.add_argument("--target-xmrg-template", type=Path, default=None)
    p.add_argument("--target-config", type=Path, default=None)
    p.add_argument("--target-con", type=Path, default=None)
    p.add_argument("--target-shp", type=Path, default=None)
    p.add_argument("--buffer-cells", type=int, default=0)

    p.set_defaults(func=cmd_batch_forecast_nc_to_xmrg)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
