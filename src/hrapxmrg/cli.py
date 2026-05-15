"""Command-line interface for hrapxmrg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .batch import batch_raster_to_xmrg
from .filenames import rdhm_filename
from .pipeline import (
    ascii_to_xmrg,
    format_xmrg_report,
    raster_to_xmrg,
    raster_to_xmrg_manifest,
    write_raster_to_xmrg_manifest,
    xmrg_to_ascii,
)
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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
