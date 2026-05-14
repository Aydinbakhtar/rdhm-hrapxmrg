"""Command-line interface for hrapxmrg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import ascii_to_xmrg, xmrg_to_ascii, format_xmrg_report
from .validate import validate_xmrg_file, scan_rdhm_log_for_missing_forcing
from .xmrg import read_xmrg


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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
