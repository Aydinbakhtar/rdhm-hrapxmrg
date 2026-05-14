"""CLI entry point for hrapxmrg.

Commands
--------
inspect
    Read an XMRG file and print metadata and stored-value statistics.
ascii-to-xmrg
    Convert an HRAP ASCII grid to a gzip-compressed XMRG file.
validate
    Check a generated XMRG file against variable-specific rules.
scan-log
    Scan an RDHM log file for missing-forcing messages.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

from .config import DEFAULT_DOMAIN, HRAPDomain
from .readers.hrap_ascii import HRAPAsciiReader
from .units import celsius_to_f100_int16
from .validator import (
    ValidationError,
    scan_rdhm_log,
    summarize_log_scan,
    validate_filename,
    validate_xmrg_file,
)
from .variable_registry import get_variable
from .xmrg_io import XMRGReader, XMRGWriter


# ------------------------------------------------------------------ #
# CLI root
# ------------------------------------------------------------------ #

@click.group()
@click.version_option(package_name="hrapxmrg")
def main() -> None:
    """hrapxmrg – HRAP/XMRG preprocessing toolkit for HL-RDHM/RDHM."""


# ------------------------------------------------------------------ #
# inspect
# ------------------------------------------------------------------ #

@main.command("inspect")
@click.argument("xmrg_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--nodata",
    default=None,
    type=float,
    help="Nodata sentinel value (excluded from statistics).  "
         "If not given, no cells are excluded.",
)
@click.option(
    "--variable", "-v",
    default=None,
    help="Variable name (tair/tmax/tmin/prep) for scale-aware reporting.",
)
def cmd_inspect(xmrg_file: str, nodata: float | None, variable: str | None) -> None:
    """Read an XMRG file and print metadata and stored-value statistics.

    XMRG_FILE is the path to a gzip-compressed XMRG binary file.
    """
    try:
        reader = XMRGReader(xmrg_file)
    except Exception as exc:
        click.echo(f"ERROR: Could not read {xmrg_file}: {exc}", err=True)
        sys.exit(1)

    d = reader.domain
    click.echo(f"File       : {xmrg_file}")
    click.echo(f"Domain     : xor={d.xor}, yor={d.yor}, maxx={d.maxx}, maxy={d.maxy}")
    click.echo(f"Grid size  : {d.ncols} columns × {d.nrows} rows")
    click.echo(f"Data dtype : {reader.dtype}")
    click.echo(f"Ext header : {'yes' if reader.has_extended_header else 'no'}")

    stats = reader.stats(nodata=nodata)
    click.echo(f"Stored min : {stats['min']:.4g}")
    click.echo(f"Stored max : {stats['max']:.4g}")
    click.echo(f"Stored mean: {stats['mean']:.4g}")
    click.echo(f"Nodata cnt : {stats['nodata_count']} / {stats['count']}")

    # Variable-aware physical reporting
    if variable is not None:
        try:
            spec = get_variable(variable)
        except KeyError as exc:
            click.echo(f"WARNING: {exc}", err=True)
        else:
            if spec.scale_convention == "F100_int16":
                click.echo(
                    f"\n--- Physical values for {variable!r} (F100 → °F) ---"
                )
                for key in ("min", "max", "mean"):
                    phys = spec.physical_from_stored(stats[key])
                    click.echo(f"  Physical {key}: {phys:.2f} °F")


# ------------------------------------------------------------------ #
# ascii-to-xmrg
# ------------------------------------------------------------------ #

@main.command("ascii-to-xmrg")
@click.argument("ascii_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option(
    "--variable", "-v",
    default=None,
    help="Variable name (tair/tmax/tmin/prep).  "
         "When specified, applies the correct unit conversion "
         "(e.g. Celsius → F100 int16 for temperature).",
)
@click.option(
    "--dtype",
    type=click.Choice(["int16", "float32"]),
    default="int16",
    show_default=True,
    help="Storage data type for the XMRG binary.",
)
@click.option(
    "--nodata-out",
    default=-9999,
    type=int,
    show_default=True,
    help="Stored nodata sentinel written for missing cells.",
)
def cmd_ascii_to_xmrg(
    ascii_file: str,
    output_file: str,
    variable: str | None,
    dtype: str,
    nodata_out: int,
) -> None:
    """Convert an HRAP ASCII grid to a gzip-compressed XMRG file.

    ASCII_FILE is the source HRAP ASCII raster (ESRI ASCII format with
    HRAP coordinates).

    OUTPUT_FILE is the destination XMRG .gz file path.

    When --variable is 'tair', 'tmax', or 'tmin', the ASCII values are
    treated as raw Celsius and converted to int16 Fahrenheit × 100
    (F = C*9/5+32, stored=round(F*100)) following the authoritative
    RDHM temperature convention.
    """
    # --- Read ASCII ---
    try:
        reader = HRAPAsciiReader(ascii_file)
    except Exception as exc:
        click.echo(f"ERROR reading ASCII: {exc}", err=True)
        sys.exit(1)

    data_f64 = reader.data.copy()  # north-first, float64
    nodata_mask = reader.nodata_mask()
    domain = reader.domain

    # --- Unit conversion ---
    storage_dtype = np.int16 if dtype == "int16" else np.float32

    if variable is not None and variable in ("tair", "tmax", "tmin"):
        click.echo(
            f"Converting Celsius → F100 int16 (F=C*9/5+32, stored=round(F*100)) "
            f"for variable {variable!r}."
        )
        out_data = celsius_to_f100_int16(
            data_f64, nodata_mask=nodata_mask, nodata_stored=nodata_out
        )
    else:
        # Raw pass-through with nodata substitution
        if storage_dtype == np.int16:
            out_data = np.where(nodata_mask, nodata_out, np.round(data_f64)).astype(np.int16)
        else:
            out_data = np.where(
                nodata_mask, float(nodata_out), data_f64
            ).astype(np.float32)

    # --- Write XMRG ---
    writer = XMRGWriter(
        domain=domain,
        storage_dtype=storage_dtype,
        rows_north_first=True,  # ASCII is north-first; writer flips to south-first
    )
    try:
        writer.write(out_data, output_file)
    except Exception as exc:
        click.echo(f"ERROR writing XMRG: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Written: {output_file}")
    click.echo(
        f"Domain : xor={domain.xor}, yor={domain.yor}, "
        f"maxx={domain.maxx}, maxy={domain.maxy}"
    )
    click.echo(f"Dtype  : {out_data.dtype}")


# ------------------------------------------------------------------ #
# validate
# ------------------------------------------------------------------ #

@main.command("validate")
@click.argument("xmrg_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--variable", "-v",
    default=None,
    help="Variable name (tair/tmax/tmin/prep) for range validation.",
)
@click.option(
    "--xor", default=None, type=int, help="Expected HRAP xor (overrides default domain)."
)
@click.option(
    "--yor", default=None, type=int, help="Expected HRAP yor (overrides default domain)."
)
@click.option(
    "--maxx", default=None, type=int, help="Expected HRAP maxx (overrides default domain)."
)
@click.option(
    "--maxy", default=None, type=int, help="Expected HRAP maxy (overrides default domain)."
)
@click.option(
    "--nodata",
    default=None,
    type=float,
    help="Stored nodata sentinel (excluded from range checks).",
)
@click.option(
    "--check-filename/--no-check-filename",
    default=False,
    help="Also validate the filename against RDHM naming conventions.",
)
def cmd_validate(
    xmrg_file: str,
    variable: str | None,
    xor: int | None,
    yor: int | None,
    maxx: int | None,
    maxy: int | None,
    nodata: float | None,
    check_filename: bool,
) -> None:
    """Check a generated XMRG file against variable-specific rules.

    XMRG_FILE is the path to a gzip-compressed XMRG binary file.

    Exit code 0 means validation passed; exit code 1 means failed.
    """
    # Build expected domain
    expected_domain: HRAPDomain | None = None
    if any(v is not None for v in (xor, yor, maxx, maxy)):
        base = DEFAULT_DOMAIN
        expected_domain = HRAPDomain(
            xor=xor if xor is not None else base.xor,
            yor=yor if yor is not None else base.yor,
            maxx=maxx if maxx is not None else base.maxx,
            maxy=maxy if maxy is not None else base.maxy,
        )

    result = validate_xmrg_file(
        xmrg_file,
        variable=variable,
        expected_domain=expected_domain,
        nodata=nodata,
    )

    if check_filename and variable is not None:
        fname_result = validate_filename(Path(xmrg_file).name, variable=variable)
        result.errors.extend(fname_result.errors)
        result.warnings.extend(fname_result.warnings)

    click.echo(str(result))
    if not result.passed:
        sys.exit(1)


# ------------------------------------------------------------------ #
# scan-log
# ------------------------------------------------------------------ #

@main.command("scan-log")
@click.argument("log_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--variable", "-v",
    "variables",
    multiple=True,
    default=("tair", "tmax", "tmin"),
    show_default=True,
    help="Variable(s) to check for missing-forcing messages. "
         "May be given multiple times.",
)
@click.option(
    "--fail-on-missing/--no-fail-on-missing",
    default=True,
    show_default=True,
    help="Exit with code 1 if any missing messages are found.",
)
def cmd_scan_log(
    log_file: str,
    variables: tuple[str, ...],
    fail_on_missing: bool,
) -> None:
    """Scan an RDHM log for missing-forcing messages.

    LOG_FILE is the path to an RDHM run log (plain text).

    Example: ``hrapxmrg scan-log rdhm_run.log --variable tair --variable tmax``
    """
    try:
        matches = scan_rdhm_log(log_file, variables=list(variables))
    except FileNotFoundError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)

    summary = summarize_log_scan(matches)
    click.echo(summary)

    if fail_on_missing:
        any_missing = any(len(hits) > 0 for hits in matches.values())
        if any_missing:
            sys.exit(1)
