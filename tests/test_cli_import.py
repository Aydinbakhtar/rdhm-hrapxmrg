from hrapxmrg.cli import build_parser


def test_cli_parser_builds():
    parser = build_parser()
    assert parser.prog == "hrapxmrg"


def test_cli_parser_includes_filename_command():
    parser = build_parser()
    args = parser.parse_args(["filename", "--variable", "prep", "--date", "1999-05-01", "--daily"])
    assert args.command == "filename"


def test_cli_parser_includes_batch_raster_to_xmrg_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "batch-raster-to-xmrg",
            "--input-dir",
            "input",
            "--pattern",
            "*.tif",
            "--date-regex",
            r"(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})",
            "--variable",
            "prep",
            "--daily-precip",
            "--output-dir",
            "output",
            "--target-config",
            "domain.yaml",
        ]
    )
    assert args.command == "batch-raster-to-xmrg"


def test_cli_parser_includes_prism_commands():
    parser = build_parser()
    info_args = parser.parse_args(["prism-info", "--input", "prism_ppt_20250101.zip"])
    assert info_args.command == "prism-info"

    convert_args = parser.parse_args(
        [
            "prism-to-xmrg",
            "--input",
            "prism_ppt_20250101.zip",
            "--output-dir",
            "out",
            "--daily-precip",
            "--target-config",
            "domain.yaml",
        ]
    )
    assert convert_args.command == "prism-to-xmrg"


def test_cli_parser_includes_daily_to_hourly_commands():
    parser = build_parser()
    ppt_args = parser.parse_args(
        [
            "daily-ppt-to-hourly-xmrg",
            "--input",
            "daily_ppt.tif",
            "--output-dir",
            "out",
            "--date",
            "2025-01-01",
            "--target-config",
            "domain.yaml",
        ]
    )
    assert ppt_args.command == "daily-ppt-to-hourly-xmrg"

    temp_args = parser.parse_args(
        [
            "daily-temp-to-hourly-tair",
            "--tmin",
            "tmin.tif",
            "--tmax",
            "tmax.tif",
            "--output-dir",
            "out",
            "--date",
            "2025-01-01",
            "--target-config",
            "domain.yaml",
        ]
    )
    assert temp_args.command == "daily-temp-to-hourly-tair"
