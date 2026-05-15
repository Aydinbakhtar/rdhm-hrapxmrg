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
