from hrapxmrg.cli import build_parser


def test_cli_parser_builds():
    parser = build_parser()
    assert parser.prog == "hrapxmrg"


def test_cli_parser_includes_filename_command():
    parser = build_parser()
    args = parser.parse_args(["filename", "--variable", "prep", "--date", "1999-05-01", "--daily"])
    assert args.command == "filename"
