from hrapxmrg.cli import build_parser


def test_cli_parser_builds():
    parser = build_parser()
    assert parser.prog == "hrapxmrg"
