import pytest

from hrapxmrg.filenames import rdhm_filename


def test_daily_precip_filename_uses_next_day_00z():
    assert rdhm_filename("prep", "1999-05-01", daily=True) == "xmrg0502199900z.gz"


def test_hourly_precip_filename_uses_end_time():
    assert rdhm_filename("prep", "1999-05-01", hour=1) == "xmrg0501199901z.gz"


def test_midnight_precip_filename_uses_00z_end_time():
    assert rdhm_filename("prep", "1999-05-01", hour=0) == "xmrg0501199900z.gz"


def test_temperature_filename_uses_valid_time():
    assert rdhm_filename("tair", "1999-05-01", hour=12) == "tair0501199912z.gz"


def test_tmax_and_tmin_filenames():
    assert rdhm_filename("tmax", "1999-05-01", hour=12) == "tmax0501199912z.gz"
    assert rdhm_filename("tmin", "1999-05-01", hour=12) == "tmin0501199912z.gz"


def test_filename_can_suppress_gzip_suffix():
    assert rdhm_filename("tair", "1999-05-01", hour=12, suffix_gz=False) == "tair0501199912z"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variable": "tair", "date_value": "1999-05-01", "daily": True},
        {"variable": "prep", "date_value": "1999-05-01"},
        {"variable": "tair", "date_value": "1999-05-01"},
        {"variable": "prep", "date_value": "1999-05-01", "hour": 1, "daily": True},
        {"variable": "prep", "date_value": "1999-05-01", "hour": 24},
        {"variable": "badvar", "date_value": "1999-05-01", "hour": 12},
    ],
)
def test_invalid_filename_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        rdhm_filename(**kwargs)
