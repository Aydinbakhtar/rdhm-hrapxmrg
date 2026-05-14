"""Tests for hrapxmrg.units – unit conversion functions."""

import numpy as np
import pytest

from hrapxmrg.units import (
    celsius_to_f100_int16,
    f100_int16_to_celsius,
    f100_int16_to_fahrenheit,
    fahrenheit_to_f100_int16,
    kelvin_to_f100_int16,
)


class TestCelsiusToF100:
    """celsius_to_f100_int16 – authoritative temperature conversion rule."""

    def test_freezing_point(self):
        """0 °C = 32 °F → stored 3200."""
        result = celsius_to_f100_int16(np.array([0.0]))
        assert result[0] == 3200

    def test_boiling_point(self):
        """100 °C = 212 °F → stored 21200."""
        result = celsius_to_f100_int16(np.array([100.0]))
        assert result[0] == 21200

    def test_minus_40(self):
        """-40 °C = -40 °F → stored -4000."""
        result = celsius_to_f100_int16(np.array([-40.0]))
        assert result[0] == -4000

    def test_prism_sample_tmax_10c(self):
        """PRISM tmax ≈ 10.2 °C → 50.36 °F → stored 5036."""
        result = celsius_to_f100_int16(np.array([10.2]))
        assert result[0] == 5036

    def test_prism_sample_tmin_minus3c(self):
        """PRISM tmin ≈ -3.45 °C → 25.79 °F → stored 2579."""
        result = celsius_to_f100_int16(np.array([-3.45]))
        # -3.45 * 9/5 + 32 = 25.79 → round(25.79*100) = 2579
        assert result[0] == 2579

    def test_array_shape_preserved(self):
        """Output shape matches input shape."""
        arr = np.zeros((3, 4))
        result = celsius_to_f100_int16(arr)
        assert result.shape == (3, 4)
        assert result.dtype == np.int16

    def test_nodata_mask(self):
        """Nodata cells are filled with nodata_stored sentinel."""
        arr = np.array([10.0, 20.0, 30.0])
        mask = np.array([False, True, False])
        result = celsius_to_f100_int16(arr, nodata_mask=mask, nodata_stored=-9999)
        assert result[1] == -9999
        assert result[0] != -9999
        assert result[2] != -9999

    def test_int16_dtype(self):
        """Output dtype is always int16."""
        result = celsius_to_f100_int16(np.array([5.0, -5.0, 15.0]))
        assert result.dtype == np.int16

    def test_not_fahrenheit_times_10(self):
        """Ensure we do NOT use F×10 (the old wrong convention).

        For 0 °C: F×10 would give 320, but correct F×100 = 3200.
        """
        result = celsius_to_f100_int16(np.array([0.0]))
        assert result[0] == 3200, "Must use F×100, not F×10"
        assert result[0] != 320, "Must NOT use F×10"

    def test_round_trip_approximately(self):
        """celsius → F100 → fahrenheit → celsius should be close."""
        c_orig = np.array([-20.0, 0.0, 10.0, 25.0])
        f100 = celsius_to_f100_int16(c_orig)
        c_back = f100_int16_to_celsius(f100)
        np.testing.assert_allclose(c_orig, c_back, atol=0.005)


class TestKelvinToF100:
    def test_freezing_0c_273k(self):
        """273.15 K = 0 °C = 32 °F → stored 3200."""
        result = kelvin_to_f100_int16(np.array([273.15]))
        assert result[0] == 3200

    def test_body_temp(self):
        """310.15 K = 37 °C = 98.6 °F → stored 9860."""
        result = kelvin_to_f100_int16(np.array([310.15]))
        assert result[0] == 9860

    def test_matches_celsius_path(self):
        """kelvin_to_f100 should produce the same result as celsius_to_f100."""
        k_arr = np.array([280.0, 295.0, 300.0])
        c_arr = k_arr - 273.15
        from_k = kelvin_to_f100_int16(k_arr)
        from_c = celsius_to_f100_int16(c_arr)
        np.testing.assert_array_equal(from_k, from_c)


class TestFahrenheitToF100:
    def test_freezing(self):
        """32 °F → stored 3200."""
        result = fahrenheit_to_f100_int16(np.array([32.0]))
        assert result[0] == 3200

    def test_round_trip(self):
        f_orig = np.array([50.0, 75.5, -10.0])
        f100 = fahrenheit_to_f100_int16(f_orig)
        f_back = f100_int16_to_fahrenheit(f100)
        np.testing.assert_allclose(f_orig, f_back, atol=0.005)


class TestF100ToFahrenheit:
    def test_nodata_becomes_nan(self):
        """Nodata sentinel becomes NaN."""
        arr = np.array([-9999, 3200], dtype=np.int16)
        result = f100_int16_to_fahrenheit(arr, nodata_stored=-9999)
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(32.0)

    def test_physical_values(self):
        """32 °F stored as 3200, 212 °F stored as 21200."""
        arr = np.array([3200, 21200], dtype=np.int16)
        result = f100_int16_to_fahrenheit(arr)
        assert result[0] == pytest.approx(32.0)
        assert result[1] == pytest.approx(212.0)
