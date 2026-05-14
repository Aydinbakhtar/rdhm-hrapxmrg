"""hrapxmrg – HRAP/XMRG preprocessing library and CLI for HL-RDHM/RDHM."""

__version__ = "0.1.0"

from .config import HRAPDomain, DEFAULT_DOMAIN
from .xmrg_io import XMRGReader, XMRGWriter
from .units import celsius_to_f100_int16, kelvin_to_f100_int16, fahrenheit_to_f100_int16

__all__ = [
    "__version__",
    "HRAPDomain",
    "DEFAULT_DOMAIN",
    "XMRGReader",
    "XMRGWriter",
    "celsius_to_f100_int16",
    "kelvin_to_f100_int16",
    "fahrenheit_to_f100_int16",
]
