"""HRAP/XMRG tools for RDHM forcing generation."""

from .ascii_grid import AsciiGrid, read_ascii_grid
from .hrap import HRAP_CRS_PROJ4, TargetGrid
from .variables import VariableSpec, get_variable_spec
from .xmrg import XMRGMeta, read_xmrg, write_xmrg
from .pipeline import ascii_to_xmrg
from .validate import ValidationResult, validate_xmrg_file

__all__ = [
    "AsciiGrid",
    "read_ascii_grid",
    "HRAP_CRS_PROJ4",
    "TargetGrid",
    "VariableSpec",
    "get_variable_spec",
    "XMRGMeta",
    "read_xmrg",
    "write_xmrg",
    "ascii_to_xmrg",
    "ValidationResult",
    "validate_xmrg_file"
]
