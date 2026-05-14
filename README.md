# rdhm-hrapxmrg

A Python library and command-line tool for building **RDHM-ready HRAP/XMRG forcing files**.

This repository is designed for HL-RDHM / RDHM preprocessing workflows where forcing data may start as:

- HRAP ASCII grids
- PRISM BIL/ZIP files
- NetCDF climate or forecast files
- GeoTIFF rasters
- Existing XMRG files used as structural templates

The first development goal is conservative and testable:

1. Read HRAP ASCII.
2. Write RDHM-compatible gzipped XMRG.
3. Read XMRG back.
4. Validate header, shape, scale, filename, and physical range.
5. Then add PRISM / NetCDF / GeoTIFF reprojection.

## Project-specific authoritative rule

For the current Upper Rio Grande / Rio Grande Headwaters RDHM setup:

```text
Temperature XMRG for tair/tmax/tmin = Fahrenheit x 100 stored as int16
```

PRISM temperature files are raw Celsius float32. Convert as:

```python
F = C * 9.0 / 5.0 + 32.0
stored_int16 = round(F * 100)
```

Do **not** use Fahrenheit x 10 for this project unless a new one-day RDHM smoke test proves that convention for another setup.

## Suggested repo name

GitHub repository:

```text
rdhm-hrapxmrg
```

Python import name:

```python
import hrapxmrg
```

CLI command:

```bash
hrapxmrg --help
```

## Install for development

```bash
git clone git@github.com:<your-user>/rdhm-hrapxmrg.git
cd rdhm-hrapxmrg
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest
```

For geospatial raster support later:

```bash
python -m pip install -e ".[geo,dev]"
```

## First safe milestone

Use one existing HRAP ASCII file and one known RDHM/asctoxmrg XMRG file:

```bash
hrapxmrg ascii-to-xmrg \
  --input /data/aydin/project/RDHM/rdhm-prep_05/forcing/obs/hrap_asc/ppt/0101200012z.asc \
  --output /tmp/test_xmrg/0101200012z.gz \
  --variable prep \
  --header-type int32 \
  --dtype int16 \
  --scale 1 \
  --orientation as-is
```

Then read it back:

```bash
hrapxmrg inspect /tmp/test_xmrg/0101200012z.gz
```

For temperature:

```bash
hrapxmrg ascii-to-xmrg \
  --input tair_01012015_hrap_F.asc \
  --output /tmp/tair0101201512z.gz \
  --variable tair \
  --dtype int16 \
  --scale 100
```

## Development philosophy

This library should refuse to produce suspicious forcing files. It should never silently create wrong RDHM input.

Validation gates should check:

- filename format
- HRAP domain
- XMRG record structure
- data type
- readback values
- physical ranges
- missing values
- file counts
- RDHM smoke-test logs
