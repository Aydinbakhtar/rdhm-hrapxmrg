# Project rules for Upper Rio Grande RDHM XMRG forcing

## 1. Temperature scaling

Authoritative rule for this RDHM setup:

```text
tair/tmax/tmin XMRG = Fahrenheit x 100 stored as int16
```

Conversion from PRISM:

```text
PRISM raw = Celsius float32
F = C x 9/5 + 32
stored int16 = round(F x 100)
```

Smoke-test evidence showed Fx10 files produced RDHM tair output 10x too small.
Do not use Fx10 for this project unless a new one-day smoke test proves it.

## 2. Temperature filenames

Use variable prefixes:

```text
tairMMDDYYYY12z.gz
tmaxMMDDYYYY12z.gz
tminMMDDYYYY12z.gz
```

For hourly runtime folders:

```text
tairMMDDYYYY00z.gz ... tairMMDDYYYY23z.gz
```

## 3. Precipitation naming

Precipitation naming must be confirmed against the existing working folder:

```text
forcing/obs/xmrg/prep/YYYY
```

Do not force an `xmrg` prefix until confirmed.

## 4. HRAP domain

Current target domain:

```text
ncols = 64
nrows = 87
xllcorner / xor = 341
yllcorner / yor = 313
cellsize = 1 HRAP cell
```

## 5. Validation before calibration

No calibration run should be trusted until:

- XMRG files read back correctly.
- Dimensions and HRAP origin match.
- Temperature values are physically plausible after dividing stored int16 by 100.
- Log has no missing tair/tmax/tmin messages.
- A one-day or short-period smoke test passes.
- Output time series are not identical across parameter runs when parameters differ.
