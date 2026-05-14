# RDHM asctoxmrg compatibility test

Test workspace on aral:

/data-alph/aydin/project/RDHM/rdhm_xmrg_py

RDHM asctoxmrg executable:

/data-alph/zcui/rdhm_trunk_3.5.14.3/asctoxmrg/asctoxmrg

Synthetic HRAP ASCII test:
- ncols = 64
- nrows = 87
- xor/xllcorner = 341
- yor/yllcorner = 313
- no nodata cells
- controlled physical values from 0.23 to 15.13

Python command:

hrapxmrg ascii-to-xmrg \
  --input ascii/synthetic_valid_64x87.asc \
  --output python_xmrg/synthetic_python.gz \
  --variable prep \
  --header-type int32 \
  --dtype int16 \
  --scale 100 \
  --orientation flipud

RDHM asctoxmrg command:

/data-alph/zcui/rdhm_trunk_3.5.14.3/asctoxmrg/asctoxmrg \
  -i ascii/synthetic_valid_64x87.asc \
  -o asctoxmrg_xmrg/synthetic_asctoxmrg \
  -f xmrg

Result:
- Python metadata: int32 header, int16 data, xor=341, yor=313, maxx=64, maxy=87
- asctoxmrg metadata: int32 header, int16 data, xor=341, yor=313, maxx=64, maxy=87
- Stored stats both: min=23, mean=768, max=1513
- Physical values with divisor 100: min=0.23, mean=7.68, max=15.13
- max absolute difference = 0
- number different = 0

Conclusion:
The Python writer matches RDHM asctoxmrg numerically when using:
- header_type = int32
- dtype = int16
- scale = 100
- orientation = flipud

The files are not byte-identical because asctoxmrg writes a secondary metadata header, but the decoded numeric grid is identical.
