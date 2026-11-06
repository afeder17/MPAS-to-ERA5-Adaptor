# MPAS-to-ERA5-Adaptor

Converts MPAS-CAM model output to ERA5-compatible format for use with WAM2layers moisture tracking.

## Features
- Bilinear (Delaunay) interpolation for u, v, q to preserve spatial coherence
- Nearest-neighbor for P, E, SP (local quantities)  
- Configurable output grid (default 1°)
- Numba-accelerated vertical interpolation
- Multiprocessing for batch conversion

## Scripts

- `mpas_to_era5.py` — original converter (uniform-mesh output in a single input directory).
- `mpas_to_era5_1deg_rr.py` — converter used for the regionally refined (120–15 km) runs: discovers case output across campaign and scratch roots (`--case`), handles gzipped files, vectorised regridding. Also runs on uniform-mesh output (`--input-dir`).

## Changelog

### v2.0.0 (2026-11-06) — accumulation-interval correction
`tp` and `e` are written in the ERA5 convention as accumulations over the data interval
(metres of water per step). Versions up to v1 multiplied the instantaneous MPAS rates
(`PRECT`, `LHFLX`) by 3600 s, which is correct only for hourly input; for the 3-hourly
MPAS streams used in our work it under-states `tp` and `e` by a factor of 3. Both
scripts now read the interval from the file's time axis (`_accum_seconds`), with a
3-hourly fallback and a `--accum-seconds` override in `mpas_to_era5_1deg_rr.py`.
Output produced with earlier versions from 3-hourly input can be corrected by
multiplying `tp` and `e` by 3; the other variables are unaffected.

### v1 (2026-04)
Bilinear wind interpolation, 1-degree grid, PRECC+PRECL handling.
