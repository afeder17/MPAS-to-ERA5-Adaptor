"""
MPAS to ERA5 Converter - 1-DEGREE BILINEAR WINDS VERSION (RR-capable)

Written by Andrew Feder and Claude AI

Bilinear interpolation for u, v, q fields to preserve spatial coherence.
Nearest-neighbor for P, E, SP (local quantities).
Numba-accelerated vertical interpolation.

This version adds discovery for the regionally refined (RR 120-15km) runs,
mirroring the RR progress scanner:
  * searches BOTH campaign and scratch, recursively, campaign preferred
  * matches *.cam.h1a.YYYY-MM-DD(-SSSSS).nc and the .nc.gz variant
  * transparently decompresses gzipped history files
  * skips truncated files with an empty time dimension
The regridding machinery is mesh-agnostic, so the RR variable-resolution mesh
needs no special handling beyond the larger cell count.

RR usage (preferred): pass a case name, both roots are searched.
    python mpas_to_era5_1deg_rr.py \
        /glade/derecho/scratch/afeder/era5_1deg_pi_phoenix_rr \
        --case pi-rr120-15km.npe2560 \
        --time-shift -1.5 --start-year 1 --end-year 2 --end-month 3 --cores 64

Legacy usage (single directory, e.g. the 120km runs):
    python mpas_to_era5_1deg_rr.py \
        /glade/derecho/scratch/afeder/era5_1deg_newpi_phoenix \
        --input-dir /glade/derecho/scratch/dazlich/feder_amip1850.120km.npe512/run/cam.h1a \
        --time-shift -1.5 --start-year 1 --end-year 2 --end-month 3 --cores 64
"""

import os
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

import re
import gzip
import tempfile
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree, Delaunay
import argparse
ACCUM_SECONDS_DEFAULT = 10800.0  # fallback if the time axis has a single step
from pathlib import Path
from multiprocessing import Pool
import warnings
from datetime import timedelta
import time as time_module
from numba import njit, prange
import pickle

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# RR discovery config (mirrors the progress scanner)
# ---------------------------------------------------------------------------
CAMPAIGN_ROOT = Path('/glade/campaign/univ/ucsu0085/simulations-cpu/feder_phd_runs')
SCRATCH_ROOT  = Path('/glade/derecho/scratch/dazlich')

# matches '....h1a.0001-01' (monthly) and '....h1a.0001-01-01' or
# '....h1a.0001-01-01-00000' (sub-daily), with optional .gz. The day and
# seconds groups are optional so the same regex validates any h1a stream.
h1a_re = re.compile(
    r'\.h1a\.(\d{4})-(\d{2})(?:-(\d{2}))?(?:-(\d{5}))?\.nc(?:\.gz)?$'
)

# ERA5 hybrid levels
ERA5_LEVELS = np.array([20, 40, 60, 80, 90, 95, 100, 105, 110, 115, 120, 123, 125, 128, 130, 131, 132, 133, 134, 135, 136, 137])

A_COEFF = np.array([316.420746, 2900.391357, 10065.978516, 19620.042969, 20087.085938, 18489.707031, 15596.695313, 11901.339844, 8163.375000, 4993.796875, 2659.140625, 1659.476563, 1143.250000, 568.062500, 302.476563, 202.484375, 122.101563, 62.781250, 22.835938, 3.757813, 0.000000, 0.000000])

B_COEFF = np.array([0.000000, 0.000000, 0.000340, 0.068448, 0.203491, 0.308598, 0.438391, 0.576692, 0.704669, 0.809536, 0.887408, 0.922096, 0.940860, 0.963352, 0.975078, 0.980072, 0.984542, 0.988500, 0.991984, 0.995003, 0.997630, 1.000000])

# 1-DEGREE grid, DESCENDING latitude, 0-360 longitude
LAT_025 = np.arange(90, -90.1, -1.0)
LON_025 = np.arange(0, 360, 1.0)

# Globals for workers
WORKER_NN_MAPPING = None
WORKER_BILINEAR = None
WORKER_TIME_SHIFT = 0
WORKER_OUTPUT_DIR = None


# ---------------------------------------------------------------------------
# Gzip-aware open + RR file discovery (ported from the progress scanner)
# ---------------------------------------------------------------------------

def _accum_seconds(ds, fallback):
    """Accumulation interval of the input stream in seconds, from the file's time axis.

    ERA5 'tp' and 'e' are accumulations over the data interval (metres of water per
    step). MPAS PRECT and LHFLX are instantaneous rates, so they must be multiplied
    by the interval length. Earlier versions hard-coded 3600 s (hourly), which is a
    factor 3 too small for 3-hourly MPAS output.
    """
    try:
        tv = ds["time"].values
        if len(tv) > 1:
            d = tv[1] - tv[0]
            if isinstance(d, np.timedelta64):
                return float(d / np.timedelta64(1, "s"))
            if hasattr(d, "total_seconds"):
                return float(d.total_seconds())
            # numeric time axis (decode_times=False): scale by the units attribute
            units = str(ds["time"].attrs.get("units", "")).lower()
            per = {"day": 86400.0, "hour": 3600.0, "minute": 60.0, "second": 1.0}
            for k, v in per.items():
                if units.startswith(k):
                    return float(d) * v
    except Exception:
        pass
    return float(fallback)


def open_mpas(path, **kwargs):
    """Open an MPAS history file, transparently handling .gz.
    Returns (ds, tmp_path_or_None). Caller removes tmp_path when done."""
    if str(path).endswith('.gz'):
        with gzip.open(path, 'rb') as gz:
            data = gz.read()
        tmp = tempfile.NamedTemporaryFile(suffix='.nc', delete=False)
        tmp.write(data)
        tmp.close()
        return xr.open_dataset(tmp.name, **kwargs), tmp.name
    return xr.open_dataset(path, **kwargs), None


def _passes_filter(y, mo, sy, ey, sm, em):
    return (sy <= y <= ey) and (sm <= mo <= em)


def find_h1a_by_case(case, roots, start_year, end_year, start_month, end_month):
    """{(y, mo, d, secs): path} -> sorted list; campaign (roots[0]) wins."""
    by_key = {}
    for root in roots:                         # campaign first => preferred
        cdir = Path(root) / case
        if not cdir.exists():
            continue
        for f in cdir.rglob('*.h1a.*'):
            m = h1a_re.search(f.name)
            if not m:
                continue
            y, mo = int(m.group(1)), int(m.group(2))
            if not _passes_filter(y, mo, start_year, end_year, start_month, end_month):
                continue
            key = (y, mo, m.group(3) or '', m.group(4) or '')
            if key not in by_key:
                by_key[key] = str(f)
    return [by_key[k] for k in sorted(by_key)]


def find_h1a_in_dir(input_dir, start_year, end_year, start_month, end_month):
    """Recursive, gz-aware discovery within a single directory (legacy mode)."""
    by_key = {}
    for f in Path(input_dir).rglob('*.h1a.*'):
        m = h1a_re.search(f.name)
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        if not _passes_filter(y, mo, start_year, end_year, start_month, end_month):
            continue
        key = (y, mo, m.group(3) or '', m.group(4) or '')
        if key not in by_key:
            by_key[key] = str(f)
    return [by_key[k] for k in sorted(by_key)]


# ---------------------------------------------------------------------------
# Numba-accelerated vertical interpolation
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def vert_interp_batch(data_3d, p_src_3d, p_tgt_3d):
    nlev_src = data_3d.shape[0]
    nlev_tgt = p_tgt_3d.shape[0]
    nlat = data_3d.shape[1]
    nlon = data_3d.shape[2]
    out = np.empty((nlev_tgt, nlat, nlon), dtype=np.float32)

    for ilat in prange(nlat):
        p_valid = np.empty(nlev_src, dtype=np.float64)
        d_valid = np.empty(nlev_src, dtype=np.float64)
        idx     = np.empty(nlev_src, dtype=np.int64)

        for ilon in range(nlon):
            n_valid = 0
            for i in range(nlev_src):
                ps = p_src_3d[i, ilat, ilon]
                ds = data_3d[i, ilat, ilon]
                if ps > 0 and not np.isnan(ps) and not np.isnan(ds):
                    p_valid[n_valid] = np.log(np.float64(ps))
                    d_valid[n_valid] = np.float64(ds)
                    n_valid += 1

            if n_valid < 2:
                for k in range(nlev_tgt):
                    out[k, ilat, ilon] = np.nan
                continue

            for i in range(n_valid):
                idx[i] = i
            for i in range(1, n_valid):
                key_p = p_valid[idx[i]]
                key_i = idx[i]
                j = i - 1
                while j >= 0 and p_valid[idx[j]] > key_p:
                    idx[j + 1] = idx[j]
                    j -= 1
                idx[j + 1] = key_i

            for k in range(nlev_tgt):
                lp = np.log(np.float64(p_tgt_3d[k, ilat, ilon]))

                if lp <= p_valid[idx[0]]:
                    if n_valid >= 2:
                        p0 = p_valid[idx[0]]; p1 = p_valid[idx[1]]
                        d0 = d_valid[idx[0]]; d1 = d_valid[idx[1]]
                        if p1 != p0:
                            out[k, ilat, ilon] = np.float32(d0 + (lp - p0) / (p1 - p0) * (d1 - d0))
                        else:
                            out[k, ilat, ilon] = np.float32(d0)
                    else:
                        out[k, ilat, ilon] = np.float32(d_valid[idx[0]])
                elif lp >= p_valid[idx[n_valid - 1]]:
                    if n_valid >= 2:
                        p0 = p_valid[idx[n_valid - 2]]; p1 = p_valid[idx[n_valid - 1]]
                        d0 = d_valid[idx[n_valid - 2]]; d1 = d_valid[idx[n_valid - 1]]
                        if p1 != p0:
                            out[k, ilat, ilon] = np.float32(d0 + (lp - p0) / (p1 - p0) * (d1 - d0))
                        else:
                            out[k, ilat, ilon] = np.float32(d1)
                    else:
                        out[k, ilat, ilon] = np.float32(d_valid[idx[n_valid - 1]])
                else:
                    for m in range(n_valid - 1):
                        if p_valid[idx[m]] <= lp <= p_valid[idx[m + 1]]:
                            p0 = p_valid[idx[m]]; p1 = p_valid[idx[m + 1]]
                            d0 = d_valid[idx[m]]; d1 = d_valid[idx[m + 1]]
                            frac = (lp - p0) / (p1 - p0)
                            out[k, ilat, ilon] = np.float32(d0 + frac * (d1 - d0))
                            break
    return out


def _warm_up_jit():
    dummy = np.ones((3, 4, 4), dtype=np.float32)
    p_src = np.array([1000, 800, 500], dtype=np.float32)[:, None, None] * np.ones((1, 4, 4), dtype=np.float32)
    p_tgt = np.array([900, 600], dtype=np.float32)[:, None, None] * np.ones((1, 4, 4), dtype=np.float32)
    vert_interp_batch(dummy, p_src, p_tgt)


# ---------------------------------------------------------------------------
# Bilinear interpolation
# ---------------------------------------------------------------------------
def build_bilinear_mapping(lat_mpas, lon_mpas):
    """Build Delaunay triangulation for bilinear interpolation of u/v/q."""
    print("  Building bilinear (Delaunay) mapping...")

    lon_mpas_360 = np.where(lon_mpas < 0, lon_mpas + 360, lon_mpas)

    # Replicate near boundaries for periodicity
    near_0 = lon_mpas_360 < 10
    near_360 = lon_mpas_360 > 350

    lat_ext = np.concatenate([lat_mpas, lat_mpas[near_0], lat_mpas[near_360]])
    lon_ext = np.concatenate([lon_mpas_360, lon_mpas_360[near_0] + 360, lon_mpas_360[near_360] - 360])
    idx_ext = np.concatenate([np.arange(len(lat_mpas)), np.where(near_0)[0], np.where(near_360)[0]])

    points = np.column_stack([lat_ext, lon_ext])

    lon_2d, lat_2d = np.meshgrid(LON_025, LAT_025)
    target_points = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])

    tri = Delaunay(points)
    simplex_idx = tri.find_simplex(target_points)
    outside = simplex_idx < 0
    print(f"    Points outside triangulation: {outside.sum()} / {len(target_points)} ({outside.sum()/len(target_points)*100:.1f}%)")

    # Precompute barycentric weights for inside points
    inside_mask = ~outside
    simplices = tri.simplices[simplex_idx[inside_mask]]
    transforms = tri.transform[simplex_idx[inside_mask]]
    delta = target_points[inside_mask] - transforms[:, 2, :]
    bary = np.einsum('ijk,ik->ij', transforms[:, :2, :], delta)
    bary_weights = np.column_stack([bary, 1 - bary.sum(axis=1)]).astype(np.float32)
    vertex_mpas_idx = idx_ext[simplices]

    # Nearest-neighbor fallback for outside points
    tree = cKDTree(points)
    nn_mpas_idx = np.zeros(outside.sum(), dtype=np.int64)
    if outside.sum() > 0:
        _, nn_idx = tree.query(target_points[outside])
        nn_mpas_idx = idx_ext[nn_idx]

    info = {
        'inside_mask': inside_mask,
        'vertex_mpas_idx': vertex_mpas_idx,
        'bary_weights': bary_weights,
        'nn_mpas_idx': nn_mpas_idx,
        'shape': (len(LAT_025), len(LON_025)),
        'n_target': len(target_points),
    }
    print(f"    Bilinear mapping: {len(LAT_025)} x {len(LON_025)} grid")
    return info


def bilinear_regrid(data_1d, info):
    """Fast bilinear regrid using precomputed weights (single level)."""
    result = np.zeros(info['n_target'], dtype=np.float32)
    inside = info['inside_mask']
    result[inside] = (data_1d[info['vertex_mpas_idx']] * info['bary_weights']).sum(axis=1)
    outside = ~inside
    if outside.sum() > 0:
        result[outside] = data_1d[info['nn_mpas_idx']]
    return result.reshape(info['shape'])


def bilinear_regrid_levels(data_2d, info):
    """Vectorized bilinear regrid over all levels at once.
    data_2d: (nlev, ncol) -> (nlev, nlat, nlon).
    Replaces the per-level Python loop with one gather + einsum."""
    nlev = data_2d.shape[0]
    result = np.zeros((nlev, info['n_target']), dtype=np.float32)
    inside = info['inside_mask']
    # (nlev, n_inside, 3) gather, weighted-sum over the 3 triangle vertices
    gathered = data_2d[:, info['vertex_mpas_idx']]
    result[:, inside] = np.einsum('lij,ij->li', gathered, info['bary_weights'])
    outside = ~inside
    if outside.sum() > 0:
        result[:, outside] = data_2d[:, info['nn_mpas_idx']]
    return result.reshape(nlev, *info['shape'])


# ---------------------------------------------------------------------------
# Nearest-neighbor mapping (for P, E, SP)
# ---------------------------------------------------------------------------
def build_nearest_mapping(lat_mpas, lon_mpas):
    """Build nearest-cell mapping from MPAS to target grid."""
    print("  Building nearest-neighbor mapping...")

    lon_mpas_360 = np.where(lon_mpas < 0, lon_mpas + 360, lon_mpas)

    mpas_x = np.cos(np.radians(lat_mpas)) * np.cos(np.radians(lon_mpas_360))
    mpas_y = np.cos(np.radians(lat_mpas)) * np.sin(np.radians(lon_mpas_360))
    mpas_z = np.sin(np.radians(lat_mpas))

    tree = cKDTree(np.column_stack([mpas_x, mpas_y, mpas_z]))

    lon_2d, lat_2d = np.meshgrid(LON_025, LAT_025)
    tgt_x = np.cos(np.radians(lat_2d)) * np.cos(np.radians(lon_2d))
    tgt_y = np.cos(np.radians(lat_2d)) * np.sin(np.radians(lon_2d))
    tgt_z = np.sin(np.radians(lat_2d))

    tgt_xyz = np.column_stack([tgt_x.ravel(), tgt_y.ravel(), tgt_z.ravel()])
    _, indices = tree.query(tgt_xyz)

    print(f"    Mapping: {len(LAT_025)} x {len(LON_025)} -> {len(lat_mpas)} MPAS cells")
    return indices.reshape(lat_2d.shape)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def calc_pressure(ps, a, b):
    return a[:, None, None] + b[:, None, None] * ps[None, :, :]


def compute_tcw(q_3d, p_3d, ps_2d):
    g = 9.81
    tcw = np.zeros_like(ps_2d)
    for k in range(q_3d.shape[0] - 1):
        dp = np.abs(p_3d[k] - p_3d[k+1])
        tcw += (q_3d[k] + q_3d[k+1]) / 2 * dp
    return (tcw / g).astype(np.float32)


def create_latitude_coord():
    return xr.DataArray(LAT_025, dims=['latitude'], attrs={
        'units': 'degrees_north', 'standard_name': 'latitude',
        'long_name': 'latitude', 'stored_direction': 'decreasing'})


def create_longitude_coord():
    return xr.DataArray(LON_025, dims=['longitude'], attrs={
        'units': 'degrees_east', 'standard_name': 'longitude', 'long_name': 'longitude'})


def create_level_coord():
    return xr.DataArray(ERA5_LEVELS.astype(np.float64), dims=['model_level'], attrs={
        'units': '1', 'standard_name': 'atmosphere_hybrid_sigma_pressure_coordinate',
        'long_name': 'hybrid level', 'positive': 'down'})


def save_model_level_var(data_4d, var_name, times, output_path, long_name, units, standard_name):
    ds = xr.Dataset(
        {var_name: xr.DataArray(data_4d, dims=['valid_time', 'model_level', 'latitude', 'longitude'],
                                attrs={'units': units, 'long_name': long_name, 'standard_name': standard_name})},
        coords={'valid_time': xr.DataArray(times, dims=['valid_time']),
                'model_level': create_level_coord(),
                'latitude': create_latitude_coord(),
                'longitude': create_longitude_coord()},
        attrs={'Conventions': 'CF-1.6', 'source': 'MPAS-CAM converted to ERA5 format (1-deg bilinear winds)'}
    )
    ds.to_netcdf(output_path, encoding={
        'valid_time': {'units': 'seconds since 1970-01-01', 'calendar': 'noleap'},
        var_name: {'dtype': 'float32'}})
    ds.close()


def save_surface_var(data_3d, var_name, times, output_path, long_name, units, standard_name=None):
    var_attrs = {'units': units, 'long_name': long_name}
    if standard_name:
        var_attrs['standard_name'] = standard_name
    ds = xr.Dataset(
        {var_name: xr.DataArray(data_3d, dims=['valid_time', 'latitude', 'longitude'], attrs=var_attrs)},
        coords={'valid_time': xr.DataArray(times, dims=['valid_time']),
                'latitude': create_latitude_coord(),
                'longitude': create_longitude_coord()},
        attrs={'Conventions': 'CF-1.6', 'source': 'MPAS-CAM converted to ERA5 format (1-deg bilinear winds)'}
    )
    ds.to_netcdf(output_path, encoding={
        'valid_time': {'units': 'seconds since 1970-01-01', 'calendar': 'noleap'},
        var_name: {'dtype': 'float32'}})
    ds.close()


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------
def init_worker(nn_mapping_path, bilinear_path, time_shift, output_dir):
    global WORKER_NN_MAPPING, WORKER_BILINEAR, WORKER_TIME_SHIFT, WORKER_OUTPUT_DIR
    WORKER_NN_MAPPING = np.load(nn_mapping_path)
    with open(bilinear_path, 'rb') as f:
        WORKER_BILINEAR = pickle.load(f)
    WORKER_TIME_SHIFT = time_shift
    WORKER_OUTPUT_DIR = output_dir


def process_file(mpas_file):
    global WORKER_NN_MAPPING, WORKER_BILINEAR, WORKER_TIME_SHIFT, WORKER_OUTPUT_DIR

    filename = Path(mpas_file).name
    file_start = time_module.time()
    tmp_path = None

    try:
        ds, tmp_path = open_mpas(mpas_file)        # gz-aware

        # Guard against truncated files with an empty time dimension
        time_vals = ds.time.values
        if len(time_vals) == 0:
            ds.close()
            print(f"  SKIP (empty time): {filename}", flush=True)
            return f"~ {filename}: empty time dimension"

        q = ds.Q.values
        u = ds.U.values
        v = ds.V.values
        ps = ds.PS.values
        use_pmid = 'PMID' in ds.variables
        pmid = ds.PMID.values if use_pmid else None

        nlev_mpas = q.shape[1]
        nlat, nlon = len(LAT_025), len(LON_025)
        nn_flat = WORKER_NN_MAPPING.ravel()

        tp = e = None
        accum_s = _accum_seconds(ds, ACCUM_SECONDS_DEFAULT)
        if 'PRECT' in ds.data_vars:
            tp = ds.PRECT.values * accum_s
        elif 'PRECC' in ds.data_vars and 'PRECL' in ds.data_vars:
            tp = (ds.PRECC.values + ds.PRECL.values) * accum_s
        if 'LHFLX' in ds.data_vars:
            e = -ds.LHFLX.values * accum_s / (1000.0 * 2.5e6)

        ds.close()

        n_timesteps = len(time_vals)

        u_daily = np.zeros((n_timesteps, len(ERA5_LEVELS), nlat, nlon), dtype=np.float32)
        v_daily = np.zeros_like(u_daily)
        q_daily = np.zeros_like(u_daily)
        sp_daily = np.zeros((n_timesteps, nlat, nlon), dtype=np.float32)
        tcw_daily = np.zeros_like(sp_daily)
        e_daily = np.zeros_like(sp_daily) if e is not None else None
        tp_daily = np.zeros_like(sp_daily) if tp is not None else None
        times_daily = []

        for idx in range(n_timesteps):
            time_val = time_vals[idx]

            if WORKER_TIME_SHIFT != 0:
                time_val = time_val + timedelta(hours=WORKER_TIME_SHIFT)

            times_daily.append(time_val)

            # BILINEAR for u, v, q (all levels at once)
            q_h = bilinear_regrid_levels(q[idx].astype(np.float32), WORKER_BILINEAR)
            u_h = bilinear_regrid_levels(u[idx].astype(np.float32), WORKER_BILINEAR)
            v_h = bilinear_regrid_levels(v[idx].astype(np.float32), WORKER_BILINEAR)

            # NEAREST for surface pressure
            ps_h = ps[idx][nn_flat].reshape(nlat, nlon).astype(np.float32)

            # Pressure on MPAS levels (nearest, tied to SP)
            if use_pmid:
                p_mpas = pmid[idx][:, nn_flat].reshape(nlev_mpas, nlat, nlon).astype(np.float32)
            else:
                sigma = np.linspace(1.0, 0.0, nlev_mpas).astype(np.float32)
                p_mpas = sigma[:, None, None] * ps_h[None, :, :]

            p_era5 = calc_pressure(ps_h, A_COEFF, B_COEFF).astype(np.float32)

            q_v = vert_interp_batch(q_h, p_mpas, p_era5)
            u_v = vert_interp_batch(u_h, p_mpas, p_era5)
            v_v = vert_interp_batch(v_h, p_mpas, p_era5)

            u_daily[idx] = u_v
            v_daily[idx] = v_v
            q_daily[idx] = np.maximum(q_v, 0)
            sp_daily[idx] = ps_h
            tcw_daily[idx] = compute_tcw(q_v, p_era5, ps_h)

            # NEAREST for P and E
            if e is not None:
                e_daily[idx] = e[idx][nn_flat].reshape(nlat, nlon).astype(np.float32)
            if tp is not None:
                tp_daily[idx] = tp[idx][nn_flat].reshape(nlat, nlon).astype(np.float32)

        t0 = times_daily[0]
        date_str = f"{t0.year:04d}-{t0.month:02d}-{t0.day:02d}"
        month_dir = os.path.join(WORKER_OUTPUT_DIR, f"{t0.year:04d}", f"{t0.month:02d}")
        os.makedirs(month_dir, exist_ok=True)
        times_np = np.array(times_daily)

        save_model_level_var(u_daily, 'u', times_np, os.path.join(month_dir, f"ERA5_{date_str}_ml_u.nc"),
                            'U component of wind', 'm s**-1', 'eastward_wind')
        save_model_level_var(v_daily, 'v', times_np, os.path.join(month_dir, f"ERA5_{date_str}_ml_v.nc"),
                            'V component of wind', 'm s**-1', 'northward_wind')
        save_model_level_var(q_daily, 'q', times_np, os.path.join(month_dir, f"ERA5_{date_str}_ml_q.nc"),
                            'Specific humidity', 'kg kg**-1', 'specific_humidity')
        save_surface_var(sp_daily, 'sp', times_np, os.path.join(month_dir, f"ERA5_{date_str}_sp.nc"),
                        'Surface pressure', 'Pa', 'surface_air_pressure')
        save_surface_var(tcw_daily, 'tcw', times_np, os.path.join(month_dir, f"ERA5_{date_str}_tcw.nc"),
                        'Total column water', 'kg m**-2')

        n_files = 5
        if e_daily is not None:
            save_surface_var(e_daily, 'e', times_np, os.path.join(month_dir, f"ERA5_{date_str}_e.nc"),
                            'Evaporation', 'm of water equivalent', 'lwe_thickness_of_water_evaporation_amount')
            n_files += 1
        if tp_daily is not None:
            save_surface_var(tp_daily, 'tp', times_np, os.path.join(month_dir, f"ERA5_{date_str}_tp.nc"),
                            'Total precipitation', 'm')
            n_files += 1

        elapsed = time_module.time() - file_start
        print(f"  Completed: {filename} -> {n_files} files ({n_timesteps} timesteps, {elapsed:.1f}s)", flush=True)
        return f"\u2713 {filename}"

    except Exception as ex:
        import traceback
        print(f"  FAILED: {filename}: {ex}", flush=True)
        return f"\u2717 {filename}: {ex}\n{traceback.format_exc()}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser(description='Convert MPAS to ERA5 format (1-deg bilinear winds, RR-capable)')
    parser.add_argument('output_dir')
    parser.add_argument('--case', default=None,
                        help='RR case name; searches campaign + scratch roots recursively.')
    parser.add_argument('--input-dir', default=None,
                        help='Legacy single-directory mode (e.g. the 120km runs).')
    parser.add_argument('--campaign-root', default=str(CAMPAIGN_ROOT))
    parser.add_argument('--scratch-root', default=str(SCRATCH_ROOT))
    parser.add_argument('--time-shift', type=float, default=0)
    parser.add_argument('--start-year', type=int, default=1)
    parser.add_argument('--end-year', type=int, default=9999)
    parser.add_argument('--start-month', type=int, default=1)
    parser.add_argument('--end-month', type=int, default=12)
    parser.add_argument('--cores', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel worker processes. Default: all cores '
                             '(the per-file bilinear regrid is single-threaded, so process '
                             'parallelism beats threads for this workload).')
    parser.add_argument('--accum-seconds', type=float, default=None, help='accumulation interval (s) if it cannot be read from the time axis; default 10800')
    parser.add_argument('--numba-threads', type=int, default=None,
                        help='Numba threads per worker (only speeds vert_interp_batch). Default: 1.')

    args = parser.parse_args()
    if args.accum_seconds:
        global ACCUM_SECONDS_DEFAULT; ACCUM_SECONDS_DEFAULT = args.accum_seconds

    if (args.case is None) == (args.input_dir is None):
        parser.error("provide exactly one of --case (RR mode) or --input-dir (legacy mode)")

    import numba
    total_cores = args.cores or os.cpu_count()
    # The per-file bilinear regrid is single-threaded NumPy and dominates runtime;
    # only vert_interp_batch uses Numba threads. So favor workers over threads.
    numba_threads = args.numba_threads if args.numba_threads is not None else 1
    n_workers = args.workers if args.workers is not None else max(1, total_cores // numba_threads)
    numba.set_num_threads(numba_threads)

    print("="*70)
    print("MPAS TO ERA5 CONVERTER - 1-DEG BILINEAR WINDS (NUMBA, RR-capable)")
    print("="*70)
    if args.case:
        roots = [Path(args.campaign_root), Path(args.scratch_root)]
        print(f"  Mode:       RR case discovery (campaign preferred)")
        print(f"  Case:       {args.case}")
        print(f"  Campaign:   {args.campaign_root}")
        print(f"  Scratch:    {args.scratch_root}")
    else:
        print(f"  Mode:       legacy single-directory")
        print(f"  Input:      {args.input_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Grid:       {len(LAT_025)} x {len(LON_025)} (1-degree)")
    print(f"  Time shift: {args.time_shift} hours")
    print(f"  Years:      {args.start_year} - {args.end_year}")
    print(f"  Months:     {args.start_month} - {args.end_month}")
    print(f"  Total cores: {total_cores}")
    print(f"  Workers:    {n_workers}")
    print(f"  Numba thds: {numba_threads}")

    print("\nWarming up Numba JIT...")
    _warm_up_jit()
    print("  JIT ready")

    # Discover files
    if args.case:
        mpas_files = find_h1a_by_case(
            args.case, [Path(args.campaign_root), Path(args.scratch_root)],
            args.start_year, args.end_year, args.start_month, args.end_month)
    else:
        mpas_files = find_h1a_in_dir(
            args.input_dir,
            args.start_year, args.end_year, args.start_month, args.end_month)

    print(f"\nFound {len(mpas_files)} h1a files in range")
    if not mpas_files:
        print("No files found!")
        return

    n_gz = sum(1 for f in mpas_files if f.endswith('.gz'))
    if n_gz:
        print(f"  ({n_gz} are gzipped and will be decompressed on read)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build mappings from the first file (gz-aware)
    print("\nLoading MPAS grid...")
    ds0, tmp0 = open_mpas(mpas_files[0])
    try:
        lat_mpas = ds0['lat'].values if 'lat' in ds0 else ds0.coords['lat'].values
        lon_mpas = ds0['lon'].values if 'lon' in ds0 else ds0.coords['lon'].values
    finally:
        ds0.close()
        if tmp0 and os.path.exists(tmp0):
            os.remove(tmp0)
    print(f"  MPAS mesh: {len(lat_mpas)} cells")

    nn_mapping = build_nearest_mapping(lat_mpas, lon_mpas)
    nn_path = os.path.join(args.output_dir, 'nearest_mapping.npy')
    np.save(nn_path, nn_mapping)
    del nn_mapping

    bilinear_info = build_bilinear_mapping(lat_mpas, lon_mpas)
    bl_path = os.path.join(args.output_dir, 'bilinear_info.pkl')
    with open(bl_path, 'wb') as f:
        pickle.dump(bilinear_info, f)
    del bilinear_info

    # Process
    print(f"\nProcessing {len(mpas_files)} files with {n_workers} workers...")
    print("-" * 70)

    start_time = time_module.time()
    completed = failed = skipped = 0

    with Pool(n_workers, initializer=init_worker,
              initargs=(nn_path, bl_path, args.time_shift, args.output_dir)) as pool:
        for result in pool.imap_unordered(process_file, mpas_files):
            if result.startswith('\u2713'):
                completed += 1
            elif result.startswith('~'):
                skipped += 1
            else:
                failed += 1

            total = completed + failed + skipped
            if total % 50 == 0:
                elapsed = time_module.time() - start_time
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(mpas_files) - total) / rate if rate > 0 else 0
                pct = 100 * total / len(mpas_files)
                print(f"\n  === PROGRESS: [{total:5d}/{len(mpas_files)}] {pct:5.1f}% | "
                      f"{completed} ok, {skipped} skipped, {failed} failed | "
                      f"{rate:.2f} files/s | ETA: {eta/60:.1f} min ===\n", flush=True)

    total_time = time_module.time() - start_time
    print("-" * 70)
    print(f"COMPLETE! {completed} succeeded, {skipped} skipped, {failed} failed")
    print(f"Total time: {total_time/60:.1f} min ({total_time:.0f} s)")
    print(f"Average rate: {len(mpas_files)/total_time:.2f} files/s")


if __name__ == '__main__':
    main()