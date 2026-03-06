"""
MPAS to ERA5 Converter - CONSERVATIVE REGRIDDING VERSION

Written by Andrew Feder and Claude AI

Optimized with Numba-accelerated vertical interpolation.

Usage
-----
    python mpas_to_era5.py <input_dir> <output_dir> [options]

Arguments
---------
    input_dir       Directory containing MPAS *.h1a.*.nc files
    output_dir      Directory for ERA5-formatted output (created if needed)

Options
-------
    --time-shift HOURS      Shift timestamps by HOURS (float, default: 0)
    --start-year YEAR       First year to process (default: 1)
    --end-year YEAR         Last year to process (default: 9999)
    --start-month MONTH     First month to process (default: 1)
    --end-month MONTH       Last month to process (default: 12)
    --cores N               Total CPU cores to use (default: all available).
                            Auto-splits between file-parallel workers and
                            Numba threads for vertical interpolation.

Examples
--------
    # Process all files
    python mpas_to_era5.py /glade/derecho/scratch/.../mpas_out ./era5_out

    # Process years 50-100, shift timestamps by -12 hours, use 64 cores
    python mpas_to_era5.py /glade/derecho/scratch/.../mpas_out ./era5_out \
        --start-year 50 --end-year 100 --time-shift -12 --cores 64

    # Process only JJA (June-August) from year 80 onward
    python mpas_to_era5.py /glade/derecho/scratch/.../mpas_out ./era5_out \
        --start-year 80 --start-month 6 --end-month 8

    # Process a single month (e.g., January year 90)
    python mpas_to_era5.py /glade/derecho/scratch/.../mpas_out ./era5_out \
        --start-year 90 --end-year 90 --start-month 1 --end-month 1

Output Structure
----------------
    output_dir/
        conservative_mapping.npy          # Reusable MPAS-to-grid mapping
        YYYY/MM/
            ERA5_YYYY-MM-DD_ml_u.nc       # U-wind on model levels
            ERA5_YYYY-MM-DD_ml_v.nc       # V-wind on model levels
            ERA5_YYYY-MM-DD_ml_q.nc       # Specific humidity on model levels
            ERA5_YYYY-MM-DD_sp.nc         # Surface pressure
            ERA5_YYYY-MM-DD_tcw.nc        # Total column water
            ERA5_YYYY-MM-DD_e.nc          # Evaporation (if LHFLX available)
            ERA5_YYYY-MM-DD_tp.nc         # Precipitation (if PRECT available)

Notes
-----
    - Output uses 22 ERA5 hybrid model levels (level 20-137), 0.25° grid,
      descending latitude (90 to -90), and 0-360° longitude.
    - Vertical interpolation is performed in log-pressure space.
    - Horizontal regridding uses nearest-cell (not area-weighted) mapping.
    - Time encoding uses noleap calendar for compatibility with CESM output.
    - Month filtering requires start_month <= end_month (no wrap-around).
      For DJF, run December and Jan-Feb as separate invocations.
    - For Derecho PBS jobs, set --cores to match your allocation (e.g., 128).
"""

import os
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'  # fork-safe threading backend

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
import argparse
from pathlib import Path
from multiprocessing import Pool
import warnings
from datetime import timedelta
import time as time_module
from numba import njit, prange

warnings.filterwarnings('ignore')

# ERA5 hybrid levels
ERA5_LEVELS = np.array([20, 40, 60, 80, 90, 95, 100, 105, 110, 115, 120, 123, 125, 128, 130, 131, 132, 133, 134, 135, 136, 137])

A_COEFF = np.array([316.420746, 2900.391357, 10065.978516, 19620.042969, 20087.085938, 18489.707031, 15596.695313, 11901.339844, 8163.375000, 4993.796875, 2659.140625, 1659.476563, 1143.250000, 568.062500, 302.476563, 202.484375, 122.101563, 62.781250, 22.835938, 3.757813, 0.000000, 0.000000])

B_COEFF = np.array([0.000000, 0.000000, 0.000340, 0.068448, 0.203491, 0.308598, 0.438391, 0.576692, 0.704669, 0.809536, 0.887408, 0.922096, 0.940860, 0.963352, 0.975078, 0.980072, 0.984542, 0.988500, 0.991984, 0.995003, 0.997630, 1.000000])

# DESCENDING latitude, 0-360 longitude for WAM2layers
LAT_025 = np.arange(90, -90.25, -0.25)
LON_025 = np.arange(0, 360, 0.25)

# Globals for workers
WORKER_MAPPING = None
WORKER_TIME_SHIFT = 0
WORKER_OUTPUT_DIR = None


# ---------------------------------------------------------------------------
# Numba-accelerated vertical interpolation
# ---------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def vert_interp_batch(data_3d, p_src_3d, p_tgt_3d):
    """
    Vectorized vertical interpolation in log-pressure space using Numba.

    Parameters
    ----------
    data_3d  : (nlev_src, nlat, nlon) float32 - source data on MPAS levels
    p_src_3d : (nlev_src, nlat, nlon) float32 - pressure on MPAS levels
    p_tgt_3d : (nlev_tgt, nlat, nlon) float32 - pressure on ERA5 levels

    Returns
    -------
    out : (nlev_tgt, nlat, nlon) float32
    """
    nlev_src = data_3d.shape[0]
    nlev_tgt = p_tgt_3d.shape[0]
    nlat = data_3d.shape[1]
    nlon = data_3d.shape[2]
    out = np.empty((nlev_tgt, nlat, nlon), dtype=np.float32)

    for ilat in prange(nlat):
        # Scratch arrays allocated per-row to avoid contention
        p_valid = np.empty(nlev_src, dtype=np.float64)
        d_valid = np.empty(nlev_src, dtype=np.float64)
        idx     = np.empty(nlev_src, dtype=np.int64)

        for ilon in range(nlon):
            # ---- gather valid source levels ----
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

            # ---- sort by log-pressure (insertion sort, small n) ----
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

            # ---- interpolate each target level ----
            for k in range(nlev_tgt):
                lp = np.log(np.float64(p_tgt_3d[k, ilat, ilon]))

                # Below lowest source level -> extrapolate from first segment
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

                # Above highest source level -> extrapolate from last segment
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

                # Interior -> linear interpolation
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
    """Trigger JIT compilation at import time so workers don't all compile."""
    dummy = np.ones((3, 4, 4), dtype=np.float32)
    p_src = np.array([1000, 800, 500], dtype=np.float32)[:, None, None] * np.ones((1, 4, 4), dtype=np.float32)
    p_tgt = np.array([900, 600], dtype=np.float32)[:, None, None] * np.ones((1, 4, 4), dtype=np.float32)
    vert_interp_batch(dummy, p_src, p_tgt)


# ---------------------------------------------------------------------------


def init_worker(mapping_path, time_shift, output_dir):
    global WORKER_MAPPING, WORKER_TIME_SHIFT, WORKER_OUTPUT_DIR
    WORKER_MAPPING = np.load(mapping_path)
    WORKER_TIME_SHIFT = time_shift
    WORKER_OUTPUT_DIR = output_dir


def build_conservative_mapping(lat_mpas, lon_mpas):
    """Build nearest-cell mapping from MPAS to target grid."""
    print("  Building conservative (nearest-cell) mapping...")

    # Convert MPAS lon to 0-360 if needed
    lon_mpas_360 = np.where(lon_mpas < 0, lon_mpas + 360, lon_mpas)

    # Cartesian coords for spherical distance
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

    print(f"    Mapping: {len(LAT_025)} × {len(LON_025)} → {len(lat_mpas)} MPAS cells")
    return indices.reshape(lat_2d.shape)


def calc_pressure(ps, a, b):
    """Calculate pressure on ERA5 hybrid levels (vectorized)."""
    return a[:, None, None] + b[:, None, None] * ps[None, :, :]


def compute_tcw(q_3d, p_3d, ps_2d):
    """Total column water."""
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
        attrs={'Conventions': 'CF-1.6', 'source': 'MPAS-CAM converted to ERA5 format (conservative regridding)'}
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
        attrs={'Conventions': 'CF-1.6', 'source': 'MPAS-CAM converted to ERA5 format (conservative regridding)'}
    )
    ds.to_netcdf(output_path, encoding={
        'valid_time': {'units': 'seconds since 1970-01-01', 'calendar': 'noleap'},
        var_name: {'dtype': 'float32'}})
    ds.close()


def process_file(mpas_file):
    global WORKER_MAPPING, WORKER_TIME_SHIFT, WORKER_OUTPUT_DIR

    filename = Path(mpas_file).name
    print(f"  Starting: {filename}", flush=True)
    file_start = time_module.time()

    try:
        ds = xr.open_dataset(mpas_file)

        q = ds.Q.values
        t = ds.T.values
        u = ds.U.values
        v = ds.V.values
        ps = ds.PS.values
        use_pmid = 'PMID' in ds.variables
        pmid = ds.PMID.values if use_pmid else None
        time_vals = ds.time.values

        lat_mpas = ds['lat'].values if 'lat' in ds else ds.coords['lat'].values
        lon_mpas = ds['lon'].values if 'lon' in ds else ds.coords['lon'].values

        nlev_mpas = q.shape[1]
        nlat, nlon = len(LAT_025), len(LON_025)

        tp = e = None
        if 'PRECT' in ds.data_vars:
            tp = ds.PRECT.values * 3600.0
        if 'LHFLX' in ds.data_vars:
            e = -ds.LHFLX.values * 3600.0 / (1000.0 * 2.5e6)

        ds.close()

        # Flatten mapping for vectorized indexing
        mapping_flat = WORKER_MAPPING.ravel()

        # Use ALL timesteps
        timestep_indices = list(range(len(time_vals)))
        n_timesteps = len(timestep_indices)

        u_daily = np.zeros((n_timesteps, len(ERA5_LEVELS), nlat, nlon), dtype=np.float32)
        v_daily = np.zeros_like(u_daily)
        q_daily = np.zeros_like(u_daily)
        sp_daily = np.zeros((n_timesteps, nlat, nlon), dtype=np.float32)
        tcw_daily = np.zeros_like(sp_daily)
        e_daily = np.zeros_like(sp_daily) if e is not None else None
        tp_daily = np.zeros_like(sp_daily) if tp is not None else None
        times_daily = []

        for idx, it in enumerate(timestep_indices):
            time_val = time_vals[it]

            if WORKER_TIME_SHIFT != 0:
                time_val = time_val + timedelta(hours=WORKER_TIME_SHIFT)

            times_daily.append(time_val)

            # Vectorized horizontal regrid: index all levels at once
            q_h = q[it][:, mapping_flat].reshape(nlev_mpas, nlat, nlon).astype(np.float32)
            u_h = u[it][:, mapping_flat].reshape(nlev_mpas, nlat, nlon).astype(np.float32)
            v_h = v[it][:, mapping_flat].reshape(nlev_mpas, nlat, nlon).astype(np.float32)

            ps_h = ps[it][mapping_flat].reshape(nlat, nlon).astype(np.float32)

            # Pressure on MPAS levels
            if use_pmid:
                p_mpas = pmid[it][:, mapping_flat].reshape(nlev_mpas, nlat, nlon).astype(np.float32)
            else:
                sigma = np.linspace(1.0, 0.0, nlev_mpas).astype(np.float32)
                p_mpas = sigma[:, None, None] * ps_h[None, :, :]

            # Pressure on ERA5 levels
            p_era5 = calc_pressure(ps_h, A_COEFF, B_COEFF).astype(np.float32)

            # Numba-accelerated vertical interpolation
            q_v = vert_interp_batch(q_h, p_mpas, p_era5)
            u_v = vert_interp_batch(u_h, p_mpas, p_era5)
            v_v = vert_interp_batch(v_h, p_mpas, p_era5)

            u_daily[idx] = u_v
            v_daily[idx] = v_v
            q_daily[idx] = q_v
            sp_daily[idx] = ps_h
            tcw_daily[idx] = compute_tcw(q_v, p_era5, ps_h)

            if e is not None:
                e_daily[idx] = e[it][mapping_flat].reshape(nlat, nlon).astype(np.float32)
            if tp is not None:
                tp_daily[idx] = tp[it][mapping_flat].reshape(nlat, nlon).astype(np.float32)

        # Get date from first timestamp (cftime object)
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
        return f"✓ {filename}"

    except Exception as ex:
        import traceback
        print(f"  FAILED: {filename}: {ex}", flush=True)
        return f"✗ {filename}: {ex}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description='Convert MPAS to ERA5 format (conservative regridding)')
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--time-shift', type=float, default=0)
    parser.add_argument('--start-year', type=int, default=1)
    parser.add_argument('--end-year', type=int, default=9999)
    parser.add_argument('--start-month', type=int, default=1)
    parser.add_argument('--end-month', type=int, default=12)
    parser.add_argument('--cores', type=int, default=None,
                        help='Total cores available. Auto-splits between file-parallel workers '
                             'and Numba threads. Default: all available cores.')

    args = parser.parse_args()

    # Auto-tune worker/thread split
    import numba
    total_cores = args.cores or os.cpu_count()
    # Heuristic: give Numba ~8-16 threads (diminishing returns beyond that for
    # the lat-parallel interp), use remaining cores for file-level parallelism.
    # Minimum 1 worker, minimum 4 Numba threads.
    numba_threads = min(max(4, total_cores // 8), 16)
    n_workers = max(1, total_cores // numba_threads)
    # Set Numba thread count before JIT warm-up
    numba.set_num_threads(numba_threads)

    print("="*70)
    print("MPAS TO ERA5 CONVERTER - CONSERVATIVE REGRIDDING (NUMBA)")
    print("="*70)
    print(f"  Input:      {args.input_dir}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Time shift: {args.time_shift} hours")
    print(f"  Years:      {args.start_year} - {args.end_year}")
    print(f"  Months:     {args.start_month} - {args.end_month}")
    print(f"  Total cores: {total_cores}")
    print(f"  Workers:    {n_workers}  (file-level parallelism)")
    print(f"  Numba thds: {numba_threads}  (per-worker vert. interp)")

    # Warm up JIT before forking workers
    print("\nWarming up Numba JIT...")
    _warm_up_jit()
    print("  JIT ready ✓")

    # Find files
    all_files = sorted(Path(args.input_dir).glob("*.h1a.*.nc"))
    print(f"\nFound {len(all_files)} total files (*.h1a.*.nc)")

    mpas_files = []
    for f in all_files:
        try:
            name_parts = f.stem.split('.')
            date_part = name_parts[-1]
            date_str = date_part.rsplit('-', 1)[0]
            year, month, day = map(int, date_str.split('-'))

            if args.start_year <= year <= args.end_year and args.start_month <= month <= args.end_month:
                mpas_files.append(str(f))
        except Exception as e:
            pass

    print(f"Filtered to {len(mpas_files)} files")

    if not mpas_files:
        print("No files found!")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Build mapping
    print("\nBuilding conservative mapping...")
    ds0 = xr.open_dataset(mpas_files[0])
    lat_mpas = ds0['lat'].values if 'lat' in ds0 else ds0.coords['lat'].values
    lon_mpas = ds0['lon'].values if 'lon' in ds0 else ds0.coords['lon'].values
    ds0.close()

    mapping = build_conservative_mapping(lat_mpas, lon_mpas)
    mapping_path = os.path.join(args.output_dir, 'conservative_mapping.npy')
    np.save(mapping_path, mapping)
    print(f"  Saved: {mapping_path}")
    del mapping

    # Process
    print(f"\nProcessing {len(mpas_files)} files with {n_workers} workers...")
    print("-" * 70)

    start_time = time_module.time()
    completed = failed = 0

    with Pool(n_workers, initializer=init_worker,
              initargs=(mapping_path, args.time_shift, args.output_dir)) as pool:
        for result in pool.imap_unordered(process_file, mpas_files):
            if result.startswith('✓'):
                completed += 1
            else:
                failed += 1

            total = completed + failed
            if total % 100 == 0:
                elapsed = time_module.time() - start_time
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(mpas_files) - total) / rate if rate > 0 else 0
                pct = 100 * total / len(mpas_files)
                print(f"\n  === PROGRESS: [{total:5d}/{len(mpas_files)}] {pct:5.1f}% | "
                      f"{completed} ok, {failed} failed | "
                      f"{rate:.2f} files/s | ETA: {eta/60:.1f} min ===\n", flush=True)

    total_time = time_module.time() - start_time
    print("-" * 70)
    print(f"COMPLETE! {completed} succeeded, {failed} failed")
    print(f"Total time: {total_time/60:.1f} min ({total_time:.0f} s)")
    print(f"Average rate: {len(mpas_files)/total_time:.2f} files/s")


if __name__ == '__main__':
    main()
