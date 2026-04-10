# MPAS-to-ERA5-Adaptor

Converts MPAS-CAM model output to ERA5-compatible format for use with WAM2layers moisture tracking.

## Features
- Bilinear (Delaunay) interpolation for u, v, q to preserve spatial coherence
- Nearest-neighbor for P, E, SP (local quantities)  
- Configurable output grid (default 1°)
- Numba-accelerated vertical interpolation
- Multiprocessing for batch conversion
