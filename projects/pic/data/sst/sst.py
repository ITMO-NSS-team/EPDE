"""EPDE PDE-discovery on satellite L4 sea-surface-temperature.

SST is a spatiotemporal field T(t, lat, lon), so unlike the ODE systems this is a
PDE-discovery run (advection-diffusion / heat-transport family):
    dT/dt = k*(d2T/dx2 + d2T/dy2) - (cx*dT/dx + cy*dT/dy) + ...
following the (t, y, x) convention of ns.py: coordinate_tensors = meshgrid(t,y,x,
indexing='ij'), data=[field], per-axis max_deriv_order and boundary.

Data source: CDS 'satellite-sea-surface-temperature' L4 combined product (variable
analysed_sst, Kelvin, dims time/lat/lon, 0.05deg gap-free). The global field is huge,
so an open-ocean box is subset (area request) and coarsened. Open ocean avoids the
land/ice NaN mask.

REQUIRES (real-data path): pip install cdsapi xarray netCDF4, plus a ~/.cdsapirc with a
CDS API key. The --synthetic path needs only numpy + epde and is used to verify wiring.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import numpy as np

from epde.interface.prepared_tokens import CustomTokens, ConstantToken, CustomEvaluator
from epde.interface.interface import EpdeSearch
from epde import TrigonometricTokens, GridTokens, CacheStoredTokens
import epde.globals as global_var


# Open-ocean North-Atlantic box (no land -> no NaN mask), strong subtropical gradients.
REGION = {'name': 'n_atlantic', 'lat': (25.0, 40.0), 'lon': (-45.0, -25.0)}
COARSEN = 10        # 0.05deg -> 0.5deg spatial coarsen factor
N_TIME = 60         # number of daily steps to use
TIME_STRIDE = 1     # 1 = daily


def noise_data(data, noise_level):
    # add noise level to the input data
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def download_sst(target, area, n_days=N_TIME * TIME_STRIDE):
    """Download the L4 SST for an area box. area = [North, West, South, East].

    SIZE: a global day at 0.05deg is ~26M cells (~100MB); 181 days ~18GB. Two guards
    keep this small: (1) 'area' subsets to the box server-side, (2) only the months
    spanning the n_days actually used are requested (not the full Jan-Jun)."""
    import cdsapi
    dataset = "satellite-sea-surface-temperature"
    n_months = min(6, max(1, -(-n_days // 28)))   # ceil(n_days/28), capped at 6
    months = [f"{m:02d}" for m in range(1, 1 + n_months)]
    request = {
        "variable": "all",
        "processinglevel": "level_4",
        "sensor_on_satellite": "combined_product",
        "version": "3_0",
        "temporal_resolution": "daily",
        "year": ["2025"],
        "month": months,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "area": area,   # <-- server-side box subset; omit this and the download is global (~GBs/day)
    }
    print(f"[download] area(N,W,S,E)={area}, months={months} (~{n_days} days used)")
    cdsapi.Client().retrieve(dataset, request).download(target)
    return target


def _to_km(lat, lon):
    """Equirectangular tangent-plane conversion of lat/lon (deg) to x,y (km)."""
    lat0 = float(np.mean(lat))
    y_km = (lat - lat[0]) * 110.574
    x_km = (lon - lon[0]) * 111.320 * np.cos(np.deg2rad(lat0))
    return y_km, x_km


def load_sst(path, region, coarsen, n_time, time_stride):
    """Load analysed_sst, subset region/time, coarsen, return ((t,y,x), field[T,Ny,Nx])."""
    import xarray as xr
    import zipfile
    # CDS delivers the area-subset L4 product as a ZIP of per-day .nc files even when
    # the requested target is named .nc -- detect by magic bytes and extract.
    if zipfile.is_zipfile(path):
        extract_dir = os.path.splitext(path)[0] + '_files'
        with zipfile.ZipFile(path) as z:
            z.extractall(extract_dir)
        print(f"[load_sst] extracted {len(os.listdir(extract_dir))} daily files -> {extract_dir}")
        src = os.path.join(extract_dir, '*.nc')
    elif os.path.isfile(path) and path.endswith('.nc'):
        src = path
    else:
        src = os.path.join(path, '*.nc')
    ds = xr.open_mfdataset(src, combine='by_coords') if ('*' in src) else xr.open_dataset(src)

    da = ds['analysed_sst']
    lat_asc = da['lat'].values[0] < da['lat'].values[-1]
    lat_sl = slice(*region['lat']) if lat_asc else slice(region['lat'][1], region['lat'][0])
    da = da.sel(lat=lat_sl, lon=slice(*region['lon']))
    da = da.isel(time=slice(0, n_time * time_stride, time_stride))
    da = da.coarsen(lat=coarsen, lon=coarsen, boundary='trim').mean()

    field = np.asarray(da.values, dtype=np.float64) - 273.15   # Kelvin -> Celsius
    nan_frac = np.isnan(field).mean()
    if nan_frac > 0.05:
        raise ValueError(f'{nan_frac:.1%} NaN (land/ice) in the box -- pick an open-ocean region.')
    if nan_frac > 0:
        # small residual gaps: fill each time slice with its spatial mean
        for k in range(field.shape[0]):
            sl = field[k]
            sl[np.isnan(sl)] = np.nanmean(sl)

    t = np.arange(field.shape[0], dtype=np.float64) * time_stride          # days
    y_km, x_km = _to_km(da['lat'].values, da['lon'].values)
    npts = field.size
    print(f"[load_sst] field {field.shape} (T,Ny,Nx) = {npts:,} pts, "
          f"T in [{field.min():.1f},{field.max():.1f}]C, "
          f"dy~{np.mean(np.diff(y_km)):.0f}km dx~{np.mean(np.diff(x_km)):.0f}km")
    # FD EPDE builds several derivative tensors per point; keep the input modest.
    if npts > 2_000_000:
        raise ValueError(f'{npts:,} grid points is large for FD discovery -- raise COARSEN '
                         f'or lower N_TIME (current: COARSEN={coarsen}, N_TIME={n_time}).')
    return (t, y_km, x_km), field


def _synthetic_field(kappa=0.5, nt=30, ny=25, nx=25, t0=10.0):
    """Exact 2D heat-kernel solution of dT/dt = kappa*(T_xx + T_yy); recovery target."""
    t = np.arange(nt, dtype=np.float64)
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-10, 10, ny)
    T, Y, X = np.meshgrid(t, y, x, indexing='ij')
    denom = 4.0 * kappa * (T + t0)
    field = np.exp(-(X ** 2 + Y ** 2) / denom) / (np.pi * denom)
    return (t, y, x), field, kappa


def sst_discovery(coords, field, noise_level=0.0, discrepancy_metric='wape', seed=0,
                  preprocessor='poly'):
    import random
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    field = noise_data(field, noise_level)
    grid = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')   # (t, y, x)

    # Per-axis small boundary cut (t, y, x).
    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True,
                                 boundary=[2, 3, 3], coordinate_tensors=grid,
                                 verbose_params={'show_iter_idx': True}, device=device,
                                 discrepancy_metric=discrepancy_metric)

    # poly (Savitzky-Golay) over FD: smoothing removes the high-freq discretization
    # error that an FD-only run lets degenerate forms overfit (verified on the
    # synthetic heat-kernel field: under FD a 5-term overfit Pareto-dominates the
    # truth; under poly it no longer does and the truth ranks best on scale-inv).
    epde_search_obj.set_preprocessor(default_preprocessor_type=preprocessor, preprocessor_kwargs={})
    epde_search_obj.set_moeadd_params(population_size=32, training_epochs=20)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

    # 1st order in time, 2nd in space (diffusion). data_fun_pow=1 (passive scalar);
    # factors up to 2 still admit nonlinear advection T*dT/dx if present.
    epde_search_obj.fit(data=[field], variable_names=['T'], max_deriv_order=(1, 2, 2),
                        equation_terms_max_number=8, data_fun_pow=1, additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-8, 1e-0))

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()
    return epde_search_obj


def sst_discovery_synthetic(discrepancy_metric='wape', preprocessor='poly'):
    coords, field, kappa = _synthetic_field()
    print(f"[synthetic] target: dT/dt = {kappa}*(T_xx + T_yy), field {field.shape}, "
          f"metric={discrepancy_metric}, preprocessor={preprocessor}")
    return sst_discovery(coords, field, discrepancy_metric=discrepancy_metric,
                         preprocessor=preprocessor)


if __name__ == "__main__":
    import argparse
    import torch
    print('cuda', torch.cuda.is_available())

    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true', help='verify wiring on an exact heat-kernel field')
    ap.add_argument('--download', action='store_true', help='fetch the region box from CDS first')
    ap.add_argument('--data', default=os.path.join(os.path.dirname(__file__), 'sst_l4.nc'))
    ap.add_argument('--metric', default='wape',
                    help="discrepancy objective: wape | l2 | scale_invariant")
    ap.add_argument('--preprocessor', default='poly',
                    help="derivative preprocessor: poly (Savitzky-Golay) | FD")
    args = ap.parse_args()

    if args.synthetic:
        sst_discovery_synthetic(discrepancy_metric=args.metric, preprocessor=args.preprocessor)
    else:
        area = [REGION['lat'][1], REGION['lon'][0], REGION['lat'][0], REGION['lon'][1]]  # N,W,S,E
        if args.download or not os.path.exists(args.data):
            download_sst(args.data, area=area)
        coords, field = load_sst(args.data, REGION, COARSEN, N_TIME, TIME_STRIDE)
        sst_discovery(coords, field, discrepancy_metric=args.metric,
                      preprocessor=args.preprocessor)
