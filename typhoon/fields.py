from pathlib import Path
import numpy as np
import xarray as xr

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
SURFACE = ["u10", "v10", "u100", "v100", "t2m", "sp", "msl", "tcwv"]
CHANNELS = SURFACE + [f"{v}{lev}" for v in ["u", "v", "z", "t", "r"] for lev in LEVELS]
ALIASES = {"10u": "u10", "10v": "v10", "100u": "u100", "100v": "v100", "2t": "t2m"}


def canonical(ds, stamp=None, global_grid=True):
    rename = {
        k: v
        for k, v in {
            "valid_time": "time",
            "latitude": "lat",
            "longitude": "lon",
            "pressure_level": "level",
            "isobaricInhPa": "level",
        }.items()
        if k in ds.dims or k in ds.coords
    }
    ds = ds.rename(rename)
    ds = ds.rename_vars(
        {
            k: v
            for k, v in ALIASES.items()
            if k in ds.data_vars and v not in ds.data_vars
        }
    )
    if "expver" in ds.dims:
        versions = [ds.sel(expver=v, drop=True) for v in ds.expver.values]
        ds = versions[0]
        for other in versions[1:]:
            ds = ds.combine_first(other)
    if "time" in ds.coords and stamp is not None:
        if "time" in ds.dims:
            ds = ds.sel(time=np.datetime64(stamp, "ns"))
        elif np.datetime64(ds.time.values, "ns") != np.datetime64(stamp, "ns"):
            raise ValueError("State time differs from the requested initialization")
    if "time" in ds.dims:
        if ds.sizes["time"] != 1:
            raise ValueError(
                "Select one explicit time, do not silently take first frame"
            )
        ds = ds.isel(time=0)
    ds = ds.assign_coords(lon=ds.lon % 360).sortby("lon").sortby("lat", ascending=False)
    if len(np.unique(ds.lon)) != len(ds.lon):
        raise ValueError("Duplicate longitudes")
    if global_grid and (
        ds.sizes["lat"] != 721
        or ds.sizes["lon"] != 1440
        or not np.allclose(ds.lat, np.linspace(90, -90, 721))
        or not np.allclose(ds.lon, np.arange(1440) * 0.25)
    ):
        raise ValueError(
            "Expected global 0.25 degree 721x1440 grid; explicit regridding is required"
        )
    if "level" in ds.coords and not set(LEVELS).issubset(set(ds.level.values.tolist())):
        raise ValueError("Missing pressure levels or incorrect pressure units")
    return ds


def read_pair(folder, stamp=None, global_grid=True):
    folder = Path(folder)
    with xr.open_dataset(folder / "sfc.nc") as s, xr.open_dataset(
        folder / "pl.nc"
    ) as p:
        s = canonical(s, stamp, global_grid).load()
        p = canonical(p, stamp, global_grid).load()
    if not np.array_equal(s.lat, p.lat) or not np.array_equal(s.lon, p.lon):
        raise ValueError("Surface and upper-air grids differ")
    return p, s


def array(da):
    return np.asarray(da.transpose("lat", "lon").values, dtype=np.float32)


def tianxing_state(pl, sfc):
    x = np.stack(
        [array(sfc[v]) for v in SURFACE]
        + [
            array(pl[v].sel(level=lev))
            for v in ["u", "v", "z", "t", "r"]
            for lev in LEVELS
        ]
    )
    if not np.isfinite(x).all():
        raise ValueError("Non-finite initial state; fill source data explicitly")
    if np.nanmedian(x[6]) < 10000:
        raise ValueError("MSLP must be in Pa, not hPa")
    if np.nanmedian(array(pl.z.sel(level=500))) < 20000:
        raise ValueError("Z must be geopotential (m2 s-2), not height")
    r = np.asarray(pl.r)
    if pl.r.attrs.get("units") not in ["%", "percent"]:
        raise ValueError(
            "TianXing requires relative humidity in percent; verify units explicitly"
        )
    return x[None]


def pangu_state(pl, sfc):
    upper = np.stack(
        [
            np.stack([array(pl[v].sel(level=lev)) for lev in reversed(LEVELS)])
            for v in ["z", "q", "t", "u", "v"]
        ]
    )
    surface = np.stack([array(sfc[v]) for v in ["msl", "u10", "v10", "t2m"]])
    if not np.isfinite(upper).all() or not np.isfinite(surface).all():
        raise ValueError("Non-finite Pangu input")
    return upper, surface
