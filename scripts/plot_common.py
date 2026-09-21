"""Shared publication-figure styles and archived forecast series."""

from pathlib import Path
import json, hashlib
import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
PC = ccrs.PlateCarree()
MM = 1 / 25.4
read = lambda p: json.loads((ROOT / p).read_text(encoding="utf8"))
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.15,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)
COL = {
    "control": "#C46A25",
    "optimal": "#0072B2",
    "era5init": "#8662A6",
    "reference": "#242424",
}
LS = {"control": "--", "optimal": "-", "era5init": ":", "reference": "-"}
MARK = {"control": "^", "optimal": "o", "era5init": "s", "reference": "."}
LAB = {
    "control": "CTRL",
    "optimal": "Optimal",
    "era5init": "ERA5T start",
    "reference": "Observed track",
}
REV = read("data/tracks.json")
ER = read("data/reanalysis_tracks.json")["results"]
SYS = {r["model"]: r["rows"] for r in read("data/ridge.json")}
CLEAR = read("data/steering.json")["rows"]
LOOKUP = {(r["model"], r["state"], r["lead_hours"]): r for r in CLEAR}
STATS = []
manifest = []


def save(fig, name, sources, method=""):
    OUT.mkdir(exist_ok=True, parents=True)
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_text() == "Pangu-Weather":
            text.set_text("Pangu-Weather")
    for ext, dpi in [("png", 320), ("pdf", 600)]:
        fig.savefig(
            OUT / (name + "." + ext), dpi=dpi, bbox_inches="tight", pad_inches=0.045
        )
    manifest.append(
        {
            "figure": name,
            "sources": sources,
            "sha256": {
                p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                for p in sources
                if (ROOT / p).is_file()
            },
        }
    )
    plt.close(fig)
    print(name, flush=True)


def field(model, state, h):
    return xr.load_dataset(ROOT / f"data/{model}_{state}_{h:03d}.nc")


def group(model, exp="baseline"):
    return next(
        r
        for r in REV["results"]
        if r["case"] == "SAUDEL_LATE"
        and r["model"] == model
        and r["experiment"] == exp
        and r["variant"] == "local3_gap1"
    )


def rows(model, state):
    if state == "era5init":
        return next(r for r in ER if r["model"] == model)["rows"]
    r = group(model, "baseline" if state in ["control", "reference"] else "corrected")[
        "rows"
    ]
    return [dict(x, center=x["reference"]) for x in r] if state == "reference" else r


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.6, length=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=0.4, color="#dce0e3")


def title(ax, letter, label):
    ax.set_title(f"({letter}) {label}", loc="left", fontweight="normal", pad=7)


def base(ax, extent=[103, 142, 10, 37], step=10, left=True, bottom=True):
    ax.set_extent(extent, crs=PC)
    ax.set_facecolor("white")
    ax.coastlines("50m", linewidth=0.4, color="#777777", zorder=5)
    ax.set_xticks(
        np.arange(np.ceil(extent[0] / step) * step, extent[1] + 0.1, step), crs=PC
    )
    sy = 5 if extent[3] - extent[2] < 25 else 10
    ax.set_yticks(np.arange(np.ceil(extent[2] / sy) * sy, extent[3] + 0.1, sy), crs=PC)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    if len(ax.get_xticks()) and abs(ax.get_xticks()[-1] - extent[1]) < 0.01:
        ax.get_xticklabels()[-1].set_ha("right")
    ax.tick_params(labelleft=left, labelbottom=bottom, length=2, width=0.5, pad=2)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
        s.set_color("#888888")


def key_handles(states, reference_label=None):
    return [
        Line2D(
            [],
            [],
            color=COL[s],
            ls=LS[s],
            marker=MARK[s],
            ms=3,
            lw=1.1,
            label=reference_label if s == "reference" and reference_label else LAB[s],
        )
        for s in states
    ]


def line(ax, x, y, state, label=None, **kw):
    ax.plot(
        x,
        y,
        color=COL[state],
        ls=LS[state],
        marker=MARK[state],
        markevery=4,
        ms=2.5,
        label=label or LAB[state],
        **kw,
    )


def track(ax, rr, state, label=None):
    x = [r["center"]["lon"] if r.get("center") else np.nan for r in rr]
    y = [r["center"]["lat"] if r.get("center") else np.nan for r in rr]
    line(ax, x, y, state, label, transform=PC, zorder=8 if state == "reference" else 7)


def cbar(fig, im, ax, label, ticks=None):
    cb = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.12,
        fraction=0.045,
        aspect=30,
        extend="both",
        ticks=ticks,
    )
    cb.set_label(label, labelpad=2)
    cb.ax.tick_params(labelsize=7.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    return cb


def focus(ax, box):
    w, e, s, n = box
    ax.add_patch(
        Rectangle(
            (w, s),
            e - w,
            n - s,
            fill=False,
            ec="#B12A32",
            lw=0.9,
            transform=PC,
            zorder=10,
        )
    )


def mark(ax, row):
    c, r = row["center"], row["reference"]
    ax.plot(
        c["lon"],
        c["lat"],
        "o",
        mfc="white",
        mec="#222222",
        ms=4,
        mew=0.9,
        transform=PC,
        zorder=12,
    )
    if r:
        ax.plot(
            r["lon"],
            r["lat"],
            "*",
            mfc="#222222",
            mec="white",
            mew=0.35,
            ms=6,
            transform=PC,
            zorder=11,
        )


def contour(ax, ds, key, levels, lw=0.5, color="#707070"):
    a = ds[key].values
    ls = [v for v in levels if np.nanmin(a) < v < np.nanmax(a)]
    if not ls:
        return
    cs = ax.contour(
        ds.lon,
        ds.lat,
        a,
        levels=ls,
        colors=color,
        linewidths=lw,
        transform=PC,
        zorder=6,
    )
    ax.clabel(cs, fmt="%d", fontsize=7, inline_spacing=3)


def scalar_panel(ax, d, r, key, left=True, annular_scale=120):
    base(ax, left=left)
    im = None
    if key == "mslp_hpa":
        im = ax.pcolormesh(
            d.lon,
            d.lat,
            d[key],
            cmap="Blues_r",
            norm=Normalize(980, 1030),
            shading="nearest",
            rasterized=True,
            transform=PC,
        )
        contour(ax, d, key, np.arange(984, 1033, 8), color="#616B75")
    elif key == "z500_m":
        contour(ax, d, key, [5760, 5800, 5840, 5920, 5960], color="#737373")
        contour(ax, d, key, [5880], lw=1.35, color="#8A5B22")
    else:
        ax.add_feature(
            cf.LAND.with_scale("50m"), facecolor="#f3f3f1", edgecolor="none", zorder=0
        )
        t = d.sel(lat=slice(37, 10), lon=slice(103, 142)).isel(
            lat=slice(0, None, 14), lon=slice(0, None, 14)
        )
        q = ax.quiver(
            t.lon,
            t.lat,
            t.u_850_500,
            t.v_850_500,
            color="#747D84",
            scale=120,
            width=0.0035,
            transform=PC,
            zorder=6,
        )
        c = r["center"]
        s = r["steering"]
        expanded = annular_scale != 120
        blue = ax.quiver(
            [c["lon"]],
            [c["lat"]],
            [s["u_850_500"]],
            [s["v_850_500"]],
            color=COL["optimal"],
            scale=annular_scale,
            width=0.008 if expanded else 0.011,
            transform=PC,
            zorder=11 if expanded else 14,
        )
        blue.set_gid("annular-wind")
        if expanded:
            from matplotlib import patheffects

            blue.set_path_effects(
                [patheffects.withStroke(linewidth=0.8, foreground="white")]
            )
        ax.quiverkey(
            q,
            0.88,
            1.045,
            10,
            "10 m s$^{-1}$",
            labelpos="W",
            fontproperties={"size": 7},
        )
    focus(
        ax,
        (
            [107, 122, 21, 31]
            if r["lead_hours"] <= 72
            else [119, 138, 16, 33] if r["lead_hours"] <= 120 else [106, 123, 16, 29]
        ),
    )
    mark(ax, r)
    if key == "mslp_hpa":
        a = d[key].sel(lat=slice(37, 10), lon=slice(103, 142)).values
        STATS.append(
            dict(
                model=r["model"],
                state=r["state"],
                h=r["lead_hours"],
                variable=key,
                min=float(np.nanmin(a)),
                max=float(np.nanmax(a)),
                limits=[980, 1030],
                out_of_range=float(np.mean((a < 980) | (a > 1030))),
            )
        )
    return im


def fig1():
    fig = plt.figure(figsize=(180 * MM, 152 * MM))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.32, 1],
        left=0.09,
        right=0.985,
        bottom=0.095,
        top=0.91,
        hspace=0.35,
        wspace=0.28,
    )
    for j, (model, label) in enumerate(
        [("tianxing", "TianXing"), ("pangu_6h", "Pangu-Weather")]
    ):
        ax = fig.add_subplot(gs[0, j], projection=PC)
        base(ax, [103, 128, 12, 32], step=5)
        title(ax, "ab"[j], label)
        bx = fig.add_subplot(gs[1, j])
        clean(bx)
        title(bx, "cd"[j], label)
        for state in ["control", "era5init", "optimal", "reference"]:
            rr = rows(model, state)
            track(ax, rr, state)
            if state != "reference":
                rr = [r for r in rr if r["lead_hours"] > 0]
                line(
                    bx,
                    [r["lead_hours"] for r in rr],
                    [r["distance_km"] for r in rr],
                    state,
                )
        bx.set(
            xlim=(0, 168),
            ylim=(0, 1350),
            xticks=[0, 24, 48, 72, 96, 120, 144, 168],
            xlabel="Forecast lead (h)",
            ylabel="Track error (km)" if j == 0 else "",
        )
    fig.legend(
        handles=key_handles(["reference", "control", "optimal", "era5init"]),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.53, 0.995),
        handlelength=2.3,
        columnspacing=1.3,
    )
    save(
        fig,
        "fig01_forecast_comparison",
        ["data/tracks.json", "data/reanalysis_tracks.json"],
        "All28future six-hour positions; same tracking; no missing-time interpolation. Markers every24h, lines retain6h states. Colors encode initialization, panels encode model.",
    )


def absolute_pair(model, name, annular_scale=120):
    fig, axes = plt.subplots(
        2, 3, figsize=(180 * MM, 110 * MM), subplot_kw={"projection": PC}
    )
    fig.subplots_adjust(
        left=0.075, right=0.975, bottom=0.14, top=0.94, hspace=0.26, wspace=0.16
    )
    for i, state in enumerate(["control", "optimal"]):
        d = field(model, state, 120)
        r = LOOKUP[model, state, 120]
        for j, key in enumerate(["mslp_hpa", "z500_m", "wind"]):
            ax = axes[i, j]
            im = scalar_panel(ax, d, r, key, left=j == 0, annular_scale=annular_scale)
            title(ax, "abcdef"[i * 3 + j], ["MSLP (hPa)", "Z500 (m)", "Layer wind"][j])
            if j == 0:
                ax.text(
                    -0.18,
                    0.5,
                    "CTRL" if i == 0 else "Optimal",
                    transform=ax.transAxes,
                    va="center",
                    ha="center",
                    rotation=90,
                    fontweight="bold",
                    fontsize=8.5,
                )
    pos = axes[1, 0].get_position()
    cax = fig.add_axes([pos.x0, 0.058, pos.width, 0.016])
    cb = fig.colorbar(
        (
            axes[1, 0].collections[1]
            if False
            else plt.cm.ScalarMappable(norm=Normalize(980, 1030), cmap="Blues_r")
        ),
        cax=cax,
        orientation="horizontal",
        extend="both",
        ticks=[980, 1000, 1020],
    )
    cax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            mfc="white",
            mec="#222222",
            ls="none",
            ms=4,
            label="Forecast center",
        ),
        Line2D(
            [],
            [],
            marker="*",
            color="#222222",
            ls="none",
            ms=6,
            label="Observed center",
        ),
        Line2D([], [], color=COL["optimal"], lw=2, label="Annular wind"),
    ]
    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.032),
        ncol=1,
        frameon=False,
        fontsize=7,
        handlelength=1.5,
        labelspacing=0.25,
    )
    if annular_scale != 120:
        blue = next(q for q in axes[1, 2].collections if q.get_gid() == "annular-wind")
        axes[1, 2].quiverkey(
            blue,
            0.64,
            0.068,
            5,
            "Annular: 5 m s$^{-1}$",
            coordinates="figure",
            labelpos="W",
            labelcolor=COL["optimal"],
            fontproperties={"size": 7},
        )
    save(
        fig,
        name,
        ["data/steering.json"],
        "Separate CTRL/OPT at120h. NativeMSLP; height contours every40m with5880m emphasized;100km-filtered map winds, raw300–800km annular means. Gray quiver scale=120; blue annular scale="
        + str(annular_scale)
        + ". Common scales across states and models. Main figure="
        + str(model == "tianxing"),
    )


def s1():
    refs = [r for r in REV["reference_results"] if r["variant"] == "local3_gap1"]
    r = next(r for r in refs if r["case"] == "SAUDEL_LATE")
    rr = r["rows"]
    fig = plt.figure(figsize=(180 * MM, 88 * MM))
    ax = fig.add_subplot(1, 2, 1, projection=PC)
    base(ax, [103, 128, 12, 32], step=5)
    title(ax, "a", "Reference circulation centers")
    track(ax, [dict(x, center=x["reference"]) for x in rr], "reference")
    track(ax, rr, "optimal", label="ERA5T analysis")
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    bx = fig.add_subplot(1, 2, 2)
    clean(bx)
    title(bx, "b", "ERA5T–ATCF distance")
    ss = [x for x in rr if x["lead_hours"] > 0]
    bx.plot(
        [x["lead_hours"] for x in ss],
        [x["distance_km"] for x in ss],
        color=COL["optimal"],
        marker="o",
        ms=2.8,
    )
    bx.set(
        xlim=(0, 168),
        ylim=(0, 140),
        xticks=[0, 48, 96, 144, 168],
        xlabel="Lead relative to initialization (h)",
        ylabel="Center distance (km)",
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.90, wspace=0.30)
    save(
        fig,
        "figS01_reference",
        ["data/tracks.json"],
        "Same causal circulation diagnostic forERA5T; all28future distances shown; continuous archivedATCF reference without interpolation.",
    )


def s7():
    A = read("data/vorticity_budget.json")
    B = read("data/advection.json")
    phase = lambda d, src: next(
        r
        for r in d["phases"]
        if r["source"] == src
        and r["region"] == "eastern_core"
        and r["level"] == "low_mean"
        and r["start_hour"] == 24
    )
    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 145 * MM))
    fig.subplots_adjust(
        left=0.12, right=0.985, bottom=0.105, top=0.92, hspace=0.42, wspace=0.28
    )
    terms = ["relative_advection", "planetary_advection", "stretching", "residual"]
    labels = ["Relative\nadvection", "Planetary\nadvection", "Stretching", "Remainder"]
    parts = [
        "wind_error_on_reference",
        "reference_wind_on_vorticity_error",
        "nonlinear_error_advection",
    ]
    pls = [
        "Wind error ×\nreference gradient",
        "Reference wind ×\nerror gradient",
        "Nonlinear\nadvection",
    ]
    for j, model in enumerate(["tianxing", "pangu_6h"]):
        for i, (d, keys, labels0) in enumerate([(A, terms, labels), (B, parts, pls)]):
            ax = axes[i, j]
            clean(ax)
            for n, state in enumerate(["control", "era5init", "optimal"]):
                p = phase(d, state + "_" + model)
                v = p["error_growth_projection"] if i == 0 else p["components"]
                x = np.arange(len(keys)) + (n - 1) * 0.23
                ax.bar(
                    x,
                    [v[k] for k in keys],
                    width=0.21,
                    color=COL[state],
                    hatch=["//", "..", ""][n],
                    lw=0.25,
                    edgecolor="white",
                )
            ax.axhline(0, color="#333333", lw=0.6)
            ax.set_xticks(np.arange(len(keys)), labels0, fontsize=7)
            ax.set_ylim((-5, 8) if i == 0 else (-4, 5))
            ax.set_ylabel(
                "Cumulative contribution\n[(10$^{-5}$ s$^{-1}$)$^2$]" if j == 0 else ""
            )
            title(
                ax,
                "abcd"[i * 2 + j],
                ("TianXing" if j == 0 else "Pangu-Weather")
                + (" | resolved budget" if i == 0 else " | advection terms"),
            )
    fig.legend(
        handles=[
            plt.Rectangle(
                (0, 0), 1, 1, facecolor=COL[s], hatch=["//", "..", ""][i], label=LAB[s]
            )
            for i, s in enumerate(["control", "era5init", "optimal"])
        ],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.995),
        ncol=3,
        frameon=False,
    )
    save(
        fig,
        "figS05_resolved_processes",
        ["data/vorticity_budget.json", "data/advection.json"],
        "24–72h fixed12–25N127–142E. Full signed horizontal terms and retained remainder; bottom exact finite-amplitude advection decomposition; all levels/regions/phases retained in source tables. These are not causal fractions.",
    )
