"""Initial increments, regional retention, and early circulation evolution."""

from plot_common import *

OUT = ROOT / "data"
CFG = {"variants": ["eastasia_hard_keep", "eastasia_hard_remove"]}


def save(fig, key, caption, sources):
    import plot_common as g

    if fig._suptitle:
        fig._suptitle.remove()
    g.save(fig, key, sources)


def base(ax, extent, left=True, bottom=True):
    ax.set_extent(extent, crs=PC)
    ax.coastlines("50m", linewidth=0.4, color="#666666")
    step = 10 if extent[1] - extent[0] > 30 else 5
    ax.set_xticks(
        np.arange(np.ceil(extent[0] / step) * step, extent[1] + 0.1, step), crs=PC
    )
    ax.set_yticks(np.arange(np.ceil(extent[2] / 10) * 10, extent[3] + 0.1, 10), crs=PC)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelleft=left, labelbottom=bottom, length=2, width=0.5, pad=2)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color("#888888")


def initial():
    with xr.open_dataset(OUT / "initial_fields.nc") as f:
        d = f.sel(lat=slice(40, 5), lon=slice(105, 150)).load()
    lat, lon = d.lat.values, d.lon.values
    fig, axes = plt.subplots(2, 3, figsize=(7.09, 4.8), subplot_kw={"projection": PC})
    fig.subplots_adjust(
        left=0.05, right=0.99, bottom=0.10, top=0.94, wspace=0.22, hspace=0.40
    )
    z = (
        d.optimal_zeta850 + d.optimal_zeta700 - d.control_zeta850 - d.control_zeta700
    ).values / 2
    panels = [
        ("Lower-level circulation", z, 2.5, r"$10^{-5}$ s$^{-1}$", 850),
        ("925 hPa specific humidity", d.delta_q925.values, 0.8, "g kg$^{-1}$", 925),
        ("700 hPa specific humidity", d.delta_q700.values, 0.5, "g kg$^{-1}$", 700),
        ("850 hPa temperature", d.delta_t850.values, 0.1, "K", 850),
        ("700 hPa temperature", d.delta_t700.values, 0.1, "K", 700),
        ("500 hPa geopotential height", d.delta_z500.values, 4, "m", 500),
    ]
    for i, (ax, (label, arr, limit, unit, p)) in enumerate(zip(axes.flat, panels)):
        base(ax, [105, 150, 5, 40])
        ax.set_title(f"({chr(97+i)}) {label}", loc="left", pad=6)
        val = np.where(d.sp.values > p * 100, arr, np.nan)
        im = ax.pcolormesh(
            lon,
            lat,
            val,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="nearest",
            rasterized=True,
            transform=PC,
        )
        if i == 0:
            du = (d.delta_u850 + d.delta_u700) / 2
            dv = (d.delta_v850 + d.delta_v700) / 2
            q = ax.quiver(
                lon[::14],
                lat[::14],
                du.values[::14, ::14],
                dv.values[::14, ::14],
                color="#222222",
                scale=45,
                width=0.003,
                transform=PC,
            )
            ax.quiverkey(
                q,
                0.80,
                1.065,
                3,
                "3 m s$^{-1}$",
                labelpos="W",
                fontproperties={"size": 7},
            )
            bg = (d.control_zeta850 + d.control_zeta700) / 2
            cs = ax.contour(
                lon,
                lat,
                bg,
                levels=[2, 4, 8],
                colors="#555555",
                linewidths=0.65,
                transform=PC,
            )
            labels = ax.clabel(cs, fmt="%d", fontsize=7, inline_spacing=2)
            for label in labels:
                label.set_bbox(
                    dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.1)
                )
            ax.text(
                0.98,
                0.97,
                r"IFS: $10^{-5}$ s$^{-1}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
            )
        elif i in [1, 2]:
            bg = d["control_q" + str(p)]
            lev = [8, 12, 16] if p == 925 else [4, 8, 12]
            cs = ax.contour(
                lon,
                lat,
                bg,
                levels=lev,
                colors="#777777",
                linewidths=0.45,
                transform=PC,
            )
            ax.clabel(cs, fmt="%d", fontsize=6)
        elif i in [3, 4]:
            cs = ax.contour(
                lon,
                lat,
                d["control_t" + str(p)],
                levels=np.arange(260, 305, 5),
                colors="#777777",
                linewidths=0.45,
                transform=PC,
            )
            ax.clabel(cs, fmt="%d", fontsize=6)
        else:
            cs = ax.contour(
                lon,
                lat,
                d.control_z500,
                levels=[5840, 5880, 5920],
                colors="#777777",
                linewidths=0.5,
                transform=PC,
            )
            ax.clabel(cs, fmt="%d", fontsize=6)
            ax.contour(
                lon,
                lat,
                d.delta_msl,
                levels=[-0.2, 0.2],
                colors="#4a1486",
                linewidths=0.7,
                transform=PC,
            )
        ax.add_patch(
            Rectangle(
                (125, 12),
                20,
                13,
                fill=False,
                ec="#C33939",
                lw=0.8,
                transform=PC,
                zorder=10,
            )
        )
        ax.plot(124.9, 28.6, "*", color="#111111", ms=6, transform=PC, zorder=11)
        cb = fig.colorbar(
            im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.07, extend="both"
        )
        cb.set_label(unit, labelpad=1)
    save(fig, "figS02_signed_initial_structure", None, ["data/initial_fields.nc"])


def early():
    with xr.open_dataset(OUT / "early_fields.nc") as f:
        d = f.sel(lat=slice(30, 10), lon=slice(120, 145)).mean("level").load()
    lat, lon = d.lat.values, d.lon.values
    hours = [0, 6, 12, 24]
    fig, axes = plt.subplots(
        2, 4, figsize=(180 / 25.4, 118 / 25.4), subplot_kw={"projection": PC}
    )
    fig.subplots_adjust(
        left=0.078, right=0.99, bottom=0.19, top=0.92, wspace=0.16, hspace=0.28
    )
    for j, model in enumerate(["tianxing", "pangu_6h"]):
        for k, h in enumerate(hours):
            ax = axes[j, k]
            a = d.sel(model=model, lead=h)
            base(ax, [120, 145, 10, 30], left=k == 0)
            ax.set_xticks([120, 130, 140], crs=PC)
            ax.set_title(f"({chr(97+4*j+k)}) +{h} h", loc="left", pad=5)
            im = ax.pcolormesh(
                lon,
                lat,
                a.delta_zeta,
                cmap="RdBu_r",
                vmin=-4,
                vmax=4,
                shading="nearest",
                rasterized=True,
                transform=PC,
            )
            cs = ax.contour(
                lon,
                lat,
                a.control_zeta,
                levels=[1, 2, 4],
                colors="#555555",
                linewidths=0.6,
                transform=PC,
            )
            labels = ax.clabel(cs, fmt="%d", fontsize=7, inline_spacing=2)
            for label in labels:
                label.set_bbox(
                    dict(facecolor="white", edgecolor="none", alpha=0.95, pad=0.1)
                )
            q = ax.quiver(
                lon[::14],
                lat[::14],
                a.delta_u.values[::14, ::14],
                a.delta_v.values[::14, ::14],
                scale=45,
                width=0.005,
                transform=PC,
            )
            if j == 0 and k == 3:
                ax.quiverkey(
                    q,
                    0.68,
                    1.14,
                    3,
                    "3 m s$^{-1}$",
                    labelpos="W",
                    fontproperties={"size": 7},
                )
            ax.add_patch(
                Rectangle(
                    (127, 12),
                    15,
                    13,
                    fill=False,
                    ec="#C33939",
                    lw=0.65,
                    transform=PC,
                    zorder=10,
                )
            )
        pos = axes[j, 0].get_position()
        fig.text(
            0.012,
            (pos.y0 + pos.y1) / 2,
            "TianXing" if j == 0 else "Pangu-Weather",
            rotation=90,
            va="center",
            ha="center",
            fontsize=8,
        )
    cax = fig.add_axes([0.25, 0.10, 0.50, 0.025])
    fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        extend="both",
        label=r"Optimal − CTRL relative vorticity ($10^{-5}$ s$^{-1}$)",
    )
    save(fig, "figS04_early_circulation_response", None, ["data/early_fields.nc"])


def support():
    result = json.loads((OUT / "regional_tracks.json").read_text(encoding="utf8"))[
        "results"
    ]
    audit = json.loads((ROOT / "data/tracks.json").read_text(encoding="utf8"))[
        "results"
    ]
    baseline = lambda m, e: next(
        r
        for r in audit
        if r["case"] == "SAUDEL_LATE"
        and r["model"] == m
        and r["experiment"] == e
        and r["variant"] == "local3_gap1"
    )
    fig = plt.figure(figsize=(180 / 25.4, 174 / 25.4))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.97,
        bottom=0.075,
        top=0.96,
        wspace=0.24,
        hspace=0.34,
        height_ratios=[0.9, 1.1],
    )
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=120))
    ax.set_global()
    ax.coastlines("110m", linewidth=0.35, color="#666666")
    with xr.open_dataset(ROOT / "data/energy.nc") as d:
        im = ax.pcolormesh(
            d.lon[::2],
            d.lat[::2],
            np.maximum(d.total.values[::2, ::2], 1e-10),
            cmap="Blues",
            norm=LogNorm(1e-4, 100),
            shading="nearest",
            rasterized=True,
            transform=PC,
        )
    ax.plot(
        [90, 170, 170, 90, 90], [0, 0, 55, 55, 0], color="#C33939", lw=1, transform=PC
    )
    ax.set_title("(a) Global correction and retained domain", loc="left", pad=9)
    cb = fig.colorbar(
        im, ax=ax, orientation="horizontal", fraction=0.07, pad=0.08, extend="both"
    )
    cb.set_label("Column moist energy (J kg$^{-1}$)", labelpad=1)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("(b) Seven-day mean track error", loc="left")
    colors = ["#C46A25", "#0072B2", "#009E73", "#8662A6"]
    labels = ["CTRL", "Full Optimal", "Region only", "Outside only"]
    styles = ["--", "-", "-.", ":"]
    markers = ["^", "o", "s", "D"]
    for k, model in enumerate(["tianxing", "pangu_6h"]):
        groups = [baseline(model, "baseline"), baseline(model, "corrected")] + [
            next(r for r in result if r["model"] == model and r["variant"] == v)
            for v in CFG["variants"]
        ]
        for i, g in enumerate(groups):
            err = [
                r["distance_km"]
                for r in g["rows"][1:]
                if r.get("distance_km") is not None
            ]
            v = float(np.mean(err))
            ax.bar(
                k + (i - 1.5) * 0.18,
                v,
                width=0.17,
                color=colors[i],
                label=labels[i] if k == 0 else None,
                hatch=["//", "", "..", "xx"][i],
                edgecolor="white",
                linewidth=0.4,
            )
            ax.text(k + (i - 1.5) * 0.18, v + 7, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1], ["TianXing", "Pangu-Weather"])
    ax.set_ylabel("Mean track error (km)")
    ax.set_ylim(0, max(500, ax.get_ylim()[1] + 25))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        frameon=False,
        columnspacing=0.8,
    )
    for k, model in enumerate(["tianxing", "pangu_6h"]):
        ax = fig.add_subplot(gs[1, k], projection=PC)
        base(ax, [104, 128, 15, 31], left=True)
        ax.set_title(
            f"({chr(99+k)}) " + ("TianXing" if k == 0 else "Pangu-Weather"), loc="left", pad=6
        )
        groups = [baseline(model, "baseline"), baseline(model, "corrected")] + [
            next(r for r in result if r["model"] == model and r["variant"] == v)
            for v in CFG["variants"]
        ]
        rr = groups[0]["rows"]
        ax.plot(
            [r["reference"]["lon"] for r in rr],
            [r["reference"]["lat"] for r in rr],
            color="#222222",
            lw=1.3,
            label="Reference",
            transform=PC,
            zorder=9,
        )
        for i, g in enumerate(groups):
            rr = g["rows"]
            ax.plot(
                [r["center"]["lon"] if r.get("center") else np.nan for r in rr],
                [r["center"]["lat"] if r.get("center") else np.nan for r in rr],
                color=colors[i],
                ls=styles[i],
                lw=1.1,
                marker=markers[i],
                markevery=4,
                ms=2.8,
                transform=PC,
                label=labels[i],
            )
        if k == 0:
            ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    save(
        fig,
        "figS03_regional_support",
        None,
        ["data/regional_tracks.json", "data/tracks.json", "data/energy.nc"],
    )
