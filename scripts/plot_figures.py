"""Reproduce the four main figures and eight supporting figures."""

from plot_common import *
import plot_common as g
import plot_regions as reg
import matplotlib.patheffects as pe

D = read("data/diagnostics.json")
B = read("data/budget_regions.json")
C = COL
L = {**LAB, "reference": "ERA5T analysis"}
S = LS


def initial(wind_shading="zonal", speed_palette="soft_blue"):
    if wind_shading not in ["zonal", "speed"]:
        raise ValueError("wind_shading must be 'zonal' or 'speed'")
    fig = plt.figure(figsize=(7.09, 5.9))
    boxes = [
        [0.08, 0.60, 0.375, 0.31],
        [0.58, 0.60, 0.375, 0.31],
        [0.08, 0.115, 0.375, 0.31],
        [0.58, 0.115, 0.375, 0.31],
    ]
    ax = fig.add_axes(boxes[0], projection=reg.PC)
    reg.base(ax, [85, 155, 0, 50])
    d = xr.load_dataset(ROOT / "data/energy.nc").sel(
        lat=slice(50, 0), lon=slice(85, 155)
    )
    im = ax.pcolormesh(
        d.lon,
        d.lat,
        d.total,
        cmap="Blues",
        norm=LogNorm(1e-4, 100),
        shading="nearest",
        rasterized=True,
        transform=reg.PC,
    )
    ax.add_patch(
        Rectangle((125, 12), 20, 13, fill=False, ec="#C33939", lw=0.9, transform=reg.PC)
    )
    ax.plot(124.9, 28.6, "*", color="#222222", ms=6, transform=reg.PC)
    ax.set_title("(a) Column perturbation energy", loc="left")
    cb = fig.colorbar(
        im,
        cax=fig.add_axes([0.08, 0.543, 0.375, 0.014]),
        orientation="horizontal",
        extend="both",
        ticks=[1e-4, 1e-2, 1, 100],
    )
    cb.set_label("Column moist energy (J kg$^{-1}$)", labelpad=1)
    d = xr.load_dataset(ROOT / "data/initial_fields.nc").sel(
        lat=slice(26, 11), lon=slice(124.5, 145.5)
    )
    ax = fig.add_axes(boxes[1], projection=reg.PC)
    reg.base(ax, [124.5, 145.5, 11, 26])
    ax.set_yticks([15, 20, 25], crs=reg.PC)
    z = (
        d.optimal_zeta850 + d.optimal_zeta700 - d.control_zeta850 - d.control_zeta700
    ) / 2
    im = ax.pcolormesh(
        d.lon,
        d.lat,
        z,
        cmap="RdBu_r",
        vmin=-2.5,
        vmax=2.5,
        shading="nearest",
        rasterized=True,
        transform=reg.PC,
    )
    du = (d.delta_u850 + d.delta_u700) / 2
    dv = (d.delta_v850 + d.delta_v700) / 2
    q = ax.quiver(
        d.lon[::8],
        d.lat[::8],
        du.values[::8, ::8],
        dv.values[::8, ::8],
        scale=40,
        width=0.0036,
        transform=reg.PC,
        color="#333333",
    )
    ax.quiverkey(
        q, 0.90, 1.065, 3, "3 m s$^{-1}$", labelpos="W", fontproperties={"size": 7}
    )
    background = (d.control_zeta850 + d.control_zeta700) / 2
    # The plotted IFS field peaks at 7.074 in units of 1e-5 s^-1:
    # the previously requested level 8 has no contour in this domain.
    contours = ax.contour(
        d.lon,
        d.lat,
        background,
        levels=[2, 4],
        colors="#555555",
        linewidths=0.75,
        transform=reg.PC,
    )
    labels = ax.clabel(contours, fmt="%d", fontsize=7, inline=True, inline_spacing=3)
    for label in labels:
        label.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.1))
    ax.text(
        0.98,
        0.97,
        r"IFS contours: $10^{-5}$ s$^{-1}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#444444",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
    )
    ax.add_patch(
        Rectangle((128, 18), 5, 5, fill=False, ec="#C33939", lw=0.9, transform=reg.PC)
    )
    ax.set_title("(b) Circulation increment", loc="left")
    cb = fig.colorbar(
        im,
        cax=fig.add_axes([0.58, 0.543, 0.375, 0.014]),
        orientation="horizontal",
        extend="both",
        ticks=[-2, -1, 0, 1, 2],
    )
    cb.set_label(r"Vorticity increment ($10^{-5}$ s$^{-1}$)", labelpad=1)
    if speed_palette == "cmaps.sunshine_9lev":
        import cmaps

        speed_cmap = cmaps.sunshine_9lev
    else:
        speed_cmap = (
            matplotlib.colors.ListedColormap(
                plt.get_cmap("Blues")(np.linspace(0.04, 0.82, 256)), name="wind_blues"
            )
            if speed_palette == "soft_blue"
            else plt.get_cmap(speed_palette)
        )
    for j in range(2):
        ax = fig.add_axes(boxes[j + 2], projection=reg.PC)
        reg.base(ax, [124.5, 145.5, 11, 26])
        ax.set_yticks([15, 20, 25], crs=reg.PC)
        u = d.control_u700 + (d.delta_u700 if j else 0)
        v = d.control_v700 + (d.delta_v700 if j else 0)
        field = np.hypot(u, v) if wind_shading == "speed" else u
        cmap = speed_cmap if wind_shading == "speed" else "RdBu_r"
        limits = (0, 20) if wind_shading == "speed" else (-24, 24)
        norm = (
            matplotlib.colors.BoundaryNorm(np.arange(0, 22, 2), speed_cmap.N)
            if wind_shading == "speed" and speed_palette == "cmaps.sunshine_9lev"
            else matplotlib.colors.Normalize(*limits)
        )
        im = ax.pcolormesh(
            d.lon,
            d.lat,
            field,
            cmap=cmap,
            norm=norm,
            shading="nearest",
            rasterized=True,
            transform=reg.PC,
        )
        q = ax.quiver(
            d.lon[::8],
            d.lat[::8],
            u.values[::8, ::8],
            v.values[::8, ::8],
            scale=160,
            width=0.0036,
            transform=reg.PC,
            color="#333333",
        )
        if wind_shading == "speed":
            arrow_outline = (
                0.20
                if speed_palette == "soft_blue"
                else (0.12 if speed_palette == "cmaps.sunshine_9lev" else 0.35)
            )
            q.set_path_effects(
                [pe.Stroke(linewidth=arrow_outline, foreground="white"), pe.Normal()]
            )
        ax.quiverkey(
            q,
            0.90,
            1.065,
            10,
            "10 m s$^{-1}$",
            labelpos="W",
            fontproperties={"size": 7},
        )
        box = Rectangle(
            (128, 18), 5, 5, fill=False, ec="#C33939", lw=0.9, transform=reg.PC
        )
        if wind_shading == "speed" and speed_palette == "cmaps.sunshine_9lev":
            box.set_path_effects(
                [pe.Stroke(linewidth=1.4, foreground="white"), pe.Normal()]
            )
        ax.add_patch(box)
        ax.plot(131, 20, "o", mfc="white", mec="#222222", ms=3.5, transform=reg.PC)
        ax.set_title(
            f'({"cd"[j]}) '
            + ("IFS initial wind" if j == 0 else "Optimal initial wind"),
            loc="left",
        )
    ticks = [0, 5, 10, 15, 20] if wind_shading == "speed" else [-24, -12, 0, 12, 24]
    if wind_shading == "speed" and speed_palette == "cmaps.sunshine_9lev":
        ticks = [0, 4, 8, 12, 16, 20]
    cb = fig.colorbar(
        im,
        cax=fig.add_axes([0.26, 0.045, 0.50, 0.015]),
        orientation="horizontal",
        extend="neither" if wind_shading == "speed" else "both",
        ticks=ticks,
    )
    cb.set_label(
        "700 hPa "
        + ("wind speed" if wind_shading == "speed" else "zonal wind")
        + " (m s$^{-1}$)",
        labelpad=1,
    )
    stem = "fig02_initial_correction" + ("_speed" if wind_shading == "speed" else "")
    save(fig, stem, ["data/initial_fields.nc", "data/energy.nc"])
    if wind_shading == "speed":
        palette_description = (
            "256 uniformly spaced samples from Matplotlib Blues, from 0.04 to 0.82"
            if speed_palette == "soft_blue"
            else "Matplotlib " + speed_palette
        )
        if speed_palette == "cmaps.sunshine_9lev":
            palette_description = "Original cmaps.sunshine_9lev, all 10 colors retained, with equally spaced boundaries at 0, 2, ..., 20 m/s. Red boxes have a white outline for visibility against warm colors"
        g.manifest[-1]["alteration"] = (
            "Panels c and d show sqrt(u700**2 + v700**2), computed from each complete initial state. Shared linear range 0–20 m/s. Sequential colormap: "
            + palette_description
            + ". Wind arrows have a thin white outline for contrast; their components and scale are unchanged. All other fields are unchanged."
        )


def evolution():
    fig, axes = plt.subplots(3, 2, figsize=(7.09, 6.75))
    fig.subplots_adjust(
        left=0.135, right=0.985, bottom=0.065, top=0.955, hspace=0.74, wspace=0.36
    )
    for ax in axes.flat:
        clean(ax)
    for j, model in enumerate(["tianxing", "pangu_6h"]):
        name = ["TianXing", "Pangu-Weather"][j]
        ax = axes[0, j]
        for state in ["control", "optimal", "era5init"]:
            rr = [
                r
                for r in D["K_curves"]
                if r["source"] == state + "_" + model
                and r["region"] == "eastern_core"
                and r["lead_hours"] <= 72
            ]
            ax.plot(
                [r["lead_hours"] for r in rr],
                [r["K_scaled"] for r in rr],
                color=C[state],
                ls=S[state],
                label=L[state],
                lw=1.45,
            )
        ax.axvspan(24, 72, color="#666666", alpha=0.06, zorder=0)
        ax.set(
            xlim=(0, 72),
            ylim=(0, 10),
            xticks=[0, 24, 48, 72],
            xlabel="Forecast lead (h)",
            ylabel=r"$K_\zeta$ / $(10^{-5}$ s$^{-1})^2$",
        )
        ax.set_title(f'({"ab"[j]}) {name} | circulation error', loc="left")
        if j == 0:
            ax.legend(frameon=False, loc="upper left", fontsize=6.8)
        for state in ["control", "optimal", "reference"]:
            key = {
                "control": "ifs_control",
                "optimal": "ifs_optimal",
                "reference": "era5t_reference",
            }[state]
            rr = g.SYS[model][:17]
            axes[1, j].plot(
                [r["lead_hours"] for r in rr],
                [
                    r["fields"][key]["detail"][
                        "z500_5880_west_intersection_from150E_at_latitude"
                    ]["30"]
                    for r in rr
                ],
                color=C[state],
                ls=S[state],
                lw=1.45,
                label=L[state],
            )
            rr = [
                r
                for r in g.CLEAR
                if r["model"] == ("era5t" if state == "reference" else model)
                and r["state"] == state
            ]
            axes[2, j].plot(
                [r["lead_hours"] for r in rr],
                [r["steering"]["v_850_500"] for r in rr],
                color=C[state],
                ls=S[state],
                lw=1.45,
                label=L[state],
            )
        axes[1, j].set_title(f'({"cd"[j]}) {name} | ridge western edge', loc="left")
        axes[1, j].set(
            xlim=(0, 96),
            ylim=(116, 134),
            xticks=[0, 24, 48, 72, 96],
            xlabel="Forecast lead (h)",
            ylabel="5880 m edge at 30°N (°E)",
        )
        if j == 1:
            axes[1, j].legend(frameon=False, loc="upper right", fontsize=6.8)
        axes[2, j].set_title(f'({"ef"[j]}) {name} | meridional steering', loc="left")
        axes[2, j].set(
            xlim=(0, 168),
            ylim=(-4.5, 4.5),
            xticks=[0, 48, 96, 144, 168],
            xlabel="Forecast lead (h)",
            ylabel=r"850–500 hPa $v$ (m s$^{-1}$)",
        )
        axes[2, j].axhline(0, color="#777777", lw=0.6)
    save(
        fig,
        "fig03_environmental_evolution",
        ["data/diagnostics.json", "data/ridge.json", "data/steering.json"],
    )


def boxes():
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 2.6), layout="constrained")
    for j, model in enumerate(["tianxing", "pangu_6h"]):
        ax = axes[j]
        clean(ax)
        x = np.arange(5)
        for k, state in enumerate(["control", "optimal", "era5init"]):
            vals = [
                next(
                    r
                    for r in B["rows"]
                    if r["model"] == model
                    and r["state"] == state
                    and r["region"] == region
                )["contributions"]["relative_advection"]
                for region in [
                    "core",
                    "expanded",
                    "west_shift",
                    "east_shift",
                    "north_shift",
                ]
            ]
            ax.bar(
                x + (k - 1) * 0.25,
                vals,
                width=0.24,
                color=C[state],
                hatch=["//", "", ".."][k],
                edgecolor="white",
                lw=0.25,
                label=L[state],
            )
        ax.axhline(0, color="#555555", lw=0.6)
        ax.set(
            xticks=x,
            xticklabels=["Core", "Larger", "West", "East", "North"],
            ylim=(-2, 6),
            ylabel="Advection contribution\n" + r"[$(10^{-5}$ s$^{-1})^2$]",
        )
        ax.set_title(f'({"ab"[j]}) ' + ["TianXing", "Pangu-Weather"][j], loc="left")
        if j == 0:
            ax.legend(frameon=False, loc="upper left", fontsize=7)
    save(fig, "figS06_budget_regions", ["data/budget_regions.json"])


def steering():
    fig, axes = plt.subplots(3, 2, figsize=(7.09, 7.0))
    fig.subplots_adjust(
        left=0.12, right=0.985, bottom=0.065, top=0.96, hspace=0.67, wspace=0.35
    )
    for ax in axes.flat:
        clean(ax)
    for j, model in enumerate(["tianxing", "pangu_6h"]):
        name = ["TianXing", "Pangu-Weather"][j]
        r = next(
            r
            for r in D["steering"]
            if r["model"] == model
            and r["component"] == "v"
            and r["layer_hpa"] == "850–500"
            and r["lead_hours"] == 120
        )
        ax = axes[0, j]
        for k, state in enumerate(["control", "optimal"]):
            x = np.arange(2) + (k - 0.5) * 0.28
            v = [r[state + "_control"], r[state + "_optimal"]]
            ax.bar(
                x,
                v,
                width=0.26,
                color=C[state],
                hatch="//" if k == 0 else "",
                edgecolor="white",
                lw=0.3,
                label=L[state] + " wind field",
            )
            for a, b in zip(x, v):
                ax.text(
                    a,
                    b + (0.16 if b >= 0 else -0.16),
                    f"{b:+.2f}",
                    ha="center",
                    va="bottom" if b >= 0 else "top",
                    fontsize=7,
                )
        ax.set(
            xticks=[0, 1],
            xticklabels=["CTRL center", "Optimal center"],
            ylim=(-6, 4.5),
            ylabel=r"850–500 hPa $v$ (m s$^{-1}$)",
        )
        ax.set_title(f'({"ab"[j]}) {name} | +120 h', loc="left")
        ax.axhline(0, color="#555555", lw=0.6)
        if j == 0:
            ax.legend(frameon=False, loc="lower left", fontsize=6.6)
        for i, layer in enumerate(["850–500", "850–700"], 1):
            ax = axes[i, j]
            rr = [
                r
                for r in D["steering"]
                if r["model"] == model
                and r["component"] == "v"
                and r["layer_hpa"] == layer
                and r["lead_hours"] >= 72
            ]
            h = np.array([r["lead_hours"] for r in rr])
            for key, col, a, b in [
                (
                    "field",
                    "#8663A8",
                    "field_at_control_center",
                    "field_at_optimal_center",
                ),
                (
                    "position",
                    "#009E73",
                    "position_in_control_field",
                    "position_in_optimal_field",
                ),
            ]:
                v = np.array([r[a] for r in rr])
                w = np.array([r[b] for r in rr])
                ax.fill_between(
                    h, np.minimum(v, w), np.maximum(v, w), color=col, alpha=0.18, lw=0
                )
                ax.plot(
                    h,
                    [r[key + "_symmetric"] for r in rr],
                    color=col,
                    lw=1.45,
                    label=(
                        "Wind-field change" if key == "field" else "Center displacement"
                    ),
                )
            ax.plot(
                h,
                [r["total"] for r in rr],
                color="#222222",
                ls="--",
                lw=1.2,
                label="Total difference",
            )
            ax.axhline(0, color="#777777", lw=0.6)
            ax.axvline(120, color="#999999", ls=":", lw=0.6)
            ax.set(
                xlim=(72, 168),
                xticks=[72, 96, 120, 144, 168],
                ylim=(-7, 11),
                xlabel="Forecast lead (h)",
                ylabel=r"Contribution to $\Delta v$ (m s$^{-1}$)",
            )
            ax.set_title(f'({"cdef"[(i-1)*2+j]}) {name} | {layer} hPa', loc="left")
            if i == 1 and j == 0:
                ax.legend(frameon=False, loc="upper left", fontsize=6.6)
    save(fig, "figS08_steering_sampling", ["data/diagnostics.json"])


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "figures")
    p.add_argument("--only", choices=["all", "main", "supplement"], default="all")
    args = p.parse_args()
    g.OUT = args.output.resolve()
    if args.only in ["all", "main"]:
        g.fig1()
        initial("speed", "cmaps.sunshine_9lev")
        evolution()
        g.absolute_pair("tianxing", "fig04_system_configuration", annular_scale=40)
    if args.only in ["all", "supplement"]:
        g.s1()
        reg.initial()
        reg.support()
        reg.early()
        g.s7()
        boxes()
        g.absolute_pair("pangu_6h", "figS07_pangu_configuration", annular_scale=40)
        steering()
    (g.OUT / "sources.json").write_text(
        json.dumps(g.manifest, indent=2), encoding="utf8"
    )


if __name__ == "__main__":
    main()
