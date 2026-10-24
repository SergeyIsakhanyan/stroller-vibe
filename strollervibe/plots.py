"""Static charts for the report (matplotlib, PNG)."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
REFS = [(0.315, "0.315"), (0.63, "0.63")]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.labelcolor": INK2,
})


def to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    try:  # 256-colour palette PNG: ~3x smaller, visually identical for flat charts
        from PIL import Image
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return buf.getvalue()


def walk_figure(res) -> bytes:
    s, m = res.series, res.metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.45})
    t = s["t"] - s["t"][0]
    for a, b in _runs(~s["moving"]):
        ax1.axvspan(t[a], t[min(b, len(t) - 1)], color=GRID, alpha=0.7, lw=0)
    ax1.plot(t, s["az_w"], color=MUTED, lw=0.5, alpha=0.7, label="Wk-weighted vertical acceleration")
    ax1.plot(t, s["run_rms_z"], color=SERIES[0], lw=2, label="1-s running RMS")
    ax1.axhline(m["aw_z"], color=INK2, lw=1, ls="--")
    ax1.annotate(f"a_wz {m['aw_z']:.2f}", xy=(1.0, m["aw_z"]), xycoords=("axes fraction", "data"),
                 xytext=(4, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)
    if (~s["moving"]).any():
        ax1.plot([], [], color=GRID, lw=8, label="stopped (excluded)")
    meta = res.meta
    title = " · ".join(x for x in [res.walk_id, str(meta.get("surface", "")).replace("_", " "),
                                    str(meta.get("stroller", "")), str(meta.get("load", ""))] if x)
    ax1.set_title(title)
    ax1.set_xlabel("time in analysed part (s)")
    ax1.set_ylabel("m/s²")
    ax1.set_xlim(0, t[-1])
    ax1.legend(loc="upper left", fontsize=8.5, ncol=3, bbox_to_anchor=(0, 1.0))
    lim = max(np.max(np.abs(s["az_w"])), 0.1)
    ax1.set_ylim(-lim * 1.05, lim * 1.35)

    f = s["f"]
    keep = (f >= 0.5) & (f <= s["fs"] / 2)
    ax2.loglog(f[keep], s["psd_u"][keep], color=MUTED, lw=1.5, label="unweighted (0.4 Hz high-pass)")
    ax2.loglog(f[keep], s["psd_w"][keep], color=SERIES[0], lw=2, label="Wk-weighted")
    pk = m["peak_hz"]
    if np.isfinite(pk):
        i = np.argmin(np.abs(f - pk))
        ax2.plot([f[i]], [s["psd_w"][i]], "o", ms=8, color=SERIES[0], mec=SURFACE, mew=2)
        ax2.annotate(f"{pk:.1f} Hz", (f[i], s["psd_w"][i]), xytext=(6, 6), textcoords="offset points", color=INK, fontsize=9)
    ax2.set_title("Vertical vibration spectrum")
    ax2.set_xlabel("frequency (Hz)")
    ax2.set_ylabel("PSD ((m/s²)²/Hz)")
    ax2.legend(loc="lower left", fontsize=8.5)
    ax2.grid(True, which="minor", color=GRID, lw=0.3)
    return to_png(fig)


def comparison_figure(walks) -> bytes:
    loads = list(dict.fromkeys(walks["load"]))
    order = walks.groupby("surface")["aw_z"].median().sort_values().index.tolist()
    strollers = sorted(walks["stroller"].unique())
    fig, axes = plt.subplots(len(loads), 1, figsize=(8.5, 3.4 * len(loads) + 0.9), squeeze=False)
    ymax = max(walks["aw_z"].max() * 1.15, 0.7)
    for ax, load in zip(axes[:, 0], loads):
        sub = walks[walks["load"] == load]
        surf = [s for s in order if s in set(sub["surface"])]
        for ref, lab in REFS:
            ax.axhline(ref, color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.annotate(lab, xy=(1.0, ref), xycoords=("axes fraction", "data"), xytext=(4, 0),
                        textcoords="offset points", va="center", color=MUTED, fontsize=8.5)
        for xi, s in enumerate(surf):
            pts = sub[sub["surface"] == s]
            med = pts["aw_z"].median()
            ax.plot([xi - 0.32, xi + 0.32], [med, med], color=INK, lw=2, solid_capstyle="round", zorder=3)
            for k, st in enumerate(strollers):
                p = pts[pts["stroller"] == st]
                if p.empty:
                    continue
                off = (k - (len(strollers) - 1) / 2) * 0.16 if len(strollers) > 1 else 0
                jitter = np.linspace(-0.04, 0.04, len(p)) if len(p) > 1 else np.zeros(1)
                ax.scatter(xi + off + jitter, p["aw_z"], s=46, color=SERIES[k % 8], marker=MARKERS[k % 8],
                           edgecolors=SURFACE, linewidths=1.5, zorder=4)
        ax.set_xticks(range(len(surf)))
        ax.set_xticklabels([s.replace("_", "\n") for s in surf])
        ax.set_xlim(-0.6, max(len(surf), 3) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"load: {load.replace('_', ' ')}", fontsize=10.5)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("a_wz (m/s²)")
    handles = [plt.Line2D([], [], marker=MARKERS[k % 8], color=SERIES[k % 8], ls="", ms=7, mec=SURFACE, mew=1.2)
               for k in range(len(strollers))]
    labels = list(strollers)
    handles.append(plt.Line2D([], [], color=INK, lw=2))
    labels.append("median")
    handles.append(plt.Line2D([], [], color=MUTED, lw=1, ls=(0, (4, 3))))
    labels.append("ISO 2631-1 Annex C bands (adult reference)")
    fig.legend(handles, labels, loc="upper left", ncol=2, fontsize=8.5, bbox_to_anchor=(0.0, 0.955), frameon=False)
    fig.suptitle("Vertical vibration (Wk-weighted RMS) by surface", x=0.01, y=0.995, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88 if len(loads) == 1 else 0.92))
    return to_png(fig)


def speed_figure(walks) -> bytes | None:
    d = walks.dropna(subset=["speed_kmh"])
    if len(d) < 3 or d["speed_kmh"].nunique() < 2:
        return None
    surfaces = d.groupby("surface")["aw_z"].median().sort_values().index.tolist()
    n = len(surfaces)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols + 0.8, 2.5 * rows + 0.6), sharex=True, sharey=True, squeeze=False)
    for ax, s in zip(axes.flat, surfaces):
        ax.scatter(d["speed_kmh"], d["aw_z"], s=18, color=GRID, edgecolors="none", zorder=2)
        p = d[d["surface"] == s]
        ax.scatter(p["speed_kmh"], p["aw_z"], s=40, color=SERIES[0], edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.set_title(s.replace("_", " "), fontsize=10)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("a_wz (m/s²)")
    for ax in axes[-1, :]:
        ax.set_xlabel("speed (km/h)")
    fig.suptitle("Vertical vibration vs walking speed (grey: all other surfaces)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return to_png(fig)


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
