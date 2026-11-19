"""Rate-accuracy (RD) curves as PNG: one chart per (sequence, task), one line per model."""

from __future__ import annotations

from pathlib import Path

# Categorical slots in fixed order (dataviz reference palette, light surface). A model keeps the
# colour of its position in the config's model list, so colours follow the model, not its rank.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]   # secondary encoding: never colour alone
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def plot_rd(series: list[dict], title: str, metric_label: str, out_path: Path, log_x: bool = False,
            bitrate_unit: str | None = "kbps") -> Path:
    """series: [{"label": str, "slot": int, "points": [{"qp", "bitrate", "value"}]}]; value in %.
    bitrate_unit: x-axis unit label (None = "Bitrate" without a unit)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=400)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    drawn = 0
    qp_at: dict[int, float] = {}
    for s in series:
        pts = sorted((p for p in s["points"] if p.get("bitrate") is not None and p.get("value") is not None),
                     key=lambda p: p["bitrate"])
        if not pts:
            continue
        slot = s["slot"] % len(SERIES_COLORS)
        x = [p["bitrate"] for p in pts]
        y = [p["value"] for p in pts]
        ax.plot(x, y, color=SERIES_COLORS[slot], linewidth=2, marker=MARKERS[slot], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=s["label"], zorder=3)
        drawn += 1
        for p in pts:
            qp_at.setdefault(p["qp"], p["bitrate"])
    _style_axes(ax, metric_label, log_x, bitrate_unit)
    n_rows = _qp_labels(ax, qp_at)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8 + 10 * n_rows)
    if drawn >= 2:
        leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
        for t in leg.get_texts():
            t.set_color(INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def _style_axes(ax, metric_label: str, log_x: bool, bitrate_unit: str | None) -> None:
    """Shared axis styling of the RD plots: recessive grid and spines, bitrate axis from 0 (or log), y from <= 0."""
    if log_x:
        ax.set_xscale("log")
        from matplotlib.ticker import FuncFormatter
        fmt = FuncFormatter(lambda v, _: f"{v:g}")
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if str(v)[0] in "25" else ""))
    ax.set_xlabel("Bitrate" + (f" ({bitrate_unit})" if bitrate_unit else "") + (", log scale" if log_x else ""),
                  color=INK2, fontsize=9)
    ax.set_ylabel(metric_label, color=INK2, fontsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.grid(True, which="minor", axis="x", color=GRID, linewidth=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(which="both", colors=INK2, labelsize=8)
    if not log_x:
        ax.set_xlim(left=0)
    ymin = ax.get_ylim()[0]
    ax.set_ylim(bottom=min(0, ymin))


def _qp_labels(ax, qp_at: dict[int, float], min_gap: float = 0.06, max_rows: int = 4) -> int:
    """QP labels along the top edge (a QP has the same bitrate for every model), so they never collide
    with markers. Labels closer than min_gap (axes fraction) to one in the same row move up a row."""
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    to_frac = lambda x: ax.transAxes.inverted().transform(ax.transData.transform((x, 0)))[0]  # noqa: E731
    last = [-1.0] * max_rows
    used = 0
    for qp, x in sorted(qp_at.items(), key=lambda kv: kv[1]):
        fx = to_frac(x)
        row = next((r for r in range(max_rows) if fx - last[r] >= min_gap), max_rows - 1)
        last[row] = fx
        used = max(used, row + 1)
        ax.text(x, 1.01 + 0.04 * row, f"QP{qp}", transform=trans, ha="center", va="bottom",
                fontsize=7, color=INK2)
        ax.plot([x, x], [1.0, 1.01 + 0.04 * row], transform=trans, color=GRID, linewidth=0.6,
                clip_on=False)
    return used



def plot_bd(series: list[dict], title: str, metric_label: str, out_path: Path, log_x: bool = False,
            bitrate_unit: str | None = "kbps") -> Path:
    """Anchor vs test RD curves (BD comparison). series: [{"label", "slot", "anchor": curve, "test": curve}],
    curve = {"kept": [(bitrate, value %)], "removed": [(bitrate, value %)]}. Anchor solid, test dashed, same colour
    and marker per model; Pareto points filled and joined, removed (dominated) points hollow and not joined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=400)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    any_removed = False
    handles = []
    for s in series:
        slot = s["slot"] % len(SERIES_COLORS)
        col, mk = SERIES_COLORS[slot], MARKERS[slot]
        drawn = False
        for side, ls in (("anchor", "-"), ("test", "--")):
            c = s.get(side) or {}
            kept = sorted(c.get("kept") or [])
            if kept:
                ax.plot([p[0] for p in kept], [p[1] for p in kept], color=col, linestyle=ls, linewidth=2,
                        marker=mk, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
                drawn = True
            rem = c.get("removed") or []
            if rem:
                any_removed = True
                ax.plot([p[0] for p in rem], [p[1] for p in rem], linestyle="none", marker=mk, markersize=7,
                        markerfacecolor=SURFACE, markeredgecolor=col, markeredgewidth=1.5, zorder=4)
                drawn = True
        if drawn:
            handles.append(Line2D([], [], color=col, linewidth=2, marker=mk, markersize=6,
                                  markeredgecolor=SURFACE, label=s["label"]))
    _style_axes(ax, metric_label, log_x, bitrate_unit)
    ax.set_title(title, color=INK, fontsize=9, loc="left", pad=8)
    handles += [Line2D([], [], color=INK2, linewidth=1.5, linestyle="-", label="anchor"),
                Line2D([], [], color=INK2, linewidth=1.5, linestyle="--", label="test")]
    if any_removed:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=SURFACE,
                              markeredgecolor=INK2, label="not on Pareto front (excluded)"))
    leg = ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path
