"""Charts for the sweep and for single sessions (matplotlib, saved as PNG).

Conventions shared by every chart:

* One color per strategy everywhere: inventory-aware = blue, naive = orange.
  Reference distributions are muted gray, and P&L components get their own
  pair of colors. All combinations pass a color-vision-deficiency check.
* 95% confidence intervals are drawn as light bands around each line.
* Every multi-series chart has a legend plus a few direct labels, and a
  headline that states the takeaway, computed from the data.
* The numbers behind every sweep chart are in results/sweep_results.csv.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FuncFormatter,
    MultipleLocator,
    PercentFormatter,
)

if TYPE_CHECKING:
    from session import SessionResult
    from sweep import SweepPoint, SweepResult

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
REFERENCE = "#898781"
MODE_COLOR = {"inventory-aware": "#2a78d6", "naive": "#eb6834"}
MODE_NAME = {"inventory-aware": "Inventory-aware", "naive": "Naive"}
CAPTURE_COLOR = "#4a3aa7"
ADVERSE_COLOR = "#1baf7a"

LINE = 1.5  # points; about 2 px when the 200-dpi image is shown at README width
MARKER = 6.5
RING = 1.5
BAND_ALPHA = 0.15
DPI = 200

X_LABELS = {
    "base_spread": "Base spread (bid-ask width when flat)",
    "inventory_penalty": "Inventory penalty ($ of quote skew per unit held)",
    "spread_widening": "Spread widening ($ per unit of |inventory|)",
    "max_inventory": "Max inventory (units)",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "text.parse_math": False,  # labels are full of dollar signs, not LaTeX
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.labelsize": 9.5,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
})


# ----------------------------------------------------------------------------- helpers


def _figure(height: float, nrows: int = 1, ncols: int = 1, width: float = 8.0, **subplot_kw):
    """Figure with fixed margins (in inches) so headlines line up with the plot area."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False, **subplot_kw)
    fig.subplots_adjust(
        left=0.85 / width, right=1 - 0.3 / width, top=1 - 0.95 / height, bottom=0.62 / height,
        hspace=0.45, wspace=0.12,
    )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)
    return fig, axes


def _headline(fig, title: str, subtitle: str) -> None:
    """Title and subtitle aligned with the plot area, shrunk if needed to fit the width."""
    left, height = fig.subplotpars.left, fig.get_figheight()
    max_width = fig.bbox.width * (1 - left) - 0.2 * fig.dpi
    renderer = fig.canvas.get_renderer()
    lines = ((0.3, title, 12.5, "bold", INK), (0.55, subtitle, 9.0, "normal", INK_2))
    for y, text, size, weight, color in lines:
        artist = fig.text(left, 1 - y / height, text, fontsize=size, fontweight=weight, color=color)
        while artist.get_window_extent(renderer).width > max_width and artist.get_fontsize() > 7.5:
            artist.set_fontsize(artist.get_fontsize() - 0.25)


def _save(fig, path: Path | str) -> None:
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _money(value: float, decimals: int = 0, signed: bool = False) -> str:
    sign = "−" if value < 0 else ("+" if signed and value > 0 else "")
    return f"{sign}${abs(value):,.{decimals}f}"


def _param_text(param: str, value: float) -> str:
    from sweep import format_value

    return format_value(param, value)


def _param_axis(ax, param: str) -> None:
    ax.set_xlabel(X_LABELS[param])
    if param == "base_spread":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _money(v, 2)))


LABEL_BACKING = dict(boxstyle="square,pad=0.1", fc=SURFACE, ec="none")


def _dot(ax, x: float, y: float, color: str) -> None:
    ax.plot([x], [y], "o", ms=MARKER, color=color, mec=SURFACE, mew=RING, zorder=5)


def _series(sweep: SweepResult, mode: str) -> tuple[list[SweepPoint], np.ndarray]:
    points = sweep.series(mode)
    return points, np.array([p.value for p in points])


# ----------------------------------------------------------------------------- sweep charts


def plot_sharpe(sweep: SweepResult, path: Path | str) -> None:
    fig, axes = _figure(4.8)
    ax = axes[0, 0]
    best_first = sorted(sweep.modes, key=lambda m: -sweep.best(m).result.sharpe)
    for mode in sweep.modes:
        points, x = _series(sweep, mode)
        lo, hi = np.array([p.sharpe_ci for p in points]).T
        color = MODE_COLOR[mode]
        ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0)
        ax.plot(x, [p.result.sharpe for p in points], color=color, lw=LINE, label=MODE_NAME[mode])
        best = sweep.best(mode)
        _dot(ax, best.value, best.result.sharpe, color)
        above = mode == best_first[0]
        ax.annotate(
            f"{MODE_NAME[mode]} peak: {_param_text(sweep.param, best.value)}, "
            f"Sharpe {best.result.sharpe:.2f}",
            (best.value, best.result.sharpe), xytext=(9, 7) if above else (0, -13),
            textcoords="offset points", ha="left" if above else "center", va="bottom" if above else "top",
            color=INK, fontsize=9,
        )
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.margins(y=0.1)
    _param_axis(ax, sweep.param)
    ax.set_ylabel("Sharpe-like ratio (mean / std of P&L)")
    ax.legend(loc="lower right")
    overall = sweep.best()
    _headline(
        fig,
        f"Risk-adjusted return peaks at a {_param_text(sweep.param, overall.value)} "
        f"{sweep.param.replace('_', ' ')} ({MODE_NAME[overall.mode].lower()} bot)",
        f"Per-session Sharpe-like ratio, {sweep.n_trials:,} sessions per point, identical markets "
        f"for every setting. Bands: 95% bootstrap CI.",
    )
    _save(fig, path)


def plot_risk_of_ruin(sweep: SweepResult, path: Path | str) -> None:
    fig, axes = _figure(4.8)
    ax = axes[0, 0]
    overall = sweep.best()
    at_best = {}
    for mode in sweep.modes:
        points, x = _series(sweep, mode)
        lo, hi = np.array([p.result.risk_of_ruin_ci for p in points]).T
        color = MODE_COLOR[mode]
        ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0)
        ax.plot(x, [p.result.risk_of_ruin for p in points], color=color, lw=LINE, label=MODE_NAME[mode])
        point = next(p for p in points if p.value == overall.value)
        at_best[mode] = point.result.risk_of_ruin
        _dot(ax, point.value, point.result.risk_of_ruin, color)
        ax.annotate(f"{point.result.risk_of_ruin:.1%}", (point.value, point.result.risk_of_ruin),
                    xytext=(7, 5), textcoords="offset points", color=INK, fontsize=9, fontweight="bold")
    ax.axvline(overall.value, color=AXIS, lw=0.8, zorder=1)
    ax.annotate(f"best setting: {_param_text(sweep.param, overall.value)}", (overall.value, 1),
                xycoords=("data", "axes fraction"), xytext=(5, -2), textcoords="offset points",
                va="top", color=INK_2, fontsize=9)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _param_axis(ax, sweep.param)
    ax.set_ylabel("Risk of ruin (share of sessions)")
    ax.legend(loc="upper right")
    value_text = _param_text(sweep.param, overall.value)
    if len(at_best) == 2:
        title = (f"At a {value_text} {sweep.param.replace('_', ' ')}, "
                 f"{at_best['inventory-aware']:.1%} of inventory-aware sessions hit the loss limit vs "
                 f"{at_best['naive']:.1%} for the naive bot")
    else:
        title = f"Risk of ruin vs {sweep.param.replace('_', ' ')}"
    _headline(
        fig, title,
        f"Share of sessions whose P&L touched {_money(-sweep.loss_limit)} at any point, "
        f"{sweep.n_trials:,} sessions per point. Bands: 95% Wilson CI.",
    )
    _save(fig, path)


def plot_pnl_histograms(sweep: SweepResult, path: Path | str) -> None:
    modes = sweep.modes
    fig, axes = _figure(4.9, 1, len(modes), sharex=True, sharey=True)
    pairs = {m: (sweep.best(m), sweep.worst(m)) for m in modes}
    everything = np.concatenate([p.result.pnl for pair in pairs.values() for p in pair])
    lo, hi = np.quantile(everything, [0.002, 0.998])
    bins = np.arange(2 * np.floor(lo / 2), 2 * np.ceil(hi / 2) + 2, 2.0)

    for ax, mode in zip(axes.flat, modes, strict=True):
        best, worst = pairs[mode]
        for point, color, role in ((worst, REFERENCE, "Worst"), (best, MODE_COLOR[mode], "Best")):
            r = point.result
            value = _param_text(sweep.param, point.value)
            label = (f"{role}: {value}\nmean {_money(r.mean_pnl, 1, signed=True)}\n"
                     f"std ${r.std_pnl:.1f} · ruin {r.risk_of_ruin:.1%}")
            ax.hist(r.pnl, bins, histtype="stepfilled", color=color, alpha=BAND_ALPHA, lw=0)
            counts, edges, _ = ax.hist(r.pnl, bins, histtype="step", color=color, lw=LINE, label=label)
            peak = int(np.argmax(counts))
            ax.annotate(value, ((edges[peak] + edges[peak + 1]) / 2, counts[peak]), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=9,
                        bbox=LABEL_BACKING, zorder=6)
        ax.axvline(-sweep.loss_limit, color=INK_2, lw=0.8, zorder=1)
        ax.annotate("loss limit", (-sweep.loss_limit, 1), xycoords=("data", "axes fraction"),
                    xytext=(-4, -2), textcoords="offset points", ha="right", va="top", color=INK_2,
                    fontsize=8.5)
        ax.set_title(f"{MODE_NAME[mode]} bot", loc="left", fontsize=10, fontweight="bold",
                     color=INK, pad=8)
        ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.0), fontsize=8, handlelength=1.2,
                  labelspacing=1.0)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _money(v)))
        ax.set_xlabel("Session P&L")
    axes[0, 0].set_ylabel("Sessions (per $2 bin)")
    axes[0, 0].margins(y=0.12)
    _headline(
        fig,
        f"P&L distributions of the best and worst {sweep.param.replace('_', ' ')} for each bot",
        f"{sweep.n_trials:,} sessions per setting. Best = highest Sharpe-like ratio; "
        f"worst = lowest mean P&L.",
    )
    _save(fig, path)


def plot_attribution(sweep: SweepResult, path: Path | str) -> None:
    mode = sweep.best().mode
    points, x = _series(sweep, mode)
    capture = np.array([p.result.attribution["spread capture"] for p in points])
    cost = -np.array([p.result.attribution["adverse selection"] for p in points])
    net = np.array([p.result.mean_pnl for p in points])
    net_lo, net_hi = np.array([p.result.mean_pnl_ci for p in points]).T
    other = np.array([p.result.attribution["inventory P&L"] + p.result.attribution["liquidation"]
                      for p in points])

    fig, axes = _figure(4.8)
    ax = axes[0, 0]
    color = MODE_COLOR[mode]
    ax.plot(x, capture, color=CAPTURE_COLOR, lw=LINE, label="Spread captured")
    ax.plot(x, cost, color=ADVERSE_COLOR, lw=LINE, label="Adverse-selection cost")
    ax.fill_between(x, net_lo, net_hi, color=color, alpha=BAND_ALPHA, lw=0)
    ax.plot(x, net, color=color, lw=LINE, label=f"Net P&L ({MODE_NAME[mode].lower()} bot)")
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)

    i_cap, i_net = int(np.argmax(capture)), int(np.argmax(net))
    ax.annotate("spread captured", (x[i_cap], capture[i_cap]), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", color=INK, fontsize=9)
    i_cost = int(np.searchsorted(x, x[0] + 0.8 * (x[-1] - x[0])))
    ax.annotate("adverse-selection cost", (x[i_cost], cost[i_cost]), xytext=(0, -34),
                textcoords="offset points", ha="center", va="top", color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8, shrinkA=2, shrinkB=3))
    tied = sweep.tied_on_mean(mode)
    peak_text = _param_text(sweep.param, tied[0].value)
    if len(tied) > 1:
        peak_text += f"–{_param_text(sweep.param, tied[-1].value)}"
    _dot(ax, x[i_net], net[i_net], color)
    ax.annotate(f"net P&L peaks\nat {peak_text}", (x[i_net], net[i_net]), xytext=(0, 9),
                textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=9,
                linespacing=1.3)

    crossings = np.flatnonzero((net[:-1] < 0) & (net[1:] >= 0))
    title = f"Where the P&L comes from as the {sweep.param.replace('_', ' ')} changes"
    if len(crossings):
        i = crossings[0]
        breakeven = x[i] + (x[i + 1] - x[i]) * -net[i] / (net[i + 1] - net[i])
        _dot(ax, breakeven, 0.0, color)
        ax.annotate(f"break-even ≈ {_param_text(sweep.param, breakeven)}", (breakeven, 0),
                    xytext=(8, -10), textcoords="offset points", ha="left", va="top", color=INK,
                    fontsize=9)
        if sweep.param == "base_spread":
            title = (f"Below {_param_text(sweep.param, breakeven)}, adverse selection costs more than "
                     f"the spread earns; net P&L peaks at {peak_text}")

    _param_axis(ax, sweep.param)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _money(v)))
    ax.set_ylabel("Mean P&L per session")
    ax.legend(loc="upper right")
    _headline(
        fig, title,
        f"Mean P&L per session by source, {MODE_NAME[mode].lower()} bot, {sweep.n_trials:,} sessions "
        f"per point. Net also includes inventory P&L and closing costs (under ${np.abs(other).max():.2f}).",
    )
    _save(fig, path)


def plot_sweep(sweep: SweepResult, out_dir: Path | str) -> list[str]:
    """Write every sweep chart to `out_dir`; returns the file names."""
    out_dir = Path(out_dir)
    charts = {
        f"sharpe_vs_{sweep.param}.png": plot_sharpe,
        f"risk_of_ruin_vs_{sweep.param}.png": plot_risk_of_ruin,
        "pnl_hist_best_vs_worst.png": plot_pnl_histograms,
        f"pnl_attribution_vs_{sweep.param}.png": plot_attribution,
    }
    for name, plot in charts.items():
        plot(sweep, out_dir / name)
    return list(charts)


# ----------------------------------------------------------------------------- one session


def plot_session(results: Sequence[SessionResult], path: Path | str, subtitle: str | None = None) -> None:
    """Price, inventory and P&L of one market traded by one or more bots."""
    first = results[0]
    ticks = np.arange(first.paths.true_price.shape[1])
    limit = first.config.market_maker.max_inventory
    loss_limit = first.config.loss_limit

    fig, axes = _figure(7.0, 3, 1, sharex=True)
    ax_price, ax_inventory, ax_pnl = axes[:, 0]
    ax_price.plot(ticks, first.paths.true_price[0], color=INK, lw=1.0)
    ax_price.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _money(v, 2)))

    for result in results:
        mode = "inventory-aware" if result.config.market_maker.inventory_aware else "naive"
        color = MODE_COLOR[mode]
        ax_inventory.plot(ticks, result.inventory, color=color, lw=1.2, drawstyle="steps-post",
                          label=MODE_NAME[mode])
        ax_pnl.plot(ticks, result.pnl, color=color, lw=1.2, label=MODE_NAME[mode])
        for ax, series in ((ax_inventory, result.inventory), (ax_pnl, result.pnl)):
            ax.annotate(f"{MODE_NAME[mode]}", (ticks[-1], series[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", color=INK, fontsize=8.5,
                        bbox=LABEL_BACKING)

    for y in (limit, -limit):
        ax_inventory.axhline(y, color=INK_2, lw=0.8, zorder=1)
    ax_inventory.annotate(f"position limit ±{limit}", (0, limit), xytext=(2, 3),
                          textcoords="offset points", va="bottom", color=INK_2, fontsize=8.5)
    ax_pnl.axhline(-loss_limit, color=INK_2, lw=0.8, zorder=1)
    ax_pnl.annotate(f"loss limit {_money(-loss_limit)}", (0, -loss_limit), xytext=(2, -3),
                    textcoords="offset points", va="top", color=INK_2, fontsize=8.5)
    ax_pnl.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax_pnl.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _money(v)))

    for ax, title in ((ax_price, "True price"), (ax_inventory, "Inventory (units)"),
                      (ax_pnl, "Mark-to-market P&L")):
        ax.set_title(title, loc="left", fontsize=9.5, color=INK_2, pad=4)
    ax_inventory.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2, borderaxespad=0.2)
    ax_pnl.set_xlabel("Tick")
    fig.subplots_adjust(right=1 - 1.15 / fig.get_figwidth())

    spread = first.config.market_maker.base_spread
    _headline(
        fig,
        "One session, one market, two bots",
        subtitle or f"Both bots quote a {_money(spread, 2)} base spread; only the inventory-aware bot "
                    f"leans its quotes against its position.",
    )
    _save(fig, path)
