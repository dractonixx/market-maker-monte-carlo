"""Parameter sweep: which strategy setting is actually best, and why.

For every value of one strategy parameter (the base spread by default) and
every quoting mode, run a full Monte Carlo batch and compare the outcome
distributions. Two design choices make the comparison sharp:

* Common random numbers: every setting is run on the *same* simulated markets
  (same price paths, same counterparty arrivals), so differences between
  settings come from the strategy and not from luck.
* Paired bootstrap: Sharpe-ratio confidence intervals use one shared
  resampling plan for all settings. Two settings are "statistically tied" if
  the 95% interval for the difference of their Sharpe ratios contains zero.

"Best" means the highest Sharpe-like ratio (mean / std of session P&L).
"Worst" means the lowest mean P&L, because Sharpe ratios do not rank losing
strategies sensibly (for a strategy that loses on average, more noise gives a
less negative Sharpe).

Outputs, written to --out (default results/):
    sweep_results.csv               every metric with 95% CIs, one row per setting
    sweep_summary.txt               the table and findings printed below
    sharpe_vs_<param>.png           Sharpe-like ratio vs the parameter
    risk_of_ruin_vs_<param>.png     risk of ruin vs the parameter
    pnl_hist_best_vs_worst.png      P&L distributions of the best and worst settings
    pnl_attribution_vs_<param>.png  spread capture vs adverse selection
    example_session.png             one session, both quoting modes, same market

Usage:
    python sweep.py --trials 5000
    python sweep.py --param inventory_penalty --values 0 0.001 0.002 0.004 0.008
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from monte_carlo import (
    DEFAULT_SEED,
    DEFAULT_TRIALS,
    Z_95,
    MonteCarloResult,
    bootstrap_indices,
    bootstrap_sharpe,
    run_monte_carlo_many,
)
from session import SessionConfig, run_session

MODES = {"inventory-aware": True, "naive": False}

DEFAULT_VALUES = {
    "base_spread": [round(0.02 * i, 2) for i in range(26)],  # $0.00 to $0.50
    "inventory_penalty": [0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016],
    "spread_widening": [0.0, 0.001, 0.002, 0.004, 0.008],
    "max_inventory": [5, 10, 15, 20, 30, 50],
}

PARAM_LABELS = {
    "base_spread": "base spread",
    "inventory_penalty": "inventory penalty",
    "spread_widening": "spread widening",
    "max_inventory": "max inventory",
}

EXAMPLE_SESSION_SEED = 7


def format_value(param: str, value: float) -> str:
    if param == "max_inventory":
        return f"{int(value)}"
    if param == "base_spread":
        return f"${value:.2f}"
    return f"${value:g}"


@dataclass(frozen=True)
class SweepPoint:
    """Monte Carlo outcome of one (mode, parameter value) setting."""

    mode: str
    value: float
    result: MonteCarloResult
    boot_sharpe: np.ndarray  # Sharpe of each resample in the sweep's shared bootstrap plan

    @property
    def sharpe_ci(self) -> tuple[float, float]:
        lo, hi = np.quantile(self.boot_sharpe, [0.025, 0.975])
        return float(lo), float(hi)


def paired_mean_gap_ci(a: SweepPoint, b: SweepPoint) -> tuple[float, float]:
    """95% CI for (mean P&L of a) - (mean P&L of b). Pairing session by session is valid
    because both settings ran on the same simulated markets."""
    gap = a.result.pnl - b.result.pnl
    half_width = Z_95 * gap.std(ddof=1) / np.sqrt(len(gap))
    return float(gap.mean() - half_width), float(gap.mean() + half_width)


@dataclass(frozen=True)
class SweepResult:
    param: str
    modes: tuple[str, ...]
    points: tuple[SweepPoint, ...]
    n_trials: int
    seed: int

    @property
    def loss_limit(self) -> float:
        return self.points[0].result.config.loss_limit

    def series(self, mode: str) -> list[SweepPoint]:
        return sorted((p for p in self.points if p.mode == mode), key=lambda p: p.value)

    def best(self, mode: str | None = None) -> SweepPoint:
        """Highest Sharpe-like ratio (within `mode`, or overall)."""
        candidates = self.series(mode) if mode else self.points
        return max(candidates, key=lambda p: p.result.sharpe)

    def worst(self, mode: str) -> SweepPoint:
        """Lowest mean P&L within `mode`."""
        return min(self.series(mode), key=lambda p: p.result.mean_pnl)

    def top_mean(self, mode: str) -> SweepPoint:
        """Highest mean P&L within `mode`."""
        return max(self.series(mode), key=lambda p: p.result.mean_pnl)

    def tied_with_best(self, mode: str) -> list[SweepPoint]:
        """Settings whose Sharpe is not significantly below the best (paired bootstrap, 95%)."""
        best = self.best(mode)
        return [p for p in self.series(mode) if np.quantile(best.boot_sharpe - p.boot_sharpe, 0.025) <= 0]

    def tied_on_mean(self, mode: str) -> list[SweepPoint]:
        """Settings whose mean P&L is not significantly below the top mean (paired, 95%)."""
        top = self.top_mean(mode)
        return [p for p in self.series(mode) if paired_mean_gap_ci(top, p)[0] <= 0]

    def near_best(self, mode: str, tolerance: float = 0.05) -> list[SweepPoint]:
        """Settings within `tolerance` (relative) of the best Sharpe: the practically-equivalent zone."""
        target = (1 - tolerance) * self.best(mode).result.sharpe
        return [p for p in self.series(mode) if p.result.sharpe >= target]


def run_sweep(
    param: str = "base_spread",
    values: list[float] | None = None,
    modes: tuple[str, ...] = tuple(MODES),
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    base: SessionConfig | None = None,
    n_boot: int = 1_000,
) -> SweepResult:
    """Monte Carlo every (mode, value) setting on one shared set of simulated markets."""
    base = base or SessionConfig()
    values = DEFAULT_VALUES[param] if values is None else values
    cast = type(getattr(base.market_maker, param))
    grid = [(mode, value) for mode in modes for value in values]
    configs = [base.with_strategy(**{param: cast(v), "inventory_aware": MODES[m]}) for m, v in grid]

    results = run_monte_carlo_many(configs, n_trials, seed)
    plan = bootstrap_indices(n_trials, n_boot, seed)
    points = tuple(
        SweepPoint(mode, float(value), result, bootstrap_sharpe(result.pnl, plan))
        for (mode, value), result in zip(grid, results, strict=True)
    )
    return SweepResult(param, tuple(modes), points, n_trials, seed)


# ----------------------------------------------------------------------------- reporting


def format_table(sweep: SweepResult) -> str:
    best_overall = sweep.best()
    header = (
        f"{PARAM_LABELS[sweep.param]:>17}  {'mean P&L [95% CI]':>24}  {'std':>6}  "
        f"{'Sharpe [95% CI]':>21}  {'ruin [95% CI]':>21}  {'fills':>6}  {'|inv|':>5}  "
        f"{'@limit':>6}  {'capture':>7}  {'adv.sel':>7}"
    )
    lines = []
    for mode in sweep.modes:
        tied = {p.value for p in sweep.tied_with_best(mode)}
        best = sweep.best(mode)
        lines += ["", f"{mode} quoting", header, "-" * len(header)]
        for p in sweep.series(mode):
            r = p.result
            m_lo, m_hi = r.mean_pnl_ci
            s_lo, s_hi = p.sharpe_ci
            r_lo, r_hi = r.risk_of_ruin_ci
            attribution = r.attribution
            tag = "  <- best" if p is best else "  ~ tied" if p.value in tied else ""
            tag += " overall" if p is best_overall else ""
            lines.append(
                f"{format_value(sweep.param, p.value):>17}  "
                f"{r.mean_pnl:+8.2f} [{m_lo:+7.2f},{m_hi:+7.2f}]  {r.std_pnl:6.2f}  "
                f"{r.sharpe:+6.2f} [{s_lo:+6.2f},{s_hi:+6.2f}]  "
                f"{r.risk_of_ruin:6.1%} [{r_lo:5.1%},{r_hi:6.1%}]  "
                f"{r.fill_rate:6.1%}  {r.mean_abs_inventory:5.1f}  {r.time_at_limit:6.1%}  "
                f"{attribution['spread capture']:+7.2f}  {attribution['adverse selection']:+7.2f}{tag}"
            )
    return "\n".join(lines)


def findings(sweep: SweepResult) -> list[str]:
    """Plain-language conclusions, each backed by numbers from the sweep."""
    label = PARAM_LABELS[sweep.param]

    def fmt(value: float) -> str:
        return format_value(sweep.param, value)

    def span(points: list[SweepPoint]) -> str:
        lo, hi = points[0].value, points[-1].value
        return fmt(lo) if lo == hi else f"{fmt(lo)} to {fmt(hi)}"

    def ruin_text(r: MonteCarloResult) -> str:
        lo, hi = r.risk_of_ruin_ci
        if r.trials.ruined.sum() == 0:
            return f"0% (0 of {r.n_trials:,} sessions; 95% CI up to {hi:.2%})"
        return f"{r.risk_of_ruin:.2%} (95% CI {lo:.2%} to {hi:.2%})"

    out = []
    overall = sweep.best()
    r = overall.result
    s_lo, s_hi = overall.sharpe_ci
    tied = sweep.tied_with_best(overall.mode)
    text = (
        f"BEST SETTING: {overall.mode} quoting with {label} {fmt(overall.value)}: Sharpe-like ratio "
        f"{r.sharpe:.2f} (95% CI {s_lo:.2f} to {s_hi:.2f}), mean P&L ${r.mean_pnl:+.2f} per session "
        f"(95% CI {r.mean_pnl_ci[0]:+.2f} to {r.mean_pnl_ci[1]:+.2f}), risk of ruin {ruin_text(r)}. "
    )
    if len(tied) == 1:
        text += (
            f"Because every setting ran on the same markets, the comparison is sharp: every other "
            f"{label} has a significantly lower Sharpe (paired bootstrap, 95%). "
        )
    else:
        text += (
            f"{label.capitalize()}s from {span(tied)} are statistically tied with it "
            f"(paired bootstrap, 95%). "
        )
    if r.sharpe > 0:
        text += (
            f"The peak is flat in practical terms: every {label} from {span(sweep.near_best(overall.mode))} "
            f"is within 5% of the best Sharpe."
        )
    out.append(text)

    for mode in sweep.modes:
        best, top = sweep.best(mode), sweep.top_mean(mode)
        mean_tied = sweep.tied_on_mean(mode)
        br, tr = best.result, top.result
        if best in mean_tied and len(mean_tied) == 1:
            text = (
                f"{mode.upper()}: mean P&L and Sharpe both peak at {fmt(best.value)} "
                f"(${br.mean_pnl:+.2f}, Sharpe {br.sharpe:.2f}). "
            )
        elif best in mean_tied:
            text = (
                f"{mode.upper()}: mean P&L and Sharpe peak together. Mean P&L is statistically flat "
                f"from {span(mean_tied)} (${tr.mean_pnl:+.2f}) and Sharpe peaks at {fmt(best.value)} "
                f"({br.sharpe:.2f}). "
            )
        else:
            text = (
                f"{mode.upper()}: mean P&L peaks at {fmt(top.value)} (${tr.mean_pnl:+.2f}) but Sharpe "
                f"peaks at {fmt(best.value)} ({br.sharpe:.2f}; "
                f"tied from {span(sweep.tied_with_best(mode))}). "
                f"Going from {fmt(top.value)} to {fmt(best.value)} gives up "
                f"{1 - br.mean_pnl / tr.mean_pnl:.0%} of mean P&L but cuts the P&L std from "
                f"${tr.std_pnl:.2f} to ${br.std_pnl:.2f} and risk of ruin from "
                f"{tr.risk_of_ruin:.1%} to {br.risk_of_ruin:.1%}. "
            )
        losing = [p for p in sweep.series(mode) if p.result.mean_pnl_ci[1] < 0]
        if losing:
            worst = sweep.worst(mode)
            a = worst.result.attribution
            text += (
                f"Every {label} up to {fmt(losing[-1].value)} loses money (95% CI below zero); at "
                f"{fmt(worst.value)} spread capture is ${a['spread capture']:+.2f} per session while "
                f"adverse selection costs ${a['adverse selection']:+.2f}. "
            )
        a = br.attribution
        if a["spread capture"] > 0:
            text += (
                f"At {fmt(best.value)}, adverse selection gives back "
                f"{-a['adverse selection'] / a['spread capture']:.0%} of the "
                f"${a['spread capture']:.2f} of spread captured."
            )
        out.append(text)

    if len(sweep.modes) == 2:
        first, second = sweep.modes
        pa = next(p for p in sweep.series(first) if p.value == overall.value)
        pb = next(p for p in sweep.series(second) if p.value == overall.value)
        a, b = pa.result, pb.result
        gap_lo, gap_hi = paired_mean_gap_ci(pa, pb)
        out.append(
            f"SAME {label.upper()} ({fmt(overall.value)}), {first} vs {second}: mean P&L "
            f"${a.mean_pnl:+.2f} vs ${b.mean_pnl:+.2f} (paired difference 95% CI {gap_lo:+.2f} to "
            f"{gap_hi:+.2f}), std ${a.std_pnl:.2f} vs ${b.std_pnl:.2f} ({a.std_pnl / b.std_pnl - 1:+.0%}), "
            f"Sharpe {a.sharpe:.2f} vs {b.sharpe:.2f}, risk of ruin {a.risk_of_ruin:.2%} vs "
            f"{b.risk_of_ruin:.2%}, time at the inventory limit {a.time_at_limit:.1%} vs "
            f"{b.time_at_limit:.1%}."
        )
    return out


def write_csv(sweep: SweepResult, path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mode", sweep.param, "mean_pnl", "mean_pnl_ci_lo", "mean_pnl_ci_hi", "std_pnl",
            "sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "risk_of_ruin", "risk_of_ruin_ci_lo",
            "risk_of_ruin_ci_hi", "pnl_5th_percentile", "pnl_worst_5pct_mean", "mean_max_drawdown",
            "fill_rate", "mean_abs_inventory", "time_at_limit", "spread_capture",
            "adverse_selection", "inventory_pnl", "liquidation",
        ])
        for p in sorted(sweep.points, key=lambda p: (sweep.modes.index(p.mode), p.value)):
            r = p.result
            writer.writerow([
                p.mode, p.value, *np.round([
                    r.mean_pnl, *r.mean_pnl_ci, r.std_pnl, r.sharpe, *p.sharpe_ci, r.risk_of_ruin,
                    *r.risk_of_ruin_ci, r.tail_pnl, r.tail_mean_pnl, r.mean_max_drawdown,
                    r.fill_rate, r.mean_abs_inventory, r.time_at_limit, *r.attribution.values(),
                ], 5),
            ])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep one strategy parameter with a full Monte Carlo batch at each value."
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="sessions per setting")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed")
    parser.add_argument("--param", choices=sorted(DEFAULT_VALUES), default="base_spread")
    parser.add_argument("--values", type=float, nargs="+", help="values to sweep (default: a preset grid)")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--out", type=Path, default=Path("results"), help="output directory")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    values = args.values or DEFAULT_VALUES[args.param]
    n_settings = len(values) * len(args.modes)
    print(f"Sweeping {args.param} over {len(values)} values x {len(args.modes)} modes: "
          f"{n_settings} settings x {args.trials:,} sessions (seed {args.seed})...")
    start = time.perf_counter()
    sweep = run_sweep(args.param, values, tuple(args.modes), args.trials, args.seed)
    elapsed = time.perf_counter() - start

    config = sweep.points[0].result.config
    report = "\n".join([
        f"Sweep of {args.param}: {args.trials:,} sessions per setting, master seed {args.seed}, "
        f"{config.price.n_steps} ticks per session.",
        f"Every setting sees the same {args.trials:,} simulated markets (common random numbers).",
        f"Risk of ruin = share of sessions whose P&L touched -${sweep.loss_limit:.0f}. "
        f"@limit = share of ticks with |inventory| >= {config.market_maker.max_inventory}.",
        "capture / adv.sel = mean spread captured / mean adverse-selection P&L per session ($).",
        format_table(sweep),
        "",
        "FINDINGS",
        *[f"- {line}" for line in findings(sweep)],
    ])
    print(report)
    print(f"\n({n_settings * args.trials:,} sessions simulated in {elapsed:.1f}s)")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sweep_summary.txt").write_text(report + "\n")
    write_csv(sweep, args.out / "sweep_results.csv")
    written = ["sweep_summary.txt", "sweep_results.csv"]

    if not args.no_plots:
        import plots

        written += plots.plot_sweep(sweep, args.out)
        best = sweep.best()
        example = [
            run_session(best.result.config.with_strategy(inventory_aware=MODES[mode]), EXAMPLE_SESSION_SEED)
            for mode in args.modes
        ]
        plots.plot_session(example, args.out / "example_session.png")
        written.append("example_session.png")

    print(f"\nWrote to {args.out}/: " + ", ".join(written))


if __name__ == "__main__":
    main()
