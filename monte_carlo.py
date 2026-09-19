"""Monte Carlo evaluation: the distribution of outcomes for a strategy.

One session is an anecdote: a good strategy can lose on a trending day and a
bad one can get lucky. `run_monte_carlo` runs thousands of independent
sessions and summarises the P&L distribution (mean, standard deviation,
Sharpe-like ratio, tail risk) and the risk of ruin: the fraction of sessions
whose P&L ever touches the loss limit.

Seeds. Trial i of a run with master seed m gets its own seed, the i-th child
of `SeedSequence(m)` (see `trial_seed`), so every trial is an independent
stream that can be replayed on its own:

    python session.py --replay <m> <i>

Common random numbers. The market a trial sees depends only on its seed, never
on the strategy. `run_monte_carlo_many` uses this to draw each chunk of
markets once and run every strategy on it, which makes comparisons between
strategies paired: they differ only in the strategy, not in luck.

Run `python monte_carlo.py --trials 5000` to evaluate the default strategy.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from market_maker import MarketMakerConfig
from session import BatchResult, MarketInputs, SessionConfig, simulate_batch

DEFAULT_SEED = 2026
DEFAULT_TRIALS = 5_000
TAIL = 0.05  # share of sessions counted as the "tail"
Z_95 = 1.959964


def trial_seed(master_seed: int, trial: int) -> np.random.SeedSequence:
    """Seed of trial `trial`: identical to `SeedSequence(master_seed).spawn(n)[trial]`."""
    return np.random.SeedSequence(master_seed, spawn_key=(trial,))


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (well behaved near 0 and 1)."""
    p = successes / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half_width = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def bootstrap_indices(n_trials: int, n_boot: int, seed: int) -> np.ndarray:
    """Resampling plan of shape (n_boot, n_trials). Reusing one plan across strategies
    keeps bootstrap comparisons between them paired."""
    return np.random.default_rng(seed).integers(0, n_trials, size=(n_boot, n_trials), dtype=np.int32)


def bootstrap_sharpe(pnl: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Sharpe-like ratio of each bootstrap resample of `pnl`."""
    sample = pnl[indices]
    return sample.mean(axis=1) / sample.std(axis=1, ddof=1)


@dataclass(frozen=True)
class MonteCarloResult:
    """Outcomes of `n_trials` independent sessions of one strategy."""

    config: SessionConfig
    trials: BatchResult
    seed: int

    @property
    def n_trials(self) -> int:
        return len(self.trials.final_pnl)

    @property
    def pnl(self) -> np.ndarray:
        return self.trials.final_pnl

    @property
    def mean_pnl(self) -> float:
        return float(self.pnl.mean())

    @property
    def std_pnl(self) -> float:
        return float(self.pnl.std(ddof=1))

    @property
    def mean_pnl_ci(self) -> tuple[float, float]:
        half_width = Z_95 * self.std_pnl / np.sqrt(self.n_trials)
        return self.mean_pnl - half_width, self.mean_pnl + half_width

    @property
    def sharpe(self) -> float:
        """Mean over standard deviation of session P&L (a per-session Sharpe ratio)."""
        return self.mean_pnl / self.std_pnl if self.std_pnl > 0 else float("nan")

    @property
    def risk_of_ruin(self) -> float:
        return float(self.trials.ruined.mean())

    @property
    def risk_of_ruin_ci(self) -> tuple[float, float]:
        return wilson_interval(int(self.trials.ruined.sum()), self.n_trials)

    @property
    def tail_pnl(self) -> float:
        """5th-percentile session P&L: 95% of sessions do at least this well (-VaR)."""
        return float(np.quantile(self.pnl, TAIL))

    @property
    def tail_mean_pnl(self) -> float:
        """Average P&L of the worst 5% of sessions (-expected shortfall)."""
        return float(self.pnl[self.pnl <= self.tail_pnl].mean())

    @property
    def fill_rate(self) -> float:
        return float(self.trials.n_fills.sum() / self.trials.n_quotes.sum())

    @property
    def mean_abs_inventory(self) -> float:
        return float(self.trials.mean_abs_inventory.mean())

    @property
    def time_at_limit(self) -> float:
        return float(self.trials.time_at_limit.mean())

    @property
    def mean_max_drawdown(self) -> float:
        return float(self.trials.max_drawdown.mean())

    @property
    def attribution(self) -> dict[str, float]:
        """Average P&L by source; the four components sum to the mean P&L."""
        t = self.trials
        return {
            "spread capture": float(t.spread_capture.mean()),
            "adverse selection": float(t.adverse_selection.mean()),
            "inventory P&L": float(t.inventory_pnl.mean()),
            "liquidation": float(t.liquidation.mean()),
        }

    def sharpe_ci(self, n_boot: int = 1_000, seed: int = 0) -> tuple[float, float]:
        """95% bootstrap confidence interval for the Sharpe-like ratio."""
        boot = bootstrap_sharpe(self.pnl, bootstrap_indices(self.n_trials, n_boot, seed))
        lo, hi = np.quantile(boot, [0.025, 0.975])
        return float(lo), float(hi)

    def summary(self) -> str:
        mm = self.config.market_maker
        mode = "inventory-aware" if mm.inventory_aware else "naive"
        lo, hi = self.mean_pnl_ci
        s_lo, s_hi = self.sharpe_ci()
        r_lo, r_hi = self.risk_of_ruin_ci
        quantiles = np.quantile(self.pnl, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        worst = int(np.argmin(self.pnl))
        parts = "\n".join(f"    {k:<18} {v:+8.2f}" for k, v in self.attribution.items())
        replay = f"python session.py --replay {self.seed} {worst} --spread {mm.base_spread}"
        replay += "" if mm.inventory_aware else " --naive"
        return (
            f"{mode} market maker, base spread ${mm.base_spread:.2f}: "
            f"{self.n_trials:,} sessions (master seed {self.seed})\n"
            f"  mean P&L            ${self.mean_pnl:+.2f}  (95% CI {lo:+.2f} to {hi:+.2f})\n"
            f"  std dev             ${self.std_pnl:.2f}\n"
            f"  Sharpe-like ratio    {self.sharpe:.2f}  (95% CI {s_lo:.2f} to {s_hi:.2f})\n"
            f"  risk of ruin         {self.risk_of_ruin:.2%}  (95% CI {r_lo:.2%} to {r_hi:.2%}; "
            f"P&L touched -${self.config.loss_limit:.0f})\n"
            f"  worst 5% of runs    P&L <= ${self.tail_pnl:+.2f}, averaging ${self.tail_mean_pnl:+.2f}\n"
            f"  mean max drawdown   ${self.mean_max_drawdown:.2f}\n"
            f"  fill rate            {self.fill_rate:.1%} of quotes\n"
            f"  mean |inventory|     {self.mean_abs_inventory:.1f} units; "
            f"{self.time_at_limit:.1%} of ticks at the {mm.max_inventory}-unit limit\n"
            f"  P&L quantiles       1%: {quantiles[0]:+.1f}  5%: {quantiles[1]:+.1f}  "
            f"25%: {quantiles[2]:+.1f}  50%: {quantiles[3]:+.1f}  75%: {quantiles[4]:+.1f}  "
            f"95%: {quantiles[5]:+.1f}  99%: {quantiles[6]:+.1f}\n"
            f"  mean P&L attribution\n{parts}\n"
            f"  worst session: trial {worst} (${self.pnl[worst]:+.2f}); replay with\n"
            f"    {replay}"
        )


def run_monte_carlo_many(
    configs: Sequence[SessionConfig],
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    chunk_size: int = 1_000,
) -> list[MonteCarloResult]:
    """Evaluate several strategies on the same `n_trials` simulated markets.

    Markets are drawn one chunk at a time, so memory use is bounded by
    `chunk_size`, not `n_trials`. The results do not depend on `chunk_size`.
    """
    n_steps = configs[0].price.n_steps
    if any(c.price.n_steps != n_steps for c in configs):
        raise ValueError("all configs must use the same session length")

    parts: list[list[BatchResult]] = [[] for _ in configs]
    for start in range(0, n_trials, chunk_size):
        stop = min(start + chunk_size, n_trials)
        inputs = MarketInputs.draw(n_steps, [trial_seed(seed, i) for i in range(start, stop)])
        for part, config in zip(parts, configs, strict=True):
            part.append(simulate_batch(config, inputs))
    return [MonteCarloResult(c, BatchResult.concat(p), seed) for c, p in zip(configs, parts, strict=True)]


def run_monte_carlo(
    config: SessionConfig | None = None,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    chunk_size: int = 1_000,
) -> MonteCarloResult:
    """Run `n_trials` independent sessions of one strategy."""
    return run_monte_carlo_many([config or SessionConfig()], n_trials, seed, chunk_size)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation of one strategy.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--spread", type=float, default=MarketMakerConfig.base_spread, help="base spread ($)")
    parser.add_argument("--naive", action="store_true", help="ignore inventory when quoting")
    args = parser.parse_args()

    config = SessionConfig().with_strategy(base_spread=args.spread, inventory_aware=not args.naive)
    print(run_monte_carlo(config, args.trials, args.seed).summary())


if __name__ == "__main__":
    main()
