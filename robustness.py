"""Robustness: is the best spread an artifact of the default parameters?

The headline sweep fixes the inventory penalty (gamma) and the market (sigma, k).
This script re-runs the full spread sweep while varying each of them:

1. Inventory penalty, gamma from $0.0005 to $0.02 per unit. If the best spread
   moved with gamma, the headline result would only reflect the choice of gamma.
2. Market conditions: volatility sigma and the flow's price sensitivity k.
   First-order theory (README, "Why the optimum is where it is") predicts that
   the spread maximising mean P&L is 2 * (1/k + k * sigma^2). This checks it.

Usage:
    python robustness.py --trials 5000     # about two minutes; writes results/robustness.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from monte_carlo import DEFAULT_SEED, DEFAULT_TRIALS
from order_flow import OrderFlowConfig
from price_sim import PriceConfig
from session import SessionConfig
from sweep import run_sweep

SPREADS = [round(0.02 * i, 2) for i in range(31)]  # $0.00 to $0.60
PENALTIES = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
MARKETS = [(0.03, 20), (0.05, 20), (0.07, 20), (0.05, 10), (0.05, 30), (0.08, 12)]  # (sigma, k)


def predicted_optimal_spread(sigma: float, k: float) -> float:
    """Full spread maximising mean P&L in the small-fill-probability approximation."""
    return 2 * (1 / k + k * sigma**2)


def optimal_spreads(base: SessionConfig, n_trials: int, seed: int) -> dict[str, float]:
    sweep = run_sweep("base_spread", SPREADS, ("inventory-aware",), n_trials, seed, base, n_boot=500)
    best, tied = sweep.best("inventory-aware"), sweep.tied_on_mean("inventory-aware")
    return {
        "sharpe_optimal_spread": best.value,
        "sharpe": best.result.sharpe,
        "mean_pnl": best.result.mean_pnl,
        "std_pnl": best.result.std_pnl,
        "mean_optimal_spread": sweep.top_mean("inventory-aware").value,
        "mean_optimal_tied_lo": tied[0].value,
        "mean_optimal_tied_hi": tied[-1].value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the best spread against gamma, sigma and k.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    start = time.perf_counter()
    rows = []

    sigma, k = PriceConfig.sigma, OrderFlowConfig.price_sensitivity
    print(f"1. Inventory penalty (market: sigma ${sigma}/tick, k {k:g}/$), "
          f"{args.trials:,} sessions per setting")
    print(f"   {'gamma $/unit':>12}  {'Sharpe-optimal spread':>21}  {'Sharpe':>6}  "
          f"{'mean P&L':>8}  {'std':>6}")
    for gamma in PENALTIES:
        r = optimal_spreads(SessionConfig().with_strategy(inventory_penalty=gamma), args.trials, args.seed)
        rows.append({"experiment": "inventory_penalty", "sigma": sigma, "k": k, "inventory_penalty": gamma,
                     "predicted_mean_optimal_spread": predicted_optimal_spread(sigma, k), **r})
        best = f"${r['sharpe_optimal_spread']:.2f}"
        print(f"   {gamma:>12g}  {best:>21}  {r['sharpe']:6.2f}  "
              f"{r['mean_pnl']:+8.2f}  {r['std_pnl']:6.2f}")

    gamma = SessionConfig().market_maker.inventory_penalty
    print(f"\n2. Market conditions (gamma ${gamma}/unit)")
    print(f"   {'sigma':>5}  {'k':>3}  {'k*sigma':>7}  {'predicted':>9}  {'mean-optimal (tied)':>19}  "
          f"{'Sharpe-optimal':>14}")
    for sigma, k in MARKETS:
        base = SessionConfig(price=PriceConfig(sigma=sigma), order_flow=OrderFlowConfig(price_sensitivity=k))
        r = optimal_spreads(base, args.trials, args.seed)
        predicted = predicted_optimal_spread(sigma, k)
        rows.append({"experiment": "market", "sigma": sigma, "k": k, "inventory_penalty": gamma,
                     "predicted_mean_optimal_spread": predicted, **r})
        tied = f"${r['mean_optimal_tied_lo']:.2f}"
        if r["mean_optimal_tied_hi"] != r["mean_optimal_tied_lo"]:
            tied += f"-${r['mean_optimal_tied_hi']:.2f}"
        predicted_text, best = f"${predicted:.3f}", f"${r['sharpe_optimal_spread']:.2f}"
        print(f"   {sigma:5.2f}  {k:3g}  {k * sigma:7.2f}  {predicted_text:>9}  {tied:>19}  {best:>14}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "robustness.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: round(v, 5) if isinstance(v, float) else v for key, v in row.items()}
                         for row in rows)
    print(f"\nWrote {args.out}/robustness.csv ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
