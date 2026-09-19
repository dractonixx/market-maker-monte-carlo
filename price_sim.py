"""True-price simulator: the "market" the bot is making a market in.

The true price is what a perfectly informed trader thinks the asset is worth.
It follows an arithmetic random walk

    S[t+1] = S[t] + sigma * Z[t],                     Z[t] ~ N(0, 1) i.i.d.

or, with mean reversion switched on, a discrete-time Ornstein-Uhlenbeck (AR(1))
process that is pulled back towards a long-run level mu:

    S[t+1] = S[t] + kappa * (mu - S[t]) + sigma * Z[t].

Both versions have the same one-tick volatility `sigma`. Switching mean
reversion on therefore changes the long-horizon risk of holding inventory
without changing the short-horizon risk of having a quote picked off, which
keeps the two effects separable.

Randomness is kept apart from the dynamics: `prices_from_shocks` is a
deterministic map from standard-normal shocks to price paths. The Monte Carlo
and sweep code rely on this to evaluate every strategy on identical paths.

Run `python price_sim.py` to check the simulated volatility against theory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PriceConfig:
    """Parameters of the true-price process (prices in dollars, time in ticks)."""

    s0: float = 100.0  # price at the open
    sigma: float = 0.05  # std dev of the one-tick price change ($)
    n_steps: int = 1_000  # ticks per trading session
    mean_reversion: float = 0.0  # kappa: share of the gap to mu closed per tick (0 = random walk)
    long_run_mean: float | None = None  # mu; defaults to s0

    def __post_init__(self) -> None:
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if not 0.0 <= self.mean_reversion < 1.0:
            raise ValueError("mean_reversion must be in [0, 1)")

    @property
    def mu(self) -> float:
        return self.s0 if self.long_run_mean is None else self.long_run_mean

    def theoretical_terminal_std(self) -> float:
        """Std dev of S[n_steps] when the path starts at its long-run mean."""
        if self.mean_reversion == 0.0:
            return self.sigma * np.sqrt(self.n_steps)
        phi = 1.0 - self.mean_reversion
        return self.sigma * np.sqrt((1.0 - phi ** (2 * self.n_steps)) / (1.0 - phi**2))


def prices_from_shocks(config: PriceConfig, shocks: np.ndarray) -> np.ndarray:
    """Turn standard-normal shocks into true-price paths.

    Args:
        config: price-process parameters.
        shocks: array of shape (n_paths, n_steps) of i.i.d. N(0, 1) draws.

    Returns:
        Array of shape (n_paths, n_steps + 1); column 0 is the opening price.
    """
    shocks = np.atleast_2d(shocks)
    n_paths, n_steps = shocks.shape
    if n_steps != config.n_steps:
        raise ValueError(f"expected {config.n_steps} shocks per path, got {n_steps}")

    moves = config.sigma * shocks
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = config.s0

    if config.mean_reversion == 0.0:
        paths[:, 1:] = config.s0 + np.cumsum(moves, axis=1)
        return paths

    kappa, mu = config.mean_reversion, config.mu
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] + kappa * (mu - paths[:, t]) + moves[:, t]
    return paths


def simulate_prices(
    config: PriceConfig, n_paths: int = 1, seed: int | None = None
) -> np.ndarray:
    """Simulate `n_paths` independent price paths, shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    return prices_from_shocks(config, rng.standard_normal((n_paths, config.n_steps)))


def _describe(config: PriceConfig, n_paths: int, seed: int) -> str:
    paths = simulate_prices(config, n_paths, seed)
    moves = np.diff(paths, axis=1)
    lag1 = np.corrcoef(moves[:, 1:].ravel(), moves[:, :-1].ravel())[0, 1]
    return (
        f"  one-tick std      {moves.std():.4f}   (sigma = {config.sigma:.4f})\n"
        f"  terminal std      {paths[:, -1].std():.4f}   (theory = {config.theoretical_terminal_std():.4f})\n"
        f"  lag-1 autocorr    {lag1:+.4f}   (random walk = 0; mean reversion < 0)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check the true-price simulator.")
    parser.add_argument("--paths", type=int, default=20_000)
    parser.add_argument("--sigma", type=float, default=PriceConfig.sigma)
    parser.add_argument("--mean-reversion", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random_walk = PriceConfig(sigma=args.sigma)
    reverting = PriceConfig(sigma=args.sigma, mean_reversion=args.mean_reversion)
    print(f"{args.paths:,} paths of {random_walk.n_steps:,} ticks each\n")
    print("Random walk:")
    print(_describe(random_walk, args.paths, args.seed))
    print(f"\nMean-reverting (kappa = {args.mean_reversion}):")
    print(_describe(reverting, args.paths, args.seed))


if __name__ == "__main__":
    main()
