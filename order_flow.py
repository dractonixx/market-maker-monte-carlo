"""Order flow: random counterparties trading against the market maker's quotes.

Each tick, buyers and sellers arrive as independent Poisson processes whose
intensity decays exponentially with how unattractive the bot's quote is
relative to the *true* price (the fill model of Avellaneda & Stoikov, 2008):

    lambda(d) = A * exp(-k * d)

  d = ask - true_price   for a buyer considering the bot's ask,
  d = true_price - bid   for a seller considering the bot's bid.

A quote is traded against during a tick with probability 1 - exp(-lambda(d)),
one unit at a time. A quote better than the true price (d < 0) is a gift, so it
is taken almost surely.

Counterparties react to the true price *after* it has moved during the tick,
but the bot set its quotes *before* the move. Whenever the move exceeds the
bot's half-spread, its stale quote is on the wrong side of the true price and
gets picked off. This is adverse selection, and it is the reason a market
maker quoting a zero spread loses money.

Run `python order_flow.py` to print the fill-probability curve.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

# Cap on the exponent in lambda(d); beyond it the fill probability is 1 to machine precision.
_MAX_EXPONENT = 50.0


@dataclass(frozen=True)
class OrderFlowConfig:
    """Parameters of the counterparty arrival model."""

    arrival_rate: float = 1.0  # A: counterparties per tick per side willing to trade at d = 0
    price_sensitivity: float = 20.0  # k (per $): intensity falls by a factor e every 1/k dollars

    def __post_init__(self) -> None:
        if self.arrival_rate < 0:
            raise ValueError("arrival_rate must be non-negative")
        if self.price_sensitivity < 0:
            raise ValueError("price_sensitivity must be non-negative")


def fill_probability(distance: np.ndarray | float, config: OrderFlowConfig) -> np.ndarray:
    """Probability that a quote is traded against within one tick.

    Args:
        distance: how much worse than the true price the quote is for the
            counterparty, in dollars (negative = better than the true price).
        config: order-flow parameters.
    """
    exponent = np.minimum(-config.price_sensitivity * np.asarray(distance, dtype=float), _MAX_EXPONENT)
    intensity = config.arrival_rate * np.exp(exponent)
    return -np.expm1(-intensity)  # 1 - exp(-intensity), accurate for small intensities


def sample_fills(
    bid: np.ndarray,
    ask: np.ndarray,
    true_price: np.ndarray,
    u_bid: np.ndarray,
    u_ask: np.ndarray,
    config: OrderFlowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Decide which quotes are traded against this tick.

    Args:
        bid, ask: the bot's quotes; NaN means that side is not quoted and cannot fill.
        true_price: the true price the counterparties see (after this tick's move).
        u_bid, u_ask: uniform(0, 1) draws. Passing randomness in, instead of
            drawing it here, lets different strategies be run on identical
            order-arrival draws (common random numbers).
        config: order-flow parameters.

    Returns:
        (bid_filled, ask_filled) boolean arrays: a seller hit the bid (the bot
        buys one unit) / a buyer lifted the ask (the bot sells one unit).
    """
    p_bid = np.nan_to_num(fill_probability(true_price - bid, config), nan=0.0)
    p_ask = np.nan_to_num(fill_probability(ask - true_price, config), nan=0.0)
    return u_bid < p_bid, u_ask < p_ask


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the fill-probability curve.")
    parser.add_argument("--arrival-rate", type=float, default=OrderFlowConfig.arrival_rate)
    parser.add_argument("--price-sensitivity", type=float, default=OrderFlowConfig.price_sensitivity)
    args = parser.parse_args()
    config = OrderFlowConfig(args.arrival_rate, args.price_sensitivity)

    print(f"A = {config.arrival_rate}/tick, k = {config.price_sensitivity}/$ "
          f"(intensity falls by e every ${1 / config.price_sensitivity:.3f})\n")
    print(" quote vs true price    P(fill this tick)")
    for d in (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        label = "better" if d < 0 else "worse " if d > 0 else "at    "
        print(f"   ${abs(d):.2f} {label}          {fill_probability(d, config):6.1%}")

    half_spread, move = 0.10, 0.05
    print(f"\nAdverse selection: an ask ${half_spread:.2f} above the price fills with probability "
          f"{fill_probability(half_spread, config):.1%}.\nIf the price jumps up ${move:.2f} before "
          f"the bot can requote, that stale ask fills with probability "
          f"{fill_probability(half_spread - move, config):.1%}, exactly when selling is a bad idea.")


if __name__ == "__main__":
    main()
