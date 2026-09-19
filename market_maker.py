"""The market-making bot: a quoting policy.

Every tick the bot sees a reference price (the true price at the start of the
tick) and its current inventory, and posts one bid and one ask. The bot holds no
state of its own; the session does the cash and inventory accounting, so the
bot is just the map (reference price, inventory) -> (bid, ask).

Naive mode (inventory_aware=False) quotes symmetrically and ignores inventory:

    bid = m - s/2,    ask = m + s/2           (m = reference price, s = base spread)

Inventory-aware mode leans its quotes against its position:

    r    = m - gamma * q                      reservation price (skew)
    h    = s/2 + eta * |q|                    half-spread (optional widening)
    bid  = r - h,     ask = r + h

When long (q > 0), both quotes move down. The ask gets more attractive (more
likely to sell) and the bid less attractive (less likely to buy), which pushes
inventory back toward zero. `gamma` is the inventory penalty in dollars per
unit. The skew is the stationary form of the Avellaneda-Stoikov reservation
price r = m - q * gamma_AS * sigma^2 * (T - t). As a hard backstop, the bot
stops bidding at +max_inventory and stops offering at -max_inventory.

Run `python market_maker.py` to print quotes at different inventory levels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass(frozen=True)
class MarketMakerConfig:
    """Strategy parameters. All prices in dollars, inventory in units."""

    base_spread: float = 0.20  # full bid-ask spread when flat
    inventory_aware: bool = True  # False: symmetric quotes that ignore inventory
    inventory_penalty: float = 0.002  # gamma: quote shift per unit of inventory
    spread_widening: float = 0.0  # eta: extra half-spread per unit of |inventory|
    max_inventory: int = 20  # position limit; also the risk limit the session reports on

    def __post_init__(self) -> None:
        if self.base_spread < 0:
            raise ValueError("base_spread must be non-negative")
        if self.inventory_penalty < 0 or self.spread_widening < 0:
            raise ValueError("inventory_penalty and spread_widening must be non-negative")
        if self.max_inventory < 1:
            raise ValueError("max_inventory must be at least 1")


class Quotes(NamedTuple):
    bid: np.ndarray  # NaN where the bot is not bidding
    ask: np.ndarray  # NaN where the bot is not offering


class MarketMaker:
    """Quoting policy parameterised by a `MarketMakerConfig`."""

    def __init__(self, config: MarketMakerConfig) -> None:
        self.config = config

    def quote(self, reference_price: np.ndarray, inventory: np.ndarray) -> Quotes:
        """Bid and ask for each session in a batch (inputs broadcast elementwise)."""
        cfg = self.config
        reference_price = np.asarray(reference_price, dtype=float)
        inventory = np.asarray(inventory, dtype=float)
        half_spread = cfg.base_spread / 2

        if not cfg.inventory_aware:
            return Quotes(reference_price - half_spread, reference_price + half_spread)

        reservation = reference_price - cfg.inventory_penalty * inventory
        half_spread = half_spread + cfg.spread_widening * np.abs(inventory)
        bid = np.where(inventory < cfg.max_inventory, reservation - half_spread, np.nan)
        ask = np.where(inventory > -cfg.max_inventory, reservation + half_spread, np.nan)
        return Quotes(bid, ask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show how quotes respond to inventory.")
    parser.add_argument("--base-spread", type=float, default=MarketMakerConfig.base_spread)
    parser.add_argument("--inventory-penalty", type=float, default=MarketMakerConfig.inventory_penalty)
    parser.add_argument("--spread-widening", type=float, default=MarketMakerConfig.spread_widening)
    parser.add_argument("--max-inventory", type=int, default=MarketMakerConfig.max_inventory)
    args = parser.parse_args()

    aware = MarketMaker(MarketMakerConfig(
        base_spread=args.base_spread,
        inventory_penalty=args.inventory_penalty,
        spread_widening=args.spread_widening,
        max_inventory=args.max_inventory,
    ))
    naive = MarketMaker(MarketMakerConfig(base_spread=args.base_spread, inventory_aware=False))

    reference = 100.0
    limit = args.max_inventory
    print(f"Reference price ${reference:.2f}, base spread ${args.base_spread:.2f}, "
          f"gamma ${args.inventory_penalty}/unit, eta ${args.spread_widening}/unit, limit {limit}\n")
    print(f"{'inventory':>9} | {'naive bid':>9} {'naive ask':>9} | {'aware bid':>9} {'aware ask':>9}")
    for q in (-limit, -limit // 2, -5, 0, 5, limit // 2, limit):
        n, a = naive.quote(reference, q), aware.quote(reference, q)
        cells = [f"{float(x):9.3f}" if np.isfinite(x) else f"{'--':>9}" for x in (*n, *a)]
        print(f"{q:>9} | {cells[0]} {cells[1]} | {cells[2]} {cells[3]}")
    print("\n'--' = side not quoted: the inventory-aware bot stops adding to a position at its limit.")


if __name__ == "__main__":
    main()
