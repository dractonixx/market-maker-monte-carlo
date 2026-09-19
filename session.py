"""One trading session: price path + order flow + market maker, tick by tick.

Timeline of tick t:

  1. The bot sees the true price S[t] and its inventory q[t] and posts a bid/ask.
  2. The market moves: S[t] -> S[t+1].
  3. Counterparties, who see S[t+1], trade against the (now stale) quotes.
  4. Cash and inventory update, and the position is marked to market at S[t+1].

At the close, leftover inventory is flattened at the true price minus
`liquidation_cost` per unit. The session's P&L splits exactly into four parts
(the tests check this identity):

  spread capture     sum over fills of the quoted edge vs. S[t]
  adverse selection  sum over fills of the move S[t] -> S[t+1] against the bot
  inventory P&L      sum over ticks of q[t] * (S[t+1] - S[t])
  liquidation        -liquidation_cost * |q[T]|

The engine is vectorised across sessions: it steps through time once and
updates a whole batch of independent sessions per tick. A single session is a
batch of one, so `run_session` and the Monte Carlo share the same code path.
All randomness for a session comes from one seeded generator, drawn up front
(see `MarketInputs`), so any session can be replayed exactly from its seed.

Run `python session.py --seed 7` to simulate and summarise one session.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from market_maker import MarketMaker, MarketMakerConfig
from order_flow import OrderFlowConfig, sample_fills
from price_sim import PriceConfig, prices_from_shocks

SeedLike = int | np.random.SeedSequence | None


@dataclass(frozen=True)
class SessionConfig:
    """Everything needed to simulate a session: market, strategy and risk limits."""

    price: PriceConfig = field(default_factory=PriceConfig)
    market_maker: MarketMakerConfig = field(default_factory=MarketMakerConfig)
    order_flow: OrderFlowConfig = field(default_factory=OrderFlowConfig)
    loss_limit: float = 20.0  # $; a session whose P&L ever reaches -loss_limit counts as ruined
    liquidation_cost: float = 0.05  # $ per unit to flatten leftover inventory at the close

    def with_strategy(self, **changes) -> SessionConfig:
        """Copy of this config with some `MarketMakerConfig` fields replaced."""
        return replace(self, market_maker=replace(self.market_maker, **changes))


@dataclass(frozen=True)
class MarketInputs:
    """All the randomness for a batch of sessions, drawn before any trading.

    Drawing the market up front means the strategy cannot influence what the
    market does. It also means two strategies given the same inputs face exactly
    the same price paths and counterparty arrivals (common random numbers).
    """

    price_shocks: np.ndarray  # (n_sessions, n_steps) standard normals
    u_bid: np.ndarray  # (n_sessions, n_steps) uniforms deciding whether sellers hit the bid
    u_ask: np.ndarray  # (n_sessions, n_steps) uniforms deciding whether buyers lift the ask

    @classmethod
    def draw(cls, n_steps: int, seeds: Sequence[SeedLike]) -> MarketInputs:
        """One independent, individually reproducible session per seed."""
        n = len(seeds)
        shocks, u_bid, u_ask = np.empty((n, n_steps)), np.empty((n, n_steps)), np.empty((n, n_steps))
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            rng.standard_normal(out=shocks[i])
            rng.random(out=u_bid[i])
            rng.random(out=u_ask[i])
        return cls(shocks, u_bid, u_ask)

    @property
    def n_sessions(self) -> int:
        return self.price_shocks.shape[0]


@dataclass(frozen=True)
class SessionPaths:
    """Tick-level history of a batch, recorded only when asked for (it is large)."""

    true_price: np.ndarray  # (n, T+1)
    bid: np.ndarray  # (n, T), NaN when not quoted
    ask: np.ndarray  # (n, T), NaN when not quoted
    bid_filled: np.ndarray  # (n, T) bool: the bot bought
    ask_filled: np.ndarray  # (n, T) bool: the bot sold
    inventory: np.ndarray  # (n, T+1)
    pnl: np.ndarray  # (n, T+1), marked to market


@dataclass(frozen=True)
class BatchResult:
    """Per-session outcomes for a batch of sessions (every array has shape (n,))."""

    final_pnl: np.ndarray  # after flattening at the close
    min_pnl: np.ndarray  # lowest mark-to-market P&L at any point, including the close
    max_drawdown: np.ndarray  # largest peak-to-trough fall in P&L
    ruined: np.ndarray  # bool: P&L touched -loss_limit at some point
    time_at_limit: np.ndarray  # fraction of ticks with |inventory| >= max_inventory
    mean_abs_inventory: np.ndarray
    final_inventory: np.ndarray  # before flattening
    n_fills: np.ndarray
    n_quotes: np.ndarray  # quotes posted (two per tick, fewer at the inventory limit)
    spread_capture: np.ndarray
    adverse_selection: np.ndarray
    inventory_pnl: np.ndarray
    liquidation: np.ndarray
    paths: SessionPaths | None = None

    @property
    def fill_rate(self) -> np.ndarray:
        """Fraction of posted quotes that were traded against."""
        return self.n_fills / np.maximum(self.n_quotes, 1)

    @staticmethod
    def concat(batches: Sequence[BatchResult]) -> BatchResult:
        """Stack per-session results from several batches (paths are dropped)."""
        names = [f for f in BatchResult.__dataclass_fields__ if f != "paths"]
        return BatchResult(**{f: np.concatenate([getattr(b, f) for b in batches]) for f in names})


def simulate_batch(config: SessionConfig, inputs: MarketInputs, record_paths: bool = False) -> BatchResult:
    """Run one session per row of `inputs`, all in lockstep."""
    n, n_steps = inputs.price_shocks.shape
    prices = prices_from_shocks(config.price, inputs.price_shocks)
    bot = MarketMaker(config.market_maker)
    limit = config.market_maker.max_inventory

    cash = np.zeros(n)
    inventory = np.zeros(n, dtype=np.int64)
    peak, max_drawdown, min_pnl = np.zeros(n), np.zeros(n), np.zeros(n)
    ticks_at_limit, abs_inventory_sum = np.zeros(n, dtype=np.int64), np.zeros(n)
    n_fills, n_quotes = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    spread_capture, adverse_selection, inventory_pnl = np.zeros(n), np.zeros(n), np.zeros(n)

    if record_paths:
        rec = SessionPaths(
            true_price=prices,
            bid=np.empty((n, n_steps)),
            ask=np.empty((n, n_steps)),
            bid_filled=np.empty((n, n_steps), dtype=bool),
            ask_filled=np.empty((n, n_steps), dtype=bool),
            inventory=np.zeros((n, n_steps + 1), dtype=np.int64),
            pnl=np.zeros((n, n_steps + 1)),
        )

    for t in range(n_steps):
        price_now, price_next = prices[:, t], prices[:, t + 1]
        move = price_next - price_now

        # 1. quote on what the bot knows now; 2-3. the market moves, then counterparties trade.
        bid, ask = bot.quote(price_now, inventory)
        bought, sold = sample_fills(bid, ask, price_next, inputs.u_bid[:, t], inputs.u_ask[:, t],
                                    config.order_flow)

        # P&L attribution, using the inventory held *during* the move.
        inventory_pnl += inventory * move
        spread_capture += np.where(bought, price_now - bid, 0.0) + np.where(sold, ask - price_now, 0.0)
        adverse_selection += (bought.astype(np.int64) - sold) * move

        # 4. accounting.
        cash += np.where(sold, ask, 0.0) - np.where(bought, bid, 0.0)
        inventory += bought.astype(np.int64) - sold
        pnl = cash + inventory * price_next

        peak = np.maximum(peak, pnl)
        max_drawdown = np.maximum(max_drawdown, peak - pnl)
        min_pnl = np.minimum(min_pnl, pnl)
        abs_inventory = np.abs(inventory)
        ticks_at_limit += abs_inventory >= limit
        abs_inventory_sum += abs_inventory
        n_fills += bought.astype(np.int64) + sold
        n_quotes += np.isfinite(bid).astype(np.int64) + np.isfinite(ask)

        if record_paths:
            rec.bid[:, t], rec.ask[:, t] = bid, ask
            rec.bid_filled[:, t], rec.ask_filled[:, t] = bought, sold
            rec.inventory[:, t + 1], rec.pnl[:, t + 1] = inventory, pnl

    liquidation = -config.liquidation_cost * np.abs(inventory)
    final_pnl = cash + inventory * prices[:, -1] + liquidation
    min_pnl = np.minimum(min_pnl, final_pnl)
    max_drawdown = np.maximum(max_drawdown, peak - final_pnl)

    return BatchResult(
        final_pnl=final_pnl,
        min_pnl=min_pnl,
        max_drawdown=max_drawdown,
        ruined=min_pnl <= -config.loss_limit,
        time_at_limit=ticks_at_limit / n_steps,
        mean_abs_inventory=abs_inventory_sum / n_steps,
        final_inventory=inventory,
        n_fills=n_fills,
        n_quotes=n_quotes,
        spread_capture=spread_capture,
        adverse_selection=adverse_selection,
        inventory_pnl=inventory_pnl,
        liquidation=liquidation,
        paths=rec if record_paths else None,
    )


@dataclass(frozen=True)
class SessionResult:
    """Outcome of one session, with its full tick-by-tick history."""

    config: SessionConfig
    final_pnl: float
    max_drawdown: float
    time_at_limit: float  # fraction of ticks with |inventory| >= max_inventory
    ruined: bool
    n_fills: int
    fill_rate: float
    attribution: dict[str, float]  # the four P&L components; they sum to final_pnl
    paths: SessionPaths  # arrays of shape (1, ...) for this session

    @property
    def inventory(self) -> np.ndarray:
        return self.paths.inventory[0]

    @property
    def pnl(self) -> np.ndarray:
        return self.paths.pnl[0]

    def summary(self) -> str:
        mm = self.config.market_maker
        mode = "inventory-aware" if mm.inventory_aware else "naive"
        parts = "\n".join(f"    {name:<18} {value:+9.2f}" for name, value in self.attribution.items())
        return (
            f"{mode} market maker, base spread ${mm.base_spread:.2f}, "
            f"{self.config.price.n_steps} ticks\n"
            f"  final P&L          ${self.final_pnl:+.2f}\n"
            f"  max drawdown       ${self.max_drawdown:.2f}\n"
            f"  fills              {self.n_fills} ({self.fill_rate:.1%} of quotes)\n"
            f"  inventory          min {self.inventory.min()}, max {self.inventory.max()}, "
            f"close {self.inventory[-1]}\n"
            f"  time at limit      {self.time_at_limit:.1%} of ticks with |inventory| >= {mm.max_inventory}\n"
            f"  ruined             {'YES' if self.ruined else 'no'} "
            f"(loss limit ${self.config.loss_limit:.0f})\n"
            f"  P&L attribution\n{parts}"
        )


def run_session(config: SessionConfig | None = None, seed: SeedLike = None) -> SessionResult:
    """Simulate one full session. The same seed always gives the same session."""
    config = config or SessionConfig()
    batch = simulate_batch(config, MarketInputs.draw(config.price.n_steps, [seed]), record_paths=True)
    return SessionResult(
        config=config,
        final_pnl=float(batch.final_pnl[0]),
        max_drawdown=float(batch.max_drawdown[0]),
        time_at_limit=float(batch.time_at_limit[0]),
        ruined=bool(batch.ruined[0]),
        n_fills=int(batch.n_fills[0]),
        fill_rate=float(batch.fill_rate[0]),
        attribution={
            "spread capture": float(batch.spread_capture[0]),
            "adverse selection": float(batch.adverse_selection[0]),
            "inventory P&L": float(batch.inventory_pnl[0]),
            "liquidation": float(batch.liquidation[0]),
        },
        paths=batch.paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate one trading session.")
    parser.add_argument("--seed", type=int, default=7, help="seed for this session")
    parser.add_argument("--spread", type=float, default=MarketMakerConfig.base_spread, help="base spread ($)")
    parser.add_argument("--naive", action="store_true", help="ignore inventory when quoting")
    parser.add_argument("--mean-reversion", type=float, default=0.0)
    parser.add_argument("--replay", nargs=2, type=int, metavar=("MC_SEED", "TRIAL"),
                        help="replay trial TRIAL of a Monte Carlo run with master seed MC_SEED")
    parser.add_argument("--plot", metavar="PATH",
                        help="save a chart of this market traded by both the inventory-aware and naive bot")
    args = parser.parse_args()

    config = SessionConfig(price=PriceConfig(mean_reversion=args.mean_reversion)).with_strategy(
        base_spread=args.spread, inventory_aware=not args.naive
    )
    if args.replay:
        from monte_carlo import trial_seed

        seed = trial_seed(*args.replay)
    else:
        seed = args.seed

    result = run_session(config, seed)
    print(result.summary())
    if args.plot:
        from plots import plot_session

        # Chart both quoting modes on this session's market so their inventory can be compared.
        other = run_session(config.with_strategy(inventory_aware=args.naive), seed)
        plot_session([other, result] if args.naive else [result, other], args.plot)
        print(f"\nsaved {args.plot} (both quoting modes on this market)")


if __name__ == "__main__":
    main()
