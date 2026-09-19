"""Sanity tests: the simulator behaves the way market-making economics says it must.

Sessions are shortened to 250 ticks to keep the suite fast; the effects being
tested are large enough to be unmistakable at that length.
"""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from market_maker import MarketMaker, MarketMakerConfig
from monte_carlo import run_monte_carlo, run_monte_carlo_many, trial_seed
from order_flow import OrderFlowConfig, fill_probability
from price_sim import PriceConfig, simulate_prices
from session import SessionConfig, run_session
from sweep import run_sweep

FAST = SessionConfig(price=PriceConfig(n_steps=250))


def naive(spread: float, config: SessionConfig = FAST) -> SessionConfig:
    return config.with_strategy(base_spread=spread, inventory_aware=False)


# ----------------------------------------------------------------------------- economics


def test_zero_spread_loses_money_on_average():
    """Quoting at the true price earns nothing on each trade but still gets picked off.

    The closing cost is switched off so that adverse selection is the only way to lose.
    """
    result = run_monte_carlo(naive(0.0, replace(FAST, liquidation_cost=0.0)), n_trials=400, seed=1)
    assert result.mean_pnl_ci[1] < 0
    adverse = result.trials.adverse_selection
    assert adverse.mean() + 3 * adverse.std(ddof=1) / np.sqrt(len(adverse)) < 0


def test_wider_spreads_reduce_fill_rate():
    spreads = [0.0, 0.05, 0.1, 0.2, 0.4]
    fill_rates = [r.fill_rate for r in run_monte_carlo_many([naive(s) for s in spreads], 200, seed=2)]
    assert all(tighter > wider for tighter, wider in pairwise(fill_rates))


def test_a_sensible_spread_makes_money():
    assert run_monte_carlo(FAST, n_trials=400, seed=3).mean_pnl_ci[0] > 0


def test_without_volatility_every_fill_earns_exactly_half_the_spread():
    """No price moves means no adverse selection and no inventory risk: P&L is pure spread capture."""
    still = SessionConfig(price=PriceConfig(sigma=0.0, n_steps=250), liquidation_cost=0.0)
    session = run_session(naive(0.2, still), seed=4)
    assert session.n_fills > 0
    assert session.final_pnl == pytest.approx(0.1 * session.n_fills)


def test_pnl_attribution_adds_up_to_final_pnl():
    session = run_session(naive(0.1), seed=5)
    assert sum(session.attribution.values()) == pytest.approx(session.final_pnl, abs=1e-9)


# ----------------------------------------------------------------------------- inventory control


def test_quotes_skew_against_inventory():
    bot = MarketMaker(MarketMakerConfig(base_spread=0.2, inventory_penalty=0.01))
    flat, long = bot.quote(100.0, 0), bot.quote(100.0, 10)
    assert long.bid < flat.bid and long.ask < flat.ask  # long: lean towards selling
    assert long.ask - long.bid == pytest.approx(flat.ask - flat.bid)  # a shift, not a wider spread


def test_inventory_aware_bot_never_exceeds_its_position_limit():
    config = FAST.with_strategy(inventory_penalty=0.0, max_inventory=5)  # only the hard limit acts
    peak = max(np.abs(run_session(config, seed).inventory).max() for seed in range(20))
    assert peak == 5  # reached, never exceeded


def test_inventory_skew_cuts_risk_at_the_same_spread():
    aware, naive_bot = run_monte_carlo_many([FAST, naive(FAST.market_maker.base_spread)], 400, seed=6)
    assert aware.std_pnl < 0.7 * naive_bot.std_pnl
    assert aware.mean_abs_inventory < naive_bot.mean_abs_inventory


# ----------------------------------------------------------------------------- reproducibility


def test_every_monte_carlo_trial_can_be_replayed_on_its_own():
    config = naive(0.2)
    batch = run_monte_carlo(config, n_trials=50, seed=7, chunk_size=16)
    for trial in (0, 15, 16, 49):  # includes both sides of a chunk boundary
        replay = run_session(config, trial_seed(7, trial))
        assert replay.final_pnl == pytest.approx(batch.pnl[trial], abs=1e-9)


def test_results_do_not_depend_on_chunk_size():
    a = run_monte_carlo(FAST, n_trials=60, seed=8, chunk_size=7).pnl
    b = run_monte_carlo(FAST, n_trials=60, seed=8, chunk_size=60).pnl
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)


# ----------------------------------------------------------------------------- building blocks


def test_random_walk_volatility_matches_theory():
    config = PriceConfig(n_steps=400)
    terminal = simulate_prices(config, n_paths=20_000, seed=9)[:, -1]
    assert terminal.std() == pytest.approx(config.theoretical_terminal_std(), rel=0.03)


def test_mean_reversion_shrinks_long_horizon_risk_but_not_one_tick_risk():
    walk = simulate_prices(PriceConfig(n_steps=400), n_paths=5_000, seed=10)
    reverting = simulate_prices(PriceConfig(n_steps=400, mean_reversion=0.02), n_paths=5_000, seed=10)
    assert np.diff(reverting).std() == pytest.approx(np.diff(walk).std(), rel=0.02)
    assert reverting[:, -1].std() < 0.5 * walk[:, -1].std()


def test_fill_probability_falls_with_distance_and_saturates_for_gifts():
    p = fill_probability(np.array([-0.5, -0.1, 0.0, 0.1, 0.3]), OrderFlowConfig())
    assert np.all(np.diff(p) < 0)
    assert p[0] == pytest.approx(1.0)


def test_sweep_picks_a_sensible_spread_and_plots(tmp_path):
    import plots

    sweep = run_sweep("base_spread", [0.0, 0.2, 0.6], n_trials=200, seed=11, base=FAST, n_boot=200)
    assert sweep.best().value == 0.2
    assert all(sweep.worst(mode).value == 0.0 for mode in sweep.modes)
    written = plots.plot_sweep(sweep, tmp_path)
    assert all((tmp_path / name).stat().st_size > 0 for name in written)
