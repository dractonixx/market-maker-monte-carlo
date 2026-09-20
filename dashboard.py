"""Local dashboard for the market-making simulator (Streamlit).

Nothing is re-implemented here: every number comes from the same engine as the
command line (`session.py`, `monte_carlo.py`, `sweep.py`) and every chart is
drawn by `plots.py`. Left at the default settings and seed, the dashboard
reproduces the figures in results/ exactly.

    pip install -r requirements-dashboard.txt
    streamlit run dashboard.py
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

import streamlit as st

import plots
from market_maker import MarketMakerConfig
from monte_carlo import DEFAULT_SEED, run_monte_carlo_many
from order_flow import OrderFlowConfig
from price_sim import PriceConfig
from robustness import predicted_optimal_spread
from session import SessionConfig, run_session
from sweep import MODES, findings, format_value, run_sweep

st.set_page_config(page_title="Market-making simulator", layout="wide")

DEFAULTS = SessionConfig()


# ----------------------------------------------------------------------------- plumbing


def build_config(**params) -> SessionConfig:
    """The same `SessionConfig` the command line builds, from the sidebar settings."""
    return SessionConfig(
        price=PriceConfig(
            sigma=params["sigma"],
            mean_reversion=params["mean_reversion"],
            n_steps=params["n_steps"],
        ),
        order_flow=OrderFlowConfig(
            arrival_rate=params["arrival_rate"],
            price_sensitivity=params["price_sensitivity"],
        ),
        market_maker=MarketMakerConfig(
            base_spread=params["base_spread"],
            inventory_penalty=params["inventory_penalty"],
            spread_widening=params["spread_widening"],
            max_inventory=params["max_inventory"],
        ),
        loss_limit=params["loss_limit"],
    )


def money_safe(text: str) -> str:
    """Escape dollar signs so Streamlit's markdown does not read them as LaTeX math."""
    return text.replace("$", r"\$")


def render(draw: Callable[[Path], None]) -> bytes:
    """Run one of the `plots` functions into a temporary file and return the PNG."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = Path(handle.name)
    try:
        draw(path)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


@st.cache_data(show_spinner=False)
def one_session(seed: int, **params) -> tuple[bytes, list[dict]]:
    config = build_config(**params)
    results = [run_session(config.with_strategy(inventory_aware=aware), seed) for aware in (True, False)]
    rows = [
        {
            "Bot": "Inventory-aware" if result.config.market_maker.inventory_aware else "Naive",
            "Final P&L": result.final_pnl,
            "Max drawdown": result.max_drawdown,
            "Fills": result.n_fills,
            "Peak |inventory|": int(abs(result.inventory).max()),
            "Time at limit": result.time_at_limit,
            "Hit loss limit": "yes" if result.ruined else "no",
            **{name.capitalize(): value for name, value in result.attribution.items()},
        }
        for result in results
    ]
    return render(lambda path: plots.plot_session(results, path)), rows


@st.cache_data(show_spinner=False)
def monte_carlo(trials: int, seed: int, **params) -> tuple[bytes, list[dict], dict]:
    config = build_config(**params)
    results = run_monte_carlo_many(
        [config.with_strategy(inventory_aware=aware) for aware in MODES.values()], trials, seed
    )
    by_mode = dict(zip(MODES, results, strict=True))
    rows = [
        {
            "Bot": mode,
            "Mean P&L": result.mean_pnl,
            "95% CI": f"{result.mean_pnl_ci[0]:+.2f} to {result.mean_pnl_ci[1]:+.2f}",
            "Std dev": result.std_pnl,
            "Sharpe-like": result.sharpe,
            "Risk of ruin": result.risk_of_ruin,
            "Worst 5% average": result.tail_mean_pnl,
            "Mean max drawdown": result.mean_max_drawdown,
            "Fill rate": result.fill_rate,
            "Mean |inventory|": result.mean_abs_inventory,
            "Time at limit": result.time_at_limit,
        }
        for mode, result in by_mode.items()
    ]
    headline = {
        mode: {
            "mean_pnl": result.mean_pnl,
            "sharpe": result.sharpe,
            "sharpe_ci": result.sharpe_ci(),
            "risk_of_ruin": result.risk_of_ruin,
            "risk_of_ruin_ci": result.risk_of_ruin_ci,
            "attribution": result.attribution,
        }
        for mode, result in by_mode.items()
    }
    return render(lambda path: plots.plot_pnl_distribution(by_mode, path)), rows, headline


@st.cache_data(show_spinner=False)
def spread_sweep(trials: int, seed: int, spreads: tuple[float, ...], **params):
    sweep = run_sweep("base_spread", list(spreads), tuple(MODES), trials, seed,
                      build_config(**params), n_boot=500)
    charts = {
        "Sharpe-like ratio": render(lambda path: plots.plot_sharpe(sweep, path)),
        "Risk of ruin": render(lambda path: plots.plot_risk_of_ruin(sweep, path)),
        "Where the P&L comes from": render(lambda path: plots.plot_attribution(sweep, path)),
        "Best vs worst spread": render(lambda path: plots.plot_pnl_histograms(sweep, path)),
    }
    rows = [
        {
            "Bot": point.mode,
            "Base spread": point.value,
            "Mean P&L": point.result.mean_pnl,
            "Std dev": point.result.std_pnl,
            "Sharpe-like": point.result.sharpe,
            "Risk of ruin": point.result.risk_of_ruin,
            "Fill rate": point.result.fill_rate,
            "Mean |inventory|": point.result.mean_abs_inventory,
        }
        for point in sorted(sweep.points, key=lambda p: (p.mode, p.value))
    ]
    best = sweep.best()
    return charts, findings(sweep), rows, {"mode": best.mode, "value": best.value,
                                           "sharpe": best.result.sharpe}


# ----------------------------------------------------------------------------- sidebar


def sidebar() -> dict:
    """Sidebar controls; returns the settings as plain values, ready for `build_config`."""
    bar = st.sidebar
    bar.title("Settings")
    bar.caption("The simulated market and the bot's quoting rule. Defaults match the repo.")

    bar.subheader("Market")
    sigma = bar.slider(
        "Volatility per tick ($)", 0.01, 0.15, DEFAULTS.price.sigma, 0.005,
        help="Standard deviation of the true price's move each tick.",
    )
    mean_reversion = bar.slider(
        "Mean reversion (0 = random walk)", 0.0, 0.05, 0.0, 0.005,
        help="Share of the gap back to the opening price closed each tick.",
    )
    n_steps = bar.select_slider(
        "Ticks per session", [250, 500, 1000, 2000], value=DEFAULTS.price.n_steps
    )
    arrival_rate = bar.slider(
        "Counterparties per tick, per side", 0.1, 3.0, DEFAULTS.order_flow.arrival_rate, 0.1,
        help="How much flow is willing to trade at the true price.",
    )
    price_sensitivity = bar.slider(
        "Price sensitivity of the flow (per $)", 5.0, 50.0,
        DEFAULTS.order_flow.price_sensitivity, 1.0,
        help="Higher means flow walks away faster as the bot's quote gets worse.",
    )

    bar.subheader("Strategy")
    base_spread = bar.slider(
        "Base spread ($)", 0.0, 0.60, DEFAULTS.market_maker.base_spread, 0.01,
        help="Full bid-ask width quoted when the bot is flat.",
    )
    inventory_penalty = bar.slider(
        "Inventory penalty ($ per unit)", 0.0, 0.02,
        DEFAULTS.market_maker.inventory_penalty, 0.0005, format="%.4f",
        help="How far quotes are skewed against the current position.",
    )
    spread_widening = bar.slider(
        "Spread widening ($ per unit)", 0.0, 0.01, 0.0, 0.0005, format="%.4f",
        help="Extra half-spread per unit held. The sweep shows this costs Sharpe.",
    )
    max_inventory = bar.slider(
        "Position limit (units)", 5, 50, DEFAULTS.market_maker.max_inventory, 5
    )

    bar.subheader("Risk and Monte Carlo")
    loss_limit = bar.slider(
        "Loss limit ($)", 5.0, 100.0, DEFAULTS.loss_limit, 5.0,
        help="A session counts as ruined if its P&L ever touches minus this amount.",
    )
    trials = bar.select_slider("Sessions per setting", [500, 1000, 2000, 5000], value=2000)
    seed = bar.number_input("Master seed", value=DEFAULT_SEED, step=1)

    return {
        "sigma": sigma,
        "mean_reversion": mean_reversion,
        "n_steps": n_steps,
        "arrival_rate": arrival_rate,
        "price_sensitivity": price_sensitivity,
        "base_spread": base_spread,
        "inventory_penalty": inventory_penalty,
        "spread_widening": spread_widening,
        "max_inventory": max_inventory,
        "loss_limit": loss_limit,
        "trials": trials,
        "seed": seed,
    }


# ----------------------------------------------------------------------------- app


def main() -> None:
    settings = sidebar()
    trials, seed = settings.pop("trials"), int(settings.pop("seed"))
    config = build_config(**settings)

    st.title("Market-making simulator")
    st.caption("A market maker quoting both sides of a simulated market, against adverse selection "
               "and inventory risk. Same engine as the command line: at the default settings these "
               "are the numbers in the README.")

    session_tab, distribution_tab, sweep_tab = st.tabs(
        ["One session", "Monte Carlo", "Spread sweep"]
    )

    with session_tab:
        st.write("One market, traded by both bots. A single session proves nothing, which is the "
                 "point of the other two tabs, but it shows the mechanism.")
        session_seed = st.number_input("Session seed", value=7, step=1,
                                       help="Any whole number. The same seed always gives the same session.")
        chart, rows = one_session(int(session_seed), **settings)
        st.image(chart)
        st.dataframe(rows, hide_index=True, column_config={
            "Final P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Max drawdown": st.column_config.NumberColumn(format="$%.2f"),
            "Time at limit": st.column_config.NumberColumn(format="percent"),
            "Spread capture": st.column_config.NumberColumn(format="$%.2f"),
            "Adverse selection": st.column_config.NumberColumn(format="$%.2f"),
            "Inventory p&l": st.column_config.NumberColumn(format="$%.2f"),
            "Liquidation": st.column_config.NumberColumn(format="$%.2f"),
        })

    with distribution_tab:
        st.write(f"{trials:,} independent sessions per bot at the current settings.")
        chart, rows, headline = monte_carlo(trials, seed, **settings)
        for mode, columns in zip(MODES, st.columns(len(MODES)), strict=True):
            stats = headline[mode]
            with columns:
                st.subheader(mode.capitalize())
                st.metric("Mean P&L per session", f"${stats['mean_pnl']:+,.2f}")
                st.metric("Sharpe-like ratio", f"{stats['sharpe']:.2f}",
                          help=f"95% CI {stats['sharpe_ci'][0]:.2f} to {stats['sharpe_ci'][1]:.2f}")
                st.metric("Risk of ruin", f"{stats['risk_of_ruin']:.2%}",
                          help=f"95% CI {stats['risk_of_ruin_ci'][0]:.2%} to "
                               f"{stats['risk_of_ruin_ci'][1]:.2%}")
                capture = stats["attribution"]["spread capture"]
                given_back = -stats["attribution"]["adverse selection"] / capture if capture > 0 else 0
                st.caption(money_safe(f"Captured ${capture:,.2f} of spread and gave back "
                                      f"{given_back:.0%} to adverse selection."))
        st.image(chart)
        st.dataframe(rows, hide_index=True, column_config={
            "Mean P&L": st.column_config.NumberColumn(format="$%.2f"),
            "Std dev": st.column_config.NumberColumn(format="$%.2f"),
            "Sharpe-like": st.column_config.NumberColumn(format="%.2f"),
            "Risk of ruin": st.column_config.NumberColumn(format="percent"),
            "Worst 5% average": st.column_config.NumberColumn(format="$%.2f"),
            "Mean max drawdown": st.column_config.NumberColumn(format="$%.2f"),
            "Fill rate": st.column_config.NumberColumn(format="percent"),
            "Mean |inventory|": st.column_config.NumberColumn(format="%.1f"),
            "Time at limit": st.column_config.NumberColumn(format="percent"),
        })

    with sweep_tab:
        predicted = predicted_optimal_spread(settings["sigma"], settings["price_sensitivity"])
        st.write(money_safe(f"Every spread below, run on the same {trials:,} markets, for both "
                            f"bots. First-order theory puts the P&L-maximising spread at "
                            f"**${predicted:.2f}** for this market."))
        low, high = st.slider("Spreads to test ($)", 0.0, 0.80, (0.0, 0.50), 0.02)
        step = st.select_slider("Step ($)", [0.01, 0.02, 0.05], value=0.02)
        spreads = tuple(round(low + step * i, 4) for i in range(int((high - low) / step) + 1))
        st.caption(f"{len(spreads)} spreads x {len(MODES)} bots x {trials:,} sessions = "
                   f"{len(spreads) * len(MODES) * trials:,} sessions.")

        if st.button("Run sweep", type="primary") or st.session_state.get("swept"):
            st.session_state["swept"] = True
            with st.spinner("Simulating..."):
                charts, conclusions, rows, best = spread_sweep(trials, seed, spreads, **settings)
            st.success(money_safe(f"Best: {best['mode']} quoting at a "
                                  f"{format_value('base_spread', best['value'])} spread "
                                  f"(Sharpe {best['sharpe']:.2f})."))
            for line in conclusions:
                st.markdown(money_safe(f"- {line}"))
            for name, chart in charts.items():
                st.subheader(name)
                st.image(chart)
            st.dataframe(rows, hide_index=True, column_config={
                "Base spread": st.column_config.NumberColumn(format="$%.2f"),
                "Mean P&L": st.column_config.NumberColumn(format="$%.2f"),
                "Std dev": st.column_config.NumberColumn(format="$%.2f"),
                "Sharpe-like": st.column_config.NumberColumn(format="%.2f"),
                "Risk of ruin": st.column_config.NumberColumn(format="percent"),
                "Fill rate": st.column_config.NumberColumn(format="percent"),
                "Mean |inventory|": st.column_config.NumberColumn(format="%.1f"),
            })

    with st.expander("Reproduce these settings in Python"):
        st.code(f"from session import SessionConfig, run_session\n\nconfig = {config!r}\n\n"
                f"session = run_session(config, seed={seed})\nprint(session.summary())", language="python")


if __name__ == "__main__":
    main()
