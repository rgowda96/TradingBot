"""
Realistic fee & cost model for ETH/USDT paper trading.

Costs modelled:
  1. Taker fee        — 0.05% per side (Binance VIP0 futures taker)
  2. Maker fee        — 0.02% per side (limit orders; used for TP exits)
  3. Slippage         — market-impact estimate based on trade size vs volume
  4. Bid-ask spread   — 0.01% half-spread on entry (market order crosses spread)
  5. Funding rate     — perpetual futures funding ~0.01% per 8h (paid/received)
                        direction-aware: longs pay when rate > 0, shorts receive
  6. Borrow cost      — for short positions: 0.01%/day (spot margin borrow)

All costs returned as absolute USD amounts so they show clearly in the log.
"""

# ── Fee schedule (Binance USDT-M futures, VIP0) ───────────────────────────────
TAKER_FEE     = 0.0005   # 0.05% — used for market entries and SL exits
MAKER_FEE     = 0.0002   # 0.02% — used for TP exits (limit order assumption)
HALF_SPREAD   = 0.0001   # 0.01% — half bid-ask spread on entry fill

# Slippage: estimated market impact (scales with order size relative to vol)
SLIPPAGE_BASE = 0.00005  # 0.005% base slippage (tiny orders)
SLIPPAGE_VOL_FACTOR = 0.001  # additional 0.1% per 1% of hourly volume consumed

# Funding rate (per 8-hour period, approximate neutral rate)
FUNDING_RATE_8H = 0.0001   # 0.01% per 8h = 0.03%/day = ~11%/year

# Borrow cost for shorts on spot margin
BORROW_RATE_PER_DAY = 0.0001  # 0.01%/day


def entry_cost(qty: float, entry_px: float, avg_volume: float = 0) -> dict:
    """
    Total cost of entering a position.
    avg_volume: average bar volume in ETH (used for slippage estimate).
    Returns dict of itemised costs in USD.
    """
    notional = qty * entry_px

    taker   = notional * TAKER_FEE
    spread  = notional * HALF_SPREAD

    # Slippage: larger orders relative to volume cost more
    if avg_volume > 0:
        vol_frac  = qty / avg_volume
        slip_pct  = SLIPPAGE_BASE + SLIPPAGE_VOL_FACTOR * vol_frac
    else:
        slip_pct  = SLIPPAGE_BASE
    slippage = notional * slip_pct

    total = taker + spread + slippage
    return {
        "taker":    round(taker,    4),
        "spread":   round(spread,   4),
        "slippage": round(slippage, 4),
        "total":    round(total,    4),
    }


def exit_cost(qty: float, exit_px: float, exit_reason: str,
              avg_volume: float = 0) -> dict:
    """
    Total cost of closing a position.
    TP exits use maker fee (limit order), SL/others use taker.
    """
    notional = qty * exit_px

    fee_rate = MAKER_FEE if exit_reason == "TP" else TAKER_FEE
    fee      = notional * fee_rate

    # Slippage on exit (only for market orders — not TP)
    if exit_reason != "TP":
        if avg_volume > 0:
            vol_frac  = qty / avg_volume
            slip_pct  = SLIPPAGE_BASE + SLIPPAGE_VOL_FACTOR * vol_frac
        else:
            slip_pct  = SLIPPAGE_BASE
        slippage = notional * slip_pct
    else:
        slippage = 0.0

    total = fee + slippage
    return {
        "fee_type": "maker" if exit_reason == "TP" else "taker",
        "fee":      round(fee,      4),
        "slippage": round(slippage, 4),
        "total":    round(total,    4),
    }


def funding_cost(qty: float, entry_px: float, direction: int,
                 bars_held: int, bars_per_8h: int) -> dict:
    """
    Cumulative funding cost over the holding period.
    direction: +1 long, -1 short.
    bars_held: number of candle bars held.
    bars_per_8h: how many bars equal 8 hours (e.g. 8 for 1h, 2 for 4h, 32 for 15m).
    Longs pay funding when rate > 0 (typical bull market); shorts receive.
    """
    periods   = bars_held / bars_per_8h
    notional  = qty * entry_px
    # Longs pay positive funding; shorts receive it (negative cost)
    net_rate  = FUNDING_RATE_8H * direction
    total     = notional * net_rate * periods
    return {
        "periods":  round(periods, 2),
        "rate_8h":  FUNDING_RATE_8H,
        "total":    round(total, 4),
    }


def borrow_cost(qty: float, entry_px: float, direction: int,
                bars_held: int, bars_per_day: int) -> dict:
    """Spot margin borrow cost for short positions."""
    if direction != -1:
        return {"total": 0.0}
    days     = bars_held / bars_per_day
    notional = qty * entry_px
    total    = notional * BORROW_RATE_PER_DAY * days
    return {
        "days":  round(days, 3),
        "total": round(total, 4),
    }


def bars_per_period(tf: str) -> tuple:
    """Returns (bars_per_8h, bars_per_day) for a given timeframe."""
    mapping = {
        "1m":  (480, 1440),
        "5m":  (96,  288),
        "15m": (32,  96),
        "1h":  (8,   24),
        "4h":  (2,   6),
        "1d":  (0.33, 1),
        "1w":  (0.048, 0.143),
    }
    return mapping.get(tf, (8, 24))


def total_trade_cost(
    tf:          str,
    qty:         float,
    entry_px:    float,
    exit_px:     float,
    direction:   int,
    bars_held:   int,
    exit_reason: str,
    avg_volume:  float = 0,
) -> dict:
    """
    Full cost breakdown for one round-trip trade.
    Returns itemised dict + grand total in USD.
    """
    bph, bpd = bars_per_period(tf)

    ec  = entry_cost(qty, entry_px, avg_volume)
    xc  = exit_cost(qty, exit_px, exit_reason, avg_volume)
    fc  = funding_cost(qty, entry_px, direction, bars_held, bph)
    bc  = borrow_cost(qty, entry_px, direction, bars_held, bpd)

    grand = ec["total"] + xc["total"] + fc["total"] + bc["total"]

    return {
        "entry_taker":   ec["taker"],
        "entry_spread":  ec["spread"],
        "entry_slip":    ec["slippage"],
        "exit_fee":      xc["fee"],
        "exit_fee_type": xc["fee_type"],
        "exit_slip":     xc["slippage"],
        "funding":       fc["total"],
        "borrow":        bc["total"],
        "grand_total":   round(grand, 4),
        "cost_pct":      round(grand / (qty * entry_px) * 100, 4) if entry_px > 0 else 0,
    }


def format_costs(costs: dict) -> str:
    return (
        f"fees=${costs['entry_taker']+costs['exit_fee']:.3f} "
        f"spread=${costs['entry_spread']:.3f} "
        f"slip=${costs['entry_slip']+costs['exit_slip']:.3f} "
        f"fund=${costs['funding']:.3f} "
        f"borrow=${costs['borrow']:.3f} "
        f"TOTAL=${costs['grand_total']:.3f} ({costs['cost_pct']:.3f}%)"
    )
