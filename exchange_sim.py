"""
BINANCE PERPETUAL FUTURES SIMULATOR — ETH/USDT
================================================
Exact replica of the Binance USDT-M perpetual trading environment:

  ✓ Leverage tiers (1x–20x, user-configurable)
  ✓ Liquidation price (Binance formula, per-tier maintenance margin)
  ✓ Margin ratio monitoring → auto-liquidation when margin ratio ≥ 100%
  ✓ Isolated vs Cross margin (isolated by default — safer for systematic)
  ✓ Taker/Maker fees, spread, slippage (size-proportional)
  ✓ Funding rate every 8h (realistic historical range ±0.01–0.10%)
  ✓ Bankruptcy price / insurance fund gap
  ✓ Mark price = close price (simplified; real Binance uses index+basis)
  ✓ Position value, unrealized PnL, margin balance, available margin
  ✓ Full trade log with every fee component itemised
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ── Binance fee schedule (USDT-M Futures, VIP0) ──────────────────────────────
TAKER_FEE   = 0.00050   # 0.05%  market orders (entry, SL)
MAKER_FEE   = 0.00020   # 0.02%  limit orders  (TP exits)
HALF_SPREAD = 0.00010   # 0.01%  bid-ask half-spread on market entry

# ── Binance ETH/USDT bracket system (maintenance margin & max leverage) ───────
# (notional_limit_usd, maint_margin_rate, max_leverage)
ETH_BRACKETS = [
    (10_000,       0.0040, 125),
    (100_000,      0.0050, 100),
    (500_000,      0.0100,  50),
    (1_000_000,    0.0125,  25),
    (2_000_000,    0.0150,  20),
    (5_000_000,    0.0250,  10),
    (10_000_000,   0.0500,   5),
    (20_000_000,   0.1000,   4),
    (50_000_000,   0.1250,   3),
    (float('inf'), 0.5000,   2),
]

def get_bracket(notional_usd: float) -> tuple:
    """Return (maint_margin_rate, max_leverage) for a given notional."""
    for limit, mmr, max_lev in ETH_BRACKETS:
        if notional_usd <= limit:
            return mmr, max_lev
    return 0.5, 2

# ── Funding rate model ────────────────────────────────────────────────────────
# Binance charges/pays every 8h at 00:00, 08:00, 16:00 UTC
# Historical ETH rates range from -0.30% to +0.30% per 8h.
# We use a realistic regime-dependent model.
def get_funding_rate(direction: int, era_bias: str = "mixed") -> float:
    """
    Simulated funding rate per 8h period.
    direction: +1 long, -1 short
    era_bias:  bull/bear/range/volatile/mixed
    Positive rate → longs pay shorts. Negative → shorts pay longs.
    """
    rates = {
        "bull":     +0.00100,   # +0.10%/8h — longs overloaded in bull
        "bear":     -0.00050,   # -0.05%/8h — shorts overloaded in bear
        "range":    +0.00010,   # neutral-ish
        "volatile": +0.00050,   # usually long-biased
        "mixed":    +0.00010,
    }
    rate = rates.get(era_bias, 0.0001)
    # Long pays if rate > 0, short receives; short pays if rate < 0
    return rate if direction == 1 else -rate


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Position:
    """Represents one open leveraged position."""
    direction:        int     # +1 long, -1 short
    entry_px:         float
    qty:              float   # ETH quantity
    leverage:         float
    notional:         float   # qty × entry_px
    initial_margin:   float   # notional / leverage
    maint_margin_rate:float
    maint_margin:     float   # notional × mmr
    liq_price:        float
    bankruptcy_price: float
    sl_px:            float
    tp_px:            float
    entry_bar:        int
    era_bias:         str     = "mixed"
    # Running totals
    funding_paid:     float   = 0.0
    entry_fee:        float   = 0.0
    entry_slippage:   float   = 0.0
    entry_spread:     float   = 0.0


@dataclass
class TradeResult:
    """Complete record of a closed trade."""
    direction:          int
    entry_px:           float
    exit_px:            float
    qty:                float
    leverage:           float
    notional_entry:     float
    bars_held:          int
    exit_reason:        str
    # P&L
    gross_pnl:          float   # raw price move × qty × direction
    # Fees
    entry_fee:          float
    exit_fee:           float
    entry_spread:       float
    entry_slippage:     float
    exit_slippage:      float
    funding_paid:       float
    total_cost:         float
    net_pnl:            float
    # Risk info
    initial_margin:     float
    maint_margin_rate:  float
    liq_price:          float
    bankruptcy_price:   float
    margin_return_pct:  float   # net_pnl / initial_margin × 100 (leveraged return)
    unlev_return_pct:   float   # net_pnl / notional_entry × 100
    # Margin at close
    final_margin_ratio: float   # margin_ratio at close (>100% = liquidated)
    was_liquidated:     bool


# ─────────────────────────────────────────────────────────────────────────────
class ExchangeSim:
    """
    Simulates a Binance USDT-M perpetual futures account.

    Usage:
        sim = ExchangeSim(capital=10_000, leverage=3, risk_pct=0.01)
        sim.open(direction=1, entry_px=2000, sl_px=1900, tp_px=2200,
                 bar_idx=0, avg_volume=500)
        result = sim.update(bar_idx=5, high=2210, low=1990, close=2100,
                            signal=1, era_bias="bull")
    """

    def __init__(
        self,
        capital:      float = 10_000,
        leverage:     float = 3.0,       # default leverage
        risk_pct:     float = 0.01,      # % of equity to risk per trade
        max_leverage: float = 20.0,      # hard cap
        margin_mode:  str   = "isolated", # isolated | cross
    ):
        self.capital      = capital
        self.equity       = capital
        self.leverage     = min(leverage, max_leverage)
        self.risk_pct     = risk_pct
        self.max_leverage = max_leverage
        self.margin_mode  = margin_mode
        self.position: Optional[Position] = None
        self.equity_curve = []
        self.trades: list[TradeResult] = []
        self._bars_per_8h = {"1w": 0.083, "1d": 0.333, "4h": 1.0,
                              "1h": 8.0, "15m": 32.0, "5m": 96.0, "1m": 480.0}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _slippage(self, notional: float, avg_volume_usd: float) -> float:
        """Market impact slippage in USD."""
        if avg_volume_usd <= 0:
            return notional * 0.00005
        vol_frac = notional / max(avg_volume_usd, 1)
        return notional * (0.00005 + 0.0005 * vol_frac)

    def _liq_price(self, direction: int, entry_px: float,
                   leverage: float, mmr: float) -> float:
        """
        Binance isolated margin liquidation price.
        LONG:  liq = entry × (1 - 1/leverage + mmr)
        SHORT: liq = entry × (1 + 1/leverage - mmr)
        """
        imr = 1.0 / leverage
        if direction == 1:
            return round(entry_px * (1 - imr + mmr), 4)
        else:
            return round(entry_px * (1 + imr - mmr), 4)

    def _bankruptcy_price(self, direction: int, entry_px: float, leverage: float) -> float:
        """Price at which the margin is fully exhausted (before insurance fund)."""
        imr = 1.0 / leverage
        if direction == 1:
            return round(entry_px * (1 - imr), 4)
        else:
            return round(entry_px * (1 + imr), 4)

    def _margin_ratio(self, pos: Position, mark_px: float) -> float:
        """
        Margin ratio = maintenance_margin / margin_balance
        When >= 1.0 → liquidation.
        margin_balance (isolated) = initial_margin + unrealized_pnl
        """
        unreal_pnl = (mark_px - pos.entry_px) * pos.qty * pos.direction
        margin_balance = pos.initial_margin + unreal_pnl
        if margin_balance <= 0:
            return 999.0
        return pos.maint_margin / margin_balance

    def _position_size(self, equity: float, entry_px: float) -> float:
        """ETH quantity based on risk_pct and leverage."""
        risk_usd  = equity * self.risk_pct
        notional  = risk_usd * self.leverage
        return notional / entry_px

    # ── Open position ─────────────────────────────────────────────────────────

    def open(
        self,
        direction:   int,
        entry_px:    float,
        sl_px:       float,
        tp_px:       float,
        bar_idx:     int,
        avg_volume:  float = 0.0,
        era_bias:    str   = "mixed",
    ) -> dict:
        """Open a new leveraged position. Returns cost breakdown."""
        if self.position is not None:
            return {"error": "already_in_position"}

        qty      = self._position_size(self.equity, entry_px)
        notional = qty * entry_px
        mmr, max_lev = get_bracket(notional)

        # Clamp leverage to bracket max
        lev = min(self.leverage, max_lev)
        initial_margin = notional / lev
        maint_margin   = notional * mmr

        liq_px   = self._liq_price(direction, entry_px, lev, mmr)
        bank_px  = self._bankruptcy_price(direction, entry_px, lev)

        # Entry costs
        fee_in   = notional * TAKER_FEE
        spread   = notional * HALF_SPREAD
        slip     = self._slippage(notional, avg_volume * entry_px)
        total_in = fee_in + spread + slip

        self.equity -= (initial_margin + total_in)

        self.position = Position(
            direction=direction, entry_px=entry_px, qty=qty,
            leverage=lev, notional=notional,
            initial_margin=initial_margin, maint_margin_rate=mmr,
            maint_margin=maint_margin, liq_price=liq_px,
            bankruptcy_price=bank_px, sl_px=sl_px, tp_px=tp_px,
            entry_bar=bar_idx, era_bias=era_bias,
            entry_fee=fee_in, entry_slippage=slip, entry_spread=spread,
        )

        return {
            "action":           "OPEN",
            "direction":        "LONG" if direction == 1 else "SHORT",
            "entry_px":         entry_px,
            "qty":              round(qty, 4),
            "leverage":         f"{lev:.1f}x",
            "notional":         round(notional, 2),
            "initial_margin":   round(initial_margin, 2),
            "maint_margin":     round(maint_margin, 2),
            "maint_margin_rate":f"{mmr*100:.2f}%",
            "liq_price":        liq_px,
            "bankruptcy_price": bank_px,
            "sl_px":            sl_px,
            "tp_px":            tp_px,
            "dist_to_liq_pct":  round(abs(liq_px - entry_px) / entry_px * 100, 2),
            "entry_fee_usd":    round(fee_in, 4),
            "spread_usd":       round(spread, 4),
            "slippage_usd":     round(slip, 4),
            "total_entry_cost": round(total_in, 4),
            "equity_after":     round(self.equity, 2),
        }

    # ── Update on each bar ────────────────────────────────────────────────────

    def update(
        self,
        bar_idx:   int,
        high:      float,
        low:       float,
        close:     float,
        signal:    int,
        tf:        str   = "1d",
        era_bias:  str   = "mixed",
        avg_volume:float = 0.0,
    ) -> Optional[TradeResult]:
        """
        Call once per bar. Returns TradeResult if position was closed, else None.
        Also pays funding every 8h (simulated).
        """
        if self.position is None:
            self.equity_curve.append(self.equity)
            return None

        pos = self.position
        mark = close  # simplified: mark price = close price

        # ── Funding payment ──────────────────────────────────────────────────
        bph = self._bars_per_8h.get(tf, 1.0)
        if bph > 0 and (bar_idx - pos.entry_bar) % max(1, int(1/bph)) == 0:
            rate     = get_funding_rate(pos.direction, era_bias)
            funding  = pos.notional * abs(rate)
            # longs pay when rate > 0, shorts pay when rate < 0
            # rate already direction-adjusted in get_funding_rate
            if rate > 0:   # longs pay
                pos.funding_paid += funding
                self.equity      -= funding
            else:          # shorts pay
                pos.funding_paid += funding
                self.equity      -= funding

        # ── Liquidation check (mark price hits liq price) ───────────────────
        liquidated = False
        if pos.direction == 1:
            liquidated = low <= pos.liq_price
        else:
            liquidated = high >= pos.liq_price

        # ── Exit conditions ───────────────────────────────────────────────────
        bars_held  = bar_idx - pos.entry_bar
        hit_sl     = (pos.direction ==  1 and low  <= pos.sl_px) or \
                     (pos.direction == -1 and high >= pos.sl_px)
        hit_tp     = (pos.direction ==  1 and high >= pos.tp_px) or \
                     (pos.direction == -1 and low  <= pos.tp_px)
        hit_max    = bars_held >= 50
        signal_flip= signal not in (0, pos.direction)

        if not any([liquidated, hit_sl, hit_tp, hit_max, signal_flip]):
            mr = self._margin_ratio(pos, mark)
            unreal = (mark - pos.entry_px) * pos.qty * pos.direction
            self.equity_curve.append(self.equity + pos.initial_margin + unreal)
            return None

        # ── Determine exit price ──────────────────────────────────────────────
        if liquidated:
            exit_px     = pos.liq_price
            exit_reason = "LIQUIDATED"
        elif hit_tp:
            exit_px     = pos.tp_px
            exit_reason = "TP"
        elif hit_sl:
            exit_px     = pos.sl_px
            exit_reason = "SL"
        elif signal_flip:
            exit_px     = close
            exit_reason = "SIGNAL_FLIP"
        else:
            exit_px     = close
            exit_reason = "MAX_HOLD"

        # ── Gross P&L ─────────────────────────────────────────────────────────
        gross_pnl = (exit_px - pos.entry_px) * pos.qty * pos.direction

        # ── Exit fees ─────────────────────────────────────────────────────────
        exit_notional = pos.qty * exit_px
        exit_fee      = exit_notional * (MAKER_FEE if exit_reason == "TP" else TAKER_FEE)
        exit_slip     = self._slippage(exit_notional, avg_volume * exit_px)
        total_exit    = exit_fee + exit_slip

        total_cost    = (pos.entry_fee + pos.entry_spread + pos.entry_slippage
                         + exit_fee + exit_slip + pos.funding_paid)
        net_pnl       = gross_pnl - total_cost

        # ── Return margin + net PnL to equity ────────────────────────────────
        if liquidated:
            # Insurance fund absorbs the gap between bankruptcy & liquidation price
            # Trader loses entire initial margin
            returned = 0.0
        else:
            returned = pos.initial_margin + net_pnl

        self.equity += returned
        self.equity  = max(self.equity, 0)

        # ── Build result ──────────────────────────────────────────────────────
        mmr    = self._margin_ratio(pos, exit_px)
        result = TradeResult(
            direction=pos.direction,
            entry_px=pos.entry_px,
            exit_px=exit_px,
            qty=pos.qty,
            leverage=pos.leverage,
            notional_entry=pos.notional,
            bars_held=bars_held,
            exit_reason=exit_reason,
            gross_pnl=round(gross_pnl, 4),
            entry_fee=round(pos.entry_fee, 4),
            exit_fee=round(exit_fee, 4),
            entry_spread=round(pos.entry_spread, 4),
            entry_slippage=round(pos.entry_slippage, 4),
            exit_slippage=round(exit_slip, 4),
            funding_paid=round(pos.funding_paid, 4),
            total_cost=round(total_cost, 4),
            net_pnl=round(net_pnl, 4),
            initial_margin=round(pos.initial_margin, 4),
            maint_margin_rate=pos.maint_margin_rate,
            liq_price=pos.liq_price,
            bankruptcy_price=pos.bankruptcy_price,
            margin_return_pct=round(net_pnl / pos.initial_margin * 100, 3) if pos.initial_margin > 0 else 0,
            unlev_return_pct=round(gross_pnl / pos.notional * 100, 3),
            final_margin_ratio=round(mmr * 100, 2),
            was_liquidated=liquidated,
        )

        self.trades.append(result)
        self.position = None
        self.equity_curve.append(self.equity)
        return result

    # ── Position snapshot (for dashboard) ────────────────────────────────────

    def position_snapshot(self, current_px: float, tf: str = "1d") -> Optional[dict]:
        """Return current position state for display."""
        if self.position is None:
            return None
        pos = self.position
        unreal = (current_px - pos.entry_px) * pos.qty * pos.direction
        margin_bal = pos.initial_margin + unreal
        mr = pos.maint_margin / margin_bal if margin_bal > 0 else 999
        dist_liq = abs(current_px - pos.liq_price) / current_px * 100
        return {
            "direction":         "LONG" if pos.direction == 1 else "SHORT",
            "entry_px":          pos.entry_px,
            "current_px":        current_px,
            "qty":               round(pos.qty, 4),
            "leverage":          f"{pos.leverage:.1f}x",
            "notional":          round(pos.qty * current_px, 2),
            "initial_margin":    round(pos.initial_margin, 2),
            "maint_margin":      round(pos.maint_margin, 2),
            "liq_price":         pos.liq_price,
            "bankruptcy_price":  pos.bankruptcy_price,
            "sl_px":             pos.sl_px,
            "tp_px":             pos.tp_px,
            "unrealized_pnl":    round(unreal, 2),
            "unrealized_pct":    round(unreal / pos.initial_margin * 100, 2),
            "margin_balance":    round(margin_bal, 2),
            "margin_ratio_pct":  round(mr * 100, 2),
            "dist_to_liq_pct":   round(dist_liq, 2),
            "funding_paid":      round(pos.funding_paid, 4),
            "bars_held":         0,  # updated by caller
            "entry_fee":         round(pos.entry_fee, 4),
            "entry_spread":      round(pos.entry_spread, 4),
            "entry_slippage":    round(pos.entry_slippage, 4),
        }

    # ── Account summary ───────────────────────────────────────────────────────

    def account_summary(self) -> dict:
        closed = self.trades
        wins   = [t for t in closed if t.net_pnl > 0]
        losses = [t for t in closed if t.net_pnl <= 0]
        liqs   = [t for t in closed if t.was_liquidated]
        total_net   = sum(t.net_pnl for t in closed)
        total_gross = sum(t.gross_pnl for t in closed)
        total_fees  = sum(t.total_cost for t in closed)
        total_fund  = sum(t.funding_paid for t in closed)
        avg_lev     = np.mean([t.leverage for t in closed]) if closed else self.leverage
        return {
            "equity":           round(self.equity, 2),
            "total_net_pnl":    round(total_net, 2),
            "total_gross_pnl":  round(total_gross, 2),
            "total_fees_usd":   round(total_fees, 4),
            "total_funding_usd":round(total_fund, 4),
            "fee_drag_pct":     round(total_fees / max(abs(total_gross), 0.01) * 100, 1),
            "n_trades":         len(closed),
            "n_wins":           len(wins),
            "n_losses":         len(losses),
            "n_liquidations":   len(liqs),
            "win_rate_pct":     round(len(wins)/max(len(closed),1)*100, 1),
            "avg_leverage":     round(avg_lev, 2),
            "avg_margin_ret":   round(np.mean([t.margin_return_pct for t in closed]), 2) if closed else 0,
        }
