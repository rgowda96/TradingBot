# Daily Analysis — 2026-06-10

## Environment Status

**All external APIs blocked (403 "Host not in allowlist")** — network policy in this container
restricts outbound calls. Affected: CoinGecko, GeckoTerminal, DexScreener, DeFiLlama.
This blocked all live data ingestion. Code logic fixes were made based on static analysis.

---

## Backtest Stats (universal)

- DB total: 0 pools (no new pools could be fetched — GeckoTerminal 403)
- Networks attempted: base, solana, eth, arbitrum, optimism, bsc, polygon_pos
- All networks exhausted at page 1 → checkpoint reset for next run when APIs unblock

## Backtest Stats (known-winners)

- 14 tokens attempted, 14 CoinGecko 403 errors
- 0 catchable tokens with data → no signal stats available
- Most recent working run: 2026-06-01 (Jupiter only — already at $1.5B MCap, above bot range)

## Micro Scan

- 0 tokens scanned (all sources blocked)
- Watchlist: 0 active tokens
- Previous scan_results.json (stale, Jun 10 03:03): 56 tokens cached
  - 3 with score ≥60: APU (64), GM (64), STATED (60)
  - 25/56 tokens aged < 7 days (below quality filter min_pair_age_days: 7)
  - vol_accel avg 1.0, vol_liq avg 0.02 — most tokens in quiet/accumulation phase

## RL Model

- 0 outcome records — RL never trained
- All 10 signal arms still cold (need 3 observations each before going active)

## Convergence Target Status

| Target | Status |
|--------|--------|
| Universal backtest: sleeping_giant + slow_build win rate ≥ 50% | ⏸ No data |
| Universal backtest: ≥30% winning pools caught within 14d of launch | ⏸ No data |
| Micro scanner: ≥10 tokens/scan with conviction ≥6.0 | ⏸ APIs blocked |
| LinUCB: ≥3 active arms with mean_reward >0.10 | ⏸ No outcome data |
| Known-winners: all catchable categories fire within 21d | ⏸ CoinGecko blocked |

---

## Code Changes Made

### backtest.py — vol_surge_mult aligned with production (8× not 5×)
The `simulate()` function defaulted to `vol_surge_mult=5.0`, but the production bot uses
`spike_volume_multiplier: 8` in config.yaml, and `backtest_universal.py` uses `SURGE_MULT=8.0`.
The mismatch meant the known-winners backtest found volume-surge signals more easily than the
bot actually fires them — inflated hit-rate statistics. Fixed to 8.0.
Also updated the report label from "5× vs 7d baseline" to "8× vs 7d baseline".

### bot.py — Fix stale docstrings (two)
1. `_conviction_score` docstring said `_CONV_FLOOR (2.5)` — actual value is 2.0 since the
   floor was lowered (comment on line 935 already notes this). Fixed to 2.0.
2. `_quality_filter` docstring said "age ≥ 14d" — config.yaml has `min_pair_age_days: 7`
   and the code default is also 7. Fixed to "age ≥ 7d".

## Why No Threshold Changes
Without universal backtest data (GeckoTerminal blocked), there is no evidence base to
safely change sleeping_giant_days, spike_volume_multiplier, or resistance_breakout_pct.
All threshold changes in config.yaml must be supported by win-rate data from backtest_db.json.

## Next Steps (when APIs unblock)
1. Run `python3 backtest_universal.py --max 300` — will start fresh from page 1 on all chains
2. Run `python3 micro_scanner.py` — watchlist is empty, will populate from scratch
3. Run `python3 backtest.py` — will now use 8× volume surge multiplier (correct vs production)
4. Once backtest_db.json accumulates 50+ pools: calibrate thresholds per convergence targets
