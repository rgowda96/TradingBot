# Daily Analysis — 2026-06-11

## Environment Status

**All external APIs blocked (403 "Host not in allowlist")** — network policy in this container
restricts outbound calls. Affected: CoinGecko, GeckoTerminal, DexScreener, DeFiLlama.
This is the second consecutive day of full API blockage. All live data ingestion blocked.

---

## Backtest Stats (universal)

- DB total: 0 pools (GeckoTerminal 403 — no new pools fetched)
- Networks attempted: base, solana, eth, arbitrum, optimism, bsc, polygon_pos
- All exhausted at page 1 → checkpoint reset for next run when APIs unblock
- Signal type win rates: no data
- False positive patterns: no data

## Backtest Stats (known-winners)

- 14 tokens attempted, 14 CoinGecko 403 errors
- 0 catchable tokens with data
- Earliest signal day: no data (all CoinGecko calls blocked)

## Micro Scan

- 0 tokens scanned across all 14 sources (all 403 blocked)
- Watchlist: 0 active tokens
- Cached scan_results.json (stale, prev run — 56 tokens):
  - Top 3 by score: Apu Apustaja (64), GM Everyday (64), useStated (60)
  - useStated (4.9d old) and kPEG (1.7d old) fail min_pair_age_days: 7
  - Apu Apustaja blocked by meme name filter (apustaja in MEME_CLONE_PATTERNS)
  - Burger Money (283d, $36K mcap, h24=+15%) — only non-meme with score 59 and positive momentum

## RL Model

- outcome_log.jsonl: not found (0 records)
- signal_outcomes.json: 0 records
- RL never trained — all 10 arms cold, need 3 observations each before activating

---

## Convergence Target Status

| Target | Status |
|--------|--------|
| Universal backtest: sleeping_giant + slow_build win rate ≥ 50% | ⏸ No data (APIs blocked) |
| Universal backtest: ≥30% winning pools caught within 14d of launch | ⏸ No data (APIs blocked) |
| Micro scanner: ≥10 tokens/scan with conviction ≥6.0 | ⏸ APIs blocked |
| LinUCB: ≥3 active arms with mean_reward >0.10 | ⏸ No outcome data |
| Known-winners: all catchable categories fire within 21d | ⏸ CoinGecko blocked |

---

## Code Changes Made

### bot.py — Fix two remaining stale floor comments (2.5 → 2.0)

Yesterday's run fixed the `_conviction_score` docstring at line 1275 but missed two other
places in bot.py where the old floor value of 2.5 appeared:

1. **Line 925** (philosophy block comment): `Any dimension < 2.5` → `Any dimension < 2.0`
2. **Line 1291** (inline comment in `_conviction_score`): `below 2.5` → `below 2.0`

Both were inconsistent with the actual `_CONV_FLOOR = 2.0` value (and its inline comment at
line 935 which already correctly said "Lowered 2.5 → 2.0"). The docstring at line 1275 was
corrected yesterday. Now all three places agree on 2.0.

---

## Why No Threshold Changes

Without universal backtest data (GeckoTerminal blocked), there is no evidence base for:
- `sleeping_giant_days` change (need sleeping_giant win rate + avg fire day from DB)
- `spike_volume_multiplier` change (need volume_surge win rate from DB)
- `resistance_breakout_pct` change (need breakout win rate from DB)

All threshold changes in config.yaml require win-rate data from backtest_db.json.

## Why No KNOWN_WINNERS Expansion

Cannot verify CoinGecko IDs without API access. No new high-confidence pools from universal
backtest (0 pools processed). Adding entries without CoinGecko verification risks bad IDs.

## Why No Quality Filter Changes

No rug pattern data from universal backtest (0 pools → 0 rugged examples). Quality filters
are sound — adding new checks without evidence would only increase false negatives.

---

## Next Steps (when APIs unblock)

1. `python3 backtest_universal.py --max 300` — DB starts at 0, will build from scratch
2. `python3 micro_scanner.py` — watchlist empty, will populate from fresh scan
3. `python3 backtest.py` — known-winners test with corrected 8× surge multiplier
4. Once backtest_db.json accumulates 50+ pools: calibrate thresholds per convergence targets
5. Sleeping giant: if avg fire day > 21 in DB, lower sleeping_giant_days: 30 → 21
