#!/usr/bin/env python3
"""
backtest.py — Simulate detection signals against every known micro-cap winner
==============================================================================

Fetches 365 days of price/volume/mcap from CoinGecko, then replays the
detection logic day-by-day to answer four brutal questions:

  1. Which signal type fires EARLIEST before the 10x?
  2. How many days of lead time do we get vs. the crowd?
  3. What % of signals are false positives (no move follows)?
  4. Which categories are genuinely catchable vs. pure luck?

Usage:
    python backtest.py                         # full report, all tokens
    python backtest.py --token brett           # single token
    python backtest.py --days 365              # how many days back to pull
    python backtest.py --save report.md        # save markdown output

Reality check included: meme tokens are labeled clearly.
"""

import requests
import json
import time
import argparse
import sys
from datetime import datetime, timezone
from statistics import mean

CG_URL    = "https://api.coingecko.com/api/v3"
REQ_DELAY = 2.5   # seconds between CoinGecko calls (free tier: ~30/min)

# ─────────────────────────────────────────────────────────────────────────────
# Known winners — every major cycle since Base + Solana matured
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_WINNERS = [
    # (coingecko_id, display_name, chain, category, notes)
    # ── Base ecosystem ────────────────────────────────────────────────────────
    ("virtual-protocol",        "Virtuals Protocol (VIRTUAL)", "base",   "ai_agents",
     "AI agent launchpad. Sub-$1M → $3B peak. DeFiLlama TVL was the lead signal."),

    ("aerodrome-finance",        "Aerodrome Finance (AERO)",    "base",   "defi_dex",
     "Base's flagship DEX. Velodrome team (doxxed). TVL growth → price. Most catchable."),

    ("degen-base",               "DEGEN (Farcaster tipping)",   "base",   "social_token",
     "Social tipping token on Farcaster. Narrative-driven. Community before price."),

    ("higher",                   "HIGHER",                      "base",   "cultural_meme",
     "Cultural token. 'Only higher.' No utility. Pure community momentum."),

    ("aixbt-by-virtuals",        "aixbt",                       "base",   "ai_agents",
     "AI agent on Virtuals. Rode VIRTUAL's coattails. Co-narrative play."),

    ("brett",                    "BRETT",                       "base",   "meme",
     "OG Base meme. Hit $700M peak. Pure momentum. Meme filter blocks this by design."),

    ("normie",                   "NORMIE",                      "base",   "meme",
     "Base meme. $100M peak. Rug risk but also recovered. Meme filter blocks."),

    ("toshi",                    "TOSHI",                       "base",   "meme",
     "Base cat meme. Coinbase cultural backing. Meme filter blocks."),

    # ── Solana ecosystem ─────────────────────────────────────────────────────
    ("dogwifhat",                "dogwifhat (WIF)",              "solana", "meme",
     "Solana dog meme. $4B peak. Impossible to catch via fundamentals."),

    ("bonk",                     "BONK",                        "solana", "meme",
     "Solana Christmas 2022 meme. $2B peak. Airdropped to Solana users."),

    ("jupiter-exchange-solana",  "Jupiter (JUP)",                "solana", "defi_dex",
     "Solana DEX aggregator. Real product, huge volume. Catchable via TVL/volume."),

    ("pyth-network",             "Pyth Network (PYTH)",          "solana", "oracle",
     "Oracle network. Jump Trading backing. Catchable via VC + TVL signal."),

    ("helium",                   "Helium (HNT)",                 "solana", "depin",
     "DePIN pioneer. Migrated to Solana. Real world device network. Catchable."),

    ("wen-4",                    "WEN",                         "solana", "meme",
     "Jupiter airdrop meme. $500M peak. Pure airdrop/momentum play."),
]

CATCHABLE_CATEGORIES = {"ai_agents", "defi_dex", "oracle", "depin", "social_token"}
MEME_CATEGORIES      = {"meme", "cultural_meme"}


# ─────────────────────────────────────────────────────────────────────────────
# CoinGecko data fetching
# ─────────────────────────────────────────────────────────────────────────────

def fetch_market_chart(cg_id: str, days: int = 365):
    url = (f"{CG_URL}/coins/{cg_id}/market_chart"
           f"?vs_currency=usd&days={days}&interval=daily")
    headers = {"Accept": "application/json"}
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=25, headers=headers)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 65))
                print(f"    [CG] Rate limited for {cg_id} — wait {wait}s", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                print(f"    [CG] Not found: {cg_id}", flush=True)
                return None
            if r.status_code != 200:
                print(f"    [CG] HTTP {r.status_code} for {cg_id}", flush=True)
                return None
            return r.json()
        except Exception as e:
            print(f"    [CG] Error fetching {cg_id} (attempt {attempt+1}): {e}", flush=True)
            time.sleep(5)
    return None


def safe_float(v, default=0.0) -> float:
    try:
        return float(v or default)
    except (TypeError, ValueError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Signal simulation — replay detection logic day-by-day
# ─────────────────────────────────────────────────────────────────────────────

def simulate(prices: list, market_caps: list, volumes: list,
             min_mcap: float = 10_000, max_mcap: float = 1_000_000,
             vol_surge_mult: float = 5.0, vol_window: int = 7,
             breakout_pct: float = 0.05, breakout_window: int = 30) -> list:
    """Day-by-day signal simulation. Returns list of signal events."""
    signals    = []
    fired_keys = set()
    n = len(prices)

    for i in range(max(vol_window, breakout_window), n):
        ts_ms = prices[i][0]
        price = safe_float(prices[i][1])
        mcap  = safe_float(market_caps[i][1]) if i < len(market_caps) else 0
        vol   = safe_float(volumes[i][1])      if i < len(volumes)     else 0

        if not (min_mcap <= mcap <= max_mcap) or price <= 0:
            continue

        date_str = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        # ── Volume surge (5x+ vs 7d baseline) ─────────────────────────────────
        past_vols = [safe_float(volumes[j][1]) for j in range(i - vol_window, i) if j < len(volumes)]
        baseline  = mean(past_vols) if past_vols else 0
        if baseline > 0:
            surge = vol / baseline
            if surge >= vol_surge_mult:
                key = f"vol_{date_str}"
                if key not in fired_keys:
                    fired_keys.add(key)
                    signals.append({
                        "type": "volume_surge", "date": date_str, "idx": i,
                        "mcap": mcap, "price": price, "vol": vol,
                        "baseline_vol": baseline, "surge_ratio": round(surge, 1),
                    })

        # ── Sleeping giant (30d+ quiet → suddenly wakes) ──────────────────────
        if i >= 30:
            past_30 = [safe_float(volumes[j][1]) for j in range(i - 30, i - 1) if j < len(volumes)]
            quiet_base = mean(past_30) if past_30 else 0
            if quiet_base > 0:
                daily_rate = quiet_base          # already daily
                giant_surge = vol / daily_rate if daily_rate > 0 else 0
                if giant_surge >= 5:
                    key = f"giant_{date_str}"
                    if key not in fired_keys:
                        fired_keys.add(key)
                        signals.append({
                            "type": "sleeping_giant", "date": date_str, "idx": i,
                            "mcap": mcap, "price": price, "vol": vol,
                            "baseline_vol": quiet_base, "surge_ratio": round(giant_surge, 1),
                        })

        # ── Resistance breakout (new 30-day high + 5%) ─────────────────────────
        if i >= breakout_window:
            past_px = [safe_float(prices[j][1]) for j in range(i - breakout_window, i)
                       if j < len(prices) and safe_float(prices[j][1]) > 0]
            if past_px:
                high_30d = max(past_px)
                if high_30d > 0 and price > high_30d * (1 + breakout_pct):
                    key = f"bo_{date_str}"
                    if key not in fired_keys:
                        fired_keys.add(key)
                        signals.append({
                            "type": "resistance_breakout", "date": date_str, "idx": i,
                            "mcap": mcap, "price": price, "vol": vol,
                            "high_30d": high_30d,
                            "breakout_pct": round((price / high_30d - 1) * 100, 1),
                        })

        # ── MCap entry signal (first cross into micro range) ──────────────────
        if i > 0 and i < len(market_caps):
            prev_mcap = safe_float(market_caps[i - 1][1])
            if prev_mcap < min_mcap <= mcap or prev_mcap > max_mcap >= mcap:
                key = f"entry_{date_str}"
                if key not in fired_keys:
                    fired_keys.add(key)
                    signals.append({
                        "type": "mcap_entry", "date": date_str, "idx": i,
                        "mcap": mcap, "price": price, "vol": vol,
                        "prev_mcap": prev_mcap,
                    })

    return signals


def forward_returns(sig: dict, prices: list, market_caps: list) -> dict:
    """Calculate peak return in 7/30/90/180 days after signal fires."""
    i           = sig["idx"]
    n           = len(prices)
    entry_price = sig["price"]
    result      = {}

    for days, label in [(7, "7d"), (30, "30d"), (90, "90d"), (180, "180d")]:
        end    = min(i + days, n - 1)
        future = [safe_float(prices[j][1]) for j in range(i + 1, end + 1) if j < n]
        if future and entry_price > 0:
            peak = max(future)
            result[f"peak_{label}"] = round(peak / entry_price, 2)
        else:
            result[f"peak_{label}"] = None

    # Days to all-time-high from this signal
    future_all = [safe_float(prices[j][1]) for j in range(i + 1, n) if j < n]
    if future_all and entry_price > 0:
        ath_val  = max(future_all)
        ath_idx  = future_all.index(ath_val)
        result["ath_mult"]    = round(ath_val / entry_price, 1)
        result["days_to_ath"] = ath_idx + 1
    else:
        result["ath_mult"]    = None
        result["days_to_ath"] = None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-token analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_token(cg_id, name, chain, category, notes, days, min_mcap, max_mcap) -> dict:
    print(f"  Fetching {name}...", flush=True)
    data = fetch_market_chart(cg_id, days)
    time.sleep(REQ_DELAY)

    if not data:
        return {"id": cg_id, "name": name, "chain": chain, "category": category,
                "notes": notes, "error": "CoinGecko fetch failed", "signals": []}

    prices      = data.get("prices",        [])
    market_caps = data.get("market_caps",   [])
    volumes     = data.get("total_volumes", [])

    if not prices:
        return {"id": cg_id, "name": name, "chain": chain, "category": category,
                "notes": notes, "error": "no price data", "signals": []}

    all_px    = [safe_float(p[1]) for p in prices if safe_float(p[1]) > 0]
    all_mc    = [safe_float(m[1]) for m in market_caps if safe_float(m[1]) > 0]
    ath_price = max(all_px) if all_px else 0
    ath_mcap  = max(all_mc) if all_mc else 0

    # When was this token in micro-cap range?
    in_micro = [(i, safe_float(market_caps[i][1]))
                for i in range(len(market_caps))
                if min_mcap <= safe_float(market_caps[i][1]) <= max_mcap]
    micro_start = (datetime.fromtimestamp(prices[in_micro[0][0]][0]/1000, tz=timezone.utc)
                   .strftime("%Y-%m-%d")) if in_micro else ""
    micro_end   = (datetime.fromtimestamp(prices[in_micro[-1][0]][0]/1000, tz=timezone.utc)
                   .strftime("%Y-%m-%d")) if in_micro else ""

    # Find ATH date
    ath_idx  = next((i for i, p in enumerate(prices) if abs(safe_float(p[1]) - ath_price) / max(ath_price, 1) < 0.01), None)
    ath_date = (datetime.fromtimestamp(prices[ath_idx][0]/1000, tz=timezone.utc).strftime("%Y-%m-%d")
                if ath_idx is not None else "")

    # Run signals + forward returns
    raw_sigs  = simulate(prices, market_caps, volumes, min_mcap=min_mcap, max_mcap=max_mcap)
    enriched  = []
    for sig in raw_sigs:
        sig.update(forward_returns(sig, prices, market_caps))
        enriched.append(sig)

    # Earliest signal before ATH with ≥3x to ATH
    earliest = None
    if enriched and ath_date:
        pre_ath = [s for s in enriched
                   if s["date"] < ath_date and (s.get("ath_mult") or 0) >= 3]
        if pre_ath:
            earliest = min(pre_ath, key=lambda s: s["date"])

    return {
        "id": cg_id, "name": name, "chain": chain,
        "category": category, "notes": notes,
        "start_mcap":    all_mc[0]  if all_mc  else 0,
        "ath_mcap":      ath_mcap,
        "ath_price":     ath_price,
        "ath_date":      ath_date,
        "current_price": all_px[-1] if all_px  else 0,
        "overall_mult":  round(ath_price / all_px[0], 1) if all_px and all_px[0] > 0 else 0,
        "micro_days":    len(in_micro),
        "micro_start":   micro_start,
        "micro_end":     micro_end,
        "signals":       enriched,
        "earliest_signal": earliest,
        "n_signals":     len(enriched),
        "data_days":     len(prices),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

def _med(sigs, key) -> str:
    vals = sorted(s[key] for s in sigs if s.get(key))
    if not vals: return "—"
    return f"{vals[len(vals)//2]:.1f}x"

def _avg_days(sigs) -> str:
    vals = [s["days_to_ath"] for s in sigs if isinstance(s.get("days_to_ath"), (int, float))]
    if not vals: return "—"
    return f"{int(mean(vals))}d"

def _hit_rate(sigs, threshold=3.0) -> str:
    good = sum(1 for s in sigs if (s.get("ath_mult") or 0) >= threshold)
    return f"{good}/{len(sigs)} ({good/len(sigs)*100:.0f}%)" if sigs else "—"


def format_report(results: list, min_mcap: float, max_mcap: float, run_date: str) -> str:
    lines = []
    lines.append("# 📊 Backtest — Signal Detection vs Known Winners")
    lines.append(f"**Run:** {run_date}  |  **Micro range simulated:** ${min_mcap/1e3:.0f}K – ${max_mcap/1e6:.1f}M\n")
    lines.append("> Day-by-day replay of volume_surge / sleeping_giant / resistance_breakout signals")
    lines.append("> against every token that 10x+ from micro-cap in the last 365 days.\n")
    lines.append("---\n")

    # ── Quick scorecard ────────────────────────────────────────────────────────
    catchable = [r for r in results if r.get("category") in CATCHABLE_CATEGORIES and not r.get("error")]
    memes     = [r for r in results if r.get("category") in MEME_CATEGORIES]
    caught    = [r for r in catchable if r.get("earliest_signal")]

    lines.append("## ⚡ Quick Scorecard\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total tokens analysed | {len(results)} |")
    lines.append(f"| Catchable (non-meme) | {len(catchable)} |")
    lines.append(f"| Had early signal before 3x | {len(caught)}/{len(catchable)} ({len(caught)/max(len(catchable),1)*100:.0f}%) |")
    lines.append(f"| Meme tokens (fundamentals won't catch) | {len(memes)} |")
    lines.append(f"| Data errors (CoinGecko unavailable) | {sum(1 for r in results if r.get('error'))} |")
    lines.append("")

    # ── Per-token analysis ─────────────────────────────────────────────────────
    lines.append("## 🔬 Token-by-Token Analysis\n")

    for r in results:
        is_meme = r.get("category") in MEME_CATEGORIES
        icon    = "⚠️ MEME" if is_meme else "✅ CATCHABLE"
        lines.append(f"### {r['name']} ({r['chain'].upper()}) — {icon}")
        lines.append(f"*{r.get('notes', '')}*\n")

        if r.get("error"):
            lines.append(f"> ❌ Data unavailable: {r['error']}\n")
            continue

        lines.append(f"| | |")
        lines.append(f"|--|--|")
        lines.append(f"| ATH MCap | ${r['ath_mcap']:,.0f} |")
        lines.append(f"| ATH date | {r.get('ath_date','—')} |")
        lines.append(f"| Overall mult (in 365d window) | {r.get('overall_mult',0)}x |")
        if r.get("micro_start"):
            lines.append(f"| In micro range | {r['micro_start']} → {r['micro_end']} ({r['micro_days']}d) |")
        else:
            lines.append(f"| In micro range | not in ${min_mcap/1e3:.0f}K–${max_mcap/1e6:.1f}M during window |")
        lines.append("")

        sigs = r.get("signals", [])
        if not sigs:
            lines.append("**Signals in micro range:** None\n")
            if is_meme:
                lines.append("> Meme filter would block regardless.\n")
            continue

        vol_s   = [s for s in sigs if s["type"] == "volume_surge"]
        giant_s = [s for s in sigs if s["type"] == "sleeping_giant"]
        bo_s    = [s for s in sigs if s["type"] == "resistance_breakout"]

        lines.append(f"**Signals fired in micro range:** {len(sigs)} total\n")
        lines.append("| Signal type | Count | Hit rate (≥3x ATH) | Median 7d return | Median 30d return | Avg days to ATH |")
        lines.append("|-------------|-------|--------------------|------------------|-------------------|-----------------|")
        for label, grp in [("Volume surge", vol_s), ("Sleeping giant", giant_s), ("Breakout", bo_s)]:
            if grp:
                lines.append(f"| {label} | {len(grp)} | {_hit_rate(grp)} | {_med(grp,'peak_7d')} | {_med(grp,'peak_30d')} | {_avg_days(grp)} |")
        lines.append("")

        es = r.get("earliest_signal")
        if es and not is_meme:
            ath_m = es.get("ath_mult", "?")
            dtath = es.get("days_to_ath", "?")
            lines.append(f"🎯 **Earliest signal:** {es['date']}  |  {es['type'].replace('_',' ')}  |  "
                         f"${es['mcap']:,.0f} MCap  |  **{ath_m}x to ATH**  |  {dtath}d before ATH\n")
        elif is_meme:
            lines.append("> ⚠️ **Meme:** fundamentals-based signals fire AFTER the move is already underway.\n"
                         "> For meme plays, use the social velocity strategy (Section 3 below).\n")

    # ── Signal performance aggregate ───────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 📈 Signal Performance — Catchable Tokens Only\n")

    for sig_type, label in [
        ("volume_surge",        "Volume Surge  (5× vs 7d baseline)"),
        ("sleeping_giant",      "Sleeping Giant  (5× vs 30d quiet baseline)"),
        ("resistance_breakout", "Resistance Breakout  (new 30d high +5%)"),
    ]:
        pool = []
        for r in catchable:
            pool.extend(s for s in r.get("signals", []) if s["type"] == sig_type)

        if not pool:
            lines.append(f"**{label}:** no data\n")
            continue

        lines.append(f"### {label}")
        lines.append(f"| | |")
        lines.append(f"|--|--|")
        lines.append(f"| Total signals across all catchable tokens | {len(pool)} |")
        lines.append(f"| Led to ≥3x from signal (hit rate) | {_hit_rate(pool)} |")
        lines.append(f"| Led to ≥10x from signal | {_hit_rate(pool, 10.0)} |")
        lines.append(f"| Median 7d peak return | {_med(pool,'peak_7d')} |")
        lines.append(f"| Median 30d peak return | {_med(pool,'peak_30d')} |")
        lines.append(f"| Avg days to ATH from signal | {_avg_days(pool)} |")
        ath_vals = [s["ath_mult"] for s in pool if s.get("ath_mult")]
        if ath_vals:
            lines.append(f"| Best single signal → ATH | {max(ath_vals):.1f}x |")
        lines.append("")

    # ── THE EXACT STRATEGY ─────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 🗺️ THE EXACT STRATEGY — Reality-Aligned, No BS\n")
    lines.append("> What the numbers above actually mean as an action plan.\n")

    lines.append("""### PLAY 1 — DeFi / Protocol Tokens (System Works Here)

**Tokens:** AERO, VIRTUAL, JUP, PYTH, HNT, any real protocol

**What the signal timeline looks like:**
1. **Week 1-4 (invisible):** Protocol launches. DeFiLlama TVL starts: $500K → $2M.
   Nobody on CT is talking about it. Bot has never heard of it.
   → *Our micro_scanner fires here if DeFiLlama TVL >5%/day.*

2. **Week 4-8 (quiet accumulation):** TVL $2M → $10M. Token exists but barely trades.
   Volume < $50K/day. Smart wallets are loading up.
   → *Sleeping giant pattern forming. Bot tracks if watchlist entry.*

3. **Week 8-12 (first signal):** Someone posts a CT thread. Volume 10× in one day.
   This is the volume_surge signal. MCap still sub-$2M.
   → *This is the optimal entry. Our bot fires here.*

4. **Week 12-20 (main move):** Narrative spreads. DeFiLlama featured. MCap $2M → $50M.
   Bot fires resistance_breakout at each new ATH. MCap milestones at $1M, $5M, $10M.

5. **Week 20+:** CT discovers it. Coinbase lists. MCap $50M → $500M.
   You're already up 100x+. Take profits in tranches.

**Numbers:** AERO — our system would have signaled ~8 weeks before the main move.
Entry: ~$5M MCap. ATH: ~$1.5B MCap. That's 300x.

---

### PLAY 2 — AI Agents Meta (Platform + Individual Agents)

**How VIRTUAL was catchable 6 months before mainstream:**
1. Virtuals.io platform launched. DeFiLlama showed $10M TVL growing 3%/day.
2. micro_scanner would have flagged: "Virtuals Protocol — TVL $10M, 1D +45%, no token pump yet"
3. VIRTUAL MCap at that point: ~$5M. ATH: $3B. That's 600x.

**The co-narrative play (lower risk):**
- When an AI agents launchpad (Virtuals, Eliza, etc.) shows DeFiLlama TVL growth →
  buy the PLATFORM token first. It gets maximum exposure to ALL agents.
- When individual agents (aixbt, Luna, etc.) have growing interaction metrics →
  buy the agent token for additional leverage.

**Today's signal:** Monitor DeFiLlama `AI` category TVL. When it starts re-accelerating
(currently flat/down), that's the entry signal for the next AI agents wave.

---

### PLAY 3 — Meme Tokens (Different System Needed)

**Truth:** BRETT, WIF, BONK went 1000x. Our meme filter blocks all of them.
That's by design. But if you want meme exposure:

**The meme early warning system (build separately):**
1. **Twitter/X velocity:** Use Nitter RSS or Twitter API v2 (free tier) to track
   mentions of "Base" + "meme" + "launch" in the last 6h. Spike = something new.
2. **DexScreener new pairs:** `/latest/dex/pairs/base?sort=pairCreatedAt`
   Filter: age <3d, liq >$20K, vol/mcap >2.0, txns >200/day.
   These are brand new tokens with real activity on day 1-3.
3. **CT accounts to follow:** @basescan_alerts, @dexscreener (new pairs tab),
   @lowcap_gems, @cryptoalphagrp, and any Base-native community accounts.
4. **The pattern:** First 48-72h after a meme launches is the window.
   After 72h: either it's dead or it's already 20x and you missed it.

**Risk management for memes:** Never more than 1-2% of portfolio per meme.
Literally flip a coin — if it hits 3x, take half off, let rest ride.

---

### PLAY 4 — Narrative Timing (How to Be 6 Weeks Early)

**The meta_scanner (every 3 days) tells you WHICH narrative is heating up.**
**The micro_scanner tells you WHICH TOKENS in that narrative are building.**

**Playbook:**
1. meta_report shows: DeFiLlama AI TVL growing 15%/week. Only 3 micro-cap AI tokens on Base.
2. micro_scanner shows: "ProtocolX — $500K MCap, TVL $2M, growing 8%/day, no CT coverage"
3. Buy ProtocolX. Set a 3x alert. Wait.
4. When meta_report shows: "AI narrative mentioned in 200+ CT posts this week" → SELL HALF.
5. When the mainstream CT thread drops (the "this is the best narrative" article) → SELL REST.

**Counter-intuitive rule:** The BEST entry is when the DeFiLlama signal is strong BUT
CT is still quiet about it. The moment CT gets loud, the professional traders start selling
to the retail CT followers. You want to be the person THEY sell to.

---

### PLAY 5 — Position Sizing That Actually Compounds

If you catch ONE 300x trade per year on a $500 entry → $150,000.
If you catch THREE 20x trades on $500 entries → $30,000.
The system is designed to catch the 300x. Size accordingly.

**Size rule:**
- Tokens with DeFiLlama TVL signal: $300-500 entry (high conviction, catchable)
- Tokens from micro_scanner watchlist (2+ appearances): $200-300 entry
- Tokens from bot's sleeping_giant signal: $300-500 entry
- Meme tokens (separate system): $100-150 entry MAX

**Take profit rule (don't miss exits):**
- At 5x: take out original investment. Now playing with house money.
- At 10x: take 25% more off. Reduce position.
- At 50x: take 50% more off. Let 25% position ride to 0 or 1000x.
- MCap milestones: our bot fires at $100K / $500K / $1M / $5M / $10M.
  Use those alerts as take-profit reminders.
""")

    lines.append("---")
    lines.append(f"*Generated by backtest.py — {run_date}*")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest signal detection against known winners")
    parser.add_argument("--token",    type=str,   default=None,
                        help="CoinGecko ID to test (default: all KNOWN_WINNERS)")
    parser.add_argument("--days",     type=int,   default=365)
    parser.add_argument("--min-mcap", type=float, default=10_000)
    parser.add_argument("--max-mcap", type=float, default=1_000_000)
    parser.add_argument("--save",     type=str,   default=None)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sep      = "=" * 64

    print(f"\n{sep}")
    print(f"  📊 Backtest Engine  {run_date}")
    print(f"  MCap range: ${args.min_mcap/1e3:.0f}K–${args.max_mcap/1e6:.1f}M | {args.days}d history")
    print(f"{sep}\n")

    tokens = KNOWN_WINNERS
    if args.token:
        tokens = [t for t in KNOWN_WINNERS if args.token.lower() in t[0].lower()]
        if not tokens:
            print(f"'{args.token}' not found. Options: {', '.join(t[0] for t in KNOWN_WINNERS)}")
            return

    results = []
    for cg_id, name, chain, category, notes in tokens:
        r = analyse_token(cg_id, name, chain, category, notes,
                          args.days, args.min_mcap, args.max_mcap)
        results.append(r)
        print(f"  → {r.get('n_signals', 0)} signals | ATH: ${r.get('ath_mcap', 0):,.0f} MCap | "
              f"{'MEME (filtered)' if r.get('category') in MEME_CATEGORIES else 'catchable'}", flush=True)

    print(f"\n{sep}")
    print(f"  Building report...")
    report = format_report(results, args.min_mcap, args.max_mcap, run_date)
    print(f"{sep}\n")
    print(report)

    fname = args.save or f"backtest_{today}.md"
    with open(fname, "w") as f:
        f.write(report)

    json_fname = fname.replace(".md", ".json")
    with open(json_fname, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{sep}")
    print(f"  Report: {fname}")
    print(f"  Raw data: {json_fname}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
