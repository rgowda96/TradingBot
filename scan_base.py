#!/usr/bin/env python3
"""
Comprehensive Base-chain gem scanner.

Fetches every known token on Base, applies hard quality filters, then scores
each survivor across 15+ parameters to rank by potential to reach $10M+.
Prints a full parameter breakdown and ranked list.
"""
import json
import os
import time
from collections import defaultdict

import requests

# ── API helpers ──────────────────────────────────────────────────────────────

_DEX_SESSION = requests.Session()
_DEX_SESSION.headers.update({"User-Agent": "gem-scanner/1.0"})
_GECKO_SESSION = requests.Session()
_GECKO_SESSION.headers.update({"Accept": "application/json", "User-Agent": "gem-scanner/1.0"})

def _dex_tokens(chain, addresses):
    """Fetch up to 30 token pairs at once from DexScreener."""
    url = f"https://api.dexscreener.com/tokens/v1/{chain}/{','.join(addresses)}"
    try:
        r = _DEX_SESSION.get(url, timeout=15)
        if r.status_code == 429:
            time.sleep(5)
            r = _DEX_SESSION.get(url, timeout=15)
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []
    except Exception as e:
        print(f"  [warn] dex batch error: {e}")
        return []

def _dex_search(term, chain="base"):
    """DexScreener free-text search."""
    try:
        r = _DEX_SESSION.get(
            "https://api.dexscreener.com/latest/dex/search",
            params={"q": term}, timeout=15)
        r.raise_for_status()
        return [p for p in (r.json().get("pairs") or []) if p.get("chainId") == chain]
    except Exception:
        return []

def _dex_profiles():
    """DexScreener latest token profiles."""
    try:
        r = _DEX_SESSION.get("https://api.dexscreener.com/token-profiles/latest/v1", timeout=15)
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []
    except Exception:
        return []

def _gecko_new_pools(chain="base"):
    """GeckoTerminal newest Base pools — freshest discoveries."""
    addrs = set()
    try:
        r = _GECKO_SESSION.get(
            f"https://api.geckoterminal.com/api/v2/networks/{chain}/new_pools",
            params={"page": 1, "include": "base_token"}, timeout=15)
        if not r.ok:
            return addrs
        for item in r.json().get("included", []):
            if item.get("type") == "token":
                addr = (item.get("attributes", {}).get("address") or "").lower()
                if addr.startswith("0x"):
                    addrs.add(addr)
    except Exception:
        pass
    return addrs

def _gecko_trending(chain="base"):
    """GeckoTerminal trending pools."""
    addrs = set()
    try:
        r = _GECKO_SESSION.get(
            f"https://api.geckoterminal.com/api/v2/networks/{chain}/trending_pools",
            params={"include": "base_token"}, timeout=15)
        if not r.ok:
            return addrs
        for item in r.json().get("included", []):
            if item.get("type") == "token":
                addr = (item.get("attributes", {}).get("address") or "").lower()
                if addr.startswith("0x"):
                    addrs.add(addr)
    except Exception:
        pass
    return addrs

# ── Address collection ────────────────────────────────────────────────────────

def collect_all_addresses():
    """Pull together every Base token address we know about."""
    all_addrs = set()

    # 1. discovered_addresses.json (main corpus — 1980+ tokens)
    if os.path.exists("discovered_addresses.json"):
        with open("discovered_addresses.json") as f:
            data = json.load(f)
        base = [a.lower() for a in data.get("base", []) if a]
        all_addrs.update(base)
        print(f"[+] discovered_addresses.json: {len(base)} base tokens")

    # 2. GeckoTerminal newest + trending pools (real-time feed)
    gecko_new = _gecko_new_pools("base")
    all_addrs.update(gecko_new)
    print(f"[+] GeckoTerminal new_pools: {len(gecko_new)} addresses")

    gecko_trend = _gecko_trending("base")
    all_addrs.update(gecko_trend)
    print(f"[+] GeckoTerminal trending: {len(gecko_trend)} addresses")

    # 3. DexScreener profiles (recently promoted/listed)
    profiles = _dex_profiles()
    prof_addrs = {p["tokenAddress"].lower() for p in profiles
                  if p.get("chainId") == "base" and p.get("tokenAddress")}
    all_addrs.update(prof_addrs)
    print(f"[+] DexScreener profiles: {len(prof_addrs)} addresses")

    # 4. Targeted search terms for gems not indexed above
    extra_terms = [
        "based", "ai", "agent", "dog", "cat", "pepe", "gem", "degen",
        "meme", "chad", "ape", "baby", "turbo", "launch", "new",
    ]
    search_found = set()
    for term in extra_terms:
        for p in _dex_search(term):
            addr = (p.get("baseToken") or {}).get("address", "").lower()
            if addr:
                search_found.add(addr)
        time.sleep(0.3)
    all_addrs.update(search_found)
    print(f"[+] DexScreener search ({len(extra_terms)} terms): {len(search_found)} addresses")

    # 5. State.json — tokens already being monitored by the bot
    if os.path.exists("state.json"):
        with open("state.json") as f:
            state = json.load(f)
        monitored = state.get("monitored_tokens", [])
        all_addrs.update(a.lower() for a in monitored)
        print(f"[+] state.json monitored: {len(monitored)} addresses")

    print(f"\n→ Total unique addresses: {len(all_addrs)}")
    return list(all_addrs)

# ── Token aggregation (same logic as bot.py) ─────────────────────────────────

def fetch_all_tokens(addresses, chain="base", batch=30, pause=0.4):
    """Fetch + aggregate pair data for all addresses."""
    print(f"\n[~] Fetching pair data for {len(addresses)} addresses in batches of {batch}…")
    # Map address → aggregated token view
    tokens = {}
    done = 0
    for i in range(0, len(addresses), batch):
        chunk = addresses[i:i+batch]
        pairs = _dex_tokens(chain, chunk)
        for pair in pairs:
            base = pair.get("baseToken") or {}
            addr = (base.get("address") or "").lower()
            if not addr:
                continue
            if addr not in tokens:
                tokens[addr] = {
                    "addr":           addr,
                    "name":           base.get("name", "?"),
                    "symbol":         base.get("symbol", "?"),
                    "best_pair":      pair,
                    "total_liq":      0.0,
                    "vol":            {"m5": 0.0, "h1": 0.0, "h6": 0.0, "h24": 0.0},
                    "buys_h1": 0, "sells_h1": 0,
                    "buys_h6": 0, "sells_h6": 0,
                    "buys_h24": 0, "sells_h24": 0,
                    "pair_created_at": None,   # oldest = true token age
                }
            t = tokens[addr]
            liq = (pair.get("liquidity") or {}).get("usd") or 0.0
            if liq > ((t["best_pair"].get("liquidity") or {}).get("usd") or 0.0):
                t["best_pair"] = pair
            t["total_liq"] += liq
            vol = pair.get("volume") or {}
            for tf in ("m5", "h1", "h6", "h24"):
                t["vol"][tf] += vol.get(tf) or 0.0
            for tf_key, bk, sk in [("h1","buys_h1","sells_h1"),
                                    ("h6","buys_h6","sells_h6"),
                                    ("h24","buys_h24","sells_h24")]:
                tx = (pair.get("txns") or {}).get(tf_key) or {}
                t[bk] += tx.get("buys") or 0
                t[sk] += tx.get("sells") or 0
            pts = pair.get("pairCreatedAt")
            if pts and (t["pair_created_at"] is None or pts < t["pair_created_at"]):
                t["pair_created_at"] = pts
        done += len(chunk)
        if done % 300 == 0 or done >= len(addresses):
            print(f"  fetched {done}/{len(addresses)} ({len(tokens)} tokens seen)")
        time.sleep(pause)
    return tokens

# ── Hard quality filters ──────────────────────────────────────────────────────

HARD_FILTERS = {
    "max_mcap":         100_000,    # sub-$100K only
    "min_mcap":         5_000,      # skip dust
    "min_liq":          5_000,      # need real pool depth
    "min_pair_age_days": 1,         # no same-day rugs
    "min_h24_txns":     20,         # organic trading minimum
    "max_fdv_mcap_ratio": 5.0,      # no hidden unlock dumps
    "require_social_or_website": True,
}

def apply_hard_filters(tokens, now):
    """Return (passing, failing_reason) for each token."""
    passed = {}
    reasons = defaultdict(int)
    for addr, t in tokens.items():
        best = t["best_pair"]
        mcap = best.get("marketCap") or best.get("fdv") or 0
        liq  = t["total_liq"]

        # MCap range
        if mcap > HARD_FILTERS["max_mcap"]:
            reasons["mcap > $100K"] += 1; continue
        if mcap < HARD_FILTERS["min_mcap"]:
            reasons["mcap < $5K (dust)"] += 1; continue

        # Liquidity
        if liq < HARD_FILTERS["min_liq"]:
            reasons["liq < $5K"] += 1; continue

        # Stablecoin / wrapped asset
        try:
            price = float(best.get("priceUsd") or 0)
            h24chg = abs(float((best.get("priceChange") or {}).get("h24") or 0))
            if price > 0 and abs(price - 1.0) < 0.02 and h24chg < 0.5:
                reasons["stablecoin/wrapped"] += 1; continue
        except (TypeError, ValueError):
            pass

        # Pair age
        pts = t["pair_created_at"]
        min_age = HARD_FILTERS["min_pair_age_days"]
        if pts and min_age > 0:
            age_days = (now * 1000 - pts) / 86_400_000
            if age_days < min_age:
                reasons[f"pair age < {min_age}d"] += 1; continue

        # Organic trading
        txns_h24 = t["buys_h24"] + t["sells_h24"]
        if txns_h24 < HARD_FILTERS["min_h24_txns"]:
            reasons["< 20 h24 txns"] += 1; continue

        # FDV / MCap ratio
        fdv = best.get("fdv") or 0
        if mcap >= 50_000 and fdv > 0 and fdv > mcap * HARD_FILTERS["max_fdv_mcap_ratio"]:
            reasons["FDV/MCap > 5x"] += 1; continue

        # Social / website
        if HARD_FILTERS["require_social_or_website"]:
            info = best.get("info") or {}
            if not info.get("websites") and not info.get("socials"):
                reasons["no social/website"] += 1; continue

        passed[addr] = t

    return passed, reasons

# ── Full scoring system ───────────────────────────────────────────────────────

def score_token(t, now):
    """
    Score a token across all 15 parameters (0-100 composite).

    The parameters, weights, and what they measure:
    ┌─────────────────────────────────┬──────┬────────────────────────────────────────────────┐
    │ Parameter                       │ Pts  │ Why it matters                                 │
    ├─────────────────────────────────┼──────┼────────────────────────────────────────────────┤
    │ 1. Market cap tier              │ 10   │ Smaller = bigger upside multiplier              │
    │ 2. Liquidity / MCap ratio       │ 10   │ Well-backed = harder to rug, real market        │
    │ 3. H1 buy pressure              │  8   │ Current hour buyers dominating = fresh demand   │
    │ 4. H6 sustained buy pressure    │  8   │ 6h buyers show lasting conviction not 1hr pump  │
    │ 5. Multi-TF buy alignment       │  7   │ h1 + h6 + h24 all dominated by buyers          │
    │ 6. Volume acceleration          │  8   │ h1 rate > h6 rate > h24 rate = picking up       │
    │ 7. Community engagement ratio   │  8   │ Txns per $100K mcap — grassroots interest       │
    │ 8. Pair age (project maturity)  │  8   │ Survived weeks/months = battle-tested           │
    │ 9. Social presence depth        │  7   │ Website + twitter + telegram = real team        │
    │ 10. Volume / Liq ratio          │  5   │ High vol vs pool = capital rotating in          │
    │ 11. Price trend alignment       │  5   │ m5/h1/h6/h24 price changes all positive         │
    │ 12. Breakout proximity          │  5   │ Price within 15% below ATH = coiling ready      │
    │ 13. Trading depth (txns h24)    │  4   │ Absolute transaction count (community size)     │
    │ 14. FDV discipline              │  4   │ Low FDV/MCap = no hidden supply to dump         │
    │ 15. Volume vs MCap ratio        │  3   │ Daily vol / mcap > 1x = highly active           │
    └─────────────────────────────────┴──────┴────────────────────────────────────────────────┘
    """
    best  = t["best_pair"]
    mcap  = best.get("marketCap") or best.get("fdv") or 0
    liq   = t["total_liq"]
    h1v   = t["vol"]["h1"]
    h6v   = t["vol"]["h6"]
    h24v  = t["vol"]["h24"]
    m5v   = t["vol"]["m5"]
    txns_h24 = t["buys_h24"] + t["sells_h24"]
    txns_h1  = t["buys_h1"] + t["sells_h1"]
    txns_h6  = t["buys_h6"] + t["sells_h6"]

    pchg = best.get("priceChange") or {}
    try:
        m5_chg  = float(pchg.get("m5")  or 0)
        h1_chg  = float(pchg.get("h1")  or 0)
        h6_chg  = float(pchg.get("h6")  or 0)
        h24_chg = float(pchg.get("h24") or 0)
    except (TypeError, ValueError):
        m5_chg = h1_chg = h6_chg = h24_chg = 0.0

    info = (best.get("info") or {})
    websites = info.get("websites") or []
    socials  = info.get("socials")  or []
    fdv = best.get("fdv") or 0

    pts = t["pair_created_at"]
    age_days = (now * 1000 - pts) / 86_400_000 if pts else 0.0

    scores = {}

    # 1. Market cap tier (max 10) — smaller mcap = bigger multiplier to $10M
    if mcap <= 10_000:       scores["mcap_tier"] = 10
    elif mcap <= 25_000:     scores["mcap_tier"] = 9
    elif mcap <= 50_000:     scores["mcap_tier"] = 7
    elif mcap <= 75_000:     scores["mcap_tier"] = 5
    else:                    scores["mcap_tier"] = 3

    # 2. Liquidity / MCap ratio (max 10) — deeper pools = real market
    liq_mcap = liq / mcap if mcap > 0 else 0
    if liq_mcap >= 1.0:     scores["liq_mcap"] = 10
    elif liq_mcap >= 0.5:   scores["liq_mcap"] = 8
    elif liq_mcap >= 0.25:  scores["liq_mcap"] = 6
    elif liq_mcap >= 0.10:  scores["liq_mcap"] = 3
    else:                   scores["liq_mcap"] = 0

    # 3. H1 buy pressure (max 8) — current hour buying dominance
    br_h1 = t["buys_h1"] / txns_h1 if txns_h1 >= 3 else None
    if br_h1 is None:       scores["buy_h1"] = 2
    elif br_h1 >= 0.80:     scores["buy_h1"] = 8
    elif br_h1 >= 0.65:     scores["buy_h1"] = 6
    elif br_h1 >= 0.55:     scores["buy_h1"] = 4
    elif br_h1 >= 0.40:     scores["buy_h1"] = 2
    else:                   scores["buy_h1"] = 0

    # 4. H6 sustained buy pressure (max 8) — sustained vs single-hour pump
    br_h6 = t["buys_h6"] / txns_h6 if txns_h6 >= 5 else None
    if br_h6 is None:       scores["buy_h6"] = 2
    elif br_h6 >= 0.75:     scores["buy_h6"] = 8
    elif br_h6 >= 0.60:     scores["buy_h6"] = 6
    elif br_h6 >= 0.50:     scores["buy_h6"] = 4
    elif br_h6 >= 0.40:     scores["buy_h6"] = 2
    else:                   scores["buy_h6"] = 0

    # 5. Multi-timeframe buy alignment (max 7) — all TFs showing buyers
    # True pre-breakout gems: buyers dominate on every timeframe simultaneously
    buy_tfs_above55 = sum([
        1 if (br_h1 is not None and br_h1 >= 0.55) else 0,
        1 if (br_h6 is not None and br_h6 >= 0.55) else 0,
        1 if (txns_h24 > 0 and t["buys_h24"] / txns_h24 >= 0.55) else 0,
    ])
    scores["buy_align"] = {3: 7, 2: 4, 1: 2, 0: 0}[buy_tfs_above55]

    # 6. Volume acceleration (max 8) — is volume picking up right now?
    h1_rate  = h1v / 1.0
    h6_rate  = h6v / 6.0   if h6v > 0 else 0
    h24_rate = h24v / 24.0 if h24v > 0 else 0
    accel = 0
    if h6_rate > 0 and h1_rate >= h6_rate * 1.5:   accel += 3
    elif h6_rate > 0 and h1_rate >= h6_rate:        accel += 2
    if h24_rate > 0 and h1_rate >= h24_rate * 2.0:  accel += 3
    elif h24_rate > 0 and h1_rate >= h24_rate * 1.5: accel += 2
    elif h24_rate > 0 and h1_rate >= h24_rate:       accel += 1
    if m5v > 0 and h1v > 0 and m5v / h1v >= 0.30:  accel += 2  # last 5min = 30%+ of h1
    scores["vol_accel"] = min(8, accel)

    # 7. Community engagement ratio (max 8) — txns per $100K mcap
    engagement = (txns_h24 / (mcap / 100_000)) if mcap > 0 and txns_h24 > 0 else 0
    if engagement >= 400:    scores["engagement"] = 8
    elif engagement >= 200:  scores["engagement"] = 6
    elif engagement >= 100:  scores["engagement"] = 4
    elif engagement >= 50:   scores["engagement"] = 2
    else:                    scores["engagement"] = 0

    # 8. Pair age / project maturity (max 8) — survived the market
    if age_days >= 180:      scores["age"] = 8
    elif age_days >= 90:     scores["age"] = 7
    elif age_days >= 30:     scores["age"] = 5
    elif age_days >= 14:     scores["age"] = 3
    elif age_days >= 7:      scores["age"] = 2
    elif age_days >= 1:      scores["age"] = 1
    else:                    scores["age"] = 0

    # 9. Social presence depth (max 7)
    has_web = bool(websites)
    has_tw  = any(s.get("type") == "twitter"  for s in socials)
    has_tg  = any(s.get("type") == "telegram" for s in socials)
    has_dc  = any(s.get("type") == "discord"  for s in socials)
    social_pts = (2 if has_web else 0) + (2 if has_tw else 0) + \
                 (2 if has_tg else 0) + (1 if has_dc else 0)
    scores["socials"] = min(7, social_pts)

    # 10. Volume / Liquidity ratio (max 5) — capital cycling through pool
    vol_liq = h1v / liq if liq > 0 and h1v > 0 else 0
    if vol_liq >= 0.5:      scores["vol_liq"] = 5
    elif vol_liq >= 0.2:    scores["vol_liq"] = 3
    elif vol_liq >= 0.05:   scores["vol_liq"] = 1
    else:                   scores["vol_liq"] = 0

    # 11. Price trend alignment (max 5) — all timeframes pointing up
    price_tfs_pos = sum([m5_chg > 0, h1_chg > 0, h6_chg > 0, h24_chg > 0])
    scores["price_align"] = {4: 5, 3: 3, 2: 1, 1: 0, 0: 0}[price_tfs_pos]

    # 12. Breakout proximity — bonus if price is coiling near ATH ceiling
    # (We don't have cached ATH reference here, so we use h24 context as proxy:
    #  if h24 price change is small-positive and h1 just turned green = coiling)
    scores["coiling_proxy"] = 0
    if 0 < h24_chg < 20 and h1_chg > 0 and h6_chg > -5:
        scores["coiling_proxy"] = 3
    if 0 < h24_chg < 10 and h1_chg > 1 and txns_h1 >= 5:
        scores["coiling_proxy"] = 5

    # 13. Trading depth — absolute h24 transaction count
    if txns_h24 >= 500:      scores["txn_depth"] = 4
    elif txns_h24 >= 200:    scores["txn_depth"] = 3
    elif txns_h24 >= 100:    scores["txn_depth"] = 2
    elif txns_h24 >= 50:     scores["txn_depth"] = 1
    else:                    scores["txn_depth"] = 0

    # 14. FDV discipline (max 4) — low unlock risk
    if fdv <= 0 or mcap <= 0:   scores["fdv_disc"] = 2
    elif fdv <= mcap * 1.1:     scores["fdv_disc"] = 4   # fully circulating
    elif fdv <= mcap * 2.0:     scores["fdv_disc"] = 3
    elif fdv <= mcap * 3.0:     scores["fdv_disc"] = 2
    else:                       scores["fdv_disc"] = 0

    # 15. Vol / MCap ratio (max 3) — daily volume vs market cap (velocity)
    vol_mcap = h24v / mcap if mcap > 0 and h24v > 0 else 0
    if vol_mcap >= 2.0:     scores["vol_mcap"] = 3
    elif vol_mcap >= 0.5:   scores["vol_mcap"] = 2
    elif vol_mcap >= 0.1:   scores["vol_mcap"] = 1
    else:                   scores["vol_mcap"] = 0

    total = sum(scores.values())
    max_total = 10 + 10 + 8 + 8 + 7 + 8 + 8 + 8 + 7 + 5 + 5 + 5 + 4 + 4 + 3  # = 100
    return total, scores, {
        "mcap":         mcap,
        "liq":          liq,
        "liq_mcap_pct": round(liq_mcap * 100, 1),
        "h24v":         h24v,
        "h1v":          h1v,
        "h6v":          h6v,
        "age_days":     round(age_days, 1),
        "txns_h24":     txns_h24,
        "txns_h1":      txns_h1,
        "engagement":   round(engagement, 0),
        "br_h1":        round(br_h1 * 100, 0) if br_h1 is not None else None,
        "br_h6":        round(br_h6 * 100, 0) if br_h6 is not None else None,
        "br_h24":       round(t["buys_h24"] / txns_h24 * 100, 0) if txns_h24 > 0 else None,
        "m5_chg":       m5_chg, "h1_chg": h1_chg,
        "h6_chg":       h6_chg, "h24_chg": h24_chg,
        "fdv":          fdv,
        "fdv_ratio":    round(fdv / mcap, 1) if mcap > 0 and fdv > 0 else None,
        "vol_mcap":     round(vol_mcap, 2),
        "accel":        accel,
        "mult_to_10M":  round(10_000_000 / mcap) if mcap > 0 else None,
        "has_web":      bool(websites),
        "has_tw":       any(s.get("type") == "twitter"  for s in socials),
        "has_tg":       any(s.get("type") == "telegram" for s in socials),
        "url":          best.get("url", ""),
        "contract":     t["addr"],
        "name":         t["name"],
        "symbol":       t["symbol"],
        "price":        best.get("priceUsd", "?"),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(ranked, reasons_dict, total_fetched, now):
    SCORE_MAX = 100

    print("\n" + "="*100)
    print("  BASE CHAIN GEM SCAN — TOKENS UNDER $100K WITH $10M+ POTENTIAL")
    print(f"  Scanned: {total_fetched} tokens  |  Passed quality filters: {len(ranked)}")
    print("="*100)

    print("\n── PARAMETERS USED ────────────────────────────────────────────────────────────────────────────")
    print("""
HARD FILTERS (token must pass ALL of these or it's excluded entirely):
  1.  Market cap:           $5K – $100K   (entry-level gems only)
  2.  Liquidity:            ≥ $5K         (need real pool depth to enter)
  3.  Pair age:             ≥ 1 day       (eliminates same-hour rug launches)
  4.  H24 transactions:     ≥ 20          (minimum organic trading activity)
  5.  FDV / MCap ratio:     ≤ 5×          (max 5× total supply vs circulating)
  6.  Social or website:    required      (real projects announce themselves)
  7.  Not a stablecoin:     auto-filtered (price near $1 + flat change)

SCORING PARAMETERS (0–100 composite — higher = more likely to run):
  #   Parameter                   Max  What it measures
  ─── ─────────────────────────── ──── ─────────────────────────────────────────────
  1   Market cap tier             /10  Smaller mcap = bigger multiplier to $10M
  2   Liquidity / MCap ratio      /10  Pool depth vs size = how hard to rug
  3   H1 buy pressure             / 8  Current-hour buying dominance (fresh demand)
  4   H6 sustained buy pressure   / 8  6-hour buyers → conviction vs one-hour pump
  5   Multi-TF buy alignment      / 7  h1+h6+h24 all dominated by buyers
  6   Volume acceleration         / 8  Is volume rate picking up right now?
  7   Community engagement ratio  / 8  Txns per $100K mcap → grassroots interest
  8   Pair age / project maturity / 8  Project survived weeks/months of market
  9   Social presence depth       / 7  Website + Twitter + Telegram + Discord
  10  Volume / Liquidity ratio    / 5  Capital cycling through pool (active market)
  11  Price trend alignment       / 5  m5/h1/h6/h24 price changes all positive
  12  Coiling proxy               / 5  h24 small-positive + h1 just turned green
  13  Trading depth (h24 txns)    / 4  Absolute community transaction count
  14  FDV discipline              / 4  Low unlock risk (FDV ≈ MCap = fully diluted)
  15  Vol / MCap ratio            / 3  Daily volume relative to market cap (velocity)
  ─── ─────────────────────────── ────
      TOTAL                       /100
    """)

    print("── FILTER REJECTION BREAKDOWN ──────────────────────────────────────────────────────────────")
    for reason, count in sorted(reasons_dict.items(), key=lambda x: -x[1]):
        pct = count / max(1, total_fetched) * 100
        bar = "█" * min(40, int(pct / 2))
        print(f"  {reason:<30} {count:>5} ({pct:4.1f}%)  {bar}")

    print("\n── RANKED RESULTS ──────────────────────────────────────────────────────────────────────────")
    HEADER = (
        f"{'#':>3}  {'Token':<22} {'Score':>5}  {'MCap':>9}  {'Mult':>5}  "
        f"{'Liq%':>5}  {'BrH1':>5}  {'BrH6':>5}  {'Engmt':>6}  "
        f"{'Age':>5}  {'Accel':>5}  {'h1vol':>7}  {'h24vol':>8}  "
        f"{'Soc':>4}  {'Txns24':>6}"
    )
    print(HEADER)
    print("─" * len(HEADER))

    for rank, (score, scores, meta) in enumerate(ranked[:50], 1):
        pct  = score / SCORE_MAX * 100
        name = f"{meta['name'][:14]}({meta['symbol'][:6]})"
        mult_txt = f"{meta['mult_to_10M']:,}×" if meta['mult_to_10M'] else "?"
        liq_pct  = f"{meta['liq_mcap_pct']:.0f}%"
        brh1     = f"{meta['br_h1']:.0f}%" if meta['br_h1'] is not None else "n/a"
        brh6     = f"{meta['br_h6']:.0f}%" if meta['br_h6'] is not None else "n/a"
        soc      = ("W" if meta["has_web"] else "-") + ("𝕏" if meta["has_tw"] else "-") + ("T" if meta["has_tg"] else "-")
        accel    = f"{meta['accel']}/8"
        print(
            f"{rank:>3}  {name:<22} {pct:>4.0f}%  ${meta['mcap']:>8,.0f}  {mult_txt:>5}  "
            f"{liq_pct:>5}  {brh1:>5}  {brh6:>5}  {meta['engagement']:>6,.0f}  "
            f"{meta['age_days']:>4.0f}d  {accel:>5}  ${meta['h1v']:>6,.0f}  "
            f"${meta['h24v']:>7,.0f}  {soc:>4}  {meta['txns_h24']:>6,}"
        )

    print("\n── TOP 20 DETAILED PROFILES ────────────────────────────────────────────────────────────────")
    for rank, (score, scores, meta) in enumerate(ranked[:20], 1):
        pct = score / SCORE_MAX * 100
        fdv_txt = f"  FDV/MCap: {meta['fdv_ratio']:.1f}×" if meta['fdv_ratio'] else ""
        chg_txt = (f"m5:{meta['m5_chg']:+.1f}%  h1:{meta['h1_chg']:+.1f}%  "
                   f"h6:{meta['h6_chg']:+.1f}%  h24:{meta['h24_chg']:+.1f}%")
        soc_parts = []
        if meta["has_web"]: soc_parts.append("web")
        if meta["has_tw"]:  soc_parts.append("𝕏/twitter")
        if meta["has_tg"]:  soc_parts.append("telegram")
        soc_str = " + ".join(soc_parts) if soc_parts else "none"
        bar_filled = round(pct / 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)

        # Score breakdown
        breakdown = "  ".join(
            f"{k}:{v}" for k, v in scores.items()
        )

        print(f"""
#{rank}  {meta['name']} ({meta['symbol']})  [{bar}] {pct:.0f}/100
    MCap:         ${meta['mcap']:,.0f}  [{meta['mult_to_10M']:,}× to $10M]{fdv_txt}
    Liquidity:    ${meta['liq']:,.0f}  ({meta['liq_mcap_pct']:.0f}% of mcap)
    Buy ratio:    h1:{meta['br_h1'] or '?'}%  h6:{meta['br_h6'] or '?'}%  h24:{meta['br_h24'] or '?'}%
    Volume:       h1:${meta['h1v']:,.0f}  h6:${meta['h6v']:,.0f}  h24:${meta['h24v']:,.0f}  accel:{meta['accel']}/8
    Vol/MCap:     {meta['vol_mcap']:.2f}×  Engagement: {meta['engagement']:.0f} txns/$100K
    Txns h24:     {meta['txns_h24']:,}  Age: {meta['age_days']:.0f}d
    Price chg:    {chg_txt}
    Socials:      {soc_str}
    Score detail: {breakdown}
    Contract:     {meta['contract']}
    URL:          {meta['url']}""")

    # Summary stats
    if ranked:
        scores_all = [s for s, _, _ in ranked]
        above_60 = sum(1 for s in scores_all if s/SCORE_MAX*100 >= 60)
        above_40 = sum(1 for s in scores_all if s/SCORE_MAX*100 >= 40)
        print(f"\n── SUMMARY ─────────────────────────────────────────────────────────────────────────────────")
        print(f"  Tokens passing all hard filters:   {len(ranked)}")
        print(f"  Score ≥ 60/100 (HIGH conviction):  {above_60}")
        print(f"  Score ≥ 40/100 (WATCHLIST):        {above_40}")
        print(f"  Median score:                      {sorted(scores_all)[len(scores_all)//2]:.0f}/100")
        print()
        print("  Most common reasons tokens were filtered out:")
        for r, c in sorted(reasons_dict.items(), key=lambda x:-x[1])[:5]:
            print(f"    {r}: {c}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    now = time.time()
    print("="*60)
    print("  BASE CHAIN COMPREHENSIVE GEM SCAN")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Phase 1: collect all addresses
    print("\n[PHASE 1] Collecting all known Base token addresses…")
    addresses = collect_all_addresses()

    # Phase 2: fetch live pair data
    print("\n[PHASE 2] Fetching live DexScreener pair data…")
    tokens = fetch_all_tokens(addresses)
    print(f"  → {len(tokens)} tokens returned data from DexScreener")

    # Phase 3: hard filters
    print("\n[PHASE 3] Applying hard quality filters…")
    passed, reasons = apply_hard_filters(tokens, now)
    total_rejected = sum(reasons.values())
    print(f"  → {len(passed)} tokens passed  ({total_rejected} rejected)")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:8]:
        print(f"    {reason}: {count}")

    # Phase 4: score and rank
    print("\n[PHASE 4] Scoring survivors across all 15 parameters…")
    ranked = []
    for addr, t in passed.items():
        total_score, score_breakdown, meta = score_token(t, now)
        ranked.append((total_score, score_breakdown, meta))
    ranked.sort(key=lambda x: -x[0])
    print(f"  → Ranked {len(ranked)} tokens")

    # Phase 5: report
    print_report(ranked, reasons, len(tokens), now)

    # Save results to JSON for further analysis
    output = []
    for score, scores, meta in ranked[:100]:
        output.append({
            "score": score,
            "score_pct": round(score/100*100, 1),
            "scores": scores,
            **meta,
        })
    with open("scan_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[+] Top 100 results saved to scan_results.json")


if __name__ == "__main__":
    main()
