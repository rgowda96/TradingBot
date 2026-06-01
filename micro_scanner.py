#!/usr/bin/env python3
"""
micro_scanner.py — Under-the-radar micro-cap tracker
=====================================================

Tracks $10K–$1M MCap tokens across Base + Solana from MULTIPLE sources,
not just one API. Maintains a persistent watchlist so tokens appearing
across 2+ scans get flagged — that is the real signal.

DATA SOURCES (all free, no API key required):
  1. GeckoTerminal — trending pools (Base + Solana)
  2. GeckoTerminal — top pools by 6H price change (catches moves not yet trending)
  3. GeckoTerminal — top pools by 24H volume (confirms sustained interest)
  4. DeFiLlama    — Base protocols with growing TVL under $10M
                    (catches protocols building before their token pumps)
  5. CoinGecko    — Base ecosystem tokens in $10K–$1M MCap range
  6. DexScreener  — boosted + profiled Base tokens (paid = has backing)

Watchlist file: micro_watchlist.json (committed to repo — persistent memory)

Key insight:
  trending once    = noise
  trending 2+ scans = something building quietly
  TVL growing + small MCap = the rarest signal
  MCap crosses $1M while on watchlist = caught it early

Usage:
    python micro_scanner.py
    python micro_scanner.py --min-mcap 10000 --max-mcap 1000000
    python micro_scanner.py --chain base
    python micro_scanner.py --chain solana,base --pages 6
"""

import requests
import json
import os
import sys
import re
import time
import argparse
from datetime import datetime, timezone
from collections import defaultdict

WATCHLIST_FILE = "micro_watchlist.json"
GECKO_URL      = "https://api.geckoterminal.com/api/v2"
CG_URL         = "https://api.coingecko.com/api/v3"
LLAMA_URL      = "https://api.llama.fi"
DEX_URL        = "https://api.dexscreener.com"
BIRDEYE_URL    = "https://public-api.birdeye.so"
GOPLUS_URL     = "https://api.gopluslabs.io/api/v1"

MEME_KEYWORDS = {
    "doge","pepe","shib","inu","moon","safe","elon","baby","floki","bonk",
    "wif","mog","brett","toshi","ponke","pnut","harambe","tremp","boden",
    "grok","wojak","cope","ape","frog","cat coin","dog wif","memecoin",
    "meme coin","meme token",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_meme(name: str, symbol: str = "") -> bool:
    text = (name + " " + symbol).lower()
    return any(kw in text for kw in MEME_KEYWORDS)

def days_since(iso_str: str) -> int:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return -1

def load_watchlist() -> dict:
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"version": 2, "tokens": {}, "defi_protocols": {}, "last_updated": ""}

def save_watchlist(wl: dict):
    wl["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(wl, f, indent=2)

def safe_float(val, default=0.0) -> float:
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default

# ─────────────────────────────────────────────────────────────────────────────
# Conviction scorer (for DEX pool data)
# ─────────────────────────────────────────────────────────────────────────────

def conviction_score(pool: dict) -> float:
    """0–100 composite conviction score from pool attributes."""
    attrs  = pool.get("attributes", {})
    pc     = attrs.get("price_change_percentage", {})
    h6     = safe_float(pc.get("h6"))
    h24    = safe_float(pc.get("h24"))
    vol24h = safe_float((attrs.get("volume_usd") or {}).get("h24"))
    mcap   = max(safe_float(attrs.get("market_cap_usd") or attrs.get("fdv_usd")), 1)
    liq    = safe_float(attrs.get("reserve_in_usd"))
    score  = 0.0

    # 6H momentum
    if h6 > 200:   score += 40
    elif h6 > 100: score += 32
    elif h6 > 50:  score += 24
    elif h6 > 20:  score += 14
    elif h6 > 5:   score += 7
    elif h6 < -40: score -= 12

    # 24H confirms trend
    if h24 > 200:   score += 22
    elif h24 > 100: score += 16
    elif h24 > 50:  score += 10
    elif h24 > 20:  score += 5
    elif h24 < -50: score -= 10

    # Volume / MCap ratio
    ratio = vol24h / mcap
    if ratio > 2.0:   score += 25
    elif ratio > 1.0: score += 18
    elif ratio > 0.5: score += 12
    elif ratio > 0.2: score += 6
    elif ratio > 0.05: score += 2

    # Liquidity depth (rug safety)
    if liq > 200_000: score += 15
    elif liq > 100_000: score += 11
    elif liq > 50_000:  score += 7
    elif liq > 10_000:  score += 3
    elif liq < 2_000:   score -= 20

    # Buy/sell pressure
    t_h1  = (attrs.get("transactions") or {}).get("h1") or {}
    buys  = int(t_h1.get("buys", 0) or 0)
    sells = int(t_h1.get("sells", 0) or 0)
    total = buys + sells
    if total > 0:
        br = buys / total
        if br > 0.75:   score += 12
        elif br > 0.60: score += 6

    return max(0.0, min(score, 100.0))

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 + 2 + 3: GeckoTerminal
# ─────────────────────────────────────────────────────────────────────────────

def fetch_gt_pools(chain: str, sort: str, pages: int,
                   min_mcap: float, max_mcap: float,
                   min_liq: float, min_vol: float, min_age: int) -> list:
    """
    Fetch pools from GeckoTerminal with configurable sort.
    sort options: trending (uses /trending_pools), h6_price_change_percent_desc,
                  h24_volume_usd_desc, h24_tx_count_desc
    """
    results = []
    headers = {"Accept": "application/json;version=20230302"}

    for page in range(1, pages + 1):
        if sort == "trending":
            url = f"{GECKO_URL}/networks/{chain}/trending_pools?page={page}"
        else:
            url = f"{GECKO_URL}/networks/{chain}/pools?page={page}&sort={sort}"

        try:
            r = requests.get(url, timeout=20, headers=headers)
            if r.status_code == 429:
                print(f"    [GT/{chain}/{sort[:8]}] Rate limited p{page} — wait 35s", flush=True)
                time.sleep(35)
                r = requests.get(url, timeout=20, headers=headers)
            if r.status_code == 400:
                print(f"    [GT/{chain}/{sort[:8]}] Bad sort param '{sort}' — skipping", flush=True)
                break
            if r.status_code != 200:
                print(f"    [GT/{chain}/{sort[:8]}] HTTP {r.status_code} p{page} — "
                      f"body: {r.text[:120]}", flush=True)
                break

            data = r.json().get("data", [])
            if not data:
                break

            added = 0
            for pool in data:
                attrs = pool.get("attributes", {})
                mcap  = safe_float(attrs.get("market_cap_usd") or attrs.get("fdv_usd"))
                liq   = safe_float(attrs.get("reserve_in_usd"))
                vol24 = safe_float((attrs.get("volume_usd") or {}).get("h24"))

                if not (min_mcap <= mcap <= max_mcap): continue
                if liq < min_liq:  continue
                if vol24 < min_vol: continue

                created = attrs.get("pool_created_at", "")
                age = days_since(created) if created else -1
                if min_age > 0 and 0 <= age < min_age: continue
                attrs["_age_days"] = age

                pool["_chain"]   = chain
                pool["_mcap"]    = mcap
                pool["_liq"]     = liq
                pool["_vol24h"]  = vol24
                pool["_score"]   = conviction_score(pool)
                pool["_source"]  = f"gt_{sort[:8]}"
                results.append(pool)
                added += 1

            time.sleep(1.2)
        except Exception as e:
            print(f"    [GT/{chain}/{sort[:6]}] p{page} error: {e}", flush=True)
            break

    return results

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 4: DeFiLlama — TVL-growing protocols (pre-token pump signal)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_defillama_base_gems(max_tvl: float = 10_000_000,
                               min_tvl: float = 50_000,
                               min_1d_change: float = 5.0) -> list:
    """
    Base protocols with growing TVL under max_tvl.
    These are the rarest signals: a protocol building TVL before its
    token is widely known. No price needed — TVL growth IS the signal.
    """
    try:
        r = requests.get(f"{LLAMA_URL}/protocols", timeout=25)
        if r.status_code != 200:
            print(f"    [DeFiLlama] HTTP {r.status_code}", flush=True)
            return []

        protocols = r.json()
        gems = []
        for p in protocols:
            # Must be on Base
            if "Base" not in (p.get("chains") or []):
                continue
            tvl = safe_float(p.get("tvl"))
            if not (min_tvl <= tvl <= max_tvl):
                continue
            change_1d = safe_float(p.get("change_1d"))
            if change_1d < min_1d_change:
                continue

            gems.append({
                "_source":     "defillama",
                "_chain":      "base",
                "name":        p.get("name", "?"),
                "symbol":      p.get("symbol", "—"),
                "tvl":         tvl,
                "change_1d":   change_1d,
                "change_7d":   safe_float(p.get("change_7d")),
                "chains":      p.get("chains", []),
                "url":         p.get("url", ""),
                "description": p.get("description", ""),
                "has_token":   bool(p.get("symbol") and p.get("symbol") != "-"),
            })

        gems.sort(key=lambda x: (-x["change_7d"] if x["change_7d"] > 0 else 0,
                                  -x["change_1d"]))
        print(f"    [DeFiLlama] {len(gems)} Base protocols TVL ${min_tvl/1e3:.0f}K–${max_tvl/1e6:.0f}M growing >{min_1d_change}%/day", flush=True)
        return gems

    except Exception as e:
        print(f"    [DeFiLlama] Error: {e}", flush=True)
        return []

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 5: CoinGecko — small-cap Base ecosystem
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cg_small_cap_base(min_mcap: float, max_mcap: float,
                             pages: int = 3) -> list:
    """CoinGecko Base ecosystem tokens under max_mcap."""
    results = []
    seen    = set()
    headers = {"Accept": "application/json"}

    for page in range(1, pages + 1):
        url = (f"{CG_URL}/coins/markets"
               f"?vs_currency=usd&category=base-ecosystem"
               f"&order=volume_desc&per_page=100&page={page}"
               "&sparkline=false&price_change_percentage=24h,7d")
        try:
            r = requests.get(url, timeout=20, headers=headers)
            if r.status_code == 429:
                print(f"    [CG small-cap] Rate limit p{page} — wait 65s", flush=True)
                time.sleep(65)
                r = requests.get(url, timeout=20, headers=headers)
            if r.status_code != 200:
                print(f"    [CG small-cap] HTTP {r.status_code} p{page}", flush=True)
                break

            data = r.json()
            if not data or not isinstance(data, list):
                break

            added = 0
            for c in data:
                if c["id"] in seen:
                    continue
                mcap = safe_float(c.get("market_cap"))
                if not (min_mcap <= mcap <= max_mcap):
                    continue
                c["_source"] = "coingecko"
                c["_mcap"]   = mcap
                c["_chain"]  = "base"
                results.append(c)
                seen.add(c["id"])
                added += 1

            time.sleep(2.2)
        except Exception as e:
            print(f"    [CG small-cap] p{page} error: {e}", flush=True)
            break

    return results

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 6: DexScreener boosts + profiles
# ─────────────────────────────────────────────────────────────────────────────

def fetch_dex_boosts(chain_id: str = "base") -> list:
    results = []
    for endpoint in [
        "https://api.dexscreener.com/token-boosts/latest/v1",
        "https://api.dexscreener.com/token-profiles/latest/v1",
    ]:
        try:
            r = requests.get(endpoint, timeout=15)
            if r.status_code == 200:
                for item in r.json():
                    if item.get("chainId") == chain_id:
                        item["_source"] = "dexscreener"
                        results.append(item)
        except Exception:
            pass
    return results

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 7: GeckoTerminal — brand new pools (< 7 days old, first signs of life)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_gt_new_pools(chain: str, min_liq: float, min_vol: float,
                       max_age_days: int = 7,
                       min_mcap: float = 5_000,
                       max_mcap: float = 1_000_000) -> list:
    """
    Brand-new pools: age ≤ max_age_days, liq + vol already showing.
    These are the EARLIEST possible signal — first 1-7 days of life.
    High risk / highest potential reward. Always cross-check with GoPlus.
    """
    results = []
    headers = {"Accept": "application/json;version=20230302"}
    url = f"{GECKO_URL}/networks/{chain}/new_pools?include=base_token&page=1"
    try:
        r = requests.get(url, timeout=20, headers=headers)
        if r.status_code == 429:
            time.sleep(35)
            r = requests.get(url, timeout=20, headers=headers)
        if r.status_code != 200:
            print(f"    [GT/{chain.upper()}/new] HTTP {r.status_code}", flush=True)
            return []

        for pool in r.json().get("data", []):
            attrs   = pool.get("attributes", {})
            created = attrs.get("pool_created_at", "")
            age     = days_since(created) if created else -1
            if age < 0 or age > max_age_days:
                continue

            mcap  = safe_float(attrs.get("market_cap_usd") or attrs.get("fdv_usd"))
            liq   = safe_float(attrs.get("reserve_in_usd"))
            vol24 = safe_float((attrs.get("volume_usd") or {}).get("h24"))

            if not (min_mcap <= mcap <= max_mcap): continue
            if liq  < min_liq:  continue
            if vol24 < min_vol: continue

            # Minimum activity gate — must have real txns, not just deployed
            txns_h24 = (attrs.get("transactions") or {}).get("h24") or {}
            buys  = int(txns_h24.get("buys", 0)  or 0)
            sells = int(txns_h24.get("sells", 0) or 0)
            if buys + sells < 20:
                continue

            attrs["_age_days"] = age
            pool["_chain"]  = chain
            pool["_mcap"]   = mcap
            pool["_liq"]    = liq
            pool["_vol24h"] = vol24
            pool["_score"]  = conviction_score(pool)
            pool["_source"] = "gt_new"
            results.append(pool)

        print(f"    [GT/{chain.upper()}/new] {len(results)} new pools ≤{max_age_days}d with activity", flush=True)
    except Exception as e:
        print(f"    [GT/{chain.upper()}/new] Error: {e}", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 8: GeckoTerminal — multi-network cross-chain momentum signals
# ─────────────────────────────────────────────────────────────────────────────

# Tokens gaining traction on ETH mainnet / Arbitrum often arrive on Base 2-4 weeks later.
# Tracking these networks gives an advance warning of what's coming to Base.
CROSS_CHAIN_NETWORKS = [
    "eth",        # Ethereum mainnet — protocols launch here first
    "arbitrum",   # Arbitrum — DeFi hub, often mirrors Base trends
    "optimism",   # OP stack sibling of Base
]

def fetch_gt_crosschain(min_mcap: float = 100_000,
                        max_mcap: float = 10_000_000,
                        min_liq: float  = 50_000,
                        pages: int      = 2) -> list:
    """
    Trending tokens on ETH/Arbitrum/Optimism.
    Tokens pumping on ETH but not yet on Base = arbitrage opportunity.
    Also: if a protocol gains TVL on these chains, Base deployment often follows.
    """
    results = []
    headers = {"Accept": "application/json;version=20230302"}

    for network in CROSS_CHAIN_NETWORKS:
        for page in range(1, pages + 1):
            url = f"{GECKO_URL}/networks/{network}/trending_pools?page={page}"
            try:
                r = requests.get(url, timeout=20, headers=headers)
                if r.status_code == 429:
                    time.sleep(35)
                    r = requests.get(url, timeout=20, headers=headers)
                if r.status_code != 200:
                    break

                added = 0
                for pool in r.json().get("data", []):
                    attrs = pool.get("attributes", {})
                    mcap  = safe_float(attrs.get("market_cap_usd") or attrs.get("fdv_usd"))
                    liq   = safe_float(attrs.get("reserve_in_usd"))
                    if not (min_mcap <= mcap <= max_mcap): continue
                    if liq < min_liq: continue
                    pool["_chain"]  = network
                    pool["_mcap"]   = mcap
                    pool["_liq"]    = liq
                    pool["_vol24h"] = safe_float((attrs.get("volume_usd") or {}).get("h24"))
                    pool["_score"]  = conviction_score(pool)
                    pool["_source"] = f"gt_xchain_{network}"
                    results.append(pool)
                    added += 1

                time.sleep(1.0)
            except Exception as e:
                print(f"    [GT/xchain/{network}] p{page} error: {e}", flush=True)
                break

    print(f"    [GT/xchain] {len(results)} cross-chain signals (ETH/ARB/OP trending)", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 9: CoinGecko trending + recently added
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cg_trending() -> list:
    """
    CoinGecko's trending search — 15 most-searched coins last 24h globally.
    These are on everyone's radar RIGHT NOW. Great for confirming narratives.
    """
    results = []
    try:
        r = requests.get(f"{CG_URL}/search/trending", timeout=20,
                         headers={"Accept": "application/json"})
        if r.status_code == 429:
            time.sleep(65)
            r = requests.get(f"{CG_URL}/search/trending", timeout=20)
        if r.status_code != 200:
            print(f"    [CG/trending] HTTP {r.status_code}", flush=True)
            return []

        for item in r.json().get("coins", []):
            coin = item.get("item") or {}
            results.append({
                "_source":  "cg_trending",
                "id":       coin.get("id", ""),
                "name":     coin.get("name", "?"),
                "symbol":   coin.get("symbol", "?").upper(),
                "rank":     coin.get("market_cap_rank"),
                "score":    safe_float(coin.get("score")),
                "thumb":    coin.get("thumb", ""),
                "price_btc": safe_float(coin.get("price_btc")),
            })
        print(f"    [CG/trending] {len(results)} trending coins (global, last 24h)", flush=True)
    except Exception as e:
        print(f"    [CG/trending] Error: {e}", flush=True)
    return results


def fetch_cg_recently_added(min_mcap: float = 5_000,
                             max_mcap: float = 5_000_000) -> list:
    """
    CoinGecko coins added to the platform in the last 7 days.
    Being listed on CoinGecko is a legitimacy signal — scams rarely get listed.
    New + small + growing = high potential.
    """
    results = []
    try:
        r = requests.get(f"{CG_URL}/coins/list/new", timeout=20,
                         headers={"Accept": "application/json"})
        if r.status_code == 429:
            time.sleep(65)
            r = requests.get(f"{CG_URL}/coins/list/new", timeout=20)
        if r.status_code != 200:
            print(f"    [CG/new] HTTP {r.status_code}", flush=True)
            return []

        for coin in r.json():
            # Get basic market data for each new coin (batch of 50 limit)
            results.append({
                "_source": "cg_new_listing",
                "id":      coin.get("id", ""),
                "name":    coin.get("name", "?"),
                "symbol":  coin.get("symbol", "?").upper(),
                "activated_at": coin.get("activated_at"),
            })
        print(f"    [CG/new] {len(results)} recently listed tokens", flush=True)
    except Exception as e:
        print(f"    [CG/new] Error: {e}", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 10: DexScreener — global trending + latest boosted (all chains)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_dex_global_trending() -> list:
    """
    DexScreener's top boosted tokens across ALL chains.
    Tokens that teams pay to boost are serious projects — cheap to verify.
    Cross-chain boosts on Base = high confidence signal.
    """
    results = []
    for endpoint in [
        f"{DEX_URL}/token-boosts/top/v1",
        f"{DEX_URL}/token-boosts/latest/v1",
    ]:
        try:
            r = requests.get(endpoint, timeout=15)
            if r.status_code == 200:
                for item in r.json():
                    item["_source"]  = "dex_boost_global"
                    item["_chain"]   = item.get("chainId", "unknown")
                    results.append(item)
        except Exception as e:
            print(f"    [DEX/global] Error {endpoint}: {e}", flush=True)

    print(f"    [DEX/global boosts] {len(results)} total across all chains", flush=True)
    return results


def fetch_dex_token_search_extended(terms: list, chains: list) -> list:
    """
    DexScreener text search for additional narrative-aligned terms.
    Catches tokens not in GeckoTerminal trending that match our narrative buckets.
    """
    results = []
    NARRATIVE_TERMS = [
        "ai", "agent", "depin", "rwa", "yield", "vault", "lend",
        "borrow", "perp", "option", "oracle", "bridge", "stake",
        "launch", "pad", "factory", "protocol", "finance",
    ]
    # Only search terms not already in the main config list
    extra_terms = [t for t in NARRATIVE_TERMS if t not in (terms or [])]

    for term in extra_terms[:8]:    # cap at 8 to limit API calls
        try:
            r = requests.get(f"{DEX_URL}/latest/dex/search?q={term}", timeout=15)
            if r.status_code != 200:
                continue
            for pair in r.json().get("pairs", []):
                if pair.get("chainId") not in chains:
                    continue
                mcap = safe_float(pair.get("marketCap") or pair.get("fdv") or 0)
                liq  = safe_float((pair.get("liquidity") or {}).get("usd") or 0)
                if mcap < 5_000 or mcap > 2_000_000: continue
                if liq  < 1_000: continue
                base_tok = pair.get("baseToken") or {}
                results.append({
                    "_source":  f"dex_search_{term}",
                    "_chain":   pair.get("chainId"),
                    "_mcap":    mcap,
                    "_liq":     liq,
                    "_vol24h":  safe_float((pair.get("volume") or {}).get("h24")),
                    "_score":   0,  # DexScreener pairs not pool format, skip GT scorer
                    "address":  base_tok.get("address", ""),
                    "name":     base_tok.get("name", "?"),
                    "symbol":   base_tok.get("symbol", "?"),
                    "url":      pair.get("url", ""),
                })
            time.sleep(0.5)
        except Exception:
            pass

    print(f"    [DEX/extended search] {len(results)} narrative-aligned tokens", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 11: DeFiLlama — yield opportunities growing (money flowing into DeFi)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_defillama_yields(min_apy: float   = 20.0,
                           max_tvl: float   = 5_000_000,
                           min_tvl: float   = 100_000,
                           chains: list     = None) -> list:
    """
    Yield pools on Base/Solana with high APY and small TVL.
    High APY + growing TVL = capital is flowing in.
    Protocol has a token or is about to launch one.
    Source: https://yields.llama.fi/pools (no rate limit, no key)
    """
    target_chains = set(c.capitalize() for c in (chains or ["Base", "Solana"]))
    results = []
    try:
        r = requests.get("https://yields.llama.fi/pools", timeout=30)
        if r.status_code != 200:
            print(f"    [DeFiLlama/yield] HTTP {r.status_code}", flush=True)
            return []

        pools = r.json().get("data", [])
        for p in pools:
            if p.get("chain") not in target_chains:
                continue
            tvl = safe_float(p.get("tvlUsd"))
            apy = safe_float(p.get("apy"))
            if not (min_tvl <= tvl <= max_tvl): continue
            if apy < min_apy: continue

            results.append({
                "_source":   "defillama_yield",
                "project":   p.get("project", "?"),
                "symbol":    p.get("symbol",  "?"),
                "chain":     p.get("chain",   "?").lower(),
                "tvl":       tvl,
                "apy":       apy,
                "apy_base":  safe_float(p.get("apyBase")),
                "apy_reward": safe_float(p.get("apyReward")),
                "pool_id":   p.get("pool",    ""),
                "url":       p.get("url",     ""),
            })

        results.sort(key=lambda x: -x["apy"])
        results = results[:20]   # top 20 highest APY small pools
        print(f"    [DeFiLlama/yield] {len(results)} high-APY small pools on Base/Solana", flush=True)
    except Exception as e:
        print(f"    [DeFiLlama/yield] Error: {e}", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 12: DeFiLlama — protocol revenue (real money generating protocols)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_defillama_revenue(chains: list = None,
                             max_mcap_mult: float = 20.0) -> list:
    """
    Protocols generating real revenue on Base/Solana.
    Revenue / MCap < 0.05 = severely undervalued relative to earnings.
    This is the FUNDAMENTALS signal — revenue is undeniable.
    Source: https://api.llama.fi/overview/fees (no key, no rate limit)
    """
    target_chains = set(c.capitalize() for c in (chains or ["Base", "Solana"]))
    results = []
    try:
        r = requests.get(f"{LLAMA_URL}/overview/fees?excludeTotalDataChart=true"
                          "&excludeTotalDataChartBreakdown=true", timeout=30)
        if r.status_code != 200:
            print(f"    [DeFiLlama/fees] HTTP {r.status_code}", flush=True)
            return []

        for p in r.json().get("protocols", []):
            # Must be active on target chains
            p_chains = set(p.get("chains") or [])
            if not p_chains.intersection(target_chains):
                continue
            daily_rev = safe_float(p.get("total24h"))
            if daily_rev < 1_000:   # min $1K/day real revenue
                continue
            results.append({
                "_source":       "defillama_revenue",
                "name":          p.get("name",   "?"),
                "displayName":   p.get("displayName", "?"),
                "chains":        list(p_chains),
                "daily_revenue": daily_rev,
                "weekly_revenue": safe_float(p.get("total7d")),
                "monthly_revenue": safe_float(p.get("total30d")),
                "revenue_change_1d": safe_float(p.get("change_1d")),
                "category":      p.get("category", ""),
                "url":           p.get("url", ""),
            })

        results.sort(key=lambda x: -x["daily_revenue"])
        results = results[:20]
        print(f"    [DeFiLlama/revenue] {len(results)} revenue-generating protocols on Base/Solana", flush=True)
    except Exception as e:
        print(f"    [DeFiLlama/revenue] Error: {e}", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 13: Birdeye — Solana + Base token list (optional API key)
# Set BIRDEYE_API_KEY env var. Free tier: 1M credits/month.
# https://docs.birdeye.so/
# ─────────────────────────────────────────────────────────────────────────────

def fetch_birdeye_trending(chain: str = "base",
                           min_mcap: float = 10_000,
                           max_mcap: float = 1_000_000,
                           api_key: str    = "") -> list:
    """
    Birdeye's token list sorted by 24H volume change — catches early momentum.
    Requires BIRDEYE_API_KEY env var. Skips gracefully if not set.
    Free tier: 1M credits/month (enough for daily scans).
    """
    if not api_key:
        return []

    birdeye_chain = {"base": "base", "solana": "solana"}.get(chain, "base")
    results = []
    headers = {
        "accept":    "application/json",
        "x-api-key": api_key,
        "x-chain":   birdeye_chain,
    }

    try:
        url = (f"{BIRDEYE_URL}/defi/tokenlist"
               f"?sort_by=v24hChangePercent&sort_type=desc"
               f"&offset=0&limit=50&min_liquidity=1000")
        r = requests.get(url, timeout=20, headers=headers)
        if r.status_code == 401:
            print(f"    [Birdeye] Invalid API key — skipping", flush=True)
            return []
        if r.status_code != 200:
            print(f"    [Birdeye] HTTP {r.status_code}", flush=True)
            return []

        for tok in r.json().get("data", {}).get("tokens", []):
            mcap = safe_float(tok.get("mc") or tok.get("marketCap"))
            liq  = safe_float(tok.get("liquidity") or tok.get("realLiquidity"))
            if not (min_mcap <= mcap <= max_mcap): continue
            if liq < 1_000: continue

            results.append({
                "_source":  "birdeye",
                "_chain":   chain,
                "_mcap":    mcap,
                "_liq":     liq,
                "_vol24h":  safe_float(tok.get("v24hUSD")),
                "_score":   0,
                "name":     tok.get("name",   "?"),
                "symbol":   tok.get("symbol", "?").upper(),
                "address":  tok.get("address", ""),
                "v24h_change_pct": safe_float(tok.get("v24hChangePercent")),
                "price_change_24h": safe_float(tok.get("priceChange24hPercent")),
            })

        print(f"    [Birdeye/{chain}] {len(results)} tokens by vol change (API key set)", flush=True)
    except Exception as e:
        print(f"    [Birdeye/{chain}] Error: {e}", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 14: GoPlus Security — token risk check (free, no key required)
# Run post-discovery to filter out honeypots and high-risk tokens
# https://gopluslabs.io/
# ─────────────────────────────────────────────────────────────────────────────

# Base chain ID = 8453 on GoPlus
GOPLUS_CHAIN_IDS = {"base": "8453", "solana": "solana"}

def check_goplus_risk(addresses: list, chain: str = "base") -> dict:
    """
    Batch risk check for discovered token addresses.
    Returns dict: address → risk_info with fields:
        is_honeypot, is_mintable, has_blacklist, sell_tax, buy_tax, holder_count
    Only called for high-score tokens to save API budget.
    Free tier: no key required, but rate-limited.
    """
    chain_id = GOPLUS_CHAIN_IDS.get(chain)
    if not chain_id or not addresses:
        return {}

    results = {}
    # GoPlus takes up to 20 addresses per call
    for i in range(0, len(addresses), 20):
        batch = addresses[i:i+20]
        addr_str = ",".join(batch)
        try:
            r = requests.get(
                f"{GOPLUS_URL}/token_security/{chain_id}",
                params={"contract_addresses": addr_str},
                timeout=20,
            )
            if r.status_code == 429:
                time.sleep(10)
                continue
            if r.status_code != 200:
                continue

            for addr, info in r.json().get("result", {}).items():
                results[addr.lower()] = {
                    "is_honeypot":   info.get("is_honeypot") == "1",
                    "is_mintable":   info.get("is_mintable") == "1",
                    "has_blacklist": info.get("is_blacklisted") == "1",
                    "sell_tax":      safe_float(info.get("sell_tax")) * 100,
                    "buy_tax":       safe_float(info.get("buy_tax")) * 100,
                    "holder_count":  int(info.get("holder_count") or 0),
                    "open_source":   info.get("is_open_source") == "1",
                    "lp_locked":     info.get("lp_total_supply", "0") != "0",
                }
            time.sleep(0.5)
        except Exception as e:
            print(f"    [GoPlus] Error: {e}", flush=True)
            break

    print(f"    [GoPlus] Risk checked {len(results)} addresses", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication across sources
# ─────────────────────────────────────────────────────────────────────────────

def dedup_pools(pools: list) -> list:
    """Dedup by pool address. Higher-scoring source wins."""
    seen: dict = {}
    for pool in sorted(pools, key=lambda p: p.get("_score", 0), reverse=True):
        attrs   = pool.get("attributes", {})
        address = attrs.get("address", "") or pool.get("id", id(pool))
        chain   = pool.get("_chain", "base")
        key     = f"{chain}_{address}"
        if key not in seen:
            seen[key] = pool
    return list(seen.values())

# ─────────────────────────────────────────────────────────────────────────────
# Watchlist management
# ─────────────────────────────────────────────────────────────────────────────

def update_watchlist(wl: dict, pools: list,
                     defillama: list, today: str) -> dict:
    new_entries  = []
    repeat_hits  = []
    graduates    = []

    # ── DEX pools ──────────────────────────────────────────────────────────
    for pool in pools:
        attrs = pool.get("attributes", {})
        chain = pool.get("_chain", "base")
        mcap  = pool.get("_mcap",  0)
        score = pool.get("_score", 0)
        liq   = pool.get("_liq",   0)
        vol   = pool.get("_vol24h",0)

        address = attrs.get("address", "") or pool.get("id", "")
        key     = f"{chain}_{address}"
        pname   = attrs.get("name", "?")
        sym     = pname.split("/")[0].strip() if "/" in pname else pname
        meme    = is_meme(pname, sym)

        pc  = attrs.get("price_change_percentage", {})
        h6  = safe_float(pc.get("h6"))
        h24 = safe_float(pc.get("h24"))
        age = attrs.get("_age_days", -1)
        src = pool.get("_source", "gt")

        snap = {"date": today, "mcap": mcap, "h6": h6, "h24": h24,
                "liq": liq, "score": score, "src": src}

        existing = wl["tokens"].get(key)
        if existing is None:
            entry = {
                "name":          sym,
                "pool_name":     pname,
                "chain":         chain,
                "address":       address,
                "first_seen":    today,
                "last_seen":     today,
                "appearances":   1,
                "history":       [snap],
                "max_score":     score,
                "liq":           liq,
                "age_days":      age,
                "is_meme":       meme,
                "sources_seen":  [src],
                "status":        "watching",
                "graduated_at":  None,
                "graduated_mcap": None,
            }
            wl["tokens"][key] = entry
            new_entries.append(entry)
        else:
            existing["last_seen"]    = today
            existing["appearances"] += 1
            existing["liq"]          = liq
            existing["history"].append(snap)
            existing["history"]      = existing["history"][-15:]
            if score > existing.get("max_score", 0):
                existing["max_score"] = score
            srcs = existing.get("sources_seen", [])
            if src not in srcs:
                srcs.append(src)
            existing["sources_seen"] = srcs

            if mcap >= 1_000_000 and existing["status"] == "watching":
                existing["status"]        = "graduated"
                existing["graduated_at"]  = today
                existing["graduated_mcap"] = mcap
                graduates.append(existing)
            else:
                repeat_hits.append((existing, score))

    # ── DeFiLlama protocols ─────────────────────────────────────────────────
    llama_section = wl.setdefault("defi_protocols", {})
    for p in defillama:
        key  = f"llama_{p['name'].lower().replace(' ', '_')}"
        snap = {"date": today, "tvl": p["tvl"],
                "change_1d": p["change_1d"], "change_7d": p["change_7d"]}
        if key not in llama_section:
            llama_section[key] = {
                "name":       p["name"],
                "symbol":     p["symbol"],
                "has_token":  p["has_token"],
                "first_seen": today,
                "last_seen":  today,
                "appearances": 1,
                "history":    [snap],
                "max_1d":     p["change_1d"],
                "url":        p.get("url", ""),
            }
        else:
            e = llama_section[key]
            e["last_seen"]    = today
            e["appearances"] += 1
            e["history"].append(snap)
            e["history"]      = e["history"][-15:]
            if p["change_1d"] > e.get("max_1d", 0):
                e["max_1d"] = p["change_1d"]

    return {
        "new":       new_entries,
        "repeats":   sorted(repeat_hits, key=lambda x: (-x[0]["appearances"], -x[1])),
        "graduates": graduates,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_report(pools: list, cg_coins: list, defillama: list,
                 dex_boosts: list, changes: dict, wl: dict,
                 run_date: str, chains: list,
                 min_mcap: float, max_mcap: float,
                 new_pools: list = None,
                 cg_trending: list = None,
                 cg_new: list = None,
                 llama_yields: list = None,
                 llama_revenue: list = None,
                 crosschain: list = None,
                 goplus_risks: dict = None) -> str:

    new_pools     = new_pools     or []
    cg_trending   = cg_trending   or []
    cg_new        = cg_new        or []
    llama_yields  = llama_yields  or []
    llama_revenue = llama_revenue or []
    crosschain    = crosschain    or []
    goplus_risks  = goplus_risks  or {}

    lines = []
    lines.append("# 🔬 Micro-Cap Radar — Full Multi-Source Report")
    lines.append(
        f"**Date:** {run_date}  |  **Chains:** {', '.join(c.upper() for c in chains)}  |  "
        f"**MCap:** ${min_mcap/1e3:.0f}K – ${max_mcap/1e6:.1f}M\n"
    )
    lines.append(
        "**Sources:** GeckoTerminal (trending+vol+tx+new) · DeFiLlama (TVL+yield+revenue) · "
        "CoinGecko (categories+trending+new) · DexScreener (boosts+search) · "
        "Cross-chain (ETH/ARB/OP) · GoPlus (security)\n"
    )
    lines.append("---\n")

    watching  = sum(1 for t in wl["tokens"].values() if t["status"] == "watching")
    graduated = sum(1 for t in wl["tokens"].values() if t["status"] == "graduated")
    llama_ct  = len(wl.get("defi_protocols", {}))
    lines.append(
        f"*Watchlist: **{len(wl['tokens'])}** DEX tokens ({watching} active, {graduated} graduated) "
        f"+ **{llama_ct}** DeFiLlama protocols tracked*\n"
    )

    # ── GRADUATES ────────────────────────────────────────────────────────────
    if changes["graduates"]:
        lines.append("## 🏆 GRADUATED — Caught Early, Now Above $1M!\n")
        for t in changes["graduates"]:
            hist   = t.get("history", [{}])
            entry  = hist[0].get("mcap", 0)
            now    = t.get("graduated_mcap", 0)
            mult   = (now / entry) if entry > 0 else 0
            lines.append(
                f"**{t['name']}** on {t['chain'].upper()} | "
                f"First seen {t['first_seen']} @ ${entry:,.0f} → ${now:,.0f} "
                f"(**{mult:.1f}x**) | {t['appearances']} appearances"
            )
        lines.append("")

    # ── PERSISTENT DEX SIGNALS ───────────────────────────────────────────────
    lines.append("## 🔁 Persistent DEX Signals (2+ Scan Appearances)\n")
    lines.append("*The real edge — quietly building while nobody notices*\n")

    persistent = sorted(
        [(k, v) for k, v in wl["tokens"].items()
         if v["appearances"] >= 2 and v["status"] == "watching"],
        key=lambda x: (-x[1]["appearances"], -x[1]["max_score"])
    )

    if persistent:
        for key, t in persistent[:20]:
            hist  = t.get("history", [])
            m0    = hist[0]["mcap"]  if hist else 0
            m1    = hist[-1]["mcap"] if hist else 0
            pct   = ((m1 - m0) / m0 * 100) if m0 > 0 else 0
            arrow = "📈" if pct > 5 else ("📉" if pct < -5 else "➡️")
            fire  = "🔥" * min(t["appearances"], 5)
            meme  = " ⚠️meme" if t.get("is_meme") else ""
            srcs  = "+".join(t.get("sources_seen", []))[:20]
            lines.append(
                f"{fire} **{t['name']}** | {t['chain'].upper()} | "
                f"${m1:,.0f} MCap | {t['appearances']}× | "
                f"{arrow} {pct:+.0f}% from entry | {t['max_score']}/100{meme} | [{srcs}]"
            )
    else:
        lines.append("*First run — baseline being established. Real signals appear from scan 2 onwards.*")
    lines.append("")

    # ── DEFILLAMA TVL SIGNAL ─────────────────────────────────────────────────
    lines.append("## 📊 DeFiLlama — Base Protocols Growing TVL (Pre-Token Pump Signal)\n")
    lines.append("*TVL growing + small size + no famous token = someone is moving money in quietly*\n")

    if defillama:
        # Split: has token vs no token yet
        with_token    = [p for p in defillama if p["has_token"]][:8]
        without_token = [p for p in defillama if not p["has_token"]][:5]

        if with_token:
            lines.append("### Has tradeable token")
            for p in with_token:
                sym = p["symbol"] if p["symbol"] and p["symbol"] != "-" else "?"
                # check if persistent in llama watchlist
                llama_key = f"llama_{p['name'].lower().replace(' ', '_')}"
                llama_e   = wl.get("defi_protocols", {}).get(llama_key)
                repeat    = f" 🔁×{llama_e['appearances']}" if llama_e and llama_e["appearances"] > 1 else ""
                lines.append(
                    f"- **{p['name']} ({sym})** | TVL ${p['tvl']:,.0f} | "
                    f"1D {p['change_1d']:+.1f}% | 7D {p['change_7d']:+.1f}%{repeat}"
                )
            lines.append("")

        if without_token:
            lines.append("### No token yet — watch for launch")
            for p in without_token:
                lines.append(
                    f"- **{p['name']}** | TVL ${p['tvl']:,.0f} | "
                    f"1D {p['change_1d']:+.1f}% | 7D {p['change_7d']:+.1f}% | {p.get('url','')}"
                )
            lines.append("")
    else:
        lines.append("*No DeFiLlama signals this scan*\n")

    # ── PERSISTENT DEFILLAMA PROTOCOLS ───────────────────────────────────────
    llama_persistent = sorted(
        [(k, v) for k, v in wl.get("defi_protocols", {}).items()
         if v["appearances"] >= 2],
        key=lambda x: (-x[1]["appearances"], -x[1].get("max_1d", 0))
    )
    if llama_persistent:
        lines.append("### Persistent DeFiLlama Protocols (2+ scans)\n")
        for key, p in llama_persistent[:8]:
            hist   = p.get("history", [{}])
            tvl0   = hist[0].get("tvl", 0)
            tvl1   = hist[-1].get("tvl", 0)
            growth = ((tvl1 - tvl0) / tvl0 * 100) if tvl0 > 0 else 0
            tok    = f" ({p['symbol']})" if p.get("has_token") else " [no token yet]"
            lines.append(
                f"🔥×{p['appearances']} **{p['name']}{tok}** | "
                f"TVL ${tvl1:,.0f} ({growth:+.0f}% since first scan)"
            )
        lines.append("")

    # ── NEW DEX ENTRIES ───────────────────────────────────────────────────────
    new_sorted = sorted(changes["new"], key=lambda x: x.get("max_score", 0), reverse=True)
    base_new   = [t for t in new_sorted if t["chain"] == "base"]
    sol_new    = [t for t in new_sorted if t["chain"] == "solana"]

    if base_new:
        lines.append("## 🆕 New Base Entries This Scan\n")
        for t in base_new[:12]:
            h   = t.get("history", [{}])
            h6  = h[-1].get("h6",   0)
            h24 = h[-1].get("h24",  0)
            mc  = h[-1].get("mcap", 0)
            sc  = t.get("max_score", 0)
            liq = t.get("liq", 0)
            age = t.get("age_days", -1)
            meme = " ⚠️meme" if t.get("is_meme") else ""
            src  = h[-1].get("src", "")
            lines.append(
                f"- **{t['name']}** — ${mc:,.0f} MCap | Liq ${liq:,.0f} | "
                f"6H {h6:+.1f}% | 24H {h24:+.1f}% | {age}d | {sc}/100{meme} [{src}]"
            )
        lines.append("")

    if sol_new:
        lines.append("## 🆕 New Solana Entries (watch only — bot covers Base)\n")
        for t in sol_new[:8]:
            h   = t.get("history", [{}])
            h6  = h[-1].get("h6",  0)
            h24 = h[-1].get("h24", 0)
            mc  = h[-1].get("mcap",0)
            sc  = t.get("max_score", 0)
            lines.append(
                f"- **{t['name']}** — ${mc:,.0f} MCap | 6H {h6:+.1f}% | 24H {h24:+.1f}% | {sc}/100"
            )
        lines.append("")

    # ── CoinGecko small-cap findings ─────────────────────────────────────────
    if cg_coins:
        lines.append("## 🦎 CoinGecko Base Small-Caps (Not in DEX Radar)\n")
        cg_sorted = sorted(cg_coins, key=lambda c: safe_float(c.get("price_change_percentage_24h")), reverse=True)
        for c in cg_sorted[:10]:
            name  = c.get("name", "?")
            sym   = c.get("symbol", "?").upper()
            mcap  = c.get("market_cap", 0)
            c24h  = safe_float(c.get("price_change_percentage_24h"))
            c7d   = safe_float(c.get("price_change_percentage_7d_in_currency"))
            vol   = c.get("total_volume", 0)
            lines.append(
                f"- **{name} ({sym})** — ${mcap:,.0f} MCap | "
                f"24H {c24h:+.1f}% | 7D {c7d:+.1f}% | Vol ${vol:,.0f}"
            )
        lines.append("")

    # ── TOP 20 DEX POOLS THIS SCAN ────────────────────────────────────────────
    lines.append("## 📋 Top 20 DEX Pools This Scan (by conviction score)\n")

    for chain_filter in ["base", "solana"]:
        chain_pools = sorted(
            [p for p in pools if p.get("_chain") == chain_filter],
            key=lambda p: p.get("_score", 0), reverse=True
        )[:10]
        if not chain_pools:
            continue

        suffix = " (bot trades this)" if chain_filter == "base" else " (watch only)"
        lines.append(f"### {chain_filter.upper()}{suffix}")

        for i, p in enumerate(chain_pools, 1):
            attrs  = p.get("attributes", {})
            pname  = attrs.get("name", "?")
            mcap   = p.get("_mcap",  0)
            liq    = p.get("_liq",   0)
            vol    = p.get("_vol24h",0)
            age    = attrs.get("_age_days", -1)
            score  = p.get("_score", 0)
            src    = p.get("_source", "?")[:8]
            pc     = attrs.get("price_change_percentage", {})
            h6     = safe_float(pc.get("h6"))
            h24    = safe_float(pc.get("h24"))
            txns   = (attrs.get("transactions") or {})
            t_h1   = (txns.get("h1") or {})
            bs     = f"{int(t_h1.get('buys',0) or 0)}B/{int(t_h1.get('sells',0) or 0)}S"

            address = attrs.get("address", "") or p.get("id", "")
            wl_e    = wl["tokens"].get(f"{chain_filter}_{address}")
            repeat  = f" 🔁×{wl_e['appearances']}" if wl_e and wl_e["appearances"] > 1 else ""

            lines.append(
                f"{i:>2}. **{pname}** — ${mcap:,.0f} MCap | Liq ${liq:,.0f} | "
                f"6H {h6:+.1f}% | 24H {h24:+.1f}% | Vol ${vol:,.0f} | "
                f"{age}d | {bs} | {score}/100{repeat} [{src}]"
            )
        lines.append("")

    # ── TINY GEMS < $100K ──────────────────────────────────────────────────────
    tiny = sorted(
        [p for p in pools if p.get("_mcap", 0) < 100_000 and p.get("_score", 0) >= 35],
        key=lambda p: p["_score"], reverse=True
    )[:6]
    if tiny:
        lines.append("## 💎 Tiny Gems Under $100K MCap (Score ≥35)\n")
        lines.append("*Maximum risk / maximum reward. Always rug-check first.*\n")
        for p in tiny:
            attrs = p.get("attributes", {})
            pname = attrs.get("name", "?")
            mcap  = p.get("_mcap", 0)
            liq   = p.get("_liq",  0)
            score = p.get("_score",0)
            pc    = attrs.get("price_change_percentage", {})
            h6    = safe_float(pc.get("h6"))
            lines.append(
                f"- **{pname}** | {p.get('_chain','').upper()} | "
                f"${mcap:,.0f} MCap | Liq ${liq:,.0f} | 6H {h6:+.1f}% | {score}/100"
            )
        lines.append("")

    # ── DEXSCREENER BOOSTS ────────────────────────────────────────────────────
    if dex_boosts:
        lines.append("## 📡 DexScreener Paid Boosts / Profiles (Base)\n")
        lines.append("*Someone paid to promote these = conviction + capital backing them*\n")
        for b in dex_boosts[:8]:
            name   = b.get("description", "—") or b.get("tokenAddress", "?")
            url    = b.get("url", "")
            amount = safe_float(b.get("amount"))
            tag    = f"${amount:,.0f} boost" if amount > 0 else "profile"
            lines.append(f"- [{tag}] **{name}** — {url}")
        lines.append("")

    # ── BRAND NEW POOLS (< 7 days old) ────────────────────────────────────────
    if new_pools:
        lines.append("## 🆕🔥 Brand New Pools (< 7 Days Old — Earliest Entry)\n")
        lines.append("*Highest risk + highest reward. Always run GoPlus check before buying.*\n")
        for p in sorted(new_pools, key=lambda x: x.get("_score", 0), reverse=True)[:8]:
            attrs = p.get("attributes", {})
            pname = attrs.get("name", "?")
            mcap  = p.get("_mcap",  0)
            liq   = p.get("_liq",   0)
            age   = attrs.get("_age_days", -1)
            score = p.get("_score", 0)
            pc    = attrs.get("price_change_percentage", {})
            h6    = safe_float(pc.get("h6"))
            chain = p.get("_chain", "base").upper()
            addr  = attrs.get("address", "")

            # GoPlus risk flag
            risk_info = goplus_risks.get(addr.lower())
            risk_txt  = ""
            if risk_info:
                flags = []
                if risk_info.get("is_honeypot"):  flags.append("🚨 HONEYPOT")
                if risk_info.get("is_mintable"):  flags.append("⚠️ mintable")
                if risk_info.get("sell_tax", 0) > 10: flags.append(f"⚠️ {risk_info['sell_tax']:.0f}% sell tax")
                risk_txt = "  " + " ".join(flags) if flags else "  ✅ clean"

            lines.append(
                f"- **{pname}** | {chain} | ${mcap:,.0f} MCap | Liq ${liq:,.0f} | "
                f"6H {h6:+.1f}% | {age}d old | {score}/100{risk_txt}"
            )
        lines.append("")

    # ── GOPLUS RISK SUMMARY ────────────────────────────────────────────────────
    if goplus_risks:
        honeypots = [a for a, r in goplus_risks.items() if r.get("is_honeypot")]
        if honeypots:
            lines.append(f"## 🚨 GoPlus Honeypot Alerts ({len(honeypots)} detected)\n")
            lines.append("*These addresses failed the honeypot check. Do NOT buy.*\n")
            for addr in honeypots[:5]:
                lines.append(f"- `{addr}`")
            lines.append("")

    # ── CROSS-CHAIN MOMENTUM (ETH/ARB/OP) ─────────────────────────────────────
    if crosschain:
        lines.append("## 🌐 Cross-Chain Momentum (ETH / Arbitrum / Optimism)\n")
        lines.append("*Tokens trending on mainnet/Arbitrum often arrive on Base 2-4 weeks later.*\n")
        xchain_sorted = sorted(crosschain, key=lambda p: p.get("_score", 0), reverse=True)[:8]
        for p in xchain_sorted:
            attrs = p.get("attributes", {})
            pname = attrs.get("name", "?")
            mcap  = p.get("_mcap", 0)
            liq   = p.get("_liq",  0)
            chain = p.get("_chain", "?").upper()
            pc    = attrs.get("price_change_percentage", {})
            h6    = safe_float(pc.get("h6"))
            h24   = safe_float(pc.get("h24"))
            score = p.get("_score", 0)
            lines.append(
                f"- **{pname}** | {chain} | ${mcap:,.0f} MCap | Liq ${liq:,.0f} | "
                f"6H {h6:+.1f}% | 24H {h24:+.1f}% | {score}/100"
            )
        lines.append("")

    # ── COINGECKO TRENDING (global) ────────────────────────────────────────────
    if cg_trending:
        lines.append("## 🔥 CoinGecko Trending (Global — Top 15 Most Searched)\n")
        lines.append("*These are what everyone is searching right now. Narrative confirmation.*\n")
        for i, coin in enumerate(cg_trending[:10], 1):
            rank = coin.get("rank", "?")
            lines.append(
                f"{i:>2}. **{coin['name']} ({coin['symbol']})** | "
                f"CMC Rank {rank}"
            )
        lines.append("")

    # ── COINGECKO RECENTLY ADDED ───────────────────────────────────────────────
    if cg_new:
        lines.append("## 🆕 CoinGecko Recently Listed (Last 7 Days)\n")
        lines.append("*Being listed on CoinGecko = passed basic legitimacy checks. Watch these.*\n")
        for coin in cg_new[:10]:
            ts = coin.get("activated_at", "")
            ts_txt = ts[:10] if ts else "?"
            lines.append(f"- **{coin['name']} ({coin['symbol']})** — listed {ts_txt} | id: `{coin['id']}`")
        lines.append("")

    # ── DEFILLAMA YIELD SIGNALS ────────────────────────────────────────────────
    if llama_yields:
        lines.append("## 💰 DeFiLlama — High-APY Small Yield Pools (Base + Solana)\n")
        lines.append("*High APY + small TVL = protocol needs liquidity badly = token incentive coming.*\n")
        for p in llama_yields[:10]:
            reward_note = f" (reward: {p['apy_reward']:.0f}%)" if p.get("apy_reward", 0) > 0 else ""
            lines.append(
                f"- **{p['project']} — {p['symbol']}** | {p['chain'].upper()} | "
                f"TVL ${p['tvl']:,.0f} | APY {p['apy']:.0f}%{reward_note}"
            )
        lines.append("")

    # ── DEFILLAMA REVENUE LEADERS ──────────────────────────────────────────────
    if llama_revenue:
        lines.append("## 📈 DeFiLlama — Real Revenue Generators (Base + Solana)\n")
        lines.append("*Protocols generating $1K+/day in fees = real users, real product.*\n")
        for p in llama_revenue[:10]:
            chg  = p.get("revenue_change_1d", 0)
            sign = "+" if chg >= 0 else ""
            lines.append(
                f"- **{p['displayName']}** | {', '.join(p['chains'][:2])} | "
                f"Daily ${p['daily_revenue']:,.0f} | 7D ${p['weekly_revenue']:,.0f} | "
                f"1D chg {sign}{chg:.0f}%"
            )
        lines.append("")

    # ── HOW TO USE ─────────────────────────────────────────────────────────────
    lines.append("## 🧭 How to Read This Report\n")
    lines.append("| Signal | What it means |")
    lines.append("|--------|---------------|")
    lines.append("| 🔁 2× appearances | Someone is steadily buying — not a flash pump |")
    lines.append("| 🔁🔁🔁 3×+ | Very high conviction. Size in. |")
    lines.append("| DeFiLlama TVL growing + no famous token | Rarest signal. Pre-pump. Watch for token launch. |")
    lines.append("| Score >60 + liq >$50K | Safest micro-cap entry |")
    lines.append("| Score >60 + liq <$5K | Too risky — can rug instantly |")
    lines.append("| [gt_trending] + [gt_h6] same token | Two independent signals = high confidence |")
    lines.append("| 🏆 Graduate | Token was on watchlist and hit $1M — system worked |")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by micro_scanner.py — {run_date}*")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-source micro-cap under-the-radar tracker")
    parser.add_argument("--min-mcap",   type=float, default=10_000)
    parser.add_argument("--max-mcap",   type=float, default=1_000_000)
    parser.add_argument("--min-liq",    type=float, default=1_000)
    parser.add_argument("--min-vol",    type=float, default=1_000)
    parser.add_argument("--min-age",    type=int,   default=5,
                        help="Min pool age days (default 5; 0 = include brand new)")
    parser.add_argument("--chain",      type=str,   default="base,solana")
    parser.add_argument("--pages",      type=int,   default=4,
                        help="GeckoTerminal pages per chain per sort (default 4)")
    parser.add_argument("--llama-max-tvl", type=float, default=10_000_000,
                        help="DeFiLlama max TVL to include (default $10M)")
    parser.add_argument("--output",     type=str,   default=None)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    import os
    birdeye_key = os.environ.get("BIRDEYE_API_KEY", "")

    chains   = [c.strip().lower() for c in args.chain.split(",")]
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sep = "=" * 64

    source_count = 14  # total data sources
    print(f"\n{sep}")
    print(f"  🔬 Micro-Cap Radar  {run_date}")
    print(f"  Chains: {', '.join(c.upper() for c in chains)}")
    print(f"  MCap ${args.min_mcap/1e3:.0f}K–${args.max_mcap/1e6:.1f}M | "
          f"Min liq ${args.min_liq:,.0f} | Min vol ${args.min_vol:,.0f}")
    print(f"  Sources: {source_count} data sources active")
    if birdeye_key:
        print(f"  Birdeye: API key found ✓")
    else:
        print(f"  Birdeye: set BIRDEYE_API_KEY env var to enable")
    print(f"{sep}\n")

    wl = load_watchlist()
    print(f"  Watchlist loaded: {len(wl['tokens'])} DEX tokens, "
          f"{len(wl.get('defi_protocols', {}))} DeFiLlama protocols\n")

    # ── SOURCE 1-3: GeckoTerminal standard sorts ────────────────────────────
    all_pools  = []
    new_pools  = []

    for chain in chains:
        if chain not in {"base", "solana"}:
            print(f"  [SKIP] Unknown chain: {chain}")
            continue

        print(f"  [GT/{chain.upper()}] trending pools...", flush=True)
        p1 = fetch_gt_pools(chain, "trending", args.pages,
                             args.min_mcap, args.max_mcap,
                             args.min_liq, args.min_vol, args.min_age)
        print(f"  → {len(p1)} pools", flush=True)

        print(f"  [GT/{chain.upper()}] 24H volume sort...", flush=True)
        p2 = fetch_gt_pools(chain, "h24_volume_usd_desc", min(args.pages, 3),
                             args.min_mcap, args.max_mcap,
                             args.min_liq, args.min_vol, args.min_age)
        print(f"  → {len(p2)} pools", flush=True)

        print(f"  [GT/{chain.upper()}] 24H tx count sort...", flush=True)
        p3 = fetch_gt_pools(chain, "h24_tx_count_desc", min(args.pages, 3),
                             args.min_mcap, args.max_mcap,
                             args.min_liq, args.min_vol, args.min_age)
        print(f"  → {len(p3)} pools", flush=True)

        # ── SOURCE 7: New pools < 7 days ─────────────────────────────────────
        print(f"  [GT/{chain.upper()}] new pools (< 7 days)...", flush=True)
        np = fetch_gt_new_pools(chain, args.min_liq, args.min_vol,
                                 min_mcap=args.min_mcap, max_mcap=args.max_mcap)
        new_pools.extend(np)
        print(f"  → {len(np)} new pools\n", flush=True)

        all_pools.extend(p1 + p2 + p3)

    all_pools = dedup_pools(all_pools)
    all_pools.sort(
        key=lambda p: safe_float((p.get("attributes", {}).get("price_change_percentage") or {}).get("h6")),
        reverse=True
    )
    print(f"  After dedup + 6H sort: {len(all_pools)} unique pools\n", flush=True)

    # ── SOURCE 4: DeFiLlama TVL ─────────────────────────────────────────────
    print("  [DeFiLlama] Base protocols with growing TVL...", flush=True)
    defillama = fetch_defillama_base_gems(max_tvl=args.llama_max_tvl)

    # ── SOURCE 11: DeFiLlama yield signals ──────────────────────────────────
    print("  [DeFiLlama] Yield pools (high APY small TVL)...", flush=True)
    llama_yields = fetch_defillama_yields(chains=chains)

    # ── SOURCE 12: DeFiLlama revenue ────────────────────────────────────────
    print("  [DeFiLlama] Protocol revenue generators...", flush=True)
    llama_revenue = fetch_defillama_revenue(chains=chains)

    # ── SOURCE 5: CoinGecko small-cap categories ─────────────────────────────
    print(f"\n  [CoinGecko] Base ecosystem small-caps...", flush=True)
    cg_coins = fetch_cg_small_cap_base(args.min_mcap, args.max_mcap, pages=2)
    print(f"  → {len(cg_coins)} CoinGecko tokens in range", flush=True)

    # ── SOURCE 9: CoinGecko trending ─────────────────────────────────────────
    print("  [CoinGecko] Global trending...", flush=True)
    cg_trending = fetch_cg_trending()
    time.sleep(2)

    # ── SOURCE 9b: CoinGecko recently added ──────────────────────────────────
    print("  [CoinGecko] Recently listed...", flush=True)
    cg_new = fetch_cg_recently_added()
    time.sleep(2)

    # ── SOURCE 6: DexScreener boosts ─────────────────────────────────────────
    print("\n  [DexScreener] Base boosts + profiles...", flush=True)
    dex_boosts = fetch_dex_boosts("base")
    print(f"  → {len(dex_boosts)} signals", flush=True)

    # ── SOURCE 10: DexScreener global trending ───────────────────────────────
    print("  [DexScreener] Global boosts (all chains)...", flush=True)
    dex_global = fetch_dex_global_trending()

    # ── SOURCE 10b: DexScreener extended search ──────────────────────────────
    print("  [DexScreener] Extended narrative search...", flush=True)
    dex_search_extra = fetch_dex_token_search_extended(
        terms=[],
        chains=chains
    )

    # ── SOURCE 8: Cross-chain momentum ──────────────────────────────────────
    print("\n  [GT/xchain] ETH / Arbitrum / Optimism trending...", flush=True)
    crosschain = fetch_gt_crosschain()

    # ── SOURCE 13: Birdeye (optional) ────────────────────────────────────────
    birdeye_pools = []
    if birdeye_key:
        print("\n  [Birdeye] Trending tokens by vol change...", flush=True)
        for chain in chains:
            bp = fetch_birdeye_trending(chain=chain,
                                        min_mcap=args.min_mcap,
                                        max_mcap=args.max_mcap,
                                        api_key=birdeye_key)
            birdeye_pools.extend(bp)

    # ── SOURCE 14: GoPlus security check on top new pools ────────────────────
    goplus_risks = {}
    high_score_new_addrs = [
        p.get("attributes", {}).get("address", "")
        for p in new_pools
        if p.get("_score", 0) >= 40 and p.get("attributes", {}).get("address")
    ]
    if high_score_new_addrs:
        print(f"\n  [GoPlus] Checking {len(high_score_new_addrs)} high-score new pool addresses...", flush=True)
        for chain in chains:
            chain_addrs = [a for a in high_score_new_addrs if a.startswith("0x")]  # base
            if chain_addrs:
                goplus_risks.update(check_goplus_risk(chain_addrs[:20], chain="base"))

    # ── Update watchlist ──────────────────────────────────────────────────────
    all_discovered = all_pools + new_pools
    print("\n  Updating watchlist...")
    changes = update_watchlist(wl, all_discovered, defillama, today)
    save_watchlist(wl)
    print(f"  → +{len(changes['new'])} new | "
          f"{len(changes['repeats'])} repeats | "
          f"{len(changes['graduates'])} graduates\n")

    # ── Build report ──────────────────────────────────────────────────────────
    report = build_report(
        pools         = all_pools,
        cg_coins      = cg_coins,
        defillama     = defillama,
        dex_boosts    = dex_boosts + dex_search_extra,
        changes       = changes,
        wl            = wl,
        run_date      = run_date,
        chains        = chains,
        min_mcap      = args.min_mcap,
        max_mcap      = args.max_mcap,
        new_pools     = new_pools,
        cg_trending   = cg_trending,
        cg_new        = cg_new,
        llama_yields  = llama_yields,
        llama_revenue = llama_revenue,
        crosschain    = crosschain,
        goplus_risks  = goplus_risks,
    )

    print(f"\n{sep}\n")
    print(report)
    print(f"\n{sep}\n")

    fname = f"micro_report_{today}.md"
    with open(fname, "w") as f:
        f.write(report)
    print(f"  Report: {fname}")
    print(f"  Watchlist: {WATCHLIST_FILE} ({len(wl['tokens'])} DEX + "
          f"{len(wl.get('defi_protocols',{}))} DeFi protocols)")

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"  Also: {args.output}")


if __name__ == "__main__":
    main()
