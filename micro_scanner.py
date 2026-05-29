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
                print(f"    [GT/{chain}/{sort[:6]}] Rate limit p{page} — wait 30s", flush=True)
                time.sleep(30)
                r = requests.get(url, timeout=20, headers=headers)
            if r.status_code != 200:
                print(f"    [GT/{chain}/{sort[:6]}] HTTP {r.status_code} p{page}", flush=True)
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
                 min_mcap: float, max_mcap: float) -> str:

    lines = []
    lines.append("# 🔬 Micro-Cap Radar — Full Multi-Source Report")
    lines.append(
        f"**Date:** {run_date}  |  **Chains:** {', '.join(c.upper() for c in chains)}  |  "
        f"**MCap:** ${min_mcap/1e3:.0f}K – ${max_mcap/1e6:.1f}M\n"
    )
    lines.append("**Sources:** GeckoTerminal (trending+6H+vol) · DeFiLlama · CoinGecko · DexScreener\n")
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

    chains   = [c.strip().lower() for c in args.chain.split(",")]
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sep = "=" * 64

    print(f"\n{sep}")
    print(f"  🔬 Micro-Cap Radar  {run_date}")
    print(f"  Chains: {', '.join(c.upper() for c in chains)}")
    print(f"  MCap ${args.min_mcap/1e3:.0f}K–${args.max_mcap/1e6:.1f}M | "
          f"Min liq ${args.min_liq:,.0f} | Min vol ${args.min_vol:,.0f}")
    print(f"  Sources: GeckoTerminal (3 sorts) · DeFiLlama · CoinGecko · DexScreener")
    print(f"{sep}\n")

    wl = load_watchlist()
    print(f"  Watchlist loaded: {len(wl['tokens'])} DEX tokens, "
          f"{len(wl.get('defi_protocols', {}))} DeFiLlama protocols\n")

    # ── Fetch from all sources ──────────────────────────────────────────────
    all_pools = []

    for chain in chains:
        if chain not in {"base", "solana"}:
            print(f"  [SKIP] Unknown chain: {chain}")
            continue

        # GT trending
        print(f"  [GT/{chain.upper()}] trending pools...")
        p1 = fetch_gt_pools(chain, "trending", args.pages,
                             args.min_mcap, args.max_mcap,
                             args.min_liq, args.min_vol, args.min_age)
        print(f"  → {len(p1)} pools")

        # GT 6H movers (different from trending — catches pumps before they trend)
        print(f"  [GT/{chain.upper()}] 6H price change sort...")
        p2 = fetch_gt_pools(chain, "h6_price_change_percent_desc", args.pages,
                             args.min_mcap, args.max_mcap,
                             args.min_liq, args.min_vol, args.min_age)
        print(f"  → {len(p2)} pools")

        # GT 24H volume (sustained interest)
        print(f"  [GT/{chain.upper()}] 24H volume sort...")
        p3 = fetch_gt_pools(chain, "h24_volume_usd_desc", min(args.pages, 3),
                             args.min_mcap, args.max_mcap,
                             args.min_liq, args.min_vol, args.min_age)
        print(f"  → {len(p3)} pools\n")

        all_pools.extend(p1 + p2 + p3)

    all_pools = dedup_pools(all_pools)
    print(f"  After dedup: {len(all_pools)} unique pools\n")

    # DeFiLlama Base protocols
    print("  [DeFiLlama] Base protocols with growing TVL...")
    defillama = fetch_defillama_base_gems(max_tvl=args.llama_max_tvl)

    # CoinGecko small-cap Base tokens
    print(f"\n  [CoinGecko] Base ecosystem small-caps...")
    cg_coins = fetch_cg_small_cap_base(args.min_mcap, args.max_mcap, pages=2)
    print(f"  → {len(cg_coins)} CoinGecko tokens in range\n")

    # DexScreener boosts
    print("  [DexScreener] Base boosts + profiles...")
    dex_boosts = fetch_dex_boosts("base")
    print(f"  → {len(dex_boosts)} signals\n")

    # Update watchlist
    print("  Updating watchlist...")
    changes = update_watchlist(wl, all_pools, defillama, today)
    save_watchlist(wl)
    print(f"  → +{len(changes['new'])} new | "
          f"{len(changes['repeats'])} repeats | "
          f"{len(changes['graduates'])} graduates\n")

    # Build report
    report = build_report(
        pools      = all_pools,
        cg_coins   = cg_coins,
        defillama  = defillama,
        dex_boosts = dex_boosts,
        changes    = changes,
        wl         = wl,
        run_date   = run_date,
        chains     = chains,
        min_mcap   = args.min_mcap,
        max_mcap   = args.max_mcap,
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
