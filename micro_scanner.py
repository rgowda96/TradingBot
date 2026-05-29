#!/usr/bin/env python3
"""
micro_scanner.py — Under-the-radar micro-cap tracker
=====================================================

Tracks $10K–$1M MCap tokens on Base + Solana that appear in the 6H
trending data. Maintains a persistent watchlist across every scan so
tokens that keep showing up get flagged — that's the real signal.

Key insight:
  A token trending once = noise.
  A token trending across 2–3 consecutive 3-day scans = something building.
  A token crossing from <$500K to >$1M while on the watchlist = we caught it early.

Data sources:
  - GeckoTerminal trending pools (Base + Solana, free API)
  - DexScreener boost / profile API (paid promos = has backing)

Watchlist file: micro_watchlist.json  (committed to repo — persistent memory)

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
import argparse
import time
from datetime import datetime, timezone
from collections import defaultdict

WATCHLIST_FILE = "micro_watchlist.json"
GECKO_URL      = "https://api.geckoterminal.com/api/v2"
CHAIN_IDS      = {"base": "base", "solana": "solana"}

MEME_KEYWORDS = {
    "doge", "pepe", "shib", "inu", "moon", "safe", "elon", "baby",
    "floki", "bonk", "wif", "mog", "brett", "toshi", "ponke", "pnut",
    "harambe", "tremp", "boden", "grok", "wojak", "cope", "ape",
    "frog", "cat coin", "dog coin", "dog wif",
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
    return {"version": 1, "tokens": {}, "last_updated": ""}


def save_watchlist(wl: dict):
    wl["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(WATCHLIST_FILE, "w") as f:
        json.dump(wl, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Conviction scorer
# ─────────────────────────────────────────────────────────────────────────────

def conviction_score(pool: dict) -> float:
    """0–100 composite conviction score."""
    attrs = pool.get("attributes", {})
    pc    = attrs.get("price_change_percentage", {})

    h1   = float(pc.get("h1",  0) or 0)
    h6   = float(pc.get("h6",  0) or 0)
    h24  = float(pc.get("h24", 0) or 0)
    vol24h = float((attrs.get("volume_usd") or {}).get("h24", 0) or 0)
    mcap   = float(attrs.get("market_cap_usd") or attrs.get("fdv_usd") or 1)
    liq    = float(attrs.get("reserve_in_usd", 0) or 0)

    score = 0

    # ── 6H momentum (primary signal) ──────────────────────────────────────────
    if h6 > 200:  score += 40
    elif h6 > 100: score += 32
    elif h6 > 50:  score += 24
    elif h6 > 20:  score += 14
    elif h6 > 5:   score += 7
    elif h6 < -40: score -= 12

    # ── 24H confirms trend ────────────────────────────────────────────────────
    if h24 > 200:  score += 22
    elif h24 > 100: score += 16
    elif h24 > 50:  score += 10
    elif h24 > 20:  score += 5
    elif h24 < -50: score -= 10

    # ── Volume / MCap ratio (genuine trading interest) ─────────────────────────
    ratio = vol24h / max(mcap, 1)
    if ratio > 2.0:  score += 25   # trading 2x its entire cap in 24H
    elif ratio > 1.0: score += 18
    elif ratio > 0.5: score += 12
    elif ratio > 0.2: score += 6
    elif ratio > 0.05: score += 2

    # ── Liquidity depth (safety — can exit without crashing price) ────────────
    if liq > 200_000: score += 15
    elif liq > 100_000: score += 11
    elif liq > 50_000:  score += 7
    elif liq > 10_000:  score += 3
    elif liq < 2_000:   score -= 20   # rug risk

    # ── Buy/sell pressure in last hour ────────────────────────────────────────
    t_h1   = attrs.get("transactions", {}).get("h1", {}) or {}
    buys   = int(t_h1.get("buys",  0) or 0)
    sells  = int(t_h1.get("sells", 0) or 0)
    total  = buys + sells
    if total > 0:
        br = buys / total
        if br > 0.75:  score += 12
        elif br > 0.60: score += 6

    return max(0.0, min(float(score), 100.0))


# ─────────────────────────────────────────────────────────────────────────────
# Data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_gt_trending(chain: str, pages: int,
                       min_mcap: float, max_mcap: float,
                       min_liq: float, min_vol: float,
                       min_age: int) -> list:
    """GeckoTerminal trending pools filtered to our MCap/liq/vol/age range."""
    results = []
    headers = {"Accept": "application/json;version=20230302"}

    for page in range(1, pages + 1):
        url = (f"{GECKO_URL}/networks/{chain}/trending_pools"
               f"?page={page}&include=base_token,dex")
        try:
            r = requests.get(url, timeout=20, headers=headers)
            if r.status_code == 429:
                print(f"    [GT/{chain}] Rate limit p{page} — wait 30s", flush=True)
                time.sleep(30)
                r = requests.get(url, timeout=20, headers=headers)
            if r.status_code != 200:
                print(f"    [GT/{chain}] HTTP {r.status_code} p{page}", flush=True)
                break
            data = r.json().get("data", [])
            if not data:
                break

            added = 0
            for pool in data:
                attrs = pool.get("attributes", {})

                # MCap
                mcap = float(attrs.get("market_cap_usd") or attrs.get("fdv_usd") or 0)
                if not (min_mcap <= mcap <= max_mcap):
                    continue

                # Liquidity
                liq = float(attrs.get("reserve_in_usd", 0) or 0)
                if liq < min_liq:
                    continue

                # 24H volume
                vol24h = float((attrs.get("volume_usd") or {}).get("h24", 0) or 0)
                if vol24h < min_vol:
                    continue

                # Age
                created = attrs.get("pool_created_at", "")
                age = days_since(created) if created else -1
                if min_age > 0 and 0 <= age < min_age:
                    continue
                attrs["_age_days"] = age

                # Annotate
                pool["_chain"]  = chain
                pool["_mcap"]   = mcap
                pool["_liq"]    = liq
                pool["_vol24h"] = vol24h
                pool["_score"]  = conviction_score(pool)
                results.append(pool)
                added += 1

            print(f"    [GT/{chain}] page {page}: +{added} ({len(results)} total)", flush=True)
            time.sleep(1.2)

        except Exception as e:
            print(f"    [GT/{chain}] p{page} error: {e}", flush=True)
            break

    return results


def fetch_dex_boosts_base() -> list:
    """DexScreener boosted Base tokens — paid promos signal confidence."""
    try:
        r = requests.get(
            "https://api.dexscreener.com/token-boosts/latest/v1", timeout=15
        )
        if r.status_code == 200:
            return [t for t in r.json() if t.get("chainId") == "base"]
    except Exception as e:
        print(f"    [DEX boosts] error: {e}", flush=True)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Watchlist management
# ─────────────────────────────────────────────────────────────────────────────

def update_watchlist(wl: dict, pools: list, today: str) -> dict:
    """Update watchlist; return dict of changes."""
    new_entries  = []
    repeat_hits  = []
    graduates    = []

    for pool in pools:
        attrs = pool.get("attributes", {})
        chain = pool.get("_chain", "base")
        mcap  = pool.get("_mcap",  0)
        score = pool.get("_score", 0)
        liq   = pool.get("_liq",   0)
        vol   = pool.get("_vol24h", 0)

        address  = attrs.get("address", "") or pool.get("id", "unknown")
        key      = f"{chain}_{address}"
        pname    = attrs.get("name", "?")
        sym      = pname.split("/")[0].strip() if "/" in pname else pname
        meme_flag = is_meme(pname, sym)

        pc   = attrs.get("price_change_percentage", {})
        h6   = float(pc.get("h6",  0) or 0)
        h24  = float(pc.get("h24", 0) or 0)
        age  = attrs.get("_age_days", -1)

        snapshot = {"date": today, "mcap": mcap, "h6": h6, "h24": h24,
                    "liq": liq, "score": score}

        existing = wl["tokens"].get(key)

        if existing is None:
            entry = {
                "name":         sym,
                "pool_name":    pname,
                "chain":        chain,
                "address":      address,
                "first_seen":   today,
                "last_seen":    today,
                "appearances":  1,
                "history":      [snapshot],
                "max_score":    score,
                "liq":          liq,
                "age_days":     age,
                "is_meme":      meme_flag,
                "status":       "watching",
                "graduated_at": None,
                "graduated_mcap": None,
            }
            wl["tokens"][key] = entry
            new_entries.append(entry)
        else:
            existing["last_seen"]    = today
            existing["appearances"] += 1
            existing["liq"]          = liq
            existing["history"].append(snapshot)
            existing["history"]      = existing["history"][-12:]   # keep last 12 scans
            if score > existing.get("max_score", 0):
                existing["max_score"] = score

            # Graduation: MCap crossed $1M
            if mcap >= 1_000_000 and existing["status"] == "watching":
                existing["status"]        = "graduated"
                existing["graduated_at"]  = today
                existing["graduated_mcap"] = mcap
                graduates.append(existing)
            else:
                repeat_hits.append((existing, score))

    return {
        "new":       new_entries,
        "repeats":   sorted(repeat_hits, key=lambda x: (-x[0]["appearances"], -x[1])),
        "graduates": graduates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_report(pools: list, changes: dict, wl: dict,
                 dex_boosts: list, run_date: str, chains: list,
                 min_mcap: float, max_mcap: float) -> str:

    lines = []
    lines.append("# 🔬 Micro-Cap Radar Report")
    lines.append(f"**Date:** {run_date}  |  **Chains:** {', '.join(c.upper() for c in chains)}  |  "
                 f"**MCap:** ${min_mcap/1e3:.0f}K – ${max_mcap/1e6:.1f}M\n")
    lines.append("---\n")

    watching  = sum(1 for t in wl["tokens"].values() if t["status"] == "watching")
    graduated = sum(1 for t in wl["tokens"].values() if t["status"] == "graduated")
    lines.append(f"*Watchlist: {len(wl['tokens'])} total — "
                 f"{watching} active watching — {graduated} graduated to >$1M*\n")

    # ── GRADUATES (the wins) ──────────────────────────────────────────────────
    if changes["graduates"]:
        lines.append("## 🏆 GRADUATED — Caught Early, Now Above $1M!\n")
        for t in changes["graduates"]:
            hist       = t.get("history", [{}])
            entry_mcap = hist[0].get("mcap", 0)
            now_mcap   = t.get("graduated_mcap", 0)
            mult       = (now_mcap / entry_mcap) if entry_mcap > 0 else 0
            lines.append(
                f"**{t['name']}** on {t['chain'].upper()} | "
                f"First seen {t['first_seen']} @ ${entry_mcap:,.0f} → "
                f"Graduated ${now_mcap:,.0f} ({mult:.1f}x) | "
                f"{t['appearances']} scan appearances"
            )
        lines.append("")

    # ── PERSISTENT SIGNALS (2+ appearances) ──────────────────────────────────
    lines.append("## 🔁 Persistent Signals — Appearing Across Multiple Scans\n")
    lines.append("*The real edge. Nobody notices, but they keep showing up.*\n")

    persistent = sorted(
        [(k, v) for k, v in wl["tokens"].items()
         if v["appearances"] >= 2 and v["status"] == "watching"],
        key=lambda x: (-x[1]["appearances"], -x[1]["max_score"])
    )

    if persistent:
        for key, t in persistent[:20]:
            hist   = t.get("history", [])
            mcap0  = hist[0]["mcap"]  if hist else 0
            mcap1  = hist[-1]["mcap"] if hist else 0
            pct    = ((mcap1 - mcap0) / mcap0 * 100) if mcap0 > 0 else 0
            trend  = "📈" if pct > 5 else ("📉" if pct < -5 else "➡️")
            meme   = " ⚠️ meme" if t.get("is_meme") else ""
            streak = "🔥" * min(t["appearances"], 5)
            lines.append(
                f"{streak} **{t['name']}** | {t['chain'].upper()} | "
                f"${mcap1:,.0f} MCap | {t['appearances']}x seen | "
                f"{trend} {pct:+.0f}% from entry | Score {t['max_score']}/100{meme}"
            )
    else:
        lines.append("*First run — baseline being built. Check back in 3 days.*")
    lines.append("")

    # ── NEW ENTRIES ───────────────────────────────────────────────────────────
    new_sorted = sorted(changes["new"], key=lambda x: x.get("max_score", 0), reverse=True)
    base_new   = [t for t in new_sorted if t["chain"] == "base"]
    sol_new    = [t for t in new_sorted if t["chain"] == "solana"]

    if base_new:
        lines.append("## 🆕 New Base Chain Entries\n")
        for t in base_new[:12]:
            h    = t.get("history", [{}])
            h6   = h[-1].get("h6",   0)
            h24  = h[-1].get("h24",  0)
            mcap = h[-1].get("mcap", 0)
            liq  = t.get("liq", 0)
            age  = t.get("age_days", -1)
            sc   = t.get("max_score", 0)
            meme = " ⚠️ meme" if t.get("is_meme") else ""
            lines.append(
                f"- **{t['name']}** — ${mcap:,.0f} MCap | Liq ${liq:,.0f} | "
                f"6H {h6:+.1f}% | 24H {h24:+.1f}% | {age}d old | {sc}/100{meme}"
            )
        lines.append("")

    if sol_new:
        lines.append("## 🆕 New Solana Entries (watch only — bot covers Base)\n")
        for t in sol_new[:8]:
            h    = t.get("history", [{}])
            h6   = h[-1].get("h6",  0)
            h24  = h[-1].get("h24", 0)
            mcap = h[-1].get("mcap", 0)
            sc   = t.get("max_score", 0)
            lines.append(
                f"- **{t['name']}** — ${mcap:,.0f} MCap | 6H {h6:+.1f}% | 24H {h24:+.1f}% | {sc}/100"
            )
        lines.append("")

    # ── TOP 20 THIS SCAN ──────────────────────────────────────────────────────
    lines.append("## 📋 Top 20 This Scan (by conviction score)\n")

    for chain_filter in ["base", "solana"]:
        chain_pools = sorted(
            [p for p in pools if p.get("_chain") == chain_filter],
            key=lambda p: p.get("_score", 0), reverse=True
        )[:10]
        if not chain_pools:
            continue

        label = chain_filter.upper() + (" (bot trades this)" if chain_filter == "base" else " (watch only)")
        lines.append(f"### {label}")
        for i, p in enumerate(chain_pools, 1):
            attrs  = p.get("attributes", {})
            pname  = attrs.get("name", "?")
            mcap   = p.get("_mcap",  0)
            liq    = p.get("_liq",   0)
            vol    = p.get("_vol24h",0)
            age    = attrs.get("_age_days", -1)
            score  = p.get("_score", 0)
            pc     = attrs.get("price_change_percentage", {})
            h6     = float(pc.get("h6",  0) or 0)
            h24    = float(pc.get("h24", 0) or 0)
            txns   = attrs.get("transactions", {})
            t_h1   = (txns.get("h1") or {})
            buys   = int(t_h1.get("buys",  0) or 0)
            sells  = int(t_h1.get("sells", 0) or 0)
            bs     = f"{buys}B/{sells}S" if (buys + sells) > 0 else "—"

            # Check if it's already in watchlist (repeat appearance)
            address = attrs.get("address", "") or p.get("id", "")
            key     = f"{chain_filter}_{address}"
            wl_entry = wl["tokens"].get(key)
            repeat_tag = f" 🔁×{wl_entry['appearances']}" if wl_entry and wl_entry["appearances"] > 1 else ""

            lines.append(
                f"{i:>2}. **{pname}** — ${mcap:,.0f} MCap | Liq ${liq:,.0f} | "
                f"6H {h6:+.1f}% | 24H {h24:+.1f}% | Vol ${vol:,.0f} | "
                f"{age}d | {bs} | {score}/100{repeat_tag}"
            )
        lines.append("")

    # ── TINY GEMS under $100K ─────────────────────────────────────────────────
    tiny = sorted(
        [p for p in pools if p.get("_mcap", 0) < 100_000 and p.get("_score", 0) >= 35],
        key=lambda p: p["_score"], reverse=True
    )[:6]
    if tiny:
        lines.append("## 💎 Tiny Gems Under $100K MCap (Score ≥35)\n")
        lines.append("*Highest risk / highest reward. Always rug-check before entry.*\n")
        for p in tiny:
            attrs = p.get("attributes", {})
            pname = attrs.get("name", "?")
            mcap  = p.get("_mcap", 0)
            liq   = p.get("_liq",  0)
            score = p.get("_score",0)
            pc    = attrs.get("price_change_percentage", {})
            h6    = float(pc.get("h6", 0) or 0)
            lines.append(
                f"- **{pname}** | {p.get('_chain','').upper()} | "
                f"${mcap:,.0f} MCap | Liq ${liq:,.0f} | 6H {h6:+.1f}% | {score}/100"
            )
        lines.append("")

    # ── DEXSCREENER BOOSTS ────────────────────────────────────────────────────
    if dex_boosts:
        lines.append("## 📡 DexScreener Paid Boosts (Base) — Has Backing\n")
        for b in dex_boosts[:8]:
            name = b.get("description", "—")
            url  = b.get("url", "")
            amt  = b.get("amount", 0)
            lines.append(f"- [{amt:,} boost pts] **{name}** — {url}")
        lines.append("")

    # ── STRATEGY REMINDER ─────────────────────────────────────────────────────
    lines.append("## 🧭 How to Use This\n")
    lines.append("1. **🔁 Persistent signals** = the real edge. Appearing 2-3× = someone is steadily buying.")
    lines.append("2. **🆕 New entries** = just entered radar. Watch for 1 more scan before sizing in.")
    lines.append("3. **💎 Tiny gems** = max risk. Check liq depth — if liq < $5K, any sell dumps it.")
    lines.append("4. **🏆 Graduates** = validation that this system works. MCap went 2x+ while on watchlist.")
    lines.append("5. Always check DexScreener for rug indicators: locked liq, team wallet %, top-10 holder %.")
    lines.append("")
    lines.append("---")
    lines.append(f"*Generated by micro_scanner.py — {run_date}*")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Micro-cap under-the-radar tracker")
    parser.add_argument("--min-mcap", type=float, default=10_000)
    parser.add_argument("--max-mcap", type=float, default=1_000_000)
    parser.add_argument("--min-liq",  type=float, default=1_000)
    parser.add_argument("--min-vol",  type=float, default=1_000)
    parser.add_argument("--min-age",  type=int,   default=5,
                        help="Min pool age in days (default 5, use 0 to include brand new)")
    parser.add_argument("--chain",    type=str,   default="base,solana")
    parser.add_argument("--pages",    type=int,   default=5,
                        help="GeckoTerminal pages per chain (default 5 = 100 pools)")
    parser.add_argument("--output",   type=str,   default=None)
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    chains   = [c.strip().lower() for c in args.chain.split(",")]
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  🔬 Micro-Cap Radar  {run_date}")
    print(f"  Chains: {', '.join(c.upper() for c in chains)}")
    print(f"  MCap: ${args.min_mcap/1e3:.0f}K – ${args.max_mcap/1e6:.1f}M  |  "
          f"Min liq ${args.min_liq:,.0f}  |  Min vol ${args.min_vol:,.0f}")
    print(f"{sep}\n")

    # Load watchlist from disk
    wl = load_watchlist()
    print(f"  Watchlist loaded: {len(wl['tokens'])} entries\n")

    # Fetch data
    all_pools = []
    for chain in chains:
        if chain not in CHAIN_IDS:
            print(f"  [SKIP] Unknown chain: {chain}")
            continue
        print(f"  [1] Fetching GeckoTerminal trending — {chain.upper()}...")
        pools = fetch_gt_trending(
            chain    = CHAIN_IDS[chain],
            pages    = args.pages,
            min_mcap = args.min_mcap,
            max_mcap = args.max_mcap,
            min_liq  = args.min_liq,
            min_vol  = args.min_vol,
            min_age  = args.min_age,
        )
        print(f"  → {len(pools)} pools in range on {chain.upper()}\n")
        all_pools.extend(pools)

    print("  [2] Fetching DexScreener Base boosts...")
    dex_boosts = fetch_dex_boosts_base()
    print(f"  → {len(dex_boosts)} boosted tokens\n")

    print("  [3] Updating watchlist...")
    changes = update_watchlist(wl, all_pools, today)
    save_watchlist(wl)
    print(f"  → +{len(changes['new'])} new | "
          f"{len(changes['repeats'])} repeats | "
          f"{len(changes['graduates'])} graduates\n")

    print("  [4] Building report...")
    report = build_report(
        pools     = all_pools,
        changes   = changes,
        wl        = wl,
        dex_boosts= dex_boosts,
        run_date  = run_date,
        chains    = chains,
        min_mcap  = args.min_mcap,
        max_mcap  = args.max_mcap,
    )

    print(f"\n{sep}\n")
    print(report)
    print(f"\n{sep}\n")

    # Save dated report
    fname = f"micro_report_{today}.md"
    with open(fname, "w") as f:
        f.write(report)
    print(f"  Report saved: {fname}")
    print(f"  Watchlist:    {WATCHLIST_FILE} ({len(wl['tokens'])} entries)")

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"  Also saved: {args.output}")


if __name__ == "__main__":
    main()
