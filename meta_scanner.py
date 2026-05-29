#!/usr/bin/env python3
"""
meta_scanner.py — 3-day crypto meta intelligence report
========================================================

Runs on a schedule (every 3 days) and produces a full market brief:
  1. Base-chain micro-cap tokens with momentum (CoinGecko)
  2. Narrative classification — what meta is forming or dying
  3. DexScreener boosted / profiled signals
  4. Phase estimate (Seed → Sprout → Named → Parabolic → Death)
  5. Dated .md report committed to repo for history

Usage:
    python meta_scanner.py                    # prints to stdout + saves dated .md
    python meta_scanner.py --output out.md    # also saves to named file
    python meta_scanner.py --min-mcap 100000 --max-mcap 50000000
"""

import requests
import json
import re
import time
import sys
import os
import argparse
from datetime import datetime, timezone
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Narrative buckets — (keywords, display label)
# ─────────────────────────────────────────────────────────────────────────────
NARRATIVE_BUCKETS = {
    "AI_AGENTS": (
        ["ai agent", " ai ", "agent", "intelligence", "llm", "gpt", "neural",
         "autonomous", "reasoning", "inference", "copilot", "openai", "anthropic"],
        "🤖 AI / Agents",
    ),
    "DEPIN": (
        ["depin", "node operator", "infrastructure", "physical network",
         "hardware", "sensor", "bandwidth", "compute", "gpu network",
         "mining", "hotspot", "coverage"],
        "📡 DePIN",
    ),
    "RWA": (
        ["real world asset", "tokenized", "rwa", "backed by", "equity token",
         "bond", "treasury", "t-bill", "fund token", "real estate",
         "commodity", "gold backed"],
        "🏦 RWA",
    ),
    "SOCIAL": (
        ["social", "creator", "content creator", "fan token", "identity",
         "reputation", "influencer", "community token", "tip", "social graph"],
        "👥 Social / Creator",
    ),
    "GAMING": (
        ["game", "gaming", "play to earn", "nft game", "guild", "quest",
         "metaverse", "virtual world", "pvp", "loot", "in-game"],
        "🎮 Gaming / NFT",
    ),
    "PAYMENTS": (
        ["payment", "remittance", "banking", "send money", "stablecoin",
         "on-ramp", "off-ramp", "cross-border", "merchant"],
        "💳 Payments",
    ),
    "ZK": (
        ["zero knowledge", "zk proof", "zk-snark", "zk-stark", "zkp",
         "privacy protocol", "shielded", "rollup", "validium"],
        "🔐 ZK / Privacy",
    ),
    "DEV_TOOLS": (
        ["developer tool", "dev tool", "sdk", "developer platform", "oracle",
         "indexer", "data availability", "api gateway", "smart contract audit",
         "debugging", "testnet"],
        "🛠 Dev Tools / Infra",
    ),
    "YIELD": (
        ["yield", "liquid staking", "restaking", "vault", "apy", "apr",
         "auto-compound", "earn interest", "money market", "lending protocol"],
        "💰 Yield / Staking",
    ),
    "LAUNCHPAD": (
        ["launchpad", "ido", "token launch", "incubator", "accelerator",
         "fundraise", "presale", "fair launch"],
        "🚀 Launchpad",
    ),
}

MEME_SIGNALS = {
    "meme coin", "memecoin", "meme token", "just for fun", "no utility",
    "community token", "fun project", "joke token",
}

MEME_NAME_KW = {
    "doge", "pepe", "shib", "inu", "moon", "safe", "elon", "baby",
    "floki", "bonk", "wif", "mog", "brett", "toshi", "ponke", "pnut",
    "harambe", "tremp", "boden", "grok", "wojak",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_meme(name: str, symbol: str, desc: str = "") -> bool:
    nl = name.lower();  dl = desc.lower()
    if any(kw in nl for kw in MEME_NAME_KW):
        return True
    if any(phrase in dl for phrase in MEME_SIGNALS):
        return True
    return False


def classify_narrative(name: str, symbol: str, description: str = "") -> list:
    """Return list of matching narrative bucket keys (can be multiple)."""
    text = f"{name} {symbol} {description}".lower()
    return [
        key for key, (keywords, _) in NARRATIVE_BUCKETS.items()
        if any(kw in text for kw in keywords)
    ]


def momentum_score(coin: dict) -> float:
    """0–100 composite momentum from price change + volume/MCap ratio."""
    c24h = coin.get("price_change_percentage_24h") or 0
    c7d  = coin.get("price_change_percentage_7d_in_currency") or 0
    vol  = coin.get("total_volume") or 0
    mcap = max(coin.get("market_cap") or 1, 1)

    score = 0
    if c24h > 50:   score += 40
    elif c24h > 20: score += 25
    elif c24h > 10: score += 15
    elif c24h > 0:  score += 5
    elif c24h < -30: score -= 10

    if c7d > 100:   score += 30
    elif c7d > 50:  score += 20
    elif c7d > 20:  score += 10
    elif c7d < -40: score -= 10

    ratio = vol / mcap
    if ratio > 0.5:   score += 30
    elif ratio > 0.2: score += 20
    elif ratio > 0.1: score += 10

    return max(0, min(score, 100))


def phase_label(avg_score: float) -> str:
    if avg_score >= 65: return "⚡ Phase 3 — PARABOLIC  (risk of top, size carefully)"
    if avg_score >= 42: return "📈 Phase 2 — NAMED      (momentum confirmed, still room)"
    if avg_score >= 18: return "🌱 Phase 1 — FORMING    (quiet accumulation, best entry)"
    return              "😴 Phase 0 — SEED        (nothing moving yet, watch & wait)"


# ─────────────────────────────────────────────────────────────────────────────
# Data fetchers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cg_base(pages: int, min_mcap: float, max_mcap: float) -> list:
    """CoinGecko Base ecosystem tokens filtered to MCap range."""
    coins = []
    seen  = set()
    headers = {"Accept": "application/json"}
    for page in range(1, pages + 1):
        url = (
            "https://api.coingecko.com/api/v3/coins/markets"
            f"?vs_currency=usd&category=base-ecosystem"
            f"&order=market_cap_desc&per_page=100&page={page}"
            "&sparkline=false&price_change_percentage=24h,7d"
        )
        try:
            r = requests.get(url, timeout=25, headers=headers)
            if r.status_code == 429:
                print("  [CG] Rate limited — sleeping 65s", flush=True)
                time.sleep(65)
                r = requests.get(url, timeout=25, headers=headers)
            if r.status_code != 200:
                print(f"  [CG] page {page} → HTTP {r.status_code}", flush=True)
                break
            data = r.json()
            if not data:
                break
            added = 0
            for c in data:
                if c["id"] in seen:
                    continue
                mcap = c.get("market_cap") or 0
                if min_mcap <= mcap <= max_mcap:
                    coins.append(c)
                    seen.add(c["id"])
                    added += 1
            print(f"  [CG] page {page}: +{added} coins ({len(coins)} total)", flush=True)
            time.sleep(2.2)
        except Exception as e:
            print(f"  [CG] page {page} error: {e}", flush=True)
            break
    return coins


def fetch_dex_base_signals() -> list:
    """DexScreener public API — boosted and newly profiled Base tokens."""
    results = []
    endpoints = [
        ("boosted",  "https://api.dexscreener.com/token-boosts/latest/v1"),
        ("profiled", "https://api.dexscreener.com/token-profiles/latest/v1"),
    ]
    for kind, url in endpoints:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            for item in r.json():
                if item.get("chainId") == "base":
                    results.append({
                        "kind":    kind,
                        "address": item.get("tokenAddress", ""),
                        "name":    item.get("description", "—"),
                        "url":     item.get("url", ""),
                        "amount":  item.get("amount", 0),
                        "icon":    item.get("icon", ""),
                    })
        except Exception as e:
            print(f"  [DEX] {kind} fetch error: {e}", flush=True)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_report(coins: list, dex_signals: list, run_date: str,
                 min_mcap: float, max_mcap: float) -> str:
    lines = []
    lines.append(f"# 🧠 Crypto Meta Intelligence Report")
    lines.append(f"**Date:** {run_date}  |  **Scope:** Base chain  |  "
                 f"**MCap range:** ${min_mcap/1e6:.2f}M – ${max_mcap/1e6:.0f}M\n")
    lines.append("---\n")

    # ── filter memes ──────────────────────────────────────────────────────────
    legit  = [c for c in coins if not _is_meme(c.get("name",""), c.get("symbol",""))]
    memes  = [c for c in coins if     _is_meme(c.get("name",""), c.get("symbol",""))]
    lines.append(f"*Scanned {len(coins)} tokens — {len(legit)} legit, {len(memes)} meme filtered*\n")

    # ── narrative buckets ──────────────────────────────────────────────────────
    bucket_map: dict = defaultdict(list)
    uncategorized   = []
    for c in legit:
        buckets = classify_narrative(c.get("name",""), c.get("symbol",""))
        if buckets:
            for b in buckets:
                bucket_map[b].append(c)
        else:
            uncategorized.append(c)

    # score each bucket
    bucket_rows = []
    for key, coin_list in bucket_map.items():
        avg = sum(momentum_score(c) for c in coin_list) / len(coin_list)
        top = sorted(coin_list, key=momentum_score, reverse=True)[:3]
        label = NARRATIVE_BUCKETS[key][1]
        bucket_rows.append((key, avg, len(coin_list), label, top))
    bucket_rows.sort(key=lambda x: x[1], reverse=True)

    # ── Section 1: Narrative leaderboard ──────────────────────────────────────
    lines.append("## 📊 Narrative Strength Leaderboard\n")
    lines.append("| Rank | Narrative | Tokens | Avg Momentum | Phase |")
    lines.append("|------|-----------|--------|-------------|-------|")
    for rank, (key, avg, count, label, _) in enumerate(bucket_rows, 1):
        bar   = "█" * int(avg/10) + "░" * (10 - int(avg/10))
        phase = phase_label(avg).split("—")[0].strip()
        lines.append(f"| {rank} | {label} | {count} | `{bar}` {avg:.0f}/100 | {phase} |")
    lines.append("")

    # ── Section 2: Deep dive — top 3 narratives ───────────────────────────────
    lines.append("## 🔍 Top Narrative Deep Dives\n")
    for key, avg, count, label, top_coins in bucket_rows[:3]:
        lines.append(f"### {label}")
        lines.append(f"**{count} tokens** — avg momentum {avg:.0f}/100 — {phase_label(avg)}\n")
        for c in top_coins:
            name  = c.get("name","")
            sym   = c.get("symbol","").upper()
            mcap  = c.get("market_cap") or 0
            c24h  = c.get("price_change_percentage_24h") or 0
            c7d   = c.get("price_change_percentage_7d_in_currency") or 0
            vol   = c.get("total_volume") or 0
            mom   = momentum_score(c)
            lines.append(
                f"- **{name} ({sym})** — MCap ${mcap:,.0f} | "
                f"24H {c24h:+.1f}% | 7D {c7d:+.1f}% | "
                f"Vol ${vol:,.0f} | Score {mom}/100"
            )
        lines.append("")

    # ── Section 3: Top 15 movers across all narratives ────────────────────────
    lines.append("## 🔥 Top 15 Movers (All Narratives)\n")
    top15 = sorted(legit, key=momentum_score, reverse=True)[:15]
    for i, c in enumerate(top15, 1):
        name  = c.get("name","")
        sym   = c.get("symbol","").upper()
        mcap  = c.get("market_cap") or 0
        c24h  = c.get("price_change_percentage_24h") or 0
        c7d   = c.get("price_change_percentage_7d_in_currency") or 0
        mom   = momentum_score(c)
        buckets = classify_narrative(name, sym)
        narr  = " + ".join(NARRATIVE_BUCKETS[b][1] for b in buckets) if buckets else "—"
        lines.append(f"{i:>2}. **{name} ({sym})** — ${mcap:,.0f} | 24H {c24h:+.1f}% | "
                     f"7D {c7d:+.1f}% | {mom}/100 | {narr}")
    lines.append("")

    # ── Section 4: DexScreener signals ────────────────────────────────────────
    if dex_signals:
        lines.append("## 📡 DexScreener Live Signals (Base chain)\n")
        for s in dex_signals[:15]:
            kind   = s["kind"]
            name   = s["name"] or "—"
            url    = s["url"]
            amount = s.get("amount", 0)
            tag    = f"[boosted ${amount:,}]" if kind == "boosted" and amount else f"[{kind}]"
            lines.append(f"- {tag} **{name}** — {url}")
        lines.append("")

    # ── Section 5: The meta verdict ────────────────────────────────────────────
    lines.append("## 🎯 The Meta Verdict\n")
    if bucket_rows:
        top_key, top_avg, top_count, top_label, top_coins = bucket_rows[0]
        lines.append(f"### Dominant: {top_label}")
        lines.append(f"{phase_label(top_avg)}")
        lines.append(f"- **{top_count} tokens** in this space, avg momentum **{top_avg:.0f}/100**")
        if top_coins:
            best = top_coins[0]
            lines.append(
                f"- Best in class: **{best.get('name','')} ({best.get('symbol','').upper()})** "
                f"${best.get('market_cap',0):,.0f} MCap"
            )
        lines.append("")

        if len(bucket_rows) > 1:
            sec_key, sec_avg, sec_count, sec_label, sec_coins = bucket_rows[1]
            lines.append(f"### Challenger: {sec_label}")
            lines.append(f"{phase_label(sec_avg)}")
            lines.append(f"- **{sec_count} tokens**, avg momentum **{sec_avg:.0f}/100**")
            if sec_coins:
                best2 = sec_coins[0]
                lines.append(
                    f"- Watch: **{best2.get('name','')} ({best2.get('symbol','').upper()})** "
                    f"${best2.get('market_cap',0):,.0f} MCap"
                )
            lines.append("")

        # Counter-intuitive call
        sleeping = [r for r in bucket_rows if r[1] < 18]
        if sleeping:
            sl_key, sl_avg, sl_count, sl_label, sl_coins = sleeping[0]
            lines.append(f"### 🔮 Counter-intuitive watch: {sl_label}")
            lines.append(
                f"Phase 0 / seed-level momentum ({sl_avg:.0f}/100) with {sl_count} tokens. "
                f"No one is talking about this. That's the point."
            )
            lines.append("")

    lines.append("---")
    lines.append(f"*Generated by meta_scanner.py — {run_date}*")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3-day crypto meta intelligence report")
    parser.add_argument("--output",    type=str,   default=None)
    parser.add_argument("--min-mcap",  type=float, default=500_000,
                        help="Min market cap USD (default $500K)")
    parser.add_argument("--max-mcap",  type=float, default=80_000_000,
                        help="Max market cap USD (default $80M)")
    parser.add_argument("--pages",     type=int,   default=4,
                        help="CoinGecko pages (default 4, = up to 400 coins)")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  🧠 Meta Scanner  {run_date}")
    print(f"  Range: ${args.min_mcap/1e6:.2f}M – ${args.max_mcap/1e6:.0f}M MCap")
    print(f"{sep}\n")

    print("  [1/3] Fetching Base ecosystem from CoinGecko...")
    coins = fetch_cg_base(args.pages, args.min_mcap, args.max_mcap)
    print(f"  → {len(coins)} tokens in range\n")

    print("  [2/3] Fetching DexScreener Base signals...")
    dex = fetch_dex_base_signals()
    print(f"  → {len(dex)} signals\n")

    print("  [3/3] Building meta report...")
    report = build_report(coins, dex, run_date, args.min_mcap, args.max_mcap)

    print(f"\n{sep}\n")
    print(report)
    print(f"\n{sep}\n")

    # save dated copy
    dated_fname = f"meta_report_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
    with open(dated_fname, "w") as f:
        f.write(report)
    print(f"  Saved: {dated_fname}")

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
