#!/usr/bin/env python3
"""Base-chain sub-$100K gem hunter.

Goal: alert ONLY on fundamentally real tokens under $100K market cap that are
starting to break out — before they make the move to $1M–$10M+.  Zero memes.

Hard requirements to alert (ALL must pass):
  • Market cap $5K–$100K  (micro-cap entry point)
  • Pair ≥ 14 days old    (past the rug window; real price history)
  • ≥ 75 h24 transactions (organic community, not a handful of bots)
  • Website               (public identity; ghost projects have no domain)
  • Twitter               (team publicly posts updates — behavioral signal)
  • Telegram              (community channel — behavioral signal)
  • FDV / MCap ≤ 5×       (limited dump-from-unlocks risk)
  • Liq / MCap ≥ 10%      (real pool depth)
  • Name not a meme       (two-layer phrase + word-boundary check)
  • Supply not meme-sized (420B / 69T / 1T+ = meme tokenomics, regardless of name)

Five signal stages (earliest → latest in the move):
  1. SLEEPING GIANT — dormant 30+ days, first hour of real volume (rarest/earliest)
  2. EARLY WAKEUP   — m5 or h1 volume suddenly 8× above the quiet baseline
  3. VOLUME SPIKE   — absolute volume threshold crossed on any timeframe
  4. BREAKOUT       — price clears its consolidation ceiling by ≥5% with volume
  5. STILL RUNNING  — price climbs ≥50% above last alert level (ride the full move)

Once a breakout fires, the token is tracked through the full run regardless of
mcap (so we get continuation alerts all the way from $100K to $10M+).
"""
import argparse
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import requests

from config import load_config
from dexscreener import DexScreener
from notifier import TelegramNotifier

log = logging.getLogger("bot")

SIGNAL_LOG_FILE = "signal_log.jsonl"

# ---------------------------------------------------------------------------
# RL model — loaded once at startup, used to augment conviction scores
# ---------------------------------------------------------------------------
try:
    from signal_rl import load_model as _rl_load_model, rl_score as _rl_score
    _RL_MODEL = _rl_load_model()
    log.info("RL model loaded: %d total observations", _RL_MODEL.n_total)
except Exception as _rl_exc:
    _RL_MODEL = None
    log.info("RL model not available (%s) — running without RL bonus", _rl_exc)


def _rl_bonus(alert):
    """Return LinUCB conviction bonus ∈ [-1.5, +1.5].

    Returns 0.0 if the model isn't trained yet or failed to load.
    The bonus is added to base conviction before the Telegram gate check,
    so signals in feature regions the model has learned to trust get
    a higher effective conviction, and regions with poor outcomes get lower.
    """
    if _RL_MODEL is None:
        return 0.0
    try:
        return _rl_score(_RL_MODEL, alert.get("type", ""), alert)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Signal logging — RL feedback loop
# ---------------------------------------------------------------------------

def _log_signal(alert, cfg_chain="base"):
    """Append one JSON line to signal_log.jsonl every time a Telegram alert fires.

    Fields captured are exactly what signal_tracker.py needs to compute outcomes:
      ts, date, token, name, symbol, chain, signal_type, conviction, mcap,
      price, liq, age_days, dims, h1_vol, h24_vol, pair_url
    """
    pair  = alert.get("pair") or {}
    token = pair.get("baseToken") or {}
    dims  = alert.get("conv_dims") or {}
    record = {
        "ts":          time.time(),
        "date":        datetime.now(tz=timezone.utc).isoformat(),
        "token":       token.get("address", ""),
        "name":        token.get("name", "?"),
        "symbol":      token.get("symbol", "?"),
        "chain":       cfg_chain,
        "signal_type": alert.get("type", "unknown"),
        "conviction":  alert.get("conviction", 0.0),
        "mcap":        alert.get("market_cap") or 0,
        "price":       _safe_float(pair.get("priceUsd")),
        "liq":         alert.get("liquidity", 0),
        "age_days":    alert.get("pair_age_days"),
        "h1_vol":      alert.get("h1_vol", 0),
        "h24_vol":     alert.get("h24_vol", 0),
        "buy_ratio":   alert.get("buy_ratio"),
        "dims": {
            "organic":    dims.get("organic",    0.0),
            "tokenomics": dims.get("tokenomics", 0.0),
            "maturity":   dims.get("maturity",   0.0),
            "momentum":   dims.get("momentum",   0.0),
        },
        "pair_url": pair.get("url", ""),
    }
    try:
        with open(SIGNAL_LOG_FILE, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        log.warning("signal_log write failed: %s", exc)


def _safe_float(v):
    """Convert a value to float, returning None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state(path):
    if os.path.exists(path):
        try:
            with open(path) as fh:
                state = json.load(fh)
            state.setdefault("monitored_tokens", [])
            state.setdefault("tokens", {})
            state.pop("pairs", None)
            return state
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read state file (%s); starting fresh", exc)
    return {"monitored_tokens": [], "tokens": {}}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_tokens(dex, cfg):
    """Collect Base-chain token addresses from all available sources."""
    chain = cfg["chain"]
    found = set()

    # 1. Screener's comprehensive GeckoTerminal + search feed — the main pipeline.
    screener_file = cfg.get("screener_addresses_file", "discovered_addresses.json")
    if os.path.exists(screener_file):
        try:
            with open(screener_file) as fh:
                disc = json.load(fh)
            for addr in disc.get(chain, []):
                if addr:
                    found.add(addr.lower())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("screener file unreadable: %s", exc)

    # 2. GeckoTerminal bootstrap results — 30-day price highs enrich our breakout reference.
    gecko_file = cfg.get("gecko_highs_file", "gecko_highs.json")
    if os.path.exists(gecko_file):
        try:
            with open(gecko_file) as fh:
                gecko_highs = json.load(fh)
            for addr in gecko_highs:
                if addr:
                    found.add(addr.lower())
        except (json.JSONDecodeError, OSError):
            pass

    # 3. DexScreener live profiles & boosts (catches newly promoted tokens fast).
    for item in dex.token_profiles_latest():
        if item.get("chainId") == chain and item.get("tokenAddress"):
            found.add(item["tokenAddress"].lower())
    for item in dex.token_boosts_latest() + dex.token_boosts_top():
        if item.get("chainId") == chain and item.get("tokenAddress"):
            found.add(item["tokenAddress"].lower())

    # 4. Free-text search terms.
    for term in cfg["discovery_search_terms"]:
        for pair in dex.search(term):
            if pair.get("chainId") != chain:
                continue
            for side in ("baseToken", "quoteToken"):
                addr = (pair.get(side) or {}).get("address")
                if addr:
                    found.add(addr.lower())

    # 5. micro_scanner.py watchlist — tokens with 2+ consecutive appearances = real signal.
    #    These are sub-$1M tokens that the 3-day intelligence scan caught as micro-gems.
    #    Feeding them into bot.py lets us monitor and alert on exact entry moments.
    micro_wl_file = cfg.get("micro_watchlist_file", "micro_watchlist.json")
    if os.path.exists(micro_wl_file):
        try:
            with open(micro_wl_file) as fh:
                micro_wl = json.load(fh)
            for addr, entry in (micro_wl.get("tokens") or {}).items():
                # Only pull tokens with multiple appearances — single-scan is noise
                if (entry.get("appearances") or 0) >= 2:
                    clean = addr.lower().replace("base_", "").replace("sol_", "")
                    if clean.startswith("0x"):
                        found.add(clean)
        except (json.JSONDecodeError, OSError) as exc:
            log.debug("micro_watchlist load error: %s", exc)

    # 6. GeckoTerminal trending + newest pools — fast catch of newly-active tokens
    #    even when screener.py hasn't cycled yet.
    _gt_base = "https://api.geckoterminal.com/api/v2/networks/base"
    for _endpoint in (f"{_gt_base}/trending_pools", f"{_gt_base}/new_pools"):
        try:
            r = requests.get(_endpoint,
                             params={"include": "base_token", "page": 1},
                             headers={"Accept": "application/json"},
                             timeout=10)
            if r.ok:
                for item in r.json().get("included", []):
                    if item.get("type") == "token":
                        addr = (item.get("attributes", {}).get("address") or "").lower()
                        if addr and addr.startswith("0x"):
                            found.add(addr)
        except Exception:
            pass

    return found


def _apply_gecko_highs(state, cfg):
    """Merge GeckoTerminal 30-day highs into per-token state (called once per cycle)."""
    gecko_file = cfg.get("gecko_highs_file", "gecko_highs.json")
    if not os.path.exists(gecko_file):
        return
    try:
        with open(gecko_file) as fh:
            gecko_highs = json.load(fh)
        for addr, high in gecko_highs.items():
            if not isinstance(high, (int, float)) or high <= 0:
                continue
            ts = state["tokens"].setdefault(addr.lower(), {})
            existing = ts.get("gecko_30d_high") or 0.0
            if high > existing:
                ts["gecko_30d_high"] = high
    except (json.JSONDecodeError, OSError):
        pass


# ---------------------------------------------------------------------------
# Pair aggregation — one view per token, not per pool
# ---------------------------------------------------------------------------

def _aggregate_by_token(pairs, monitored_set):
    """Group DexScreener pairs by base token address.

    Volume and liquidity are summed across all pools so we see the true
    token-level activity (Uniswap v2 + Aerodrome + etc. combined).
    The highest-liquidity pool is kept as the canonical pair for price/URL.
    Price change is taken from the highest-liquidity pool (most reliable).
    """
    tokens = {}
    for pair in pairs:
        base = pair.get("baseToken") or {}
        addr = (base.get("address") or "").lower()
        if not addr or addr not in monitored_set:
            continue
        if addr not in tokens:
            tokens[addr] = {
                "base":             base,
                "best_pair":        pair,
                "total_liq":        0.0,
                "total_vol":        {"m5": 0.0, "h1": 0.0, "h6": 0.0, "h24": 0.0},
                "total_buys_h1":    0,  "total_sells_h1":    0,
                "total_buys_h6":    0,  "total_sells_h6":    0,
                "total_buys_h24":   0,  "total_sells_h24":   0,
                "pair_created_at":  None,  # oldest pairCreatedAt in ms = true token age
            }
        t = tokens[addr]
        liq = (pair.get("liquidity") or {}).get("usd") or 0.0
        if liq > ((t["best_pair"].get("liquidity") or {}).get("usd") or 0.0):
            t["best_pair"] = pair
        t["total_liq"] += liq
        vol = pair.get("volume") or {}
        for tf in ("m5", "h1", "h6", "h24"):
            t["total_vol"][tf] += vol.get(tf) or 0.0
        for tf_key, bk, sk in [("h1",  "total_buys_h1",  "total_sells_h1"),
                                ("h6",  "total_buys_h6",  "total_sells_h6"),
                                ("h24", "total_buys_h24", "total_sells_h24")]:
            tx = (pair.get("txns") or {}).get(tf_key) or {}
            t[bk] += tx.get("buys") or 0
            t[sk] += tx.get("sells") or 0
        # Track oldest pair creation time (= when token first appeared on-chain)
        pair_ts = pair.get("pairCreatedAt")
        if pair_ts and (t["pair_created_at"] is None or pair_ts < t["pair_created_at"]):
            t["pair_created_at"] = pair_ts
    for t in tokens.values():
        t["price_change"] = t["best_pair"].get("priceChange") or {}
    return tokens


# ---------------------------------------------------------------------------
# Quality gate — hard shitcoin / rug filters applied before any signal fires
# ---------------------------------------------------------------------------

def _quality_filter(token_data, cfg, now):
    """Hard quality gate for the sub-$100K micro-cap tier.

    Every check targets a specific class of garbage:
      age ≥ 7d           — past the initial 48-72h rug window (config: min_pair_age_days)
      txns ≥ 75          — real organic community, not one whale and a few bots
      FDV/MCap ≤ 5×      — team didn't pre-mint 80% of supply to dump later
      website required   — bare-minimum public identity; ghost projects have no domain
      Twitter required   — real team publicly posts updates (behavioral signal)
      Telegram required  — real community channel (behavioral signal)
      meme/clone filter  — block doge/pepe clones and pump vocabulary in the name
      liq/mcap ≥ 10%    — pool has real depth; prices that move cleanly
      price stable       — not catching a -40%+ falling knife
      meme supply check  — block canonical meme supply sizes (420B, 69T, 1T+, etc.)
    """
    qf   = cfg.get("quality_filter", {})
    best = token_data["best_pair"]
    base = best.get("baseToken") or {}

    # 1. Pair age
    min_age_days = qf.get("min_pair_age_days", 7)
    pair_ts      = token_data.get("pair_created_at")
    if pair_ts and min_age_days > 0:
        age_days = (now * 1000 - pair_ts) / 86_400_000
        if age_days < min_age_days:
            return False, f"pair too new ({age_days:.1f}d < {min_age_days}d)"

    # 2. Organic trading depth
    min_txns = qf.get("min_h24_txns", 50)
    txns_h24 = token_data["total_buys_h24"] + token_data["total_sells_h24"]
    if txns_h24 < min_txns:
        return False, f"thin trading ({txns_h24} h24 txns < {min_txns})"

    # 3. FDV / MCap ratio
    max_ratio = qf.get("max_fdv_mcap_ratio", 5.0)
    mcap      = best.get("marketCap") or 0
    fdv       = best.get("fdv")       or 0
    if fdv > 0 and mcap > 0 and fdv > mcap * max_ratio:
        return False, f"FDV/MCap={fdv/mcap:.1f}x > {max_ratio:.0f}x"

    # 4. Website — real projects have a domain
    _info    = best.get("info") or {}
    websites = _info.get("websites") or []
    if not websites:
        return False, "no website"

    # 5a. Twitter — real teams publicly announce and post updates
    socials      = _info.get("socials") or []
    social_types = {(s.get("type") or "").lower() for s in socials}
    if "twitter" not in social_types:
        return False, "no Twitter (real projects post updates publicly)"

    # 5b. Telegram — optional for micro-cap tier.
    #     Many legitimate early-stage projects ($5K–$100K) only have Twitter at launch.
    #     Telegram presence is rewarded in _score_project_maturity() as a bonus, but
    #     hard-requiring it here eliminates too many real sub-$100K opportunities.
    #     Set require_telegram: true in config to re-enable the hard gate.
    if qf.get("require_telegram", False) and "telegram" not in social_types:
        return False, "no Telegram (require_telegram: true in config)"

    # 6. Meme / clone filter — name-based two-layer check (phrase + whole-word boundary)
    if qf.get("filter_meme_names", True):
        is_meme, meme_reason = _is_meme_token(
            base.get("name") or "", base.get("symbol") or "")
        if is_meme:
            return False, meme_reason

    # 7. Liquidity backing
    min_liq_pct = qf.get("min_liq_mcap_pct", 10)
    liq         = token_data["total_liq"]
    if mcap > 0 and min_liq_pct > 0 and liq / mcap * 100 < min_liq_pct:
        return False, f"liq/mcap={liq/mcap*100:.1f}% < {min_liq_pct}%"

    # 8. Price not in freefall
    try:
        h24_chg = float((best.get("priceChange") or {}).get("h24") or 0)
        if h24_chg < -40:
            return False, f"price crashed {h24_chg:.0f}% in 24h (dump in progress)"
    except (TypeError, ValueError):
        pass

    # 9. Meme tokenomics fingerprint — name-independent, behavioral check.
    #    Canonical meme supply sizes (420B, 69T, 1T+) betray a meme regardless
    #    of whether the token name sounds legitimate.  Real projects don't deploy
    #    with 420 billion tokens.
    if qf.get("filter_meme_names", True):
        is_meme_sup, sup_reason = _is_meme_supply(
            best.get("fdv"), best.get("priceUsd"))
        if is_meme_sup:
            return False, sup_reason

    return True, ""


# ---------------------------------------------------------------------------
# Meme token detection — three independent layers
# ---------------------------------------------------------------------------
#
# The goal is ZERO meme alerts.  Name matching alone is insufficient because:
#   • Meme deployers increasingly use "legit-sounding" names (e.g. "BaseAI Vault")
#   • Name filtering has to be maintained manually as new memes emerge
#   • A truly fundamental approach looks at on-chain STRUCTURE and BEHAVIOR
#
# Layer 1 (_MEME_CLONE_PATTERNS): phrase / substring patterns.
#   These are specific enough that a substring match is reliable.
#   e.g. "pepe" can't appear in a real project name that isn't a meme.
#
# Layer 2 (_MEME_WORD_PATTERNS + _MEME_WORD_RE): whole-word boundary matching.
#   These are common English words that only mean "meme" when they ARE the
#   token's identity (not buried inside a compound word).
#   "cat" → meme;  "Catalyst" → not a meme  (word boundary prevents false match)
#   "moon" → meme; "Moonwell" → not a meme
#
# Layer 3 (_MEME_SUPPLY_RANGES + _is_meme_supply): tokenomics fingerprinting.
#   Meme deployers choose supply sizes for psychological effect, not utility:
#   420B for the degen number, 1T because "looks big", 69B for the meme number.
#   Real DeFi/infra projects design supply around governance/staking economics
#   and virtually never land on these specific canonical meme figures.
#   This check fires regardless of the token name — catches renamed memes.
#
# Social completeness (in _quality_filter):
#   Requiring BOTH Twitter AND Telegram is behavioral, not name-based.
#   Real teams running real products post updates on Twitter and maintain a
#   Telegram for community support.  Anon pump groups typically only set up
#   one (usually Telegram for the pump channel) and skip the other.

_MEME_CLONE_PATTERNS = {
    # ── Crypto knockoffs ──────────────────────────────────────────────────────
    "litecoin", "litecoins", "litcoin",  # litcoin = litecoin misspelling/clone
    "bitcoins", "ethereums", "solanas",
    # ── Doge / Shib family ────────────────────────────────────────────────────
    "doge", "dogecoin", "doggy", "doggo", "dogelon",
    "shib", "shiba", "shibinu", "shibainu", "shib inu",
    "floki", "bonk", "wif", "dogwifhat", "dog wif hat",
    # ── Pepe / frog ───────────────────────────────────────────────────────────
    "pepe", "pepecoin", "wojak", "kek", "kekius",
    "froggo",
    # ── Inu suffix (space-anchored catches "X inu" without touching "continue")
    " inu",
    # ── Cat memes ─────────────────────────────────────────────────────────────
    "nyan", "nyah", "meow",
    "skitten",             # Ski Mask Kitten (specific Base meme)
    "garfield",            # Garfield cat meme clones
    "whiskers",            # "Mr Whiskers" style meme names
    "felix",               # Felix the Cat meme clones
    "grumpy cat",          # viral meme
    # ── Pump vocabulary (phrase forms) ───────────────────────────────────────
    "moonshot", "mooning", "to the moon", "wen moon",
    "100x", "1000x", "10000x", "50x", "200x", "500x",
    "lambo", "lamborghini",
    "millionaire", "billionaire",
    "wagmi", "ngmi",
    "get rich", "getrich",
    "safemoon", "safemars",
    "fair launch",
    "rug pull",
    # ── Celebrity / political bait ────────────────────────────────────────────
    "elon", "musk",
    "trump", "maga",
    "biden",
    # ── AI-name squatters ─────────────────────────────────────────────────────
    "gpt coin", "chatgpt", "openai coin", "ai coin",
    # ── Numbers that are always meme ──────────────────────────────────────────
    "420", "69",
    # ── Known named memes on Base ─────────────────────────────────────────────
    "toshi",       # Coinbase cat clone
    "normie",      # Base meme token
    "brettcoin",
    # ── Meme characters / internet figures ───────────────────────────────────
    "apustaja",    # Apu Apustaja — Finnish "helper frog" meme (Apu Apustaja token)
    "apustus",     # alternate spelling
    "pepenomics",  # Pepe economics = meme finance
    "cheems",      # Cheems dog meme
    "doomba",      # meme roomba dog
    "milady",      # Milady meme collection
    "remilia",     # Remilia / Milady creator alias
    # ── Crypto-specific slang used only as meme / pump names ─────────────────
    "to the moon", "wen moon",
    "ser ",        # "ser" as standalone prefix (e.g. "Ser Token") — pure CT slang
}

# Standalone words that flag a meme when they appear as a WHOLE WORD in name/symbol.
# Whole-word matching prevents: "cat" blocking "Catalyst", "moon" blocking "Moonwell".
_MEME_WORD_PATTERNS = frozenset({
    # Animals — almost universally memes at micro-cap level
    "cat", "cats",
    "kitten", "kittens",   # Ski Mask Kitten, etc.
    "kitty", "kitties",    # Hello Kitty clones
    "puss", "pussy",       # cat variants
    "purr",                # cat sound = cat meme
    "mittens",             # cat meme
    "tabby",               # cat breed = meme
    "dog", "dogs",
    "frog", "frogs",
    "ape", "apes",
    "chimp", "gorilla", "monkey",
    "penguin", "penguins",
    "hamster", "hamsters",
    "rabbit", "rabbits", "bunny", "bunnies",
    "duck", "ducks",
    "snake", "snakes",
    "bear",     # "bear" alone = meme; "Bearish Protocol" would clear on "Protocol" below
    "bull",     # same
    "rat", "rats",
    "pig", "hog",
    "cow", "cows",
    "horse", "horses", "pony",
    "fish",
    "crab", "crabs",
    "fox",
    "wolf", "wolves",
    "lion", "tiger", "cheetah",
    "dino", "dinosaur",
    "chicken",
    # Pump vocabulary as standalone words
    "moon",
    "rocket",
    "pump",
    "rich",
    "gem",
    # Meme culture / internet slang as token name
    "chad",
    "sigma",
    "pepe",
    "brett",    # famous Base meme
    "bald",     # Base meme
    "higher",   # Base meme (cultural token)
    "apu",      # Apu Apustaja meme character
    # Crypto Twitter / CT slang — exclusively used by meme / pump deployers
    "fren",  "frens",     # crypto slang for "friend" — never a real project name
    "anon",               # anonymous culture — always pump/meme at micro-cap
    "degen",              # degen finance = meme/gamble culture at micro-cap
    "ngmi",  "wagmi",     # already in clone list but add word-boundary form too
    "wen",                # "wen moon", "wen lambo" — pump vocabulary
    "gm",                 # good morning = CT meme culture token
    "simp",               # internet slang/meme
    "shill",              # crypto shill = bad actor signal
    "rug",                # rug pull vocabulary
    # Generic launchers
    "baby",
    "mini",
    "safe",
})

# Pre-compiled regex for efficient whole-word matching (called 1000+ times/cycle)
_MEME_WORD_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in _MEME_WORD_PATTERNS) + r')\b'
)


def _is_meme_token(name, symbol):
    """Two-layer name-based meme detection. Returns (is_meme: bool, reason: str).

    Designed for zero false negatives: if in doubt, classify as meme.
    The user wants ZERO meme alerts; missing a real project is acceptable.
    """
    name_l   = (name   or "").lower().strip()
    sym_l    = (symbol or "").lower().strip()
    combined = name_l + " " + sym_l

    # Layer 1: phrase / substring match
    for pat in _MEME_CLONE_PATTERNS:
        if pat in combined:
            return True, f"meme pattern '{pat}'"

    # Layer 2: whole-word match — catches standalone animal/pump words
    m = _MEME_WORD_RE.search(combined)
    if m:
        return True, f"meme word '{m.group()}'"

    return False, ""


# ---------------------------------------------------------------------------
# Meme supply fingerprinting (Layer 3) — behavioral, name-independent
# ---------------------------------------------------------------------------
#
# Meme deployers choose supply for psychological impact:
#   • 420,000,000,000  (420B) — the degen/weed number
#   • 69,000,000,000   (69B)  — the other degen number
#   • 69,420,000,000   (69.42B) — combined
#   • 1,000,000,000,000 (1T) — "sounds big and cheap per token"
#   • 100,000,000,000,000 (100T) — hyper-inflated clone
#   • > 1e14           (>100T) — anything above = definitionally meme
#
# Real DeFi/infrastructure projects design supply around:
#   governance weights, staking emission schedules, vesting math.
#   They use utilitarian numbers (100M, 500M, 1B at most).
#
# We compute approximate total supply as: fdv / price_usd
# (±2% tolerance to absorb floating-point drift in DexScreener's fields)

_MEME_SUPPLY_RANGES = [
    # ── 420 family ────────────────────────────────────────────────────────────
    (420_000_000       * 0.98, 420_000_000       * 1.02),   # 420M
    (420_000_000_000   * 0.98, 420_000_000_000   * 1.02),   # 420B
    (420_690_000_000   * 0.98, 420_690_000_000   * 1.02),   # 420.69B
    (4_200_000_000_000 * 0.98, 4_200_000_000_000 * 1.02),   # 4.2T
    # ── 69 family ─────────────────────────────────────────────────────────────
    (69_000_000        * 0.98, 69_000_000        * 1.02),   # 69M
    (69_000_000_000    * 0.98, 69_000_000_000    * 1.02),   # 69B
    (69_420_000_000    * 0.98, 69_420_000_000    * 1.02),   # 69.42B
    (6_900_000_000_000 * 0.98, 6_900_000_000_000 * 1.02),   # 6.9T
    (69_000_000_000_000 * 0.98, 69_000_000_000_000 * 1.02), # 69T
    # ── Trillion-scale supplies ────────────────────────────────────────────────
    # Real projects virtually never exceed 10B circulating supply.
    # Any trillion supply is chosen purely to make per-token price "look cheap".
    (1_000_000_000_000  * 0.98, 1_000_000_000_000  * 1.02), # 1T
    (10_000_000_000_000 * 0.98, 10_000_000_000_000 * 1.02), # 10T
    (100_000_000_000_000 * 0.98, 100_000_000_000_000 * 1.02), # 100T
]


def _supply_label(n):
    """Human-readable supply label (e.g. 420B, 69T, 1.2Q)."""
    if n >= 1e15: return f"{n/1e15:.3g}Q"   # quadrillion
    if n >= 1e12: return f"{n/1e12:.3g}T"   # trillion
    if n >= 1e9:  return f"{n/1e9:.3g}B"    # billion
    if n >= 1e6:  return f"{n/1e6:.3g}M"    # million
    return f"{int(n):,}"


def _is_meme_supply(fdv, price_usd):
    """Layer 3 meme detection: tokenomics fingerprint via supply-size check.

    Computes approximate total supply = fdv / price_usd and checks whether it
    matches a canonical meme supply figure.  Completely independent of the
    token name — catches renamed memes and new meme variants on day one.

    Returns (is_meme: bool, reason: str).
    """
    try:
        fdv_f   = float(fdv       or 0)
        price_f = float(price_usd or 0)
    except (TypeError, ValueError):
        return False, ""

    if fdv_f <= 0 or price_f <= 0:
        return False, ""

    approx_supply = fdv_f / price_f

    # Hard ceiling: anything above 100 trillion is definitionally meme tokenomics.
    # No legitimate DeFi or infrastructure project needs a quadrillion tokens.
    if approx_supply > 1e14:
        return True, f"meme tokenomics: {_supply_label(approx_supply)} total supply"

    # Check canonical meme supply sizes (±2% tolerance)
    for lo, hi in _MEME_SUPPLY_RANGES:
        if lo <= approx_supply <= hi:
            mid = (lo + hi) / 2.0
            return True, f"meme supply: ~{_supply_label(mid)}"

    return False, ""


def _quality_filter_midcap(token_data, cfg, now):
    """Hard quality gate for the $100K–$1M mid-cap tier.

    Goal: keep only projects a real founder shipped — not memes, clones, or
    anonymous pumps.  Every check here has a specific target class it eliminates:

      age ≥ 30d        — past the rug window; survived at least one full market cycle
      txns ≥ 200       — real community trading, not a whale running a few wallets
      FDV/MCap ≤ 3×    — team didn't give themselves 90% unlocked supply
      website required — project has a public identity beyond a token address
      Twitter required — team is posting updates, not anonymous ghosts
      Telegram required— active community channel; not a dead Discord
      liq/mcap ≥ 20%   — pool is meaningfully backed; price can actually move cleanly
      price stable     — not currently in a -50%+ dump (catching a falling knife)
      meme/clone check — block token names that match known pump-and-dump archetypes
    """
    qf   = cfg.get("quality_filter_midcap", {})
    best = token_data["best_pair"]
    base = best.get("baseToken") or {}

    # ── 1. Pair age ──────────────────────────────────────────────────────────
    min_age_days = qf.get("min_pair_age_days", 30)
    pair_ts      = token_data.get("pair_created_at")
    if pair_ts and min_age_days > 0:
        age_days = (now * 1000 - pair_ts) / 86_400_000
        if age_days < min_age_days:
            return False, f"pair too new ({age_days:.1f}d < {min_age_days}d)"

    # ── 2. Community depth ───────────────────────────────────────────────────
    min_txns = qf.get("min_h24_txns", 200)
    txns_h24 = token_data["total_buys_h24"] + token_data["total_sells_h24"]
    if txns_h24 < min_txns:
        return False, f"thin trading ({txns_h24} h24 txns < {min_txns})"

    # ── 3. FDV / MCap ratio ──────────────────────────────────────────────────
    max_ratio = qf.get("max_fdv_mcap_ratio", 3.0)
    mcap      = best.get("marketCap") or 0
    fdv       = best.get("fdv")       or 0
    if fdv > 0 and mcap > 0 and fdv > mcap * max_ratio:
        return False, f"FDV/MCap={fdv/mcap:.1f}x > {max_ratio:.0f}x"

    # ── 4. Website + Twitter required; Telegram optional ────────────────────
    info     = best.get("info") or {}
    websites = info.get("websites") or []
    socials  = info.get("socials")  or []
    social_types = {(s.get("type") or "").lower() for s in socials}
    if not websites:
        return False, "no website"
    if "twitter" not in social_types:
        return False, "no Twitter (mid-cap requires team account)"
    # Telegram optional by default — many legitimate mid-cap projects use Discord
    # or only have Twitter at this stage.  Set require_telegram: true to re-enable.
    if qf.get("require_telegram", False) and "telegram" not in social_types:
        return False, "no Telegram (require_telegram: true in config)"

    # ── 5. Liquidity backing ─────────────────────────────────────────────────
    min_liq_pct = qf.get("min_liq_mcap_pct", 20)
    liq = token_data["total_liq"]
    if mcap > 0 and liq / mcap * 100 < min_liq_pct:
        return False, f"liq/mcap={liq/mcap*100:.1f}% < {min_liq_pct}%"

    # ── 6. Price stability — not catching a falling knife ────────────────────
    try:
        h24_chg = float((best.get("priceChange") or {}).get("h24") or 0)
        if h24_chg < -50:
            return False, f"price crashed {h24_chg:.0f}% in 24h (dump in progress)"
    except (TypeError, ValueError):
        pass

    # ── 7. Meme / clone filter — name-based ─────────────────────────────────
    if qf.get("filter_meme_names", True):
        is_meme, meme_reason = _is_meme_token(
            base.get("name") or "", base.get("symbol") or "")
        if is_meme:
            return False, meme_reason

    # ── 8. Meme tokenomics fingerprint — name-independent ───────────────────
    #    Canonical meme supply sizes betray a meme regardless of name.
    if qf.get("filter_meme_names", True):
        is_meme_sup, sup_reason = _is_meme_supply(
            best.get("fdv"), best.get("priceUsd"))
        if is_meme_sup:
            return False, sup_reason

    return True, ""


def _quality_filter_largecap(token_data, cfg, now):
    """Hard quality gate for the $1M–$10M emerging large-cap tier.

    These are already-established projects — we demand full social presence,
    deep liquidity, clean tokenomics, and 2+ months of proven existence.
    """
    qf   = cfg.get("quality_filter_largecap", {})
    best = token_data["best_pair"]
    base = best.get("baseToken") or {}

    # ── 1. Pair age — must have survived at least 2 months ──────────────────
    min_age_days = qf.get("min_pair_age_days", 60)
    pair_ts = token_data.get("pair_created_at")
    if pair_ts and min_age_days > 0:
        age_days = (now * 1000 - pair_ts) / 86_400_000
        if age_days < min_age_days:
            return False, f"pair too new ({age_days:.1f}d < {min_age_days}d)"

    # ── 2. Community depth — deep daily trading activity ────────────────────
    min_txns = qf.get("min_h24_txns", 400)
    txns_h24 = token_data["total_buys_h24"] + token_data["total_sells_h24"]
    if txns_h24 < min_txns:
        return False, f"thin trading ({txns_h24} h24 txns < {min_txns})"

    # ── 3. FDV / MCap — strict supply cleanliness ────────────────────────────
    max_ratio = qf.get("max_fdv_mcap_ratio", 2.0)
    mcap = best.get("marketCap") or 0
    fdv  = best.get("fdv")       or 0
    if fdv > 0 and mcap > 0 and fdv > mcap * max_ratio:
        return False, f"FDV/MCap={fdv/mcap:.1f}x > {max_ratio:.0f}x"

    # ── 4. Full social stack ─────────────────────────────────────────────────
    info         = best.get("info") or {}
    websites     = info.get("websites") or []
    socials      = info.get("socials")  or []
    social_types = {(s.get("type") or "").lower() for s in socials}
    if not websites:
        return False, "no website"
    if "twitter" not in social_types:
        return False, "no Twitter"
    if "telegram" not in social_types:
        return False, "no Telegram"

    # ── 5. Liquidity depth ───────────────────────────────────────────────────
    min_liq_pct = qf.get("min_liq_mcap_pct", 10)
    liq = token_data["total_liq"]
    if mcap > 0 and liq / mcap * 100 < min_liq_pct:
        return False, f"liq/mcap={liq/mcap*100:.1f}% < {min_liq_pct}%"

    # ── 6. Not catching a dump ───────────────────────────────────────────────
    try:
        h24_chg = float((best.get("priceChange") or {}).get("h24") or 0)
        if h24_chg < -40:
            return False, f"price crashed {h24_chg:.0f}% (dump in progress)"
    except (TypeError, ValueError):
        pass

    # ── 7. Meme / clone filter ────────────────────────────────────────────────
    if qf.get("filter_meme_names", True):
        is_meme, reason = _is_meme_token(base.get("name") or "", base.get("symbol") or "")
        if is_meme:
            return False, reason

    return True, ""


# ---------------------------------------------------------------------------
# CoinGecko enrichment — VC / Ivy League backing detection
# ---------------------------------------------------------------------------

# CoinGecko category tags assigned to tokens in known VC portfolios.
_VC_PORTFOLIO_CATEGORIES = {
    "coinbase ventures portfolio", "a16z portfolio", "paradigm portfolio",
    "polychain capital portfolio", "multicoin capital portfolio",
    "framework ventures portfolio", "delphi digital portfolio",
    "1confirmation", "placeholder portfolio", "dragonfly capital portfolio",
    "sequoia capital portfolio", "pantera capital portfolio",
    "electric capital portfolio", "variant portfolio",
    "y combinator alumni", "binance labs portfolio", "coinbase backed",
}

# Keywords to look for in project descriptions / website text.
_VC_DESCRIPTION_KEYWORDS = [
    "coinbase ventures", "a16z", "andreessen horowitz", "paradigm capital",
    "sequoia", "pantera capital", "binance labs", "multicoin capital",
    "framework ventures", "delphi digital", "polychain capital",
    "dragonfly capital", "electric capital", "variant fund",
    "y combinator", "yc alum", "seed funding", "series a",
    "strategic investors", "notable investors", "backed by",
    "funded by", "raised $", "million funding", "investment round",
]


def _coingecko_enrich(token_addr):
    """One-time CoinGecko lookup for a Base token address.

    Returns a dict summarising VC backing, community stats, and CG listing.
    Returns {"listed": False} on any miss or error.
    Rate limit: ~30 req/min on the free CG API; callers must throttle.
    """
    try:
        url = (f"https://api.coingecko.com/api/v3"
               f"/coins/base/contract/{token_addr}")
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=10)
        if not r.ok:
            return {"listed": False, "status": r.status_code}
        data = r.json()

        cats_raw = [c.lower() for c in (data.get("categories") or [])]
        vc_cats  = [c for c in cats_raw if c in _VC_PORTFOLIO_CATEGORIES]

        desc       = (data.get("description") or {}).get("en") or ""
        desc_lower = desc.lower()
        kw_hits    = [kw for kw in _VC_DESCRIPTION_KEYWORDS if kw in desc_lower]

        comm = data.get("community_data")   or {}
        dev  = data.get("developer_data")   or {}

        return {
            "listed":            True,
            "id":                data.get("id"),
            "name":              data.get("name"),
            "symbol":            data.get("symbol"),
            "vc_categories":     vc_cats,
            "vc_keywords":       kw_hits,
            "has_vc_backing":    bool(vc_cats or kw_hits),
            "twitter_followers": comm.get("twitter_followers", 0),
            "telegram_members":  comm.get("telegram_channel_user_count", 0),
            "github_stars":      dev.get("stars", 0),
            "coingecko_score":   data.get("coingecko_score", 0),
            "description":       desc[:300],
        }
    except Exception:
        return {"listed": False}


# ---------------------------------------------------------------------------
# Multi-dimensional conviction scoring — geometric mean of 4 independent axes
# ---------------------------------------------------------------------------
#
# PHILOSOPHY
# ──────────
# Legitimate emerging tokens on Base chain show ONE specific fingerprint:
# ALL four fundamental dimensions are simultaneously healthy.  Memes, pumps,
# and rugs always break down in at least one dimension — usually more.
#
# The four dimensions are INDEPENDENT: they measure completely different things.
#   D_organic    — Is this trading COMMUNITY-DRIVEN or WHALE/BOT-MANIPULATED?
#   D_tokenomics — Are the ECONOMICS designed for utility, or for a pump exit?
#   D_maturity   — Has this project SURVIVED long enough to be real?
#   D_momentum   — Is the current move AUTHENTIC and BUILDING?
#
# WHY GEOMETRIC MEAN (not additive sum)
# ──────────────────────────────────────
# Additive scoring lets a token compensate:  weak tokenomics + massive momentum
# = falsely high score.  That's exactly the pump-and-dump pattern.
#
# Geometric mean = (D1 × D2 × D3 × D4)^(1/4)
#
#   • A 0 in ANY dimension → overall = 0  (absolute block)
#   • (10, 10, 10,  3) → 7.40  vs additive 8.25  (penalized for weak dimension)
#   • (10, 10,  3, 10) → 7.40  (doesn't matter WHICH dimension is weak)
#   • (7,   7,  7,  7) → 7.00  (balanced is rewarded, unbalanced is not)
#   • Any dimension < 2.5 → hard floor → returns 0.0
#
# The ONLY way to score HIGH (≥7) is to be genuinely strong in ALL four areas.
#
# ORGANIC SCORE uses a neutral 5.0 baseline so thin-data tokens (like a
# sleeping giant on its first hour of activity) aren't falsely penalized.
# Evidence of manipulation drives it DOWN; evidence of organicity drives it UP.
# "No data" stays neutral — absence of evidence is not evidence of manipulation.

_CONV_FLOOR = 2.0   # any dimension below this hard-kills the conviction score
# Lowered 2.5 → 2.0: early-stage tokens (7-14d, first observation) legitimately score
# low on maturity — we want to catch them before they're "proven", not after.


def _score_organic_trading(token_data):
    """D_organic — Is this trading community-driven or whale/bot-manipulated?

    Key insight: organic trading has a SPECIFIC behavioral fingerprint:
      • Buy/sell ratio is CONSISTENT across ALL timeframes (h1 ≈ h6 ≈ h24)
        — Real community interest doesn't suddenly appear for one hour then vanish
        — Pump-and-dump: h1 buy ratio sharply HIGHER than h24 (exits visible in h24)
      • Volume is distributed across the day (not 90% crammed into one hour)
        — Exception: sleeping giants ARE expected to be h1-concentrated on wakeup
      • Many small trades > few large ones (community grassroots, not whale control)

    Starts at 5.0 (neutral) — shifts UP for evidence of organicity,
    DOWN for evidence of manipulation. Thin data stays neutral (5.0).
    """
    score = 5.0   # neutral baseline: no data is not evidence of manipulation

    buys_h1  = token_data["total_buys_h1"]
    sells_h1 = token_data["total_sells_h1"]
    txns_h1  = buys_h1 + sells_h1
    buys_h6  = token_data["total_buys_h6"]
    sells_h6 = token_data["total_sells_h6"]
    txns_h6  = buys_h6 + sells_h6
    buys_h24  = token_data["total_buys_h24"]
    sells_h24 = token_data["total_sells_h24"]
    txns_h24  = buys_h24 + sells_h24

    br_h1  = (buys_h1  / txns_h1)  if txns_h1  >= 3  else None
    br_h6  = (buys_h6  / txns_h6)  if txns_h6  >= 5  else None
    br_h24 = (buys_h24 / txns_h24) if txns_h24 >= 10 else None

    # ── Cross-timeframe buy ratio consistency (MOST IMPORTANT signal) ──────────
    # Compare h1 buy ratio to h24 buy ratio.  Spread tells us if exits are
    # happening RIGHT NOW during a pump (h1 buyers >> h24 average = distributions).
    #
    # CRITICAL design choice: when the spread is large (exits in progress), we
    # do NOT award any "healthy timeframe" bonuses below — the cross-timeframe
    # inconsistency IS the signal and individual timeframe readings are misleading.
    cross_spread = 0.0
    if br_h1 is not None and br_h24 is not None:
        cross_spread = br_h1 - br_h24
        if cross_spread >= 0.30:     # h1 buyers 30%+ above h24 norm → exits in progress
            score -= 3.5
        elif cross_spread >= 0.20:   # 20% gap → moderate concern
            score -= 2.0
        elif cross_spread >= 0.12:   # 12% gap → slight concern
            score -= 0.8
        elif -0.08 <= cross_spread <= 0.12:
            score += 1.5             # tight spread → consistent organic interest

    # ── Sustained healthy buy pressure across timeframes ───────────────────────
    # Award points ONLY when cross-timeframe consistency is intact (spread < 15%).
    # If exits are in progress (large spread), individual "healthy" readings are
    # misleading — the inconsistency pattern dominates.
    if cross_spread < 0.15:
        healthy_tfs = sum(1 for r in [br_h1, br_h6, br_h24]
                          if r is not None and 0.45 <= r <= 0.72)
        score += healthy_tfs * 0.75    # max +2.25 for all three balanced

    # ── Volume concentration (soft penalty only — sleeping giants exempt) ──────
    # If 85%+ of the day's volume happened in the last hour, it MAY be a burst pump.
    # This is a soft signal: penalty is small because sleeping giants legitimately
    # show high h1 concentration on their first wakeup hour.
    vol_h1  = token_data["total_vol"]["h1"]
    vol_h24 = token_data["total_vol"]["h24"]
    if vol_h24 > 0:
        h1_conc = vol_h1 / vol_h24
        if h1_conc > 0.90:
            score -= 1.0    # very concentrated — soft flag, not a kill
        elif h1_conc > 0.75:
            score -= 0.4
        elif h1_conc <= 0.30:
            score += 1.0    # volume spread across the day = organic accumulation

    # ── Transaction size: community depth vs whale dominance ──────────────────
    # Many small trades = grassroots community interest.
    # Few enormous trades = whale/bot manipulation, not real adoption.
    # Measure: average txn value as a fraction of pool liquidity.
    # A ratio of 1.0 means each trade is moving 100% of the pool — pathological.
    liq = token_data["total_liq"]
    if txns_h24 >= 10 and vol_h24 > 0 and liq > 0:
        avg_txn_as_pct_of_liq = (vol_h24 / txns_h24) / liq
        if avg_txn_as_pct_of_liq < 0.003:   # avg txn < 0.3% of pool → micro-grassroots
            score += 1.5
        elif avg_txn_as_pct_of_liq < 0.01:  # < 1% → healthy retail-sized trades
            score += 0.75
        elif avg_txn_as_pct_of_liq > 0.20:  # > 20% of pool per tx → extreme whale/bot
            score -= 3.5
        elif avg_txn_as_pct_of_liq > 0.08:  # > 8% → significant whale activity
            score -= 2.0
        elif avg_txn_as_pct_of_liq > 0.03:  # > 3% → borderline
            score -= 0.75

    return max(0.0, min(10.0, round(score, 1)))


def _score_tokenomics(token_data):
    """D_tokenomics — Are the economics designed for utility, or for a pump exit?

    Key insight: meme/pump token deployers make SPECIFIC choices about economics
    that are completely independent of the token name:
      • Supply size: 420B, 69T, 1T are chosen for psychological manipulation
        ("looks cheap per unit") — real projects choose supply for governance math
      • Liq/MCap: real teams lock real money in the pool; rugs keep liq minimal
      • FDV/MCap: large spread means hidden supply overhang that will dump later

    Supply integrity returns 0.0 immediately for meme supply (hard kill).
    This means a token named "Quantum Finance" with 420B supply scores 0.0
    in this dimension → geometric mean = 0.0 → not alerted.
    """
    best  = token_data["best_pair"]
    mcap  = best.get("marketCap") or 0
    fdv   = best.get("fdv")       or 0
    liq   = token_data["total_liq"]
    price = float(best.get("priceUsd") or 0)

    # ── Supply integrity: the CORE tokenomics signal ──────────────────────────
    # Meme supply → immediate 0.0 (no score, no alert, regardless of token name)
    if fdv > 0 and price > 0:
        if _is_meme_supply(fdv, price)[0]:
            return 0.0   # hard kill — meme tokenomics regardless of name
        approx_supply = fdv / price
        # Utility/governance supply range: 1M–2B
        # Real projects design supply around governance weight, vesting schedules.
        # The lower in this range, the more scarcity premium; the higher, the
        # more it looks like inflation tokenomics.
        if 500_000 <= approx_supply <= 500_000_000:      # 500K–500M: ideal
            supply_pts = 4.0
        elif 500_000_000 < approx_supply <= 2_000_000_000:  # 500M–2B: fine
            supply_pts = 3.5
        elif approx_supply < 500_000:                     # < 500K: scarce, unusual
            supply_pts = 3.0
        elif 2_000_000_000 < approx_supply <= 10_000_000_000:  # 2B–10B: borderline
            supply_pts = 2.0
        else:                                              # > 10B: suspect
            supply_pts = 1.0
    else:
        supply_pts = 2.5   # no price data → neutral

    # ── Liquidity / MCap: skin-in-the-game anti-rug signal ────────────────────
    # If the team locked meaningful liquidity relative to market cap, rugging
    # requires them to take a proportional loss. High liq/mcap = alignment.
    if mcap > 0 and liq > 0:
        lm = liq / mcap
        if lm >= 0.50:    liq_pts = 3.0    # pool ≥ half the mcap = very committed
        elif lm >= 0.30:  liq_pts = 2.5
        elif lm >= 0.20:  liq_pts = 2.0
        elif lm >= 0.10:  liq_pts = 1.0    # quality-filter floor — barely passing
        else:             liq_pts = 0.0
    else:
        liq_pts = 1.5

    # ── FDV / MCap unlock overhang ────────────────────────────────────────────
    # Large FDV vs circulating mcap means there's a big supply overhang waiting to
    # unlock and dump.  Quality filter already blocks >5×, so we score within 1-5×.
    if fdv > 0 and mcap > 0:
        ratio = fdv / mcap
        if ratio <= 1.2:    fdv_pts = 3.0    # almost all supply circulating
        elif ratio <= 2.0:  fdv_pts = 2.5    # moderate unlock risk
        elif ratio <= 3.0:  fdv_pts = 2.0
        elif ratio <= 4.0:  fdv_pts = 1.0
        else:               fdv_pts = 0.5
    else:
        fdv_pts = 1.5

    # Weighted combination → normalize to 0-10
    # Supply integrity (40%) > Liq backing (35%) > FDV risk (25%)
    raw = supply_pts * 0.40 + liq_pts * 0.35 + fdv_pts * 0.25
    # Maximum possible: 4.0*0.4 + 3.0*0.35 + 3.0*0.25 = 1.6+1.05+0.75 = 3.4
    return max(0.0, min(10.0, round(raw / 3.4 * 10, 1)))


def _score_project_maturity(token_data, token_state, now):
    """D_maturity — Has this project proven itself over real time and market cycles?

    Key insight: time cannot be faked.  A meme token can buy website hosting,
    pay for a Twitter account, and list on DexScreener — but it can NOT fake
    having survived 90 days of real market cycles with an active community.

    Three components, weighted by predictive power:
      • Age (40%) — weeks/months of SURVIVAL is the ultimate anti-rug filter
      • Monitoring history (35%) — we've observed sustained activity over cycles
      • Social completeness (25%) — full organizational presence, not just a name
    """
    score = 0.0
    best = token_data["best_pair"]
    info = best.get("info") or {}

    # ── Project age: survival over real market cycles ─────────────────────────
    pair_ts = token_data.get("pair_created_at")
    if pair_ts:
        age_days = (now * 1000 - pair_ts) / 86_400_000
        if age_days >= 180:   age_pts = 4.0   # 6 months: veteran, survived many cycles
        elif age_days >= 90:  age_pts = 3.5   # 3 months: proven resilience
        elif age_days >= 60:  age_pts = 3.0   # 2 months: past initial volatility
        elif age_days >= 30:  age_pts = 2.0   # 1 month: past rug window
        elif age_days >= 14:  age_pts = 1.0   # 2 weeks: cleared the standard gate
        elif age_days >= 7:   age_pts = 0.5   # 1 week: minimal survival; early watch
        else:                 age_pts = 0.0
    else:
        age_pts = 0.5

    # ── Monitoring history: repeated observation proves sustained activity ─────
    # conviction_history records the last 5 scoring cycles we've run on this token.
    # Being scored multiple times proves it wasn't just a flash event.
    conv_hist = token_state.get("conviction_history", [])
    n = len(conv_hist)
    if n >= 5:     hist_pts = 3.0
    elif n >= 3:   hist_pts = 2.0
    elif n >= 1:   hist_pts = 1.0
    else:          hist_pts = 0.5   # first observation — not penalized, just unknown

    # Conviction trend bonus: scores RISING over recent cycles = pre-breakout buildup.
    # Read from PREVIOUS history — current score is appended after this returns.
    if n >= 3 and conv_hist[-1] > conv_hist[-2] > conv_hist[-3]:
        hist_pts = min(hist_pts + 1.0, 3.0)   # accelerating improvement
    elif n >= 2 and conv_hist[-1] > conv_hist[-2]:
        hist_pts = min(hist_pts + 0.5, 3.0)   # last cycle improved

    # ── Social completeness: full organizational presence ─────────────────────
    # Twitter is the hard gate; Telegram is now optional (quality filter changed).
    # Reward tokens that go beyond the minimum — Discord + Twitter + Telegram is
    # the full stack, rare for memes, common for real DeFi/infra projects.
    socials   = info.get("socials") or []
    soc_types = {(s.get("type") or "").lower() for s in socials}
    has_twitter  = "twitter"  in soc_types
    has_telegram = "telegram" in soc_types
    has_discord  = "discord"  in soc_types

    if has_discord and has_twitter and has_telegram:
        soc_pts = 3.0   # full stack: extremely rare for memes
    elif has_discord and (has_twitter or has_telegram):
        soc_pts = 2.7   # Discord + one other = genuine community investment
    elif has_twitter and has_telegram:
        soc_pts = 2.5   # standard serious project social presence
    elif has_twitter:
        soc_pts = 2.0   # Twitter-only: many legitimate early-stage projects; acceptable
    elif has_telegram:
        soc_pts = 1.0   # Telegram-only: suspicious (pump groups skip Twitter)
    else:
        soc_pts = 0.0

    # Weighted combination → normalize to 0-10
    raw = age_pts * 0.40 + hist_pts * 0.35 + soc_pts * 0.25
    # Maximum possible: 4.0*0.4 + 3.0*0.35 + 3.0*0.25 = 1.6+1.05+0.75 = 3.4
    return max(0.0, min(10.0, round(raw / 3.4 * 10, 1)))


def _score_momentum(token_data, token_state):
    """D_momentum — Is the current price/volume move authentic and building?

    Key insight: genuine breakouts accelerate.  Volume builds in layers:
    h24 rate → h6 rate → h1 rate each HIGHER than the previous.
    Coordinated pumps typically burst in one hour and then flatten.

    Three independent signals are combined:
      • Volume acceleration  (is momentum BUILDING across timeframes?)
      • Surge vs baseline    (how much bigger is this vs the quiet baseline?)
      • Buy pressure quality (is fresh capital still entering in h1?)
    """
    score = 0.0
    vol  = token_data["total_vol"]
    h1_v = vol["h1"]
    h6_v = vol["h6"]
    h24_v = vol["h24"]

    h1_rate  = h1_v  / 1.0
    h6_rate  = h6_v  / 6.0
    h24_rate = h24_v / 24.0

    # ── Volume acceleration: layered build-up = authentic momentum ─────────────
    # Perfect: h1_rate > h6_rate > h24_rate (volume building over each window)
    # This pattern is hard to fake: it requires sustained buying pressure, not
    # just a single candle burst.
    if h6_rate > 0 and h24_rate > 0:
        h1_vs_h6   = h1_rate  / h6_rate   if h6_rate  > 0 else 0.0
        h6_vs_h24  = h6_rate  / h24_rate  if h24_rate > 0 else 0.0
        if h1_vs_h6 >= 1.5 and h6_vs_h24 >= 1.2:
            score += 4.0   # strong double-layer acceleration
        elif h1_vs_h6 >= 1.2 and h6_vs_h24 >= 1.0:
            score += 3.0   # clean layered build
        elif h1_vs_h6 >= 1.0 and h6_vs_h24 >= 1.0:
            score += 2.0   # current hour ≥ 6h rate and 6h ≥ 24h rate
        elif h1_vs_h6 >= 1.0:
            score += 1.0   # at least h1 is above h6 rate
    elif h1_rate > 0 and h24_rate > 0 and h1_rate >= h24_rate:
        score += 0.5   # minimal: just current hour above day average

    # ── Surge vs quiet baseline: is this awakening real? ──────────────────────
    # Compare current h1 volume to the token's established quiet hourly average.
    # A 10× surge means THIS HOUR is 10× busier than any recent quiet hour.
    baseline_h24 = token_state.get("last_low_h24_vol") or 0.0
    avg_hourly   = baseline_h24 / 24 if baseline_h24 > 0 else 0.0
    if avg_hourly > 0 and h1_v > 0:
        surge = h1_v / avg_hourly
        if surge >= 30:     score += 3.5   # extraordinary: 30× baseline
        elif surge >= 15:   score += 2.5   # very strong: 15× baseline
        elif surge >= 8:    score += 1.5   # strong: 8× baseline (surge threshold)
        elif surge >= 4:    score += 0.75
        elif surge >= 2:    score += 0.25

    # ── Buy pressure in h1: is fresh capital still entering? ──────────────────
    buys_h1  = token_data["total_buys_h1"]
    sells_h1 = token_data["total_sells_h1"]
    txns_h1  = buys_h1 + sells_h1
    if txns_h1 >= 3:
        br_h1 = buys_h1 / txns_h1
        if br_h1 >= 0.65:    score += 2.5
        elif br_h1 >= 0.55:  score += 1.75
        elif br_h1 >= 0.45:  score += 1.0
        elif br_h1 >= 0.35:  score += 0.25   # barely above minimum gate

    # ── Price direction: is the move actually going somewhere? ────────────────
    try:
        pc   = token_data.get("price_change") or {}
        m5c  = float(pc.get("m5")  or 0)
        h1c  = float(pc.get("h1")  or 0)
        h24c = float(pc.get("h24") or 0)
        if m5c > 0 and h1c > 0 and h24c > 0:
            score += 1.0   # all timeframes green = clean uptrend
        elif h1c > 0 and h24c > 0:
            score += 0.5
        elif m5c < -15 or h1c < -20:
            score -= 2.0   # significant dump happening right now
    except (TypeError, ValueError):
        pass

    return max(0.0, min(10.0, round(score, 1)))


def _conviction_score(token_state, token_data, now):
    """Geometric mean conviction score: (D_organic × D_tokenomics × D_maturity × D_momentum)^¼

    Returns a tuple: (overall_score: float, dims: dict)
    where dims = {"organic": float, "tokenomics": float,
                  "maturity": float, "momentum": float}

    Any dimension < _CONV_FLOOR (2.0) → overall = 0.0 (hard kill).

    Examples:
      Legitimate 60d project:   organic=7.5 tok=9.1 mat=8.5 mom=7.5 → 8.08
      Pump-and-dump in exit:    organic=1.5 tok=7.0 mat=5.0 mom=8.0 → 0.0  (organic < floor)
      Meme with legit-sounding name: organic=6 tok=0.0 mat=6 mom=7 → 0.0  (tokenomics = 0)
      Sleeping giant day 30:    organic=5.0 tok=8.5 mat=7.3 mom=8.5 → 7.17
    """
    d_org  = _score_organic_trading(token_data)
    d_tok  = _score_tokenomics(token_data)
    d_mat  = _score_project_maturity(token_data, token_state, now)
    d_mom  = _score_momentum(token_data, token_state)

    dims = {"organic": d_org, "tokenomics": d_tok,
            "maturity": d_mat, "momentum": d_mom}

    # Hard floor: ANY dimension below 2.5 → the alert is not credible.
    if min(d_org, d_tok, d_mat, d_mom) < _CONV_FLOOR:
        return 0.0, dims

    # Geometric mean: rewards BALANCED strength, heavily penalizes any weak link.
    overall = round((d_org * d_tok * d_mat * d_mom) ** 0.25, 1)
    return min(10.0, overall), dims


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def evaluate_token(token_data, cfg, token_state, now):
    """Evaluate one token and return a list of alert dicts (may be empty).

    Alert types:
      volume_surge         — m5 or h1 disproportionate vs quiet baseline (earliest)
      volume_spike         — any timeframe crosses the absolute threshold
      resistance_breakout  — price exceeds ATH reference by breakout_pct with volume
      momentum_continuation— price climbs ≥50% above last breakout/continuation level
    """
    det       = cfg["detection"]
    best      = token_data["best_pair"]
    total_liq = token_data["total_liq"]
    total_vol = token_data["total_vol"]
    token_state["last_seen_ts"] = now

    # Track when we first started monitoring this token
    if not token_state.get("first_seen_ts"):
        token_state["first_seen_ts"] = now

    try:
        current_price = float(best.get("priceUsd") or 0)
    except (TypeError, ValueError):
        current_price = 0.0

    # ── Step 1: update all price references ─────────────────────────────────
    h24_vol = total_vol["h24"]

    try:
        h24_chg = float((token_data.get("price_change") or {}).get("h24") or 0)
    except (TypeError, ValueError):
        h24_chg = 0.0

    # Save pre-update reference values — breakout detection (Step 4) uses
    # these so the reference can't chase the current price in the same cycle.
    # prev_baseline_h24: prevents first-cycle spam where baseline is set and
    # immediately used in the same evaluation (produces 24x "spike" on every
    # quiet token seen for the first time).
    prev_quiet_high   = token_state.get("quiet_price_high")   or 0.0
    prev_consol_high  = token_state.get("consolidation_high") or 0.0
    prev_baseline_h24 = token_state.get("last_low_h24_vol")   or 0.0

    # 1a. Quiet baseline (dead token: h24 vol very low)
    if h24_vol <= det["low_volume_max"]:
        token_state["last_low_h24_ts"] = now
        if not token_state.get("first_quiet_ts"):
            token_state["first_quiet_ts"] = now
        # Rolling baseline: min of last 7 quiet readings — prevents a single stale
        # data-point from anchoring the baseline permanently after activity recovers.
        _quiet_hist = token_state.setdefault("quiet_vol_history", [])
        _quiet_hist.append(h24_vol)
        if len(_quiet_hist) > 7:
            _quiet_hist[:] = _quiet_hist[-7:]
        token_state["last_low_h24_vol"] = min(_quiet_hist)
        # Freeze resistance once any spike alert fires
        if not token_state.get("volume_spike_ts") and current_price > 0:
            if current_price > (token_state.get("quiet_price_high") or 0):
                token_state["quiet_price_high"]    = current_price
                token_state["quiet_price_high_ts"] = now

    # 1b. Consolidation reference — catches tokens with moderate activity too.
    # Updates whenever volume is below consolidation_volume_max AND price isn't
    # crashing AND we're not in an active spike window (≤24h after a spike fired).
    # NOTE: we intentionally do NOT update consolidation_high when the price is
    # already in a surge (h1 price change > 15%) to keep the reference stable
    # against a rising price.
    # For mid-cap tokens ($100K–$1M) scale the threshold up so their "normal"
    # volume still qualifies as consolidation and a reference ceiling gets set.
    _raw_mcap = best.get("marketCap") or best.get("fdv") or 0
    _mc_mult  = det.get("midcap_vol_multiplier", 4.0)
    consolidation_vol_max = det.get("consolidation_volume_max", 500_000)
    if _raw_mcap > det.get("max_market_cap_usd", 100_000):
        consolidation_vol_max = consolidation_vol_max * _mc_mult
    spike_ts = token_state.get("volume_spike_ts")
    not_in_spike = not spike_ts or (now - spike_ts > det.get("alert_cooldown_seconds", 86400))

    try:
        h1_chg = float((token_data.get("price_change") or {}).get("h1") or 0)
    except (TypeError, ValueError):
        h1_chg = 0.0

    if (current_price > 0
            and h24_vol < consolidation_vol_max
            and h24_chg > -40.0
            and h1_chg < 15.0          # not already surging — keep reference stable
            and not_in_spike):
        if current_price > (token_state.get("consolidation_high") or 0):
            token_state["consolidation_high"]    = current_price
            token_state["consolidation_high_ts"] = now

    # 1c. Running ATH high — used for continuation detection.
    # We save the value BEFORE updating so Step 5 can compare against last cycle's max.
    prev_all_time_high = token_state.get("all_time_seen_high") or 0.0
    if current_price > prev_all_time_high:
        token_state["all_time_seen_high"] = current_price

    # ── Baseline validity: need at least one reference to proceed ────────────
    last_low_h24_ts = token_state.get("last_low_h24_ts")
    has_quiet_base  = bool(last_low_h24_ts and
                           now - last_low_h24_ts <= det["baseline_lookback_seconds"])
    has_consol_base = bool(token_state.get("consolidation_high"))

    if not has_quiet_base and not has_consol_base:
        return []

    # ── Step 2: hard filters ─────────────────────────────────────────────────
    market_cap = best.get("marketCap") or best.get("fdv")

    # If we already fired a breakout on this token, keep tracking the continuation
    # regardless of how high its mcap grows (we want to ride $100K → $10M+).
    already_in_breakout = bool(token_state.get("breakout_reference_price"))
    midcap_mode   = False   # $100K–$1M tier — stricter quality, scaled thresholds
    largecap_mode = False   # $1M–$10M emerging large-cap tier (new)

    max_small    = det["max_market_cap_usd"]                     # $100K
    max_midcap   = det.get("max_midcap_usd",   1_000_000)        # $1M
    max_largecap = det.get("max_largecap_usd", 10_000_000)       # $10M
    min_mcap     = det.get("min_market_cap_usd", 0)

    if not already_in_breakout:
        if min_mcap > 0 and market_cap is not None and market_cap < min_mcap:
            return []
        if market_cap is not None and market_cap > max_small:
            if market_cap <= max_midcap:
                midcap_mode = True
            elif market_cap <= max_largecap:
                largecap_mode = True
            else:
                return []
    else:
        # Stop tracking once the token has grown to >$50M (mission complete)
        if market_cap is not None and market_cap > 50_000_000:
            return []

    if total_liq < det["min_liquidity_usd"]:
        return []

    # Stablecoin / wrapped-asset filter.
    try:
        h24_chg_abs = abs(float((token_data.get("price_change") or {}).get("h24") or 0))
        price_near_one = current_price > 0 and abs(current_price - 1.0) < 0.02
        if price_near_one and h24_chg_abs < 0.5:
            return []
    except (TypeError, ValueError):
        pass

    buys_h1  = token_data["total_buys_h1"]
    sells_h1 = token_data["total_sells_h1"]
    txns_h1  = buys_h1 + sells_h1
    min_buy_ratio = det.get("min_buy_ratio", 0.0)
    min_txns_gate = det.get("min_txns_h1", 0)
    if txns_h1 >= max(min_txns_gate, 3) and min_buy_ratio > 0:
        if (buys_h1 / txns_h1) < min_buy_ratio:
            return []

    # Distribution mode flag: if >70% of h24 txns are sells the token is in
    # active distribution — block volume-based entry signals but still allow
    # breakout / coil / continuation signals which reflect price action.
    _buys_h24_raw  = token_data["total_buys_h24"]
    _sells_h24_raw = token_data["total_sells_h24"]
    _txns_h24_raw  = _buys_h24_raw + _sells_h24_raw
    _h24_sell_dominated = (
        _txns_h24_raw >= 20 and
        _buys_h24_raw / _txns_h24_raw < 0.30
    )

    # ── Quality gate ─────────────────────────────────────────────────────────
    if not already_in_breakout:
        # Full quality filter for fresh tokens — tier-appropriate
        if midcap_mode:
            passes, _reason = _quality_filter_midcap(token_data, cfg, now)
        elif largecap_mode:
            passes, _reason = _quality_filter_largecap(token_data, cfg, now)
        else:
            passes, _reason = _quality_filter(token_data, cfg, now)
        if not passes:
            return []
    else:
        # Tokens already in a confirmed breakout skip the MARKET-CAP range check
        # so we can ride the move from $100K → $10M+.  But every other quality
        # gate is re-applied in full — no token gets a permanent free pass.
        #
        # Gaps this closes vs. the old "absolute minimum" bypass:
        #   • Meme name / symbol check  — was MISSING, now enforced
        #   • Twitter AND Telegram both required (not just one of the two)
        #   • Minimum h24 transactions  — catches ghost / abandoned trading
        #   • Minimum pool liquidity    — catches drained pools
        _bo_info     = best.get("info") or {}
        _bo_soc      = {(s.get("type") or "").lower()
                        for s in (_bo_info.get("socials") or [])}
        _bo_age      = ((now * 1000 - token_data["pair_created_at"]) / 86_400_000
                        if token_data.get("pair_created_at") else 999)
        _bo_fdv      = best.get("fdv") or 0
        _bo_price    = float(best.get("priceUsd") or 0)
        _bo_mcap     = best.get("marketCap") or 0
        _bo_base_tok = best.get("baseToken") or {}
        _bo_is_meme, _ = _is_meme_token(
            _bo_base_tok.get("name") or "", _bo_base_tok.get("symbol") or "")
        _bo_txns     = token_data["total_buys_h24"] + token_data["total_sells_h24"]
        _bo_liq      = token_data.get("total_liq", 0)

        _bo_bad = (
            not _bo_info.get("websites")               # no website
            or "twitter"  not in _bo_soc               # no Twitter
            or "telegram" not in _bo_soc               # no Telegram
            or _bo_age < 7                             # pair too new
            or _bo_is_meme                             # meme name / symbol
            or (_bo_fdv > 0 and _bo_price > 0          # meme supply fingerprint
                and _is_meme_supply(_bo_fdv, _bo_price)[0])
            or (_bo_fdv > 0 and _bo_mcap > 0           # extreme FDV overhang
                and _bo_fdv > _bo_mcap * 12)
            or _bo_liq < 3_000                         # pool has effectively dried up
            or (0 < _bo_txns < 30)                     # ghost / abandoned trading
        )
        if _bo_bad:
            # Evict from breakout tracking — next cycle re-enters full quality filter
            token_state.pop("breakout_reference_price", None)
            return []

    # ── Shared context ───────────────────────────────────────────────────────
    baseline_h24  = prev_baseline_h24                            # pre-update value only
    avg_hourly    = baseline_h24 / 24   if baseline_h24 > 0 else 0.0
    avg_5min      = baseline_h24 / 288  if baseline_h24 > 0 else 0.0
    buy_ratio     = (buys_h1 / txns_h1) if txns_h1 > 0 else None
    quiet_days    = (now - token_state["first_quiet_ts"]) / 86_400 \
                    if token_state.get("first_quiet_ts") else 0.0
    price_change  = token_data.get("price_change") or {}
    h1_vol        = total_vol["h1"]

    try:
        m5_chg = float(price_change.get("m5") or 0)
    except (TypeError, ValueError):
        m5_chg = 0.0
    price_ok_for_surge = m5_chg >= det.get("min_price_change_m5", -10.0)

    # Best ATH reference: use pre-update values so the ceiling can't chase the
    # current price within the same cycle (which would prevent breakout from firing).
    ath_reference = max(
        prev_quiet_high,
        prev_consol_high,
        token_state.get("gecko_30d_high") or 0.0,
    )

    # Volume threshold multiplier — mid-cap tokens need 4× more volume to signal
    # (a $500K mcap token generates ~4× more volume than a $50K micro-cap normally)
    _vol_mult = (det.get("midcap_vol_multiplier",   4.0)  if midcap_mode   else
                 det.get("largecap_vol_multiplier", 10.0) if largecap_mode else 1.0)

    # Volume confirmation needed for ATH breakout and continuation
    breakout_min_vol = det.get("breakout_min_h1_vol", 500) * _vol_mult
    vol_confirming   = (h1_vol >= breakout_min_vol or
                        total_vol["h24"] >= det["spike_volume_min"] * _vol_mult * 2)

    # Monitoring age — used by coiling + breakout + continuation signals
    min_monitoring_secs = det.get("min_consolidation_age_seconds", 14_400)
    monitoring_age      = now - token_state.get("first_seen_ts", now)

    # Compute conviction once; reuse for every alert this cycle.
    # Returns (overall_score, dim_scores_dict).
    conviction, conv_dims = _conviction_score(token_state, token_data, now)
    # Update trend history so next cycle can reward rising quality
    _conv_hist = token_state.setdefault("conviction_history", [])
    _conv_hist.append(round(conviction, 1))
    if len(_conv_hist) > 5:
        _conv_hist[:] = _conv_hist[-5:]

    # Precompute quality metadata used by formatter and future scoring
    pair_ts   = token_data.get("pair_created_at")
    pair_age_days = (now * 1000 - pair_ts) / 86_400_000 if pair_ts else None
    _info     = best.get("info") or {}
    has_website = bool(_info.get("websites"))
    _socials    = _info.get("socials") or []
    social_types = [s.get("type") for s in _socials if s.get("type")]
    txns_h24  = token_data["total_buys_h24"] + token_data["total_sells_h24"]

    # ── Pulse snapshot ────────────────────────────────────────────────────────
    # Stored every cycle for quality-passing tokens so the hourly watchlist
    # digest can display fresh data without extra DexScreener calls.
    # Written here (after quality filter, after conviction) so only legitimate
    # tokens with real scores make it into the watchlist.
    _base_tok = best.get("baseToken") or {}
    token_state["_pulse_snapshot"] = {
        "name":          _base_tok.get("name", "?"),
        "symbol":        _base_tok.get("symbol", "?"),
        "mcap":          market_cap,
        "h1_vol":        h1_vol,
        "h24_vol":       total_vol["h24"],
        "conviction":    conviction,
        "conv_dims":     conv_dims,
        "price_usd":     best.get("priceUsd"),
        "price_change":  price_change,
        "pair_url":      best.get("url", ""),
        "pair_age_days": pair_age_days,
        "has_website":   has_website,
        "social_types":  social_types,
        "txns_h24":      txns_h24,
        "ts":            now,
    }

    # ── Daily accumulation snapshot ───────────────────────────────────────────
    # One entry per 24 hours, kept for 30 days.  Powers the slow-build
    # accumulation signal: week-over-week txn growth while staying quiet.
    # This is how we catch Virtuals/GitClaw/Aeon-style projects at Week 4-8
    # before the explosive move happens at Week 12+.
    _daily = token_state.setdefault("daily_snapshots", [])
    _last_daily_ts = _daily[-1]["ts"] if _daily else 0
    if now - _last_daily_ts >= 86_400:          # one snapshot per 24 hours
        _br_h24 = (token_data["total_buys_h24"]
                   / max(1, txns_h24))
        _daily.append({
            "ts":           now,
            "txns_h24":     txns_h24,
            "liq":          total_liq,
            "price":        current_price,
            "h24_vol":      h24_vol,
            "conviction":   conviction,
            "buy_ratio_h24": round(_br_h24, 3),
        })
        if len(_daily) > 30:                    # 30 days of history
            _daily[:] = _daily[-30:]

    # Volume acceleration indicator — h1 rate vs h6 and h24 rates
    _h6_vol  = total_vol["h6"]
    _h1_rate = h1_vol          / 1.0
    _h6_rate = _h6_vol         / 6.0
    _h24_rate = total_vol["h24"] / 24.0
    vol_accelerating = (_h1_rate >= _h6_rate and _h1_rate >= _h24_rate)

    def _base_alert(atype):
        return {
            "type":             atype,
            "pair":             best,
            "baseline_h24_vol": baseline_h24,
            "liquidity":        total_liq,
            "market_cap":       market_cap,
            "quiet_price_high": ath_reference,   # pre-update ceiling used for breakout
            "buy_ratio":        buy_ratio,
            "quiet_days":       quiet_days,
            "conviction":       conviction,
            "conv_dims":        conv_dims,        # per-dimension breakdown
            "price_change":     price_change,
            "pair_age_days":    pair_age_days,
            "has_website":      has_website,
            "social_types":     social_types,
            "txns_h24":         txns_h24,
            "vol_accelerating": vol_accelerating,
            "midcap_mode":      midcap_mode,
            "largecap_mode":    largecap_mode,
        }

    alerts = []
    surge_mult = det.get("spike_volume_multiplier", 0)

    # ── Step 3.0a: Slow Build — week-over-week accumulation (earliest signal) ─
    # Fires when a legitimate token shows quiet, consistent growth over 7+ days
    # without any spike event.  This is the EXACT fingerprint of tokens like
    # Virtuals, GitClaw, Aeon at Weeks 4-8 before the explosive Week 12 move:
    #
    #   • h24 transactions growing ≥30% vs 7 days ago   (organic wallet growth)
    #   • Volume still below consolidation ceiling       (big money not in yet)
    #   • Price holding — not crashing (higher lows)     (real demand floor)
    #   • Liquidity not draining                         (team/LPs still committed)
    #   • Conviction ≥ 6.0 and not falling              (fundamentals intact)
    #   • No alerts in the last 7 days                  (truly pre-signal stage)
    #
    # This fires at most once per 7 days per token (slow_build_cooldown_seconds).
    _sb_cd       = det.get("slow_build_cooldown_seconds", 604_800)   # 7 days
    _sb_min_conv = det.get("slow_build_min_conviction",   6.0)
    _sb_snaps    = token_state.get("daily_snapshots", [])

    if (len(_sb_snaps) >= 7
            and conviction >= _sb_min_conv
            and h24_vol <= det.get("consolidation_volume_max", 500_000)
            and not already_in_breakout):

        _sb_today    = _sb_snaps[-1]
        _sb_week_ago = _sb_snaps[-7]

        _sb_txns_now  = _sb_today.get("txns_h24",  0)
        _sb_txns_then = _sb_week_ago.get("txns_h24", 0)
        _sb_liq_now   = _sb_today.get("liq",  0)
        _sb_liq_then  = _sb_week_ago.get("liq", 0)
        _sb_px_now    = _sb_today.get("price",  0)
        _sb_px_then   = _sb_week_ago.get("price", 0)

        # Conviction trend: last entry ≥ entry 3 cycles ago (not falling)
        _sb_conv_ok = (
            len(token_state.get("conviction_history", [])) < 3
            or token_state["conviction_history"][-1]
               >= token_state["conviction_history"][-3]
        )

        # No recent spike/breakout/surge alert in the last 7 days
        _sb_no_recent = True
        for _sb_key in ["sleeping_giant_alert_ts", "breakout_alert_ts",
                         "coil_alert_ts", "continuation_alert_ts"]:
            if now - token_state.get(_sb_key, 0) < 604_800:
                _sb_no_recent = False
                break
        if _sb_no_recent:
            for _sb_tf in ["m5_surge", "h1_surge", "h1", "h6", "h24"]:
                _sub = token_state.get(_sb_tf) or {}
                if _sub.get("last_alert_ts") and now - _sub["last_alert_ts"] < 604_800:
                    _sb_no_recent = False
                    break

        if (_sb_txns_then >= 30
                and _sb_txns_now  >= 50
                and _sb_txns_now / _sb_txns_then >= 1.30       # ≥30% more txns
                and (_sb_px_then  <= 0 or _sb_px_now >= _sb_px_then * 0.75)  # holding
                and (_sb_liq_then <= 0 or _sb_liq_now >= _sb_liq_then * 0.80) # liq ok
                and _sb_conv_ok
                and _sb_no_recent):

            _sb_last = token_state.get("slow_build_alert_ts", 0)
            if now - _sb_last >= _sb_cd:
                token_state["slow_build_alert_ts"] = now
                _sb_txns_growth = _sb_txns_now / max(1, _sb_txns_then)
                _sb_px_chg      = ((_sb_px_now / _sb_px_then - 1) * 100
                                   if _sb_px_then > 0 else 0.0)
                _sb_liq_chg     = ((_sb_liq_now / _sb_liq_then - 1) * 100
                                   if _sb_liq_then > 0 else 0.0)
                a = _base_alert("slow_build_accumulation")
                a.update(
                    txns_growth=_sb_txns_growth,
                    txns_now=_sb_txns_now,
                    txns_week_ago=_sb_txns_then,
                    price_chg_7d=_sb_px_chg,
                    liq_chg_7d=_sb_liq_chg,
                    days_of_data=len(_sb_snaps),
                )
                alerts.append(a)

    # ── Step 3.0b: Dead-Token Resurrection ───────────────────────────────────
    # A token that the bot has previously seen at a MUCH higher price (≥5× the
    # current price), was then confirmed dormant, and is now surging again.
    # Different from sleeping_giant: this token had a real life before — it rose,
    # fell hard, went dormant, and is now showing the first signs of a comeback.
    # Pattern: "was $1, crashed to $0.05, silent for 3 months, now $0.08 with surge vol."
    _res_cd         = det.get("resurrection_cooldown_seconds", 604_800)   # 7 days
    _res_min_fall   = det.get("resurrection_min_fall_mult",    5.0)       # ATH ≥5× current
    _res_surge_mult = det.get("resurrection_surge_mult",       4)         # vol ≥4× baseline

    _hist_ath = max(
        prev_all_time_high,                                  # highest ever seen by bot
        token_state.get("gecko_30d_high")     or 0.0,
        token_state.get("consolidation_high") or 0.0,
    )

    if (
        _hist_ath > 0
        and current_price > 0
        and current_price < _hist_ath                        # still below old peak
        and _hist_ath / current_price >= _res_min_fall       # fell significantly (≥5×)
        and not already_in_breakout
        and has_quiet_base                                   # confirmed dormant recently
        and h1_vol >= det.get("breakout_min_h1_vol", 500)
        and not _h24_sell_dominated
    ):
        _dormant_h24 = prev_baseline_h24 or 0.0
        _vol_surging = (
            (_dormant_h24 > 0 and h24_vol >= _dormant_h24 * _res_surge_mult)
            or (h1_vol >= det.get("min_surge_h1_vol", 2_000))
        )
        _last_res = token_state.get("resurrection_alert_ts") or 0
        if _vol_surging and now - _last_res >= _res_cd:
            token_state["resurrection_alert_ts"] = now
            a = _base_alert("dead_token_resurrection")
            a.update(
                all_time_high=_hist_ath,
                fall_pct=(_hist_ath / current_price - 1) * 100,
                current_price=current_price,
                h24_vol=total_vol["h24"],
                h1_vol=h1_vol,
            )
            alerts.append(a)

    # ── Step 3.0: Sleeping Giant wakeup ──────────────────────────────────────
    # A token dormant for 30+ days suddenly showing h1 volume ≥5× its quiet
    # hourly rate is a rare, very early signal — the very first candle of life.
    giant_min_days = det.get("sleeping_giant_days", 30)
    giant_cd       = det.get("sleeping_giant_cooldown_seconds", 172_800)
    giant_mult     = det.get("sleeping_giant_surge_mult", 5)
    _first_quiet   = token_state.get("first_quiet_ts")
    quiet_days_giant = (now - _first_quiet) / 86_400 if _first_quiet else 0.0
    if (has_quiet_base
            and _first_quiet is not None
            and quiet_days_giant >= giant_min_days
            and avg_hourly > 0
            and h1_vol >= avg_hourly * giant_mult
            and not _h24_sell_dominated):
        last_giant = token_state.get("sleeping_giant_alert_ts")
        if last_giant is None or now - last_giant >= giant_cd:
            token_state["sleeping_giant_alert_ts"] = now
            a = _base_alert("sleeping_giant_wakeup")
            a.update(
                sleep_days=quiet_days_giant,
                wake_mult=h1_vol / avg_hourly,
                h1_vol=h1_vol,
                avg_hourly=avg_hourly,
            )
            alerts.append(a)

    # ── Step 3a: m5 micro-surge (fires within the first candle) ─────────────
    # Requires quiet baseline to compute avg_5min meaningfully
    m5_vol     = total_vol["m5"]
    min_m5_vol = det.get("min_surge_m5_vol", 500) * _vol_mult
    if (price_ok_for_surge and surge_mult > 0 and avg_5min > 0 and has_quiet_base
            and m5_vol >= min_m5_vol and m5_vol >= avg_5min * surge_mult
            and not _h24_sell_dominated):
        m5_state = token_state.setdefault("m5_surge", {})
        last_m5 = m5_state.get("last_alert_ts")
        if last_m5 is None or now - last_m5 >= det["alert_cooldown_seconds"]:
            m5_state["last_alert_ts"] = now
            token_state.setdefault("volume_spike_ts", now)
            a = _base_alert("volume_surge")
            a.update(surge_window="m5", avg_baseline_rate=avg_5min,
                     surge_vol=m5_vol, surge_ratio=m5_vol / avg_5min)
            alerts.append(a)

    # ── Step 3b: h1 surge ratio (fires within the first hour) ────────────────
    min_h1_vol = det.get("min_surge_h1_vol", 1_000) * _vol_mult
    if (price_ok_for_surge and surge_mult > 0 and avg_hourly > 0 and has_quiet_base
            and h1_vol >= min_h1_vol and h1_vol >= avg_hourly * surge_mult
            and not _h24_sell_dominated):
        h1_state = token_state.setdefault("h1_surge", {})
        last_h1 = h1_state.get("last_alert_ts")
        if last_h1 is None or now - last_h1 >= det["alert_cooldown_seconds"]:
            h1_state["last_alert_ts"] = now
            token_state.setdefault("volume_spike_ts", now)
            a = _base_alert("volume_surge")
            a.update(surge_window="h1", avg_baseline_rate=avg_hourly,
                     surge_vol=h1_vol, surge_ratio=h1_vol / avg_hourly)
            alerts.append(a)

    # ── Step 3c: absolute volume spike across configured timeframes ──────────
    # Only fires when we have an established quiet baseline (last_low_h24_vol > 0).
    # Without a baseline we can't distinguish "always-active token" from "waking up",
    # which causes first-cycle spam for every token just added to monitoring.
    for tf in cfg.get("timeframes", ["h1", "h6", "h24"]):
        if _h24_sell_dominated:
            continue
        volume = total_vol.get(tf) or 0.0
        if volume < det["spike_volume_min"] * _vol_mult:
            continue
        if baseline_h24 <= 0:
            continue
        tf_state = token_state.setdefault(tf, {})
        last_tf = tf_state.get("last_alert_ts")
        if last_tf is not None and now - last_tf < det["alert_cooldown_seconds"]:
            continue
        tf_state["last_alert_ts"] = now
        token_state.setdefault("volume_spike_ts", now)
        a = _base_alert("volume_spike")
        a.update(timeframe=tf, spike_volume=volume)
        alerts.append(a)

    # ── Step 3d: Pre-breakout coiling alert ─────────────────────────────────
    # Fires when price is close to (but still below) the ATH reference while
    # volume is building and buyers are dominant.  Gives the earliest possible
    # entry signal — before the 5% breakout threshold is crossed.
    coiling_lower_pct = det.get("coiling_lower_pct", 15)       # zone starts here% below ceiling
    coiling_zone_low  = ath_reference * (1 - coiling_lower_pct / 100)
    coiling_min_vol   = det.get("coiling_min_h1_vol", 1_000)
    coiling_min_buy   = det.get("coiling_buy_ratio_min", 0.50)
    coiling_cd        = det.get("coiling_cooldown_seconds", 43_200)
    if (ath_reference > 0
            and current_price > 0
            and coiling_zone_low <= current_price < ath_reference
            and monitoring_age >= min_monitoring_secs
            and h1_vol >= coiling_min_vol
            and vol_confirming
            and (buy_ratio is None or buy_ratio >= coiling_min_buy)):
        last_coil = token_state.get("coiling_alert_ts")
        if last_coil is None or now - last_coil >= coiling_cd:
            token_state["coiling_alert_ts"] = now
            a = _base_alert("pre_breakout_coil")
            a.update(
                ath_reference=ath_reference,
                current_price=current_price,
                pct_to_breakout=(ath_reference / current_price - 1) * 100,
                h1_vol=h1_vol,
                h24_vol=total_vol["h24"],
            )
            alerts.append(a)

    # ── Step 4: ATH breakout — the primary "final takeoff" signal ───────────
    # Fires when price crosses above the best available reference ceiling.
    # No longer requires volume_spike_ts — catches consolidating tokens too.
    breakout_threshold = 1 + det.get("resistance_breakout_pct", 5) / 100
    breakout_cooldown  = det.get("breakout_cooldown_seconds", 43_200)

    if (ath_reference > 0
            and monitoring_age >= min_monitoring_secs
            and current_price >= ath_reference * breakout_threshold
            and vol_confirming):
        last_bo = token_state.get("breakout_alert_ts")
        if last_bo is None or now - last_bo >= breakout_cooldown:
            token_state["breakout_alert_ts"]      = now
            token_state["breakout_reference_price"] = current_price
            token_state.setdefault("volume_spike_ts", now)
            # Record which reference was broken so the alert can say so
            q = token_state.get("quiet_price_high")   or 0.0
            c = token_state.get("consolidation_high") or 0.0
            g = token_state.get("gecko_30d_high")     or 0.0
            if g >= c and g >= q and g > 0:
                ath_source = "30d-history"
            elif c >= q:
                ath_source = "consolidation"
            else:
                ath_source = "quiet-period"
            a = _base_alert("resistance_breakout")
            a.update(
                quiet_price_high=ath_reference,
                ath_source=ath_source,
                current_price=current_price,
                breakout_pct=(current_price / ath_reference - 1) * 100,
                h24_vol=total_vol["h24"],
                h1_vol=h1_vol,
            )
            alerts.append(a)

    # ── Step 5: momentum continuation ───────────────────────────────────────
    # After an ATH breakout fires, alert when price climbs continuation_pct%
    # above the level where we last alerted (breakout or prior continuation).
    # Uses the price reference saved when the breakout/continuation fired so
    # a slow grind upward doesn't constantly re-trigger.
    continuation_pct = det.get("continuation_pct", 50)
    continuation_cd  = det.get("continuation_cooldown_seconds", 14_400)  # 4h
    bo_ref_price     = token_state.get("breakout_reference_price") or 0.0

    if (bo_ref_price > 0
            and current_price > bo_ref_price * (1 + continuation_pct / 100)
            and vol_confirming):
        last_cont = token_state.get("continuation_alert_ts")
        if last_cont is None or now - last_cont >= continuation_cd:
            token_state["continuation_alert_ts"]    = now
            token_state["breakout_reference_price"] = current_price  # advance reference
            a = _base_alert("momentum_continuation")
            a.update(
                prev_reference=bo_ref_price,
                current_price=current_price,
                continuation_pct=(current_price / bo_ref_price - 1) * 100,
                h24_vol=total_vol["h24"],
                h1_vol=h1_vol,
            )
            alerts.append(a)

    # ── Step 6: MCap milestone tracker ───────────────────────────────────────
    # Fires ONCE each time a quality-passing token crosses a key market-cap
    # level upward — regardless of whether another signal fired this cycle.
    # Covers the full journey: $100K → $500K → $1M → $5M → $10M.
    # A token at $19K today that reaches $10M = 526× — we alert at every step.
    _MCAP_MILESTONES = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    _milestones_hit  = set(token_state.get("mcap_milestones_hit") or [])

    if (market_cap
            and conviction >= det.get("milestone_min_conviction", 5.0)):
        for _ms in _MCAP_MILESTONES:
            _ms_key = int(_ms)
            if _ms_key not in _milestones_hit and market_cap >= _ms:
                _milestones_hit.add(_ms_key)
                token_state["mcap_milestones_hit"] = list(_milestones_hit)
                # Fire only once per milestone — no cooldown key needed
                if not token_state.get(f"mcap_ms_{_ms_key}_ts"):
                    token_state[f"mcap_ms_{_ms_key}_ts"] = now
                    a = _base_alert("mcap_milestone")
                    a.update(milestone=_ms, market_cap=market_cap)
                    alerts.append(a)
                break  # one milestone alert per cycle maximum

    return alerts


# ---------------------------------------------------------------------------
# Alert formatting — tiered, actionable Telegram messages
#
# Tier system:
#   🌱 WATCH  — early accumulation signals (act small, set alerts)
#   ⚡ ENTRY  — primary entry signals (size in now)
#   🔥 RIDE   — continuation signals (add or hold)
#   📊 INFO   — factual milestones (no action required)
# ---------------------------------------------------------------------------

def _token_header(pair):
    token = pair.get("baseToken") or {}
    return token.get("name", "?"), token.get("symbol", "?")


def _conviction_bar(score):
    filled = round(score)
    bar    = "█" * filled + "░" * (10 - filled)
    tier   = "HIGH 🔥" if score >= 7 else ("MED 👀" if score >= 4 else "LOW")
    return f"{bar} {score:.1f}/10 {tier}"


def _dim_bar(score):
    """Compact 5-cell bar for a dimension sub-score."""
    cells  = round(score / 2)   # 0-10 → 0-5 filled cells
    return "■" * cells + "□" * (5 - cells)


def _mcap_line(mcap):
    """Return compact mcap + multiplier string."""
    if not mcap or mcap <= 0:
        return "mcap n/a"
    mult = 10_000_000 / mcap
    if mcap >= 1_000_000:
        return f"${mcap/1_000_000:.1f}M mcap · {mult:.0f}× to $10M"
    if mcap >= 1_000:
        return f"${mcap/1_000:.0f}K mcap · {mult:.0f}× to $10M"
    return f"${mcap:.0f} mcap · {mult:.0f}× to $10M"


def _trade_plan(mcap, signal_type):
    """Return a simple trade-plan line calibrated to current mcap."""
    if not mcap or mcap <= 0:
        return "Entry: now  ·  Stop: −35%"

    # Target: 3× and 10× from current mcap
    t1 = mcap * 3
    t2 = mcap * 10
    t1_s = f"${t1/1_000:.0f}K" if t1 < 1_000_000 else f"${t1/1_000_000:.1f}M"
    t2_s = f"${t2/1_000:.0f}K" if t2 < 1_000_000 else f"${t2/1_000_000:.1f}M"

    if signal_type in ("slow_build_accumulation", "pre_breakout_coil"):
        size_note = "small (early watch)"
    elif signal_type in ("sleeping_giant_wakeup", "volume_surge", "resistance_breakout"):
        size_note = "normal entry"
    elif signal_type in ("momentum_continuation", "dead_token_resurrection"):
        size_note = "add / hold existing"
    else:
        size_note = "normal entry"

    return (f"Entry: now ({size_note})  ·  "
            f"Targets: {t1_s} mcap (+200%) → {t2_s} (+900%)  ·  Stop: −35%")


def _build_alert_body(alert, tier_header: str, why_text: str) -> str:
    """Assemble a complete tiered Telegram alert message.

    Structure:
        {tier_header}
        ━━━━━━━━━━━━━━━
        Name (SYM) · BASE · $XXK mcap · 234× to $10M

        WHY: {why_text}

        📊 Conviction: ████████░░ X.X/10 HIGH
        📐 Org X.X · Tok X.X · Mat X.X · Mom X.X

        📈 $price  m5 ±X%  h1 ±X%  h24 ±X%
        💧 Liq $X  ·  Vol h1 $X  ·  Txns/d X  ·  Age Xd
        🔗 Socials  ·  Web ✓/✗  ·  FDV/MCap X.X×  ·  Buys X% h1

        📋 TRADE PLAN
           {trade_plan}

        Contract: 0x...
        {url}
    """
    pair     = alert["pair"]
    name, symbol = _token_header(pair)
    mcap     = alert.get("market_cap") or 0
    liq      = alert.get("liquidity", 0)
    stype    = alert.get("type", "")
    conv     = alert.get("conviction", 0.0)

    # Price + changes
    price_usd = pair.get("priceUsd", "?")
    pchg      = alert.get("price_change") or {}
    m5_c  = pchg.get("m5",  "?")
    h1_c  = pchg.get("h1",  "?")
    h24_c = pchg.get("h24", "?")
    def fmt_pct(v):
        if v == "?" or v is None:
            return "?"
        try:
            f = float(v)
            return f"{f:+.1f}%"
        except (ValueError, TypeError):
            return str(v)
    price_line = (f"📈 ${price_usd}  "
                  f"m5 {fmt_pct(m5_c)}  h1 {fmt_pct(h1_c)}  h24 {fmt_pct(h24_c)}")

    # Liquidity + volume
    h1_vol   = alert.get("h1_vol") or alert.get("surge_vol", 0)
    h1_vol_s = f"${h1_vol:,.0f}" if h1_vol else "n/a"
    txns_h24 = alert.get("txns_h24", 0)
    age_days = alert.get("pair_age_days")
    age_s    = f"{age_days:.0f}d" if age_days is not None else "n/a"
    liq_line = (f"💧 Liq ${liq:,.0f}  ·  Vol h1 {h1_vol_s}  "
                f"·  {txns_h24:,} txns/d  ·  Age {age_s}")

    # Socials + quality
    socials      = alert.get("social_types") or []
    social_icons = {"twitter": "𝕏", "telegram": "TG", "discord": "DC"}
    social_s     = " ".join(social_icons.get(s, s.upper()) for s in socials) or "none"
    web_s        = "Web ✓" if alert.get("has_website") else "Web ✗"
    buy_ratio    = alert.get("buy_ratio")
    buy_s        = f"Buys {buy_ratio*100:.0f}% h1" if buy_ratio is not None else ""
    fdv          = alert.get("fdv_mcap_ratio")
    fdv_s        = f"FDV/MCap {fdv:.1f}×" if fdv else ""
    quality_parts = [s for s in [social_s, web_s, fdv_s, buy_s] if s]
    quality_line  = "🔗 " + "  ·  ".join(quality_parts)

    # Conviction bar
    conv_line = f"📊 Conviction: {_conviction_bar(conv)}"

    # Dimensions
    dims  = alert.get("conv_dims") or {}
    d_org = dims.get("organic",    0.0)
    d_tok = dims.get("tokenomics", 0.0)
    d_mat = dims.get("maturity",   0.0)
    d_mom = dims.get("momentum",   0.0)
    dim_line = (f"📐 Org {d_org:.1f}  ·  Tok {d_tok:.1f}  "
                f"·  Mat {d_mat:.1f}  ·  Mom {d_mom:.1f}")

    # Header line
    header_line = f"{name} ({symbol}) · BASE · {_mcap_line(mcap)}"

    # Trade plan
    plan_line = _trade_plan(mcap, stype)

    # Contract + URL
    token    = pair.get("baseToken") or {}
    contract = token.get("address", "?")
    url      = pair.get("url", "")

    sep = "━" * 35
    return (
        f"{tier_header}\n"
        f"{sep}\n"
        f"{header_line}\n"
        f"\n"
        f"WHY: {why_text}\n"
        f"\n"
        f"{conv_line}\n"
        f"{dim_line}\n"
        f"\n"
        f"{price_line}\n"
        f"{liq_line}\n"
        f"{quality_line}\n"
        f"\n"
        f"📋 TRADE PLAN\n"
        f"   {plan_line}\n"
        f"\n"
        f"Contract: {contract}\n"
        f"{url}"
    )


def _format_resurrection(alert):
    fall_pct = alert.get("fall_pct", 0)
    ath      = alert.get("all_time_high", 0)
    h1_vol   = alert.get("h1_vol", 0)
    h24_vol  = alert.get("h24_vol", 0)
    cur_px   = alert.get("current_price", 0)
    tier_pfx = "LARGE-CAP " if alert.get("largecap_mode") else ("MID-CAP " if alert.get("midcap_mode") else "")
    why = (
        f"Previously fell {fall_pct:.0f}% from ATH (${ath:.4g}) and went "
        f"dormant. Now: h1 vol ${h1_vol:,.0f} / h24 ${h24_vol:,.0f}. "
        f"Dead tokens that revive often run hard — the story is already priced in reverse."
    )
    return _build_alert_body(
        alert,
        tier_header=f"🔥 RIDE SIGNAL — {tier_pfx}DEAD-TOKEN RESURRECTION",
        why_text=why,
    )


def _format_milestone(alert):
    mcap      = alert.get("market_cap") or 0
    milestone = alert.get("milestone", 0)
    ms_labels = {
        100_000:    ("$100K",  "micro-cap to emerging"),
        500_000:    ("$500K",  "gaining real traction"),
        1_000_000:  ("$1M",    "crossed $1M — real project"),
        5_000_000:  ("$5M",    "approaching $10M"),
        10_000_000: ("$10M",   "hit $10M 🚀"),
    }
    ms_str, ms_label = ms_labels.get(milestone, (f"${milestone/1e6:.1f}M", "milestone"))
    remaining = max(0, 10_000_000 - mcap)
    rem_s = f"${remaining/1_000:.0f}K" if remaining < 1_000_000 else f"${remaining/1_000_000:.1f}M"
    tier_pfx = "LARGE-CAP " if alert.get("largecap_mode") else ("MID-CAP " if alert.get("midcap_mode") else "")
    why = (
        f"Crossed {ms_str} mcap ({ms_label}). "
        f"{rem_s} remaining to $10M. "
        f"Factual milestone — useful for tracking the full move from entry."
    )
    return _build_alert_body(
        alert,
        tier_header=f"📊 INFO — {tier_pfx}MCAP MILESTONE: {ms_str}",
        why_text=why,
    )


def format_alert(alert):
    t = alert.get("type")
    if t == "resistance_breakout":
        return _format_breakout(alert)
    if t == "pre_breakout_coil":
        return _format_coil(alert)
    if t == "sleeping_giant_wakeup":
        return _format_giant(alert)
    if t == "slow_build_accumulation":
        return _format_slow_build(alert)
    if t == "dead_token_resurrection":
        return _format_resurrection(alert)
    if t == "mcap_milestone":
        return _format_milestone(alert)
    if t == "volume_surge":
        return _format_surge(alert)
    if t == "momentum_continuation":
        return _format_continuation(alert)
    return _format_spike(alert)


def _format_giant(alert):
    sleep_days = alert.get("sleep_days", 0)
    wake_mult  = alert.get("wake_mult", 0)
    h1_vol     = alert.get("h1_vol", 0)
    baseline   = alert.get("baseline_h24_vol", 0)
    tier_pfx   = "MID-CAP " if alert.get("midcap_mode") else ""
    why = (
        f"Silent for {sleep_days:.0f} days, then h1 volume hit {wake_mult:.0f}× "
        f"its quiet hourly rate (${h1_vol:,.0f} vs ${baseline/24:,.0f}/hr baseline). "
        f"This fires in the first candle of real activity — before price moves. "
        f"Rarest and earliest entry signal."
    )
    return _build_alert_body(
        alert,
        tier_header=f"⚡ ENTRY SIGNAL — {tier_pfx}SLEEPING GIANT WAKEUP ({sleep_days:.0f}d dormant)",
        why_text=why,
    )


def _format_slow_build(alert):
    txns_growth = alert.get("txns_growth", 0)
    txns_now    = alert.get("txns_now", 0)
    txns_then   = alert.get("txns_week_ago", 0)
    px_chg_7d   = alert.get("price_chg_7d", 0)
    liq_chg_7d  = alert.get("liq_chg_7d", 0)
    days        = alert.get("days_of_data", 0)
    why = (
        f"Daily transactions grew {txns_growth*100:.0f}% in 7 days "
        f"({txns_then} → {txns_now}/d) while volume stayed quiet. "
        f"Price {px_chg_7d:+.1f}%, liquidity {liq_chg_7d:+.1f}% — no hype yet. "
        f"Patient accumulation over {days} days. This is what early Virtuals/GitClaw looked like."
    )
    return _build_alert_body(
        alert,
        tier_header="🌱 WATCH SIGNAL — SLOW BUILD ACCUMULATION",
        why_text=why,
    )


def _format_coil(alert):
    pct_away  = alert.get("pct_to_breakout", 0)
    ath_ref   = alert.get("ath_reference", 0)
    h1_vol    = alert.get("h1_vol", 0)
    tier_pfx  = "MID-CAP " if alert.get("midcap_mode") else ""
    why = (
        f"Price is {pct_away:.1f}% below its resistance ceiling (${ath_ref:.6g}), "
        f"with h1 vol ${h1_vol:,.0f} building and dominant buy pressure. "
        f"The coil is tightening — breakout signal fires when it clears. "
        f"Early entry vs waiting for the breakout."
    )
    return _build_alert_body(
        alert,
        tier_header=f"🌱 WATCH SIGNAL — {tier_pfx}PRE-BREAKOUT COIL",
        why_text=why,
    )


def _format_spike(alert):
    tf       = alert.get("timeframe", "h1")
    baseline = alert.get("baseline_h24_vol", 0)
    spike    = alert.get("spike_volume", 0)
    multiple = spike / baseline if baseline > 0 else 0
    mult_txt = f"{multiple:,.0f}×" if multiple > 0 else "from near-zero"
    resistance = alert.get("quiet_price_high")
    tier_pfx   = "MID-CAP " if alert.get("midcap_mode") else ""
    res_note   = f" Ceiling at ${resistance:.6g} — watch for breakout." if resistance else ""
    why = (
        f"{tf.upper()} volume hit ${spike:,.0f} ({mult_txt} the quiet baseline of ${baseline:,.0f}/d). "
        f"Absolute spike — something just changed.{res_note}"
    )
    return _build_alert_body(
        alert,
        tier_header=f"⚡ ENTRY SIGNAL — {tier_pfx}VOLUME SPIKE ({tf.upper()})",
        why_text=why,
    )


def _format_surge(alert):
    window        = alert.get("surge_window", "h1").upper()
    surge_vol     = alert.get("surge_vol", 0)
    surge_ratio   = alert.get("surge_ratio", 0)
    baseline_rate = alert.get("avg_baseline_rate", 0)
    baseline_h24  = alert.get("baseline_h24_vol", 0)
    resistance    = alert.get("quiet_price_high")
    tier_pfx      = "MID-CAP " if alert.get("midcap_mode") else ""
    res_note      = f" Resistance ceiling ${resistance:.6g} — breakout signal fires next." if resistance else ""
    why = (
        f"{window} volume just hit ${surge_vol:,.0f} — that is {surge_ratio:.0f}× "
        f"the quiet hourly rate (${baseline_rate:,.0f}/hr, h24 baseline ${baseline_h24:,.0f}). "
        f"Early wakeup fires in the first hour.{res_note}"
    )
    return _build_alert_body(
        alert,
        tier_header=f"⚡ ENTRY SIGNAL — {tier_pfx}EARLY WAKEUP ({window} {surge_ratio:.0f}× SURGE)",
        why_text=why,
    )


def _format_breakout(alert):
    breakout_pct = alert.get("breakout_pct", 0)
    ceiling      = alert.get("quiet_price_high", 0)
    h1_vol       = alert.get("h1_vol", 0)
    source       = alert.get("ath_source", "")
    src_note     = f" ({source})" if source else ""
    tier_pfx     = "MID-CAP " if alert.get("midcap_mode") else ""
    why = (
        f"Price cleared resistance{src_note} at ${ceiling:.6g} by +{breakout_pct:.1f}%, "
        f"with h1 vol ${h1_vol:,.0f} confirming the move. "
        f"Breakout above consolidation ceiling with volume = primary entry signal. "
        f"This is the setup all earlier signals were leading to."
    )
    return _build_alert_body(
        alert,
        tier_header=f"⚡ ENTRY SIGNAL — {tier_pfx}ATH BREAKOUT (+{breakout_pct:.0f}% above ceiling)",
        why_text=why,
    )


def _format_continuation(alert):
    cont_pct  = alert.get("continuation_pct", 0)
    prev_ref  = alert.get("prev_reference", 0)
    cur_px    = alert.get("current_price", 0)
    h1_vol    = alert.get("h1_vol", 0)
    why = (
        f"Price climbed {cont_pct:.0f}% above the last alert level (${prev_ref:.6g} → "
        f"${cur_px:.6g}), h1 vol ${h1_vol:,.0f} still strong. "
        f"Momentum is intact — the move is not done. Add to winners or hold. "
        f"Trail stop below last support."
    )
    return _build_alert_body(
        alert,
        tier_header=f"🔥 RIDE SIGNAL — STILL RUNNING (+{cont_pct:.0f}% since last alert)",
        why_text=why,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def prune_token_state(state, ttl, now):
    stale = [
        addr for addr, ts in state["tokens"].items()
        if now - ts.get("last_seen_ts", 0) > ttl
    ]
    for addr in stale:
        del state["tokens"][addr]


def _send_daily_digest(notifier, state, now):
    """Send one Telegram message per day summarising everything the system did.

    Covers:
      • Signals sent in last 24 h (from signal_log.jsonl)
      • RL model health: active arms, total observations, per-arm win rate
      • Backtest DB coverage (backtest_db.json) if available
      • Top 5 watchlist tokens by conviction
      • Links to where to see more detail

    Designed to replace manual log-checking. Fires once per day regardless
    of whether the bot sent any alerts.
    """
    date_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%a %d %b %Y")

    # ── 1. Signals in last 24 h ──────────────────────────────────────────────
    sig_counts = {}
    total_sigs_24h = 0
    if os.path.exists(SIGNAL_LOG_FILE):
        cutoff = now - 86400
        with open(SIGNAL_LOG_FILE) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if rec.get("ts", 0) >= cutoff:
                        st = rec.get("signal_type", "?")
                        sig_counts[st] = sig_counts.get(st, 0) + 1
                        total_sigs_24h += 1
                except Exception:
                    pass
    all_time_sigs = 0
    if os.path.exists(SIGNAL_LOG_FILE):
        with open(SIGNAL_LOG_FILE) as fh:
            all_time_sigs = sum(1 for _ in fh)

    sig_tier = {
        "slow_build_accumulation":  "🌱",
        "pre_breakout_coil":        "🌱",
        "sleeping_giant_wakeup":    "⚡",
        "volume_surge":             "⚡",
        "volume_spike":             "⚡",
        "resistance_breakout":      "⚡",
        "momentum_continuation":    "🔥",
        "dead_token_resurrection":  "🔥",
        "mcap_milestone":           "📊",
        "vc_backed_project":        "📊",
    }
    if sig_counts:
        sig_lines = "\n".join(
            f"  {sig_tier.get(st,'  ')} {st.replace('_',' ')}: {n}"
            for st, n in sorted(sig_counts.items(), key=lambda x: -x[1])
        )
    else:
        sig_lines = "  (none — quiet market or bot just started)"

    # ── 2. RL model status ───────────────────────────────────────────────────
    rl_lines = "  No model yet (need 7+ days of signals)"
    rl_outcomes = 0
    if os.path.exists("outcome_log.jsonl"):
        with open("outcome_log.jsonl") as fh:
            rl_outcomes = sum(1 for _ in fh)
    if os.path.exists("rl_model.json"):
        try:
            with open("rl_model.json") as fh:
                rl_data = json.load(fh)
            arms = rl_data.get("arms", {})
            active = [(st, a["n_obs"]) for st, a in arms.items() if a.get("n_obs", 0) >= 3]
            n_total = rl_data.get("n_total", 0)
            trained_ts = rl_data.get("trained")
            trained_ago = ""
            if trained_ts:
                mins = (now - trained_ts) / 60
                trained_ago = (f"{mins:.0f}m ago" if mins < 120
                               else f"{mins/60:.1f}h ago")
            if active:
                arm_txt = ", ".join(
                    f"{st.replace('_',' ')} (n={n})" for st, n in active[:3]
                )
                rl_lines = (
                    f"  Observations: {n_total} | Outcomes: {rl_outcomes}\n"
                    f"  Active arms ({len(active)}): {arm_txt}\n"
                    f"  Last trained: {trained_ago}"
                )
            else:
                rl_lines = (
                    f"  Outcomes: {rl_outcomes} | Observations: {n_total}\n"
                    f"  No active arms yet (need 3+ outcomes per signal type)"
                )
        except Exception:
            rl_lines = f"  {rl_outcomes} outcomes logged, model not yet readable"

    # ── 3. Backtest DB coverage ──────────────────────────────────────────────
    bt_lines = "  No backtest DB yet (remote agent hasn't run)"
    if os.path.exists("backtest_db.json"):
        try:
            with open("backtest_db.json") as fh:
                bt_data = json.load(fh)
            pools = bt_data.get("pools", {})
            analysed = sum(1 for v in pools.values()
                          if isinstance(v, dict) and not v.get("skipped"))
            winners  = sum(1 for v in pools.values()
                          if isinstance(v, dict)
                          and v.get("outcome") in ("10x_plus","5x","2x","winner"))
            wr = winners / analysed * 100 if analysed else 0
            bt_lines = (
                f"  Pools analysed: {analysed:,} | Winners found: {winners:,} ({wr:.0f}%)\n"
                f"  Total DB entries: {len(pools):,}"
            )
        except Exception:
            bt_lines = "  DB exists but unreadable"

    # ── 4. Top watchlist tokens ──────────────────────────────────────────────
    tokens = state.get("tokens", {})
    candidates = []
    for addr, ts in tokens.items():
        snap = ts.get("_pulse_snapshot")
        if not snap:
            continue
        if now - snap.get("ts", 0) > 7200:   # fresh data only
            continue
        conv = snap.get("conviction", 0)
        if conv >= 5.0:
            candidates.append(snap)
    candidates.sort(key=lambda s: s.get("conviction", 0), reverse=True)

    if candidates:
        watch_lines = "\n".join(
            f"  {i+1}. {c.get('name','?')} ({c.get('symbol','?')})  "
            f"conv {c.get('conviction',0):.1f}  "
            f"${c.get('mcap',0)/1000:.0f}K"
            for i, c in enumerate(candidates[:5])
        )
    else:
        watch_lines = "  (no high-conviction tokens in current scan)"

    # ── 5. Assemble and send ─────────────────────────────────────────────────
    sep = "━" * 32
    message = (
        f"📡 DAILY SYSTEM DIGEST — {date_str}\n"
        f"{sep}\n"
        f"\n"
        f"📨 Signals sent last 24h: {total_sigs_24h}  (all-time: {all_time_sigs})\n"
        f"{sig_lines}\n"
        f"\n"
        f"🤖 RL Model (LinUCB)\n"
        f"{rl_lines}\n"
        f"\n"
        f"📊 Universal Backtest DB\n"
        f"{bt_lines}\n"
        f"\n"
        f"👀 Top Watchlist Now\n"
        f"{watch_lines}\n"
        f"\n"
        f"🔗 View logs\n"
        f"  Remote agents: claude.ai/code/routines\n"
        f"  Daily commits: github.com/rgowda96/TradingBot/commits\n"
        f"  Local: tail -f bot.log | train_loop.log"
    )
    try:
        notifier.send(message)
        log.info("daily digest sent")
    except Exception as exc:
        log.warning("daily digest send failed: %s", exc)


def send_watchlist_pulse(cfg, notifier, state, now):
    """Send an hourly digest of high-conviction tokens building toward a signal.

    Scans every tracked token's _pulse_snapshot (written by evaluate_token on
    each quality-passing evaluation), picks the ones with conviction ≥ 5.5 and
    fresh data, and sends a compact Telegram message showing the top 8.

    This fires even when no individual alert has triggered — so the user always
    has a regular 'here's what we're watching' update regardless of market quiet.
    """
    tokens          = state.get("tokens", {})
    pulse_interval  = cfg.get("watchlist_pulse_interval_seconds", 3600)
    freshness_limit = pulse_interval + 600   # one interval + 10 min buffer

    candidates = []
    for addr, ts in tokens.items():
        snap = ts.get("_pulse_snapshot")
        if not snap:
            continue
        # Data must be fresh — seen in this or the previous poll cycle
        if now - snap.get("ts", 0) > freshness_limit:
            continue
        conv = snap.get("conviction", 0)
        if conv < 5.5:
            continue
        # Require at least 2 historical data points so we're not showing
        # brand-new single-cycle observations
        if len(ts.get("conviction_history", [])) < 2:
            continue
        candidates.append((conv, addr, snap))

    if not candidates:
        log.info("watchlist pulse: no tokens with conviction ≥ 5.5 and fresh data")
        return

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = candidates[:8]

    lines = [
        f"📊 HOURLY WATCHLIST UPDATE — {len(candidates)} qualifying token(s) tracked\n"
        f"Showing top {len(top)} by conviction  |  threshold ≥5.5\n"
        "─────────────────────────────────────────"
    ]

    for i, (conv, addr, snap) in enumerate(top, 1):
        name    = snap.get("name", "?")
        symbol  = snap.get("symbol", "?")
        mcap    = snap.get("mcap") or 0
        h1_vol  = snap.get("h1_vol") or 0
        h24_vol = snap.get("h24_vol") or 0
        dims    = snap.get("conv_dims") or {}
        pchg    = snap.get("price_change") or {}
        url     = snap.get("pair_url", "")
        age     = snap.get("pair_age_days")
        txns    = snap.get("txns_h24", 0)

        mcap_str = f"${mcap:,.0f}" if mcap > 0 else "n/a"
        mult_str = f"  [{10_000_000 / mcap:,.0f}× to $10M]" if mcap > 0 else ""
        age_str  = f"{age:.0f}d" if age else "n/a"
        tier     = "HIGH 🔥" if conv >= 7 else "MED 👀"

        d_org = dims.get("organic",    0.0)
        d_tok = dims.get("tokenomics", 0.0)
        d_mat = dims.get("maturity",   0.0)
        d_mom = dims.get("momentum",   0.0)

        m5_chg  = pchg.get("m5",  "?")
        h1_chg  = pchg.get("h1",  "?")
        h24_chg = pchg.get("h24", "?")

        lines.append(
            f"\n{i}. {name} ({symbol}) — {mcap_str}{mult_str}\n"
            f"   Conviction: {conv:.1f}/10 {tier}  |  Age: {age_str}  |  Txns: {txns:,}\n"
            f"   🔄 Org:{d_org:.1f}  💎 Tok:{d_tok:.1f}  ⏳ Mat:{d_mat:.1f}  ⚡ Mom:{d_mom:.1f}\n"
            f"   h1 vol: ${h1_vol:,.0f}   h24 vol: ${h24_vol:,.0f}\n"
            f"   Δ m5:{m5_chg}%  h1:{h1_chg}%  h24:{h24_chg}%\n"
            f"   {url}"
        )

    message = "\n".join(lines)
    notifier.send(message)
    log.info("watchlist pulse sent — %d qualifying tokens, top %d shown",
             len(candidates), len(top))


def run_cycle(dex, cfg, notifier, state, now, force_discovery=False):
    monitored = state["monitored_tokens"]

    if force_discovery or now - state.get("last_discovery_ts", 0) >= cfg["discovery_interval_seconds"]:
        before = len(monitored)
        seen   = set(monitored)
        for addr in discover_tokens(dex, cfg):
            if addr not in seen:
                monitored.append(addr)
                seen.add(addr)
        cap = cfg["max_monitored_tokens"]
        if len(monitored) > cap:
            del monitored[:len(monitored) - cap]
        state["last_discovery_ts"] = now
        log.info("discovery: %d tokens (+%d new)", len(monitored), len(monitored) - before)

    # Merge any new GeckoTerminal 30d highs into token state each cycle
    _apply_gecko_highs(state, cfg)

    monitored_set   = set(monitored)
    tokens_seen     = 0
    tokens_alerting = 0
    alerts_logged   = 0   # total alerts written to log (all types, all conviction levels)
    tg_sent         = 0   # actual Telegram messages sent this cycle

    for batch in chunks(monitored, cfg["request_batch_size"]):
        raw_pairs = dex.tokens(cfg["chain"], batch)
        token_map = _aggregate_by_token(raw_pairs, monitored_set)
        tokens_seen += len(token_map)

        for token_addr, token_data in token_map.items():
            token_state  = state["tokens"].setdefault(token_addr, {})
            token_alerts = evaluate_token(token_data, cfg, token_state, now)
            if not token_alerts:
                continue

            # Per-cycle dedup — priority order (weakest → strongest):
            #   milestone (always fires) + slow_build + resurrection + giant
            #   + volume signals + coil + breakout + continuation
            milestones    = [a for a in token_alerts if a["type"] == "mcap_milestone"]
            resurrections = [a for a in token_alerts if a["type"] == "dead_token_resurrection"]
            giants        = [a for a in token_alerts if a["type"] == "sleeping_giant_wakeup"]
            slow_builds   = [a for a in token_alerts if a["type"] == "slow_build_accumulation"]
            breakouts     = [a for a in token_alerts if a["type"] == "resistance_breakout"]
            continuations = [a for a in token_alerts if a["type"] == "momentum_continuation"]
            coils         = [a for a in token_alerts if a["type"] == "pre_breakout_coil"]
            signals       = [a for a in token_alerts
                             if a["type"] not in ("resistance_breakout", "momentum_continuation",
                                                  "pre_breakout_coil", "sleeping_giant_wakeup",
                                                  "slow_build_accumulation", "mcap_milestone",
                                                  "dead_token_resurrection")]
            if signals:
                signals.sort(key=lambda a: (a.get("conviction", 0),
                                            a.get("surge_ratio", 0)), reverse=True)
                signals = signals[:1]
            # Breakout subsumes all early-warning signals
            if breakouts:
                coils         = []
                giants        = []
                slow_builds   = []
                resurrections = []
            # Giant/resurrection suppress slow_build (stronger early signal)
            if giants or resurrections:
                slow_builds = []
            # Milestones always fire alongside everything else
            deduped = milestones + slow_builds + resurrections + giants + signals + coils + breakouts + continuations

            tokens_alerting += 1
            for alert in deduped:
                message = format_alert(alert)
                atype = alert.get("type", "")
                conviction = alert.get("conviction", 0)
                is_midcap = alert.get("midcap_mode", False)
                label = {
                    "resistance_breakout":   "MID-CAP BREAKOUT"       if is_midcap else "ATH BREAKOUT",
                    "pre_breakout_coil":     "MID-CAP COIL"           if is_midcap else "PRE-BREAKOUT COIL",
                    "volume_surge":          "MID-CAP SURGE"          if is_midcap else "EARLY WAKEUP SURGE",
                    "momentum_continuation": "MOMENTUM CONTINUATION",
                    "sleeping_giant_wakeup":  "MID-CAP SLEEPING GIANT" if is_midcap else "SLEEPING GIANT",
                    "slow_build_accumulation":"SLOW BUILD ACCUMULATION",
                    "dead_token_resurrection":"DEAD TOKEN RESURRECTION",
                    "mcap_milestone":         "MCAP MILESTONE",
                }.get(atype, ("LARGE-CAP SPIKE" if alert.get("largecap_mode")
                              else ("MID-CAP SPIKE" if is_midcap else "VOLUME SPIKE")))
                # RL bonus: LinUCB adds/subtracts up to ±1.5 from conviction
                # based on what it has learned about this feature region.
                # Untrained arms return 0.0 (no influence until 3+ observations).
                bonus    = _rl_bonus(alert)
                eff_conv = conviction + bonus   # effective conviction for gate check

                log.info(
                    "%s [conv=%.1f rl%+.2f → eff=%.1f]:\n%s",
                    label, conviction, bonus, eff_conv, message,
                )
                # Telegram quality gate: mid-cap signals require higher conviction
                # since they're less "entry-level early" and need stronger signal quality.
                tg_min = cfg.get("telegram_min_conviction", {})
                if is_midcap:
                    min_conv = tg_min.get(f"midcap_{atype}", tg_min.get("midcap_default", 6))
                else:
                    min_conv = tg_min.get(atype, tg_min.get("default", 0))
                if eff_conv >= min_conv and tg_sent < cfg.get("max_alerts_per_cycle", 50):
                    notifier.send(message)
                    _log_signal(alert, cfg_chain=cfg.get("chain", "base"))
                    tg_sent += 1
                alerts_logged += 1

        time.sleep(cfg["batch_pause_seconds"])

    # ── CoinGecko enrichment — VC / Ivy League backing detection ────────────
    # Runs at most cg_checks_per_cycle times per cycle (default 3).
    # Checks tokens with conviction ≥ 5.0 that haven't been CG-checked in 7d.
    # If CoinGecko lists the token and has VC portfolio category tags or VC
    # keywords in the project description, fires an "IVY LEAGUE" Telegram alert.
    _cg_budget  = cfg.get("cg_checks_per_cycle", 3)
    _cg_done    = 0
    _cg_min_age = 604_800   # re-check at most once per 7 days

    for _cg_addr, _cg_ts in list(state["tokens"].items()):
        if _cg_done >= _cg_budget:
            break
        _cg_snap = _cg_ts.get("_pulse_snapshot") or {}
        _cg_conv = _cg_snap.get("conviction") or 0
        if _cg_conv < 5.0:
            continue
        if now - (_cg_ts.get("cg_checked_ts") or 0) < _cg_min_age:
            continue   # recently checked
        if _cg_ts.get("vc_backed_alert_ts"):
            continue   # already alerted

        cg = _coingecko_enrich(_cg_addr)
        _cg_ts["cg_checked_ts"] = now
        _cg_done += 1

        if not cg.get("listed"):
            continue

        _cg_ts["cg_info"] = cg
        log.debug("CoinGecko listed: %s (vc=%s)", cg.get("name"), cg.get("has_vc_backing"))

        if cg.get("has_vc_backing"):
            _cg_ts["vc_backed_alert_ts"] = now
            _cg_name   = _cg_snap.get("name",   "?") or cg.get("name",   "?")
            _cg_symbol = _cg_snap.get("symbol", "?") or cg.get("symbol", "?")
            _cg_mcap   = _cg_snap.get("mcap",    0) or 0
            _cg_mult   = int(10_000_000 / _cg_mcap) if _cg_mcap > 0 else 0
            _mult_s    = f"  [{_cg_mult:,}× to $10M]" if _cg_mult else ""
            _vc_cats   = ", ".join(c.title() for c in cg.get("vc_categories", [])[:3])
            _vc_kw     = ", ".join(cg.get("vc_keywords", [])[:3])
            _backing   = _vc_cats or _vc_kw or "CoinGecko-verified project"
            _cg_msg = (
                f"🏛️ IVY LEAGUE PROJECT DETECTED{_mult_s}\n"
                f"{_cg_name} ({_cg_symbol})\n"
                f"Backing: {_backing}\n"
                f"CG score: {cg.get('coingecko_score', 0):.1f}  "
                f"Twitter: {cg.get('twitter_followers', 0):,} followers\n"
                f"MCap: ${_cg_mcap:,.0f}   Conviction: {_cg_conv:.1f}/10\n"
                f"{_cg_snap.get('pair_url', '')}"
            )
            log.info("VC BACKED PROJECT [%s]:\n%s", _cg_symbol, _cg_msg)
            _cg_min_conv = cfg.get("telegram_min_conviction", {}).get("vc_backed_project", 5)
            if _cg_conv >= _cg_min_conv and tg_sent < cfg.get("max_alerts_per_cycle", 50):
                notifier.send(_cg_msg)
                # Log VC alert for RL tracker (minimal synthetic alert dict)
                _log_signal({
                    "type":        "vc_backed_project",
                    "conviction":  _cg_conv,
                    "market_cap":  _cg_mcap,
                    "pair":        {"baseToken": {"address": _cg_addr, "name": _cg_name, "symbol": _cg_symbol},
                                    "url": _cg_snap.get("pair_url", "")},
                    "liquidity":   _cg_snap.get("liq", 0),
                    "pair_age_days": None,
                    "h1_vol":      0,
                    "h24_vol":     0,
                    "conv_dims":   {},
                }, cfg_chain=cfg.get("chain", "base"))
                tg_sent += 1

    # ── Hourly watchlist pulse ───────────────────────────────────────────────
    pulse_interval = cfg.get("watchlist_pulse_interval_seconds", 3600)
    if now - state.get("last_watchlist_pulse_ts", 0) >= pulse_interval:
        send_watchlist_pulse(cfg, notifier, state, now)
        state["last_watchlist_pulse_ts"] = now

    # ── Daily system digest ─────────────────────────────────────────────────
    # Fires once per day — a full status report covering signals, RL model,
    # backtest coverage, and top watchlist. Replaces having to check logs manually.
    if now - state.get("last_daily_digest_ts", 0) >= 86400:
        _send_daily_digest(notifier, state, now)
        state["last_daily_digest_ts"] = now

    prune_token_state(state, cfg["pair_state_ttl_seconds"], now)
    log.info(
        "cycle done — evaluated %d tokens | %d alerting | %d logged | %d Telegram sent",
        tokens_seen, tokens_alerting, alerts_logged, tg_sent,
    )
    return tg_sent


def main():
    parser = argparse.ArgumentParser(description="Base-chain ATH breakout bot")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--once",   action="store_true", help="single cycle then exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")],
    )

    cfg      = load_config(args.config)
    dex      = DexScreener()
    notifier = TelegramNotifier(cfg["telegram_bot_token"], cfg["telegram_chat_id"])
    state    = load_state(cfg["state_file"])

    det = cfg["detection"]
    log.info(
        "started | quiet<=$%s | consol<=$%s | spike>=$%s | mcap<=$%s | liq>=$%s | "
        "surge %sx | buy_ratio>=%s%% | breakout>=%s%% | continuation>=%s%%",
        f"{det['low_volume_max']:,}",
        f"{det.get('consolidation_volume_max', 500_000):,}",
        f"{det['spike_volume_min']:,}",
        f"{det['max_market_cap_usd']:,}",
        f"{det['min_liquidity_usd']:,}",
        det.get("spike_volume_multiplier", 0),
        int(det.get("min_buy_ratio", 0) * 100),
        det.get("resistance_breakout_pct", 5),
        det.get("continuation_pct", 50),
    )

    try:
        while True:
            now = time.time()
            try:
                run_cycle(dex, cfg, notifier, state, now, force_discovery=args.once)
            except Exception:
                log.exception("cycle failed")
            save_state(cfg["state_file"], state)
            if args.once:
                break
            time.sleep(cfg["poll_interval_seconds"])
    except KeyboardInterrupt:
        log.info("stopped by user")
    finally:
        save_state(cfg["state_file"], state)


if __name__ == "__main__":
    main()
