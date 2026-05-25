#!/usr/bin/env python3
"""
DexScreener Token Screener — comprehensive discovery, image-only, live-updating.

Three background workers per chain:
  gecko_worker   — pages through every pool on every DEX via GeckoTerminal.
                   Discovers ALL tokens and stores their images directly.
  search_worker  — searches 700+ terms on DexScreener to catch anything
                   GeckoTerminal hasn't indexed yet.
  cache_worker   — every 90 s fetches live pair data from DexScreener for
                   ALL accumulated addresses, filters to image-only, and
                   publishes the final token list.
"""

import json
import logging
import os
import string
import threading
import time

import requests
from flask import Flask, jsonify, render_template, request

from dexscreener import DexScreener

log = logging.getLogger("screener")
app = Flask(__name__)

SUPPORTED_CHAINS = [
    "base", "ethereum", "bsc", "solana", "arbitrum", "polygon", "avalanche"
]

ADDRESSES_FILE    = "discovered_addresses.json"
PAIR_REFRESH_SECS = 90
PAIR_BATCH_SIZE   = 30
PAIR_BATCH_PAUSE  = 0.35
SEARCH_PAUSE      = 0.5
GECKO_PAGE_PAUSE  = 0.5
GECKO_CYCLE_SLEEP = 600   # re-scan GeckoTerminal every 10 minutes

# GeckoTerminal chain IDs
# Pre-fetched DEX slugs per chain — avoids an extra API call that can get rate-limited
GECKO_DEXES: dict[str, list[str]] = {
    "base": [
        "aerodrome-base", "aerodrome-slipstream", "aerodrome-slipstream-2",
        "uniswap-v3-base", "uniswap-v4-base", "uniswap-v2-base",
        "pancakeswap-v3-base", "pancakeswap-v2-base", "pancakeswap-infinity-clmm-base",
        "virtuals-base", "virtuals-unicorn-base",
        "baseswap", "baseswap-v3",
        "sushiswap-v3-base", "sushiswap-v2-base",
        "curve-base", "balancer-v2-base", "balancer-v3-base",
        "kyberswap-elastic-base", "maverick-v2-base", "smardex-base",
        "alien-base", "alien-base-v3",
        "dackieswap-v3-base", "dackieswap-v2-base",
        "squadswap-base", "squadswap-v3-base",
        "swapbased", "swapbased-v3",
        "equalizer-base", "solidly-v3-base",
        "rocketswap", "horizondex-base", "velocimeter-base",
        "throne-v2-base", "throne-v3-base",
        "archly-base", "baso-finance",
        "diamondswap", "moonbase", "bluedex",
        "leetswap-base", "cbswap", "synthswap", "synthswap-v3",
        "oasisswap-base", "ensurer", "cloudbase", "soswap",
        "icecreamswap-base", "bakeryswap-base", "kokonutswap-base",
        "lfgswap-base", "crescentswap-base", "swapline-base",
        "sobal-base", "derpdex-base", "pixelswap-base",
        "degenbrains-base", "yum-yum-swap", "bunnyswap",
        "plantbaseswap", "dex-on-crypto-base", "iziswap-base",
        "area-51-alien-base", "sharkswap-base", "torus", "citadelswap",
        "satori-base", "robots-farm-base", "memebox-base",
        "marswap-base", "dejungle", "ethervista-base",
        "x7-finance-base", "fwx-base", "kim-v4-base", "kim-v2-base",
        "henjin", "treble-v2", "treble-v4", "treble-v4-limit",
        "9mm-v3-base", "9mm-v2-base", "deltaswap-base",
        "rubicon-base", "hydrex-integral", "kewlswap-base",
        "quickswap-v2-base", "quickswap-v4-base",
        "omni-exchange-v2-base", "omni-exchange-v3-base", "omni-exchange-v4-base",
        "thirdfy", "fluid-base", "mint-club-base", "deli-swap",
    ],
}

GECKO_CHAIN = {
    "base":      "base",
    "ethereum":  "eth",
    "bsc":       "bsc",
    "solana":    "solana",
    "arbitrum":  "arbitrum",
    "polygon":   "polygon_pos",
    "avalanche": "avax",
}

# DexScreener search terms — all 676 two-letter combos + crypto keywords
_SEARCH_TERMS = (
    [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase] +
    [
        "pepe", "doge", "shib", "inu", "floki", "bonk", "wojak", "snek",
        "cat", "dog", "frog", "ape", "chimp", "shark", "whale", "fish",
        "bull", "bear", "pig", "crab", "moon", "sun", "star", "fire",
        "ice", "earth", "wind", "thunder", "dark", "light", "shadow",
        "king", "queen", "god", "devil", "angel", "alien", "robot",
        "chad", "based", "degen", "gem", "meme", "baby", "mini",
        "super", "turbo", "mega", "giga", "ultra", "alpha", "beta",
        "sigma", "omega", "delta", "gamma", "safe", "fair", "rug",
        "pump", "rich", "poor", "cash", "money", "gold", "silver",
        "eth", "btc", "sol", "avax", "arb", "base", "weth",
        "usdc", "usdbc", "cbeth", "defi", "dex", "amm", "pool",
        "vault", "stake", "farm", "yield", "dao", "burn", "mint",
        "launch", "bridge", "layer", "swap", "lock", "nft", "coin",
        "token", "finance", "crypto", "chain",
        "brett", "toshi", "bald", "normie", "higher", "onchain",
        "virtual", "aero", "aerodrome", "uniswap", "curve",
        "blue", "red", "green", "black", "white", "purple", "orange",
        "mars", "space", "galaxy", "nova", "cosmos",
    ]
)
SEARCH_TERMS: dict[str, list[str]] = {c: _SEARCH_TERMS for c in SUPPORTED_CHAINS}

dex = DexScreener()

# ── Persistent address store ───────────────────────────────────────────────────
_discovered: dict[str, set] = {}
_disc_lock  = threading.Lock()


def _load_discovered() -> None:
    if not os.path.exists(ADDRESSES_FILE):
        return
    try:
        with open(ADDRESSES_FILE) as fh:
            data = json.load(fh)
        with _disc_lock:
            for chain, addrs in data.items():
                _discovered[chain] = set(addrs)
        total = sum(len(v) for v in _discovered.values())
        log.info("Loaded %d previously discovered addresses", total)
    except Exception as exc:
        log.warning("Could not load %s: %s", ADDRESSES_FILE, exc)


def _save_discovered() -> None:
    try:
        with _disc_lock:
            data = {c: list(a) for c, a in _discovered.items()}
        tmp = ADDRESSES_FILE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, ADDRESSES_FILE)
    except Exception as exc:
        log.warning("Could not save addresses: %s", exc)


def _add_addresses(chain: str, addrs: set) -> int:
    with _disc_lock:
        s = _discovered.setdefault(chain, set())
        before = len(s)
        s.update(addrs)
        return len(s) - before


def _get_addresses(chain: str) -> list:
    with _disc_lock:
        return list(_discovered.get(chain, set()))


# ── Image / profile metadata ───────────────────────────────────────────────────
_meta: dict[str, dict] = {}    # addr → { icon, has_profile, has_boost }
_meta_lock = threading.Lock()


def _upsert_meta(addr: str, icon=None, has_profile=False, has_boost=False) -> None:
    with _meta_lock:
        ex = _meta.get(addr, {})
        _meta[addr] = {
            "icon":        icon or ex.get("icon"),
            "has_profile": has_profile or ex.get("has_profile", False),
            "has_boost":   has_boost   or ex.get("has_boost",   False),
        }


def _get_meta(addr: str) -> dict:
    with _meta_lock:
        return dict(_meta.get(addr, {}))


# ── DexScreener profile / boost refresh ───────────────────────────────────────
def _refresh_dex_profiles(chain: str) -> None:
    new: set = set()
    for item in dex.token_profiles_latest():
        if item.get("chainId") != chain:
            continue
        addr = (item.get("tokenAddress") or "").lower()
        if addr:
            _upsert_meta(addr, icon=item.get("icon"), has_profile=True)
            new.add(addr)
    for item in dex.token_boosts_latest() + dex.token_boosts_top():
        if item.get("chainId") != chain:
            continue
        addr = (item.get("tokenAddress") or "").lower()
        if addr:
            _upsert_meta(addr, has_boost=True)
            new.add(addr)
    added = _add_addresses(chain, new)
    if added:
        log.info("[profiles:%s] +%d from profiles/boosts", chain, added)


# ── Token builder ─────────────────────────────────────────────────────────────
def _build_token(pair: dict, meta: dict, now_ms: float):
    icon = meta.get("icon") or (pair.get("info") or {}).get("imageUrl")
    if not icon:
        return None
    base       = pair.get("baseToken") or {}
    liq        = (pair.get("liquidity") or {}).get("usd") or 0
    mcap       = pair.get("marketCap") or pair.get("fdv") or 0
    fdv        = pair.get("fdv") or 0
    vol        = pair.get("volume") or {}
    txns24     = (pair.get("txns") or {}).get("h24") or {}
    buys       = txns24.get("buys") or 0
    sells      = txns24.get("sells") or 0
    change     = (pair.get("priceChange") or {}).get("h24") or 0
    created_at = pair.get("pairCreatedAt") or 0
    age_h      = (now_ms - created_at) / 3_600_000 if created_at else None
    return {
        "address":      (base.get("address") or "").lower(),
        "name":         base.get("name", "?"),
        "symbol":       base.get("symbol", "?"),
        "icon":         icon,
        "pair_address": pair.get("pairAddress", ""),
        "dex":          pair.get("dexId", "?"),
        "price_usd":    pair.get("priceUsd") or "0",
        "liquidity":    liq,
        "market_cap":   mcap,
        "fdv":          fdv,
        "vol_h24":      vol.get("h24") or 0,
        "vol_h6":       vol.get("h6") or 0,
        "vol_h1":       vol.get("h1") or 0,
        "change_h24":   change,
        "txns_h24":     buys + sells,
        "buys_h24":     buys,
        "sells_h24":    sells,
        "age_hours":    age_h,
        "has_profile":  meta.get("has_profile", False),
        "has_boost":    meta.get("has_boost",   False),
        "url":          pair.get("url", ""),
    }


# ── Token cache ────────────────────────────────────────────────────────────────
_cache: dict = {}
_cache_lock = threading.Lock()


def _set_cache(chain: str, tokens: list, disc: int) -> None:
    with _cache_lock:
        _cache[chain] = {"tokens": tokens, "last_updated": time.time(),
                         "discovered_count": disc}


def _get_cache(chain: str) -> dict:
    with _cache_lock:
        return dict(_cache.get(chain, {"tokens": [], "last_updated": 0,
                                       "discovered_count": 0}))


def _get_tokens(chain: str) -> list:
    return list(_get_cache(chain).get("tokens", []))


# ══════════════════════════════════════════════════════════════════════════════
# WORKER 1 — GeckoTerminal: pages every DEX on the chain to get ALL pools
# ══════════════════════════════════════════════════════════════════════════════
def _gecko_worker(chain: str) -> None:
    gecko_chain = GECKO_CHAIN.get(chain, chain)
    session = requests.Session()
    session.headers.update({"Accept": "application/json",
                             "User-Agent": "token-screener/1.0"})
    base_url = f"https://api.geckoterminal.com/api/v2/networks/{gecko_chain}"

    log.info("[gecko:%s] Started", chain)

    def _fetch_page(url, page):
        try:
            r = session.get(url, params={"page": page,
                                         "include": "base_token,quote_token"},
                            timeout=15)
            if r.status_code in (429, 503):
                time.sleep(5)
                return None
            if not r.ok:
                return None
            return r.json()
        except Exception as exc:
            log.debug("[gecko:%s] fetch error: %s", chain, exc)
            return None

    def _drain_endpoint(url, label):
        """Page through an endpoint until empty, harvesting addresses + images."""
        total_new = 0
        page = 1
        while True:
            data = _fetch_page(url, page)
            if not data or not data.get("data"):
                break
            new_addrs: set = set()
            for item in data.get("included", []):
                if item.get("type") != "token":
                    continue
                attrs = item.get("attributes", {})
                addr  = (attrs.get("address") or "").lower()
                img   = attrs.get("image_url") or ""
                if not addr or not addr.startswith("0x"):
                    continue
                new_addrs.add(addr)
                if img and "missing" not in img:
                    _upsert_meta(addr, icon=img)
            added = _add_addresses(chain, new_addrs)
            total_new += added
            page += 1
            time.sleep(GECKO_PAGE_PAUSE)
        if total_new:
            log.info("[gecko:%s] %s → +%d new addresses (total=%d)",
                     chain, label, total_new, len(_get_addresses(chain)))

    while True:
        try:
            # ── 1. Global pools list (top by volume) ───────────────────────
            _drain_endpoint(f"{base_url}/pools", "all_pools")
            time.sleep(2)

            # ── 2. Newest pools (recently launched) ────────────────────────
            _drain_endpoint(f"{base_url}/new_pools", "new_pools")
            time.sleep(2)

            # ── 3. Per-DEX pool sweep — every pool on every exchange ────────
            # Use hardcoded list first; refresh dynamically for unknown chains
            dex_ids = list(GECKO_DEXES.get(chain, []))
            if not dex_ids:
                try:
                    r = session.get(
                        f"https://api.geckoterminal.com/api/v2/networks/{gecko_chain}/dexes",
                        timeout=15)
                    if r.ok:
                        dex_ids = [d["id"] for d in r.json().get("data", [])]
                except Exception as exc:
                    log.warning("[gecko:%s] dexes fetch: %s", chain, exc)

            log.info("[gecko:%s] Scanning %d DEXes pool-by-pool…", chain, len(dex_ids))
            for dex_id in dex_ids:
                _drain_endpoint(f"{base_url}/dexes/{dex_id}/pools", f"dex:{dex_id}")
                time.sleep(GECKO_PAGE_PAUSE)

            _save_discovered()
            log.info("[gecko:%s] Full cycle done. Discovered: %d addrs. Sleeping %ds.",
                     chain, len(_get_addresses(chain)), GECKO_CYCLE_SLEEP)

        except Exception:
            log.exception("[gecko:%s] Unhandled error", chain)

        time.sleep(GECKO_CYCLE_SLEEP)


# ══════════════════════════════════════════════════════════════════════════════
# WORKER 2 — DexScreener search: 700+ search terms sweep
# ══════════════════════════════════════════════════════════════════════════════
def _search_worker(chain: str) -> None:
    terms     = SEARCH_TERMS.get(chain, _SEARCH_TERMS)
    idx       = 0
    save_tick = 0
    log.info("[search:%s] Started — %d terms", chain, len(terms))

    while True:
        term = terms[idx % len(terms)]
        idx += 1
        try:
            pairs = dex.search(term)
            new_addrs: set = set()
            for pair in pairs:
                if pair.get("chainId") != chain:
                    continue
                addr = (pair.get("baseToken") or {}).get("address", "").lower()
                if not addr:
                    continue
                new_addrs.add(addr)
                img = (pair.get("info") or {}).get("imageUrl")
                if img:
                    _upsert_meta(addr, icon=img)
            added = _add_addresses(chain, new_addrs)
            if added:
                log.info("[search:%s] term=%r +%d  total=%d",
                         chain, term, added, len(_get_addresses(chain)))
            save_tick += 1
            if save_tick % 50 == 0:
                _save_discovered()
        except Exception as exc:
            log.warning("[search:%s] error term=%r: %s", chain, term, exc)

        time.sleep(SEARCH_PAUSE)


# ══════════════════════════════════════════════════════════════════════════════
# WORKER 3 — Cache: fetches pair data for ALL discovered addresses every 90 s
# ══════════════════════════════════════════════════════════════════════════════
def _cache_worker(chain: str) -> None:
    log.info("[cache:%s] Started", chain)

    while True:
        try:
            _refresh_dex_profiles(chain)

            addresses  = _get_addresses(chain)
            disc_count = len(addresses)

            if not addresses:
                time.sleep(10)
                continue

            log.info("[cache:%s] Fetching pair data for %d addresses…", chain, disc_count)
            now_ms    = time.time() * 1000
            best_pair: dict[str, dict] = {}

            for i in range(0, len(addresses), PAIR_BATCH_SIZE):
                batch = addresses[i : i + PAIR_BATCH_SIZE]
                try:
                    for pair in dex.tokens(chain, batch):
                        addr = (pair.get("baseToken") or {}).get("address", "").lower()
                        if not addr:
                            continue
                        liq     = (pair.get("liquidity") or {}).get("usd") or 0
                        cur_liq = ((best_pair.get(addr) or {}).get("liquidity") or {}).get("usd") or 0
                        if liq >= cur_liq:
                            best_pair[addr] = pair
                except Exception as exc:
                    log.warning("[cache:%s] batch error: %s", chain, exc)
                time.sleep(PAIR_BATCH_PAUSE)

            tokens = []
            for addr, pair in best_pair.items():
                meta = _get_meta(addr)
                if not meta:
                    img = (pair.get("info") or {}).get("imageUrl")
                    if img:
                        meta = {"icon": img, "has_profile": False, "has_boost": False}
                    else:
                        continue
                t = _build_token(pair, meta, now_ms)
                if t:
                    tokens.append(t)

            _set_cache(chain, tokens, disc_count)
            log.info("[cache:%s] ✓ %d image-verified tokens / %d addresses",
                     chain, len(tokens), disc_count)

        except Exception:
            log.exception("[cache:%s] Unhandled error", chain)

        time.sleep(PAIR_REFRESH_SECS)


# ── Watchlist criteria ─────────────────────────────────────────────────────────
WATCHLIST_CRITERIA = {
    "chain":    "base",
    "liq_min":  1_000,
    "mcap_min": 10_000,
    "mcap_max": 500_000,
    "age_min":  100,
    "age_max":  3_000,
    "vol_min":  1,
    "vol_max":  10_000,
}


def _apply_watchlist(tokens: list) -> list:
    c = WATCHLIST_CRITERIA
    def passes(t):
        age = t["age_hours"]
        return (
            t["liquidity"]  >= c["liq_min"]  and
            t["market_cap"] >= c["mcap_min"] and
            (not t["market_cap"] or t["market_cap"] <= c["mcap_max"]) and
            age is not None and c["age_min"] <= age <= c["age_max"] and
            c["vol_min"] <= t["vol_h24"] <= c["vol_max"]
        )
    result = [t for t in tokens if passes(t)]
    result.sort(key=lambda t: t["market_cap"] or 0)
    return result


# ── Filter helper ──────────────────────────────────────────────────────────────
def _apply_filters(tokens: list, args) -> list:
    def fp(k):
        v = args.get(k, "")
        try: return float(v) if v.strip() else None
        except ValueError: return None

    liq_min = fp("liq_min");   liq_max  = fp("liq_max")
    mcap_min= fp("mcap_min");  mcap_max = fp("mcap_max")
    fdv_min = fp("fdv_min");   fdv_max  = fp("fdv_max")
    age_min = fp("age_min");   age_max  = fp("age_max")
    txns_min= fp("txns_min");  txns_max = fp("txns_max")
    buys_min= fp("buys_min");  buys_max = fp("buys_max")
    sells_min=fp("sells_min"); sells_max= fp("sells_max")
    vol_min = fp("vol_min");   vol_max  = fp("vol_max")
    chg_min = fp("chg_min");   chg_max  = fp("chg_max")
    only_profile = args.get("profile") == "1"
    only_boosted = args.get("boosted") == "1"

    def passes(t):
        liq = t["liquidity"]; mcap = t["market_cap"]; fdv = t["fdv"]
        age = t["age_hours"]; vol  = t["vol_h24"];    chg = t["change_h24"]
        txns= t["txns_h24"];  buys = t["buys_h24"];   sells= t["sells_h24"]
        if liq_min   and liq  < liq_min:  return False
        if liq_max   and liq  > liq_max:  return False
        if mcap_min  and mcap < mcap_min: return False
        if mcap_max  and mcap and mcap > mcap_max: return False
        if fdv_min   and fdv  < fdv_min:  return False
        if fdv_max   and fdv  and fdv > fdv_max:  return False
        if age_min   and (age is None or age < age_min): return False
        if age_max   and (age is None or age > age_max): return False
        if txns_min  and txns < txns_min: return False
        if txns_max  and txns > txns_max: return False
        if buys_min  and buys < buys_min: return False
        if buys_max  and buys > buys_max: return False
        if sells_min and sells < sells_min: return False
        if sells_max and sells > sells_max: return False
        if vol_min   and vol  < vol_min:  return False
        if vol_max   and vol  > vol_max:  return False
        if chg_min is not None and chg < chg_min: return False
        if chg_max is not None and chg > chg_max: return False
        if only_profile and not t["has_profile"]: return False
        if only_boosted and not t["has_boost"]:   return False
        return True

    return [t for t in tokens if passes(t)]


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", chains=SUPPORTED_CHAINS)

@app.route("/watchlist")
def watchlist():
    return render_template("watchlist.html", criteria=WATCHLIST_CRITERIA)

@app.route("/api/watchlist")
def api_watchlist():
    tokens   = _get_tokens(WATCHLIST_CRITERIA["chain"])
    filtered = _apply_watchlist(tokens)
    cc       = _get_cache(WATCHLIST_CRITERIA["chain"])
    return jsonify({"tokens": filtered, "total": len(filtered),
                    "last_updated": cc["last_updated"],
                    "discovered_count": cc["discovered_count"],
                    "criteria": WATCHLIST_CRITERIA})

@app.route("/api/tokens")
def api_tokens():
    chain = request.args.get("chain", "base").lower()
    if chain not in SUPPORTED_CHAINS:
        return jsonify({"error": "unsupported chain"}), 400
    sort_by  = request.args.get("sort", "vol_h24")
    sort_dir = request.args.get("dir", "desc")
    SORT_OK  = {"vol_h24","vol_h6","vol_h1","liquidity","market_cap",
                "fdv","change_h24","txns_h24","buys_h24","sells_h24","age_hours"}
    if sort_by not in SORT_OK:
        sort_by = "vol_h24"
    tokens   = _get_tokens(chain)
    filtered = _apply_filters(tokens, request.args)
    filtered.sort(key=lambda t: (t[sort_by] or 0), reverse=(sort_dir != "asc"))
    cc = _get_cache(chain)
    return jsonify({"tokens": filtered[:500], "total": len(filtered),
                    "last_updated": cc["last_updated"],
                    "discovered_count": cc["discovered_count"]})

@app.route("/api/status")
def api_status():
    chain = request.args.get("chain", "base").lower()
    cc    = _get_cache(chain)
    return jsonify({
        "chain":            chain,
        "image_tokens":     len(cc.get("tokens", [])),
        "discovered_addrs": len(_get_addresses(chain)),
        "last_updated":     cc.get("last_updated", 0),
        "age_seconds":      int(time.time() - cc.get("last_updated", 0)),
    })


# ── Startup ────────────────────────────────────────────────────────────────────
def _start_workers(chain: str) -> None:
    for target, name in [(_gecko_worker,  f"gecko-{chain}"),
                         (_search_worker, f"search-{chain}"),
                         (_cache_worker,  f"cache-{chain}")]:
        threading.Thread(target=target, args=(chain,),
                         daemon=True, name=name).start()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=5001)
    parser.add_argument("--chain", default="base")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    _load_discovered()
    _start_workers(args.chain)
    log.info("Screener on http://0.0.0.0:%d  chain=%s", args.port, args.chain)
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
