#!/usr/bin/env python3
"""
base_intelligence.py
────────────────────
Base Chain Intelligence Scanner — finds projects with elite founder backgrounds.

Exactly the kind of research that would have surfaced Virtuals Protocol
(Imperial College London founders) months before the 10,000× move.

Data sources (all free, no API keys):
  • CoinGecko     — descriptions, VC categories, GitHub links, community stats
  • GeckoTerminal — all active Base pools with volume/liquidity
  • DexScreener   — social links, pair data, website URLs
  • GitHub API    — repo stars, contributors, README analysis
  • Website scraper — /team /about /founders pages

Elite background detection:
  • Top-tier universities  (Imperial, MIT, Stanford, Oxford, Cambridge, ETH…)
  • Prestigious employers  (Google/DeepMind, Goldman, Citadel, Jane Street…)
  • VC portfolio categories on CoinGecko
  • Academic credentials   (PhD, professor, researcher, postdoc)
  • Prior exit indicators  (acquired, IPO, co-founder, raised $Xm)
  • Open-source cred       (GitHub stars, commit velocity)

Two-pass scanning strategy:
  Pass 1 (fast) — CoinGecko description + VC categories only.  No external HTTP.
                  Scans all projects in the mcap range in ~2 minutes.
  Pass 2 (deep) — website team pages + GitHub README + Twitter bio.
                  Only runs on projects that scored > 0 in Pass 1.
                  Keeps total runtime under 5 minutes.

Usage:
    python3 base_intelligence.py              # full scan → report.json + terminal
    python3 base_intelligence.py --quick      # Pass 1 only (CoinGecko, fastest ~1 min)
    python3 base_intelligence.py --token 0x… # deep-dive a single token address
    python3 base_intelligence.py --refresh    # force re-fetch (ignore 48h cache)
    python3 base_intelligence.py --min-mcap 50000 --max-mcap 50000000
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# Meme token detection — skip before scoring to prevent false positives
# ─────────────────────────────────────────────────────────────────────────────

MEME_NAME_KEYWORDS = {
    "doge", "pepe", "shib", "inu", "moon", "safe", "elon", "baby",
    "wojak", "floki", "bonk", "wif", "mog", "brett", "toshi",
    "tremp", "boden", "ponke", "pnut", "andy", "harambe", "grok",
    "kek", "wagmi", "ngmi", "fren", "cope", "seethe", "rug", "pump",
    "lambo", "copium", "doomer", "chad", "sigma", "alpha male",
    "ape together", "diamond hands", "hodl", "rekt", "ngmi", "gm ",
    "fomo", "fud", "based god", "based af",
}

MEME_CG_CATEGORIES = {
    "meme-token", "memecoins", "dog-themed-coins", "cat-themed-coins",
    "frog-themed-tokens", "solana meme coins", "base meme coins",
    "meme", "memes", "animal-racing",
}

# Supply fingerprint numbers that meme deployers use for psychological effect
MEME_SUPPLY_PATTERNS = re.compile(r"\b(69|420|1337|6969|42069)\b")


def _is_meme_token(name: str, symbol: str, categories: list, description: str = "") -> bool:
    """
    Return True if token appears to be a meme/joke token.
    Checks name keywords, CoinGecko categories, and description signals.
    """
    nl   = name.lower()
    sl   = symbol.lower()
    cats = {c.lower() for c in (categories or [])}
    dl   = (description or "").lower()

    # Hard category match
    if cats & MEME_CG_CATEGORIES:
        return True

    # Name contains meme keyword
    if any(kw in nl for kw in MEME_NAME_KEYWORDS):
        return True

    # Symbol looks memey (69, 420, etc.)
    if MEME_SUPPLY_PATTERNS.search(symbol):
        return True

    # Description explicitly says meme
    if any(phrase in dl for phrase in ("meme coin", "memecoin", "meme token",
                                        "just for fun", "no utility", "community token",
                                        "fun project", "joke token")):
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Elite background keyword databases
# ─────────────────────────────────────────────────────────────────────────────

ELITE_UNIVERSITIES = {
    # ── United Kingdom ────────────────────────────────────────────────────────
    "imperial college":            ("Imperial College London", 10),
    "imperial college london":     ("Imperial College London", 10),
    "imperial college of london":  ("Imperial College London", 10),
    "university of oxford":        ("Oxford University",       10),
    "oxford university":           ("Oxford University",       10),
    "university of cambridge":     ("Cambridge University",    10),
    "cambridge university":        ("Cambridge University",    10),
    "ucl":                         ("UCL",                      8),
    "university college london":   ("UCL",                      8),
    "lse":                         ("LSE",                      8),
    "london school of economics":  ("LSE",                      8),
    "king's college london":       ("King's College London",    7),
    "edinburgh university":        ("University of Edinburgh",  7),
    # ── United States ─────────────────────────────────────────────────────────
    "mit":                         ("MIT",                     10),
    "massachusetts institute of technology": ("MIT",           10),
    "stanford university":         ("Stanford",                10),
    "stanford":                    ("Stanford",                10),
    "harvard university":          ("Harvard",                 10),
    "harvard":                     ("Harvard",                 10),
    "caltech":                     ("Caltech",                  9),
    "california institute of technology": ("Caltech",           9),
    "carnegie mellon":             ("CMU",                      9),
    "cmu":                         ("CMU",                      9),
    "princeton university":        ("Princeton",                9),
    "princeton":                   ("Princeton",                9),
    "yale university":             ("Yale",                     9),
    "columbia university":         ("Columbia",                 8),
    "university of chicago":       ("UChicago",                 8),
    "uchicago":                    ("UChicago",                 8),
    "cornell university":          ("Cornell",                  8),
    "university of pennsylvania":  ("UPenn/Wharton",            8),
    "wharton":                     ("Wharton",                  9),
    "uc berkeley":                 ("UC Berkeley",              9),
    "berkeley":                    ("UC Berkeley",              8),
    "georgia tech":                ("Georgia Tech",             7),
    "university of michigan":      ("UMichigan",                7),
    "nyu stern":                   ("NYU Stern",                7),
    # ── Europe ───────────────────────────────────────────────────────────────
    "eth zurich":                  ("ETH Zürich",              10),
    "epfl":                        ("EPFL",                     9),
    "tu munich":                   ("TU Munich",                8),
    "technical university munich": ("TU Munich",                8),
    "tu delft":                    ("TU Delft",                 8),
    "technische universiteit delft": ("TU Delft",               8),
    "tum":                         ("TU Munich",                8),
    "sorbonne":                    ("Sorbonne",                 7),
    "ecole polytechnique":         ("École Polytechnique",      9),
    "hec paris":                   ("HEC Paris",                8),
    "insead":                      ("INSEAD",                   8),
    # ── Asia-Pacific ──────────────────────────────────────────────────────────
    "nus":                         ("NUS",                      9),
    "national university of singapore": ("NUS",                 9),
    "ntu singapore":               ("NTU Singapore",            8),
    "iit bombay":                  ("IIT Bombay",               9),
    "iit delhi":                   ("IIT Delhi",                9),
    "iit":                         ("IIT",                      8),
    "tsinghua university":         ("Tsinghua",                 9),
    "tsinghua":                    ("Tsinghua",                 9),
    "peking university":           ("Peking University",        8),
    "university of toronto":       ("U of Toronto",             8),
    "waterloo":                    ("UWaterloo",                 8),
    "mcgill":                      ("McGill",                   7),
    # ── Research / Academic credentials (anywhere) ────────────────────────────
    "phd":                         ("PhD",                      6),
    "ph.d":                        ("PhD",                      6),
    "dphil":                       ("DPhil (Oxford)",           8),
    "postdoc":                     ("Postdoc",                  5),
    "professor":                   ("Professor",                7),
    "associate professor":         ("Associate Prof",           7),
    "research scientist":          ("Research Scientist",       6),
    "principal researcher":        ("Principal Researcher",     7),
}

ELITE_EMPLOYERS = {
    # ── Big Tech / AI ─────────────────────────────────────────────────────────
    # NOTE: Prefer longer/more specific phrases to avoid false positives.
    # "google" alone fires on "Google login / Google Play" — use "ex-google" or
    # "google engineer"; bare "google" is only kept because "googler" is specific.
    "googler":         ("Google",          8),     # "I'm a Googler" = unmistakable
    "ex-google":       ("Google (ex)",     8),
    "former google":   ("Google (ex)",     8),
    "deepmind":        ("DeepMind",       10),
    "google brain":    ("Google Brain",   10),
    "google research": ("Google Research", 9),
    "openai":          ("OpenAI",         10),
    "anthropic":       ("Anthropic",      10),
    "meta ai":         ("Meta AI",         9),
    "meta research":   ("Meta Research",   9),
    "microsoft research": ("Microsoft Research", 9),
    # "apple", "amazon", "aws" removed — too often appear as integrations/metaphors
    "stripe":          ("Stripe",          9),
    "palantir":        ("Palantir",        8),
    "anduril":         ("Anduril",         8),
    "waymo":           ("Waymo",           9),
    # ── Crypto / Web3 ─────────────────────────────────────────────────────────
    "coinbase":        ("Coinbase",        9),
    "binance":         ("Binance",         8),
    "kraken":          ("Kraken",          7),
    "uniswap labs":    ("Uniswap Labs",    9),
    # NOTE: "aave", "compound", "chainlink", "polygon", "optimism", "base" removed —
    # these are protocol/chain names that appear in descriptions as integrations,
    # not as employer signals. "coinbase" kept because "ex-Coinbase engineer" is common.
    "protocol labs":   ("Protocol Labs",   8),
    # ── Finance / Quant ──────────────────────────────────────────────────────
    "goldman sachs":   ("Goldman Sachs",  10),
    # "goldman" alone removed — double-counts when "Goldman Sachs" appears in text
    "morgan stanley":  ("Morgan Stanley",  8),
    "jp morgan":       ("JPMorgan",        8),
    "jpmorgan":        ("JPMorgan",        8),   # "JPMorgan" vs "JP Morgan" text variants
    "blackrock":       ("BlackRock",       8),
    "citadel":         ("Citadel",        10),
    "jane street":     ("Jane Street",    10),
    "two sigma":       ("Two Sigma",       9),
    "de shaw":         ("D.E. Shaw",       9),
    "hudson river":    ("Hudson River",    8),
    "renaissance technologies": ("Renaissance", 10),
    "bridgewater":     ("Bridgewater",     9),
    "point72":         ("Point72",         8),
    # ── VC / Accelerators ────────────────────────────────────────────────────
    "y combinator":    ("Y Combinator",    9),
    "ycombinator":     ("Y Combinator",    9),
    "yc alum":         ("YC Alumni",       9),
    "a16z":            ("a16z",            9),
    "andreessen horowitz": ("a16z",        9),
    "paradigm":        ("Paradigm",        9),
    "sequoia":         ("Sequoia",         9),
    "coinbase ventures": ("Coinbase Ventures", 9),
    "pantera":         ("Pantera Capital", 8),
    "dragonfly":       ("Dragonfly",       8),
    "multicoin":       ("Multicoin",       8),
    "polychain":       ("Polychain",       8),
    "binance labs":    ("Binance Labs",    8),
    "framework ventures": ("Framework Ventures", 8),
    "electric capital":("Electric Capital", 8),
}

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compiled keyword regex patterns (word-boundary aware, case-insensitive)
# Built lazily on first use to avoid startup overhead.
# ─────────────────────────────────────────────────────────────────────────────
_KW_RE_CACHE = {}

def _kw_re(kw: str):
    """Return a compiled regex that matches `kw` at word boundaries."""
    if kw not in _KW_RE_CACHE:
        _KW_RE_CACHE[kw] = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
    return _KW_RE_CACHE[kw]


EXIT_INDICATORS = {
    "acquired by":       4,
    "acquisition by":    4,
    "sold to":           3,
    "ipo":               4,
    "went public":       3,
    "series a":          3,
    "series b":          4,
    "raised $":          2,
    "raised $1m":        2,
    "raised $5m":        3,
    "raised $10m":       4,
    "raised $20m":       5,
    "raised $50m":       5,
    "backed by":         2,
    "former founder":    2,
    "co-founder":        2,
    "previously built":  2,
    "previously founded":2,
}

# CoinGecko VC portfolio categories that indicate institutional backing
VC_CG_CATEGORIES = {
    "coinbase ventures portfolio": ("Coinbase Ventures", 9),
    "a16z portfolio":              ("a16z",              9),
    "paradigm portfolio":          ("Paradigm",          9),
    "polychain capital portfolio": ("Polychain",         8),
    "multicoin capital portfolio": ("Multicoin",         8),
    "framework ventures portfolio":("Framework Ventures",8),
    "delphi digital portfolio":    ("Delphi Digital",    8),
    "1confirmation":               ("1Confirmation",     7),
    "placeholder portfolio":       ("Placeholder VC",    7),
    "dragonfly capital portfolio": ("Dragonfly",         8),
    "sequoia capital portfolio":   ("Sequoia",           9),
    "pantera capital portfolio":   ("Pantera",           8),
    "electric capital portfolio":  ("Electric Capital",  8),
    "variant portfolio":           ("Variant Fund",      8),
    "y combinator alumni":         ("YC",                9),
    "binance labs portfolio":      ("Binance Labs",      8),
}

# ─────────────────────────────────────────────────────────────────────────────
# Disk cache — 48h TTL per coin_id
# ─────────────────────────────────────────────────────────────────────────────

CACHE_FILE = "base_intelligence_cache.json"
CACHE_TTL  = 48 * 3600          # 48 hours
_cache_lock = threading.Lock()


def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _cache_get(cache: dict, coin_id: str, force_refresh: bool):
    if force_refresh or coin_id not in cache:
        return None
    entry = cache[coin_id]
    if time.time() - entry.get("_cached_at", 0) < CACHE_TTL:
        return entry
    return None


def _cache_put(cache: dict, coin_id: str, record: dict):
    record["_cached_at"] = time.time()
    with _cache_lock:
        cache[coin_id] = record
    _save_cache(cache)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

HEADERS_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
HEADERS_API = {"Accept": "application/json", "User-Agent": "base-intelligence/1.0"}

SESSION = requests.Session()
SESSION.headers.update(HEADERS_API)


def _get(url, params=None, timeout=8, retries=2, browser=False):
    h = HEADERS_BROWSER if browser else HEADERS_API
    for attempt in range(retries + 1):
        try:
            r = SESSION.get(url, params=params, headers=h, timeout=timeout)
            if r.ok:
                return r
            if r.status_code in (429, 503):
                time.sleep(2 ** attempt)
                continue
            return r
        except Exception:
            if attempt < retries:
                time.sleep(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Background detection engine
# ─────────────────────────────────────────────────────────────────────────────

def _scan_text(text: str) -> dict:
    """
    Scan free-form text for elite background signals.
    Uses word-boundary regex to prevent false positives from substrings
    (e.g. "MIT" matching inside "committed", "NUS" inside "bonus").

    Returns dict: {unis, employers, exit_signals, raw_score, evidence}.
    """
    if not text:
        return {"unis": [], "employers": [], "exits": [], "raw_score": 0, "evidence": []}

    unis, employers, exits, evidence = [], [], [], []
    score = 0

    for kw, (label, pts) in ELITE_UNIVERSITIES.items():
        m = _kw_re(kw).search(text)
        if m:
            unis.append(label)
            score += pts
            start = m.start()
            ctx   = text[max(0, start-50):start+len(kw)+80].strip()
            evidence.append(f"[UNI] {label}: «{ctx}»")

    for kw, (label, pts) in ELITE_EMPLOYERS.items():
        m = _kw_re(kw).search(text)
        if m:
            employers.append(label)
            score += pts
            start = m.start()
            ctx   = text[max(0, start-50):start+len(kw)+80].strip()
            evidence.append(f"[EMP] {label}: «{ctx}»")

    for kw, pts in EXIT_INDICATORS.items():
        m = _kw_re(kw).search(text)
        if m:
            exits.append(kw)
            score += pts
            start = m.start()
            ctx   = text[max(0, start-40):start+len(kw)+60].strip()
            evidence.append(f"[EXIT] {kw}: «{ctx}»")

    # Deduplicate
    unis      = list(dict.fromkeys(unis))
    employers = list(dict.fromkeys(employers))
    exits     = list(dict.fromkeys(exits))
    evidence  = list(dict.fromkeys(evidence))

    return {
        "unis": unis, "employers": employers, "exits": exits,
        "raw_score": score, "evidence": evidence,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data source fetchers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_base_projects_coingecko(
    max_pages: int = 4,
    min_mcap: float = 50_000,
    max_mcap: float = 100_000_000,
) -> list:
    """
    Fetch CoinGecko-listed Base ecosystem projects within the mcap range.
    Default: $50K–$100M (early gems with 100x+ potential, not USDC/Chainlink).
    Stops fetching more pages when project mcaps drop below min_mcap.
    """
    projects = []
    categories = ["base-ecosystem", "base-native"]
    for cat in categories:
        for page in range(1, max_pages + 1):
            r = _get("https://api.coingecko.com/api/v3/coins/markets", params={
                "vs_currency": "usd", "category": cat,
                "order": "market_cap_desc", "per_page": 250, "page": page,
                "sparkline": False, "price_change_percentage": "24h",
            })
            if not r:
                break
            data = r.json() if r.ok else []
            if not data:
                break
            for item in data:
                item["_source_category"] = cat
            projects.extend(data)

            # If the smallest mcap on this page is already below min_mcap, stop
            page_mcaps = [p.get("market_cap") or 0 for p in data]
            if page_mcaps and min(page_mcaps) < min_mcap:
                break
            time.sleep(0.6)

    # Deduplicate by CoinGecko id
    seen, deduped = set(), []
    for p in projects:
        if p["id"] not in seen:
            seen.add(p["id"])
            deduped.append(p)

    # Filter to the target mcap window
    filtered = [
        p for p in deduped
        if min_mcap <= (p.get("market_cap") or 0) <= max_mcap
    ]
    return filtered


def fetch_coin_detail_coingecko(coin_id: str) -> dict:
    """Fetch full CoinGecko coin detail — description, links, categories, community."""
    r = _get(f"https://api.coingecko.com/api/v3/coins/{coin_id}", params={
        "localization": False, "tickers": False, "market_data": False,
        "community_data": True, "developer_data": True,
    })
    if not r or not r.ok:
        return {}
    return r.json()


def fetch_github_readme(repo_url: str) -> str:
    """Fetch GitHub repository README for background keyword scanning."""
    m = re.match(r"https://github\.com/([^/]+/[^/]+)", repo_url)
    if not m:
        return ""
    slug = m.group(1).rstrip("/")
    for branch in ("main", "master"):
        r = _get(f"https://raw.githubusercontent.com/{slug}/{branch}/README.md", timeout=7)
        if r and r.ok:
            return r.text[:5000]
    return ""


def fetch_github_repo_info(repo_url: str) -> dict:
    """Fetch GitHub repo metadata (stars, contributors, language)."""
    m = re.match(r"https://github\.com/([^/]+/[^/]+)", repo_url)
    if not m:
        return {}
    slug = m.group(1).rstrip("/")
    r = _get(f"https://api.github.com/repos/{slug}", timeout=7, retries=1)
    if not r or not r.ok:
        return {}
    d = r.json()
    return {
        "stars":       d.get("stargazers_count", 0),
        "forks":       d.get("forks_count", 0),
        "language":    d.get("language"),
        "description": d.get("description", ""),
        "topics":      d.get("topics", []),
        "updated_at":  d.get("updated_at", ""),
    }


def scrape_team_page(base_url: str) -> str:
    """Try common team/about URL patterns and return extracted text."""
    parsed = urlparse(base_url if base_url.startswith("http") else "https://" + base_url)
    root   = f"{parsed.scheme}://{parsed.netloc}"

    candidates = [
        root + "/team",
        root + "/about",
        root + "/founders",
        root + "/about-us",
        root + "/our-team",
        root + "/company",
        root + "/meet-the-team",
        base_url,      # homepage — often has "built by" bios
    ]

    for url in candidates:
        try:
            r = _get(url, timeout=6, browser=True)
            if not r or not r.ok or len(r.text) < 500:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["nav", "footer", "script", "style", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            tl   = text.lower()
            if any(kw in tl for kw in (
                "team", "founder", "built by", "co-founder", "cto", "ceo",
                "previously", "background", "university", "phd", "ex-",
            )):
                return text[:6000]
        except Exception:
            continue
    return ""


def get_twitter_bio(handle: str) -> str:
    """
    Fetch Twitter/X profile bio via no-auth endpoints.
    Returns bio text or empty string.
    """
    if not handle:
        return ""
    handle = handle.lstrip("@").split("/")[-1]

    # Method 1: Twitter syndication widget endpoint
    r = _get(
        "https://cdn.syndication.twimg.com/widgets/followbutton/info.json",
        params={"usernames": handle},
        timeout=7,
    )
    if r and r.ok:
        try:
            data = r.json()
            if data and isinstance(data, list):
                return data[0].get("description", "")
        except Exception:
            pass

    # Method 2: Nitter instances
    for host in ("nitter.net", "nitter.cz", "nitter.1d4.us"):
        try:
            r2 = _get(f"https://{host}/{handle}", timeout=5, browser=True)
            if r2 and r2.ok:
                soup    = BeautifulSoup(r2.text, "html.parser")
                bio_div = soup.find("div", class_="profile-bio")
                if bio_div:
                    return bio_div.get_text(strip=True)
        except Exception:
            continue

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Per-project researcher  (two-pass: fast CG first, deep scrape only if signal)
# ─────────────────────────────────────────────────────────────────────────────

def research_project(
    cg_list_item: dict,
    deep: bool = True,
    force_refresh: bool = False,
    cache=None,
) -> dict:
    """
    Research a single project.

    deep=False (Pass 1):  CoinGecko description + VC categories only.  Fast.
    deep=True  (Pass 2):  Also scrapes website team pages, GitHub README, Twitter.
                          Only called for projects that already have score > 0.

    Returns a unified record with all collected signals + elite score.
    Meme tokens are detected early and returned with is_meme=True, score=0.
    """
    coin_id = cg_list_item.get("id", "")
    name    = cg_list_item.get("name", "?")
    symbol  = cg_list_item.get("symbol", "?").upper()
    mcap    = cg_list_item.get("market_cap") or 0
    price   = cg_list_item.get("current_price") or 0
    h24_chg = cg_list_item.get("price_change_percentage_24h") or 0

    # ── Check cache first ─────────────────────────────────────────────────
    cache_key = coin_id + ("_deep" if deep else "_fast")
    if cache is not None:
        cached = _cache_get(cache, cache_key, force_refresh)
        if cached:
            return cached

    record = {
        "coin_id":      coin_id,
        "name":         name,
        "symbol":       symbol,
        "mcap":         mcap,
        "price":        price,
        "h24_chg":      h24_chg,
        "cg_url":       f"https://www.coingecko.com/en/coins/{coin_id}",
        "elite_score":  0,
        "tier":         "",
        "is_meme":      False,
        "unis":         [],
        "employers":    [],
        "vc_backing":   [],
        "exits":        [],
        "evidence":     [],
        "github_stars": 0,
        "twitter":      "",
        "website":      "",
        "github_repos": [],
        "description":  "",
        "error":        None,
    }

    # ── 1. CoinGecko detailed data ─────────────────────────────────────────
    # Free tier: 30 req/min hard limit — 2.2s between calls keeps us at 27/min.
    # Do NOT sleep before a cache hit (speed); sleep only before live API calls.
    time.sleep(2.2)
    detail = fetch_coin_detail_coingecko(coin_id)
    if not detail:
        # Don't cache failed fetches — may be transient rate limit; retry next run
        record["error"] = "CG detail fetch failed"
        return record

    desc       = (detail.get("description") or {}).get("en") or ""
    categories = [c.lower() for c in (detail.get("categories") or [])]
    links      = detail.get("links") or {}
    comm       = detail.get("community_data") or {}
    dev        = detail.get("developer_data") or {}
    websites   = [w for w in (links.get("homepage") or []) if w]
    tw_handle  = links.get("twitter_screen_name") or ""
    gh_repos   = (links.get("repos_url") or {}).get("github") or []
    whitepaper = (links.get("whitepaper") or "")

    record["description"]  = desc[:400]
    record["twitter"]      = tw_handle
    record["website"]      = websites[0] if websites else ""
    record["github_repos"] = [g for g in gh_repos if g]
    record["tw_followers"] = comm.get("twitter_followers", 0)
    record["tg_members"]   = comm.get("telegram_channel_user_count", 0)
    record["gh_stars"]     = dev.get("stars", 0)
    record["gh_commits"]   = dev.get("commit_count_4_weeks", 0)

    # ── 2. Meme token gate ─────────────────────────────────────────────────
    if _is_meme_token(name, symbol, categories, desc):
        record["is_meme"] = True
        record["tier"]    = "🚫 MEME TOKEN — SKIPPED"
        if cache is not None:
            _cache_put(cache, cache_key, record)
        return record

    # ── 3. CoinGecko VC portfolio categories ──────────────────────────────
    for cat in categories:
        if cat in VC_CG_CATEGORIES:
            label, pts = VC_CG_CATEGORIES[cat]
            record["vc_backing"].append(label)
            record["elite_score"] += pts
            record["evidence"].append(f"[VC-CG] {label} (CoinGecko category)")

    # ── 4. Scan CoinGecko description for elite signals ───────────────────
    cg_scan = _scan_text(desc)
    record["unis"]      += cg_scan["unis"]
    record["employers"] += cg_scan["employers"]
    record["exits"]     += cg_scan["exits"]
    record["evidence"]  += cg_scan["evidence"]
    record["elite_score"] += cg_scan["raw_score"]

    # ── Fast-pass only?  Stop here and cache the fast record ──────────────
    if not deep:
        _finalize(record)
        if cache is not None:
            _cache_put(cache, cache_key, record)
        return record

    # ── 5. Website team page scraping (deep pass only) ────────────────────
    if websites:
        team_text = scrape_team_page(websites[0])
        if team_text:
            ws_scan = _scan_text(team_text)
            record["unis"]      += ws_scan["unis"]
            record["employers"] += ws_scan["employers"]
            record["exits"]     += ws_scan["exits"]
            record["evidence"]  += ws_scan["evidence"]
            record["elite_score"] += ws_scan["raw_score"]

    # ── 6. GitHub README analysis (deep pass only) ────────────────────────
    total_gh_stars = 0
    for gh_url in record["github_repos"][:2]:
        time.sleep(0.3)
        gh_info = fetch_github_repo_info(gh_url)
        total_gh_stars += gh_info.get("stars", 0)
        readme = fetch_github_readme(gh_url)
        if readme:
            gh_scan = _scan_text(readme)
            record["unis"]      += gh_scan["unis"]
            record["employers"] += gh_scan["employers"]
            record["evidence"]  += gh_scan["evidence"]
            record["elite_score"] += gh_scan["raw_score"]
        stars = gh_info.get("stars", 0)
        if stars >= 1000:
            record["elite_score"] += 5
            record["evidence"].append(f"[GH] {stars:,} GitHub stars")
        elif stars >= 200:
            record["elite_score"] += 3
            record["evidence"].append(f"[GH] {stars:,} GitHub stars")
        elif stars >= 50:
            record["elite_score"] += 1
    record["github_stars"] = total_gh_stars

    # ── 7. Twitter bio scraping (deep pass only) ──────────────────────────
    if tw_handle:
        time.sleep(0.3)
        bio = get_twitter_bio(tw_handle)
        if bio:
            tw_scan = _scan_text(bio)
            record["unis"]      += tw_scan["unis"]
            record["employers"] += tw_scan["employers"]
            record["evidence"]  += tw_scan["evidence"]
            record["elite_score"] += tw_scan["raw_score"]

    # ── 8. Whitepaper scan ────────────────────────────────────────────────
    if whitepaper and not whitepaper.endswith(".pdf"):
        try:
            r = _get(whitepaper, timeout=6, browser=True)
            if r and r.ok and "html" in r.headers.get("content-type", ""):
                soup    = BeautifulSoup(r.text, "html.parser")
                wp_text = soup.get_text(separator=" ", strip=True)[:4000]
                wp_scan = _scan_text(wp_text)
                record["unis"]      += wp_scan["unis"]
                record["employers"] += wp_scan["employers"]
                record["evidence"]  += wp_scan["evidence"]
                record["elite_score"] += wp_scan["raw_score"]
        except Exception:
            pass

    # ── 9. Community strength bonus ───────────────────────────────────────
    tw_f = record.get("tw_followers", 0)
    if tw_f >= 100_000:
        record["elite_score"] += 4
    elif tw_f >= 20_000:
        record["elite_score"] += 2

    gh_c = record.get("gh_commits", 0)
    if gh_c >= 100:
        record["elite_score"] += 3
        record["evidence"].append(f"[GH] {gh_c} commits last 4 weeks (active dev)")
    elif gh_c >= 30:
        record["elite_score"] += 1

    _finalize(record)
    if cache is not None:
        _cache_put(cache, cache_key, record)
    return record


def _finalize(record: dict):
    """Deduplicate lists and assign tier label."""
    record["unis"]       = list(dict.fromkeys(record["unis"]))
    record["employers"]  = list(dict.fromkeys(record["employers"]))
    record["vc_backing"] = list(dict.fromkeys(record["vc_backing"]))
    record["exits"]      = list(dict.fromkeys(record["exits"]))
    record["evidence"]   = list(dict.fromkeys(record["evidence"]))[:12]

    s = record["elite_score"]
    record["tier"] = (
        "🏛️ TIER 1 — EXCEPTIONAL" if s >= 25 else
        "🥇 TIER 2 — ELITE"        if s >= 15 else
        "🥈 TIER 3 — STRONG"       if s >= 8  else
        "🥉 TIER 4 — NOTABLE"      if s >= 3  else
        "— STANDARD"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Discovery from existing bot state (cross-reference)
# ─────────────────────────────────────────────────────────────────────────────

def load_bot_state_candidates(state_path: str = "state.json") -> list:
    """
    Pull high-conviction tokens from the bot's state.json.
    These are the tokens the bot is already watching — we enrich them.
    """
    if not os.path.exists(state_path):
        return []
    with open(state_path) as f:
        state = json.load(f)
    tokens = state.get("tokens", {})
    out = []
    for addr, ts in tokens.items():
        snap = ts.get("_pulse_snapshot") or {}
        ch   = ts.get("conviction_history") or []
        conv = ch[-1] if ch else 0
        if conv < 4.0:
            continue
        out.append({
            "address":    addr,
            "name":       snap.get("name", ""),
            "symbol":     snap.get("symbol", ""),
            "mcap":       snap.get("mcap", 0),
            "conviction": conv,
            "pair_url":   snap.get("pair_url", ""),
        })
    return sorted(out, key=lambda x: x["conviction"], reverse=True)


def enrich_bot_candidate(candidate: dict, cache=None) -> dict:
    """
    For a token from bot state, try to find CoinGecko data and research.
    Falls back to DexScreener website/Twitter if not on CoinGecko.
    """
    addr = candidate["address"]
    name = candidate.get("name", addr[:12])

    # Try CoinGecko by contract address
    time.sleep(0.5)
    r = _get(f"https://api.coingecko.com/api/v3/coins/base/contract/{addr}")
    if r and r.ok:
        data    = r.json()
        coin_id = data.get("id")
        if coin_id:
            list_item = {
                "id":     coin_id,
                "name":   data.get("name") or name,
                "symbol": data.get("symbol") or candidate.get("symbol", ""),
                "market_cap":                  candidate["mcap"],
                "current_price":               0,
                "price_change_percentage_24h": 0,
            }
            result = research_project(list_item, deep=True, cache=cache)
            result["bot_conviction"] = candidate["conviction"]
            result["pair_url"]       = candidate["pair_url"]
            result["address"]        = addr
            result["source"]         = "bot_state+cg"
            return result

    # Not on CoinGecko — lightweight DexScreener-only enrichment
    r2 = _get(f"https://api.dexscreener.com/latest/dex/tokens/{addr}")
    if not r2 or not r2.ok:
        return {**candidate, "elite_score": 0, "tier": "— STANDARD",
                "evidence": ["Not on CoinGecko, DexScreener lookup failed"]}

    pairs = r2.json().get("pairs") or []
    if not pairs:
        return {**candidate, "elite_score": 0, "tier": "— STANDARD", "evidence": []}
    best = max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))

    info    = best.get("info") or {}
    webs    = [w.get("url", "") for w in (info.get("websites") or []) if w.get("url")]
    socials = {(s.get("type", "").lower()): s.get("url", "")
               for s in (info.get("socials") or [])}
    tw_url    = socials.get("twitter", "")
    tw_handle = tw_url.rstrip("/").split("/")[-1] if tw_url else ""

    score, evidence, unis, employers = 0, [], [], []

    if webs:
        team_text = scrape_team_page(webs[0])
        if team_text:
            scan      = _scan_text(team_text)
            score    += scan["raw_score"]
            evidence += scan["evidence"]
            unis     += scan["unis"]
            employers+= scan["employers"]

    if tw_handle:
        bio = get_twitter_bio(tw_handle)
        if bio:
            scan      = _scan_text(bio)
            score    += scan["raw_score"]
            evidence += scan["evidence"]
            unis     += scan["unis"]
            employers+= scan["employers"]

    tier = (
        "🏛️ TIER 1 — EXCEPTIONAL" if score >= 25 else
        "🥇 TIER 2 — ELITE"        if score >= 15 else
        "🥈 TIER 3 — STRONG"       if score >= 8  else
        "🥉 TIER 4 — NOTABLE"      if score >= 3  else
        "— STANDARD"
    )
    return {
        **candidate,
        "elite_score": score, "tier": tier,
        "unis":        list(dict.fromkeys(unis)),
        "employers":   list(dict.fromkeys(employers)),
        "vc_backing":  [], "evidence": list(dict.fromkeys(evidence))[:10],
        "website":     webs[0] if webs else "",
        "twitter":     tw_handle,
        "source":      "bot_state+dexscreener",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def _mcap_fmt(v: float) -> str:
    if not v:
        return "n/a"
    if v >= 1_000_000_000:
        return f"${v/1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v/1_000:.0f}K"
    return f"${v:.0f}"


def print_report(results: list, top_n: int = 30):
    """Print ranked terminal report."""
    # Exclude meme tokens from the ranked display
    non_meme = [r for r in results if not r.get("is_meme")]
    scored   = [r for r in non_meme if r.get("elite_score", 0) > 0]
    scored.sort(key=lambda x: x["elite_score"], reverse=True)

    meme_count = sum(1 for r in results if r.get("is_meme"))

    print()
    print("═" * 80)
    print("  BASE CHAIN INTELLIGENCE — FOUNDER BACKGROUND SCAN")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
          f"{len(results)} total  |  {meme_count} meme-filtered  |  "
          f"{len(scored)} with detected signals")
    print("═" * 80)

    if not scored:
        print("\n  No elite background signals found in this scan run.\n")
        print("  Tip: try --min-mcap 10000 --max-mcap 500000000 to widen the search.\n")
        return

    for i, r in enumerate(scored[:top_n], 1):
        tier     = r.get("tier", "— STANDARD")
        name     = r.get("name", "?")
        symbol   = r.get("symbol", "?")
        mcap     = _mcap_fmt(r.get("mcap", 0))
        score    = r.get("elite_score", 0)
        unis     = r.get("unis", [])
        emps     = r.get("employers", [])
        vc       = r.get("vc_backing", [])
        ev       = r.get("evidence", [])
        h24      = r.get("h24_chg", 0)
        gh_s     = r.get("github_stars") or r.get("gh_stars") or 0
        conv     = r.get("bot_conviction") or r.get("conviction") or 0
        tw       = r.get("twitter", "")
        web      = r.get("website", "")
        pair_url = r.get("pair_url") or r.get("cg_url", "")

        print(f"\n{'─'*80}")
        print(f"  #{i:>2}  {tier}")
        print(f"       {name} ({symbol})   MCap: {mcap}   "
              f"24h: {'+'if h24>=0 else ''}{h24:.1f}%   "
              f"Elite Score: {score}")
        if conv:
            print(f"       Bot conviction: {conv:.1f}/10")
        if unis:
            print(f"       🎓 Universities:  {', '.join(unis)}")
        if emps:
            print(f"       💼 Employers:     {', '.join(emps[:5])}")
        if vc:
            print(f"       🏦 VC Backing:    {', '.join(vc)}")
        if gh_s > 0:
            print(f"       ⭐ GitHub Stars:  {gh_s:,}")
        if tw:
            print(f"       𝕏  Twitter:      @{tw}")
        if web:
            print(f"       🌐 Website:      {web}")
        if pair_url:
            print(f"       🔗 {pair_url}")
        if ev:
            print(f"       📎 Evidence ({len(ev)} signals):")
            for e in ev[:5]:
                print(f"          {e[:120]}")

    print(f"\n{'═'*80}")
    print(f"  Showing top {min(top_n, len(scored))} of {len(scored)} "
          f"projects with elite signals")
    print(f"  ({meme_count} meme/joke tokens filtered out)\n")


def save_report(results: list, path: str = "base_intelligence_report.json"):
    with open(path, "w") as f:
        json.dump({
            "generated_at":  datetime.now().isoformat(),
            "total_scanned": len(results),
            "meme_filtered": sum(1 for r in results if r.get("is_meme")),
            "results": sorted(
                results,
                key=lambda x: x.get("elite_score", 0),
                reverse=True,
            ),
        }, f, indent=2)
    print(f"  Full report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Unbuffered output so progress lines appear immediately even when piped
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass   # Python < 3.7

    parser = argparse.ArgumentParser(description="Base Chain Intelligence Scanner")
    parser.add_argument("--quick",    action="store_true",
                        help="Pass-1 only: CoinGecko description scan, no website/GitHub scraping")
    parser.add_argument("--token",    type=str, default="",
                        help="Deep-dive a single token by address (0x…)")
    parser.add_argument("--refresh",  action="store_true",
                        help="Force re-fetch all data (ignore 48h cache)")
    parser.add_argument("--output",   type=str, default="base_intelligence_report.json",
                        help="Output JSON path (default: base_intelligence_report.json)")
    parser.add_argument("--top",      type=int, default=30,
                        help="How many projects to show in terminal (default: 30)")
    parser.add_argument("--min-mcap", type=float, default=50_000,
                        help="Min market cap USD (default: 50000 = $50K)")
    parser.add_argument("--max-mcap", type=float, default=100_000_000,
                        help="Max market cap USD (default: 100000000 = $100M)")
    parser.add_argument("--pages",    type=int, default=4,
                        help="Max CoinGecko pages to fetch per category (default: 4)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cache = _load_cache()

    # ── Single token deep-dive ────────────────────────────────────────────
    if args.token:
        print(f"\n  🔬 Deep-diving token: {args.token}\n")
        cand = enrich_bot_candidate({
            "address":    args.token,
            "name":       args.token[:10],
            "symbol":     "?",
            "mcap":       0,
            "conviction": 0,
            "pair_url":   f"https://dexscreener.com/base/{args.token}",
        }, cache=cache)
        results = [cand]
        print_report(results, top_n=1)
        save_report(results, args.output)
        return

    results = []

    # ── Phase 1: Fetch CoinGecko-listed Base projects in target mcap range ─
    min_m = args.min_mcap
    max_m = args.max_mcap
    print(f"\n  Phase 1 — Fetching Base ecosystem projects "
          f"({_mcap_fmt(min_m)}–{_mcap_fmt(max_m)})…")
    cg_projects = fetch_base_projects_coingecko(
        max_pages=args.pages,
        min_mcap=min_m,
        max_mcap=max_m,
    )
    print(f"  Found {len(cg_projects)} projects in target range  "
          f"(cache has {len(cache)} entries)")

    # ── Pass 1: Quick CG-only scan of ALL projects ────────────────────────
    print(f"\n  Pass 1 — Quick CoinGecko description scan…")
    pass1_results = []
    for i, proj in enumerate(cg_projects, 1):
        name   = proj.get("name", "?")
        symbol = proj.get("symbol", "?").upper()
        mcap   = _mcap_fmt(proj.get("market_cap") or 0)
        print(f"  [{i:>3}/{len(cg_projects)}] {name} ({symbol}) {mcap}…",
              end=" ", flush=True)
        try:
            rec  = research_project(proj, deep=False,
                                    force_refresh=args.refresh, cache=cache)
            pass1_results.append(rec)
            if rec.get("is_meme"):
                print("🚫 meme")
            else:
                s    = rec.get("elite_score", 0)
                flag = "⭐" if s >= 8 else ("✓" if s > 0 else "·")
                print(f"{flag} score={s}")
        except Exception as e:
            print(f"err: {e}")
            pass1_results.append({
                "coin_id": proj.get("id"), "name": name, "symbol": symbol,
                "mcap": proj.get("market_cap") or 0,
                "elite_score": 0, "tier": "— STANDARD",
                "is_meme": False, "error": str(e),
            })

    # Candidates for deep dive: non-meme, score > 0 in Pass 1
    deep_candidates = [
        r for r in pass1_results
        if not r.get("is_meme") and r.get("elite_score", 0) > 0
    ]

    # ── Pass 2: Deep-scrape only the promising candidates ─────────────────
    if not args.quick and deep_candidates:
        print(f"\n  Pass 2 — Deep scrape: {len(deep_candidates)} promising candidates…")
        deep_ids = {r["coin_id"] for r in deep_candidates}
        deep_map = {p["id"]: p for p in cg_projects if p["id"] in deep_ids}

        for i, rec in enumerate(deep_candidates, 1):
            cid  = rec["coin_id"]
            proj = deep_map.get(cid)
            if not proj:
                results.append(rec)
                continue
            print(f"  [{i:>2}/{len(deep_candidates)}] Deep-diving: "
                  f"{rec['name']} ({rec['symbol']})…", end=" ", flush=True)
            try:
                deep_rec = research_project(
                    proj, deep=True,
                    force_refresh=args.refresh, cache=cache,
                )
                results.append(deep_rec)
                s    = deep_rec.get("elite_score", 0)
                flag = "⭐" if s >= 15 else ("✓" if s >= 8 else "·")
                print(f"{flag} score={s}")
            except Exception as e:
                print(f"err: {e}")
                results.append(rec)   # fall back to Pass-1 result

        # Add non-deep projects (scored 0 or meme) to results for report
        deep_result_ids = {r["coin_id"] for r in results}
        for rec in pass1_results:
            if rec["coin_id"] not in deep_result_ids:
                results.append(rec)
    else:
        results = pass1_results

    # ── Phase 2: Bot state high-conviction tokens (not already on CG) ────
    if not args.quick:
        print("\n  Phase 2 — Enriching bot's high-conviction tokens…")
        cg_names = {r.get("name", "").lower() for r in results}
        bot_candidates  = load_bot_state_candidates()
        bot_to_research = [
            c for c in bot_candidates
            if c.get("name", "").lower() not in cg_names
        ][:20]
        print(f"  {len(bot_to_research)} bot-state tokens not yet on CoinGecko")

        for i, cand in enumerate(bot_to_research, 1):
            name = cand.get("name") or cand["address"][:12]
            conv = cand.get("conviction", 0)
            print(f"  [{i:>2}/{len(bot_to_research)}] {name} (conv={conv:.1f})…",
                  end=" ", flush=True)
            try:
                record = enrich_bot_candidate(cand, cache=cache)
                results.append(record)
                score = record.get("elite_score", 0)
                flag  = "⭐" if score >= 8 else ("✓" if score > 0 else "·")
                print(f"{flag} score={score}")
            except Exception as e:
                print(f"err: {e}")

    # ── Output ────────────────────────────────────────────────────────────
    print_report(results, top_n=args.top)
    save_report(results, args.output)


if __name__ == "__main__":
    main()
