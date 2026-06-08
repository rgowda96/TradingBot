#!/usr/bin/env python3
"""
GeckoTerminal 30-day price high bootstrap.

Reads all monitored token addresses from state.json and discovered_addresses.json,
fetches 30-day OHLCV candles for each, and writes the 30-day price high into
gecko_highs.json.  bot.py reads that file every cycle to improve breakout detection.

Run once before starting the bot, or re-run periodically (e.g. weekly) to refresh.

Usage:
    python3 bootstrap_highs.py
    python3 bootstrap_highs.py --limit 500   # only bootstrap up to 500 tokens
    python3 bootstrap_highs.py --resume      # skip tokens already in gecko_highs.json
"""

import argparse
import json
import os
import sys
import time

import requests

GT      = "https://api.geckoterminal.com/api/v2"
HEADERS = {"Accept": "application/json", "User-Agent": "ath-bootstrap/1.0"}
CHAIN   = "base"

OUTPUT_FILE      = "gecko_highs.json"
STATE_FILE       = "state.json"
ADDRESSES_FILE   = "discovered_addresses.json"
PAUSE_BETWEEN    = 1.2   # seconds between API calls (free tier: ~50 req/min)
CANDLES_LIMIT    = 30    # 30 daily candles = 30-day history


def _gt_get(path, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(f"{GT}{path}", params=params, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 0)) or 15 * (attempt + 1)
                print(f"  [rate limit] sleeping {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                return {}
            time.sleep(3 * (attempt + 1))
    return {}


def get_pools_for_token(token_addr):
    """Return list of pool addresses for a token (highest liquidity first)."""
    d = _gt_get(f"/networks/{CHAIN}/tokens/{token_addr}/pools", {"page": 1})
    pools = d.get("data", [])
    # Sort by reserve_in_usd descending so we pick the most liquid pool
    try:
        pools.sort(
            key=lambda p: float(p.get("attributes", {}).get("reserve_in_usd") or 0),
            reverse=True,
        )
    except Exception:
        pass
    return [p["attributes"]["address"] for p in pools if p.get("attributes", {}).get("address")]


def get_30d_high(pool_addr):
    """Fetch 30-day daily candles and return the highest high price."""
    d = _gt_get(
        f"/networks/{CHAIN}/pools/{pool_addr}/ohlcv/day",
        {"limit": CANDLES_LIMIT, "currency": "usd"},
    )
    candles = d.get("data", {}).get("attributes", {}).get("ohlcv_list", []) or []
    if not candles:
        return None
    # candle format: [timestamp, open, high, low, close, volume]
    highs = [c[2] for c in candles if len(c) >= 3 and c[2] and c[2] > 0]
    return max(highs) if highs else None


def load_token_addresses():
    """Collect all unique token addresses from state.json and discovered_addresses.json."""
    addrs = set()

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            for a in state.get("monitored_tokens", []):
                if a:
                    addrs.add(a.lower())
        except Exception as exc:
            print(f"Warning: could not read {STATE_FILE}: {exc}")

    if os.path.exists(ADDRESSES_FILE):
        try:
            with open(ADDRESSES_FILE) as f:
                disc = json.load(f)
            for a in disc.get(CHAIN, []):
                if a:
                    addrs.add(a.lower())
        except Exception as exc:
            print(f"Warning: could not read {ADDRESSES_FILE}: {exc}")

    return sorted(addrs)


def load_existing_highs():
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_highs(highs):
    tmp = OUTPUT_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(highs, f, indent=2)
    os.replace(tmp, OUTPUT_FILE)


def main():
    parser = argparse.ArgumentParser(description="Bootstrap 30-day price highs from GeckoTerminal")
    parser.add_argument("--limit",  type=int, default=0,
                        help="Max tokens to process (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tokens already in gecko_highs.json")
    args = parser.parse_args()

    all_addrs = load_token_addresses()
    existing  = load_existing_highs()

    if args.resume:
        pending = [a for a in all_addrs if a not in existing]
        print(f"Resume mode: {len(all_addrs)} total, {len(existing)} done, "
              f"{len(pending)} remaining")
    else:
        pending = all_addrs

    if args.limit:
        pending = pending[:args.limit]

    print(f"\nBootstrapping 30-day price highs for {len(pending)} tokens on {CHAIN}...")
    print(f"Output: {OUTPUT_FILE}\n")

    highs   = dict(existing)
    success = errors = skipped = 0

    for i, addr in enumerate(pending):
        pct = (i + 1) / len(pending) * 100
        sys.stdout.write(f"\r  [{i+1}/{len(pending)} {pct:.0f}%] {addr[:12]}... "
                         f"ok={success} err={errors} skip={skipped}   ")
        sys.stdout.flush()

        try:
            # Step 1: find the best pool
            pools = get_pools_for_token(addr)
            time.sleep(PAUSE_BETWEEN)

            if not pools:
                skipped += 1
                continue

            # Step 2: get 30d candles from the best (most liquid) pool
            high = get_30d_high(pools[0])
            time.sleep(PAUSE_BETWEEN)

            if high and high > 0:
                highs[addr] = high
                success += 1
            else:
                skipped += 1

            # Save incrementally every 20 tokens so we don't lose progress
            if (i + 1) % 20 == 0:
                save_highs(highs)

        except KeyboardInterrupt:
            print("\n\nInterrupted — saving progress...")
            break
        except Exception as exc:
            errors += 1
            # Don't print error mid-line; just count it

    print(f"\n\nDone.")
    print(f"  Fetched highs  : {success}")
    print(f"  Skipped (no data): {skipped}")
    print(f"  Errors         : {errors}")

    save_highs(highs)
    print(f"\n{OUTPUT_FILE} written — {len(highs)} tokens with 30d price highs.")
    print("Restart bot.py for the new highs to take effect.\n")


if __name__ == "__main__":
    main()
