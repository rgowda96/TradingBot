#!/usr/bin/env python3
"""Base-chain volume-spike alert bot.

Watches tokens trading on the Base chain via DexScreener and fires a Telegram
alert when a token jumps from under 4-figure volume (< $10k) to 5-figure
volume ($10k+) on any of the configured timeframes (h1, h6, h24).
"""
import argparse
import json
import logging
import os
import time

from config import load_config
from dexscreener import DexScreener
from notifier import TelegramNotifier

log = logging.getLogger("bot")

# Seconds per DexScreener timeframe key — used to compute per-TF lookback.
TIMEFRAME_SECONDS = {"m5": 300, "h1": 3600, "h6": 21600, "h24": 86400}


# --------------------------------------------------------------------------
# State persistence
# --------------------------------------------------------------------------
def load_state(path):
    if os.path.exists(path):
        try:
            with open(path) as fh:
                state = json.load(fh)
            state.setdefault("monitored_tokens", [])
            state.setdefault("pairs", {})
            return state
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read state file (%s); starting fresh", exc)
    return {"monitored_tokens": [], "pairs": {}}


def save_state(path, state):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def discover_tokens(dex, cfg):
    """Collect Base-chain token addresses from DexScreener discovery feeds."""
    chain = cfg["chain"]
    found = set()

    for item in dex.token_profiles_latest():
        if item.get("chainId") == chain and item.get("tokenAddress"):
            found.add(item["tokenAddress"].lower())

    for item in dex.token_boosts_latest() + dex.token_boosts_top():
        if item.get("chainId") == chain and item.get("tokenAddress"):
            found.add(item["tokenAddress"].lower())

    for term in cfg["discovery_search_terms"]:
        for pair in dex.search(term):
            if pair.get("chainId") != chain:
                continue
            for side in ("baseToken", "quoteToken"):
                addr = (pair.get(side) or {}).get("address")
                if addr:
                    found.add(addr.lower())

    return found


# --------------------------------------------------------------------------
# Spike detection
# --------------------------------------------------------------------------
def evaluate_pair(pair, cfg, pair_state, now):
    """Update per-pair state and return a list of alert dicts (one per spiking timeframe).

    A token fires when:
      1. Its 1d (h24) volume was under low_volume_max within baseline_lookback_seconds
         — meaning it has been "sleeping" for a sustained period.
      2. Any configured spike timeframe now shows volume >= spike_volume_min.
    """
    det = cfg["detection"]
    pair_state["last_seen_ts"] = now

    # --- Step 1: maintain the quiet 1d baseline ---
    h24_vol = (pair.get("volume") or {}).get("h24")
    if h24_vol is not None and h24_vol <= det["low_volume_max"]:
        pair_state["last_low_h24_ts"] = now
        pair_state["last_low_h24_vol"] = h24_vol

    # Token must have been quiet on 1d recently to qualify for spike alerts.
    last_low_h24_ts = pair_state.get("last_low_h24_ts")
    if not last_low_h24_ts or now - last_low_h24_ts > det["baseline_lookback_seconds"]:
        return []

    # --- Step 2: check market cap and liquidity once ---
    # Use marketCap first, fall back to fdv; allow through only if both are absent.
    market_cap = pair.get("marketCap") or pair.get("fdv")
    if market_cap is not None and market_cap > det["max_market_cap_usd"]:
        return []

    liquidity = (pair.get("liquidity") or {}).get("usd")
    if liquidity is None or liquidity < det["min_liquidity_usd"]:
        return []

    # --- Step 3: check each spike timeframe ---
    alerts = []
    for tf in cfg.get("timeframes", ["h1", "h6", "h24"]):
        volume = (pair.get("volume") or {}).get(tf)
        if volume is None or volume < det["spike_volume_min"]:
            continue

        tf_state = pair_state.setdefault(tf, {})
        last_alert_ts = tf_state.get("last_alert_ts")
        if last_alert_ts is not None and now - last_alert_ts < det["alert_cooldown_seconds"]:
            continue

        tf_state["last_alert_ts"] = now
        alerts.append({
            "timeframe": tf,
            "pair": pair,
            "baseline_h24_vol": pair_state.get("last_low_h24_vol", 0.0),
            "spike_volume": volume,
            "liquidity": liquidity,
            "market_cap": market_cap,
        })

    return alerts


def format_alert(alert):
    """Build a plain-text alert that reads well in a terminal and on Telegram."""
    pair = alert["pair"]
    tf = alert["timeframe"]
    token = pair.get("baseToken") or {}
    name = token.get("name", "?")
    symbol = token.get("symbol", "?")
    baseline = alert["baseline_h24_vol"]
    spike = alert["spike_volume"]
    multiple = spike / baseline if baseline > 0 else None
    multiple_txt = "from near-zero" if multiple is None else f"{multiple:,.0f}x"

    mcap = alert.get("market_cap")
    mcap_txt = f"${mcap:,.0f}" if mcap else "n/a"

    return (
        f"=== BREAKOUT ALERT ({tf.upper()}) ===\n"
        f"{name} ({symbol})\n"
        f"Market cap:       {mcap_txt}\n"
        f"1d baseline vol:  ${baseline:,.0f}  (was sleeping)\n"
        f"{tf} vol now:      ${spike:,.0f}  ({multiple_txt})\n"
        f"Liquidity:        ${alert['liquidity']:,.0f}\n"
        f"Price:            ${pair.get('priceUsd', '?')}\n"
        f"DEX:              {pair.get('dexId', '?')}\n"
        f"{pair.get('url', '')}"
    )


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def prune_pair_state(state, ttl, now):
    stale = [
        addr for addr, ps in state["pairs"].items()
        if now - ps.get("last_seen_ts", 0) > ttl
    ]
    for addr in stale:
        del state["pairs"][addr]


def run_cycle(dex, cfg, notifier, state, now, force_discovery=False):
    monitored = state["monitored_tokens"]

    if force_discovery or now - state.get("last_discovery_ts", 0) >= cfg[
        "discovery_interval_seconds"
    ]:
        before = len(monitored)
        seen = set(monitored)
        for addr in discover_tokens(dex, cfg):
            if addr not in seen:
                monitored.append(addr)
                seen.add(addr)
        cap = cfg["max_monitored_tokens"]
        if len(monitored) > cap:
            del monitored[:len(monitored) - cap]
        state["last_discovery_ts"] = now
        log.info("discovery: %d tokens (+%d)", len(monitored), len(monitored) - before)

    alerts_sent = 0
    for batch in chunks(monitored, cfg["request_batch_size"]):
        for pair in dex.tokens(cfg["chain"], batch):
            if pair.get("chainId") != cfg["chain"]:
                continue
            pair_addr = pair.get("pairAddress")
            if not pair_addr:
                continue
            pair_state = state["pairs"].setdefault(pair_addr, {})
            for alert in evaluate_pair(pair, cfg, pair_state, now):
                message = format_alert(alert)
                log.info("SPIKE DETECTED:\n%s", message)
                notifier.send(message)
                alerts_sent += 1
        time.sleep(cfg["batch_pause_seconds"])

    prune_pair_state(state, cfg["pair_state_ttl_seconds"], now)
    return alerts_sent


def main():
    parser = argparse.ArgumentParser(description="Base-chain volume-spike alert bot")
    parser.add_argument("--config", default="config.yaml", help="path to config file")
    parser.add_argument("--once", action="store_true", help="run a single cycle and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")],
    )

    cfg = load_config(args.config)
    dex = DexScreener()
    notifier = TelegramNotifier(cfg["telegram_bot_token"], cfg["telegram_chat_id"])
    state = load_state(cfg["state_file"])

    log.info(
        "started: timeframes=%s, spike >= $%s, baseline <= $%s, liquidity >= $%s",
        cfg["timeframes"],
        f"{cfg['detection']['spike_volume_min']:,}",
        f"{cfg['detection']['low_volume_max']:,}",
        f"{cfg['detection']['min_liquidity_usd']:,}",
    )

    try:
        while True:
            now = time.time()
            try:
                run_cycle(dex, cfg, notifier, state, now, force_discovery=args.once)
            except Exception:  # keep the loop alive on transient errors
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
