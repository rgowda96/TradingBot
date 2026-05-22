#!/usr/bin/env python3
"""Base-chain volume-spike alert bot.

Watches tokens trading on the Base chain via DexScreener and fires a Telegram
alert when a token jumps from a quiet 1-hour volume (under $1000) to six
figures or more ($100,000+), while holding sufficient liquidity.
"""
import argparse
import html
import json
import logging
import os
import time

from config import load_config
from dexscreener import DexScreener
from notifier import TelegramNotifier

log = logging.getLogger("bot")


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
    """Update per-pair state and return an alert dict if a spike is detected."""
    det = cfg["detection"]
    volume = (pair.get("volume") or {}).get("h1")
    liquidity = (pair.get("liquidity") or {}).get("usd")
    if volume is None:
        return None

    pair_state["last_volume"] = volume
    pair_state["last_seen_ts"] = now

    # Record the quiet baseline whenever the token is trading under the cap.
    if det["low_volume_min"] <= volume <= det["low_volume_max"]:
        pair_state["last_low_volume"] = volume
        pair_state["last_low_ts"] = now

    # A spike must clear the six-figure volume bar...
    if volume < det["spike_volume_min"]:
        return None
    # ...with enough liquidity to actually trade against...
    if liquidity is None or liquidity < det["min_liquidity_usd"]:
        return None
    # ...and the token must have been quiet recently (a genuine jump).
    last_low_ts = pair_state.get("last_low_ts")
    if not last_low_ts or now - last_low_ts > det["lookback_seconds"]:
        return None
    # ...and we must not have alerted on it too recently.
    last_alert_ts = pair_state.get("last_alert_ts")
    if last_alert_ts is not None and now - last_alert_ts < det["alert_cooldown_seconds"]:
        return None

    pair_state["last_alert_ts"] = now
    return {
        "pair": pair,
        "baseline_volume": pair_state.get("last_low_volume", 0.0),
        "spike_volume": volume,
        "liquidity": liquidity,
    }


def format_alert(alert):
    pair = alert["pair"]
    token = pair.get("baseToken") or {}
    name = html.escape(str(token.get("name", "?")))
    symbol = html.escape(str(token.get("symbol", "?")))
    baseline = alert["baseline_volume"]
    spike = alert["spike_volume"]
    multiple = spike / baseline if baseline > 0 else float("inf")
    multiple_txt = "∞" if multiple == float("inf") else f"{multiple:,.0f}x"
    price = pair.get("priceUsd", "?")
    dex_id = html.escape(str(pair.get("dexId", "?")))
    url = pair.get("url", "")

    return (
        "\U0001F6A8 <b>BASE VOLUME SPIKE</b> \U0001F6A8\n\n"
        f"<b>{name} ({symbol})</b>\n"
        f"Quiet 1h baseline: ${baseline:,.0f}\n"
        f"Now 1h volume: <b>${spike:,.0f}</b> ({multiple_txt})\n"
        f"Liquidity: ${alert['liquidity']:,.0f}\n"
        f"Price: ${price}\n"
        f"DEX: {dex_id}\n"
        f"{url}"
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
            alert = evaluate_pair(pair, cfg, pair_state, now)
            if alert:
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
        "started: spike >= $%s/1h, baseline <= $%s, liquidity >= $%s",
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
