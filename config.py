"""Configuration loading for the Base volume-spike bot."""
import copy
import os

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at runtime
    load_dotenv = None

DEFAULTS = {
    "chain": "base",
    "state_file": "state.json",
    "poll_interval_seconds": 120,
    "discovery_interval_seconds": 300,
    "max_monitored_tokens": 5000,
    "request_batch_size": 30,
    "batch_pause_seconds": 0.5,
    "pair_state_ttl_seconds": 604800,
    "discovery_search_terms": ["base", "WETH", "USDC"],
    "timeframes": ["h1", "h6", "h24"],
    "detection": {
        "low_volume_max": 9999,
        "spike_volume_min": 10000,
        "max_market_cap_usd": 500000,
        "min_liquidity_usd": 20000,
        "baseline_lookback_seconds": 172800,
        "alert_cooldown_seconds": 86400,
    },
}


def _merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path="config.yaml"):
    """Load config.yaml merged over defaults, plus Telegram secrets from env."""
    if load_dotenv:
        load_dotenv()
    cfg = copy.deepcopy(DEFAULTS)
    if os.path.exists(path):
        with open(path) as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
    cfg["telegram_bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN", "")
    cfg["telegram_chat_id"] = os.getenv("TELEGRAM_CHAT_ID", "")
    return cfg
