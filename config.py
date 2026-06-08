"""Configuration loading for the Base ATH breakout bot."""
import copy
import os

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
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
    "discovery_search_terms": ["base", "ai", "agent", "based", "dog", "pepe", "WETH"],
    "timeframes": ["h1", "h6", "h24"],
    "screener_addresses_file": "discovered_addresses.json",
    "gecko_highs_file": "gecko_highs.json",
    "max_alerts_per_cycle": 20,
    "telegram_min_conviction": {
        "resistance_breakout":              7,
        "pre_breakout_coil":               7,
        "momentum_continuation":           6,
        "volume_surge":                    7,
        "volume_spike":                    7,
        "sleeping_giant_wakeup":           7,
        "slow_build_accumulation":         7,
        "dead_token_resurrection":         6,
        "mcap_milestone":                  6,
        "vc_backed_project":               5,
        "default":                         7,
        "midcap_resistance_breakout":      7,
        "midcap_pre_breakout_coil":        7,
        "midcap_volume_surge":             8,
        "midcap_volume_spike":             8,
        "midcap_momentum_continuation":    6,
        "midcap_sleeping_giant_wakeup":    7,
        "midcap_default":                  7,
    },
    "quality_filter_largecap": {
        "min_pair_age_days":    60,
        "min_h24_txns":         400,
        "max_fdv_mcap_ratio":   2.0,
        "min_liq_mcap_pct":     10,
        "filter_meme_names":    True,
    },
    "quality_filter_midcap": {
        "min_pair_age_days":    30,
        "min_h24_txns":         200,
        "max_fdv_mcap_ratio":   3.0,
        "min_liq_mcap_pct":     20,
        "filter_meme_names":    True,
    },
    "quality_filter": {
        "min_pair_age_days":    14,
        "min_h24_txns":         75,
        "max_fdv_mcap_ratio":   5.0,
        "filter_meme_names":    True,
        "min_liq_mcap_pct":     10,
    },
    "detection": {
        "low_volume_max":               50_000,
        "consolidation_volume_max":     500_000,
        "spike_volume_min":             5_000,
        "min_market_cap_usd":           5_000,
        "max_market_cap_usd":           100_000,
        "max_midcap_usd":               1_000_000,
        "midcap_vol_multiplier":        4.0,
        "min_liquidity_usd":            5_000,
        "baseline_lookback_seconds":    172_800,
        "alert_cooldown_seconds":       86_400,
        "spike_volume_multiplier":      8,
        "min_surge_m5_vol":             500,
        "min_surge_h1_vol":             2_000,
        "min_buy_ratio":                0.35,
        "min_txns_h1":                  5,
        "min_price_change_m5":          -10.0,
        "coiling_lower_pct":            15,
        "coiling_min_h1_vol":           1_000,
        "coiling_buy_ratio_min":        0.50,
        "coiling_cooldown_seconds":     43_200,
        "resistance_breakout_pct":      5,
        "breakout_cooldown_seconds":    21_600,
        "breakout_min_h1_vol":          500,
        "min_consolidation_age_seconds": 3_600,
        "continuation_pct":             50,
        "continuation_cooldown_seconds": 14_400,
        "sleeping_giant_days":          30,
        "sleeping_giant_cooldown_seconds": 172_800,
        "sleeping_giant_surge_mult":    5,
        "slow_build_cooldown_seconds":  604_800,
        "slow_build_min_conviction":    6.0,
        "resurrection_cooldown_seconds": 604_800,
        "resurrection_min_fall_mult":   5.0,
        "resurrection_surge_mult":      4,
        "milestone_min_conviction":     5.0,
        "max_largecap_usd":             10_000_000,
        "largecap_vol_multiplier":      10.0,
        "cg_checks_per_cycle":          3,
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
