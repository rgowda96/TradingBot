# TradingBot — Base Volume-Spike Alerter

Watches tokens trading on the **Base** chain via the DexScreener API and sends a
**Telegram alert** when a token jumps from a quiet 1-hour volume (under $1,000)
straight to **six figures or more ($100,000+)**, while still holding enough
liquidity to trade against.

It is an *alerting* bot — it does not place trades.

## How it works

1. **Discovery** — every few minutes it pulls Base-chain tokens from
   DexScreener's profile, boost, and search feeds, accumulating a watchlist.
2. **Polling** — it re-checks every watched token's 1-hour volume and liquidity.
3. **Spike detection** — for each pair it records a "quiet baseline" whenever 1h
   volume is under $1,000. When that same pair's 1h volume later clears
   $100,000 (with liquidity above the configured floor), it fires one alert.
4. **Alerting** — the alert is sent to your Telegram chat (and logged).

State is stored in `state.json` so baselines and alert cooldowns survive
restarts.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in your Telegram bot token + chat id
```

Get a bot token from [@BotFather](https://t.me/BotFather). Send your new bot a
message, then read your chat id from
`https://api.telegram.org/bot<TOKEN>/getUpdates`.

## Run

```bash
python bot.py            # run continuously
python bot.py --once     # run a single cycle (handy for testing)
```

## Configuration

Thresholds and timings live in `config.yaml`. Key `detection` settings:

| Setting | Default | Meaning |
|---|---|---|
| `low_volume_max` | `1000` | the "quiet" 1h volume ceiling |
| `spike_volume_min` | `100000` | 1h volume that counts as a six-figure spike |
| `min_liquidity_usd` | `20000` | minimum pooled liquidity to alert |
| `lookback_seconds` | `3600` | how recently the quiet baseline must have been seen |
| `alert_cooldown_seconds` | `21600` | minimum gap between alerts for the same pair |

## Note on coverage

DexScreener's public API has no "list every token on a chain" endpoint, so the
watchlist is built up over time from the trending/profile/boost/search feeds.
The longer the bot runs, the more of Base it covers. Add specific tokens of
interest as `discovery_search_terms` in `config.yaml` to seed them faster.
