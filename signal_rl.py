#!/usr/bin/env python3
"""LinUCB Contextual Bandit — real RL for signal quality learning.

Algorithm: LinUCB (Chu et al. 2011 — "Contextual Bandits with Linear Payoff Functions")

Why LinUCB and not a simpler win-rate table
==========================================
A win-rate table treats every signal of the same type identically.
LinUCB learns WHICH FEATURES predict outcomes within each signal type.
Example: a sleeping_giant at conviction=8.0, momentum=9.2, age=60d
         is very different from sleeping_giant at conviction=4.5,
         momentum=5.0, age=8d. LinUCB captures that difference.

It also does principled exploration: the UCB bonus (alpha * sqrt(x^T A^-1 x))
is large when a feature region is under-explored, so the model actively
tries signals in uncertain territory rather than blindly suppressing them.

How it fits into the bot
========================
1. bot.py fires a signal → appends to signal_log.jsonl (feature vector included)
2. Daily: python signal_rl.py harvest   → fetches 7d-old price outcomes,
                                           appends to outcome_log.jsonl
3. Daily: python signal_rl.py train     → reads outcome_log, updates LinUCB
                                           model, saves rl_model.json
4. bot.py loads rl_model.json at startup and calls rl_score(features)
   to get a [-1.5, +1.5] additive bonus on top of base conviction:
     effective_conviction = base_conviction + rl_bonus
   Tokens the model has learned to distrust score lower; tokens in
   high-reward regions of feature space score higher.

Arm structure
=============
One LinUCB arm per signal type (8 arms).
Each arm maintains:
  A  — d×d precision matrix (initialized to I, acts as ridge λ=1)
  b  — d-vector of reward-weighted feature sums

Feature vector (d = 11)
========================
  0  conviction        normalized: score / 10
  1  dim_organic       / 10
  2  dim_tokenomics    / 10
  3  dim_maturity      / 10
  4  dim_momentum      / 10
  5  log_mcap          log10(mcap) / 7  (7 = log10(10M), the target ceiling)
  6  buy_ratio         raw [0, 1]
  7  log_liq           log10(liq) / 6   (6 = log10(1M))
  8  age_norm          min(age_days, 180) / 180
  9  vol_surge_ratio   min(surge_ratio, 50) / 50  (0 if not a surge signal)
 10  bias              always 1.0

Reward
======
  r = clip(log(p_7d / p_entry), -1.5, 1.5)
  Positive reward → price went up; negative → price fell.
  Log scale: +0.69 = doubled (+100%), -0.69 = halved (-50%).
  Clipping at ±1.5 prevents one 10× moonshot from dominating training.

Usage
=====
  # Run from the TradingBot directory:
  python signal_rl.py harvest    # fetch outcomes for 7d-old signals
  python signal_rl.py train      # fit the model from outcome_log.jsonl
  python signal_rl.py status     # print current model state + stats
  python signal_rl.py all        # harvest + train + status in one shot (use in cron)

  # From bot.py:
  from signal_rl import load_model, rl_score
  rl = load_model()   # call once at startup
  bonus = rl_score(rl, signal_type, features_dict)  # call per-signal
"""
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIGNAL_LOG    = "signal_log.jsonl"
OUTCOME_LOG   = "outcome_log.jsonl"
RL_MODEL_FILE = "rl_model.json"

FEATURE_DIM   = 11          # must match feature vector construction below
ALPHA         = 1.0         # UCB exploration coefficient
MIN_OBS_TRAIN = 3           # min observations per arm before that arm influences scoring
HARVEST_AFTER_DAYS = 7      # fetch outcome once signal is this old
HARVEST_MAX_DAYS   = 120    # ignore signals older than this
REWARD_CLIP        = 1.5    # clip log returns to [-REWARD_CLIP, +REWARD_CLIP]

SIGNAL_TYPES = [
    "sleeping_giant_wakeup",
    "volume_surge",
    "volume_spike",
    "resistance_breakout",
    "pre_breakout_coil",
    "slow_build_accumulation",
    "momentum_continuation",
    "dead_token_resurrection",
    "mcap_milestone",
    "vc_backed_project",
]

DEXSCREENER_URL = "https://api.dexscreener.com/tokens/v1"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(signal_type, feat):
    """Build the d=11 feature vector from a signal dict or keyword args.

    Args:
        signal_type: string signal type (used for arm selection, not in features)
        feat: dict with keys matching signal_log.jsonl fields:
              conviction, dims (dict), mcap, liq, age_days, buy_ratio,
              h1_vol, h24_vol (last two used to derive vol_surge_ratio)

    Returns:
        np.ndarray of shape (11,), all values in [0, 1] except bias
    """
    dims   = feat.get("dims") or {}
    conv   = float(feat.get("conviction") or 0)
    d_org  = float(dims.get("organic",    0))
    d_tok  = float(dims.get("tokenomics", 0))
    d_mat  = float(dims.get("maturity",   0))
    d_mom  = float(dims.get("momentum",   0))

    mcap   = float(feat.get("mcap") or 1)
    liq    = float(feat.get("liq")  or 1)
    age    = float(feat.get("age_days") or 0)
    buy    = float(feat.get("buy_ratio") or 0.5)

    # Vol surge ratio: h1_vol relative to h24_vol baseline (hourly rate)
    h1_vol  = float(feat.get("h1_vol")  or 0)
    h24_vol = float(feat.get("h24_vol") or 1)
    hourly_baseline = max(h24_vol / 24, 1.0)
    surge_ratio = h1_vol / hourly_baseline

    # Normalize each feature into [0, 1]
    f0  = np.clip(conv / 10.0, 0, 1)
    f1  = np.clip(d_org  / 10.0, 0, 1)
    f2  = np.clip(d_tok  / 10.0, 0, 1)
    f3  = np.clip(d_mat  / 10.0, 0, 1)
    f4  = np.clip(d_mom  / 10.0, 0, 1)
    f5  = np.clip(math.log10(max(mcap, 1)) / 7.0, 0, 1)   # 7 = log10(10M)
    f6  = np.clip(buy, 0, 1)
    f7  = np.clip(math.log10(max(liq, 1)) / 6.0, 0, 1)    # 6 = log10(1M)
    f8  = np.clip(age / 180.0, 0, 1)
    f9  = np.clip(surge_ratio / 50.0, 0, 1)
    f10 = 1.0   # bias

    return np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10], dtype=np.float64)


# ---------------------------------------------------------------------------
# LinUCB arm
# ---------------------------------------------------------------------------

class LinUCBArm:
    """One LinUCB arm.  Maintains sufficient statistics (A, b) for a ridge
    regression model:  expected_reward = theta^T x
    where theta = A^{-1} b (the ridge estimate).

    The UCB score adds an exploration bonus:
        score = theta^T x + alpha * sqrt(x^T A^{-1} x)

    A starts as the d×d identity matrix (equivalent to ridge λ=1 with
    zero prior mean), ensuring the model is invertible from the first
    observation.
    """
    def __init__(self, d=FEATURE_DIM, alpha=ALPHA):
        self.d      = d
        self.alpha  = alpha
        self.A      = np.eye(d, dtype=np.float64)   # precision matrix
        self.b      = np.zeros(d, dtype=np.float64)  # reward-weighted feature sum
        self.n_obs  = 0   # number of observations used to train this arm

    # ------------------------------------------------------------------
    def score(self, x):
        """Return (ucb_score, expected_reward) for context vector x."""
        A_inv    = np.linalg.inv(self.A)
        theta    = A_inv @ self.b
        mu       = float(theta @ x)
        sigma    = float(np.sqrt(max(x @ A_inv @ x, 0.0)))
        ucb      = mu + self.alpha * sigma
        return ucb, mu

    def update(self, x, reward):
        """Online update: incorporate one (context, reward) observation."""
        self.A    += np.outer(x, x)
        self.b    += reward * x
        self.n_obs += 1

    # ------------------------------------------------------------------
    def to_dict(self):
        return {
            "A":     self.A.tolist(),
            "b":     self.b.tolist(),
            "n_obs": self.n_obs,
            "d":     self.d,
            "alpha": self.alpha,
        }

    @classmethod
    def from_dict(cls, d):
        arm       = cls(d=d["d"], alpha=d["alpha"])
        arm.A     = np.array(d["A"], dtype=np.float64)
        arm.b     = np.array(d["b"], dtype=np.float64)
        arm.n_obs = d["n_obs"]
        return arm


# ---------------------------------------------------------------------------
# Model (collection of arms + metadata)
# ---------------------------------------------------------------------------

class LinUCBModel:
    def __init__(self):
        self.arms     = {st: LinUCBArm() for st in SIGNAL_TYPES}
        self.created  = time.time()
        self.trained  = None
        self.n_total  = 0   # total observations ever trained on

    def arm(self, signal_type):
        if signal_type not in self.arms:
            self.arms[signal_type] = LinUCBArm()
        return self.arms[signal_type]

    def save(self, path=RL_MODEL_FILE):
        data = {
            "version":  2,
            "created":  self.created,
            "trained":  self.trained or time.time(),
            "n_total":  self.n_total,
            "arms":     {st: arm.to_dict() for st, arm in self.arms.items()},
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path=RL_MODEL_FILE):
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as fh:
                data = json.load(fh)
            m         = cls()
            m.created = data.get("created", time.time())
            m.trained = data.get("trained")
            m.n_total = data.get("n_total", 0)
            for st, arm_d in data.get("arms", {}).items():
                m.arms[st] = LinUCBArm.from_dict(arm_d)
            return m
        except Exception as exc:
            print(f"[rl] could not load model ({exc}), starting fresh")
            return cls()


# ---------------------------------------------------------------------------
# Public API for bot.py integration
# ---------------------------------------------------------------------------

def load_model(path=RL_MODEL_FILE):
    """Load the trained model at bot startup.  Returns a LinUCBModel."""
    return LinUCBModel.load(path)


def rl_score(model, signal_type, feat):
    """Return an additive conviction bonus ∈ [-1.5, +1.5].

    The bonus is based on the LinUCB expected reward for this signal's
    feature vector.  It's scaled so that a token in the highest-reward
    region adds +1.5 to conviction, and the worst-seen region subtracts -1.5.

    If the arm has fewer than MIN_OBS_TRAIN observations the bonus is 0.0
    (untrained arm — don't influence scoring yet).

    Usage in bot.py:
        from signal_rl import load_model, rl_score
        rl = load_model()
        bonus = rl_score(rl, alert["type"], alert)
        effective_conv = alert["conviction"] + bonus
    """
    arm = model.arm(signal_type)
    if arm.n_obs < MIN_OBS_TRAIN:
        return 0.0

    x          = extract_features(signal_type, feat)
    _, mu      = arm.score(x)
    # Scale: REWARD_CLIP maps to ±1.5 conviction bonus
    bonus      = np.clip(mu / REWARD_CLIP * 1.5, -1.5, 1.5)
    return float(bonus)


# ---------------------------------------------------------------------------
# Outcome harvesting
# ---------------------------------------------------------------------------

def _fetch_price(token_addr, chain="base"):
    """Fetch current price from DexScreener.  Returns float or None."""
    try:
        url  = f"{DEXSCREENER_URL}/{chain}/{token_addr}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        pairs = (resp.json().get("pairs") or [])
        if not pairs:
            return None
        pairs.sort(
            key=lambda p: float((p.get("liquidity") or {}).get("usd", 0) or 0),
            reverse=True,
        )
        v = pairs[0].get("priceUsd")
        return float(v) if v else None
    except Exception:
        return None


def _already_harvested(sid, outcome_log_path):
    """Check if this signal_id is already in outcome_log.jsonl."""
    if not os.path.exists(outcome_log_path):
        return False
    with open(outcome_log_path) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
                if rec.get("signal_id") == sid:
                    return True
            except Exception:
                pass
    return False


def _make_signal_id(sig):
    return f"{sig.get('token','?')}_{sig.get('signal_type','?')}_{int(sig.get('ts', 0))}"


def harvest(signal_log=SIGNAL_LOG, outcome_log=OUTCOME_LOG, verbose=True):
    """Fetch price outcomes for all signals >= HARVEST_AFTER_DAYS old.

    For each harvestable signal:
      - fetches current price from DexScreener
      - computes reward = clip(log(p_now / p_entry), -REWARD_CLIP, +REWARD_CLIP)
      - appends one JSON line to outcome_log.jsonl

    Returns the number of new outcomes harvested.
    """
    now = time.time()
    if not os.path.exists(signal_log):
        if verbose:
            print(f"[harvest] {signal_log} not found — run the bot first")
        return 0

    signals = []
    with open(signal_log) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                signals.append(json.loads(line))
            except Exception:
                pass

    if verbose:
        print(f"[harvest] {len(signals)} total signals in log")

    harvestable = [
        s for s in signals
        if HARVEST_AFTER_DAYS * 86400 <= now - s.get("ts", 0) <= HARVEST_MAX_DAYS * 86400
    ]
    if verbose:
        print(f"[harvest] {len(harvestable)} signals in harvest window ({HARVEST_AFTER_DAYS}–{HARVEST_MAX_DAYS} days old)")

    new = 0
    for i, sig in enumerate(harvestable):
        sid = _make_signal_id(sig)
        if _already_harvested(sid, outcome_log):
            continue

        token     = sig.get("token", "")
        chain     = sig.get("chain", "base")
        entry_px  = sig.get("price")

        if not token or not entry_px or entry_px <= 0:
            continue

        if verbose:
            name = sig.get("name", "?")
            stype = sig.get("signal_type", "?")
            print(f"  [{i+1}/{len(harvestable)}] {name} ({stype}) — fetching price ...", end="", flush=True)

        current_px = _fetch_price(token, chain)

        if current_px is None:
            # Token delisted / rugged — treat as max loss
            reward  = -REWARD_CLIP
            outcome = "delisted"
            if verbose:
                print(" DELISTED → reward=-1.5")
        else:
            log_ret = math.log(current_px / entry_px)
            reward  = float(np.clip(log_ret, -REWARD_CLIP, REWARD_CLIP))
            pct_chg = (current_px - entry_px) / entry_px * 100
            outcome = "win" if pct_chg > 20 else ("loss" if pct_chg < -20 else "neutral")
            if verbose:
                print(f" {pct_chg:+.0f}% → reward={reward:+.3f} ({outcome})")

        # Feature vector (stored for debugging; also recomputed during train)
        x = extract_features(sig.get("signal_type", ""), sig).tolist()

        record = {
            "signal_id":    sid,
            "signal_ts":    sig.get("ts", 0),
            "harvest_ts":   now,
            "token":        token,
            "name":         sig.get("name", "?"),
            "symbol":       sig.get("symbol", "?"),
            "chain":        chain,
            "signal_type":  sig.get("signal_type", "unknown"),
            "conviction":   sig.get("conviction", 0),
            "mcap":         sig.get("mcap", 0),
            "entry_price":  entry_px,
            "current_price": current_px,
            "reward":       reward,
            "outcome":      outcome,
            "features":     x,
            # Original signal fields (for retraining)
            "dims":         sig.get("dims", {}),
            "liq":          sig.get("liq", 0),
            "age_days":     sig.get("age_days"),
            "buy_ratio":    sig.get("buy_ratio"),
            "h1_vol":       sig.get("h1_vol", 0),
            "h24_vol":      sig.get("h24_vol", 0),
        }
        with open(outcome_log, "a") as fh:
            fh.write(json.dumps(record) + "\n")

        new += 1
        time.sleep(0.3)   # rate-limit DexScreener

    if verbose:
        print(f"[harvest] {new} new outcomes appended to {outcome_log}")
    return new


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(outcome_log=OUTCOME_LOG, model_path=RL_MODEL_FILE, verbose=True):
    """Fit the LinUCB model from outcome_log.jsonl.

    Rebuilds A and b from scratch each time (full batch refit).
    This avoids numerical drift from many sequential online updates
    and makes the model deterministic given the same data.
    """
    if not os.path.exists(outcome_log):
        if verbose:
            print(f"[train] {outcome_log} not found — run harvest first")
        return None

    records = []
    with open(outcome_log) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if verbose:
        print(f"[train] {len(records)} outcome records")

    if not records:
        if verbose:
            print("[train] nothing to train on")
        return None

    # Build a fresh model (full refit from scratch)
    model = LinUCBModel()
    model.n_total = len(records)

    by_type = {}
    for rec in records:
        st = rec.get("signal_type", "unknown")
        by_type.setdefault(st, []).append(rec)

    for st, recs in by_type.items():
        arm = model.arm(st)
        # Reset to fresh state before full refit
        arm.A     = np.eye(FEATURE_DIM, dtype=np.float64)
        arm.b     = np.zeros(FEATURE_DIM, dtype=np.float64)
        arm.n_obs = 0

        for rec in recs:
            reward = rec.get("reward", 0.0)
            # Recompute features from stored fields (more reliable than stored vector)
            x = extract_features(st, {
                "conviction": rec.get("conviction", 0),
                "dims":       rec.get("dims", {}),
                "mcap":       rec.get("mcap", 0),
                "liq":        rec.get("liq", 0),
                "age_days":   rec.get("age_days"),
                "buy_ratio":  rec.get("buy_ratio"),
                "h1_vol":     rec.get("h1_vol", 0),
                "h24_vol":    rec.get("h24_vol", 0),
            })
            arm.update(x, reward)

        if verbose:
            # Compute mean expected reward at the arm's current theta
            A_inv = np.linalg.inv(arm.A)
            theta = A_inv @ arm.b
            rewards = [r.get("reward", 0) for r in recs]
            print(
                f"  {st:35s}  n={arm.n_obs:4d}  "
                f"mean_reward={sum(rewards)/len(rewards):+.3f}  "
                f"theta_norm={np.linalg.norm(theta):.3f}"
            )

    model.trained = time.time()
    model.save(model_path)
    if verbose:
        print(f"[train] model saved to {model_path}")
    return model


# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

def status(model_path=RL_MODEL_FILE, outcome_log=OUTCOME_LOG):
    """Print current model state and per-arm statistics."""
    model = LinUCBModel.load(model_path)

    trained_s = (
        datetime.fromtimestamp(model.trained, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        if model.trained else "never"
    )
    print(f"\n{'='*70}")
    print(f"LinUCB Contextual Bandit — Model Status")
    print(f"Trained:       {trained_s}")
    print(f"Total obs:     {model.n_total}")
    print(f"Feature dim:   {FEATURE_DIM}  |  Alpha: {ALPHA}  |  Arms: {len(model.arms)}")
    print(f"{'='*70}")

    # Load outcomes for stats
    outcomes_by_type = {}
    if os.path.exists(outcome_log):
        with open(outcome_log) as fh:
            for line in fh:
                try:
                    rec = json.loads(line.strip())
                    st  = rec.get("signal_type", "?")
                    outcomes_by_type.setdefault(st, []).append(rec.get("reward", 0))
                except Exception:
                    pass

    print(f"\n{'Signal Type':<35}  {'n':>4}  {'Mean R':>7}  {'Win%':>5}  {'Bonus@conv7':>11}  {'Status'}")
    print("-" * 80)

    # Probe UCB score at a "typical good signal" — conv=7, all dims=7,
    # $50K mcap, $20K liq, 60d age, 0.6 buy ratio, 10× surge
    typical = {
        "conviction": 7.0,
        "dims": {"organic": 7.0, "tokenomics": 7.0, "maturity": 7.0, "momentum": 7.0},
        "mcap": 50_000, "liq": 20_000, "age_days": 60,
        "buy_ratio": 0.6, "h1_vol": 5_000, "h24_vol": 12_000,
    }

    for st in SIGNAL_TYPES:
        arm      = model.arm(st)
        rewards  = outcomes_by_type.get(st, [])
        n        = arm.n_obs
        mean_r   = sum(rewards) / len(rewards) if rewards else 0.0
        wins     = sum(1 for r in rewards if r > 0.18) / len(rewards) * 100 if rewards else 0.0
        trained  = n >= MIN_OBS_TRAIN

        if trained:
            x      = extract_features(st, typical)
            _, mu  = arm.score(x)
            bonus  = np.clip(mu / REWARD_CLIP * 1.5, -1.5, 1.5)
            bonus_s = f"{bonus:+.2f}"
        else:
            bonus_s = "  n/a"

        status_s = f"✅ active (n={n})" if trained else f"⬜ cold   (need {MIN_OBS_TRAIN - n} more)"
        print(f"  {st:<33}  {n:>4}  {mean_r:>+.3f}  {wins:>4.0f}%  {bonus_s:>11}  {status_s}")

    print()

    # Feature importance: which features have the highest |theta| weight
    # across all arms (averaged)
    FEAT_NAMES = [
        "conviction", "organic", "tokenomics", "maturity", "momentum",
        "log_mcap",   "buy_ratio", "log_liq",  "age_norm", "surge_ratio", "bias",
    ]
    total_theta = np.zeros(FEATURE_DIM)
    active_arms = 0
    for arm in model.arms.values():
        if arm.n_obs >= MIN_OBS_TRAIN:
            A_inv = np.linalg.inv(arm.A)
            total_theta += np.abs(A_inv @ arm.b)
            active_arms += 1

    if active_arms > 0:
        avg_theta = total_theta / active_arms
        ranked    = sorted(enumerate(avg_theta), key=lambda t: -t[1])
        print("Feature importance (avg |θ| across active arms):")
        for idx, weight in ranked[:5]:
            print(f"  {FEAT_NAMES[idx]:<15}  {weight:.4f}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "harvest":
        harvest()
    elif cmd == "train":
        train()
    elif cmd == "status":
        status()
    elif cmd == "all":
        print("=== HARVEST ===")
        harvest()
        print("\n=== TRAIN ===")
        train()
        print("\n=== STATUS ===")
        status()
    else:
        print(f"Usage: python signal_rl.py [harvest|train|status|all]")
        print()
        print("  harvest  — fetch 7d-later prices for logged signals, write outcome_log.jsonl")
        print("  train    — fit LinUCB model from outcome_log.jsonl, save rl_model.json")
        print("  status   — print per-arm stats and feature importance")
        print("  all      — harvest + train + status (use in daily cron)")
        sys.exit(1)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()
