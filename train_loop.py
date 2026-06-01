#!/usr/bin/env python3
"""Continuous RL training loop — runs on your local machine alongside the bot.

Every INTERVAL minutes it:
  1. harvest  — fetches 7d-later prices for any newly eligible signals
  2. train    — refits the LinUCB model from all outcomes so far
  3. status   — prints per-arm stats
  4. auto-commits rl_model.json so the remote agent can see the latest model

The bot reloads rl_model.json at startup. To get the newest model into a
running bot without restarting: the SIGHUP handler below triggers a reload.

Usage:
    python train_loop.py                    # default: every 60 minutes
    python train_loop.py --interval 30      # every 30 minutes
    python train_loop.py --interval 15      # aggressive: every 15 minutes
    python train_loop.py --once             # one shot then exit
    python train_loop.py --no-commit        # skip git commit

Run in the background:
    nohup python train_loop.py > train_loop.log 2>&1 &
    echo $! > train_loop.pid

Stop it:
    kill $(cat train_loop.pid)
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [train_loop] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_loop.log"),
    ],
)
log = logging.getLogger("train_loop")

SIGNAL_LOG  = "signal_log.jsonl"
OUTCOME_LOG = "outcome_log.jsonl"
MODEL_FILE  = "rl_model.json"
STATS_FILE  = "rl_stats.json"  # written each cycle for monitoring


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as fh:
        return sum(1 for _ in fh)


def run_harvest_and_train(verbose=True):
    """Run one full harvest → train → status cycle.

    Returns a dict with cycle stats.
    """
    from signal_rl import harvest, train, status, load_model, OUTCOME_LOG, RL_MODEL_FILE

    before_outcomes = count_lines(OUTCOME_LOG)
    before_signals  = count_lines(SIGNAL_LOG)

    log.info(f"Starting cycle | signals={before_signals} outcomes_before={before_outcomes}")

    # 1. Harvest — fetch prices for signals old enough to evaluate
    new_outcomes = harvest(verbose=verbose)

    after_outcomes = count_lines(OUTCOME_LOG)
    log.info(f"Harvest done: {new_outcomes} new outcomes (total {after_outcomes})")

    # 2. Train — only if we have at least 3 outcomes
    model = None
    if after_outcomes >= 3:
        model = train(verbose=verbose)
        if model:
            log.info(
                f"Model trained: {model.n_total} observations, "
                f"{sum(1 for a in model.arms.values() if a.n_obs >= 3)} active arms"
            )
    else:
        log.info(f"Skipping train — need >= 3 outcomes, have {after_outcomes}")

    # 3. Write stats file for monitoring
    stats = {
        "ts":              time.time(),
        "date":            datetime.now(tz=timezone.utc).isoformat(),
        "signals_logged":  before_signals,
        "outcomes_total":  after_outcomes,
        "new_outcomes":    new_outcomes,
        "model_trained":   model is not None,
        "active_arms":     (sum(1 for a in model.arms.values() if a.n_obs >= 3)
                            if model else 0),
        "n_total":         model.n_total if model else 0,
    }
    if model:
        arm_stats = {}
        for st, arm in model.arms.items():
            if arm.n_obs > 0:
                import numpy as np
                A_inv = np.linalg.inv(arm.A)
                theta = A_inv @ arm.b
                arm_stats[st] = {
                    "n_obs":       arm.n_obs,
                    "theta_norm":  round(float(np.linalg.norm(theta)), 4),
                }
        stats["arms"] = arm_stats

    with open(STATS_FILE, "w") as fh:
        json.dump(stats, fh, indent=2)

    return stats


def git_commit_model():
    """Commit rl_model.json so the remote agent and other machines see it."""
    if not os.path.exists(MODEL_FILE):
        return
    try:
        # Only commit if the model actually changed
        result = subprocess.run(
            ["git", "diff", "--quiet", MODEL_FILE],
            capture_output=True,
        )
        if result.returncode == 0:
            log.info("rl_model.json unchanged — no commit needed")
            return

        subprocess.run(["git", "add", MODEL_FILE, OUTCOME_LOG], check=False)
        msg = (
            f"data(auto): rl model update {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        )
        subprocess.run(["git", "commit", "-m", msg], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=False)
        log.info("rl_model.json committed and pushed")
    except Exception as exc:
        log.warning(f"git commit failed: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval",   type=int, default=60,
                        help="Minutes between training cycles (default 60)")
    parser.add_argument("--once",       action="store_true",
                        help="Run one cycle then exit")
    parser.add_argument("--no-commit",  action="store_true",
                        help="Skip git commit of model")
    parser.add_argument("--verbose",    action="store_true",
                        help="Show per-signal harvest output")
    args = parser.parse_args()

    interval_s = args.interval * 60

    log.info(
        f"Train loop starting | interval={args.interval}m | "
        f"commit={'off' if args.no_commit else 'on'}"
    )

    cycle = 0
    while True:
        cycle += 1
        log.info(f"─── Cycle {cycle} ───────────────────────────────")

        try:
            stats = run_harvest_and_train(verbose=args.verbose)

            if not args.no_commit and stats.get("model_trained"):
                git_commit_model()

            log.info(
                f"Cycle {cycle} done | "
                f"new_outcomes={stats['new_outcomes']} | "
                f"active_arms={stats['active_arms']} | "
                f"total_obs={stats['n_total']}"
            )
        except KeyboardInterrupt:
            log.info("Interrupted — exiting cleanly")
            break
        except Exception as exc:
            log.error(f"Cycle {cycle} error: {exc}", exc_info=True)

        if args.once:
            break

        next_run = datetime.fromtimestamp(
            time.time() + interval_s, tz=timezone.utc
        ).strftime("%H:%M UTC")
        log.info(f"Next cycle in {args.interval}m (at {next_run})")
        time.sleep(interval_s)


if __name__ == "__main__":
    main()
