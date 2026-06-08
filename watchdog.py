"""
Watchdog — launches autonomous_loop.py as a subprocess and restarts it
if it crashes. Also keeps the dashboard HTML in sync at /tmp/eth-dashboard/.
Runs for 7 days then exits cleanly.
"""
import subprocess
import sys
import os
import time
import shutil
from datetime import datetime, timezone, timedelta

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LOOP_SCRIPT = os.path.join(SCRIPT_DIR, "autonomous_loop.py")
HTML_SRC    = os.path.join(SCRIPT_DIR, "dashboard", "index.html")
DASHBOARD   = "/tmp/eth-dashboard"
LOG_FILE    = os.path.join(SCRIPT_DIR, "watchdog.log")
RUN_DAYS    = 7
RESTART_DELAY = 10   # seconds between crash restarts
PYTHON      = sys.executable


def _log(msg):
    ts  = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] [watchdog] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def sync_dashboard():
    os.makedirs(DASHBOARD, exist_ok=True)
    if os.path.exists(HTML_SRC):
        shutil.copy2(HTML_SRC, os.path.join(DASHBOARD, "index.html"))


def run():
    end_time = datetime.now(timezone.utc) + timedelta(days=RUN_DAYS)
    _log(f"Watchdog started. Loop will run until {end_time.date()}.")
    sync_dashboard()

    crashes = 0
    while datetime.now(timezone.utc) < end_time:
        _log(f"Starting autonomous_loop.py (restart #{crashes})...")
        try:
            proc = subprocess.Popen(
                [PYTHON, LOOP_SCRIPT],
                cwd=SCRIPT_DIR,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            proc.wait()
            rc = proc.returncode
        except Exception as e:
            _log(f"Failed to launch: {e}")
            rc = -1

        if datetime.now(timezone.utc) >= end_time:
            _log("7-day run complete. Watchdog exiting.")
            break

        crashes += 1
        _log(f"autonomous_loop.py exited with code {rc}. "
             f"Restarting in {RESTART_DELAY}s... (total crashes: {crashes})")
        sync_dashboard()   # refresh HTML on each restart
        time.sleep(RESTART_DELAY)

    _log("Watchdog done.")


if __name__ == "__main__":
    run()
