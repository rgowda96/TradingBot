"""Thin DexScreener API client with retry/backoff."""
import logging
import time

import requests

API = "https://api.dexscreener.com"
log = logging.getLogger("dexscreener")


class DexScreener:
    def __init__(self, timeout=15, max_retries=4):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "base-volume-spike-bot/1.0"})
        self.timeout = timeout
        self.max_retries = max_retries

    def _get(self, path, params=None):
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(
                    f"{API}{path}", params=params, timeout=self.timeout
                )
                if resp.status_code == 429:  # rate limited
                    wait = 2 ** attempt
                    log.warning("rate limited on %s, retrying in %ss", path, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == self.max_retries - 1:
                    log.error("request to %s failed: %s", path, exc)
                    return None
                time.sleep(2 ** attempt)
        return None

    def token_profiles_latest(self):
        """Latest token profiles across all chains."""
        return self._get("/token-profiles/latest/v1") or []

    def token_boosts_latest(self):
        return self._get("/token-boosts/latest/v1") or []

    def token_boosts_top(self):
        return self._get("/token-boosts/top/v1") or []

    def search(self, query):
        """Search pairs by free-text query. Returns a list of pair objects."""
        data = self._get("/latest/dex/search", {"q": query})
        return (data or {}).get("pairs") or []

    def tokens(self, chain, addresses):
        """Fetch all pairs for up to 30 token addresses on a chain."""
        if not addresses:
            return []
        joined = ",".join(addresses)
        data = self._get(f"/tokens/v1/{chain}/{joined}")
        return data or []
