#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Install deps if needed
pip install -q flask>=3.0.0 requests>=2.31.0

echo "Starting DexScreener Token Screener on http://localhost:5001"
python screener.py "$@"
