#!/bin/zsh
cd "/Users/rakshaksgowda/Desktop/shayan's custom strategy"
while true; do
  python3 strategy.py >> /tmp/eth_runner.log 2>&1
  sleep 60
done
