#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

python -m cooling_rate_python.train \
  --episodes 150 \
  --runs 10 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --noise-std 0.2 \
  --cpus 10 \
  --output Results/Results_DDPG \
  --quiet
