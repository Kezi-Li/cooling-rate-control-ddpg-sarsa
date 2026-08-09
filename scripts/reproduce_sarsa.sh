#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

python -m cooling_rate_python.sarsa \
  --episodes 800 \
  --runs 10 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --cpus 10 \
  --reference 1200 \
  --layers 5 \
  --substrate-length 150 \
  --output Results/Results_SARSA \
  --quiet
