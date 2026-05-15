#!/usr/bin/env bash
set -euo pipefail

autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/hopper_v0/benchmark.json \
  --seed-candidate autoresearch_gym/tasks/hopper_v0/seed_trainable.py \
  --tag smoke \
  --train-episodes 2 \
  --eval-episodes 1 \
  --no-record
