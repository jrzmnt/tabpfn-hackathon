#!/usr/bin/env bash
# Collects (state, action) demonstrations from every expert registered
# in scripts/envs.py's ENVIRONMENTS dict, saving one
# data/demos_<env>.npz per environment.
#
#   ./scripts/collect_all.sh              # all envs, default 20 episodes
#   ./scripts/collect_all.sh --episodes 5 # fewer episodes, e.g. for a quick smoke test
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ant is excluded: its rl-baselines3-zoo "-v3" checkpoint was trained on an
# old mujoco-py observation space that current gymnasium can't reproduce
# with a public kwarg -- see the ENVIRONMENTS docstring in envs.py.
# humanoidstandup is excluded from this bulk run since it's much slower
# end-to-end (17 action dims); collect it explicitly if you want it.
ENVS=(halfcheetah hopper walker2d invertedpendulum swimmer)

for env in "${ENVS[@]}"; do
    echo "=== collecting demonstrations: $env ==="
    uv run python scripts/collect_demos.py --env "$env" "$@"
    echo
done

echo "done -- see data/demos_<env>.npz for each of: ${ENVS[*]}"
