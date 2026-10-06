#!/usr/bin/env bash
# Runs the full Atari TabPFN-BC pipeline (collect -> offline check ->
# closed-loop comparison) for each game in ENVS below. Idempotent for the
# collection step (collect_atari_demos.py skips envs already collected);
# the offline check and closed-loop comparison always re-run and overwrite.
#
#   ./scripts/run_atari_experiments.sh                          # mspacman, spaceinvaders, pong
#   ./scripts/run_atari_experiments.sh mspacman                  # just one
#
# evaluate_atari_policies.py now runs each of N_ENVS=20 parallel episodes to
# its *real* end (done=True), not a short fixed cutoff -- needed for a
# scientifically meaningful return, at the cost of a much longer runtime than
# an early version of this pipeline used. Per-game closed-loop rollout time
# (TabPFNClassifier on 512-dim embeddings, ~4s/call at N_DEMO_TRANSITIONS=500)
# is bounded by that game's average real episode length:
#   mspacman ~291 steps       -> ~20 min
#   spaceinvaders ~592 steps  -> ~40 min
#   pong ~1,658 steps         -> ~1.9 h
#   breakout ~5,745 steps     -> ~6.6 h -- NOT included by default (see below)
#
# breakout is excluded from ENVS below because a true full episode there is
# too expensive to run casually. Run it separately with an explicit,
# documented truncation, e.g.:
#   uv run python scripts/collect_atari_demos.py --env breakout
#   uv run python scripts/atari_tabpfn_bc.py --env breakout
#   uv run python scripts/evaluate_atari_policies.py --env breakout --max-steps 2500
# (~2500 steps is a deliberate, reported truncation, not breakout's true
# episode length -- document that cap wherever these results are used.)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_ENVS=(mspacman spaceinvaders pong)  # breakout excluded by default -- see header comment
ENVS=("${@:-${DEFAULT_ENVS[@]}}")
TOTAL=${#ENVS[@]}
BAR_WIDTH=30

draw_bar() {
    local current=$1 total=$2 label=$3
    local filled=$(( current * BAR_WIDTH / total ))
    local empty=$(( BAR_WIDTH - filled ))
    local pct=$(( current * 100 / total ))
    local bar
    bar=$(printf '%*s' "$filled" '' | tr ' ' '#')
    bar+=$(printf '%*s' "$empty" '' | tr ' ' '-')
    printf '\n[%s] %d/%d (%d%%) -- %s\n\n' "$bar" "$current" "$total" "$pct" "$label"
}

i=0
for env in "${ENVS[@]}"; do
    i=$((i + 1))
    draw_bar "$((i - 1))" "$TOTAL" "starting $env"
    uv run python scripts/collect_atari_demos.py --env "$env"
    uv run python scripts/atari_tabpfn_bc.py --env "$env"
    uv run python scripts/evaluate_atari_policies.py --env "$env"
done

draw_bar "$TOTAL" "$TOTAL" "done"
echo "results: data/atari_demos_<env>.npz, data/atari_tabpfn_bc_results_<env>.npz, data/atari_results_<env>.npz for: ${ENVS[*]}"
