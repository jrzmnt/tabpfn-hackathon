#!/usr/bin/env bash
# Records and splits the Random | Expert | TabPFN-BC comparison GIFs for
# each Atari game in ENVS below: record_comparison_video_atari.py renders
# the combined comparison_<env>.gif, then split_and_render_gifs_atari.py
# crops it into random_<env>.gif / expert_<env>.gif / tabpfn_bc_<env>.gif.
# Requires data/atari_demos_<env>.npz to already exist for each game (run
# run_atari_experiments.sh, or at least collect_atari_demos.py, first).
#
#   ./scripts/render_all_atari_gifs.sh                    # all 4 games
#   ./scripts/render_all_atari_gifs.sh pong breakout       # a subset
#
# Per-game cost: dominated by the TabPFN-BC rollout (100 steps x ~4-8s per
# TabPFNClassifier.predict() call) -- expect ~7-10 min/game.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_ENVS=(pong breakout spaceinvaders mspacman)
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
    uv run python scripts/record_comparison_video_atari.py --env "$env"
    uv run python scripts/split_and_render_gifs_atari.py --env "$env"
done

draw_bar "$TOTAL" "$TOTAL" "done"
echo "generated: random_<env>.gif, expert_<env>.gif, tabpfn_bc_<env>.gif for: ${ENVS[*]}"
