#!/usr/bin/env bash
# Regenerates the Random | Expert | TabPFN-BC comparison GIFs for every
# environment in ENVS below: record_comparison_video.py renders the combined
# comparison_<env>.gif, then split_and_render_gifs.py crops it into
# random_<env>.gif / expert_<env>.gif / tabpfn_bc_<env>.gif.
#
#   ./scripts/render_all_gifs.sh                       # all envs below
#   ./scripts/render_all_gifs.sh invertedpendulum       # just one env
#   ./scripts/render_all_gifs.sh hopper swimmer         # a subset
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_ENVS=(invertedpendulum halfcheetah hopper swimmer)
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
    uv run python scripts/record_comparison_video.py --env "$env"
    uv run python scripts/split_and_render_gifs.py --env "$env"
done

draw_bar "$TOTAL" "$TOTAL" "done"
echo "generated: random_<env>.gif, expert_<env>.gif, tabpfn_bc_<env>.gif for: ${ENVS[*]}"
