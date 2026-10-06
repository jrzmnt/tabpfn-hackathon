#!/usr/bin/env bash
# Runs the entire Atari extension end to end: experiments (collect -> offline
# check -> closed-loop comparison) for all 4 games, GIFs for all 4 games, and
# the summary grid. One command, with a progress bar per stage.
#
#   ./scripts/run_all_atari.sh
#
# Expected total wall-clock time: ~4-4.5 hours, dominated by:
#   - mspacman/spaceinvaders/pong closed-loop comparisons (true full episodes,
#     ~20min + ~40min + ~1.9h -- see run_atari_experiments.sh's header)
#   - breakout's closed-loop comparison, run separately with an explicit,
#     documented truncation (--max-steps 2500, ~1.7h) since its true average
#     episode length (~5,745 steps) would otherwise take ~6.6h alone
#   - GIF recording for all 4 games (~500-step clips, much cheaper than the
#     full-episode numeric runs above)
# Safe to Ctrl-C and rerun: collect_atari_demos.py skips games already
# collected; everything else always re-runs and overwrites.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BREAKOUT_MAX_STEPS=2500  # documented truncation -- see header comment and README

STAGES=(
    "Experiments: mspacman, spaceinvaders, pong (full episodes)"
    "Experiment: breakout (truncated at ${BREAKOUT_MAX_STEPS} steps)"
    "GIFs: all 4 games"
    "Summary grid"
)
TOTAL=${#STAGES[@]}
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

draw_bar 0 "$TOTAL" "${STAGES[0]}"
./scripts/run_atari_experiments.sh mspacman spaceinvaders pong

draw_bar 1 "$TOTAL" "${STAGES[1]}"
uv run python scripts/collect_atari_demos.py --env breakout
uv run python scripts/atari_tabpfn_bc.py --env breakout
uv run python scripts/evaluate_atari_policies.py --env breakout --max-steps "$BREAKOUT_MAX_STEPS"

draw_bar 2 "$TOTAL" "${STAGES[2]}"
./scripts/render_all_atari_gifs.sh

draw_bar 3 "$TOTAL" "${STAGES[3]}"
uv run python scripts/make_atari_summary_grid.py

draw_bar "$TOTAL" "$TOTAL" "done"
echo "results: data/atari_results_<env>.npz, data/atari_tabpfn_bc_results_<env>.npz"
echo "media: media/atari_results_grid.png, media/{random,expert,tabpfn_bc}_<env>.gif"
echo "note: breakout's closed-loop numbers used --max-steps ${BREAKOUT_MAX_STEPS} (a documented"
echo "truncation, not its true ~5,745-step average episode length) -- report that alongside its results."
