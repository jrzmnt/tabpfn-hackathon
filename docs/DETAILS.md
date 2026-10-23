# Engineering details

The [main README](../README.md) keeps the story tight: what we tried, why
it matters, and how the results back that up. This doc has everything that
didn't need to be in the reader's way to get there: framing, the
pipeline's internals, the full file/script reference, exact per-step
`How to run` timings, and the complete limitations writeup (compatibility
bugs, latency investigation, every environment's individual story).

## Framing

This is a *supervised* special case of Behavior Cloning from Observation
(BCO): MuJoCo/Atari let us log the expert's actions directly, so each
demonstration is a `(state, action)` pair and TabPFN is fit straight on
that supervised signal. This sidesteps the harder, observation-only setting
(where actions must be inferred from state transitions alone, e.g. via an
inverse dynamics model), which remains the natural next research step, not
part of this prototype.

## Architecture

```
PPO expert (rl-baselines3-zoo, HuggingFace Hub)
        │  rollout, log (state, action) pairs
        ▼
data/demos_halfcheetah.npz  (20 episodes, 20,000 transitions)
        │  subsample context
        ▼
6x TabPFNRegressor (V3.5 "Fast"), one per action dimension
  .fit(states_context, action_dim_context)   <- in-context "training", no gradients
        │
        ▼
tabpfn_policy(obs) = stack([reg.predict(obs) for reg in regressors])
        │
        ▼
closed-loop rollout in HalfCheetah-v4, compared against expert & random
```

`TabPFNRegressor` is single-output, and MuJoCo action spaces are
multi-dimensional, so we fit one regressor per action dimension, each given
the same context states but a different target column.

### Pipeline: expert loop vs. TabPFN-BC loop

The two closed loops being compared, side by side. The key point: TabPFN
does not learn during the rollout. It is given the expert's demonstrations
as context exactly once, before the loop starts; every step then just
queries that same fixed context (in-context inference, no weight updates).

**Phase 1: demonstration collection (offline, once):**
```
state -> expert (PPO) -> action -> environment -> state+1   (repeated 20,000x)
                          │
                          ▼
            (state, action) pairs saved as data/demos_halfcheetah.npz
```

**Phase 2: TabPFN fit (offline, once, no gradients):**
```
context = sample of (state, action) pairs from Phase 1
TabPFN.fit(context)   <- in-context "training": encodes the context, no weight updates
```

**Phase 3a: expert closed loop (for comparison):**
```
state -> expert (PPO) -> action -> environment -> state+1
```

**Phase 3b: TabPFN-BC closed loop (the policy under test):**
```
state -> TabPFN.predict(state | fixed expert context) -> action -> environment -> state+1
```

Phase 3a and 3b have the same shape: a state goes in, an action comes
out, the environment advances, but 3b's "brain" is TabPFN doing in-context
prediction against a frozen context, instead of a network trained with PPO.

## Why evaluation batches across parallel episodes

TabPFN's per-call inference cost is dominated by a roughly fixed overhead
of encoding the context (~0.5s per `predict()` call in our tests),
almost independent of how many query points are in the batch. A naive
step-by-step closed-loop rollout of N sequential 1000-step episodes would
need `N * 1000 * 6` separate `predict()` calls: tens of thousands, i.e.
hours of wall-clock time just for inference.

Instead, `evaluate_policies.py` runs many episodes **in parallel** using a
vectorized environment (`N_ENVS` copies of `HalfCheetah-v4` stepped in
lockstep) and caps each episode at `EPISODE_LEN` steps. At every timestep
we make exactly one batched `predict()` call per action dimension across
all parallel envs, so the total inference cost is
`EPISODE_LEN * 6` calls regardless of how many "episodes" (parallel envs)
we run. The expert and random baselines are re-evaluated under the same
capped episode length for a fair, apples-to-apples comparison: the raw
20-episode, full-1000-step expert numbers logged during demonstration
collection (`data/demos_halfcheetah.npz`) are systematically higher because
they run four times longer per episode.

## Project layout

```
tabpfn-hackathon/
├── scripts/      MuJoCo + Atari pipeline scripts (see tables below)
├── data/         collected demonstrations + numeric results (*.npz)
├── media/        plots and GIFs (*.png, *.gif) referenced by the README
├── docs/         this file
├── pyproject.toml, uv.lock, .python-version
└── .env          TABPFN_TOKEN, git-ignored, never committed
```

## Files (`scripts/`): MuJoCo

| File | Purpose |
| --- | --- |
| `collect_demos.py` | Generic demo collector: `--env {halfcheetah,hopper,walker2d,humanoidstandup,invertedpendulum,swimmer}` picks an expert checkpoint (repo/algorithm/VecNormalize-or-not, all in `scripts/envs.py`) and rolls it out `--episodes` (default 20) with a `rich` progress bar, saves `data/demos_<env>.npz`. `ant` is registered but disabled, see Limitations. |
| `collect_data.py` | Unified collector across **both** registries (MuJoCo `envs.py` + Atari `atari_envs.py`): `--env <any>` auto-detects which one, `--n-transitions <N>` (not `--episodes`) collects whole episodes until at least N `(state, action)` pairs exist. Useful because episode length varies wildly by env (invertedpendulum ~200 steps vs. breakout ~5,745), so "20 episodes" means very different context sizes; this lets you target the context size directly. Same output paths/format as `collect_demos.py`/`collect_atari_demos.py`. |
| `tabpfn_bc.py` | Offline check: `--env <env>` fits TabPFN on a context subsample, evaluates R²/MSE on held-out demo transitions (no environment interaction). Saves `data/tabpfn_bc_results_<env>.npz`. |
| `evaluate_policies.py` | Main deliverable: `--env <env>` closed-loop comparison of expert / TabPFN-BC / random policies via parallel vectorized rollouts, saves `data/results_<env>.npz` and `media/results_<env>.png`. `--episode-len <N>` overrides the default 200-step cap (used for the HalfCheetah 1,000-step re-run in Results). |
| `record_comparison_video.py` | `--env <env>` renders one expert episode and one TabPFN-BC episode combined into `media/comparison_<env>.gif`. |
| `split_and_render_gifs.py` | `--env <env>` splits `media/comparison_<env>.gif`'s three side-by-side panels (Random / Expert / TabPFN-BC, that order) into `media/random_<env>.gif` + `media/expert_<env>.gif` + `media/tabpfn_bc_<env>.gif` (pure crop, no re-render), for placing the three side by side in the README. |
| `make_summary_grid.py` | `--envs <env> [<env> ...]` combines each env's `data/results_<env>.npz` into one 2x2 (or NxM) bar-chart grid, `media/results_grid.png`. Pure plotting from already-saved data, no TabPFN/environment calls, cheap to rerun. |
| `envs.py` | Shared `ENVIRONMENTS` registry (env id, HF repo, algorithm, `VecNormalize`-or-not) plus `resolve_env`/`load_expert`/`make_venv` helpers, imported by every script above. Not runnable on its own. |
| `collect_all.sh` | Runs `collect_demos.py` for `halfcheetah`, `hopper`, `walker2d`, `invertedpendulum`, `swimmer`: one `data/demos_<env>.npz` each. Extra args (e.g. `--episodes 5`) are forwarded to each run. `humanoidstandup` isn't included by default since it's much slower end-to-end (17 action dims); run it explicitly. |
| `render_all_gifs.sh` | Runs `record_comparison_video.py` + `split_and_render_gifs.py` for `invertedpendulum`, `halfcheetah`, `hopper`, `swimmer` (or the envs passed as args), with a progress bar between environments. |

## Files (`scripts/`): Atari

| File | Purpose |
| --- | --- |
| `atari_envs.py` | Registry (env id, HF repo, model filename) + shared helpers (`load_atari_expert`, `make_atari_venv`, `extract_features`) for the 4 games. Not runnable on its own. |
| `collect_atari_demos.py` | `--env {pong,breakout,spaceinvaders,mspacman}` rolls out the PPO expert, saves `data/atari_demos_<env>.npz` ((512-dim embedding, discrete action) pairs, not pixels). |
| `atari_tabpfn_bc.py` | Offline check: fits one `TabPFNClassifier` on a context subsample, reports held-out action-prediction accuracy vs. a majority-class baseline. Saves `data/atari_tabpfn_bc_results_<env>.npz`. |
| `evaluate_atari_policies.py` | Closed-loop comparison of expert / TabPFN-BC / random policies via parallel vectorized rollouts, saves `data/atari_results_<env>.npz`. |
| `record_comparison_video_atari.py` | Renders Random / Expert / TabPFN-BC gameplay side by side into `media/comparison_<env>.gif`. |
| `split_and_render_gifs_atari.py` | Splits that into `media/random_<env>.gif` + `media/expert_<env>.gif` + `media/tabpfn_bc_<env>.gif` (pure crop, no re-render). |
| `make_atari_summary_grid.py` | Combines the 4 games' `atari_results_<env>.npz` into `media/atari_results_grid.png`. Pure plotting, no TabPFN/environment calls. |
| `run_atari_experiments.sh` | Runs collect, then offline check, then closed-loop comparison for each game, with a progress bar. |
| `render_all_atari_gifs.sh` | Runs the record + split GIF steps for each game, with a progress bar. |
| `run_all_atari.sh` | Runs everything above end to end for all 4 games in one command: experiments (mspacman/spaceinvaders/pong full episodes, then breakout with a documented `--max-steps` truncation), GIFs, and the summary grid. ~4-4.5h total; see its header comment for the per-stage breakdown. |

## Data (`data/`)

| File pattern | Purpose |
| --- | --- |
| `demos_<env>.npz` / `atari_demos_<env>.npz` | Collected expert demonstrations for that env (states or embeddings, actions, per-episode returns). |
| `tabpfn_bc_results_<env>.npz` / `atari_tabpfn_bc_results_<env>.npz` | Offline R²/MSE or accuracy check output. |
| `results_<env>.npz` / `atari_results_<env>.npz` | Closed-loop return arrays (paired with `media/results_<env>.png` / `media/atari_results_grid.png`). |

## How to run, step by step (with timings)

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12+.

**Step 1: install dependencies:**
```bash
uv sync
```

**Step 2: authorize TabPFN-3.5 (one time only).** It needs a one-time
license acceptance before it can download model weights:
1. Log in at https://ux.priorlabs.ai and accept the license under the
   Licenses tab.
2. Copy your API key from https://ux.priorlabs.ai/account.
3. Put it in a local `.env` file (already git-ignored, never committed):
   ```bash
   echo 'TABPFN_TOKEN=your-api-key-here' > .env
   ```

**Step 3: collect expert demonstrations:**
```bash
uv run python scripts/collect_demos.py --env halfcheetah   # ~ a few minutes; produces data/demos_halfcheetah.npz
# uv run python scripts/collect_demos.py --env hopper      # or: walker2d, invertedpendulum, swimmer, humanoidstandup
# ./scripts/collect_all.sh                                 # or collect every registered env in one go
```

**Step 4: offline sanity check** (fits TabPFN on a context subsample,
checks R²/MSE against held-out states, no environment interaction, ~3 min):
```bash
uv run python scripts/tabpfn_bc.py --env halfcheetah
```

**Step 5: closed-loop comparison** (expert vs TabPFN-BC vs random policy
in the real environment, ~10-15 min for halfcheetah/hopper, this is the
main result):
```bash
uv run python scripts/evaluate_policies.py --env halfcheetah
```

**Step 6: record the side-by-side GIFs** (~7-8 min, dominated by the
TabPFN-BC rollout; Random and Expert render much faster):
```bash
uv run python scripts/record_comparison_video.py --env halfcheetah
uv run python scripts/split_and_render_gifs.py --env halfcheetah
# or, for every env with a comparable result in one go, with a progress bar:
./scripts/render_all_gifs.sh
```

**Step 7: combine the results into one grid** (seconds, pure plotting
from the `data/results_<env>.npz` files already on disk, no TabPFN or
environment calls):
```bash
uv run python scripts/make_summary_grid.py
```

Every script in steps 3-6 takes the same `--env` flag (`invertedpendulum`,
`halfcheetah`, `hopper`, `walker2d`, `swimmer`, or `humanoidstandup`);
rerun the ones you care about per environment. `humanoidstandup` in
particular is much slower (17 action dims).

For Atari, `./scripts/run_all_atari.sh` wraps the equivalent sequence for
all 4 games (~4-4.5h total, see its header comment for why, and the main
README's Atari section for per-game timings and the documented `breakout`
truncation).

## Limitations & next steps

- **Action-only cloning, not observation-only.** This prototype uses the
  expert's logged actions directly. The BCO framework this is modeled on
  targets the harder setting where only state trajectories are available
  and actions must be inferred (e.g. via an inverse dynamics model), a
  natural extension.
- **TabPFN inference latency.** Each `predict()` call has non-trivial fixed
  overhead, which is why closed-loop evaluation here uses capped,
  parallelized episodes rather than the full 1000-step horizon. Reducing
  this further (e.g. smaller context, distillation) would let this scale to
  longer horizons and more complex environments.
- **One `TabPFNRegressor` per action dimension: investigated, and it's not
  where the redundant cost actually is.** `TabPFNRegressor.fit()`/`.predict()`
  are strictly single-output (`n_outputs_: Literal[1]` in
  `tabpfn/regressor.py`), so we fit N regressors on the same context states
  `X`, one per action dimension. We looked into whether the transformer's
  encoding of `X` (shared across all N) could be computed once and reused;
  it can't: in `tabpfn/architectures/tabpfn_v3_5.py`, `y` is mixed into the
  per-row embedding (`_process_row_chunk`, ~line 2988) *before* every
  attention-heavy stage (distribution embedder, column aggregator, ICL
  transformer), so almost the entire forward pass is already
  target-conditioned. There's no reusable "encode X only" checkpoint to
  cache, and separating them would mean rewriting those internals, not a
  quick patch.

  The actual redundant cost turned out to be simpler: every
  `TabPFNRegressor.create_default_for_version(...)` call *rebuilds the whole
  transformer from scratch* (`model_loading.py`'s `load_model()`), because
  its built-model LRU cache is opt-in and disabled by default
  (`TABPFN_MODEL_CACHE_SIZE=0`). Setting `TABPFN_MODEL_CACHE_SIZE=1` (done at
  the top of every script that creates multiple `TabPFNRegressor` instances,
  1 is enough since every regressor in a given run loads the *same*
  checkpoint identity, "V3.5 Fast") lets those N instances share the
  already-built model instead of reconstructing it N times: measured
  **~31% faster** (7.11s to 4.90s for 6 dims' fit+predict on our HalfCheetah
  setup), for a one-line, zero-risk change. It mainly speeds up the one-time
  N-regressor setup cost (e.g. `tabpfn_bc.py`'s fit/predict loop); it doesn't
  touch the per-timestep `predict()` cost during an already-fit closed-loop
  rollout, so `evaluate_policies.py`'s EPISODE_LEN-step rollout gets a
  smaller win from this than the offline scripts do.

  We initially set this to `4` "for safety margin," but on
  `humanoidstandup` (348 obs dims, 17 action dims, much larger than the
  other environments) that caused an MPS (Apple GPU) out-of-memory error:
  the cache keeps built models resident on-device, and 4 simultaneous
  copies of a model sized for 348-dim inputs exceeded the unified-memory
  limit. Since a size of 1 is all any single script run actually needs, we
  lowered it to `1` everywhere: same speedup, no memory headroom wasted on
  unused cache slots.
- **`ant` is registered but disabled.** Its rl-zoo checkpoint
  (`sb3/ppo-Ant-v3`) was trained on the old mujoco-py `-v3` environment,
  whose observation layout (112 dims, including contact forces counted
  differently) doesn't match current gymnasium `Ant-v4`/`v5` (27 dims) and
  can't be restored with a public `gym.make()` kwarg the way `walker2d`'s
  could (`exclude_current_positions_from_observation=False`). `swimmer` had
  the identical problem (8 dims vs. the checkpoint's 9) until we found
  `farama-minari/Swimmer-v5-PPO-expert`, trained directly on the current
  engine, so the dimension mismatch (and any physics-transfer risk) doesn't
  apply; see the farama-minari paragraph below. `ant` doesn't have that
  escape hatch yet: `farama-minari/Ant-v5-SAC-expert` exists and looked
  promising (`mean_reward≈5846`) when we scouted it, but we haven't wired
  it in or verified it end to end.
- **Matching observation dims isn't enough: some zoo checkpoints don't
  survive the mujoco-py to mujoco physics transfer.** Two balance-sensitive
  bipedal checkpoints have correct observation dimensions but fall almost
  immediately on gymnasium's current native-mujoco engine (vs. the
  mujoco-py engine they were trained and evaluated on in 2022):
  - `walker2d` (`sb3/ppo-Walker2d-v3`, obs dims fixed via
    `exclude_current_positions_from_observation=False`): falls in ~64
    steps, mean return ≈ -3 to -5 even with the correct `VecNormalize`
    stats applied, verified directly against the pipeline's own
    `make_venv`/`load_expert`, not a normalization bug. Because of this,
    its `evaluate_policies.py`/`record_comparison_video.py` numbers aren't
    a meaningful "clone the expert" comparison (a barely-alive expert
    isn't a target worth cloning), so they're left out of the headline
    results; `data/results_walker2d.npz` and `media/results_walker2d.png`
    exist from testing but aren't part of them. Demo collection and the
    offline R² check still work fine for it.
  - `sac-Humanoid-v3` (the only public Humanoid-*walking* checkpoint whose
    observation dims match current gymnasium, 376 either way): scored
    `mean_reward≈6252` on its original setup but only `≈400` (falling
    within ~85 steps) here.

  Neither is a config problem we can fix with a kwarg: it's the physics
  engine itself producing different enough dynamics that these particular
  fragile locomotion gaits don't hold up. `humanoidstandup` sidesteps this
  for the Humanoid family by using
  `farama-minari/HumanoidStandup-v5-PPO-expert` instead, trained directly
  on the current `v5` engine (obs dims match exactly, `mean_reward≈128k`
  reproduced locally, full 1000-step episodes with no early termination),
  though its 17-dimensional action space (vs. 3-6 for the other
  environments) means 17 `TabPFNRegressor`s per step, and its
  348-dimensional observations are large enough that
  `TABPFN_MODEL_CACHE_SIZE` set too high caused an MPS out-of-memory error
  during testing (see the cache-size note above): expect this environment
  to run noticeably slower end to end than the others, and (per the R²
  numbers in the main README) to clone noticeably worse too. Once we knew
  `farama-minari` trains directly on the current engine, we used it for
  `swimmer` and `invertedpendulum` too, worth checking there first before
  assuming any missing/broken environment needs a from-scratch fix.
- **The Atari extension clones a fixed CNN embedding, not raw pixels.**
  This is a deliberate scope choice (see the main README's Atari section
  for why raw 28,224-dim pixels weren't viable), but it means TabPFN-BC
  still depends on the expert's own vision network at eval time: it isn't
  a vision-to-action policy on its own. A fully independent version would
  need its own feature extractor (e.g. a small pretrained/frozen CNN not
  tied to this specific expert), which is a bigger lift than swapping a
  registry entry.
- **Breakout exposes compounding error that short horizons hide.** An
  earlier version of this experiment capped every closed-loop episode at
  100 steps, and Breakout scored ~80% there, a seemingly solid fourth
  result. Once episodes ran to a real length (even truncated to 2,500
  steps, short of its true ~5,745-step average), that dropped to ~7%,
  despite Breakout having the *highest* offline action-prediction accuracy
  of the four games (94.2%). Good per-step accuracy but poor long-horizon
  return is the standard signature of compounding error in behavior
  cloning: small per-step mispredictions push the trajectory slightly
  outside the states TabPFN's context actually covers, and Breakout's
  precise, sustained paddle tracking punishes that drift far more than
  Pong, Space Invaders, or Ms. Pac-Man do. The standard fix in the
  imitation-learning literature is DAgger-style iterative re-labeling
  (query the expert on the clone's own visited states, add those to the
  context, refit) rather than one static context fit up front, worth
  trying here given TabPFN's in-context `.fit()` is cheap enough to redo
  repeatedly.

  **This doesn't generalize blindly, though, we checked.** MuJoCo's
  closed-loop episodes are capped at 200 (of HalfCheetah's default 1,000)
  steps purely for TabPFN's per-call-cost reasons, so we went back and ran
  HalfCheetah (6-dim actions, its weakest MuJoCo result) out to a full
  1,000-step episode expecting the same collapse. It didn't happen:
  `uv run python scripts/evaluate_policies.py --env halfcheetah --episode-len 1000`
  gave expert 5884.2 ± 50.8, TabPFN-BC 5119.5 ± 1814.1, **87.0%**, actually
  *better* than the 200-step run's ~83%, though with ~4.6x the absolute
  variance (relative std 35.4% vs. 48.6%, so not obviously worse
  proportionally either). Compounding error is real and decisive in
  Breakout; in HalfCheetah it isn't, at least not at this horizon. The
  likely reason: losing the ball in Breakout is a sharp, unrecoverable
  failure the instant the trajectory drifts, while a bad step in
  HalfCheetah's gait is just a rough stride: the episode keeps going and
  there's room to keep accumulating reward regardless. A genuine negative
  result for the hypothesis, reported as found rather than left untested.
