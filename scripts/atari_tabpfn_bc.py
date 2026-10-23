"""Offline check: fits a single TabPFNClassifier on a context subsample of
(CNN-embedding, discrete action) pairs, evaluates held-out action-prediction
accuracy against a majority-class baseline. No environment interaction.

    uv run python scripts/atari_tabpfn_bc.py --env pong
"""
import argparse
import os
import time
from pathlib import Path

# Avoids rebuilding the transformer from scratch on repeated TabPFN model
# loads -- must be set before any TabPFN model load.
os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "1")

import numpy as np
from dotenv import load_dotenv
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from atari_envs import DATA_DIR, SUPPORTED_ATARI_ENVIRONMENTS

load_dotenv()

N_CONTEXT = 1000
N_TEST = 500
TEST_BATCH = 50  # TabPFNClassifier.predict() batched to avoid MPS OOM on 512-dim features
SEED = 0


def predict_batched(clf, X, batch_size=TEST_BATCH):
    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(clf.predict(X[i:i + batch_size]))
    return np.concatenate(preds)


def main(env_key):
    data = np.load(DATA_DIR / f"atari_demos_{env_key}.npz")
    states, actions = data["states"], data["actions"]
    print(f"[{env_key}] total transitions: {len(states)}, embedding dim: {states.shape[1]}, "
          f"n_actions: {len(np.unique(actions))}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(states))
    ctx_idx = perm[:N_CONTEXT]
    test_idx = perm[N_CONTEXT:N_CONTEXT + N_TEST]
    X_ctx, y_ctx = states[ctx_idx], actions[ctx_idx]
    X_test, y_test = states[test_idx], actions[test_idx]

    clf = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5_FAST)
    t0 = time.time()
    clf.fit(X_ctx, y_ctx)
    t1 = time.time()
    preds = predict_batched(clf, X_test)
    t2 = time.time()
    print(f"[{env_key}] fit={t1-t0:.1f}s, predict({len(X_test)} pts)={t2-t1:.1f}s")

    acc = (preds == y_test).mean()
    majority_acc = (y_test == np.bincount(y_ctx).argmax()).mean()
    print(f"[{env_key}] held-out action accuracy: {acc:.3f} (majority-class baseline: {majority_acc:.3f})")

    out_path = DATA_DIR / f"atari_tabpfn_bc_results_{env_key}.npz"
    np.savez(
        out_path,
        y_test=y_test, preds=preds, accuracy=acc, majority_baseline=majority_acc,
        n_context=N_CONTEXT, n_test=len(X_test),
    )
    print(f"[{env_key}] saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=SUPPORTED_ATARI_ENVIRONMENTS, default="pong")
    args = parser.parse_args()
    main(args.env)
