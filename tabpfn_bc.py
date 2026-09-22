"""Behavior cloning with TabPFN-3.5 (open-weight, local, Fast checkpoint):
fit one TabPFNRegressor per action dimension on a context of (state, action)
pairs from the PPO expert, then evaluate on held-out states via batched
in-context prediction (see README for why batched, not step-by-step
rollout, is how we evaluate this).
"""
import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

N_CONTEXT = 1000
N_TEST = 2000
SEED = 0


def main():
    data = np.load("demos_halfcheetah.npz")
    states, actions = data["states"], data["actions"]
    n_dims = actions.shape[1]
    print(f"total transitions: {len(states)}, action dims: {n_dims}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(states))
    ctx_idx = perm[:N_CONTEXT]
    test_idx = perm[N_CONTEXT : N_CONTEXT + N_TEST]

    X_ctx, y_ctx = states[ctx_idx], actions[ctx_idx]
    X_test, y_test = states[test_idx], actions[test_idx]

    preds = np.zeros_like(y_test)
    fit_times, predict_times = [], []

    for dim in range(n_dims):
        reg = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5_FAST)
        t0 = time.time()
        reg.fit(X_ctx, y_ctx[:, dim])
        t1 = time.time()
        preds[:, dim] = reg.predict(X_test)
        t2 = time.time()
        fit_times.append(t1 - t0)
        predict_times.append(t2 - t1)
        print(f"dim {dim}: fit={t1-t0:.1f}s, predict({N_TEST} pts)={t2-t1:.1f}s")

    print(f"\ntotal fit time: {sum(fit_times):.1f}s, total predict time: {sum(predict_times):.1f}s")

    print("\n--- per-dimension behavior-cloning accuracy (held-out states) ---")
    r2s, mses = [], []
    for dim in range(n_dims):
        r2 = r2_score(y_test[:, dim], preds[:, dim])
        mse = mean_squared_error(y_test[:, dim], preds[:, dim])
        r2s.append(r2)
        mses.append(mse)
        print(f"action dim {dim}: R2={r2:.3f}, MSE={mse:.4f}")

    print(f"\nmean R2 across dims: {np.mean(r2s):.3f}")
    print(f"mean MSE across dims: {np.mean(mses):.4f}")

    np.savez(
        "tabpfn_bc_results.npz",
        y_test=y_test,
        preds=preds,
        r2s=r2s,
        mses=mses,
        fit_times=fit_times,
        predict_times=predict_times,
        n_context=N_CONTEXT,
        n_test=N_TEST,
    )
    print("saved tabpfn_bc_results.npz")


if __name__ == "__main__":
    main()
