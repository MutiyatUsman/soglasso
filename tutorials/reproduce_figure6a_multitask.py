"""
Tutorial: Reproducing Figure 6(a) using the ACTUAL multitask construction
of Section 5.1 (not the single-task stand-in in `reproduce_figure6a.py`).

Paper's exact Section 5.2 setup: T=20 tasks, overlapping groups of size
B~=6 (sliding by 4), M=100 groups, k=10 active groups (SAME across all
tasks), m=100 Gaussian measurements per task, noise std=0.1, averaged over
100 trials, regularization "clairvoyantly picked to minimize MSE."

This script uses the same T=20 (task count matches the paper exactly,
since that's the structural element that matters most -- it's what makes
the "shared macrostructure, independent microstructure" story possible),
but reduces n_groups/n_per_task/trial count/grid density for tractable
runtime (a full paper-scale sweep with proper per-method grid search would
take hours; see module docstring in `soglasso/multitask.py` for how to run
it at full scale if you have the compute budget).

Run:
    python tutorials/reproduce_figure6a_multitask.py
Outputs:
    tutorials/figure6a_multitask_reproduction.png
    tutorials/figure6a_multitask_results.csv
"""
from __future__ import annotations
import csv
import itertools
import os

import numpy as np

from soglasso.multitask import make_multitask_sog_regression
from soglasso import lasso, group_lasso, overlapping_group_lasso, sog_lasso

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- experiment configuration ---------------------------------------------
# T matches the paper exactly (T=20); other sizes reduced for tractable
# runtime while preserving the true multitask structure.
T = 20
N_PER_TASK_TRAIN = 40
N_PER_TASK_VAL = 20
N_PER_TASK_TEST = 40
N_GROUPS = 40          # paper: 100
GROUP_SIZE = 6         # paper: 6 (matches)
SHIFT = 4              # paper: 4 (matches)
K_ACTIVE = 6            # paper: 10
NOISE_STD = 0.1         # paper: 0.1 (matches)
N_TRIALS = 2             # paper: 100
ALPHAS = np.linspace(0.1, 1.0, 4)
ETA1_GRID = [0.3, 0.6, 1.0]
MU_GRID = [0.05, 0.2, 0.5]
MAX_ITER = 250
RIDGE = 1e-4


def _mse(model, Phi, y):
    return float(np.mean((model.predict(Phi) - y) ** 2))


def _fit_best(build_fn, param_grid, Phi_tr, y_tr, Phi_val, y_val):
    best_mse, best_model = np.inf, None
    for params in param_grid:
        model = build_fn(**params)
        model.fit(Phi_tr, y_tr)
        mse = _mse(model, Phi_val, y_val)
        if mse < best_mse:
            best_mse, best_model = mse, model
    return best_model


def _make_split_multitask(alpha, seed):
    """
    Generate one big multitask problem with enough samples per task to
    split into train/val/test, using the SAME sparsity pattern (x*) and
    group structure across the split (only the design/measurements differ
    per split), matching how the paper evaluates held-out MSE.
    """
    n_total = N_PER_TASK_TRAIN + N_PER_TASK_VAL + N_PER_TASK_TEST
    Phi, y, x_star, groups, p = make_multitask_sog_regression(
        T=T, n_per_task=n_total, n_groups=N_GROUPS, group_size=GROUP_SIZE,
        shift=SHIFT, k_active=K_ACTIVE, alpha=alpha, noise_std=NOISE_STD,
        random_state=seed,
    )
    # Phi is block-diagonal per task; split each task's block into
    # train/val/test slices and reassemble as block-diagonal matrices.
    def slice_blocks(start, end):
        n_slice = end - start
        Phi_s = np.zeros((T * n_slice, T * p))
        for t in range(T):
            block = Phi[t * n_total + start: t * n_total + end, t * p:(t + 1) * p]
            Phi_s[t * n_slice:(t + 1) * n_slice, t * p:(t + 1) * p] = block
        y_s = np.concatenate([
            y[t * n_total + start: t * n_total + end] for t in range(T)
        ])
        return Phi_s, y_s

    Phi_tr, y_tr = slice_blocks(0, N_PER_TASK_TRAIN)
    Phi_val, y_val = slice_blocks(N_PER_TASK_TRAIN, N_PER_TASK_TRAIN + N_PER_TASK_VAL)
    Phi_te, y_te = slice_blocks(N_PER_TASK_TRAIN + N_PER_TASK_VAL, n_total)

    return Phi_tr, y_tr, Phi_val, y_val, Phi_te, y_te, x_star, groups, p


def run_experiment():
    results = {name: {a: [] for a in ALPHAS}
               for name in ["Lasso", "Glasso", "OGlasso", "SOGlasso"]}

    for trial in range(N_TRIALS):
        for alpha in ALPHAS:
            seed = 1000 * trial + int(alpha * 100)
            Phi_tr, y_tr, Phi_val, y_val, Phi_te, y_te, x_star, groups, p = \
                _make_split_multitask(alpha, seed)

            # Lasso: singleton groups over the SAME aggregated Tp space
            singleton_groups = [np.array([i]) for i in range(T * p)]
            grid = [{"overlapping_groups": singleton_groups, "mu": 0.0,
                     "eta1": e, "eta2": RIDGE, "max_iter": MAX_ITER}
                    for e in ETA1_GRID]
            m = _fit_best(sog_lasso, grid, Phi_tr, y_tr, Phi_val, y_val)
            results["Lasso"][alpha].append(_mse(m, Phi_te, y_te))

            # Group Lasso: non-overlapping groups over base p features,
            # aggregated the same way (mu=0, base groups don't overlap)
            from soglasso.groups import make_disjoint_groups
            from soglasso.multitask import build_multitask_groups
            base_disjoint = make_disjoint_groups(p, GROUP_SIZE)
            glasso_groups = build_multitask_groups(base_disjoint, T, p)
            grid = [{"overlapping_groups": glasso_groups, "mu": 0.0,
                     "eta1": e, "eta2": RIDGE, "max_iter": MAX_ITER}
                    for e in ETA1_GRID]
            m = _fit_best(sog_lasso, grid, Phi_tr, y_tr, Phi_val, y_val)
            results["Glasso"][alpha].append(_mse(m, Phi_te, y_te))

            # Overlapping Group Lasso: mu=0, same overlapping groups as SOG
            grid = [{"overlapping_groups": groups, "mu": 0.0, "eta1": e,
                     "eta2": RIDGE, "max_iter": MAX_ITER} for e in ETA1_GRID]
            m = _fit_best(sog_lasso, grid, Phi_tr, y_tr, Phi_val, y_val)
            results["OGlasso"][alpha].append(_mse(m, Phi_te, y_te))

            # SOGlasso
            grid = [{"overlapping_groups": groups, "mu": mu, "eta1": e,
                     "eta2": RIDGE, "max_iter": MAX_ITER}
                    for e, mu in itertools.product(ETA1_GRID, MU_GRID)]
            m = _fit_best(sog_lasso, grid, Phi_tr, y_tr, Phi_val, y_val)
            results["SOGlasso"][alpha].append(_mse(m, Phi_te, y_te))

        print(f"trial {trial + 1}/{N_TRIALS} done")

    return results


def save_and_plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = os.path.join(HERE, "figure6a_multitask_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "alpha", "mean_mse", "std_mse"])
        for name, per_alpha in results.items():
            for a in ALPHAS:
                vals = per_alpha[a]
                writer.writerow([name, a, np.mean(vals), np.std(vals)])
    print(f"Saved results table to {csv_path}")

    plt.figure(figsize=(6, 4.5))
    markers = {"Lasso": "o", "Glasso": "s", "OGlasso": "^", "SOGlasso": "d"}
    for name, per_alpha in results.items():
        means = [np.mean(per_alpha[a]) for a in ALPHAS]
        plt.plot(ALPHAS, means, marker=markers[name], label=name)
    plt.xlabel(r"$\alpha$ (within-group sparsity fraction)")
    plt.ylabel("Held-out MSE")
    plt.title("Figure 6(a), multitask construction (T=20 tasks)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(HERE, "figure6a_multitask_reproduction.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    res = run_experiment()
    save_and_plot(res)
