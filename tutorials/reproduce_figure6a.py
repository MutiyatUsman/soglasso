"""
Tutorial: Reproducing Figure 6(a) of Rao, Nowak, Cox, Rogers (2014),
"Classification with Sparse Overlapping Groups" (arXiv:1402.4512).

The paper's Section 5.2 toy experiment varies the within-group sparsity
fraction alpha and compares held-out MSE for Lasso, Group Lasso (Glasso),
Overlapping Group Lasso (OGlasso), and SOGlasso. The reported qualitative
result: SOGlasso is best (or tied-best) across the full range of alpha;
OGlasso improves as alpha -> 1 (dense active groups) since it doesn't
penalize within-group sparsity; Lasso/Glasso ignore one axis of structure
each and are dominated.

This script reproduces that comparison on the single-task synthetic
generator in `soglasso.data.make_sog_regression`, with hyperparameters
chosen per-method via a small validation grid search (in place of the
paper's "clairvoyant" oracle tuning).

Run:
    python tutorials/reproduce_figure6a.py
Outputs:
    tutorials/figure6a_reproduction.png
    tutorials/figure6a_results.csv
"""
from __future__ import annotations
import csv
import itertools
import os

import numpy as np

from soglasso import (
    make_sog_regression,
    lasso,
    group_lasso,
    overlapping_group_lasso,
    sog_lasso,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- experiment configuration (mirrors paper's Sec 5.2 toy setup) --------
N_TRAIN = 100
N_VAL = 50
N_TEST = 200
N_GROUPS = 30
GROUP_SIZE = 6
SHIFT = 4
K_ACTIVE = 5
NOISE_STD = 0.1
N_TRIALS = 10  # paper uses 100 trials; reduced here to keep tutorial runtime short
ALPHAS = np.linspace(0.1, 1.0, 6)
ETA1_GRID = [0.1, 0.3, 0.5, 0.8, 1.2]
MU_GRID = [0.2, 0.5, 1.0]
MAX_ITER = 300
RIDGE = 1e-4


def _mse(model, Phi, y):
    return float(np.mean((model.predict(Phi) - y) ** 2))


def _fit_best(build_fn, param_grid, Phi_tr, y_tr, Phi_val, y_val):
    """Small grid search: pick hyperparams minimizing validation MSE."""
    best_mse, best_model = np.inf, None
    for params in param_grid:
        model = build_fn(**params)
        model.fit(Phi_tr, y_tr)
        mse = _mse(model, Phi_val, y_val)
        if mse < best_mse:
            best_mse, best_model = mse, model
    return best_model


def run_experiment():
    results = {name: {a: [] for a in ALPHAS}
               for name in ["Lasso", "Glasso", "OGlasso", "SOGlasso"]}

    for trial in range(N_TRIALS):
        for alpha in ALPHAS:
            Phi, y, x_star, groups = make_sog_regression(
                n_samples=N_TRAIN + N_VAL + N_TEST,
                n_groups=N_GROUPS, group_size=GROUP_SIZE, shift=SHIFT,
                k_active=K_ACTIVE, alpha=alpha, noise_std=NOISE_STD,
                random_state=1000 * trial + int(alpha * 100),
            )
            p = Phi.shape[1]
            Phi_tr, Phi_val, Phi_te = (Phi[:N_TRAIN],
                                        Phi[N_TRAIN:N_TRAIN + N_VAL],
                                        Phi[N_TRAIN + N_VAL:])
            y_tr, y_val, y_te = (y[:N_TRAIN],
                                  y[N_TRAIN:N_TRAIN + N_VAL],
                                  y[N_TRAIN + N_VAL:])

            # Lasso
            grid = [{"p": p, "eta1": e, "eta2": RIDGE, "max_iter": MAX_ITER}
                    for e in ETA1_GRID]
            m = _fit_best(lasso, grid, Phi_tr, y_tr, Phi_val, y_val)
            results["Lasso"][alpha].append(_mse(m, Phi_te, y_te))

            # Group Lasso (non-overlapping groups of same size)
            grid = [{"p": p, "group_size": GROUP_SIZE, "eta1": e,
                     "eta2": RIDGE, "max_iter": MAX_ITER} for e in ETA1_GRID]
            m = _fit_best(group_lasso, grid, Phi_tr, y_tr, Phi_val, y_val)
            results["Glasso"][alpha].append(_mse(m, Phi_te, y_te))

            # Overlapping Group Lasso (mu=0, same overlapping groups as SOG)
            grid = [{"overlapping_groups": groups, "eta1": e, "eta2": RIDGE,
                     "max_iter": MAX_ITER} for e in ETA1_GRID]
            m = _fit_best(overlapping_group_lasso, grid, Phi_tr, y_tr, Phi_val, y_val)
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

    csv_path = os.path.join(HERE, "figure6a_results.csv")
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
    plt.title("Reproduction of Figure 6(a): SOGlasso toy regression")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(HERE, "figure6a_reproduction.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    res = run_experiment()
    save_and_plot(res)
