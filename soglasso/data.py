"""
Synthetic data generator reproducing the qualitative setup of Section 5.2
("Toy Data, Linear Regression") in Rao, Nowak, Cox, Rogers (2014).

Paper setup: overlapping groups of size B=6 sliding by shift=4 (adjacent
groups overlap: G1={0..5}, G2={4..9}, ...), M=100 groups, k=10 active groups,
within-group sparsity fraction alpha varied, n=100 Gaussian measurements,
additive noise sigma=0.1, coefficients in active/retained positions drawn
Uniform[-1, 1], repeated over 100 trials and averaged (Figure 6a).

For clarity and to keep the reference implementation self-contained, we
reproduce the SINGLE-TASK version of this experiment (the multitask case in
the paper is a straightforward stacking of this construction across tasks,
Section 5.1, Eq. group definition G = aggregated rows of X*).
"""
from __future__ import annotations
import numpy as np

from .groups import make_overlapping_chain_groups


def make_sog_regression(
    n_samples: int = 100,
    n_groups: int = 100,
    group_size: int = 6,
    shift: int = 4,
    k_active: int = 10,
    alpha: float = 0.5,
    noise_std: float = 0.1,
    random_state: int | None = None,
):
    """
    Generate a synthetic (k, l)-group-sparse regression problem with
    overlapping groups, matching the spirit of the paper's Section 5.2 setup.

    Parameters
    ----------
    n_samples : number of Gaussian measurements (rows of Phi).
    n_groups : number of overlapping groups to construct (controls p).
    group_size : size B of each group.
    shift : how much consecutive groups shift by (controls overlap amount).
    k_active : number of active (nonzero) groups, k <= n_groups.
    alpha : fraction of coefficients retained (nonzero) within each active
            group (this is the "l/L" sparsity-within-group parameter).
    noise_std : std. dev. of additive Gaussian measurement noise.
    random_state : int or None.

    Returns
    -------
    Phi : ndarray (n_samples, p)
    y : ndarray (n_samples,)
    x_star : ndarray (p,) ground-truth coefficient vector
    groups : list of ndarray, the overlapping group index sets used
    """
    rng = np.random.RandomState(random_state)

    # Build overlapping groups and infer ambient dimension p
    dummy_p = (n_groups - 1) * shift + group_size
    groups = make_overlapping_chain_groups(dummy_p, group_size, shift)[:n_groups]
    p = int(max(g.max() for g in groups)) + 1

    # Choose k_active groups at random to be "on"
    active_groups = rng.choice(len(groups), size=min(k_active, len(groups)), replace=False)

    x_star = np.zeros(p)
    for gi in active_groups:
        idx = groups[gi]
        n_retain = max(1, int(round(alpha * len(idx))))
        retain = rng.choice(idx, size=n_retain, replace=False)
        x_star[retain] = rng.uniform(-1, 1, size=n_retain)

    Phi = rng.randn(n_samples, p)
    y = Phi @ x_star + noise_std * rng.randn(n_samples)

    return Phi, y, x_star, groups
