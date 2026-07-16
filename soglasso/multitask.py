"""
Multitask construction, following Section 5.1 of Rao, Nowak, Cox & Rogers
(2014) EXACTLY (not the single-task stand-in used in `data.py`).

Quoting the paper's own description of how the multitask problem is reduced
to the standard single-vector form used throughout the rest of the paper:

    "In the multitask learning setting, suppose the features are given by
    Phi_t, for tasks t = {1, ..., T}, and corresponding sparse vectors
    x*_t in R^p. These vectors can be arranged as columns of a matrix X*.
    Suppose we are now given M groups G~ = {G~_1, G~_2, ...} with maximum
    size B~. Note that the groups will now correspond to sets of ROWS of X*.

    Let x* = [x*_1^T x*_2^T ... x*_T^T]^T in R^{Tp}, and
    y = [y_1^T y_2^T ... y_T^T]^T in R^{Tn}. We also define
    G = {G_1, ..., G_M} to be the set of groups defined on R^{Tp} formed by
    AGGREGATING THE ROWS of X that were originally in G~, so that x is
    composed of groups G in G, and let the corresponding maximum group size
    be B = T * B~."

In words: a "feature group" (e.g. an anatomical brain region, or a gene
pathway) is defined ONCE, over the p base features. Every task gets its own
sparse coefficient vector, but the SAME k groups are active across all
tasks -- modeling the fMRI motivation that the same brain regions matter
for all subjects, while the exact voxels used can differ per subject
(within-group sparsity pattern is independent per task).

Section 5.2's toy experiment settings: T=20 tasks, overlapping groups of
size B~=6 (sliding by shift), M=100 groups, k=10 active groups, m=100
Gaussian measurements per task, noise std=0.1, repeated over 100 trials.
"""
from __future__ import annotations
import numpy as np

from .groups import make_overlapping_chain_groups


def build_multitask_groups(base_groups: list[np.ndarray], T: int, p: int) -> list[np.ndarray]:
    """
    Aggregate base (per-task) feature groups into groups over the stacked
    T*p coefficient vector x = [x_1; x_2; ...; x_T], exactly as described
    in Section 5.1: group G_k in the aggregated space contains, for every
    task t, the same base feature indices G~_k, offset into that task's
    block.

    Parameters
    ----------
    base_groups : list of index arrays over the base p-dim feature space.
    T : number of tasks.
    p : number of base features (per task).

    Returns
    -------
    list of index arrays over the aggregated T*p-dim space.
    """
    agg_groups = []
    for g in base_groups:
        idx = np.concatenate([g + t * p for t in range(T)])
        agg_groups.append(idx)
    return agg_groups


def make_multitask_sog_regression(
    T: int = 20,
    n_per_task: int = 100,
    n_groups: int = 100,
    group_size: int = 6,
    shift: int = 4,
    k_active: int = 10,
    alpha: float = 0.5,
    noise_std: float = 0.1,
    random_state: int | None = None,
):
    """
    Generate the ACTUAL multitask (k, l)-group-sparse regression problem of
    Section 5.1/5.2, reduced to the standard stacked-vector form of Eq. (5)
    used everywhere else in the paper.

    The k active groups are the SAME across all T tasks (shared macro-
    structure). Within each active group, each task independently retains a
    fraction `alpha` of coefficients at random (different micro-structure
    per task) -- this is exactly the fMRI motivation in Figure 2: same
    anatomical region active across subjects, different voxels within it.

    Returns
    -------
    Phi_big : ndarray, shape (T * n_per_task, T * p)
        Block-diagonal design matrix: Phi_big = blkdiag(Phi_1, ..., Phi_T).
    y_big : ndarray, shape (T * n_per_task,)
        Stacked measurements y = [y_1; ...; y_T].
    x_star_big : ndarray, shape (T * p,)
        Stacked ground-truth coefficients x* = [x*_1; ...; x*_T].
    groups : list of ndarray
        Overlapping groups over the aggregated T*p space (Section 5.1
        construction: same base group, replicated/offset into every task's
        block).
    p : int
        Base (per-task) feature dimension.
    """
    rng = np.random.RandomState(random_state)

    # Base (per-task) overlapping groups over p features
    dummy_p = (n_groups - 1) * shift + group_size
    base_groups = make_overlapping_chain_groups(dummy_p, group_size, shift)[:n_groups]
    p = int(max(g.max() for g in base_groups)) + 1

    # k active groups, SHARED across all T tasks (Section 5.1 macrostructure)
    active_groups = rng.choice(len(base_groups), size=min(k_active, len(base_groups)),
                                replace=False)

    # Per-task sparse coefficient vectors: same active groups, independently
    # sparse WITHIN each group per task (Section 5.1 microstructure)
    X_star = np.zeros((p, T))  # columns are x*_t, matching paper's "X*" matrix
    for t in range(T):
        for gi in active_groups:
            idx = base_groups[gi]
            n_retain = max(1, int(round(alpha * len(idx))))
            retain = rng.choice(idx, size=n_retain, replace=False)
            X_star[retain, t] = rng.uniform(-1, 1, size=n_retain)

    x_star_big = X_star.T.reshape(-1)  # [x*_1; x*_2; ...; x*_T], each block length p

    # Block-diagonal design: independent Gaussian design per task
    Phi_blocks = [rng.randn(n_per_task, p) for _ in range(T)]
    Phi_big = np.zeros((T * n_per_task, T * p))
    for t, Phi_t in enumerate(Phi_blocks):
        Phi_big[t * n_per_task:(t + 1) * n_per_task, t * p:(t + 1) * p] = Phi_t

    y_big = Phi_big @ x_star_big + noise_std * rng.randn(T * n_per_task)

    groups = build_multitask_groups(base_groups, T, p)

    return Phi_big, y_big, x_star_big, groups, p
