"""
Covariate-duplication ("replication") strategy for overlapping groups.

Following Jacob, Obozinski & Vert (2009), as referenced in the SOGlasso paper
(Section 3.2): to handle overlapping groups G = {G_1, ..., G_K} over p
features, we build an expanded design matrix Phi_expanded in R^{n x sum|G_k|}
where each group gets its own private copy of the features it contains.
Solving the non-overlapping (group-)lasso in this expanded space and then
SUMMING duplicated coefficients back together recovers a solution to the
original overlapping-group problem (Lemma 9 / Eq. 7-8 in the paper: x is
recovered as the sum over G of w_G).
"""
from __future__ import annotations
import numpy as np


def build_replication(p: int, groups: list[np.ndarray]):
    """
    Build the index map needed to replicate features according to `groups`.

    Parameters
    ----------
    p : int
        Original ambient dimension.
    groups : list of 1D int arrays
        groups[k] = array of original feature indices in group k. Groups may
        overlap arbitrarily.

    Returns
    -------
    expand_idx : ndarray, shape (p_expanded,)
        expand_idx[j] = original feature index that expanded column j maps to.
        So Phi_expanded = Phi[:, expand_idx].
    group_slices : list of ndarray
        group_slices[k] = indices (into the expanded space) belonging to group k.
    p_expanded : int
    """
    expand_idx = []
    group_slices = []
    offset = 0
    for g in groups:
        g = np.asarray(g, dtype=int)
        group_slices.append(np.arange(offset, offset + len(g)))
        expand_idx.append(g)
        offset += len(g)
    expand_idx = np.concatenate(expand_idx) if expand_idx else np.array([], dtype=int)
    return expand_idx, group_slices, offset


def collapse_to_original(w_expanded: np.ndarray, expand_idx: np.ndarray, p: int) -> np.ndarray:
    """
    Sum duplicated coefficients back into the original p-dimensional space.
    x = sum_G w_G  (Eq. 7 in the paper: elements of W(x) sum to x).
    """
    x = np.zeros(p)
    np.add.at(x, expand_idx, w_expanded)
    return x


def make_overlapping_chain_groups(p: int, group_size: int, shift: int) -> list[np.ndarray]:
    """
    Build "sliding window" overlapping groups over p features:
    G1 = {0, ..., group_size-1}, G2 = {shift, ..., shift+group_size-1}, ...
    matching the style used in the paper's toy experiment (Section 5.2).
    """
    groups = []
    start = 0
    while start < p:
        end = min(start + group_size, p)
        groups.append(np.arange(start, end))
        if end == p:
            break
        start += shift
    return groups


def make_disjoint_groups(p: int, group_size: int) -> list[np.ndarray]:
    """Non-overlapping consecutive groups of size `group_size` (Glasso case)."""
    return [np.arange(i, min(i + group_size, p)) for i in range(0, p, group_size)]


def make_singleton_groups(p: int) -> list[np.ndarray]:
    """Each feature is its own group of size 1 -> reduces penalty to plain L1 (Lasso)."""
    return [np.array([i]) for i in range(p)]
