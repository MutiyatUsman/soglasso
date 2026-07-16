"""
Proximal operators for the Sparse Overlapping Group (SOG) Lasso.

Implements the composition of soft-thresholding and group soft-thresholding
described in Section 3.2 / Eq. 16 of:

    Rao, Nowak, Cox, Rogers. "Classification with Sparse Overlapping Groups."
    arXiv:1402.4512

    w_tilde        = sign(w_grad) * max(|w_grad| - eta1*mu, 0)
    (w_{t+1})_G    = w_tilde_G / ||w_tilde_G|| * max(||w_tilde_G|| - eta1, 0)
                     if ||w_tilde_G|| != 0, else 0

This is the proximal operator of
    h_G(w) = ||w||_2 + mu * ||w||_1
applied block-wise to a (non-overlapping, replicated) group structure.
"""
from __future__ import annotations
import numpy as np


def soft_threshold(w: np.ndarray, thresh: float) -> np.ndarray:
    """Elementwise soft-thresholding: sign(w) * max(|w| - thresh, 0)."""
    return np.sign(w) * np.maximum(np.abs(w) - thresh, 0.0)


def group_soft_threshold(w: np.ndarray, thresh: float) -> np.ndarray:
    """L2 (block) soft-thresholding of a single group vector w."""
    norm = np.linalg.norm(w)
    if norm <= thresh or norm == 0.0:
        return np.zeros_like(w)
    return (w / norm) * (norm - thresh)


def sog_prox(w: np.ndarray, group_slices, eta1: float, mu: float) -> np.ndarray:
    """
    Proximal operator of eta1 * sum_G (||w_G||_2 + mu ||w_G||_1),
    for NON-overlapping groups given as a list of slice objects / index arrays
    into w (this is the replicated/expanded space where groups are disjoint).

    Parameters
    ----------
    w : ndarray, shape (p_expanded,)
        Point at which to evaluate the proximal operator (e.g. a gradient step).
    group_slices : list of array-like of int
        Indices belonging to each (non-overlapping) group in the expanded space.
    eta1 : float
        Overall regularization step size (group L2 threshold).
    mu : float
        Within-group L1/L2 tradeoff (this is lambda_1/sqrt(l) in the paper).

    Returns
    -------
    out : ndarray, same shape as w
    """
    out = np.zeros_like(w)
    # Step 1: elementwise soft-threshold by eta1*mu (the L1 part)
    w_tilde = soft_threshold(w, eta1 * mu)
    # Step 2: group soft-threshold each block by eta1 (the L2 part)
    for idx in group_slices:
        out[idx] = group_soft_threshold(w_tilde[idx], eta1)
    return out
