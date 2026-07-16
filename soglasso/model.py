"""
Sparse Overlapping Group (SOG) Lasso estimator.

Solves the Lagrangian problem given in Section 3.2 of Rao, Nowak, Cox, Rogers
(2014), "Classification with Sparse Overlapping Groups" (arXiv:1402.4512):

    xhat = argmin_x  L(x; Phi, y)  +  eta1 * h(x)  +  eta2 * ||x||^2

where h(x) = inf_{w_G} sum_G ( ||w_G||_2 + mu ||w_G||_1 ),  mu = lambda1/sqrt(l),
and L is either the squared loss (regression) or the linear classification loss
sum_i -y_i <phi_i, x> used in the paper's estimator (Eq. 5).

Overlapping groups are handled via the covariate-duplication / replication
strategy of Jacob, Obozinski & Vert (2009), exactly as described by the paper
in Section 3.2 ("We use the 'covariate duplication' method... to first reduce
the problem to the non overlapping sparse group lasso in an expanded space.
One can then use proximal methods to recover the coefficients.").

Special cases (see paper Section 2.1 "Remarks" and Section 3, "SOGlasso
Penalty" remarks) obtained from this same estimator:
    - mu = 0, singleton groups            -> standard Lasso
    - mu = 0, non-overlapping groups      -> standard Group Lasso (Glasso)
    - mu = 0, overlapping groups          -> (latent/overlapping) Group Lasso
    - mu > 0, overlapping groups          -> SOGlasso (general case)
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .proximal import sog_prox
from .groups import build_replication, collapse_to_original


class SOGLasso(BaseEstimator, RegressorMixin):
    """
    Parameters
    ----------
    groups : list of 1D int arrays
        Groups over the original p features. May overlap arbitrarily.
    eta1 : float
        Overall regularization strength (multiplies h(x)).
    mu : float
        Within-group L1/L2 tradeoff (mu = lambda1/sqrt(l) in the paper).
        mu=0 recovers the (overlapping) group lasso.
    eta2 : float
        Ridge strength on the expanded coefficients (paper's eta2||x||^2 term;
        applied in expanded space here for a fully separable proximal step,
        a standard simplification of the replication approach).
    loss : {'squared', 'linear'}
        'squared' -> regression: (1/2)||y - Phi x||^2 (used for the toy
                     multitask/regression experiments in the paper, Sec 5.2).
        'linear'  -> classification objective of Eq. 5 in the paper:
                     sum_i -y_i <phi_i, x>  (y in {-1,+1}).
    max_iter, tol : optimization controls for FISTA.
    """

    def __init__(self, groups, eta1=0.1, mu=0.5, eta2=1e-4,
                 loss="squared", max_iter=500, tol=1e-6):
        self.groups = groups
        self.eta1 = eta1
        self.mu = mu
        self.eta2 = eta2
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, Phi: np.ndarray, y: np.ndarray):
        n, p = Phi.shape
        expand_idx, group_slices, p_exp = build_replication(p, self.groups)
        Phi_dup = Phi[:, expand_idx]  # (n, p_exp)

        # Lipschitz constant of the smooth part's gradient (largest eigenvalue
        # of Phi_dup^T Phi_dup), via power iteration for efficiency.
        L = _spectral_norm_sq(Phi_dup) + 2 * self.eta2
        step = 1.0 / max(L, 1e-12)

        w = np.zeros(p_exp)
        z = w.copy()
        t = 1.0

        for _ in range(self.max_iter):
            grad = self._grad(Phi_dup, y, z)
            w_new = sog_prox(z - step * grad, group_slices,
                              self.eta1 * step, self.mu)
            t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            z = w_new + ((t - 1) / t_new) * (w_new - w)

            if np.linalg.norm(w_new - w) < self.tol * (1 + np.linalg.norm(w)):
                w = w_new
                break
            w, t = w_new, t_new

        self.coef_expanded_ = w
        self.coef_ = collapse_to_original(w, expand_idx, p)
        self._expand_idx = expand_idx
        return self

    def _grad(self, Phi_dup, y, w):
        if self.loss == "squared":
            resid = Phi_dup @ w - y
            return Phi_dup.T @ resid + 2 * self.eta2 * w
        elif self.loss == "linear":
            # gradient of sum_i -y_i <phi_i, x> w.r.t expanded w is constant:
            # -Phi_dup^T y, plus ridge gradient.
            return -(Phi_dup.T @ y) + 2 * self.eta2 * w
        else:
            raise ValueError(f"Unknown loss '{self.loss}'")

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        return Phi @ self.coef_


def _spectral_norm_sq(A: np.ndarray, n_iter: int = 50) -> float:
    """Power iteration estimate of the largest eigenvalue of A^T A."""
    n, p = A.shape
    v = np.random.RandomState(0).randn(p)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(n_iter):
        v = A.T @ (A @ v)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            return 1.0
        v /= norm
    return float(norm)
