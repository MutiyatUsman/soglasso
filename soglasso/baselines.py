"""
Baseline penalties, all expressed as special cases of the same SOGLasso
estimator -- exactly as noted in the paper's Remarks (Section 2.1 and
Section 3, "SOGlasso Penalty"):

    - mu = 0, singleton groups       -> standard Lasso
    - mu = 0, non-overlapping groups -> standard Group Lasso (Glasso)
    - mu = 0, overlapping groups     -> (latent) Overlapping Group Lasso (OGlasso)
    - mu > 0, overlapping groups     -> SOGlasso
"""
from __future__ import annotations
import numpy as np

from .model import SOGLasso
from .groups import make_singleton_groups, make_disjoint_groups


def lasso(p: int, **kwargs) -> SOGLasso:
    return SOGLasso(groups=make_singleton_groups(p), mu=0.0, **kwargs)


def group_lasso(p: int, group_size: int, **kwargs) -> SOGLasso:
    return SOGLasso(groups=make_disjoint_groups(p, group_size), mu=0.0, **kwargs)


def overlapping_group_lasso(overlapping_groups, **kwargs) -> SOGLasso:
    """AKA latent group lasso (Jacob et al., 2009); mu=0 on overlapping groups."""
    return SOGLasso(groups=overlapping_groups, mu=0.0, **kwargs)


def sog_lasso(overlapping_groups, mu: float = 0.5, **kwargs) -> SOGLasso:
    return SOGLasso(groups=overlapping_groups, mu=mu, **kwargs)
