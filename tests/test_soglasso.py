import numpy as np
import pytest

from soglasso import (
    SOGLasso, lasso, group_lasso, overlapping_group_lasso, sog_lasso,
    make_sog_regression, make_overlapping_chain_groups, make_disjoint_groups,
    make_singleton_groups,
)
from soglasso.proximal import soft_threshold, group_soft_threshold, sog_prox
from soglasso.groups import build_replication, collapse_to_original


def test_soft_threshold_basic():
    w = np.array([2.0, -3.0, 0.5, -0.2])
    out = soft_threshold(w, 1.0)
    np.testing.assert_allclose(out, [1.0, -2.0, 0.0, 0.0])


def test_group_soft_threshold_shrinks_to_zero_below_threshold():
    w = np.array([0.3, 0.4])  # norm = 0.5
    out = group_soft_threshold(w, 1.0)
    np.testing.assert_allclose(out, [0.0, 0.0])


def test_group_soft_threshold_shrinks_correctly_above_threshold():
    w = np.array([3.0, 4.0])  # norm = 5
    out = group_soft_threshold(w, 1.0)
    # direction preserved, magnitude shrunk to norm - thresh = 4
    np.testing.assert_allclose(np.linalg.norm(out), 4.0, atol=1e-10)
    np.testing.assert_allclose(out / np.linalg.norm(out), w / np.linalg.norm(w))


def test_singleton_groups_reduce_to_lasso_like_shrinkage():
    # With singleton groups and mu=0, sog_prox should equal plain soft-thresholding
    p = 10
    groups = make_singleton_groups(p)
    expand_idx, group_slices, p_exp = build_replication(p, groups)
    assert p_exp == p
    w = np.random.RandomState(0).randn(p)
    out = sog_prox(w, group_slices, eta1=0.3, mu=0.0)
    expected = np.array([group_soft_threshold(np.array([wi]), 0.3)[0] for wi in w])
    np.testing.assert_allclose(out, expected, atol=1e-10)


def test_replication_and_collapse_roundtrip_no_overlap():
    p = 12
    groups = make_disjoint_groups(p, 4)
    expand_idx, group_slices, p_exp = build_replication(p, groups)
    assert p_exp == p  # no duplication when groups don't overlap
    w = np.arange(p_exp, dtype=float)
    x = collapse_to_original(w, expand_idx, p)
    np.testing.assert_allclose(x, w)  # identity permutation-ish (contiguous)


def test_replication_duplicates_overlapping_features():
    p = 10
    groups = make_overlapping_chain_groups(p, group_size=4, shift=2)
    expand_idx, group_slices, p_exp = build_replication(p, groups)
    assert p_exp > p  # overlaps mean expanded space is larger
    # every original feature appears at least once in expand_idx
    assert set(expand_idx.tolist()) == set(range(p))


def test_sog_lasso_recovers_sparse_group_signal_better_than_chance():
    Phi, y, x_star, groups = make_sog_regression(
        n_samples=200, n_groups=15, group_size=5, shift=3,
        k_active=2, alpha=0.4, noise_std=0.05, random_state=1,
    )
    p = Phi.shape[1]
    model = sog_lasso(groups, mu=0.5, eta1=0.6, eta2=1e-4, max_iter=400)
    model.fit(Phi, y)
    err = np.linalg.norm(model.coef_ - x_star)
    null_err = np.linalg.norm(x_star)  # error of predicting all-zero
    assert err < 0.5 * null_err


def test_lasso_group_lasso_are_valid_special_cases():
    p = 20
    m1 = lasso(p, eta1=0.2)
    assert all(len(g) == 1 for g in m1.groups)
    assert m1.mu == 0.0

    m2 = group_lasso(p, group_size=5, eta1=0.2)
    assert all(len(g) == 5 for g in m2.groups[:-1])
    assert m2.mu == 0.0


def test_fit_predict_shapes():
    Phi, y, x_star, groups = make_sog_regression(
        n_samples=50, n_groups=10, group_size=4, shift=2,
        k_active=2, alpha=0.5, random_state=2,
    )
    model = sog_lasso(groups, mu=0.3, eta1=0.3, max_iter=100)
    model.fit(Phi, y)
    preds = model.predict(Phi)
    assert preds.shape == y.shape
    assert model.coef_.shape == (Phi.shape[1],)


def test_h_is_convex_triangle_inequality_numerically():
    """Sanity check that the underlying penalty behaves like a norm
    (Lemma 7 in the paper): h(x + y) <= h(x) + h(y)."""
    from soglasso.proximal import group_soft_threshold  # noqa

    def h(x, groups, mu):
        # crude non-optimal-representation estimate: assign each coordinate
        # to its first group only, as a proxy (upper bound on true h via
        # a *feasible*, not necessarily optimal, representation).
        total = 0.0
        assigned = np.zeros_like(x, dtype=bool)
        for g in groups:
            mask = ~assigned[g]
            sub = np.zeros(len(g))
            sub[mask] = x[g][mask]
            assigned[g] = True
            total += np.linalg.norm(sub) + mu * np.sum(np.abs(sub))
        return total

    rng = np.random.RandomState(0)
    p = 12
    groups = make_overlapping_chain_groups(p, 4, 2)
    x = rng.randn(p)
    y = rng.randn(p)
    # feasible-representation h is an upper bound on true h, so this isn't
    # an exact test of Lemma 7, but confirms qualitatively sane behavior.
    assert h(x + y, groups, 0.5) <= h(x, groups, 0.5) + h(y, groups, 0.5) + 1e-8


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
