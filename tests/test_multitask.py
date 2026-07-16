import numpy as np

from soglasso.multitask import make_multitask_sog_regression, build_multitask_groups


def test_multitask_shapes():
    T, n_per_task = 5, 20
    Phi, y, x_star, groups, p = make_multitask_sog_regression(
        T=T, n_per_task=n_per_task, n_groups=10, group_size=4, shift=2,
        k_active=3, alpha=0.5, random_state=0,
    )
    assert Phi.shape == (T * n_per_task, T * p)
    assert y.shape == (T * n_per_task,)
    assert x_star.shape == (T * p,)


def test_multitask_groups_span_all_tasks():
    """Each aggregated group should touch every one of the T task blocks."""
    T, p = 4, 20
    base_groups = [np.array([0, 1, 2]), np.array([5, 6, 7])]
    agg = build_multitask_groups(base_groups, T, p)
    for g_base, g_agg in zip(base_groups, agg):
        assert len(g_agg) == T * len(g_base)
        # every task's block should be represented
        for t in range(T):
            expected = g_base + t * p
            assert set(expected.tolist()).issubset(set(g_agg.tolist()))


def test_multitask_block_diagonal_structure():
    """Phi should be exactly block-diagonal across tasks (no cross terms)."""
    T, n_per_task = 3, 10
    Phi, y, x_star, groups, p = make_multitask_sog_regression(
        T=T, n_per_task=n_per_task, n_groups=8, group_size=3, shift=2,
        k_active=2, alpha=0.5, random_state=1,
    )
    for t in range(T):
        row_slice = slice(t * n_per_task, (t + 1) * n_per_task)
        for t2 in range(T):
            if t2 == t:
                continue
            col_slice = slice(t2 * p, (t2 + 1) * p)
            assert np.allclose(Phi[row_slice, col_slice], 0.0)


def test_multitask_active_groups_shared_across_tasks():
    """
    Every task should end up with some nonzero coefficients, since the
    same k groups are active across all tasks (Section 5.1's "shared
    macrostructure" construction) -- though the exact nonzero indices
    within a group may differ per task (independent microstructure).
    """
    T = 6
    Phi, y, x_star, groups, p = make_multitask_sog_regression(
        T=T, n_per_task=15, n_groups=10, group_size=4, shift=2,
        k_active=3, alpha=0.6, random_state=2,
    )
    X_star = x_star.reshape(T, p)
    for t in range(T):
        assert np.count_nonzero(X_star[t]) > 0


def test_multitask_fit_reduces_error_vs_null():
    Phi, y, x_star, groups, p = make_multitask_sog_regression(
        T=6, n_per_task=25, n_groups=12, group_size=4, shift=2,
        k_active=3, alpha=0.5, noise_std=0.05, random_state=3,
    )
    from soglasso import sog_lasso
    model = sog_lasso(groups, mu=0.3, eta1=0.5, eta2=1e-4, max_iter=200)
    model.fit(Phi, y)
    err = np.linalg.norm(model.coef_ - x_star)
    null_err = np.linalg.norm(x_star)
    assert err < null_err
