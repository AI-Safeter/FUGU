"""Behavioral contract for the Sep-CMA-ES optimizer.

These tests describe *what the optimizer must do*, not how. They avoid
hard-coding the exact strategy constants (lambda, c_sigma, c_mu, ...) so they
remain valid while those are reconciled against Ros & Hansen (2008) / TRINITY.

Reference: Ros, R., & Hansen, N. (2008). "A Simple Modification in CMA-ES
Achieving Linear Time and Space Complexity." PPSN X.
"""

import numpy as np
import pytest

from sepcmaes import SepCMAES, minimize


# --------------------------------------------------------------------------- #
# Standard black-box benchmark functions (all minimized, optimum value 0).
# --------------------------------------------------------------------------- #
def sphere(x):
    return float(np.sum(x**2))


def ellipsoid_separable(x):
    """Axis-parallel (separable) ill-conditioned ellipsoid, cond ~ 1e6.

    This is the landscape sep-CMA-ES is *designed* for: the Hessian is
    diagonal, so a diagonal covariance model suffices.
    """
    n = len(x)
    exponents = np.arange(n) / max(n - 1, 1)
    coeff = 1e6**exponents
    return float(np.sum(coeff * x**2))


def rosenbrock(x):
    """Non-separable banana valley; harder for a diagonal model but solvable."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


# --------------------------------------------------------------------------- #
# Construction / interface invariants
# --------------------------------------------------------------------------- #
def test_dimension_inferred_from_mean():
    opt = SepCMAES(mean=np.zeros(7), sigma=0.5, seed=0)
    assert opt.dim == 7


def test_default_population_size_matches_cma_default():
    # lambda = 4 + floor(3 ln n)  (Hansen tutorial default)
    opt = SepCMAES(mean=np.zeros(10), sigma=0.5, seed=0)
    assert opt.population_size == 4 + int(3 * np.log(10))


def test_ask_returns_lambda_candidates_of_dim_n():
    opt = SepCMAES(mean=np.zeros(5), sigma=0.3, seed=0)
    xs = opt.ask()
    xs = np.asarray(xs)
    assert xs.shape == (opt.population_size, 5)


def test_covariance_is_diagonal_linear_space():
    """The defining property: covariance is stored as a length-n vector,
    never an n x n matrix. This is what makes the method O(n) per sample."""
    n = 12
    opt = SepCMAES(mean=np.zeros(n), sigma=0.5, seed=0)
    C = np.asarray(opt.C)
    assert C.shape == (n,)  # diagonal only
    assert np.allclose(C, 1.0)  # initial diagonal C = I


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_reproducible_samples():
    a = SepCMAES(mean=np.zeros(6), sigma=0.4, seed=42)
    b = SepCMAES(mean=np.zeros(6), sigma=0.4, seed=42)
    assert np.allclose(np.asarray(a.ask()), np.asarray(b.ask()))


def test_different_seed_different_samples():
    a = SepCMAES(mean=np.zeros(6), sigma=0.4, seed=1)
    b = SepCMAES(mean=np.zeros(6), sigma=0.4, seed=2)
    assert not np.allclose(np.asarray(a.ask()), np.asarray(b.ask()))


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #
def _run(opt, fn, max_evals):
    best = np.inf
    while opt.count_evals < max_evals and not opt.stop():
        xs = opt.ask()
        fs = [fn(np.asarray(x)) for x in xs]
        opt.tell(xs, fs)
        best = min(best, opt.best.f)
    return best


def test_converges_on_sphere():
    opt = SepCMAES(mean=np.full(8, 3.0), sigma=2.0, seed=0)
    best = _run(opt, sphere, max_evals=4000)
    assert best < 1e-9


def test_converges_on_separable_ellipsoid():
    # The home turf of sep-CMA-ES.
    opt = SepCMAES(mean=np.full(10, 1.0), sigma=1.0, seed=0)
    best = _run(opt, ellipsoid_separable, max_evals=20000)
    assert best < 1e-8


def test_diagonal_adapts_to_separable_curvature():
    """After optimizing the separable ellipsoid, the coordinate with the
    steepest curvature (largest coefficient, last index) must have a much
    smaller sampling variance than the flattest (first index)."""
    n = 10
    opt = SepCMAES(mean=np.full(n, 1.0), sigma=1.0, seed=0)
    _run(opt, ellipsoid_separable, max_evals=20000)
    C = np.asarray(opt.C)
    assert C[-1] < C[0] / 100.0


def test_solves_rosenbrock_with_restarts():
    """Rosenbrock is *non*-separable with a local optimum for n >= 4, so a
    diagonal-covariance model cannot rotate to follow the banana valley and a
    single run may stall near (-1, 1, ..., 1). This is the documented weakness
    of sep-CMA-ES (Ros & Hansen 2008). The standard remedy is independent
    restarts: across a handful of seeds at least one run escapes to the global
    optimum at (1, ..., 1)."""
    best = min(
        _run(SepCMAES(mean=np.zeros(6), sigma=0.5, seed=s), rosenbrock, 30000)
        for s in range(4)
    )
    assert best < 1e-2


def test_single_run_makes_substantial_progress_on_sphere_subspace():
    """Even one run reliably crushes the separable part: from f(0)=... the
    optimizer drives a well-conditioned separable problem to near machine zero,
    confirming the stall above is specific to non-separable curvature, not a
    failure to optimize."""
    opt = SepCMAES(mean=np.full(6, 2.0), sigma=1.0, seed=0)
    best = _run(opt, sphere, max_evals=4000)
    assert best < 1e-10


# --------------------------------------------------------------------------- #
# minimize() convenience wrapper
# --------------------------------------------------------------------------- #
def test_minimize_finds_sphere_optimum():
    res = minimize(sphere, x0=np.full(5, 2.0), sigma0=1.0, max_evals=5000, seed=0)
    assert res.fun < 1e-9
    assert np.allclose(res.x, 0.0, atol=1e-4)
    assert res.nfev <= 5000


def test_minimize_reports_full_result_fields():
    """The OptimizeResult must be fully populated, and on deep convergence the
    success flag is set with a non-empty stop reason."""
    res = minimize(sphere, x0=np.full(5, 2.0), sigma0=1.0, max_evals=20000, seed=0)
    assert res.success is True
    assert isinstance(res.message, str) and res.message
    assert res.nit > 0
    assert res.stop  # converged before the budget -> stop fired
    assert np.allclose(res.mean, 0.0, atol=1e-4)


def test_minimize_max_evals_is_a_hard_ceiling():
    res = minimize(sphere, x0=np.full(5, 2.0), sigma0=1.0, max_evals=60, seed=0)
    assert res.nfev <= 60
    assert res.success is True  # budget reached counts as a clean finish


def test_minimize_respects_maximize_flag():
    """Maximizing -(x-1)^2 must converge to x=1 AND report fun ~= 0 in the
    caller's original sign (guards against a sign-reporting bug)."""
    obj = lambda x: -float(np.sum((x - 1.0) ** 2))
    res = minimize(
        obj, x0=np.zeros(4), sigma0=0.5, max_evals=4000, seed=0, maximize=True
    )
    assert np.allclose(res.x, 1.0, atol=1e-3)
    assert res.fun == pytest.approx(0.0, abs=1e-4)  # peak value, original sign
    assert res.fun == pytest.approx(obj(res.x), abs=1e-9)  # sign is consistent


def test_minimize_and_maximize_agree_on_same_objective():
    # min of (x-1)^2 and max of -(x-1)^2 are the same point with opposite-sign fun
    g = lambda x: float(np.sum((x - 1.0) ** 2))
    lo = minimize(g, x0=np.zeros(3), sigma0=0.5, max_evals=4000, seed=0)
    hi = minimize(
        lambda x: -g(x),
        x0=np.zeros(3),
        sigma0=0.5,
        max_evals=4000,
        seed=0,
        maximize=True,
    )
    assert np.allclose(lo.x, hi.x, atol=1e-2)
    assert lo.fun == pytest.approx(-hi.fun, abs=1e-4)


# --------------------------------------------------------------------------- #
# Step-size sanity: sigma must shrink as we home in on the sphere optimum.
# --------------------------------------------------------------------------- #
def test_sigma_shrinks_on_sphere():
    opt = SepCMAES(mean=np.full(8, 3.0), sigma=2.0, seed=0)
    sigma0 = opt.sigma
    _run(opt, sphere, max_evals=4000)
    assert opt.sigma < sigma0 * 1e-3


# --------------------------------------------------------------------------- #
# Cross-validation against the reference `cmaes` library (skipped if absent).
# Our strategy constants follow Hansen (2016) + the Ros & Hansen (2008)
# (n+2)/3 acceleration; the reference uses a slightly different default set.
# They are not bit-identical (different RNG, different constant forms) but must
# reach the same convergence order of magnitude on separable landscapes.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn, x0, sigma0, evals",
    [
        (sphere, np.full(8, 3.0), 2.0, 4000),
        (ellipsoid_separable, np.full(10, 1.0), 1.0, 20000),
    ],
)
def test_matches_reference_cmaes_on_separable(fn, x0, sigma0, evals):
    SepCMA = pytest.importorskip("cmaes").SepCMA

    mine = SepCMAES(mean=x0.copy(), sigma=sigma0, seed=0)
    mine_best = _run(mine, fn, evals)

    ref = SepCMA(mean=x0.copy(), sigma=sigma0, seed=0)
    ref_best, n = np.inf, 0
    while n < evals and not ref.should_stop():
        sols = []
        for _ in range(ref.population_size):
            x = ref.ask()
            sols.append((x, fn(x)))
            n += 1
        ref.tell(sols)
        ref_best = min(ref_best, min(v for _, v in sols))

    # Both must reach near machine precision AND genuinely agree: their final
    # log10 values differ by no more than a few orders of magnitude (they use
    # different RNGs and constant forms, so exact equality is not expected).
    assert mine_best < 1e-8 and ref_best < 1e-8
    assert abs(np.log10(mine_best) - np.log10(ref_best)) < 4.0
