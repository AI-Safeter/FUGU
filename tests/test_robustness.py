"""Robustness, edge cases, and error-path coverage for SepCMAES.

Most of these were surfaced by an adversarial review of the optimizer. The four
"reject"/"diverge gracefully" tests reproduce genuine defects (they must fail
before the fix); the bounds/stop/construction tests fill coverage of existing,
correct behavior.
"""

import numpy as np
import pytest

from sepcmaes import SepCMAES, minimize


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


# --------------------------------------------------------------------------- #
# Confirmed numerics defects (RED before fix)
# --------------------------------------------------------------------------- #
def test_extreme_step_diverges_gracefully_not_crashes():
    """A huge recombined step must not crash the step-size update with an
    OverflowError; it should saturate so the 'diverged' stop can fire."""
    opt = SepCMAES(mean=np.zeros(5), sigma=1.0, population_size=8, seed=0)
    xs = opt.ask()
    xs[:] = 1e9  # all candidates identical & enormous -> giant p_sigma
    opt.tell(xs, np.arange(8.0))  # must not raise
    assert "diverged" in opt.stop()  # divergence detected, not a crash


def test_tell_rejects_nonfinite_solutions():
    """A NaN/Inf coordinate in a candidate must be rejected, not silently
    propagated into the mean/sigma/covariance (which permanently wedges runs)."""
    opt = SepCMAES(mean=np.zeros(3), sigma=1.0, seed=0)
    xs = opt.ask()
    xs[0, 0] = np.nan
    with pytest.raises(ValueError):
        opt.tell(xs, np.zeros(xs.shape[0]))
    # state stays clean after the rejected tell
    assert np.all(np.isfinite(opt.mean)) and np.isfinite(opt.sigma)


def test_n_resample_must_be_positive():
    with pytest.raises(ValueError):
        SepCMAES(
            mean=np.zeros(3), sigma=1.0, bounds=([-1, -1, -1], [1, 1, 1]), n_resample=0
        )


def test_tell_rejects_wrong_batch_size():
    opt = SepCMAES(mean=np.zeros(4), sigma=1.0, seed=0)
    xs = opt.ask()[: opt.population_size - 1]  # one too few
    with pytest.raises(ValueError, match="solutions"):
        opt.tell(xs, np.zeros(xs.shape[0]))


# --------------------------------------------------------------------------- #
# Bounds (box constraints)
# --------------------------------------------------------------------------- #
def test_ask_respects_bounds_under_wide_sigma():
    low, high = np.full(4, -1.0), np.full(4, 1.0)
    opt = SepCMAES(mean=np.zeros(4), sigma=5.0, bounds=(low, high), seed=0)
    for _ in range(20):
        xs = opt.ask()
        assert np.all(xs >= low) and np.all(xs <= high)
        opt.tell(xs, [sphere(x) for x in xs])


def test_scalar_bounds_broadcast_to_all_coordinates():
    opt = SepCMAES(mean=np.zeros(5), sigma=3.0, bounds=(-2.0, 2.0), seed=0)
    xs = opt.ask()
    assert np.all(xs >= -2.0) and np.all(xs <= 2.0)


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        SepCMAES(mean=np.zeros(3), sigma=1.0, bounds=([0, 0, 0], [0, 1, 1]))


def test_tight_box_forces_clip_fallback_still_in_bounds():
    # n_resample=1 with a tight box almost always exhausts resampling -> clip.
    opt = SepCMAES(
        mean=np.zeros(3), sigma=10.0, bounds=(-0.01, 0.01), seed=0, n_resample=1
    )
    xs = opt.ask()
    assert np.all(xs >= -0.01) and np.all(xs <= 0.01)


# --------------------------------------------------------------------------- #
# Stop conditions
# --------------------------------------------------------------------------- #
def test_stop_on_max_iterations():
    opt = SepCMAES(mean=np.zeros(4), sigma=1.0, seed=0, max_iterations=5)
    for _ in range(5):
        assert not opt.stop()
        xs = opt.ask()
        opt.tell(xs, [sphere(x) for x in xs])
    assert "max_iterations" in opt.stop()


def test_stop_on_constant_fitness_tolfun():
    opt = SepCMAES(mean=np.zeros(3), sigma=1.0, seed=0)
    triggered = False
    for _ in range(2000):
        if opt.stop():
            triggered = "tol_fun" in opt.stop()
            break
        xs = opt.ask()
        opt.tell(xs, np.zeros(xs.shape[0]))  # perfectly flat landscape
    assert triggered


# --------------------------------------------------------------------------- #
# Construction validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mean=np.zeros(3), sigma=0.0),  # non-positive sigma
        dict(mean=np.zeros(3), sigma=-1.0),
        dict(mean=np.zeros((2, 2)), sigma=1.0),  # non-1D mean
        dict(mean=np.zeros(3), sigma=1.0, population_size=1),  # popsize <= 1
    ],
)
def test_construction_rejects_invalid_args(kwargs):
    with pytest.raises(ValueError):
        SepCMAES(**kwargs)
