"""Coverage for the torch backend's wrapper, error paths, stop conditions, and
bounds. Surfaced as gaps by the adversarial review of the port."""

import pytest

torch = pytest.importorskip("torch")

from sepcmaes.torch_backend import SepCMAES, TrinityRouterHead, minimize

F64 = torch.float64


def sphere_batch(xs):
    return torch.sum(xs**2, dim=1)


# --------------------------------------------------------------------------- #
# minimize() wrapper
# --------------------------------------------------------------------------- #
def test_torch_minimize_maximize_reports_original_sign():
    obj = lambda x: -torch.sum((x - 1.0) ** 2)  # concave, peak 0 at x=1
    res = minimize(
        obj,
        x0=torch.zeros(4, dtype=F64),
        sigma0=0.5,
        max_evals=4000,
        seed=0,
        maximize=True,
    )
    assert torch.allclose(res.x, torch.ones(4, dtype=F64), atol=1e-2)
    assert abs(res.fun) < 1e-3  # peak value, original sign


def test_torch_minimize_handles_batched_and_scalar_objectives():
    objectives = [
        lambda xs: torch.sum(xs**2, dim=1),  # batched -> (lambda,)
        lambda x: torch.sum(x**2),  # 0-dim tensor (scalar path)
        lambda x: float(torch.sum(x**2)),  # python float (scalar path)
    ]
    for obj in objectives:
        res = minimize(
            obj, x0=torch.full((4,), 2.0, dtype=F64), sigma0=1.0, max_evals=5000, seed=0
        )
        assert res.fun < 1e-8


# --------------------------------------------------------------------------- #
# Construction / tell validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mean=torch.zeros(3, dtype=F64), sigma=0.0),
        dict(mean=torch.zeros(3, dtype=F64), sigma=-1.0),
        dict(mean=torch.zeros((2, 2), dtype=F64), sigma=1.0),
        dict(mean=torch.zeros(3, dtype=F64), sigma=1.0, population_size=1),
        dict(
            mean=torch.zeros(3, dtype=F64),
            sigma=1.0,
            bounds=([-1, -1, -1], [1, 1, 1]),
            n_resample=0,
        ),
        dict(mean=torch.zeros(3, dtype=F64), sigma=1.0, bounds=([0, 0, 0], [0, 1, 1])),
    ],
)
def test_torch_construction_rejects_invalid_args(kwargs):
    with pytest.raises(ValueError):
        SepCMAES(**kwargs)


def test_torch_tell_validation():
    opt = SepCMAES(mean=torch.zeros(4, dtype=F64), sigma=1.0, seed=0)
    xs = opt.ask()
    lam = xs.shape[0]
    with pytest.raises(ValueError):  # wrong batch size
        opt.tell(xs[:-1], torch.zeros(lam - 1, dtype=F64))
    with pytest.raises(ValueError):  # non-finite solution
        bad = xs.clone()
        bad[0, 0] = float("nan")
        opt.tell(bad, torch.zeros(lam, dtype=F64))
    with pytest.raises(ValueError):  # non-finite fitness
        f = torch.zeros(lam, dtype=F64)
        f[0] = float("inf")
        opt.tell(xs, f)


# --------------------------------------------------------------------------- #
# Stop conditions
# --------------------------------------------------------------------------- #
def test_torch_stop_max_iterations():
    opt = SepCMAES(mean=torch.zeros(4, dtype=F64), sigma=1.0, seed=0, max_iterations=5)
    for _ in range(5):
        assert not opt.stop()
        xs = opt.ask()
        opt.tell(xs, sphere_batch(xs))
    assert "max_iterations" in opt.stop()


def test_torch_stop_diverged():
    opt = SepCMAES(mean=torch.zeros(4, dtype=F64), sigma=1.0, seed=0)
    opt._sigma = 1e61
    assert "diverged" in opt.stop()


def test_torch_stop_tolfun_on_flat_landscape():
    opt = SepCMAES(mean=torch.zeros(3, dtype=F64), sigma=1.0, seed=0)
    triggered = False
    for _ in range(2000):
        if opt.stop():
            triggered = "tol_fun" in opt.stop()
            break
        xs = opt.ask()
        opt.tell(xs, torch.zeros(xs.shape[0], dtype=F64))
    assert triggered


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #
def test_torch_ask_respects_per_coordinate_bounds():
    low = torch.full((4,), -1.0, dtype=F64)
    high = torch.full((4,), 1.0, dtype=F64)
    opt = SepCMAES(
        mean=torch.zeros(4, dtype=F64), sigma=5.0, seed=0, bounds=(low, high)
    )
    for _ in range(20):
        xs = opt.ask()
        assert torch.all(xs >= low) and torch.all(xs <= high)
        opt.tell(xs, sphere_batch(xs))


def test_torch_scalar_bounds_broadcast():
    opt = SepCMAES(
        mean=torch.zeros(5, dtype=F64), sigma=3.0, seed=0, bounds=(-2.0, 2.0)
    )
    xs = opt.ask()
    assert torch.all(xs >= -2.0) and torch.all(xs <= 2.0)


# --------------------------------------------------------------------------- #
# decide(sample=True)
# --------------------------------------------------------------------------- #
def test_torch_decide_sample_reproducible():
    head = TrinityRouterHead(7, 16, dtype=F64)
    head.set_params(torch.randn(head.n_params, dtype=F64))
    h = torch.randn(16, dtype=F64)
    s1 = [
        head.decide(h, sample=True, generator=torch.Generator().manual_seed(0))
        for _ in range(10)
    ]
    s2 = [
        head.decide(h, sample=True, generator=torch.Generator().manual_seed(0))
        for _ in range(10)
    ]
    assert s1 == s2


def test_torch_sampling_varies_under_flat_policy():
    head = TrinityRouterHead(7, 8, dtype=F64)  # zero weights -> uniform
    h = torch.ones(8, dtype=F64)
    seen = {
        head.decide(h, sample=True, generator=torch.Generator().manual_seed(s))
        for s in range(50)
    }
    assert len(seen) > 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_torch_decide_rejects_cross_device_generator():
    head = TrinityRouterHead(7, 8, dtype=F64).to("cuda")
    head.set_params(torch.randn(head.n_params, dtype=F64, device="cuda"))
    h = torch.ones(8, dtype=F64, device="cuda")
    with pytest.raises(ValueError):
        head.decide(h, sample=True, generator=torch.Generator())  # CPU gen, CUDA head
