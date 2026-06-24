"""Contract for the PyTorch backend.

The headline test is numerical equivalence with the verified NumPy
implementation: fed the same population and fitnesses from the same initial
state, the torch optimizer must reproduce the numpy mean/sigma/C exactly (to
float64 tolerance). Everything else (nn.Module head, batched evaluation, device
placement) is torch-specific behavior.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sepcmaes.torch_backend import SepCMAES, TrinityRouterHead, minimize
from sepcmaes.router import Role
from sepcmaes import SepCMAES as NumpySepCMAES

F64 = torch.float64


def _vec(x):
    return torch.tensor(x, dtype=F64)


# --------------------------------------------------------------------------- #
# Equivalence with the NumPy reference (the whole point of a faithful port)
# --------------------------------------------------------------------------- #
def test_torch_update_matches_numpy_exactly():
    n = 6
    rng = np.random.default_rng(0)
    npo = NumpySepCMAES(mean=np.zeros(n), sigma=1.0, seed=0)
    pto = SepCMAES(mean=torch.zeros(n, dtype=F64), sigma=1.0, seed=0)
    assert npo.population_size == pto.population_size
    lam = npo.population_size

    for _ in range(25):
        X = rng.standard_normal((lam, n))
        f = rng.standard_normal(lam)
        npo.tell(X, f)
        pto.tell(torch.tensor(X, dtype=F64), torch.tensor(f, dtype=F64))
        assert np.allclose(npo.mean, pto.mean.cpu().numpy(), atol=1e-10, rtol=1e-8)
        assert abs(npo.sigma - pto.sigma) <= 1e-9 * max(1.0, npo.sigma)
        assert np.allclose(npo.C, pto.C.cpu().numpy(), atol=1e-10, rtol=1e-8)


def test_torch_strategy_constants_match_numpy():
    npo = NumpySepCMAES(mean=np.zeros(10), sigma=0.5, seed=0)
    pto = SepCMAES(mean=torch.zeros(10, dtype=F64), sigma=0.5, seed=0)
    assert pto.population_size == npo.population_size
    assert pto.dim == npo.dim == 10


# --------------------------------------------------------------------------- #
# Convergence
# --------------------------------------------------------------------------- #
def _run(opt, fn, max_evals):
    best = float("inf")
    while opt.count_evals < max_evals and not opt.stop():
        xs = opt.ask()
        opt.tell(xs, fn(xs))
        best = min(best, opt.best.f)
    return best


def test_torch_converges_on_sphere():
    opt = SepCMAES(mean=torch.full((8,), 3.0, dtype=F64), sigma=2.0, seed=0)
    best = _run(opt, lambda xs: torch.sum(xs**2, dim=1), 4000)
    assert best < 1e-9


def test_torch_converges_on_separable_ellipsoid():
    n = 10
    coeff = _vec(1e6 ** (np.arange(n) / (n - 1)))

    def ell(xs):
        return torch.sum(coeff * xs**2, dim=1)

    opt = SepCMAES(mean=torch.full((n,), 1.0, dtype=F64), sigma=1.0, seed=0)
    assert _run(opt, ell, 20000) < 1e-8


def test_torch_ask_reproducible():
    a = SepCMAES(mean=torch.zeros(6, dtype=F64), sigma=0.4, seed=42)
    b = SepCMAES(mean=torch.zeros(6, dtype=F64), sigma=0.4, seed=42)
    assert torch.allclose(a.ask(), b.ask())


def test_torch_minimize_sphere():
    res = minimize(
        lambda x: torch.sum(x**2),
        x0=torch.full((5,), 2.0, dtype=F64),
        sigma0=1.0,
        max_evals=5000,
        seed=0,
    )
    assert res.fun < 1e-9
    assert torch.is_tensor(res.x)


# --------------------------------------------------------------------------- #
# TrinityRouterHead as a torch nn.Module
# --------------------------------------------------------------------------- #
def test_torch_head_is_nn_module_with_10240_params():
    import torch.nn as nn

    head = TrinityRouterHead(n_agents=7, feature_dim=1024)
    assert isinstance(head, nn.Module)
    assert head.n_params == 10_240


def test_torch_head_is_bias_free():
    head = TrinityRouterHead(n_agents=7, feature_dim=12, dtype=F64)
    head.set_params(torch.zeros(head.n_params, dtype=F64))
    out = head.logits(torch.randn(12, dtype=F64))
    assert out.shape == (10,)
    assert torch.allclose(out, torch.zeros_like(out))


def test_torch_head_param_roundtrip():
    head = TrinityRouterHead(n_agents=5, feature_dim=16, dtype=F64)
    theta = torch.arange(head.n_params, dtype=F64)
    head.set_params(theta)
    assert torch.equal(head.get_params(), theta)


def test_torch_head_decide_valid():
    head = TrinityRouterHead(n_agents=7, feature_dim=32, dtype=F64)
    head.set_params(torch.randn(head.n_params, dtype=F64))
    agent_id, role = head.decide(torch.randn(32, dtype=F64))
    assert 0 <= agent_id < 7
    assert isinstance(role, Role)


def test_torch_head_batched_logits_matches_loop():
    head = TrinityRouterHead(n_agents=4, feature_dim=16, dtype=F64)
    k, s = 5, 3  # 5 candidate param vectors, 3 states
    Theta = torch.randn(k, head.n_params, dtype=F64)
    H = torch.randn(s, 16, dtype=F64)
    out = head.batched_logits(Theta, H)
    assert out.shape == (k, s, 4 + 3)
    head.set_params(Theta[2])
    assert torch.allclose(out[2, 1], head.logits(H[1]), atol=1e-9)


# --------------------------------------------------------------------------- #
# Device placement (CUDA if present)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_torch_runs_on_cuda():
    dev = "cuda"
    opt = SepCMAES(
        mean=torch.full((8,), 3.0, dtype=F64, device=dev), sigma=2.0, seed=0, device=dev
    )
    best = _run(opt, lambda xs: torch.sum(xs**2, dim=1), 4000)
    assert best < 1e-6
    assert opt.mean.device.type == "cuda"


# --------------------------------------------------------------------------- #
# Bugs surfaced by adversarial review (RED before fix)
# --------------------------------------------------------------------------- #
def test_torch_update_matches_numpy_with_tied_fitnesses():
    """Equivalence must hold even when fitnesses tie. numpy's default sort is
    unstable and torch's is stable, so without forcing a stable sort on both,
    the mu-best selection (and the positionally-applied weights) diverges on
    plateaus / integer / accuracy-style rewards."""
    n = 5
    rng = np.random.default_rng(1)
    npo = NumpySepCMAES(mean=np.zeros(n), sigma=1.0, seed=0)
    pto = SepCMAES(mean=torch.zeros(n, dtype=F64), sigma=1.0, seed=0)
    lam = npo.population_size
    for _ in range(15):
        X = rng.standard_normal((lam, n))
        f = rng.integers(0, 3, size=lam).astype(float)  # many ties
        npo.tell(X, f)
        pto.tell(torch.tensor(X, dtype=F64), torch.tensor(f, dtype=F64))
        assert np.allclose(npo.mean, pto.mean.cpu().numpy(), atol=1e-10, rtol=1e-8)
        assert np.allclose(npo.C, pto.C.cpu().numpy(), atol=1e-10, rtol=1e-8)
        assert abs(npo.sigma - pto.sigma) <= 1e-9 * max(1.0, npo.sigma)


def test_torch_head_does_not_build_autograd_graph():
    """The head is optimized by a gradient-free ES, so it must not build/retain
    an autograd graph during fitness rollouts."""
    head = TrinityRouterHead(7, 16, dtype=F64)
    head.set_params(torch.randn(head.n_params, dtype=F64))
    out = head.logits(torch.randn(16, dtype=F64))
    assert out.grad_fn is None
    assert head.linear.weight.requires_grad is False


def test_torch_decide_rejects_batched_input():
    """decide() returns a single (agent, role); a batched hidden state has no
    single answer and must be rejected rather than silently mis-argmaxed."""
    head = TrinityRouterHead(7, 16, dtype=F64)
    head.set_params(torch.randn(head.n_params, dtype=F64))
    with pytest.raises(ValueError):
        head.decide(torch.randn(4, 16, dtype=F64))


def test_torch_head_dtype_follows_to_and_set_params_keeps_precision():
    head = TrinityRouterHead(5, 8)  # default
    head = head.to(torch.float64)
    assert head.dtype == torch.float64  # property must track .to()
    theta = torch.arange(head.n_params, dtype=torch.float64) / 3.0
    head.set_params(theta)
    assert head.get_params().dtype == torch.float64
    assert torch.equal(head.get_params(), theta)  # no silent truncation
