"""PyTorch backend for Sep-CMA-ES and TRINITY's router head.

A device-agnostic port of :mod:`sepcmaes.optimizer` and :mod:`sepcmaes.router`.
The update equations are identical to the NumPy reference (the test suite checks
them for bit-for-bit-close equality in float64), so this is the same algorithm,
not a re-derivation. What torch adds:

* The router head is a real ``nn.Module``, so it consumes an LLM's hidden state
  tensor directly with no numpy round-trip and moves with ``.to(device)``.
* State lives on whatever device you choose, so the optimizer, the head, and the
  LLM that scores candidates can all sit on the same GPU.
* :meth:`TrinityRouterHead.batched_logits` evaluates a whole population of
  candidate parameter vectors against a batch of states in one einsum, which is
  how you would score ``lambda`` candidates over ``m_CMA`` rollouts efficiently.

The optimizer defaults to float64 because the ES converges to ~1e-15 on
well-conditioned problems; float32 (the LLM's dtype) only reaches ~1e-6.

References: Ros & Hansen (2008); Hansen (2016); TRINITY (arXiv:2512.04695).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "sepcmaes.torch_backend requires PyTorch. Install it with "
        "`pip install sepcmaes[torch]` or `pip install torch`. The NumPy "
        "backend (`from sepcmaes import SepCMAES`) needs no extra dependency."
    ) from exc

from .optimizer import OptimizeResult, Solution
from .router import (
    Role,
    TrinityRouterHead as _NumpyHeadUnused,
)  # noqa: F401  (Role re-export)

TensorLike = Union[torch.Tensor, Sequence[float]]


class SepCMAES:
    """Separable CMA-ES on PyTorch tensors (ask-and-tell).

    Parameters mirror :class:`sepcmaes.optimizer.SepCMAES`, plus ``device`` and
    ``dtype``. All dynamic state is held on ``device`` with dtype ``dtype``.
    """

    def __init__(
        self,
        mean: TensorLike,
        sigma: float,
        *,
        population_size: Optional[int] = None,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
        bounds: Optional[tuple] = None,
        tol_x: float = 1e-12,
        tol_fun: float = 1e-12,
        max_iterations: Optional[int] = None,
        n_resample: int = 100,
    ):
        mean_t = torch.as_tensor(mean, dtype=dtype)
        if device is None:
            device = mean_t.device
        self._device = torch.device(device)
        self._dtype = dtype
        mean_t = mean_t.to(self._device).clone()
        if mean_t.ndim != 1:
            raise ValueError("mean must be a 1-D vector")
        if not (sigma > 0):
            raise ValueError("sigma (initial step size) must be positive")

        n = mean_t.numel()
        self._n = n
        self._mean = mean_t
        self._sigma = float(sigma)

        self._gen_rng = torch.Generator(device=self._device)
        if seed is not None:
            self._gen_rng.manual_seed(int(seed))

        # ----- selection & recombination (identical constants to the np ref) ----- #
        if population_size is None:
            population_size = 4 + int(math.floor(3 * math.log(n)))
        if population_size <= 1:
            raise ValueError("population_size must be > 1")
        self._lambda = int(population_size)
        self._mu = self._lambda // 2

        weights_prime = torch.tensor(
            [
                math.log((self._lambda + 1) / 2) - math.log(i + 1)
                for i in range(self._mu)
            ],
            dtype=dtype,
            device=self._device,
        )
        self._weights = weights_prime / weights_prime.sum()
        self._mu_eff = float(1.0 / torch.sum(self._weights**2))

        # ----- adaptation rates (python floats, identical formulas) ----- #
        me = self._mu_eff
        self._c_c = (4 + me / n) / (n + 4 + 2 * me / n)
        self._c_sigma = (me + 2) / (n + me + 5)
        self._d_sigma = (
            1 + 2 * max(0.0, math.sqrt((me - 1) / (n + 1)) - 1) + self._c_sigma
        )
        self._c_m = 1.0

        c1 = 2 / ((n + 1.3) ** 2 + me)
        c_mu = min(1 - c1, 2 * (me - 2 + 1 / me) / ((n + 2) ** 2 + me))
        accel = (n + 2) / 3.0
        c1 *= accel
        c_mu *= accel
        if c1 + c_mu > 1.0:
            scale = 1.0 / (c1 + c_mu)
            c1 *= scale
            c_mu *= scale
        self._c1 = c1
        self._c_mu = c_mu

        self._chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # ----- dynamic state (tensors on device) ----- #
        self._C = torch.ones(n, dtype=dtype, device=self._device)
        self._D = torch.ones(n, dtype=dtype, device=self._device)
        self._p_sigma = torch.zeros(n, dtype=dtype, device=self._device)
        self._p_c = torch.zeros(n, dtype=dtype, device=self._device)
        self._generation = 0
        self._count_evals = 0
        self._best = Solution(x=self._mean.clone(), f=math.inf)

        if bounds is not None:
            low, high = bounds
            self._low = (
                torch.as_tensor(low, dtype=dtype, device=self._device).expand(n).clone()
            )
            self._high = (
                torch.as_tensor(high, dtype=dtype, device=self._device)
                .expand(n)
                .clone()
            )
            if torch.any(self._low >= self._high):
                raise ValueError("each lower bound must be < its upper bound")
        else:
            self._low = self._high = None
        self._n_resample = int(n_resample)
        if self._n_resample < 1:
            raise ValueError("n_resample must be >= 1")

        self._tol_x = tol_x * self._sigma
        self._tol_fun = tol_fun
        self._max_iterations = max_iterations
        self._fun_window = 10 + int(math.ceil(30 * n / self._lambda))
        self._fun_history: list = []

    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        return self._n

    @property
    def population_size(self) -> int:
        return self._lambda

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def mean(self) -> torch.Tensor:
        return self._mean.clone()

    @property
    def C(self) -> torch.Tensor:
        return self._C.clone()

    @property
    def std(self) -> torch.Tensor:
        return self._sigma * self._D

    @property
    def best(self) -> Solution:
        return self._best

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def count_evals(self) -> int:
        return self._count_evals

    # ------------------------------------------------------------------ #
    def ask(self) -> torch.Tensor:
        """Return a ``(lambda, n)`` tensor of candidate solutions."""
        z = torch.randn(
            self._lambda,
            self._n,
            generator=self._gen_rng,
            dtype=self._dtype,
            device=self._device,
        )
        X = self._mean + self._sigma * self._D * z
        if self._low is None:
            return X
        # vectorized rejection sampling for box bounds, then clip as last resort
        for _ in range(self._n_resample - 1):
            bad = ~((X >= self._low) & (X <= self._high)).all(dim=1)
            if not bool(bad.any()):
                break
            k = int(bad.sum())
            zk = torch.randn(
                k,
                self._n,
                generator=self._gen_rng,
                dtype=self._dtype,
                device=self._device,
            )
            X[bad] = self._mean + self._sigma * self._D * zk
        return torch.clamp(X, self._low, self._high)

    def tell(self, solutions: TensorLike, fitnesses: TensorLike) -> None:
        """Update the distribution from evaluated candidates (minimization)."""
        X = torch.as_tensor(solutions, dtype=self._dtype, device=self._device)
        f = torch.as_tensor(fitnesses, dtype=self._dtype, device=self._device)
        if X.ndim != 2 or X.shape[1] != self._n:
            raise ValueError(f"solutions must have shape (lambda, {self._n})")
        if X.shape[0] != self._lambda:
            raise ValueError(
                f"expected {self._lambda} solutions (population_size), got {X.shape[0]}"
            )
        if f.shape[0] != X.shape[0]:
            raise ValueError("number of fitnesses must match number of solutions")
        if not bool(torch.isfinite(X).all()):
            raise ValueError("solution vectors must be finite")
        if not bool(torch.isfinite(f).all()):
            raise ValueError("fitness values must be finite")

        self._count_evals += int(X.shape[0])

        gen_best_idx = int(torch.argmin(f))
        gen_best_f = float(f[gen_best_idx])
        if gen_best_f < self._best.f:
            self._best = Solution(x=X[gen_best_idx].clone(), f=gen_best_f)
        self._fun_history.append(gen_best_f)
        if len(self._fun_history) > self._fun_window:
            self._fun_history.pop(0)

        # stable sort so tie-breaking matches the numpy reference (and the
        # cmaes library's stable list sort) when fitnesses tie.
        order = torch.argsort(f, stable=True)
        x_best = X[order[: self._mu]]
        y = (x_best - self._mean) / self._sigma  # (mu, n)
        y_w = self._weights @ y  # (n,)

        n = self._n
        self._mean = self._mean + self._c_m * self._sigma * y_w

        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * (y_w / self._D)
        ps_norm = float(torch.linalg.norm(self._p_sigma))
        log_factor = (self._c_sigma / self._d_sigma) * (ps_norm / self._chi_n - 1)
        self._sigma *= math.exp(min(log_factor, 700.0))

        denom = math.sqrt(1 - (1 - self._c_sigma) ** (2 * (self._generation + 1)))
        h_sigma = 1.0 if ps_norm / denom < (1.4 + 2 / (n + 1)) * self._chi_n else 0.0
        self._p_c = (1 - self._c_c) * self._p_c + h_sigma * math.sqrt(
            self._c_c * (2 - self._c_c) * self._mu_eff
        ) * y_w
        delta_h = (1 - h_sigma) * self._c_c * (2 - self._c_c)

        rank_one = self._c1 * (self._p_c**2 + delta_h * self._C)
        rank_mu = self._c_mu * (self._weights @ (y**2))
        self._C = (1 - self._c1 - self._c_mu) * self._C + rank_one + rank_mu
        self._C = torch.clamp_min(self._C, 1e-300)
        self._D = torch.sqrt(self._C)

        self._generation += 1

    # ------------------------------------------------------------------ #
    def stop(self) -> dict:
        conditions: dict = {}
        if not bool(torch.isfinite(self._C).all()) or self._sigma > 1e60:
            conditions["diverged"] = self._sigma
            return conditions
        if (
            self._max_iterations is not None
            and self._generation >= self._max_iterations
        ):
            conditions["max_iterations"] = self._generation
        step = self._sigma * self._D
        if bool((step < self._tol_x).all()) and bool(
            (self._sigma * torch.abs(self._p_c) < self._tol_x).all()
        ):
            conditions["tol_x"] = float(step.max())
        if len(self._fun_history) >= self._fun_window:
            spread = max(self._fun_history) - min(self._fun_history)
            if spread < self._tol_fun:
                conditions["tol_fun"] = spread
        return conditions


class TrinityRouterHead(nn.Module):
    """TRINITY's bias-free linear router head as an ``nn.Module``.

    ``forward``/``logits`` map a hidden state ``h`` of shape ``(d,)`` or
    ``(B, d)`` to ``(L+3)`` (resp. ``(B, L+3)``) logits: ``L`` agent logits and
    3 role logits (Thinker/Worker/Verifier). The flattened weight is the vector
    Sep-CMA-ES optimizes.
    """

    N_ROLES = 3

    def __init__(
        self, n_agents: int, feature_dim: int, *, dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        self.n_agents = int(n_agents)
        self.feature_dim = int(feature_dim)
        self._out = self.n_agents + self.N_ROLES
        self.linear = nn.Linear(self.feature_dim, self._out, bias=False, dtype=dtype)
        with torch.no_grad():
            self.linear.weight.zero_()
        # The head is optimized by a gradient-free ES, so it must never build an
        # autograd graph during fitness rollouts. Disable grad on the weight;
        # set_params still mutates it under no_grad. Re-enable if you ever want
        # to also train it by backprop.
        self.linear.weight.requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        # live, so it tracks .to(dtype=...) instead of going stale
        return self.linear.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    @property
    def n_roles(self) -> int:
        return self.N_ROLES

    @property
    def n_params(self) -> int:
        return self._out * self.feature_dim

    def get_params(self) -> torch.Tensor:
        return self.linear.weight.detach().reshape(-1).clone()

    def set_params(self, theta: torch.Tensor) -> "TrinityRouterHead":
        w = self.linear.weight
        # cast to the weight's live dtype/device (so .to(float64) is honored
        # and there is no surprise precision drop relative to the current head)
        theta = torch.as_tensor(theta, dtype=w.dtype, device=w.device)
        if theta.shape != (self.n_params,):
            raise ValueError(f"theta must have shape ({self.n_params},)")
        with torch.no_grad():
            w.copy_(theta.reshape(self._out, self.feature_dim))
        return self

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)

    def policy(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.logits(h)
        return (
            torch.softmax(z[..., : self.n_agents], dim=-1),
            torch.softmax(z[..., self.n_agents :], dim=-1),
        )

    def decide(
        self,
        h: torch.Tensor,
        *,
        sample: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[int, Role]:
        if h.ndim != 1:
            raise ValueError(
                "decide() expects a single state of shape (feature_dim,); "
                "use logits()/policy() for batched input"
            )
        p_agent, p_role = self.policy(h)
        if sample:
            if generator is not None and generator.device != p_agent.device:
                raise ValueError(
                    "the sampling generator must be on the same device as the head"
                )
            agent_id = int(torch.multinomial(p_agent, 1, generator=generator))
            role = Role(int(torch.multinomial(p_role, 1, generator=generator)))
        else:
            agent_id = int(torch.argmax(p_agent))
            role = Role(int(torch.argmax(p_role)))
        return agent_id, role

    def batched_logits(self, theta: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Logits for K candidate parameter vectors over S states, in one pass.

        ``theta`` is ``(K, n_params)``, ``H`` is ``(S, d)``; returns
        ``(K, S, L+3)``. This is how you score a whole Sep-CMA-ES population
        against a batch of rollouts without a Python loop.
        """
        if theta.ndim != 2 or theta.shape[1] != self.n_params:
            raise ValueError(f"theta must have shape (K, {self.n_params})")
        if H.ndim != 2 or H.shape[1] != self.feature_dim:
            raise ValueError(f"H must have shape (S, {self.feature_dim})")
        W = theta.reshape(theta.shape[0], self._out, self.feature_dim)
        return torch.einsum("kod,sd->kso", W, H.to(W.dtype))


def minimize(
    objective: Callable[[torch.Tensor], torch.Tensor],
    x0: TensorLike,
    sigma0: float,
    *,
    max_evals: int = 10000,
    seed: Optional[int] = None,
    maximize: bool = False,
    population_size: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float64,
    **kwargs,
) -> OptimizeResult:
    """Optimize a torch ``objective`` with the torch Sep-CMA-ES backend.

    ``objective`` may take a single ``(n,)`` vector or a ``(lambda, n)`` batch;
    if it returns a scalar it is mapped over the population, if it returns a
    ``(lambda,)`` vector it is used directly (the fast, batched path).
    """
    opt = SepCMAES(
        mean=x0,
        sigma=sigma0,
        seed=seed,
        population_size=population_size,
        device=device,
        dtype=dtype,
        **kwargs,
    )
    sign = -1.0 if maximize else 1.0

    def evaluate(xs: torch.Tensor) -> torch.Tensor:
        out = objective(xs)
        if torch.is_tensor(out) and out.ndim == 1 and out.shape[0] == xs.shape[0]:
            return sign * out
        return sign * torch.stack(
            [torch.as_tensor(objective(x)).reshape(()) for x in xs]
        )

    stop: dict = {}
    while True:
        stop = opt.stop()
        if stop:
            break
        if opt.count_evals > 0 and opt.count_evals + opt.population_size > max_evals:
            break
        xs = opt.ask()
        opt.tell(xs, evaluate(xs))

    diverged = "diverged" in stop
    return OptimizeResult(
        x=opt.best.x,
        fun=sign * opt.best.f,
        nfev=opt.count_evals,
        nit=opt.generation,
        success=not diverged,
        message="; ".join(stop) if stop else "evaluation budget reached",
        sigma=opt.sigma,
        mean=opt.mean,
        stop=stop,
    )
