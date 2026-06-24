"""Sep-CMA-ES — separable Covariance Matrix Adaptation Evolution Strategy.

A diagonal-covariance variant of CMA-ES with **linear** time and space
complexity in the search-space dimension ``n``. Full CMA-ES stores and
factorizes an ``n x n`` covariance matrix (O(n^2) memory, O(n^3) eigendecomp);
the separable variant restricts the covariance to its diagonal, so every
quantity is a length-``n`` vector and a generation costs O(lambda * n).

This is the optimizer behind TRINITY's evolved coordinator head: the head has
~10^4 parameters, far too many for full CMA-ES, but its loss landscape is close
to block/axis separable, which is exactly the regime where Ros & Hansen show the
diagonal model both scales and *learns faster* (the (n+2)/3 acceleration below).

References
----------
Ros, R., & Hansen, N. (2008). "A Simple Modification in CMA-ES Achieving Linear
    Time and Space Complexity." PPSN X, LNCS 5199, pp. 296-305.
Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial." arXiv:1604.00772.
Xu, J. et al. (2026). "TRINITY: An Evolved LLM Coordinator." ICLR 2026.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

ArrayLike = Sequence[float]


@dataclass
class Solution:
    """A single evaluated candidate: parameter vector ``x`` and fitness ``f``."""

    x: np.ndarray
    f: float


@dataclass
class OptimizeResult:
    """Outcome of :func:`minimize`, loosely mirroring ``scipy.optimize``."""

    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    success: bool
    message: str
    sigma: float = 0.0
    mean: Optional[np.ndarray] = None
    stop: dict = field(default_factory=dict)


class SepCMAES:
    """Separable CMA-ES with an ask-and-tell interface.

    Parameters
    ----------
    mean : array_like
        Initial distribution mean ``m_0`` (also fixes the dimension ``n``).
    sigma : float
        Initial global step size ``sigma_0`` (> 0). Should be ~ 1/4 of the
        expected coordinate-wise distance from ``mean`` to the optimum.
    population_size : int, optional
        Number of samples per generation ``lambda``. Defaults to the CMA-ES
        rule ``4 + floor(3 ln n)``.
    seed : int, optional
        Seed for the internal NumPy ``Generator`` (reproducibility).
    bounds : (low, high), optional
        Per-coordinate box bounds. Out-of-box samples are resampled (up to
        ``n_resample`` times) then clipped, following the standard ES recipe.
    tol_x, tol_fun : float
        Convergence tolerances on step size and on the recent fitness range.
    max_iterations : int, optional
        Hard cap on generations for :meth:`stop`.

    Notes
    -----
    Usage (ask/tell):

    >>> opt = SepCMAES(mean=np.zeros(10), sigma=0.5, seed=0)
    >>> while not opt.stop():
    ...     xs = opt.ask()
    ...     fs = [my_objective(x) for x in xs]
    ...     opt.tell(xs, fs)
    >>> opt.best.x, opt.best.f
    """

    def __init__(
        self,
        mean: ArrayLike,
        sigma: float,
        *,
        population_size: Optional[int] = None,
        seed: Optional[int] = None,
        bounds: Optional[tuple] = None,
        tol_x: float = 1e-12,
        tol_fun: float = 1e-12,
        max_iterations: Optional[int] = None,
        n_resample: int = 100,
    ):
        mean = np.asarray(mean, dtype=float).copy()
        if mean.ndim != 1:
            raise ValueError("mean must be a 1-D vector")
        if not (sigma > 0):
            raise ValueError("sigma (initial step size) must be positive")

        n = mean.size
        self._n = n
        self._mean = mean
        self._sigma = float(sigma)
        self._rng = np.random.default_rng(seed)

        # ----- selection & recombination (Hansen tutorial defaults) ----- #
        if population_size is None:
            population_size = 4 + int(math.floor(3 * math.log(n)))
        if population_size <= 1:
            raise ValueError("population_size must be > 1")
        self._lambda = int(population_size)
        self._mu = self._lambda // 2

        # log-decreasing positive recombination weights, then normalized.
        weights_prime = np.array(
            [
                math.log((self._lambda + 1) / 2) - math.log(i + 1)
                for i in range(self._mu)
            ]
        )
        self._weights = weights_prime / weights_prime.sum()
        # variance-effective selection mass: mu_eff = 1 / sum(w_i^2).
        self._mu_eff = 1.0 / np.sum(self._weights**2)

        # ----- adaptation rates ----- #
        # cumulation for the rank-one path p_c
        self._c_c = (4 + self._mu_eff / n) / (n + 4 + 2 * self._mu_eff / n)
        # cumulation for the step-size path p_sigma (CSA)
        self._c_sigma = (self._mu_eff + 2) / (n + self._mu_eff + 5)
        self._d_sigma = (
            1
            + 2 * max(0.0, math.sqrt((self._mu_eff - 1) / (n + 1)) - 1)
            + self._c_sigma
        )
        self._c_m = 1.0  # mean learning rate

        # rank-one (c1) and rank-mu (c_mu) covariance learning rates...
        c1 = 2 / ((n + 1.3) ** 2 + self._mu_eff)
        c_mu = min(
            1 - c1,
            2 * (self._mu_eff - 2 + 1 / self._mu_eff) / ((n + 2) ** 2 + self._mu_eff),
        )
        # ...accelerated by (n+2)/3 for the diagonal model (Ros & Hansen 2008,
        # eq. 5): the diagonal has only n free params, so it can be learned
        # faster than the full O(n^2) matrix.
        accel = (n + 2) / 3.0
        c1 *= accel
        c_mu *= accel
        # Defensive renormalization so the convex-combination weight stays >= 0
        # (the (n+2)/3 boost can exceed 1 at very small n).
        if c1 + c_mu > 1.0:
            scale = 1.0 / (c1 + c_mu)
            c1 *= scale
            c_mu *= scale
        self._c1 = c1
        self._c_mu = c_mu

        # E||N(0,I)|| — expected length of a standard normal vector.
        self._chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # ----- dynamic state ----- #
        self._C = np.ones(n)  # diagonal of the covariance (variances)
        self._D = np.ones(n)  # sqrt(C): per-coordinate standard deviations
        self._p_sigma = np.zeros(n)  # conjugate evolution path (step-size)
        self._p_c = np.zeros(n)  # evolution path (rank-one covariance)
        self._generation = 0
        self._count_evals = 0
        self._best = Solution(x=mean.copy(), f=math.inf)

        # ----- bounds & stopping ----- #
        if bounds is not None:
            low, high = bounds
            self._low = np.broadcast_to(np.asarray(low, float), (n,)).copy()
            self._high = np.broadcast_to(np.asarray(high, float), (n,)).copy()
            if np.any(self._low >= self._high):
                raise ValueError("each lower bound must be < its upper bound")
        else:
            self._low = self._high = None
        self._n_resample = int(n_resample)
        if self._n_resample < 1:
            raise ValueError("n_resample must be >= 1")

        self._tol_x = tol_x * self._sigma
        self._tol_fun = tol_fun
        self._max_iterations = max_iterations
        # window of recent best-of-generation fitnesses for the TolFun test
        self._fun_window = 10 + int(math.ceil(30 * n / self._lambda))
        self._fun_history: deque = deque(maxlen=self._fun_window)

    # ------------------------------------------------------------------ #
    # Read-only state
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
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def C(self) -> np.ndarray:
        """Diagonal of the covariance matrix as a length-``n`` vector."""
        return self._C.copy()

    @property
    def std(self) -> np.ndarray:
        """Effective per-coordinate sampling std: ``sigma * sqrt(C)``."""
        return self._sigma * self._D

    @property
    def best(self) -> Solution:
        return self._best

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def count_evals(self) -> int:
        return self._count_evals

    # ------------------------------------------------------------------ #
    # Ask / tell
    # ------------------------------------------------------------------ #
    def ask(self) -> np.ndarray:
        """Sample and return a ``(lambda, n)`` array of candidate solutions.

        Each candidate is ``x = m + sigma * D * z`` with ``z ~ N(0, I)`` — the
        per-coordinate scaling by ``D = sqrt(C)`` is the whole separable model.
        """
        return np.stack([self._sample_one() for _ in range(self._lambda)])

    def _sample_one(self) -> np.ndarray:
        for _ in range(self._n_resample):
            z = self._rng.standard_normal(self._n)
            x = self._mean + self._sigma * self._D * z
            if self._low is None or np.all((x >= self._low) & (x <= self._high)):
                return x
        # Resampling budget exhausted: clip into the box as a last resort.
        return np.clip(x, self._low, self._high)

    def tell(self, solutions: ArrayLike, fitnesses: ArrayLike) -> None:
        """Update the distribution from evaluated candidates (minimization).

        ``solutions`` are the candidate vectors (as returned by :meth:`ask`, in
        any order) and ``fitnesses`` their objective values. Lower is better.
        """
        X = np.asarray(solutions, dtype=float)
        f = np.asarray(fitnesses, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._n:
            raise ValueError(f"solutions must have shape (lambda, {self._n})")
        if X.shape[0] != self._lambda:
            raise ValueError(
                f"expected {self._lambda} solutions (population_size), "
                f"got {X.shape[0]}"
            )
        if f.shape[0] != X.shape[0]:
            raise ValueError("number of fitnesses must match number of solutions")
        if not np.all(np.isfinite(X)):
            raise ValueError("solution vectors must be finite")
        if not np.all(np.isfinite(f)):
            raise ValueError("fitness values must be finite")

        self._count_evals += X.shape[0]

        # Track the best-ever sampled solution.
        gen_best_idx = int(np.argmin(f))
        if f[gen_best_idx] < self._best.f:
            self._best = Solution(x=X[gen_best_idx].copy(), f=float(f[gen_best_idx]))
        self._fun_history.append(float(f[gen_best_idx]))

        # Rank by fitness; keep the mu best. y = (x - m) / sigma is the step in
        # search space measured in units of sigma.
        order = np.argsort(f)
        x_best = X[order[: self._mu]]
        y = (x_best - self._mean) / self._sigma  # (mu, n)
        y_w = self._weights @ y  # (n,) recombined step

        n = self._n
        # --- mean update ---
        self._mean = self._mean + self._c_m * self._sigma * y_w

        # --- step-size control (CSA) ---
        # For a diagonal C, C^{-1/2} y_w = y_w / D.
        self._p_sigma = (1 - self._c_sigma) * self._p_sigma + math.sqrt(
            self._c_sigma * (2 - self._c_sigma) * self._mu_eff
        ) * (y_w / self._D)
        ps_norm = float(np.linalg.norm(self._p_sigma))
        # Clip the exponent so a pathological step saturates sigma to a huge but
        # finite value (which the 'diverged' stop condition then catches) rather
        # than raising OverflowError from math.exp.
        log_factor = (self._c_sigma / self._d_sigma) * (ps_norm / self._chi_n - 1)
        self._sigma *= math.exp(min(log_factor, 700.0))

        # --- covariance path with Heaviside stall guard ---
        denom = math.sqrt(1 - (1 - self._c_sigma) ** (2 * (self._generation + 1)))
        h_sigma = 1.0 if ps_norm / denom < (1.4 + 2 / (n + 1)) * self._chi_n else 0.0
        self._p_c = (1 - self._c_c) * self._p_c + h_sigma * math.sqrt(
            self._c_c * (2 - self._c_c) * self._mu_eff
        ) * y_w
        delta_h = (1 - h_sigma) * self._c_c * (2 - self._c_c)  # keeps E[C] unbiased

        # --- diagonal covariance update (elementwise rank-one + rank-mu) ---
        rank_one = self._c1 * (self._p_c**2 + delta_h * self._C)
        rank_mu = self._c_mu * (self._weights @ (y**2))
        self._C = (1 - self._c1 - self._c_mu) * self._C + rank_one + rank_mu
        # Numerical floor: keep variances strictly positive.
        np.maximum(self._C, 1e-300, out=self._C)
        self._D = np.sqrt(self._C)

        self._generation += 1

    # ------------------------------------------------------------------ #
    # Stopping
    # ------------------------------------------------------------------ #
    def stop(self) -> dict:
        """Return a dict of triggered stop conditions (empty ``=>`` keep going)."""
        conditions: dict = {}

        # Divergence guard first: if sigma/C blew up, report it and skip the
        # collapse checks below (whose sigma*D would itself overflow).
        if not np.all(np.isfinite(self._C)) or self._sigma > 1e60:
            conditions["diverged"] = self._sigma
            return conditions

        if (
            self._max_iterations is not None
            and self._generation >= self._max_iterations
        ):
            conditions["max_iterations"] = self._generation

        # TolX: the whole sampling distribution has collapsed below tolerance.
        step = self._sigma * self._D
        if np.all(step < self._tol_x) and np.all(
            self._sigma * np.abs(self._p_c) < self._tol_x
        ):
            conditions["tol_x"] = float(np.max(step))

        # TolFun: best-of-generation fitness has been flat over the window.
        if len(self._fun_history) >= self._fun_window:
            spread = max(self._fun_history) - min(self._fun_history)
            if spread < self._tol_fun:
                conditions["tol_fun"] = spread

        return conditions


def minimize(
    objective: Callable[[np.ndarray], float],
    x0: ArrayLike,
    sigma0: float,
    *,
    max_evals: int = 10000,
    seed: Optional[int] = None,
    maximize: bool = False,
    population_size: Optional[int] = None,
    **kwargs,
) -> OptimizeResult:
    """Optimize ``objective`` with Sep-CMA-ES.

    Minimizes by default; pass ``maximize=True`` to maximize. ``objective`` maps
    a length-``n`` vector to a scalar. Returns the best solution found within the
    ``max_evals`` budget (or earlier, if a convergence criterion fires).

    ``max_evals`` is a hard ceiling: since candidates are evaluated a full
    generation (``population_size``) at a time, the loop stops before any batch
    that would exceed the budget, so ``nfev <= max_evals`` whenever the budget
    admits at least one generation. (A budget smaller than ``population_size``
    still runs one generation, to avoid returning an unevaluated result.)
    """
    opt = SepCMAES(
        mean=x0,
        sigma=sigma0,
        seed=seed,
        population_size=population_size,
        **kwargs,
    )
    sign = -1.0 if maximize else 1.0

    stop: dict = {}
    while True:
        stop = opt.stop()
        if stop:
            break
        # Don't start a generation that would overshoot the budget (but always
        # run at least the first one).
        if opt.count_evals > 0 and opt.count_evals + opt.population_size > max_evals:
            break
        xs = opt.ask()
        fs = [sign * float(objective(np.asarray(x))) for x in xs]
        opt.tell(xs, fs)

    diverged = "diverged" in stop
    return OptimizeResult(
        x=opt.best.x,
        fun=sign * opt.best.f,  # report in the caller's original sign
        nfev=opt.count_evals,
        nit=opt.generation,
        success=not diverged,
        message="; ".join(stop) if stop else "evaluation budget reached",
        sigma=opt.sigma,
        mean=opt.mean,
        stop=stop,
    )
