# Sep-CMA-ES: the router optimizer from TRINITY

A NumPy implementation of separable CMA-ES (Ros & Hansen, 2008): a
diagonal-covariance evolution strategy with linear time and space in the search
dimension. This is the black-box optimizer that
[TRINITY](https://arxiv.org/abs/2512.04695) uses to evolve its LLM-coordinator
router head.

TRINITY and its sibling Conductor are the research behind Sakana AI's FUGU,
their multi-LLM orchestration system. FUGU routes each query across a pool of
LLMs, assigning roles and picking which model answers; that routing decision is
what this optimizer trains. This repo implements TRINITY's side of FUGU.

No LLM is required to use this code. The optimizer is a general-purpose
gradient-free minimizer, and the router head takes a feature vector as input.

## Why separable CMA-ES?

Full CMA-ES adapts a dense `n × n` covariance matrix: `O(n²)` memory and an
`O(n³)` eigendecomposition per update. That is hopeless once `n` is in the
thousands. Sep-CMA-ES restricts the covariance to its diagonal, so every
quantity is a length-`n` vector and a generation costs `O(λ·n)`. Ros & Hansen
showed this not only scales but learns faster on separable landscapes, by
accelerating the covariance learning rate by a factor of `(n+2)/3`.

TRINITY's coordinator head has ~10⁴ parameters and a loss landscape that is
near block/axis-separable (their Definition 1), the regime where the diagonal
model loses almost nothing while saving the `O(n²)` cost.

## The two papers

Both are Sakana AI papers (ICLR 2026), two routes to the same goal of
orchestrating a pool of LLMs, at opposite ends of the design space:

| | TRINITY ([2512.04695](https://arxiv.org/abs/2512.04695)) | Conductor ([2512.04388](https://arxiv.org/abs/2512.04388)) |
|---|---|---|
| Controller | frozen 0.6B SLM + ~10K-param head | full 7B LLM, generative |
| Action | one `(agent, role)` per turn, fixed set | free-form NL plan (≤5 steps) per call |
| Optimizer | Sep-CMA-ES (this repo) | GRPO (RL) |
| Objective | `J(θ)=E_τ[R(τ)]`, binary terminal reward | shaped reward (format gate + correctness) |

This repo implements TRINITY's side: the Sep-CMA-ES optimizer and the linear
router head it evolves.

## The router head

`TrinityRouterHead` is a single bias-free linear layer `W: ℝ^d → ℝ^{L+3}`:

```
logits = W · h            # h = penultimate-token hidden state of the SLM (d=1024)
agent  = argmax(logits[:L])        # which of L worker LLMs
role   = argmax(logits[L:])        # Thinker | Worker | Verifier
```

For `L=7, d=1024` that is `10·1024 = 10,240` parameters (paper Table 6). The
flattened `W` is the vector `θ` that Sep-CMA-ES optimizes. The hidden state `h`
is an input. In the full system it comes from a frozen Qwen3-0.6B; here you
supply it, which keeps the optimizer testable without a model.

## Install

```bash
pip install -e .            # numpy only
pip install -e ".[test]"    # + pytest and the reference `cmaes` lib for cross-validation
```

## Usage

Ask-and-tell (the canonical ES loop):

```python
import numpy as np
from sepcmaes import SepCMAES

opt = SepCMAES(mean=np.zeros(10), sigma=0.5, seed=0)
while not opt.stop():
    xs = opt.ask()                     # (lambda, n) candidate solutions
    fs = [float(np.sum(x**2)) for x in xs]
    opt.tell(xs, fs)                   # minimization
print(opt.best.x, opt.best.f)
```

One-shot convenience wrapper:

```python
from sepcmaes import minimize
res = minimize(lambda x: np.sum(x**2), x0=np.full(5, 2.0), sigma0=1.0, max_evals=5000, seed=0)
res.x, res.fun, res.nfev
```

Evolving the router head (no LLM, no gradients):

```bash
python examples/evolve_router.py
# -> final E[reward] under stochastic policy : 98.4%
```

## Plugging in real LLM rollouts

The one seam to replace is the fitness function. TRINITY's objective is the
expected binary task reward, estimated by averaging `m_CMA = 16` rollouts:

```python
def fitness(theta):
    head.set_params(theta)
    rewards = []
    for _ in range(16):                     # m_CMA rollouts
        h = coordinator_slm.hidden_state(transcript)   # <- your LLM
        agent_id, role = head.decide(h, sample=True)    # stochastic policy
        reward = run_episode_and_score(agent_id, role)  # <- your task, R(τ)∈{0,1}
        rewards.append(reward)
    return np.mean(rewards)

opt.tell(xs, [-fitness(x) for x in xs])     # negate: we minimize
```

Sampling, recombination, step-size and covariance adaptation are already done
by this repo.

## Faithfulness notes

The TRINITY paper specifies only the sampling model `y = m + σ·D·z`, the
population size `λ = ⌈4 + 3 ln n⌉`, and `m_CMA = 16`; it cites Ros & Hansen
(2008) for the rest. The update equations and strategy constants here follow
Hansen's 2016 tutorial defaults plus the Ros & Hansen (2008) `(n+2)/3`
acceleration, applied to the full covariance learning rate (both the rank-one
`c1` and rank-μ `cμ` terms), matching the 2008 paper's combined `c_cov`. The
`cmaes` library applies the factor only to `cμ`; the two are empirically
equivalent on separable problems (see the cross-validation test). The default
`λ = 4 + ⌊3 ln n⌋` is the standard convention; TRINITY's `⌈4+3 ln n⌉` rounding
differs by at most one individual.

## Tests

```bash
pytest -q          # 42 tests
```

Coverage: interface invariants, determinism, convergence on sphere and the
ill-conditioned separable ellipsoid (the home turf), the diagonal adapting to
per-coordinate curvature, the `minimize`/`maximize` wrapper (including
sign-reporting and a hard `max_evals` ceiling), step-size shrinkage, stop
conditions, box-constraint handling, construction/`tell` error paths, graceful
divergence on pathological steps, the router head contract, an end-to-end
"Sep-CMA-ES evolves the head to match an oracle" test, and a cross-validation
against the reference `cmaes` library on separable landscapes (skipped if it is
not installed).

The implementation was also put through an adversarial multi-agent review
(algorithm-vs-Ros&Hansen, numerical robustness, router faithfulness, test
quality). It found no mathematical or router errors, and the robustness and
coverage findings it confirmed were fixed test-first.

Note on Rosenbrock: it is non-separable with a local optimum for `n ≥ 4`, so a
diagonal model can stall there in a single run. The test reflects the standard
remedy, independent restarts, rather than asserting a single run solves it.

## References

- Ros, R., & Hansen, N. (2008). *A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity.* PPSN X.
- Hansen, N. (2016). *The CMA Evolution Strategy: A Tutorial.* arXiv:1604.00772.
- Xu, J. et al. (2026). *TRINITY: An Evolved LLM Coordinator.* ICLR 2026. arXiv:2512.04695.
- Nielsen, S. et al. (2026). *Learning to Orchestrate Agents in Natural Language with the Conductor.* ICLR 2026. arXiv:2512.04388.
