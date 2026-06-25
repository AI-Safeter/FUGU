# Sep-CMA-ES: the router optimizer from TRINITY

[![tests](https://github.com/AI-Safeter/FUGU/actions/workflows/ci.yml/badge.svg)](https://github.com/AI-Safeter/FUGU/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![arXiv: TRINITY](https://img.shields.io/badge/arXiv-2512.04695-b31b1b)](https://arxiv.org/abs/2512.04695)
[![Sakana Fugu](https://img.shields.io/badge/Sakana-Fugu-7c3aed)](https://sakana.ai/fugu/)

A PyTorch implementation of separable CMA-ES (Ros & Hansen, 2008), the
gradient-free optimizer that [TRINITY](https://arxiv.org/abs/2512.04695) uses to
evolve its LLM-coordinator router head. This is the method behind
[Sakana AI's FUGU](https://sakana.ai/fugu/) multi-LLM orchestration service. A
NumPy backend with an identical API ships alongside it for dependency-light use.

No LLM is needed to run this. The optimizer is a general black-box minimizer and
the router head takes a feature vector as input, so the whole thing is testable
without a model.

## How it works

Full CMA-ES adapts a dense `n x n` covariance matrix, which costs `O(n^2)`
memory and an `O(n^3)` eigendecomposition per step. That dies once `n` reaches
the thousands. Sep-CMA-ES keeps only the diagonal, so every quantity is a
length-`n` vector and a generation costs `O(lambda * n)`. On separable problems
it also learns faster, by speeding up the covariance learning rate by `(n+2)/3`.

TRINITY's coordinator head has about 10^4 parameters and a near axis-separable
loss surface, which is the regime where the diagonal model loses almost nothing.

The head itself is one bias-free linear layer `W: R^d -> R^{L+3}`:

```
logits = W . h                 # h = hidden state of the coordinator SLM (d=1024)
agent  = argmax(logits[:L])    # which of L worker LLMs
role   = argmax(logits[L:])    # Thinker | Worker | Verifier
```

For `L=7, d=1024` that is `10 * 1024 = 10,240` parameters (paper Table 6). The
flattened `W` is the vector `theta` that Sep-CMA-ES optimizes; the hidden state
`h` is an input you supply (a frozen Qwen3-0.6B in the full system).

## Install

```bash
pip install -e ".[torch]"        # PyTorch backend
pip install -e ".[torch,test]"   # + pytest and the reference cmaes lib for the full suite
```

## Usage

The PyTorch path keeps the optimizer, the head, and the model on one device:

```python
import torch
from sepcmaes.torch_backend import SepCMAES, TrinityRouterHead

dev = "cuda" if torch.cuda.is_available() else "cpu"
head = TrinityRouterHead(n_agents=7, feature_dim=1024).to(dev)   # an nn.Module
opt = SepCMAES(mean=torch.zeros(head.n_params, dtype=torch.float64, device=dev),
               sigma=1.0, seed=0, device=dev)

while not opt.stop():
    xs = opt.ask()                       # (lambda, n_params) on `dev`
    logits = head.batched_logits(xs, H)  # score the whole population in one einsum
    opt.tell(xs, fitnesses)              # minimization
print(opt.best.x, opt.best.f)
```

The optimizer defaults to float64 (it converges to ~1e-15; float32 stalls near
~1e-6); the head defaults to float32 to match LLM activations.
`batched_logits(theta, H)` scores `K` candidate parameter vectors over `S` states
in one pass and returns `(K, S, L+3)`, so a whole population is evaluated across
rollouts without a Python loop. `examples/evolve_router_torch.py` reaches 99.6%
on a synthetic task (GPU if present).

The NumPy backend mirrors the same API under `from sepcmaes import SepCMAES`. A
one-shot wrapper exists in both:

```python
from sepcmaes.torch_backend import minimize
res = minimize(lambda x: (x**2).sum(), x0=torch.full((5,), 2.0), sigma0=1.0, seed=0)
res.x, res.fun, res.nfev
```

## Plugging in real LLM rollouts

The one seam to replace is the fitness function. TRINITY's objective is the
expected binary task reward, estimated by averaging `m_CMA = 16` rollouts:

```python
def fitness(theta):
    head.set_params(theta)
    rewards = []
    for _ in range(16):                              # m_CMA rollouts
        h = coordinator_slm.hidden_state(transcript) # <- your LLM
        agent_id, role = head.decide(h, sample=True)  # stochastic policy
        rewards.append(run_episode_and_score(agent_id, role))  # <- your task, R in {0,1}
    return sum(rewards) / len(rewards)

opt.tell(xs, [-fitness(x) for x in xs])              # negate: we minimize
```

Sampling, recombination, step-size and covariance adaptation are handled here.

## Tests

```bash
pytest -q          # torch tests skip if torch is not installed
```

The suite checks interface invariants, determinism, convergence on the sphere
and the ill-conditioned separable ellipsoid, the `minimize` wrapper and its
`max_evals` ceiling, stop conditions, the router head contract, and an
end-to-end "evolve the head to match an oracle" run. It cross-validates against
the reference `cmaes` library on separable landscapes, and feeds the NumPy and
torch optimizers the same population and fitnesses to assert their state matches
in float64.

## References

- Ros, R., & Hansen, N. (2008). *A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity.* PPSN X.
- Hansen, N. (2016). *The CMA Evolution Strategy: A Tutorial.* arXiv:1604.00772.
- Xu, J. et al. (2026). *TRINITY: An Evolved LLM Coordinator.* ICLR 2026. [arXiv:2512.04695](https://arxiv.org/abs/2512.04695).
- Sakana AI (2026). *Sakana Fugu: One Model to Command Them All.* [sakana.ai/fugu](https://sakana.ai/fugu/).
