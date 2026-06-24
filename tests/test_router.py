"""Behavioral contract for TRINITY's evolved coordinator head ("the router").

The router is a *bias-free linear head* W in R^{(L+3) x d}: it maps the
penultimate-token hidden state h in R^d of the coordinator SLM to L agent-
selection logits plus 3 role logits (Thinker / Worker / Verifier), under a
softmax policy. Its flattened weights are exactly the parameter vector that
Sep-CMA-ES optimizes. No LLM is invoked here: ``h`` is supplied by the caller.

Reference: Xu et al., "TRINITY: An Evolved LLM Coordinator", ICLR 2026,
Sec. 3.1 & Eq. 5 (linear head, 10,240 params for L=7, d=1024).
"""

import numpy as np
import pytest

from sepcmaes import SepCMAES
from sepcmaes.router import TrinityRouterHead, Role


def test_trinity_default_param_count_is_10240():
    """L=7 agents + 3 roles, d=1024 => (7+3)*1024 = 10,240 params (paper Table 6)."""
    head = TrinityRouterHead(n_agents=7, feature_dim=1024)
    assert head.n_params == 10_240


def test_three_roles_thinker_worker_verifier():
    assert [r.name for r in Role] == ["THINKER", "WORKER", "VERIFIER"]
    head = TrinityRouterHead(n_agents=4, feature_dim=8)
    assert head.n_roles == 3


def test_param_roundtrip():
    head = TrinityRouterHead(n_agents=5, feature_dim=16)
    theta = np.arange(head.n_params, dtype=float)
    head.set_params(theta)
    assert np.array_equal(head.get_params(), theta)


def test_logits_shape_and_no_bias():
    head = TrinityRouterHead(n_agents=7, feature_dim=12)
    head.set_params(np.zeros(head.n_params))  # W = 0
    h = np.random.default_rng(0).standard_normal(12)
    logits = head.logits(h)
    assert logits.shape == (7 + 3,)
    # bias-free: zero weights => zero logits regardless of input
    assert np.allclose(logits, 0.0)


def test_decide_returns_valid_agent_and_role():
    head = TrinityRouterHead(n_agents=7, feature_dim=32)
    head.set_params(np.random.default_rng(1).standard_normal(head.n_params))
    h = np.random.default_rng(2).standard_normal(32)
    agent_id, role = head.decide(h)
    assert 0 <= agent_id < 7
    assert isinstance(role, Role)


def test_policy_probabilities_are_valid_distributions():
    head = TrinityRouterHead(n_agents=7, feature_dim=32)
    head.set_params(np.random.default_rng(3).standard_normal(head.n_params))
    h = np.random.default_rng(4).standard_normal(32)
    p_agent, p_role = head.policy(h)
    assert p_agent.shape == (7,) and p_role.shape == (3,)
    assert np.isclose(p_agent.sum(), 1.0) and np.isclose(p_role.sum(), 1.0)
    assert np.all(p_agent >= 0) and np.all(p_role >= 0)


def test_sampling_is_seeded_and_reproducible():
    head = TrinityRouterHead(n_agents=7, feature_dim=32)
    head.set_params(np.random.default_rng(5).standard_normal(head.n_params))
    h = np.random.default_rng(6).standard_normal(32)
    a = head.decide(h, rng=np.random.default_rng(99), sample=True)
    b = head.decide(h, rng=np.random.default_rng(99), sample=True)
    assert a == b


def test_sampling_is_stochastic_and_differs_from_greedy():
    """With a near-flat policy (W=0 => uniform softmax), sampling must actually
    explore: it should produce more than one distinct decision across seeds, and
    differ from the deterministic argmax decode at least sometimes. This guards
    against a 'sample' path that secretly just returns argmax."""
    head = TrinityRouterHead(n_agents=7, feature_dim=8)
    head.set_params(np.zeros(head.n_params))  # uniform policy over 7x3 actions
    h = np.ones(8)
    greedy = head.decide(h)  # argmax of a tie -> (0, THINKER)
    sampled = [
        head.decide(h, rng=np.random.default_rng(s), sample=True) for s in range(50)
    ]
    assert len(set(sampled)) > 1  # genuinely random
    assert any(s != greedy for s in sampled)  # not a no-op argmax


# --------------------------------------------------------------------------- #
# The whole point: Sep-CMA-ES can *evolve* the router head end-to-end, with no
# LLM and no gradients — exactly TRINITY's training loop, but against a
# synthetic oracle objective so it runs in milliseconds.
# --------------------------------------------------------------------------- #
def test_sepcmaes_evolves_router_to_match_an_oracle():
    rng = np.random.default_rng(0)
    n_agents, feat = 3, 8

    # A synthetic, linearly-separable routing task: random features, and a
    # hidden "ground-truth" routing rule the head must rediscover.
    n_samples = 40
    H = rng.standard_normal((n_samples, feat))
    W_true = rng.standard_normal((n_agents + 3, feat))

    def oracle(h):
        z = W_true @ h
        return int(np.argmax(z[:n_agents])), int(np.argmax(z[n_agents:]))

    targets = [oracle(h) for h in H]

    head = TrinityRouterHead(n_agents=n_agents, feature_dim=feat)

    def fitness(theta):
        head.set_params(theta)
        correct = 0
        for h, (a_t, r_t) in zip(H, targets):
            a, r = head.decide(h)  # greedy (argmax) decode
            correct += (a == a_t) + (r.value == r_t)
        return correct / (2 * n_samples)  # fraction of correct decisions

    # untrained baseline at the all-zero vector ties everything -> ~chance
    head.set_params(np.zeros(head.n_params))
    baseline = fitness(np.zeros(head.n_params))

    opt = SepCMAES(mean=np.zeros(head.n_params), sigma=1.0, seed=0)
    best = -np.inf
    while opt.count_evals < 30_000 and not opt.stop():
        xs = opt.ask()
        opt.tell(xs, [-fitness(x) for x in xs])  # minimize negative accuracy
        best = max(best, -opt.best.f)

    assert best > 0.9  # near-perfectly matches the oracle
    assert best > baseline + 0.3  # and is a large improvement over untrained
