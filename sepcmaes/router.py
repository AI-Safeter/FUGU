"""TRINITY's evolved coordinator head: "the router" optimized by Sep-CMA-ES.

The router is a single **bias-free linear layer** that turns a coordinator
SLM's penultimate-token hidden state ``h in R^d`` into a routing decision:

    logits = W h,     W in R^{(L + 3) x d}

The first ``L`` logits select one of ``L`` worker LLMs; the last 3 select a
role in {Thinker, Worker, Verifier}. A softmax over each group defines the
policy ``pi_theta(a | s)``. The flattened ``W`` is the parameter vector ``theta``
that :class:`~sepcmaes.SepCMAES` evolves.

This module is deliberately **LLM-free**: ``h`` is an input. In the full TRINITY
system ``h`` comes from a frozen Qwen3-0.6B (d = 1024) run over the dialogue
transcript; here any feature source works, which is what lets the optimizer be
developed and tested without a model in the loop.

Reference: Xu et al., "TRINITY: An Evolved LLM Coordinator", ICLR 2026,
Sec. 3.1, Eq. 5; Table 6 (linear head = 10,240 params at L=7, d=1024).
"""

from __future__ import annotations

import enum
from typing import Optional, Tuple

import numpy as np


class Role(enum.IntEnum):
    """The three roles a selected LLM can be assigned each turn (TRINITY §2)."""

    THINKER = 0
    WORKER = 1
    VERIFIER = 2


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)  # shift for numerical stability
    e = np.exp(z)
    return e / e.sum()


class TrinityRouterHead:
    """Bias-free linear routing head ``W: R^d -> R^(L+3)``.

    Parameters
    ----------
    n_agents : int
        Number ``L`` of worker LLMs to choose among.
    feature_dim : int
        Dimension ``d`` of the input hidden state (1024 for Qwen3-0.6B).
    """

    N_ROLES = 3  # Thinker, Worker, Verifier

    def __init__(self, n_agents: int, feature_dim: int):
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        self.n_agents = int(n_agents)
        self.feature_dim = int(feature_dim)
        self._out = self.n_agents + self.N_ROLES
        # W stored as (out, d); initialized to zero (an unbiased, undecided head).
        self._W = np.zeros((self._out, self.feature_dim))

    # ------------------------------------------------------------------ #
    # Parameter vector <-> weight matrix (the Sep-CMA-ES interface)
    # ------------------------------------------------------------------ #
    @property
    def n_roles(self) -> int:
        return self.N_ROLES

    @property
    def n_params(self) -> int:
        """Length of the flat parameter vector theta = (L + 3) * d."""
        return self._out * self.feature_dim

    def get_params(self) -> np.ndarray:
        """Return a flat copy of theta (row-major over W)."""
        return self._W.reshape(-1).copy()

    def set_params(self, theta: np.ndarray) -> "TrinityRouterHead":
        """Load a flat parameter vector theta into W. Returns self for chaining."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(f"theta must have shape ({self.n_params},)")
        self._W = theta.reshape(self._out, self.feature_dim)
        return self

    # ------------------------------------------------------------------ #
    # Forward / policy / decision
    # ------------------------------------------------------------------ #
    def logits(self, h: np.ndarray) -> np.ndarray:
        """Affine-free forward pass: returns the (L + 3,) logit vector ``W h``."""
        h = np.asarray(h, dtype=float)
        if h.shape != (self.feature_dim,):
            raise ValueError(f"h must have shape ({self.feature_dim},)")
        return self._W @ h

    def policy(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(p_agent, p_role)``: softmax distributions over each group."""
        z = self.logits(h)
        return _softmax(z[: self.n_agents]), _softmax(z[self.n_agents :])

    def decide(
        self,
        h: np.ndarray,
        *,
        sample: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, Role]:
        """Pick ``(agent_id, role)`` for the current state ``h``.

        ``sample=False`` (default) decodes greedily via argmax, which is
        TRINITY's deployment-time behavior. ``sample=True`` draws from the softmax policy
        (used to estimate the stochastic objective J(theta) = E[R(tau)]).
        """
        p_agent, p_role = self.policy(h)
        if sample:
            rng = rng or np.random.default_rng()
            agent_id = int(rng.choice(self.n_agents, p=p_agent))
            role = Role(int(rng.choice(self.N_ROLES, p=p_role)))
        else:
            agent_id = int(np.argmax(p_agent))
            role = Role(int(np.argmax(p_role)))
        return agent_id, role
