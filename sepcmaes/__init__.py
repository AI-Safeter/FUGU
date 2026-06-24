"""Sep-CMA-ES: separable Covariance Matrix Adaptation Evolution Strategy.

The black-box optimizer behind TRINITY's evolved LLM coordinator head
(Ros & Hansen, 2008; Xu et al., TRINITY, ICLR 2026).
"""

from .optimizer import SepCMAES, minimize, OptimizeResult, Solution
from .router import TrinityRouterHead, Role

__all__ = [
    "SepCMAES",
    "minimize",
    "OptimizeResult",
    "Solution",
    "TrinityRouterHead",
    "Role",
]
