"""Evolve TRINITY's router head with Sep-CMA-ES — no LLM, no gradients.

This mirrors TRINITY's training loop end-to-end, but swaps the (expensive,
non-differentiable) "run the multi-LLM coordinator and score the task" fitness
for a synthetic oracle so it runs in a second. The seam is exactly the one you
would replace to plug in real LLM rollouts:

    fitness(theta) = mean over m_CMA rollouts of  R(tau)        # binary reward
                     where the rollout uses head.decide(h(state)).

Run:  python examples/evolve_router.py
"""

import numpy as np

from sepcmaes import SepCMAES, TrinityRouterHead, Role


def main() -> None:
    rng = np.random.default_rng(0)
    # A tractable instance so the loop visibly converges in ~1s. TRINITY's real
    # head is L=7, d=1024 (10,240 params), optimized over ~tens of generations
    # with each candidate scored by averaging m_CMA=16 binary task rollouts.
    n_agents, feature_dim = 4, 16  # -> (4+3)*16 = 112 params
    n_samples = 64

    # Synthetic environment: hidden states + a ground-truth routing rule.
    H = rng.standard_normal((n_samples, feature_dim))
    W_true = rng.standard_normal((n_agents + 3, feature_dim))

    def oracle(h):
        z = W_true @ h
        return int(np.argmax(z[:n_agents])), int(np.argmax(z[n_agents:]))

    targets = [oracle(h) for h in H]
    head = TrinityRouterHead(n_agents=n_agents, feature_dim=feature_dim)

    def expected_reward(theta):
        """J(theta) = E_{tau ~ pi_theta}[R(tau)]: probability the *sampled*
        policy picks both the right agent and the right role, averaged over
        states. This is TRINITY's smooth, navigable surrogate for the binary
        reward (here in closed form instead of via 16 stochastic rollouts).
        A hard argmax accuracy would be a step function -> flat plateaus that
        stall the rank-based search; smoothing is what makes ES work."""
        head.set_params(theta)
        total = 0.0
        for h, (a_t, r_t) in zip(H, targets):
            p_agent, p_role = head.policy(h)
            total += p_agent[a_t] * p_role[r_t]
        return total / n_samples

    def greedy_accuracy(theta):
        head.set_params(theta)
        return (
            sum(head.decide(h) == (a, Role(r)) for h, (a, r) in zip(H, targets))
            / n_samples
        )

    # --- Sep-CMA-ES drives theta = flatten(W), dim = (7+3)*64 = 640 ---
    opt = SepCMAES(mean=np.zeros(head.n_params), sigma=1.0, seed=0)
    print(f"router head: {head.n_params} params, lambda={opt.population_size}")
    print("  gen   evals    E[reward]")
    gen = 0
    while opt.count_evals < 60_000 and not opt.stop():
        xs = opt.ask()
        opt.tell(xs, [-expected_reward(x) for x in xs])  # minimize -J(theta)
        gen += 1
        if gen % 100 == 0:
            print(f"  {gen:4d}  {opt.count_evals:6d}   {-opt.best.f:8.3f}")

    print(f"\nfinal E[reward] under stochastic policy : {-opt.best.f:.1%}")
    print(
        f"final greedy (argmax) agent+role match  : {greedy_accuracy(opt.best.x):.1%}"
    )


if __name__ == "__main__":
    main()
