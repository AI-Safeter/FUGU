"""Evolve TRINITY's router head with the PyTorch backend, on GPU if present.

This is the torch counterpart of examples/evolve_router.py. The point it makes
that the numpy version cannot: the entire Sep-CMA-ES population is scored in one
batched pass via TrinityRouterHead.batched_logits, with no Python loop over
candidates, on whatever device the tensors live on.

Run:  python examples/evolve_router_torch.py
"""

import torch

from sepcmaes.torch_backend import SepCMAES, TrinityRouterHead


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    g = torch.Generator(device=device).manual_seed(0)

    n_agents, feature_dim, n_samples = 4, 16, 256
    H = torch.randn(n_samples, feature_dim, generator=g, device=device, dtype=dtype)
    W_true = torch.randn(
        n_agents + 3, feature_dim, generator=g, device=device, dtype=dtype
    )
    z = H @ W_true.T
    a_true = z[:, :n_agents].argmax(1)  # ground-truth agent per state
    r_true = z[:, n_agents:].argmax(1)  # ground-truth role per state

    head = TrinityRouterHead(n_agents, feature_dim, dtype=dtype).to(device)

    @torch.no_grad()
    def expected_reward(theta: torch.Tensor) -> torch.Tensor:
        # theta: (lambda, n_params) -> reward per candidate: (lambda,)
        logits = head.batched_logits(theta, H)  # (lam, S, L+3)
        p_agent = torch.softmax(logits[..., :n_agents], dim=-1)
        p_role = torch.softmax(logits[..., n_agents:], dim=-1)
        lam = theta.shape[0]
        pa = p_agent.gather(-1, a_true.view(1, -1, 1).expand(lam, -1, 1)).squeeze(-1)
        pr = p_role.gather(-1, r_true.view(1, -1, 1).expand(lam, -1, 1)).squeeze(-1)
        return (pa * pr).mean(dim=1)  # E[reward] per candidate

    opt = SepCMAES(
        mean=torch.zeros(head.n_params, dtype=dtype, device=device),
        sigma=1.0,
        seed=0,
        device=device,
        dtype=dtype,
    )
    print(f"device={device}  params={head.n_params}  lambda={opt.population_size}")
    print("  gen   evals    E[reward]")
    gen = 0
    while opt.count_evals < 60_000 and not opt.stop():
        xs = opt.ask()  # (lambda, n_params)
        opt.tell(xs, -expected_reward(xs))  # one batched pass; minimize -reward
        gen += 1
        if gen % 100 == 0:
            print(f"  {gen:4d}  {opt.count_evals:6d}   {-opt.best.f:8.3f}")

    print(f"\nfinal E[reward] under stochastic policy: {-opt.best.f:.1%}")


if __name__ == "__main__":
    main()
