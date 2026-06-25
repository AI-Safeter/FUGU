# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- PyTorch backend (`sepcmaes.torch_backend`): a device- and dtype-agnostic port
  of the optimizer and the router head (`TrinityRouterHead` is an `nn.Module`,
  with `batched_logits` for whole-population scoring). Numerically equivalent to
  the NumPy reference, checked in float64 by the test suite.
- `examples/evolve_router_torch.py`: evolves the head on GPU in one batched pass.
- Continuous integration (GitHub Actions) across Python 3.9–3.12, plus a job
  that runs the full suite with the PyTorch backend on CPU wheels.
- `CONTRIBUTING.md`, packaging metadata (classifiers, keywords, project URLs).

### Changed
- Both backends now force a stable sort when selecting the mu best, so
  tie-breaking is deterministic and identical across NumPy and PyTorch.

## [0.1.0] - 2026-06-24

### Added
- NumPy Sep-CMA-ES (Ros & Hansen, 2008): ask/tell interface and a
  `minimize`/`maximize` wrapper with a hard evaluation budget.
- `TrinityRouterHead`: the bias-free linear router head from TRINITY
  (arXiv:2512.04695), 10,240 parameters for 7 agents and a 1024-d hidden state.
- Test suite covering convergence, the separable/diagonal behavior, robustness
  and error paths, and a cross-validation against the reference `cmaes` library.
- MIT license, README, and the `evolve_router.py` example.
