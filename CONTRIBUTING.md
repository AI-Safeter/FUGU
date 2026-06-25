# Contributing

Thanks for considering a contribution. This is a small, test-first codebase, so
the bar for getting started is low.

## Setup

```bash
git clone https://github.com/AI-Safeter/FUGU
cd FUGU
pip install -e ".[test]"        # numpy + pytest + cmaes (for cross-validation)
pip install torch               # optional, only for the torch backend
```

## Running the tests

```bash
pytest -q
```

The 12 PyTorch tests skip automatically if `torch` is not installed, so the
suite is green either way. With `torch` present you also get the
numpy-vs-torch equivalence checks and (if a GPU is visible) the CUDA tests.

## How the code is laid out

- `sepcmaes/optimizer.py`: the NumPy Sep-CMA-ES (the reference implementation).
- `sepcmaes/router.py`: the NumPy TRINITY router head.
- `sepcmaes/torch_backend.py`: the PyTorch port of both, numerically equivalent
  to the NumPy reference (a test asserts this in float64).
- `tests/`: one file per area. `examples/`: runnable demos.

## Conventions

- **Test-first.** Add a failing test, watch it fail, then make it pass. Bug
  fixes start with a regression test that reproduces the bug.
- **Keep the two backends equivalent.** If you change an update equation, change
  it in both `optimizer.py` and `torch_backend.py`, and make sure
  `test_torch_update_matches_numpy_exactly` still passes. Both backends use a
  stable sort so tie-breaking matches.
- **Cite the source.** The algorithm is Ros & Hansen (2008); the router is from
  TRINITY (arXiv:2512.04695). Reference equations by their source when relevant.

## Pull requests

Open a PR against `main` with a short description of what changed and why, and
confirm `pytest -q` is green. Small, focused PRs are easiest to review.
