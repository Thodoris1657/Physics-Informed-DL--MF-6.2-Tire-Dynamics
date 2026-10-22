# Contributing

1. `pip install -e ".[all]"`
2. Create a branch, make the change, add a test in `tests/`.
3. `ruff check src tests` and `pytest -m "not slow"` must pass; run `pytest` (including the
   end-to-end test on the bundled data) before opening a pull request.
4. Physics changes: state the equation and its source in the docstring, and keep the single
   geometry definition (`vehicle.wheel_positions`) as the only place wheel positions appear.
5. Never commit TTC data to a public repository (TTC member agreement).
