# AGENTS.md — Operational rules for Codex in this repo

This file is loaded automatically at the start of every Codex session in this
repository. It specifies how to operate: environment, installs, tests, notebook runs,
git hygiene. For project context and conventions, read `SKILL.md`. For the method
specification, read `DESIGN.md`.

## Python environment

This project uses a pre-existing virtual environment at `.venv/` at the repository
root. It was cloned from the previous INR-FWI project and already contains PyTorch,
Deepwave, NumPy, Matplotlib, and their dependencies.

**Every Python or pip invocation must use the venv's interpreter.** Do not use the
system `python`, do not use Homebrew `python`, do not use `conda`. The two acceptable
patterns are:

1. Activate the venv in the shell once per session, then call `python` and `pip`
   normally:

   ```bash
   source .venv/bin/activate
   python -c "import torch; print(torch.__version__)"
   pip install <package>
   ```

2. Or invoke the venv's interpreter by full path without activation:

   ```bash
   .venv/bin/python -c "import torch; print(torch.__version__)"
   .venv/bin/pip install <package>
   ```

Pattern 1 is preferred for interactive sessions; pattern 2 is preferred for one-off
commands in scripts. Never mix them within a single multi-step task.

**Verification.** At the start of any session that will run Python code, verify the
venv is the one being used:

```bash
.venv/bin/python -c "import sys; print(sys.executable)"
```

The path must end in `.venv/bin/python`. If it does not, stop and re-activate.

## Installing packages

You have permission to install packages into `.venv` when a task genuinely requires
them. Use the venv's `pip` (see above). After any install:

1. Pin the version that was actually installed (not a wildcard) by running
   `pip freeze | grep -i <package>` and recording the exact version.
2. Add the pinned version to `pyproject.toml` under the appropriate
   `[project.dependencies]` or `[project.optional-dependencies]` group.
3. Mention the addition in the commit message.

Do not install into the system Python. Do not create a second venv.

## Running tests

```bash
.venv/bin/pytest -q
```

All tests must pass before any commit. Tests must not depend on external data files;
use the synthetic fallback in `warpfwi.data` for any test that needs a velocity model.

## Running the notebook

The notebook at `notebooks/01_warpfwi_marmousi.ipynb` is the orchestration entry
point. From the command line:

```bash
.venv/bin/jupyter lab
```

The notebook picks its device via `warpfwi.device.select_device()`. On MacOS M4 this
resolves to `cpu`; on a CUDA host it resolves to `cuda`. Do not hardcode the device.

## Git conventions

- Commit after each numbered step in the kickoff prompt completes.
- Commit messages: imperative mood, present tense, scoped prefix (e.g.
  `warp: add identity invariant test`, `notebook: expose classical FWI iters`).
- Do not commit `.venv/`, Marmousi data files, Jupyter checkpoints, or PyTorch model
  checkpoints. `.gitignore` handles these — do not bypass it.
- Do not commit generated figures unless explicitly asked.

## Hard rules

- **No `.cuda()` calls.** Ever. Use `.to(device)` with the device from
  `select_device()`.
- **No silent device fallbacks.** If the requested device is unavailable or
  unsupported by a library (e.g. MPS for Deepwave's scalar solver), raise an
  explicit error.
- **No magic numbers in training loops.** Pull from a dataclass config.
- **No inline classes or training loops in the notebook.** All logic lives in
  `src/warpfwi/`.
- **No silent scope expansion.** If a request would require features not in
  `DESIGN.md` v1, leave a TODO and ask before implementing.

## When unsure

Prefer explicit failure over silent fallback. Prefer asking over assuming. Prefer
leaving a TODO over implementing something `DESIGN.md` does not specify.
