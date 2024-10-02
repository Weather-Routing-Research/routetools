# CMA-ES Bézier

## Structure

The repository is structured into the following directories:

- `/lib`: Python source code
- `/tests`: Python code for testing via pytest

Conveniently, a set of workflows via Github Actions are already installed:

- `pre-commit`: run pre-commit hooks
- `pytest`: automatically discover and runs tests in `tests/`

Tools:

- [uv](https://docs.astral.sh/uv/): manage dependencies, Python versions and virtual environments
- [ruff](https://docs.astral.sh/ruff/): lint and format Python code
- [mypy](https://mypy.readthedocs.io/): check types
- [pytest](https://docs.pytest.org/en/): run unit tests
- [pre-commit](https://pre-commit.com/): manage pre-commit hooks
- [prettier](https://prettier.io/): format YAML and Markdown
- [codespell](https://github.com/codespell-project/codespell): check spelling in source code

## Installation

### Application

Install package and pinned dependencies with the [`uv`](https://docs.astral.sh/uv/) package manager:

1. Install `uv`. See instructions for Windows, Linux or MacOS [here](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone repository

3. Install package and dependencies in a virtual environment:

   ```{bash}
   uv sync
   ```

4. Run any command or Python script with `uv run`, for instance:

   ```{bash}
   uv run lib/main.py
   ```

   Alternatively, you can also activate the virtual env and run the scripts normally:

   ```{bash}
   source .venv/bin/activate
   ```

### Library

Install a specific version of the package with `pip` or `uv pip`:

```{bash}
pip install git+ssh://git@github.com:Weather-Routing-Research/cmaes_bezier_demo.git
```

## Setup development environment (Unix)

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and pre-commit hooks:

```{bash}
make install
```

`uv` will automatically create a virtual environment with the specified Python version in `.python-version` and install the dependencies from `uv.lock` (both standard and dev dependencies). It will also install the package in editable mode.

### Adding new dependencies

Add dependencies with:

```{bash}
uv add <PACKAGE>
```

Add dev dependencies with:

```{bash}
uv add --dev <PACKAGE>
```

Remove dependency with:

```{bash}
uv remove <PACKAGE>
```

In all cases `uv` will automatically update the `uv.lock` file and sync the virtual environment. This can also be done manually with:

```{bash}
uv sync
```

### Create a kernel for Jupyter

```{bash}
uv run ipython kernel install --user --name=cmaes
```

### Tools

#### Run pre-commit hooks

Hooks are run on modified files before any commit. To run them manually on all files use:

```{bash}
make hooks
```

#### Run linter and formatter

```{bash}
make ruff
```

#### Run tests

```{bash}
make test
```

#### Run type checker

```{bash}
make mypy
```
