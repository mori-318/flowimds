# flowimds

[日本語 README](docs/README.ja.md)

A simple and easy-to-use library for batch image directory processing.

## Usage

All primary classes are re-exported from the package root, so you can work with
the pipeline and processing steps through a concise namespace:

```python
from pathlib import Path

import flowimds as fi

pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((128, 128)), fi.GrayscaleStep()],
    input_path=Path("examples/input"),
    output_path=Path("examples/output"),
    recursive=True,
    preserve_structure=True,
)

result = pipeline.run()
print(f"Processed {result.processed_count} images")
```

When you already have a list of image paths or in-memory images, the pipeline
offers dedicated helpers:

```python
# Process an explicit list of files and persist the results
result = pipeline.run_on_paths(["/tmp/example.png"])

# Transform in-memory images without touching the filesystem
from flowimds import read_image

arrays = pipeline.run_on_arrays([read_image("/tmp/example.png")])
```

See `samples/README.md` for runnable examples that generate inputs and inspect
the produced outputs.

## Branching Strategy

We follow a GitFlow-based workflow to keep the library stable while enabling parallel development.

### Main branches

- **main**: Always holds release-ready code. Every public release is tagged (for example, `v1.2.0`).
- **develop**: Aggregates the latest changes intended for the upcoming release. Most feature work branches off from here.

### Supporting branches

1. **feature/**: Short-lived branches for individual enhancements. Branch off from `develop`, open a pull request, and merge back with `--no-ff` once reviews and checks pass.
2. **release/**: Created from `develop` when preparing a release. Finalizes version numbers and docs, then merges into both `main` (with a tag) and back into `develop`.
3. **hotfix/**: Urgent fixes branched from `main`. After verification, merge into both `main` and `develop` to keep histories aligned.

## Contribution Workflow

1. Ensure `main` is up to date: `git pull origin main`.
2. Switch to `develop` (create it from `main` if it does not exist) and branch off: `git checkout develop`, `git checkout -b feature/<topic>`.
3. Commit changes following PEP 8, add tests when appropriate, and push the branch.
4. Open a pull request targeting `develop`, make sure CI passes, and obtain at least one approval.
5. After merging, delete the feature branch to keep the repository tidy.

## Commit Message Guidelines

We adopt a lightweight rule set based on [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary):

- Format: `<type>[optional scope]: <description>` with an optional body and footer.
- Types: start with `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `ci` as needed.
- Scope: optional module name in parentheses, e.g., `feat(parser): ...`.
- Description: 50 characters or fewer, written in the imperative mood.
- Body/footers: add extra context, issue references, or `BREAKING CHANGE:` when relevant.

Operational tips:

1. Keep commits focused; prefer “one topic per commit.”
2. Include test updates under the `test` type and CI/config changes under `ci`.
3. Align pull request titles with the same format for easier tracking.
4. Optionally enforce the format with tools such as commitlint.

## Code Style and Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting. The
project configuration lives in `pyproject.toml` and enforces an 88 character
line length to match Ruff's Black-compatible style.

- Run lint checks:

  ```bash
  uv run ruff check
  ```

- Format code in place:

  ```bash
  uv run ruff format
  ```

- Verify formatting without applying changes (useful for CI):

  ```bash
  uv run ruff format --check
  ```

`ruff format` does not sort imports; rely on `ruff check` for import order and
other lint diagnostics before formatting.

## Test Fixtures

The automated tests depend on image fixtures stored under `tests/data`. If you
need to recreate them (for example, after cleaning the repository), run:

```bash
uv run python scripts/generate_test_data.py
```

This command regenerates all directories inside `tests/data` with deterministic
sample images.
