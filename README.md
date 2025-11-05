# flowimds

[日本語 README](docs/README.ja.md)

`flowimds` is an open-source Python library for batch image directory
processing. It lets you describe pipelines composed of reusable steps (resize,
grayscale, binarise, denoise, rotate, flip, …) and execute them against folders,
lists of file paths, or in-memory NumPy arrays.

## Table of contents

1. [Features](#features)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Usage guide](#usage-guide)
5. [Benchmarking](#benchmarking)
6. [Support](#support)
7. [Contributing](#contributing)
8. [Development setup](#development-setup)
9. [License](#license)
10. [Project status](#project-status)
11. [Acknowledgements](#acknowledgements)

## Features

- Batch processing for entire directories with optional recursive traversal.
- Configurable output structure that can mirror the input hierarchy.
- Rich standard step library: resize, grayscale, rotate, flip, binarise, and
  denoise.
- Multiple execution modes: directory scanning, explicit path lists, or pure
  in-memory processing.
- Deterministic test data generators for reproducible fixtures.

## Installation

### Requirements

- Python 3.12+
- `uv` or `pip` for dependency management

### Commands

```bash
pip install flowimds
```

or

```bash
uv add flowimds
```

### From source

```bash
git clone https://github.com/mori-318/flowimds.git
cd flowimds
uv sync
```

## Quick start

All primary classes are re-exported from the package root, so you can work with
the pipeline and processing steps through a concise namespace:

```python
import flowimds as fi

pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((128, 128)), fi.GrayscaleStep()],
    input_path="samples/input",
    output_path="samples/output",
    recursive=True,
    preserve_structure=True,
)

result = pipeline.run()
print(f"Processed {result.processed_count} images")
```

## Usage guide

- Compose pipelines from the built-in steps (resize, grayscale, rotate, flip,
  binarise, denoise) or any custom object exposing `apply(image)`.
- Run against directories, explicit file lists, or pure in-memory NumPy arrays
  depending on the data you have available.
- Inspect the returned `PipelineResult` to review processed counts, failed
  files, and where outputs were written.

See [docs/usage.md](docs/usage.md) for a complete walkthrough with code
snippets and configuration tips.

## Benchmarking

Use the benchmark helper script to compare the legacy implementation with the
current pipeline. The command below relies on `uv` so that dependencies and the
virtual environment are handled consistently:

```bash
uv run python scripts/benchmark_pipeline.py --count 5000 --workers 8
```

- `--count` controls how many synthetic images are generated (default `5000`).
- `--workers` sets the maximum parallel worker count (`0` auto-detects CPUs).

For reproducible comparisons across runs, specify `--seed` (default `42`). The
script prints timing summaries for each pipeline variant and cleans up the
temporary outputs afterward.

## Support

Questions and bug reports are welcome via the GitHub issue tracker.

## Contributing

We follow a GitFlow-based workflow to keep the library stable while enabling
parallel development:

- **main**: release-ready code (tagged as `vX.Y.Z`).
- **develop**: staging area for the next release.
- **feature/**, **release/**, **hotfix/** branches support focused work.

Before opening a pull request:

1. Check out a topic branch from `develop`.
2. Ensure lint and test commands pass (see [development setup](#development-setup)).
3. Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
   messages.

## Development setup

```bash
# Install dependencies
uv sync --all-extras --dev

# Lint and format (apply fixes when needed)
uv run black .
uv run ruff format .

# Lint and format (verify)
uv run black --check .
uv run ruff check .
uv run ruff format --check .

# Regenerate deterministic fixtures when needed
uv run python scripts/generate_test_data.py

# Run tests
uv run pytest
```

## License

This project is released under the [MIT License](LICENSE).

## Project status

The project is under active development, targeting the first public release.
Watch the repository for tagged versions once the release workflow is in place.

## Acknowledgements

- Built with [NumPy](https://numpy.org/).
- Image I/O powered by [OpenCV](https://opencv.org/).
- Packaging and workflow tooling driven by [uv](https://github.com/astral-sh/uv)
  and [Ruff](https://docs.astral.sh/ruff/).
