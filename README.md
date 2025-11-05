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
5. [Command-line interface](#command-line-interface)
6. [Project roadmap](#project-roadmap)
7. [Support](#support)
8. [Contributing](#contributing)
9. [Development setup](#development-setup)
10. [License](#license)
11. [Project status](#project-status)
12. [Acknowledgements](#acknowledgements)

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

### From PyPI (after publishing)

```bash
pip install flowimds
```

### From source

```bash
git clone https://github.com/your-org/flowimds.git
cd flowimds
uv sync --all-extras --dev
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

- **Explicit paths**: `pipeline.run_on_paths([...])` processes an explicit list
  of images and persists the outputs.
- **In-memory arrays**: `pipeline.run_on_arrays([...])` applies the pipeline to
  NumPy arrays without touching the filesystem.
- **Sample fixtures**: See `samples/README.md` for runnable examples that
  generate inputs and inspect the produced outputs.

## Command-line interface

The library is designed to expose a CLI in future releases. Track progress in
the [project roadmap](#project-roadmap) and follow upcoming releases for CLI
availability.

## Project roadmap

Development milestones and future work are tracked in
[`docs/plan.md`](docs/plan.md). Highlights include:

- v1.0 pipeline implementation with reusable steps and result reporting.
- Future CLI tooling for batch execution from the terminal.
- Potential AI-driven pipeline steps for classification or anomaly detection.

## Support

Questions and bug reports are welcome via the GitHub issue tracker once the
repository is public. Until then, feel free to open a discussion thread or
contact the maintainers directly.

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

# Lint and format
uv run black --check .
uv run ruff check .
uv run ruff format --check .

# Run tests
uv run pytest

# Regenerate deterministic fixtures when needed
uv run python scripts/generate_test_data.py
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
