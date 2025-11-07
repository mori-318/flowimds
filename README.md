<div align="center">
  <img src="docs/assets/flowimds_rogo.png" alt="flowimds logo" width="100%">
  <h1>flowimds</h1>
</div>

<p align="center">
  <a href="https://pypi.org/project/flowimds/"><img src="https://img.shields.io/pypi/v/flowimds.svg" alt="PyPI"></a>
  <a href="https://github.com/mori-318/flowimds/actions/workflows/publish.yml"><img src="https://img.shields.io/github/actions/workflow/status/mori-318/flowimds/publish.yml?branch=main&label=publish" alt="Publish workflow status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/mori-318/flowimds.svg" alt="License"></a>
  <a href="https://pypi.org/project/flowimds/"><img src="https://img.shields.io/pypi/pyversions/flowimds.svg" alt="Python Versions">
  </a>
</p>

> Writing one-off scripts to process every image at once is tedious. flowimds solves that pain.
> Compose pipelines to express simple, repeatable batch-processing flows.

[Japanese version](docs/README.ja.md)

## ✨ Highlights

- ♻️ **Batch processing at scale** — Traverse entire directories with optional recursive scanning.
- 🗂️ **Structure-aware outputs** — Mirror the input folder layout when preserving directory structures.
- 🧩 **Rich step library** — Combine resizing, grayscale conversion, rotations, flips, binarisation, denoising, and custom steps.
- 🔄 **Flexible execution modes** — Operate on folders, explicit file lists, or in-memory NumPy arrays.
- 🧪 **Deterministic fixtures** — Recreate test data whenever needed for reproducible pipelines.
- 🤖 **Expanding step roadmap** — More transformations, including AI-assisted steps, are planned.
- 📁 **Flattened outputs available** — Optionally disable structure preservation to write everything into a single directory.

## 🚀 Quick start

All primary classes are re-exported from the package root, so pipelines can be described through a concise namespace:

```python
# Import the flowimds package
import flowimds as fi

# Define the pipeline
# Args:
#   steps: sequence of pipeline steps
#   input_path: input directory path
#   output_path: output directory path
#   recursive: whether to traverse subdirectories (default: False)
#   preserve_structure: whether to mirror the input tree (default: True)
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((128, 128)),
        fi.GrayscaleStep(),
    ],
    input_path="samples/input",
    output_path="samples/output",
    recursive=True,
    preserve_structure=True,
)

result = pipeline.run()

# Inspect the result
# Fields:
#   processed_count: number of successfully processed images
#   failed_count: number of images that failed to process
#   failed_files: paths of the images that failed
print(f"Processed {result.processed_count} images")
```

## 📦 Installation

- Python 3.12+
- `uv` or `pip` for dependency management
- `uv` is recommended

### uv

```bash
uv add flowimds
```

### pip

```bash
pip install flowimds
```

### From source

```bash
git clone https://github.com/mori-318/flowimds.git
cd flowimds
uv sync
```

## 📚 Documentation

- [Usage guide](docs/usage.md) — configuration tips and extended examples.
- [使用ガイド](docs/usage.ja.md) — 日本語版。

## 🔬 Benchmarks

Compare the legacy (v0.2.1-) and current (v0.2.1+) pipeline implementations with the bundled helper script. Running via `uv` keeps dependencies and the virtual environment consistent:

```bash
# count: number of synthetic images to generate (default `5000`)
# workers: maximum worker threads (`0` auto-detects CPU cores)
uv run python scripts/benchmark_pipeline.py --count 5000 --workers 8
```

- `--count`: number of synthetic images to generate (default `5000`).
- `--workers`: maximum worker threads (`0` auto-detects CPU cores).
- `--seed`: specify the seed (default `42`) for reproducible comparisons.

The script prints processing times for each pipeline variant and cleans up temporary outputs afterward.

## 🆘 Support

Questions and bug reports are welcome via the GitHub issue tracker.

## 🤝 Contributing

We follow a GitFlow-based workflow to keep the library stable while enabling parallel development:

- **main** — release-ready code (tagged as `vX.Y.Z`).
- **develop** — staging area for the next release.
- **feature/** — focused branches for scoped work.
- **release/** — branches dedicated to preparing releases.
- **hotfix/** — branches for urgent fixes.
- **docs/** — branches for documentation updates.

Before opening a pull request:

1. Check out a topic branch from `develop`.
2. Ensure lint and test commands pass (see [🛠️ Development](#️-development)).
3. Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.
4. Target the `develop` branch when opening the pull request.

## 🛠️ Development

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

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📌 Project status

Stable releases are already published on PyPI (v0.2.1), and we continue to iterate toward upcoming updates. Watch the repository for new tags and changelog announcements.
