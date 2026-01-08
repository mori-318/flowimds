<div align="center">
  <img src="docs/assets/flowimds_rogo.png" alt="flowimds logo" width="100%">
  <h1>flowimds </h1>
</div>

<p align="center">
  <a href="https://pypi.org/project/flowimds/"><img src="https://img.shields.io/pypi/v/flowimds.svg" alt="PyPI"></a>
  <a href="https://github.com/mori-318/flowimds/actions/workflows/publish.yml"><img src="https://img.shields.io/github/actions/workflow/status/mori-318/flowimds/publish.yml?branch=main&label=publish" alt="Publish workflow status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/mori-318/flowimds.svg" alt="License"></a>
  <a href="https://pypi.org/project/flowimds/"><img src="https://img.shields.io/pypi/pyversions/flowimds.svg" alt="Python Versions">
  </a>
</p>

Flowimds delivers reusable image-processing pipelines for entire directories—compose steps and let the tool handle the batch work for you.

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
#   worker_count: number of parallel workers (default: ~70% of CPU cores)
#   log: whether to show progress bar (default: False)
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((128, 128)),
        fi.GrayscaleStep(),
    ],
)

# Run the pipeline
# Args:
#   input_path: directory to scan for images
#   recursive: whether to traverse subdirectories (default: False)
result = pipeline.run(input_path="samples/input", recursive=True)

# Save the results
# Args:
#   output_path: destination directory
#   preserve_structure: whether to mirror the input tree (default: False)
result.save("samples/output", preserve_structure=True)

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

Compare the legacy (v0.2.1-) and current (v1.0.2+) pipeline implementations with the bundled helper script. Running via `uv` keeps dependencies and the virtual environment consistent:

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

For contribution flow details, see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) or the Japanese guide [docs/CONTRIBUTING_ja.md](docs/CONTRIBUTING_ja.md).

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

### Docker powered environment

You can standardize the development environment inside containers built from `docker/Dockerfile`. Dependencies are installed with `uv sync --all-extras --dev` during build, so any `uv` command (e.g., `uv run pytest`) is reproducible.

Two typical workflows exist:

1. **Run the suite once in a disposable container** (container exits when tests finish):

   ```bash
   docker compose -f docker/docker-compose.yml up --build
   ```

2. **Open an interactive shell for iterative work** (recommended while developing):

   ```bash
   # Build the image (no-op if cached)
   docker compose -f docker/docker-compose.yml build

   # Start an interactive container with a clean shell
   docker compose -f docker/docker-compose.yml run --rm app bash

   # Inside the container (already at /app)
   uv sync --all-extras --dev   # install deps into the mounted .venv
   uv run pytest
   uv run black --check .
   ```

`docker compose exec app ...` works only while a container started with `up` is still running. Because the default command runs `uv run pytest` and exits immediately, use `run --rm app bash` whenever you need an interactive session.

### Dev Container

A VS Code Dev Container configuration is provided under `.devcontainer/`. If you use the **Dev Containers** extension, you can open this repository in a container and work inside a reproducible Docker-based development environment.

#### Using with VS Code

1. Install and start Docker.
2. Install the "Dev Containers" extension in VS Code (if you do not have it yet).
3. Open this repository in VS Code and run "Dev Containers: Reopen in Container" from the command palette.
4. Inside the container, install dependencies and run the usual development commands:

     ```bash
     uv sync --all-extras --dev
     uv run pytest
     uv run black --check .
     ```

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📌 Project status

Stable releases are already published on PyPI (v1.0.2), and we continue to iterate toward upcoming updates. Watch the repository for new tags and changelog announcements.
