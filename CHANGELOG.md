# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2025-11-06

### Added

- Enabled semantic highlighting support so editors surface pipeline symbols with richer colourisation.

### Fixed

- Generate unique filenames (e.g., `_no2`) when preserving a flat output structure to avoid overwriting results.

## [0.2.0] - 2025-11-06

<!-- markdownlint-disable-next-line MD024 -->
### Changed

- Introduced parallel execution option for `Pipeline` (implementation in progress).
  - By default we auto-detect the worker count (~70% of logical cores) but fall back to sequential execution when that heuristic yields 1 worker.
- Introduced logging option for `Pipeline` (implementation in progress).
  - Disabled by default.
  - Combines `print` output with an optional `tqdm` progress bar when enabled.
- Recorded benchmark results using `uv run python scripts/benchmark_pipeline.py` (auto-detected 6 worker threads on an 8-core machine; measured on a MacBook Air (13-inch, 2024) with Apple M3/16 GB RAM; both logging modes captured).

| Step combination | Baseline duration (s) | Parallel duration (log on) (s) | Parallel duration (log off) (s) | Speed-up (log on) | Speed-up (log off) |
| ---------------- | --------------------- | ----------------------------- | ------------------------------ | ----------------- | ------------------ |
| Resize → Grayscale | 0.72 | 0.35 | 0.30 | 2.06× | 2.40× |
| Resize → Denoise → Rotate | 3.91 | 1.32 | 0.97 | 2.96× | 4.03× |
| Custom composite flow | 1.12 | 0.63 | 0.59 | 1.78× | 1.90× |

## [0.1.1] - 2025-11-05

<!-- markdownlint-disable-next-line MD024 -->
### Fixed

- Ensure the PyPI distribution bundles pipeline step modules to avoid `ModuleNotFoundError` when importing `flowimds.steps`.

## [0.1.0] - 2025-11-05

<!-- markdownlint-disable-next-line MD024 -->
### Added

- Comprehensive usage guides in `docs/usage.md` and `docs/usage.ja.md`, including pipeline configuration details and step references.
- Unit tests that verify `Pipeline.run` and `Pipeline.run_on_paths` emit clear errors when required paths are missing.

### Changed
<!-- markdownlint-disable-next-line MD024 -->

- `Pipeline` now treats `input_path` and `output_path` as optional at construction time, while enforcing them for methods that require filesystem access.
- README (English and Japanese) usage sections now summarise key points and link to the detailed guides.
- Packaging metadata moved to `[project.urls]` to satisfy modern setuptools validation.
