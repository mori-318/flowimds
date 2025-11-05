# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- markdownlint-disable-next-line MD024 -->
### Added

- _Nothing yet_

## [0.1.2] - yyyy-mm-dd

<!-- markdownlint-disable-next-line MD024 -->
### Changed

- Introduced parallel execution option for `Pipeline` (implementation in progress).
- Performance comparison pending; see table template below for recording baseline vs. optimized results.

| Step combination | Baseline duration (s) | Parallel duration (s) | Speed-up | Notes |
| ---------------- | --------------------- | --------------------- | -------- | ----- |
| Resize → Grayscale | TBD | TBD | TBD | |
| Resize → Denoise → Rotate | TBD | TBD | TBD | |
| Custom composite flow | TBD | TBD | TBD | |

<!-- TODO: Update table once measurements are available -->

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
