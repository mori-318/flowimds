# AGENTS.md

## About the flowimds repository

flowimds is the open-source image-processing pipeline library that provides reusable, directory-wide image-processing pipelines. Compose steps like resizing, grayscale conversion, rotations, flips, binarization, and denoising, then run them as batch jobs that mirror the input folder hierarchy or flatten outputs into a single directory. The project targets Python 3.12+, ships under the MIT License, and is published on PyPI for easy installation.

## Directory structure

- `flowimds/` – Core package modules, including the pipeline orchestration logic, reusable step implementations, and utilities.
- `docs/` – Documentation set (English and Japanese), contribution guides, and assets used across README variants.
- `tests/` – Unit and integration suites plus shared fixtures for validating pipeline behaviors.
- `samples/` – Lightweight usage examples demonstrating how to compose and run pipelines.
- `scripts/` – Helper tooling for benchmarking, generating release notes, and creating deterministic test data.
- `.devcontainer/` – Development container definition (Dockerfile and devcontainer.json) for a reproducible VS Code / Codespaces environment.
- `.github/` – Issue/PR templates plus CI workflows that run publishing and validation pipelines.

