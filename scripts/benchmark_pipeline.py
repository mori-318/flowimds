"""Benchmark tools for measuring pipeline throughput."""

import argparse
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from flowimds.pipeline import Pipeline
import flowimds.pipeline_old as pipeline_old
from flowimds.steps import (
    BinarizeStep,
    DenoiseStep,
    GrayscaleStep,
    PipelineStep,
    ResizeStep,
    RotateStep,
)
from flowimds.utils.image_discovery import IMAGE_SUFFIXES, collect_image_paths
from flowimds.utils.image_io import read_image, write_image


@dataclass
class DatasetStats:
    """Metadata describing the generated benchmark dataset."""

    generated_images: int
    generation_seconds: float


@dataclass
class ComparisonResult:
    """Summary of legacy and current pipeline benchmark timings."""

    label: str
    legacy_seconds: float
    new_sequential_seconds: float
    new_parallel_logged_seconds: float
    new_parallel_quiet_seconds: float
    notes: str = ""

    def speedup_sequential(self) -> float:
        """Return new sequential speed-up relative to legacy sequential."""

        if self.new_sequential_seconds <= 0:
            return math.nan
        return self.legacy_seconds / self.new_sequential_seconds

    def speedup_parallel_logged(self) -> float:
        """Return logged parallel speed-up relative to legacy sequential."""

        if self.new_parallel_logged_seconds <= 0:
            return math.nan
        return self.legacy_seconds / self.new_parallel_logged_seconds

    def speedup_parallel_quiet(self) -> float:
        """Return quiet parallel speed-up relative to legacy sequential."""

        if self.new_parallel_quiet_seconds <= 0:
            return math.nan
        return self.legacy_seconds / self.new_parallel_quiet_seconds

    def to_markdown_rows(self) -> list[str]:
        """Return markdown rows comparing legacy, new sequential, and parallel."""

        sequential_speedup = self.speedup_sequential()
        sequential_speedup_str = (
            f"{sequential_speedup:.2f}×" if math.isfinite(sequential_speedup) else "N/A"
        )
        parallel_logged_speedup = self.speedup_parallel_logged()
        parallel_logged_speedup_str = (
            f"{parallel_logged_speedup:.2f}×"
            if math.isfinite(parallel_logged_speedup)
            else "N/A"
        )
        parallel_quiet_speedup = self.speedup_parallel_quiet()
        parallel_quiet_speedup_str = (
            f"{parallel_quiet_speedup:.2f}×"
            if math.isfinite(parallel_quiet_speedup)
            else "N/A"
        )
        return [
            (
                f"| {self.label} | Legacy sequential | "
                f"{self.legacy_seconds:.2f} | baseline | {self.notes} |"
            ),
            (
                f"| {self.label} | New sequential | "
                f"{self.new_sequential_seconds:.2f} | {sequential_speedup_str} | "
                f"{self.notes} |"
            ),
            (
                f"| {self.label} | New parallel (log) | "
                f"{self.new_parallel_logged_seconds:.2f} | "
                f"{parallel_logged_speedup_str} | "
                f"{self.notes} |"
            ),
            (
                f"| {self.label} | New parallel (quiet) | "
                f"{self.new_parallel_quiet_seconds:.2f} | "
                f"{parallel_quiet_speedup_str} | "
                f"{self.notes} |"
            ),
        ]


@dataclass(frozen=True)
class BenchmarkDefinition:
    """Configuration describing a benchmark scenario."""

    label: str
    steps_factory: Callable[[], Iterable[PipelineStep]]
    notes: str = ""
    recursive: bool = False
    preserve_structure: bool = False

    def build_steps(self) -> list[PipelineStep]:
        """Return a fresh list of pipeline steps."""

        return list(self.steps_factory())

    def slug(self) -> str:
        """Return a filesystem-friendly slug derived from ``label``."""

        slug = re.sub(r"[^a-z0-9]+", "-", self.label.lower())
        return slug.strip("-") or "benchmark"


def _effective_workers(requested: int) -> int:
    """Normalise the requested worker count for parallel execution."""

    if requested > 0:
        return requested
    return max(1, os.cpu_count() or 1)


def _rng(seed: int) -> np.random.Generator:
    """Return a NumPy random number generator initialised with ``seed``."""

    return np.random.default_rng(seed)


def _random_shape(
    generator: np.random.Generator,
    *,
    channels: int = 3,
    min_size: int = 64,
    max_size: int = 256,
) -> tuple[int, ...]:
    """Return a random image shape for benchmarking."""

    height = int(generator.integers(min_size, max_size))
    width = int(generator.integers(min_size, max_size))
    if channels <= 0:
        return (height, width)
    return (height, width, channels)


def _random_image(shape: tuple[int, ...], generator: np.random.Generator) -> np.ndarray:
    """Return pseudo-random image data for a given ``shape``."""

    return generator.integers(0, 256, size=shape, dtype=np.uint8)


def _apply_steps(
    steps: Iterable[PipelineStep],
    image: np.ndarray,
) -> np.ndarray:
    """Apply ``steps`` sequentially to ``image``.

    Args:
        steps: Iterable of pipeline steps to run.
        image: Input image represented as a NumPy array.

    Returns:
        np.ndarray: Transformed image produced by the final step.
    """

    result = image
    for step in steps:
        result = step.apply(result)
    return result


def _destination_path(
    source: Path,
    input_root: Path,
    output_root: Path,
    *,
    preserve_structure: bool,
) -> Path:
    """Return the output path corresponding to ``source``.

    Args:
        source: Path to the source image.
        input_root: Root directory containing all source images.
        output_root: Directory under which transformed images are written.
        preserve_structure: Whether to mirror the input directory hierarchy.

    Returns:
        Path: Destination path where the transformed image should be stored.
    """

    if preserve_structure:
        try:
            relative = source.relative_to(input_root)
        except ValueError:
            relative = Path(source.name)
        destination = output_root / relative
    else:
        destination = output_root / source.name
    return destination


def _process_single_image(
    source: Path,
    *,
    steps_factory: Callable[[], Iterable[PipelineStep]],
    input_root: Path,
    output_root: Path,
    preserve_structure: bool,
) -> Path:
    """Process ``source`` with steps from ``steps_factory`` and persist output.

    Args:
        source: Path to the image to transform.
        steps_factory: Callable that returns fresh pipeline steps.
        input_root: Root directory containing the dataset images.
        output_root: Directory in which outputs are written.
        preserve_structure: Whether to mirror the input directory hierarchy.

    Returns:
        Path: Destination path that was written.

    Raises:
        RuntimeError: If reading or writing the image fails.
    """

    steps = list(steps_factory())
    image = read_image(str(source))
    if image is None:
        msg = f"Failed to read image: {source}"  # pragma: no cover - defensive
        raise RuntimeError(msg)
    transformed = _apply_steps(steps, image)
    destination = _destination_path(
        source,
        input_root,
        output_root,
        preserve_structure=preserve_structure,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not write_image(str(destination), transformed):
        msg = f"Failed to write image: {destination}"  # pragma: no cover - defensive
        raise RuntimeError(msg)
    return destination


def _generate_benchmark_dataset(directory: Path, count: int, seed: int) -> DatasetStats:
    """Populate ``directory`` with ``count`` synthetic images.

    Args:
        directory: Destination directory where images will be written.
        count: Number of synthetic images to generate.
        seed: Seed value for deterministic random data generation.

    Returns:
        Dataset statistics including the number of generated images and the
        elapsed generation time.
    """

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)

    start = perf_counter()
    for index in range(count):
        shape = _random_shape(rng)
        image = _random_image(shape, rng)
        write_image(str(directory / f"image_{index:04d}.png"), image)
    generation_seconds = perf_counter() - start
    return DatasetStats(generated_images=count, generation_seconds=generation_seconds)


def _run_pipeline_sequential(
    steps: Iterable[PipelineStep],
    input_dir: Path,
    output_dir: Path,
    *,
    recursive: bool,
    preserve_structure: bool,
) -> float:
    """Execute ``steps`` sequentially using :class:`Pipeline`.

    Args:
        steps: Pipeline steps to execute.
        input_dir: Directory containing source images.
        output_dir: Directory where outputs are written.
        recursive: Whether to traverse ``input_dir`` recursively.
        preserve_structure: Whether to mirror the input directory structure.

    Returns:
        Execution duration in seconds.
    """

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(
        steps=list(steps),
        recursive=recursive,
        preserve_structure=preserve_structure,
        worker_count=1,
        log=True,
    )

    start = perf_counter()
    result = pipeline.run(input_path=input_dir)
    result.save(output_dir)
    return perf_counter() - start


def _run_pipeline_parallel(
    steps_factory: Callable[[], Iterable[PipelineStep]],
    input_dir: Path,
    output_dir: Path,
    *,
    recursive: bool,
    preserve_structure: bool,
    workers: int,
) -> float:
    """Execute ``steps_factory`` across images in parallel and return duration.

    Args:
        steps_factory: Callable yielding fresh pipeline steps per image.
        input_dir: Directory containing source images.
        output_dir: Directory where outputs are written.
        recursive: Whether to traverse ``input_dir`` recursively.
        preserve_structure: Whether to mirror the input directory structure.
        workers: Requested maximum worker threads (``0`` auto-detects).

    Returns:
        Total execution time in seconds.
    """

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(
        input_dir,
        recursive=recursive,
        suffixes=IMAGE_SUFFIXES,
    )
    if not image_paths:
        return 0.0

    worker_count = _effective_workers(workers)
    pipeline = Pipeline(
        steps=list(steps_factory()),
        recursive=recursive,
        preserve_structure=preserve_structure,
        worker_count=worker_count if worker_count > 0 else None,
        log=False,
    )

    start = perf_counter()
    result = pipeline.run(input_paths=image_paths)
    result.save(output_dir)
    return perf_counter() - start


def _run_definition(
    definition: BenchmarkDefinition,
    *,
    dataset_dir: Path,
    sequential_root: Path,
    parallel_root: Path,
    workers: int,
) -> ComparisonResult:
    """Execute legacy and new pipeline benchmarks for ``definition``.

    Args:
        definition: Benchmark scenario definition.
        dataset_dir: Directory containing source images.
        sequential_root: Base directory for sequential outputs.
        parallel_root: Base directory for parallel outputs.
        workers: Requested worker count for the parallel run.

    Returns:
        Comparison result containing timings for sequential and parallel runs.
    """

    legacy_dir = sequential_root / f"legacy-{definition.slug()}"
    new_sequential_dir = sequential_root / f"new-{definition.slug()}"
    parallel_logged_dir = parallel_root / f"logged-{definition.slug()}"
    parallel_quiet_dir = parallel_root / f"quiet-{definition.slug()}"

    legacy_pipeline = pipeline_old.Pipeline(
        steps=definition.build_steps(),
        input_path=dataset_dir,
        output_path=legacy_dir,
        recursive=definition.recursive,
        preserve_structure=definition.preserve_structure,
    )
    legacy_result = legacy_pipeline.run()
    legacy_seconds = legacy_result.duration_seconds

    new_pipeline_seq = Pipeline(
        steps=definition.build_steps(),
        recursive=definition.recursive,
        preserve_structure=definition.preserve_structure,
        worker_count=1,
        log=True,
    )
    new_seq_start = perf_counter()
    new_seq_result = new_pipeline_seq.run(input_path=dataset_dir)
    new_seq_result.save(new_sequential_dir)
    new_seq_seconds = perf_counter() - new_seq_start

    new_pipeline_parallel_logged = Pipeline(
        steps=definition.build_steps(),
        recursive=definition.recursive,
        preserve_structure=definition.preserve_structure,
        worker_count=workers if workers > 0 else None,
        log=True,
    )
    new_parallel_logged_start = perf_counter()
    new_parallel_logged_result = new_pipeline_parallel_logged.run(
        input_path=dataset_dir,
    )
    new_parallel_logged_result.save(parallel_logged_dir)
    new_parallel_logged_seconds = perf_counter() - new_parallel_logged_start

    new_pipeline_parallel_quiet = Pipeline(
        steps=definition.build_steps(),
        recursive=definition.recursive,
        preserve_structure=definition.preserve_structure,
        worker_count=workers if workers > 0 else None,
        log=False,
    )
    new_parallel_quiet_start = perf_counter()
    new_parallel_quiet_result = new_pipeline_parallel_quiet.run(
        input_path=dataset_dir,
    )
    new_parallel_quiet_result.save(parallel_quiet_dir)
    new_parallel_quiet_seconds = perf_counter() - new_parallel_quiet_start
    return ComparisonResult(
        label=definition.label,
        legacy_seconds=legacy_seconds,
        new_sequential_seconds=new_seq_seconds,
        new_parallel_logged_seconds=new_parallel_logged_seconds,
        new_parallel_quiet_seconds=new_parallel_quiet_seconds,
        notes=definition.notes,
    )


def main() -> None:
    """Entry point for running the benchmark from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=5000,
        help="Number of synthetic images to generate (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed value used to generate synthetic images (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Directory used to store benchmark artefacts (default: benchmarks)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Maximum number of worker threads for the parallel run. "
            "Use 0 to auto-detect based on available CPUs (default: 0)."
        ),
    )

    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    dataset_dir = output_root / "dataset"
    sequential_root = output_root / "sequential"
    parallel_root = output_root / "parallel"

    definitions = [
        BenchmarkDefinition(
            label="Resize → Grayscale",
            steps_factory=lambda: [ResizeStep((256, 256)), GrayscaleStep()],
            notes="Resize to 256px square, convert to grayscale.",
        ),
        BenchmarkDefinition(
            label="Resize only",
            steps_factory=lambda: [ResizeStep((256, 256))],
            notes="Single-step baseline for resizing.",
        ),
        BenchmarkDefinition(
            label="Grayscale only",
            steps_factory=lambda: [GrayscaleStep()],
            notes="Single-step baseline for grayscale conversion.",
        ),
        BenchmarkDefinition(
            label="Binarize only",
            steps_factory=lambda: [BinarizeStep()],
            notes="Single-step baseline for binarization (Otsu).",
        ),
        BenchmarkDefinition(
            label="Denoise only",
            steps_factory=lambda: [DenoiseStep(mode="median", kernel_size=5)],
            notes="Single-step baseline for median denoise.",
        ),
        BenchmarkDefinition(
            label="Resize → Denoise → Rotate",
            steps_factory=lambda: [
                ResizeStep((256, 256)),
                DenoiseStep(mode="median", kernel_size=5),
                RotateStep(15.0),
            ],
            notes="Median denoise and 15° rotation.",
        ),
        BenchmarkDefinition(
            label="Resize → Grayscale → Binarize → Rotate -30°",
            steps_factory=lambda: [
                ResizeStep((192, 192)),
                GrayscaleStep(),
                BinarizeStep(),
                RotateStep(-30.0),
            ],
            notes="Composite grayscale, binarise, rotate -30°.",
        ),
    ]

    try:
        dataset_stats = _generate_benchmark_dataset(dataset_dir, args.count, args.seed)

        comparisons = [
            _run_definition(
                definition,
                dataset_dir=dataset_dir,
                sequential_root=sequential_root,
                parallel_root=parallel_root,
                workers=args.workers,
            )
            for definition in definitions
        ]

        print("Benchmark dataset summary:")
        print(f"  Images generated : {dataset_stats.generated_images}")
        print(f"  Generation time  : {dataset_stats.generation_seconds:.2f} s")
        print()
        for comparison in comparisons:
            seq_speedup = comparison.speedup_sequential()
            par_logged_speedup = comparison.speedup_parallel_logged()
            par_quiet_speedup = comparison.speedup_parallel_quiet()

            print("=" * 72)
            print(f"Step combination : {comparison.label}")
            print(f"  Notes          : {comparison.notes or '-'}")
            print(f"  Legacy (seq)   : {comparison.legacy_seconds:>8.2f} s | baseline")
            if math.isfinite(seq_speedup):
                print(
                    "  New (seq)      : "
                    f"{comparison.new_sequential_seconds:>8.2f} s | "
                    f"speed-up {seq_speedup:>5.2f}×"
                )
            else:
                print(
                    "  New (seq)      : "
                    f"{comparison.new_sequential_seconds:>8.2f} s | "
                    "speed-up   N/A"
                )
            if math.isfinite(par_logged_speedup):
                print(
                    "  New (parallel) : "
                    f"{comparison.new_parallel_logged_seconds:>8.2f} s | "
                    f"speed-up {par_logged_speedup:>5.2f}× (log on)"
                )
            else:
                print(
                    "  New (parallel) : "
                    f"{comparison.new_parallel_logged_seconds:>8.2f} s | "
                    "speed-up   N/A (log on)"
                )
            if math.isfinite(par_quiet_speedup):
                print(
                    "  New (parallel) : "
                    f"{comparison.new_parallel_quiet_seconds:>8.2f} s | "
                    f"speed-up {par_quiet_speedup:>5.2f}× (log off)"
                )
            else:
                print(
                    "  New (parallel) : "
                    f"{comparison.new_parallel_quiet_seconds:>8.2f} s | "
                    "speed-up   N/A (log off)"
                )
            print()
        print("=" * 72)
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)


if __name__ == "__main__":
    main()
