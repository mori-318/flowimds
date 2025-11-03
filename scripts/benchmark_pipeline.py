"""Benchmark tools for measuring pipeline throughput."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from flowimds.pipeline import Pipeline
from flowimds.steps import ResizeStep
from flowimds.utils.image_io import write_image


@dataclass
class BenchmarkResult:
    """Container describing the outcome of a benchmark run."""

    generated_images: int
    generation_seconds: float
    processing_seconds: float


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


def _generate_benchmark_dataset(directory: Path, count: int, seed: int) -> float:
    """Populate ``directory`` with ``count`` synthetic images."""

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = _rng(seed)

    start = perf_counter()
    for index in range(count):
        shape = _random_shape(rng)
        image = _random_image(shape, rng)
        name = f"image_{index:04d}.png"
        write_image(str(directory / name), image)
    return perf_counter() - start


def _build_pipeline(input_dir: Path, output_dir: Path) -> Pipeline:
    """Return a simple pipeline used for benchmarking."""

    return Pipeline(
        steps=[ResizeStep((128, 128))],
        input_path=input_dir,
        output_path=output_dir,
        recursive=False,
        preserve_structure=False,
    )


def run_benchmark(
    input_dir: Path,
    output_dir: Path,
    *,
    count: int,
    dataset_seed: int,
) -> BenchmarkResult:
    """Generate a dataset and measure pipeline processing time."""

    generation_seconds = _generate_benchmark_dataset(input_dir, count, dataset_seed)

    pipeline = _build_pipeline(input_dir, output_dir)

    start = perf_counter()
    result = pipeline.run()
    processing_seconds = perf_counter() - start

    assert result.processed_count == count

    return BenchmarkResult(
        generated_images=count,
        generation_seconds=generation_seconds,
        processing_seconds=processing_seconds,
    )


def main() -> None:
    """Entry point for running the benchmark from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Number of synthetic images to generate (default: 1000)",
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
        help="Directory used to store benchmark datasets (default: benchmarks)",
    )

    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    input_dir = output_root / "input"
    output_dir = output_root / "output"

    try:
        result = run_benchmark(
            input_dir,
            output_dir,
            count=args.count,
            dataset_seed=args.seed,
        )

        print("Benchmark completed:")
        print(f"  Generated images : {result.generated_images}")
        print(f"  Generation time  : {result.generation_seconds:.2f} s")
        print(f"  Processing time  : {result.processing_seconds:.2f} s")
    finally:
        if output_root.exists():
            shutil.rmtree(output_root)


if __name__ == "__main__":
    main()
