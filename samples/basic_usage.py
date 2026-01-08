"""Minimal end-to-end example for the flowimds pipeline."""

from pathlib import Path

import numpy as np

import flowimds as fi


def _ensure_directories(base: Path) -> tuple[Path, Path]:
    """Return input/output directories for the sample data."""

    input_dir = base / "input"
    output_dir = base / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def _create_sample_images(input_dir: Path) -> list[Path]:
    """Generate a few colored squares for demonstration."""

    colors = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    )
    paths: list[Path] = []
    for index, color in enumerate(colors):
        image = np.full((48, 48, 3), color, dtype=np.uint8)
        path = input_dir / f"sample_{index}.png"
        fi.write_image(str(path), image)
        paths.append(path)
    return paths


def main() -> None:
    """Run the basic pipeline example."""

    base_dir = Path(__file__).resolve().parent
    input_dir, output_dir = _ensure_directories(base_dir)
    input_paths = _create_sample_images(input_dir)

    pipeline = fi.Pipeline(
        steps=[fi.ResizeStep((32, 32)), fi.GrayscaleStep()],
    )

    result = pipeline.run(input_path=input_dir, recursive=False)
    print("Directory run produced the following outputs:")
    result.save(output_dir, preserve_structure=False)
    for mapping in result.output_mappings:
        print(f" - {mapping.output_path}")

    images = [fi.read_image(str(path)) for path in input_paths]
    arrays_result = pipeline.run(input_arrays=images)
    print("In-memory run produced the following outputs:")
    arrays_result.save(output_dir, preserve_structure=False)
    for mapping in arrays_result.output_mappings:
        print(f" - {mapping.output_path}")


if __name__ == "__main__":
    main()
