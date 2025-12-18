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
    """Generate a few coloured squares for demonstration."""

    colours = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    )
    paths: list[Path] = []
    for index, colour in enumerate(colours):
        image = np.full((48, 48, 3), colour, dtype=np.uint8)
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
        input_path=input_dir,
        output_path=output_dir,
        recursive=False,
        preserve_structure=False,
    )

    result = pipeline.run()
    print("Directory run produced the following outputs:")
    for mapping in result.output_mappings:
        print(f" - {mapping.output_path}")

    images = [fi.read_image(str(path)) for path in input_paths]
    transformed = pipeline.run_on_arrays(images)
    print("In-memory run produced the following outputs:")
    for index, array in enumerate(transformed):
        destination = output_dir / f"in_memory_{index}.png"
        fi.write_image(str(destination), array)
        print(f" - {destination}")


if __name__ == "__main__":
    main()
