# flowimds Usage Guide

## Overview

`flowimds` lets you compose repeatable image-processing pipelines from small, reusable steps. A pipeline coordinates input discovery, step execution, and persistence of the transformed images. Every run returns a `PipelineResult` object with useful metadata:

- `processed_count`: number of images that completed successfully.
- `failed_count`: number of images that could not be processed.
- `failed_files`: ordered list of file paths that failed.
- `output_mappings`: mapping of input paths to the destination files that were written.
- `duration_seconds`: total execution time.
- `settings`: snapshot of the pipeline configuration used for the run.

## Running the Pipeline

### Processing a directory with `Pipeline.run`

```python
import flowimds as fi

pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        fi.GrayscaleStep(),
        fi.DenoiseStep(mode="median", kernel_size=5),
    ],
    input_path="/path/to/input",  # str or pathlib.Path
    output_path="/path/to/output",
    recursive=True,
    preserve_structure=True,
)

result = pipeline.run()
print(f"Processed {result.processed_count} images in {result.duration_seconds:.2f}s")
```

Use this form when you want the library to scan a directory for supported image types (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`). Set `recursive=True` to traverse subdirectories and `preserve_structure=True` to mirror the input tree under the output directory.

### Running against an explicit list of paths

```python
import flowimds as fi

paths = [
    "samples/input/receipt.png",
    "samples/input/avatar.jpg",
]

pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((256, 256)), fi.BinarizeStep(mode="otsu")],
    input_path="samples/input",
    output_path="samples/output",
)

result = pipeline.run_on_paths(paths)
for mapping in result.output_mappings:
    print(f"{mapping.input_path} -> {mapping.output_path}")
```

`run_on_paths` is helpful when you already know which files you want to process or when your inputs span multiple directories.

### Working entirely in memory with `run_on_arrays`

```python
import flowimds as fi
import numpy as np

images = [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(4)]

def brighten(image: np.ndarray) -> np.ndarray:
    # Custom inline step; any object with an apply(image) -> image method works.
    return np.clip(image + 40, 0, 255).astype(image.dtype)

pipeline = fi.Pipeline(
    steps=[fi.GrayscaleStep(), brighten],
)

transformed = pipeline.run_on_arrays(images)
print(f"Got {len(transformed)} transformed images")
```

`run_on_arrays` never touches the filesystem. It validates that every element in the input iterable is a NumPy array and returns a list of transformed arrays in the same order.

When you only call `run_on_arrays`, the filesystem paths remain optional. If you later call `run()` or `run_on_paths()` on the same pipeline instance, provide `input_path` / `output_path` values that point to existing directories so the pipeline can discover inputs and persist outputs.

### Inspecting `PipelineResult`

```python
def summarise(result: fi.PipelineResult) -> None:
    print(f"✅ processed: {result.processed_count}")
    print(f"⚠️ failed: {result.failed_count}")
    if result.failed_files:
        print("Failed files:")
        for path in result.failed_files:
            print(f"  - {path}")
    for mapping in result.output_mappings:
        print(f"Saved {mapping.input_path.name} to {mapping.output_path}")

result = pipeline.run()
summarise(result)
```

The `settings` dictionary attached to the result is useful for logging or audit trails when you need to confirm the run configuration.

## Configuring the Pipeline

Pipelines accept either `str` or `pathlib.Path` values for filesystem paths. The table below recaps the most common configuration flags.

| Setting | Type | Description |
| --- | --- | --- |
| `steps` | iterable of `PipelineStep` | Ordered sequence of transforms applied to each image. Any object exposing `apply(image)` can be used. |
| `input_path` | `str` or `Path` (optional) | Folder to scan for images when using `run`. Omit when only calling `run_on_arrays`. |
| `output_path` | `str` or `Path` (optional) | Folder where transformed files will be written. Required for `run` / `run_on_paths`; omit when only calling `run_on_arrays`. |
| `recursive` | `bool` | Enables recursive directory traversal when collecting images. |
| `preserve_structure` | `bool` | If `True`, mirrors the input directory hierarchy inside `output_path`. Otherwise every output is placed directly under `output_path`. |

Remember that the order of the `steps` list matters: each step receives the image returned by the previous one.

## Built-in Step Reference

### `ResizeStep`

- Purpose: resize every image to a fixed `(width, height)`.
- Constructor: `ResizeStep(size: tuple[int, int])`.
- Notes: validates the tuple uses positive integers; relies on OpenCV’s `cv2.resize` with bilinear interpolation by default.

### `GrayscaleStep`

- Purpose: convert colour images to single-channel grayscale.
- Constructor: `GrayscaleStep()`.
- Notes: accepts 2D or 3D arrays and preserves the input dtype; uses OpenCV colour conversion when needed.

### `BinarizeStep`

- Purpose: convert an image to black/white using either Otsu or fixed thresholding.
- Constructor: `BinarizeStep(mode="otsu", threshold=None, max_value=255)`.
- Notes: `mode="otsu"` computes the optimal threshold automatically; `mode="fixed"` requires `threshold` (0–`max_value`). Output shares the input dtype.

### `DenoiseStep`

- Purpose: reduce image noise with median or bilateral filtering.
- Constructor: `DenoiseStep(mode="median", kernel_size=3, diameter=9, sigma_color=75.0, sigma_space=75.0)`.
- Notes: `mode="median"` uses `kernel_size` (must be odd ≥ 3); `mode="bilateral"` keeps edges sharp through bilateral filtering. sigma values must be positive.

### `RotateStep`

- Purpose: rotate images counter-clockwise by an arbitrary angle.
- Constructor: `RotateStep(angle, expand=True, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101)`.
- Notes: right angles use fast `numpy.rot90`; `expand=False` keeps the original canvas size (cropping may occur). Border mode controls how uncovered pixels are filled.

### `FlipStep`

- Purpose: flip images horizontally and/or vertically.
- Constructor: `FlipStep(horizontal=False, vertical=False)`.
- Notes: at least one of `horizontal` or `vertical` must be `True`; internally maps to OpenCV’s `cv2.flip`.

## Working with Sample Data

- Run `python samples/basic_usage.py` to see the pipeline operate on bundled input files. The script prints progress and stores outputs under `samples/output`.
- Regenerate deterministic fixture images with `python scripts/generate_test_data.py`. This is useful when you change step behaviour and need refreshed test assets.

## Tips and Next Steps

- Start small: begin with a single step and add more once you have validated the intermediate outputs.
- Mix built-in and custom steps: any object with an `apply(image)` method can participate in the pipeline, making it easy to wrap bespoke OpenCV or NumPy routines.
- Watch the GitHub repository for upcoming CLI tooling that will expose common pipeline operations without writing Python code.
