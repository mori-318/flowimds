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
    input_path=None,
    output_path="samples/output",
)

result = pipeline.run_on_paths(paths)
for mapping in result.output_mappings:
    print(f"{mapping.input_path} -> {mapping.output_path}")
```

`run_on_paths` is helpful when you already know which files you want to process or when your inputs span multiple directories.
This method does not use the `input_path` configured on the `Pipeline` instance, so calling `run_on_paths()` when `input_path` is set will result in an error.


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

When you only use `run_on_arrays`, both `input_path` and `output_path` can be omitted.  
If you later call `run()` on the same instance, you must configure both `input_path` and `output_path` to point to valid directories. If you call `run_on_paths()`, you must configure `output_path`. If these are not set, errors will occur during input discovery or when saving outputs.

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

## Error Handling

### Understanding Failed Images

When images fail to process, they're recorded in `result.failed_files` with detailed error information:

```python
result = pipeline.run()

if result.failed_count > 0:
    print(f"⚠️  {result.failed_count} images failed:")
    for path in result.failed_files:
        print(f"  - {path}")

    # You can retry failed images with a different pipeline
    if result.failed_files:
        print("Retrying with simpler pipeline...")
        retry_pipeline = fi.Pipeline(
            steps=[fi.ResizeStep((256, 256))],  # Minimal processing
            output_path="retries",
        )
        retry_result = retry_pipeline.run_on_paths(result.failed_files)
        print(f"Recovered: {retry_result.processed_count}/{result.failed_count}")
```

**Common failure causes**:

- Unsupported image format (not in: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`)
- Corrupted image file
- Insufficient memory for large images
- Permission errors (read/write access)
- Invalid image dimensions after transformation

### Handling Validation Errors

Steps validate their parameters during initialization, providing clear error messages:

```python
# Examples of validation errors and their handling
try:
    step = fi.ResizeStep((0, 100))  # Invalid: width must be positive
except ValueError as e:
    print(f"Configuration error: {e}")
    # Fix: use positive dimensions
    step = fi.ResizeStep((100, 100))

try:
    step = fi.BinarizeStep(mode="fixed")  # Invalid: missing threshold
except ValueError as e:
    print(f"Binarize error: {e}")
    # Fix: provide threshold
    step = fi.BinarizeStep(mode="fixed", threshold=128)

try:
    step = fi.DenoiseStep(mode="median", kernel_size=2)  # Invalid: even kernel size
except ValueError as e:
    print(f"Denoise error: {e}")
    # Fix: use odd kernel size
    step = fi.DenoiseStep(mode="median", kernel_size=3)
```

**Common validation errors**:

- `ResizeStep`: dimensions must be positive integers
- `BinarizeStep`: threshold required for `mode='fixed'`, threshold must be between 0 and max_value
- `DenoiseStep`: kernel_size must be odd and ≥ 3 for median mode
- `FlipStep`: at least one of horizontal/vertical must be True
- `RotateStep`: angle can be any float, but extreme values may produce unexpected results

### Robust Pipeline Design

Build resilient pipelines that handle errors gracefully:

```python
def robust_pipeline_processing(input_path, output_path, max_retries=2):
    """Process images with error recovery and fallback strategies."""

    def create_pipeline(complexity="full"):
        if complexity == "full":
            return fi.Pipeline(
                steps=[
                    fi.ResizeStep((512, 512)),
                    fi.GrayscaleStep(),
                    fi.DenoiseStep(mode="median", kernel_size=5),
                    fi.BinarizeStep(mode="otsu"),
                ],
                output_path=output_path,
                log=True,
            )
        elif complexity == "simple":
            return fi.Pipeline(
                steps=[fi.ResizeStep((256, 256)), fi.GrayscaleStep()],
                output_path=output_path,
                log=True,
            )
        else:  # minimal
            return fi.Pipeline(
                steps=[fi.ResizeStep((128, 128))],
                output_path=output_path,
                log=True,
            )

    # Try full pipeline first
    result = create_pipeline("full").run()

    # Retry failed images with simpler pipelines
    failed_files = result.failed_files.copy()
    retry_attempts = 0

    while failed_files and retry_attempts < max_retries:
        retry_attempts += 1
        complexity = ["simple", "minimal"][retry_attempts - 1]

        print(f"Retry {retry_attempts}/{max_retries} with {complexity} pipeline...")
        retry_result = create_pipeline(complexity).run_on_paths(failed_files)

        # Update failed files list
        newly_failed = set(failed_files) - set(retry_result.output_mappings)
        failed_files = list(newly_failed)

        print(f"  Recovered: {retry_result.processed_count}, Still failing: {len(failed_files)}")

    # Final summary
    print(f"\nFinal results:")
    print(f"  Successfully processed: {result.processed_count}")
    print(f"  Recovered through retries: {sum(1 for _ in result.output_mappings) - result.processed_count}")
    print(f"  Permanently failed: {len(failed_files)}")

    if failed_files:
        print(f"  Failed files: {failed_files}")

    return result
```

### Debugging Failed Processing

When images fail to process, use these debugging strategies:

```python
def debug_failed_images(failed_files):
    """Analyze why specific images failed to process."""

    for file_path in failed_files:
        print(f"\nDebugging {file_path}:")

        # Check if file exists and is readable
        try:
            image = fi.read_image(str(file_path))
            if image is None:
                print("  - Image could not be read (possibly corrupted)")
                continue
            print(f"  - Image shape: {image.shape}")
            print(f"  - Image dtype: {image.dtype}")
        except Exception as e:
            print(f"  - Read error: {e}")
            continue

        # Try processing with individual steps
        test_steps = [
            fi.ResizeStep((256, 256)),
            fi.GrayscaleStep(),
        ]

        for i, step in enumerate(test_steps):
            try:
                processed = step.apply(image)
                print(f"  - Step {i+1} ({step.__class__.__name__}): OK")
                image = processed  # Use result for next step
            except Exception as e:
                print(f"  - Step {i+1} ({step.__class__.__name__}): FAILED - {e}")
                break
```

## Configuring the Pipeline

Pipelines accept either `str` or `pathlib.Path` values for filesystem paths. The table below recaps the most common configuration flags.

| Setting | Type | Description |
| --- | --- | --- |
| `steps` | iterable of `PipelineStep` | Ordered sequence of transforms applied to each image. Any object exposing `apply(image)` can be used. |
| `input_path` | `str` or `Path` (optional) | Folder to scan for images when using `run`. Omit when only calling `run_on_arrays`. |
| `output_path` | `str` or `Path` (optional) | Folder where transformed files will be written. Required for `run` / `run_on_paths`; omit when only calling `run_on_arrays`. |
| `recursive` | `bool` | Enables recursive directory traversal when collecting images. |
| `preserve_structure` | `bool` | If `True`, mirrors the input directory hierarchy inside `output_path`. Otherwise every output is placed directly under `output_path`. |
| `worker_count` | `int` (optional) | Maximum number of worker threads for parallel processing. `None` uses ~70% of CPU cores, `1` for sequential, `0` for all cores. |
| `log` | `bool` | Enable progress bars and informational logs during processing. |

Remember that the order of the `steps` list matters: each step receives the image returned by the previous one.

## Performance Tuning

### Parallel Processing

By default, flowimds uses approximately 70% of available CPU cores to balance performance and system responsiveness:

```python
# Explicit worker control
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        fi.GrayscaleStep(),
    ],
    input_path="input",
    output_path="output",
    worker_count=8,  # Use 8 worker threads
)
```

**Worker count guidelines**:
- `worker_count=None` (default): Auto-detect ~70% of CPU cores
- `worker_count=1`: Sequential processing (useful for debugging)
- `worker_count=0`: Uses all available CPU cores

**Performance tips**:
- For I/O-bound workloads (many small images): consider `worker_count = cpu_count * 1.5`
- For CPU-bound workloads (large images, complex transforms): use `worker_count = cpu_count * 0.7`
- Monitor memory usage with large worker counts as each worker holds images in memory

### Progress Monitoring

Enable logging to track pipeline execution and get real-time feedback:

```python
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((256, 256)),
        fi.DenoiseStep(),
    ],
    input_path="large_dataset",
    output_path="processed",
    log=True,  # Enable progress bar and logs
)

result = pipeline.run()
```

With `log=True`, you'll see:
- Progress bar (via tqdm) showing completion percentage
- Worker/core count information at startup
- Periodic progress updates for long-running operations

### Memory Considerations

- Large images consume more memory during parallel processing
- Each worker thread holds at least one image in memory
- If you encounter memory errors, reduce `worker_count` or process images in smaller batches
- Consider using `run_on_arrays` for memory-efficient in-memory processing when you don't need file persistence

```python
# Memory-efficient batch processing example
import os
from pathlib import Path

def process_in_batches(input_dir, output_dir, batch_size=100):
    """Process images in batches to control memory usage."""
    all_images = list(Path(input_dir).rglob("*.jpg"))

    for i in range(0, len(all_images), batch_size):
        batch = all_images[i:i + batch_size]
        pipeline = fi.Pipeline(
            steps=[fi.ResizeStep((512, 512))],
            output_path=output_dir,
            worker_count=4,  # Conservative for memory
        )
        result = pipeline.run_on_paths(batch)
        print(f"Batch {i//batch_size + 1}: {result.processed_count} processed")
```

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

## Troubleshooting

### Images Not Being Processed

**Problem**: `run()` returns 0 processed images

**Solutions**:

1. Verify the input directory exists and contains images
2. Check supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`
3. Enable logging to see discovery details:

```python
pipeline = fi.Pipeline(..., log=True)
result = pipeline.run()
```

4. Try `recursive=True` if images are in subdirectories

5. Check directory permissions:

```python
import os
input_dir = "/path/to/input"
print(f"Directory exists: {os.path.exists(input_dir)}")
print(f"Directory readable: {os.access(input_dir, os.R_OK)}")

# List image files in directory
from pathlib import Path
image_files = list(Path(input_dir).rglob("*"))
supported_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
images = [f for f in image_files if f.suffix.lower() in supported_extensions]
print(f"Found {len(images)} supported images")
```

### Output Files Not Created

**Problem**: Pipeline completes but no output files appear

**Solutions**:

1. Ensure `output_path` directory exists (created automatically)
2. Check write permissions for output directory
3. Inspect `result.failed_files` for specific failures
4. Verify output mappings:

```python
for mapping in result.output_mappings:
    print(f"{mapping.input_path} -> {mapping.output_path}")

# Check if output files actually exist
import os
for mapping in result.output_mappings:
    if os.path.exists(mapping.output_path):
        print(f"✓ {mapping.output_path}")
    else:
        print(f"✗ Missing: {mapping.output_path}")
```

### File Name Collisions (Flattened Output)

**Problem**: Files overwritten when `preserve_structure=False`

**Behavior**: flowimds automatically appends `_no{N}` suffixes to duplicates:

- `image.png` → `image.png`
- `image.png` (duplicate) → `image_no2.png`
- `image.png` (duplicate) → `image_no3.png`

**Example**:

```python
# Demonstrating collision handling
pipeline = fi.Pipeline(
    steps=[fi.GrayscaleStep()],
    input_path="input",  # Contains: folder1/image.png, folder2/image.png
    output_path="output",
    preserve_structure=False,  # Flatten output
    log=True,
)

result = pipeline.run()
for mapping in result.output_mappings:
    print(f"{mapping.input_path} -> {mapping.output_path}")
# Output:
# input/folder1/image.png -> output/image.png
# input/folder2/image.png -> output/image_no2.png
```

### Performance Issues

**Problem**: Processing is slower than expected

**Diagnostics**:

```python
import time
import psutil

def profile_pipeline(pipeline):
    """Profile pipeline performance and resource usage."""

    # Monitor system resources
    cpu_before = psutil.cpu_percent()
    memory_before = psutil.virtual_memory().percent

    start_time = time.time()
    result = pipeline.run()
    end_time = time.time()

    cpu_after = psutil.cpu_percent()
    memory_after = psutil.virtual_memory().percent

    print(f"Performance metrics:")
    print(f"  Duration: {end_time - start_time:.2f} seconds")
    print(f"  Images processed: {result.processed_count}")
    print(f"  Images per second: {result.processed_count / (end_time - start_time):.2f}")
    print(f"  CPU usage: {cpu_before:.1f}% -> {cpu_after:.1f}%")
    print(f"  Memory usage: {memory_before:.1f}% -> {memory_after:.1f}%")

    return result
```

**Optimization strategies**:

1. For I/O-bound workloads (many small images): increase `worker_count`
2. For CPU-bound workloads (large images, complex transforms): reduce `worker_count`
3. Monitor memory usage and reduce workers if needed
4. Consider processing in batches for very large collections
5. Use `run_on_arrays` when you don't need file persistence

### Memory Errors

**Problem**: `MemoryError` or system becomes unresponsive

**Solutions**:
```python
def memory_safe_processing(input_path, output_path):
    """Process images with memory constraints in mind."""

    # Start with conservative settings
    pipeline = fi.Pipeline(
        steps=[fi.ResizeStep((512, 512))],
        input_path=input_path,
        output_path=output_path,
        worker_count=1,  # Sequential processing
        log=True,
    )

    try:
        result = pipeline.run()
        return result
    except MemoryError:
        print("Memory error occurred, trying batch processing...")

        # Process in smaller batches
        from pathlib import Path
        all_images = list(Path(input_path).rglob("*.jpg"))
        batch_size = 50  # Adjust based on available memory

        total_processed = 0
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i + batch_size]
            batch_pipeline = fi.Pipeline(
                steps=[fi.ResizeStep((512, 512))],
                output_path=output_path,
                worker_count=1,
            )
            batch_result = batch_pipeline.run_on_paths(batch)
            total_processed += batch_result.processed_count
            print(f"Batch {i//batch_size + 1}: {batch_result.processed_count} processed")

        return fi.PipelineResult(
            processed_count=total_processed,
            failed_count=0,
            failed_files=[],
            output_mappings=[],
            duration_seconds=0,
            settings={},
        )
```

### Japanese File Names and Paths

**Note**: flowimds uses OpenCV's special handling for non-ASCII paths. Japanese characters in file names and paths are fully supported.

```python
# Japanese file names work correctly
pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((256, 256))],
    input_path="写真/入力",  # Japanese directory name
    output_path="写真/出力",  # Japanese directory name
    recursive=True,
    log=True,
)

result = pipeline.run()
print(f"Processed {result.processed_count} images with Japanese paths")
```

### Step-Specific Issues

**ResizeStep produces unexpected results**:

```python
# Check image dimensions before and after resizing
def debug_resize_step():
    test_image = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
    print(f"Original shape: {test_image.shape}")

    resize_step = fi.ResizeStep((512, 512))
    resized = resize_step.apply(test_image)
    print(f"Resized shape: {resized.shape}")

    # Note: OpenCV uses (width, height) convention
    # So (512, 512) produces 512x512 output
    ```

**BinarizeStep not working as expected**:

```python
# BinarizeStep always converts to grayscale first
def debug_binarize_step():
    # Create color test image
    color_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    print(f"Input shape: {color_image.shape}, channels: {color_image.shape[2] if len(color_image.shape) == 3 else 1}")

    binarize_step = fi.BinarizeStep(mode="otsu")
    result = binarize_step.apply(color_image)
    print(f"Output shape: {result.shape}, channels: {result.shape[2] if len(result.shape) == 3 else 1}")
    print(f"Output dtype: {result.dtype}")
    print(f"Unique values: {np.unique(result)}")
```

### Getting Help

If you encounter issues not covered here:

1. **Enable logging** to get detailed execution information
2. **Check the GitHub repository** for known issues and discussions
3. **Create a minimal reproducible example** when reporting bugs
4. **Include system information** (Python version, OS, memory)
5. **Provide sample images** that reproduce the issue (if possible)

```python
# Template for bug reports
def create_bug_report():
    """Generate information useful for bug reports."""

    import sys
    import platform
    import cv2
    import numpy as np

    print("System Information:")
    print(f"  Python: {sys.version}")
    print(f"  Platform: {platform.platform()}")
    print(f"  OpenCV: {cv2.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  flowimds: {fi.__version__ if hasattr(fi, '__version__') else 'unknown'}")

    # Memory information
    import psutil
    memory = psutil.virtual_memory()
    print(f"  Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"  Available RAM: {memory.available / (1024**3):.1f} GB")
```

## Practical Recipes

### Recipe 1: Batch Image Resizing for Web

Resize a directory of images to multiple web-friendly sizes while maintaining aspect ratio:

```python
def resize_for_web(input_dir, output_dir, sizes=[(1920, 1080), (1280, 720), (640, 360)]):
    """Resize images to multiple web-friendly formats."""

    for width, height in sizes:
        print(f"Processing {width}x{height} size...")

        pipeline = fi.Pipeline(
            steps=[fi.ResizeStep((width, height))],
            input_path=input_dir,
            output_path=f"{output_dir}/{width}x{height}",
            log=True,
            worker_count=4,
        )

        result = pipeline.run()
        print(f"  Completed: {result.processed_count} images")
        if result.failed_count > 0:
            print(f"  Failed: {result.failed_count} images")

# Usage
resize_for_web("photos/original", "photos/web")
```

### Recipe 2: Document Scanning Pipeline

Process scanned documents with deskewing, noise reduction, and binarization:

```python
def document_preprocessing(input_dir, output_dir):
    """Prepare scanned documents for OCR or archival."""

    pipeline = fi.Pipeline(
        steps=[
            fi.ResizeStep((2000, 3000)),  # Standardize size
            fi.GrayscaleStep(),           # Convert to grayscale
            fi.DenoiseStep(mode="gaussian", kernel_size=5),  # Remove noise
            fi.BinarizeStep(mode="otsu"),  # Optimal thresholding
        ],
        input_path=input_dir,
        output_path=output_dir,
        log=True,
        worker_count=2,  # Conservative for large documents
    )

    result = pipeline.run()
    print(f"Document processing complete:")
    print(f"  Successfully processed: {result.processed_count}")
    print(f"  Failed: {result.failed_count}")

    return result

# Usage
document_preprocessing("scans/input", "scans/processed")
```

### Recipe 3: Machine Learning Data Preparation

Prepare image datasets for ML training with consistent preprocessing:

```python
def prepare_ml_dataset(input_dir, output_dir, target_size=(224, 224), augment=False):
    """Prepare images for machine learning training."""

    steps = [fi.ResizeStep(target_size)]

    if augment:
        # Add data augmentation steps
        steps.extend([
            fi.RandomRotationStep(angle_range=(-15, 15)),
            fi.RandomFlipStep(horizontal=True, vertical=False),
        ])

    pipeline = fi.Pipeline(
        steps=steps,
        input_path=input_dir,
        output_path=output_dir,
        preserve_structure=True,  # Keep class folders
        log=True,
        worker_count=6,
    )

    result = pipeline.run()

    # Generate dataset statistics
    print(f"Dataset preparation complete:")
    print(f"  Total images: {result.processed_count}")
    print(f"  Processing time: {result.duration_seconds:.2f} seconds")
    print(f"  Average time per image: {result.duration_seconds/result.processed_count:.3f} seconds")

    return result

# Usage for training data
prepare_ml_dataset("dataset/raw/train", "dataset/processed/train", augment=True)
prepare_ml_dataset("dataset/raw/val", "dataset/processed/val", augment=False)
```

### Recipe 4: Thumbnail Generation with Watermarking

Create thumbnails with automatic watermarking for image galleries:

```python
class WatermarkStep:
    """Custom step to add watermark to images."""

    def __init__(self, watermark_text="© My Gallery", opacity=0.7):
        self.watermark_text = watermark_text
        self.opacity = opacity

    def apply(self, image):
        """Add semi-transparent text watermark."""
        import cv2

        # Convert to BGR for OpenCV text operations
        if len(image.shape) == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image.copy()

        # Add watermark text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        # Get text size to position it
        (text_width, text_height), baseline = cv2.getTextSize(
            self.watermark_text, font, font_scale, thickness
        )

        # Position watermark in bottom-right corner
        h, w = image_bgr.shape[:2]
        x = w - text_width - 20
        y = h - text_height - 20

        # Add text with opacity
        overlay = image_bgr.copy()
        cv2.putText(overlay, self.watermark_text, (x, y),
                   font, font_scale, (255, 255, 255), thickness)

        # Blend with original
        result = cv2.addWeighted(overlay, self.opacity, image_bgr, 1 - self.opacity, 0)

        return result

def create_thumbnails_with_watermark(input_dir, output_dir, thumbnail_size=(300, 300)):
    """Generate watermarked thumbnails for gallery display."""

    pipeline = fi.Pipeline(
        steps=[
            fi.ResizeStep(thumbnail_size),
            WatermarkStep("© My Gallery 2024"),
        ],
        input_path=input_dir,
        output_path=output_dir,
        log=True,
        worker_count=8,
    )

    result = pipeline.run()
    print(f"Thumbnail generation complete: {result.processed_count} thumbnails created")

    return result

# Usage
create_thumbnails_with_watermark("gallery/full_size", "gallery/thumbnails")
```

### Recipe 5: Medical Image Preprocessing

Standardize medical images with contrast enhancement and noise reduction:

```python
def medical_image_preprocessing(input_dir, output_dir):
    """Preprocess medical images with contrast enhancement."""

    class ContrastEnhancementStep:
        """Custom step for medical image contrast enhancement using CLAHE."""

        def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
            self.clip_limit = clip_limit
            self.tile_grid_size = tile_grid_size

        def apply(self, image):
            """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
            import cv2

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                   tileGridSize=self.tile_grid_size)
            enhanced = clahe.apply(gray)

            return enhanced

    pipeline = fi.Pipeline(
        steps=[
            fi.ResizeStep((512, 512)),  # Standardize medical image size
            ContrastEnhancementStep(clip_limit=3.0),  # Enhance contrast
            fi.DenoiseStep(mode="median", kernel_size=3),  # Remove noise
        ],
        input_path=input_dir,
        output_path=output_dir,
        log=True,
        worker_count=2,  # Conservative for medical images
    )

    result = pipeline.run()
    print(f"Medical image preprocessing complete:")
    print(f"  Processed: {result.processed_count} images")
    print(f"  Failed: {result.failed_count} images")

    return result

# Usage
medical_image_preprocessing("medical/raw", "medical/processed")
```

### Recipe 6: Real-time Processing Pipeline

Process images as they are added to a directory (useful for monitoring):

```python
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ImageProcessorHandler(FileSystemEventHandler):
    """Process new images as they appear in a directory."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.pipeline = fi.Pipeline(
            steps=[
                fi.ResizeStep((1024, 768)),
                fi.GrayscaleStep(),
                fi.DenoiseStep(mode="gaussian", kernel_size=3),
            ],
            output_path=output_dir,
            worker_count=2,
        )

    def on_created(self, event):
        """Handle new file creation."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}:
            print(f"Processing new image: {file_path.name}")

            # Wait a moment for file to be fully written
            time.sleep(1)

            try:
                result = self.pipeline.run_on_paths([file_path])
                if result.processed_count > 0:
                    print(f"  ✓ Processed successfully")
                else:
                    print(f"  ✗ Processing failed")
            except Exception as e:
                print(f"  ✗ Error: {e}")

def monitor_directory(input_dir, output_dir):
    """Monitor input directory and process new images automatically."""

    event_handler = ImageProcessorHandler(output_dir)
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=True)
    observer.start()

    print(f"Monitoring {input_dir} for new images...")
    print(f"Processed images will be saved to {output_dir}")
    print("Press Ctrl+C to stop monitoring")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nMonitoring stopped")

    observer.join()

# Usage
# monitor_directory("watch/input", "watch/output")
```

### Recipe 7: Quality Assessment Pipeline

Assess image quality and filter out low-quality images:

```python
def assess_image_quality(image):
    """Assess image quality using various metrics."""
    import cv2

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate sharpness using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Calculate brightness
    brightness = gray.mean()

    # Calculate contrast (standard deviation)
    contrast = gray.std()

    return {
        'sharpness': laplacian_var,
        'brightness': brightness,
        'contrast': contrast,
    }

class QualityFilterStep:
    """Custom step to filter images based on quality metrics."""

    def __init__(self, min_sharpness=100, min_contrast=30):
        self.min_sharpness = min_sharpness
        self.min_contrast = min_contrast
        self.quality_scores = []

    def apply(self, image):
        """Assess quality and return image if it meets criteria."""
        quality = assess_image_quality(image)
        self.quality_scores.append(quality)

        # Check quality thresholds
        if (quality['sharpness'] >= self.min_sharpness and
            quality['contrast'] >= self.min_contrast):
            return image
        else:
            # Return None to indicate the image should be filtered out
            # Note: This would need custom pipeline logic to handle
            return image

def quality_assessment_pipeline(input_dir, output_dir, good_dir, poor_dir):
    """Separate images based on quality assessment."""

    quality_step = QualityFilterStep(min_sharpness=100, min_contrast=30)

    pipeline = fi.Pipeline(
        steps=[
            fi.ResizeStep((1024, 768)),
            quality_step,
        ],
        input_path=input_dir,
        output_path=output_dir,
        log=True,
        worker_count=4,
    )

    result = pipeline.run()

    # Analyze quality scores
    if quality_step.quality_scores:
        avg_sharpness = sum(q['sharpness'] for q in quality_step.quality_scores) / len(quality_step.quality_scores)
        avg_contrast = sum(q['contrast'] for q in quality_step.quality_scores) / len(quality_step.quality_scores)

        print(f"Quality assessment complete:")
        print(f"  Images processed: {result.processed_count}")
        print(f"  Average sharpness: {avg_sharpness:.2f}")
        print(f"  Average contrast: {avg_contrast:.2f}")

    return result, quality_step.quality_scores

# Usage
result, scores = quality_assessment_pipeline("photos/all", "photos/processed", "photos/good", "photos/poor")
```

## Creating Custom Steps

### Understanding the PipelineStep Protocol

All pipeline steps in flowimds follow the `PipelineStep` protocol, which requires a single method:

```python
from flowimds.steps.base import PipelineStep
import numpy as np

class MyCustomStep:
    """Custom step implementing the PipelineStep protocol."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Transform the provided image and return the result.

        Args:
            image: Input image as numpy array (2D or 3D)

        Returns:
            Transformed image as numpy array with same or different dimensions
        """
        # Your custom processing logic here
        return processed_image
```

### Basic Custom Step Examples

#### Example 1: Simple Brightness Adjustment

```python
class BrightnessStep:
    """Adjust image brightness by adding a constant value."""

    def __init__(self, brightness_factor: float = 1.0):
        """Initialize brightness adjustment.

        Args:
            brightness_factor: Multiplier for pixel values (1.0 = no change)
        """
        self.brightness_factor = brightness_factor

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply brightness adjustment."""
        # Convert to float to avoid overflow
        adjusted = image.astype(np.float32) * self.brightness_factor

        # Clip values to valid range and convert back
        adjusted = np.clip(adjusted, 0, 255)
        return adjusted.astype(image.dtype)

# Usage
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        BrightnessStep(brightness_factor=1.2),  # Increase brightness by 20%
        fi.GrayscaleStep(),
    ],
    input_path="input",
    output_path="output",
)
```

#### Example 2: Gaussian Blur with Custom Parameters

```python
import cv2

class CustomBlurStep:
    """Apply Gaussian blur with configurable parameters."""

    def __init__(self, kernel_size: int = 5, sigma_x: float = 1.0):
        """Initialize blur step.

        Args:
            kernel_size: Size of the Gaussian kernel (must be odd)
            sigma_x: Standard deviation in X direction
        """
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), self.sigma_x)

# Usage
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((1024, 768)),
        CustomBlurStep(kernel_size=7, sigma_x=2.0),
        fi.BinarizeStep(mode="otsu"),
    ],
)
```

### Advanced Custom Step Examples

#### Example 3: Edge Detection with Multiple Algorithms

```python
class EdgeDetectionStep:
    """Edge detection with algorithm selection."""

    def __init__(self, method: str = "canny", **kwargs):
        """Initialize edge detection.

        Args:
            method: Edge detection method ('canny', 'sobel', 'laplacian')
            **kwargs: Method-specific parameters
        """
        self.method = method.lower()
        self.kwargs = kwargs

        if self.method not in {"canny", "sobel", "laplacian"}:
            raise ValueError("method must be 'canny', 'sobel', or 'laplacian'")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply edge detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.method == "canny":
            low_threshold = self.kwargs.get("low_threshold", 50)
            high_threshold = self.kwargs.get("high_threshold", 150)
            return cv2.Canny(gray, low_threshold, high_threshold)

        elif self.method == "sobel":
            ksize = self.kwargs.get("ksize", 3)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)

        elif self.method == "laplacian":
            ksize = self.kwargs.get("ksize", 3)
            return cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize).astype(np.uint8)

# Usage
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        fi.GrayscaleStep(),
        EdgeDetectionStep(method="canny", low_threshold=100, high_threshold=200),
    ],
)
```

#### Example 4: Histogram Equalization

```python
class HistogramEqualizationStep:
    """Apply histogram equalization with optional CLAHE."""

    def __init__(self, use_clahe: bool = False, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        """Initialize histogram equalization.

        Args:
            use_clahe: Use CLAHE instead of standard histogram equalization
            clip_limit: Contrast limiting threshold for CLAHE
            tile_grid_size: Size of the grid for tile-based histogram equalization
        """
        self.use_clahe = use_clahe
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            return clahe.apply(gray)
        else:
            return cv2.equalizeHist(gray)

# Usage
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        HistogramEqualizationStep(use_clahe=True, clip_limit=3.0),
    ],
)
```

### Custom Step with State

#### Example 5: Adaptive Threshold Based on Image Statistics

```python
class AdaptiveThresholdStep:
    """Adaptive thresholding based on image statistics."""

    def __init__(self, target_mean: float = 128.0, max_iterations: int = 10):
        """Initialize adaptive threshold.

        Args:
            target_mean: Target mean pixel value after thresholding
            max_iterations: Maximum iterations to find optimal threshold
        """
        self.target_mean = target_mean
        self.max_iterations = max_iterations
        self.final_threshold = None  # Store the threshold used

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Binary search for optimal threshold
        low, high = 0, 255
        best_threshold = 127

        for _ in range(self.max_iterations):
            threshold = (low + high) // 2
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            current_mean = binary.mean()

            if abs(current_mean - self.target_mean) < 1.0:
                best_threshold = threshold
                break
            elif current_mean < self.target_mean:
                high = threshold
            else:
                low = threshold
            best_threshold = threshold

        self.final_threshold = best_threshold
        _, result = cv2.threshold(gray, best_threshold, 255, cv2.THRESH_BINARY)
        return result

# Usage with threshold inspection
threshold_step = AdaptiveThresholdStep(target_mean=100.0)
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        fi.GrayscaleStep(),
        threshold_step,
    ],
)

result = pipeline.run()
print(f"Final threshold used: {threshold_step.final_threshold}")
```

### Custom Step Best Practices

#### 1. Input Validation

```python
class RobustCustomStep:
    """Custom step with comprehensive input validation."""

    def __init__(self, param1: float, param2: int):
        """Initialize with parameter validation."""
        if not isinstance(param1, (int, float)):
            raise TypeError("param1 must be a number")
        if not 0 <= param1 <= 1.0:
            raise ValueError("param1 must be between 0 and 1")
        if not isinstance(param2, int) or param2 <= 0:
            raise ValueError("param2 must be a positive integer")

        self.param1 = float(param1)
        self.param2 = param2

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply with image validation."""
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy array")
        if image.size == 0:
            raise ValueError("Input array cannot be empty")
        if len(image.shape) not in {2, 3}:
            raise ValueError("Input must be 2D or 3D array")

        # Your processing logic here
        return processed_image
```

#### 2. Memory Efficiency

```python
class MemoryEfficientStep:
    """Custom step that processes images in chunks for memory efficiency."""

    def __init__(self, chunk_size: int = 1024):
        """Initialize with chunk size for processing."""
        self.chunk_size = chunk_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Process image in chunks to save memory."""
        height, width = image.shape[:2]
        result = np.zeros_like(image)

        # Process in horizontal chunks
        for y in range(0, height, self.chunk_size):
            y_end = min(y + self.chunk_size, height)
            chunk = image[y:y_end]

            # Process chunk
            processed_chunk = self._process_chunk(chunk)
            result[y:y_end] = processed_chunk

        return result

    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process a single chunk of the image."""
        # Your chunk-specific processing logic
        return chunk
```

#### 3. Error Handling and Logging

```python
import logging

class LoggingStep:
    """Custom step with built-in logging and error handling."""

    def __init__(self, operation_name: str = "custom_operation"):
        """Initialize with logging configuration."""
        self.operation_name = operation_name
        self.logger = logging.getLogger(f"flowimds.{operation_name}")

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply with comprehensive error handling."""
        try:
            self.logger.debug(f"Processing image of shape {image.shape}")

            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")

            # Process image
            result = self._process_image(image)

            self.logger.debug(f"Successfully processed image, output shape {result.shape}")
            return result

        except Exception as e:
            self.logger.error(f"Error in {self.operation_name}: {e}")
            # Re-raise with more context
            raise RuntimeError(f"Failed to process image in {self.operation_name}: {e}") from e

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """Actual processing logic."""
        # Your processing code here
        return image
```

### Testing Custom Steps

```python
import unittest
import numpy as np

class TestCustomStep(unittest.TestCase):
    """Test suite for custom steps."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    def test_brightness_step(self):
        """Test BrightnessStep functionality."""
        step = BrightnessStep(brightness_factor=1.5)

        # Test with color image
        result = step.apply(self.test_image)
        self.assertEqual(result.shape, self.test_image.shape)
        self.assertTrue(np.all(result >= self.test_image))  # Should be brighter

        # Test with grayscale image
        result_gray = step.apply(self.test_gray)
        self.assertEqual(result_gray.shape, self.test_gray.shape)

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with self.assertRaises(ValueError):
            BrightnessStep(brightness_factor=-1.0)

    def test_edge_detection_methods(self):
        """Test different edge detection methods."""
        for method in ["canny", "sobel", "laplacian"]:
            step = EdgeDetectionStep(method=method)
            result = step.apply(self.test_image)
            self.assertEqual(len(result.shape), 2)  # Should be grayscale
            self.assertTrue(np.all(result >= 0))  # Should be non-negative

# Run tests
if __name__ == "__main__":
    unittest.main()
```

### Integrating Custom Steps

```python
# Create a pipeline with mixed built-in and custom steps
custom_pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((512, 512)),
        BrightnessStep(brightness_factor=1.2),
        CustomBlurStep(kernel_size=5, sigma_x=1.0),
        EdgeDetectionStep(method="canny", low_threshold=50, high_threshold=150),
        HistogramEqualizationStep(use_clahe=True),
    ],
    input_path="input",
    output_path="output",
    log=True,
    worker_count=4,
)

# Run the pipeline
result = custom_pipeline.run()
print(f"Processed {result.processed_count} images with custom pipeline")
```

## API Reference

### Core Classes

#### Pipeline

The main class for creating and executing image processing pipelines.

```python
class Pipeline:
    """Image processing pipeline with parallel execution support."""

    def __init__(
        self,
        steps: List[PipelineStep],
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        worker_count: int = 4,
        preserve_structure: bool = True,
        log: bool = False,
    ):
        """Initialize a new pipeline.

        Args:
            steps: List of processing steps implementing PipelineStep protocol
            input_path: Directory containing input images (optional)
            output_path: Directory for output images (optional)
            worker_count: Number of parallel workers (default: 4)
            preserve_structure: Whether to preserve directory structure (default: True)
            log: Enable detailed logging (default: False)
        """
```

**Methods:**

- `run() -> PipelineResult`: Execute pipeline on directory
- `run_on_paths(paths: List[Path]) -> PipelineResult`: Execute on specific file paths
- `run_on_arrays(images: List[np.ndarray]) -> PipelineResult`: Execute on numpy arrays

**Example:**

```python
pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((512, 512)), fi.GrayscaleStep()],
    input_path="input",
    output_path="output",
    worker_count=8,
    log=True,
)

# Run on directory
result = pipeline.run()

# Run on specific files
from pathlib import Path
specific_files = [Path("img1.jpg"), Path("img2.jpg")]
result = pipeline.run_on_paths(specific_files)

# Run on numpy arrays
import numpy as np
arrays = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)]
result = pipeline.run_on_arrays(arrays)
```

#### PipelineResult

Contains results and metadata from pipeline execution.

```python
@dataclass
class PipelineResult:
    """Results from pipeline execution."""

    processed_count: int
    failed_count: int
    failed_files: List[str]
    output_mappings: List[FileMapping]
    duration_seconds: float
    settings: Dict[str, Any]

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.processed_count + self.failed_count == 0:
            return 0.0
        return (self.processed_count / (self.processed_count + self.failed_count)) * 100
```

**FileMapping:**
```python
@dataclass
class FileMapping:
    """Mapping between input and output files."""

    input_path: str
    output_path: str
    success: bool
    error_message: Optional[str] = None
```

### Built-in Steps

#### ResizeStep

Resize images to specified dimensions.

```python
class ResizeStep:
    """Resize images to specified dimensions."""

    def __init__(self, size: Tuple[int, int]):
        """Initialize resize step.

        Args:
            size: Target size as (width, height) tuple
        """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
```

**Example:**
```python
# Resize to square
step = fi.ResizeStep((512, 512))

# Resize to rectangular
step = fi.ResizeStep((1024, 768))

# Use in pipeline
pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((800, 600))],
    input_path="input",
    output_path="output",
)
```

#### GrayscaleStep

Convert images to grayscale.

```python
class GrayscaleStep:
    """Convert color images to grayscale."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
```

**Example:**
```python
step = fi.GrayscaleStep()
result = step.apply(color_image)  # Returns 2D array
```

#### DenoiseStep

Apply noise reduction to images.

```python
class DenoiseStep:
    """Apply noise reduction to images."""

    def __init__(self, mode: str = "gaussian", kernel_size: int = 5):
        """Initialize denoise step.

        Args:
            mode: Denoising method ("gaussian", "median", "bilateral")
            kernel_size: Size of the denoising kernel
        """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to image."""
```

**Example:**
```python
# Gaussian blur
gaussian_step = fi.DenoiseStep(mode="gaussian", kernel_size=5)

# Median filter
median_step = fi.DenoiseStep(mode="median", kernel_size=3)

# Bilateral filter (preserves edges)
bilateral_step = fi.DenoiseStep(mode="bilateral")
```

#### BinarizeStep

Convert images to binary (black and white).

```python
class BinarizeStep:
    """Convert images to binary using thresholding."""

    def __init__(self, mode: str = "otsu", threshold: Optional[int] = None):
        """Initialize binarize step.

        Args:
            mode: Thresholding method ("otsu", "adaptive", "fixed")
            threshold: Fixed threshold value (required for "fixed" mode)
        """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary."""
```

**Example:**
```python
# Otsu's automatic thresholding
otsu_step = fi.BinarizeStep(mode="otsu")

# Fixed threshold
fixed_step = fi.BinarizeStep(mode="fixed", threshold=127)

# Adaptive thresholding
adaptive_step = fi.BinarizeStep(mode="adaptive")
```

#### FlipStep

Flip images horizontally and/or vertically.

```python
class FlipStep:
    """Flip images horizontally and/or vertically."""

    def __init__(self, horizontal: bool = False, vertical: bool = False):
        """Initialize flip step.

        Args:
            horizontal: Flip horizontally (default: False)
            vertical: Flip vertically (default: False)
        """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Flip image according to specified axes."""
```

**Example:**
```python
# Horizontal flip
h_flip = fi.FlipStep(horizontal=True)

# Vertical flip
v_flip = fi.FlipStep(vertical=True)

# Both horizontal and vertical
both_flip = fi.FlipStep(horizontal=True, vertical=True)
```

### Utility Functions

#### Image Loading and Saving

```python
def load_image(path: str) -> np.ndarray:
    """Load image from file path.

    Args:
        path: Path to image file

    Returns:
        Image as numpy array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid image
    """

def save_image(image: np.ndarray, path: str) -> None:
    """Save image to file path.

    Args:
        image: Image as numpy array
        path: Output file path

    Raises:
        ValueError: If image format is not supported
    """
```

#### Image Validation

```python
def validate_image(image: np.ndarray) -> bool:
    """Validate if array is a valid image.

    Args:
        image: Array to validate

    Returns:
        True if valid image, False otherwise
    """

def get_image_info(image: np.ndarray) -> Dict[str, Any]:
    """Get information about image array.

    Args:
        image: Image array

    Returns:
        Dictionary with image metadata
    """
```

### Configuration

#### Pipeline Settings

```python
class PipelineSettings:
    """Configuration settings for pipeline execution."""

    def __init__(
        self,
        max_workers: int = 4,
        chunk_size: int = 100,
        timeout_seconds: int = 300,
        retry_attempts: int = 3,
        memory_limit_mb: int = 1024,
    ):
        """Initialize pipeline settings.

        Args:
            max_workers: Maximum number of worker processes
            chunk_size: Number of images per processing chunk
            timeout_seconds: Timeout for individual image processing
            retry_attempts: Number of retry attempts for failed images
            memory_limit_mb: Memory limit per worker in MB
        """
```

#### Logging Configuration

```python
def configure_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    file_path: Optional[str] = None,
) -> None:
    """Configure logging for pipeline operations.

    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        format_string: Custom log format string
        file_path: Log file path (optional, defaults to stdout)
    """
```

### Error Types

#### Pipeline Exceptions

```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class StepExecutionError(PipelineError):
    """Error during step execution."""

    def __init__(self, step_name: str, message: str):
        self.step_name = step_name
        super().__init__(f"Error in step '{step_name}': {message}")

class ImageLoadError(PipelineError):
    """Error loading image file."""
    pass

class ImageSaveError(PipelineError):
    """Error saving image file."""
    pass

class ValidationError(PipelineError):
    """Error during input validation."""
    pass
```

### Performance Monitoring

#### Performance Metrics

```python
@dataclass
class PerformanceMetrics:
    """Performance metrics for pipeline execution."""

    total_images: int
    processed_images: int
    failed_images: int
    total_time: float
    average_time_per_image: float
    memory_usage_mb: float
    cpu_usage_percent: float

    @classmethod
    def from_pipeline_result(cls, result: PipelineResult) -> "PerformanceMetrics":
        """Create metrics from pipeline result."""
        return cls(
            total_images=result.processed_count + result.failed_count,
            processed_images=result.processed_count,
            failed_images=result.failed_count,
            total_time=result.duration_seconds,
            average_time_per_image=result.duration_seconds / max(1, result.processed_count),
            memory_usage_mb=0.0,  # Would be populated during execution
            cpu_usage_percent=0.0,  # Would be populated during execution
        )
```

#### Memory Profiling

```python
def profile_memory_usage(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function.

    Args:
        func: Function to profile

    Returns:
        Wrapped function that reports memory usage
    """

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.

    Returns:
        Dictionary with memory usage information
    """
```

### Batch Processing

#### Batch Operations

```python
def process_in_batches(
    pipeline: Pipeline,
    input_paths: List[str],
    batch_size: int = 100,
) -> List[PipelineResult]:
    """Process images in batches to manage memory usage.

    Args:
        pipeline: Pipeline to execute
        input_paths: List of input file paths
        batch_size: Number of images per batch

    Returns:
        List of results for each batch
    """

def merge_batch_results(results: List[PipelineResult]) -> PipelineResult:
    """Merge multiple batch results into single result.

    Args:
        results: List of batch results

    Returns:
        Combined pipeline result
    """
```

### Integration Examples

#### With NumPy

```python
import numpy as np
import flowimds as fi

# Create synthetic images
images = [
    np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    for _ in range(100)
]

# Process with pipeline
pipeline = fi.Pipeline(steps=[
    fi.ResizeStep((256, 256)),
    fi.GrayscaleStep(),
])

result = pipeline.run_on_arrays(images)
print(f"Processed {result.processed_count} synthetic images")
```

#### With OpenCV

```python
import cv2
import flowimds as fi

# Custom step using OpenCV
class OpenCVCustomStep:
    def apply(self, image):
        # Use OpenCV functions directly
        return cv2.medianBlur(image, 5)

# Integrate with flowimds
pipeline = fi.Pipeline(steps=[
    fi.ResizeStep((512, 512)),
    OpenCVCustomStep(),
    fi.BinarizeStep(mode="otsu"),
])
```

#### With Pillow

```python
from PIL import Image
import flowimds as fi
import numpy as np

class PillowStep:
    def apply(self, image):
        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Apply PIL operations
        pil_image = pil_image.convert('L')  # Grayscale

        # Convert back to numpy
        return np.array(pil_image)

pipeline = fi.Pipeline(steps=[
    PillowStep(),
    fi.ResizeStep((256, 256)),
])
```
