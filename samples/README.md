# Samples

The `samples` directory contains small, self-contained scripts that showcase
how to work with the `flowimds` pipeline. They generate synthetic image data on
the fly, so you can try the library without preparing any assets in advance.

## Basic pipeline example

- Script: `samples/basic_usage.py`
- Creates a temporary input directory with three coloured squares.
- Runs the pipeline with resize and grayscale steps.
- Demonstrates both `Pipeline.run()` for directory processing and
  `Pipeline.run()` with numpy arrays for in-memory transformations.

### Running the example

```bash
uv run python samples/basic_usage.py
```

The script prints the paths of the generated images. Processed outputs are
written to `samples/output` and can be inspected with any image viewer.
