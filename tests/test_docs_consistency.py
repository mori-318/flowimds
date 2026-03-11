"""Tests to keep public docs aligned with implemented APIs."""

from pathlib import Path


DOCS_EN = Path("docs/usage.md")
DOCS_JA = Path("docs/usage.ja.md")


README_EN = Path("README.md")
README_JA = Path("docs/README.ja.md")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_usage_docs_describe_current_pipeline_api() -> None:
    """Pipeline constructor and run/save arguments should match implementation."""

    en = _read(DOCS_EN)
    ja = _read(DOCS_JA)

    for content in (en, ja):
        assert "recursive: bool = False" not in content
        assert "preserve_structure: bool = False" not in content
        assert "def __init__(\n        self,\n        steps" in content
        assert (
            "run(*, input_path=None, input_paths=None, input_arrays=None, "
            "recursive=False)" in content
        )
        assert "save(output_path, preserve_structure=False)" in content


def test_usage_docs_match_current_step_modes_and_worker_note() -> None:
    """Mode examples and worker_count guidance should match current behaviour."""

    en = _read(DOCS_EN)
    ja = _read(DOCS_JA)

    for content in (en, ja):
        assert (
            'mode: Denoising method ("gaussian", "median", "bilateral")' not in content
        )
        assert (
            'mode: ノイズ除去方法（"gaussian", "median", "bilateral"）' not in content
        )
        assert 'mode: Thresholding method ("otsu", "adaptive", "fixed")' not in content
        assert 'mode: しきい値処理方法（"otsu", "adaptive", "fixed"）' not in content
        assert "worker_count=0" in content
        assert "Same as `None`" in content or "`None` と同じ" in content


def test_usage_docs_do_not_list_nonexistent_public_api_sections() -> None:
    """Docs should avoid publishing utilities and classes not exposed by flowimds."""

    en = _read(DOCS_EN)
    ja = _read(DOCS_JA)

    stale_tokens = [
        "def load_image",
        "def save_image",
        "class PipelineSettings:",
        "class PipelineError",
        "class PerformanceMetrics",
        "def merge_batch_results",
    ]

    for token in stale_tokens:
        assert token not in en
        assert token not in ja

    for content in (en, ja):
        assert "read_image" in content
        assert "write_image" in content


def test_usage_docs_input_arrays_example_uses_pipeline_step_objects() -> None:
    """input_arrays examples should not pass plain functions as steps."""

    en = _read(DOCS_EN)
    ja = _read(DOCS_JA)

    assert "steps=[fi.GrayscaleStep(), brighten]" not in en
    assert "steps=[fi.GrayscaleStep(), brighten]" not in ja
    assert "steps=[fi.GrayscaleStep(), fi.ResizeStep((128, 128))]" in en
    assert "steps=[fi.GrayscaleStep(), fi.ResizeStep((128, 128))]" in ja


def test_readme_worker_count_zero_note_matches_benchmark_behaviour() -> None:
    """README benchmark notes should match benchmark script worker semantics."""

    en = _read(README_EN)
    ja = _read(README_JA)

    assert "`0` uses the same logic as `None` (~70% of CPU cores)" not in en
    assert "`0` は `None` と同じロジック（CPUコアの約70%）" not in ja
    assert "`0` uses all logical CPU cores" in en
    assert "`0` は利用可能な論理CPUコアをすべて使用" in ja
