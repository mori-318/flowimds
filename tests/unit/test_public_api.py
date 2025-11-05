"""Tests that describe the public package API exports."""

import flowimds

from flowimds.pipeline import Pipeline, PipelineResult
from flowimds.steps import (
    BinarizeStep,
    DenoiseStep,
    FlipStep,
    GrayscaleStep,
    ResizeStep,
    RotateStep,
)


def test_public_api_exposes_pipeline_and_steps() -> None:
    """The package root should re-export key classes for convenience."""

    assert flowimds.Pipeline is Pipeline
    assert flowimds.PipelineResult is PipelineResult
    assert flowimds.ResizeStep is ResizeStep
    assert flowimds.GrayscaleStep is GrayscaleStep
    assert flowimds.RotateStep is RotateStep
    assert flowimds.FlipStep is FlipStep
    assert flowimds.BinarizeStep is BinarizeStep
    assert flowimds.DenoiseStep is DenoiseStep
