"""Pytest fixtures that provide convenient access to test data paths."""

import shutil
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session")
def tests_root() -> Path:
    """Return the absolute path to the ``tests`` directory.

    Returns:
        Path: Absolute path to the ``tests`` directory.
    """

    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def data_root(tests_root: Path) -> Path:
    """Return the test data directory.

    Args:
        tests_root: Absolute path to the ``tests`` directory.

    Returns:
        Path: Absolute path to the ``tests/data`` directory.
    """

    data_dir = tests_root / "data"
    if not data_dir.exists():
        msg = "The tests/data directory does not exist."
        raise FileNotFoundError(msg)
    return data_dir


@pytest.fixture(scope="session")
def simple_input_path(data_root: Path) -> Path:
    """Return the path to the simple input fixtures.

    Args:
        data_root: Absolute path to the ``tests/data`` directory.

    Returns:
        Path: Path to ``tests/data/simple/input``.
    """

    path = data_root / "simple" / "input"
    if not path.exists():
        msg = "The tests/data/simple/input directory does not exist."
        raise FileNotFoundError(msg)
    return path


@pytest.fixture(scope="session")
def recursive_input_path(data_root: Path) -> Path:
    """Return the path to the recursive input fixtures.

    Args:
        data_root: Absolute path to the ``tests/data`` directory.

    Returns:
        Path: Path to ``tests/data/recursive/input``.
    """

    path = data_root / "recursive" / "input"
    if not path.exists():
        msg = "The tests/data/recursive/input directory does not exist."
        raise FileNotFoundError(msg)
    return path


@pytest.fixture
def simple_input_dir(tmp_path: Path, simple_input_path: Path) -> Path:
    """Copy the simple input fixtures into a temporary directory.

    Args:
        tmp_path: Temporary directory provided by pytest.
        simple_input_path: Source path to the simple fixtures.

    Returns:
        Path: Path to the copied fixtures within the temporary directory.
    """

    dest = tmp_path / "simple_input"
    shutil.copytree(simple_input_path, dest)
    return dest


@pytest.fixture
def recursive_input_dir(tmp_path: Path, recursive_input_path: Path) -> Path:
    """Copy the recursive input fixtures into a temporary directory.

    Args:
        tmp_path: Temporary directory provided by pytest.
        recursive_input_path: Source path to the recursive fixtures.

    Returns:
        Path: Path to the copied fixtures within the temporary directory.
    """

    dest = tmp_path / "recursive_input"
    shutil.copytree(recursive_input_path, dest)
    return dest


@pytest.fixture
def output_dir(tmp_path_factory: pytest.TempPathFactory) -> Generator[Path, None, None]:
    """Provide a temporary output directory for pipeline executions.

    Args:
        tmp_path_factory: Pytest factory that creates temporary directories.

    Yields:
        Path: Empty directory available for writing pipeline outputs.
    """

    path = tmp_path_factory.mktemp("flowimds_output")
    yield path
