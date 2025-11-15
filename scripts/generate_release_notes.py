"""Script to extract release notes from ``CHANGELOG.md``."""

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract release notes for a specific version from CHANGELOG.md.",
    )
    parser.add_argument(
        "--changelog",
        type=Path,
        default=Path("CHANGELOG.md"),
        help="Path to the CHANGELOG file to parse.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Git tag (e.g., v0.2.1). The leading 'v' is removed automatically.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination file path for the extracted release notes. "
            "Defaults to stdout."
        ),
    )
    return parser.parse_args()


def normalize_version(tag: str) -> str:
    """Normalize a Git tag string to a semantic version.

    Args:
        tag (str): Git tag (for example, ``v0.2.1``).

    Returns:
        str: Version string without the leading prefix.
    """
    return tag.lstrip("vV")


def extract_section(changelog_text: str, version: str) -> str:
    """Extract the section for a specific version from the changelog.

    Args:
        changelog_text (str): Entire changelog text.
        version (str): Target version string.

    Returns:
        str: Markdown section corresponding to the target version.

    Raises:
        ValueError: Raised when no section matches the target version.
    """
    escaped_version = re.escape(version)
    heading_pattern = re.compile(rf"^## \[{escaped_version}\].*$", re.MULTILINE)
    heading_match = heading_pattern.search(changelog_text)
    if not heading_match:
        raise ValueError(f"Section for version {version} was not found.")

    start_index = heading_match.start()
    remainder = changelog_text[heading_match.end() :]
    next_heading_match = re.search(r"^## \[", remainder, re.MULTILINE)
    end_index = (
        heading_match.end() + next_heading_match.start()
        if next_heading_match
        else len(changelog_text)
    )
    return changelog_text[start_index:end_index].strip()


def output_section(section: str, output_path: Path | None) -> None:
    """Output the extracted section.

    Args:
        section (str): Extracted Markdown text.
        output_path (Path | None): Destination file path or ``None`` for stdout.
    """
    if output_path:
        output_path.write_text(section + "\n", encoding="utf-8")
    else:
        print(section)


def main() -> None:
    """Entrypoint for the CLI script."""
    args = parse_args()
    version = normalize_version(args.tag)
    changelog_text = args.changelog.read_text(encoding="utf-8")
    section = extract_section(changelog_text, version)
    output_section(section, args.output)


if __name__ == "__main__":
    main()
