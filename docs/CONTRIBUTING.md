# Contribution Guide

Contributions welcome! We will be glad for your help. You can contribute in the following ways.

- Create an Issue – Propose a new feature or report a bug.
- Pull Request – Fix bugs, improve docs, or refactor the code.
- Publish custom pipeline steps – Build reusable steps or integrations for flowimds.
- Share – Write about flowimds in blogs, talks, or social posts.
- Make your application – Try flowimds in your own projects.

Note:
Flowimds was started by [@mori-318](https://github.com/mori-318) as a community-driven project. We appreciate every proposal, but ideas that do not align with our roadmap or design goals may be declined. Please understand that it is never personal.

Although, don't worry!
flowimds is well tested, actively polished by contributors, and relied upon in real workloads. We will keep striving to make it reliable, efficient, and fun to use.

## Installing dependencies

The `flowimds` project uses [uv](https://docs.astral.sh/uv/) as its package manager. After installing uv and Python 3.12 or newer, set up the development environment:

```bash
uv sync --all-extras --dev
```

## PRs

Please ensure your pull request passes the same checks as CI:

```bash
uv run black --check .
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

- Base your work on the latest `develop` branch (`main` is reserved for stable releases).
- Keep pull requests focused and prefer smaller changes for easier reviews.
- Link related issues using keywords such as `Closes #123`.

## Custom pipeline steps

Extensions such as additional pipeline steps or utility scripts can live outside the core package. Feel free to create third-party packages that depend on flowimds or target specific environments (for example, specialized I/O backends). If you are planning to publish an integration under the `flowimds` organization, please open an issue first so we can coordinate.

## Local development

```bash
git clone https://github.com/mori-318/flowimds.git
cd flowimds
git checkout develop
uv sync --all-extras --dev
uv run pytest
```

We recommend creating topic branches from `develop`:

```bash
git checkout -b feat/my-improvement
```

After you push your branch to your fork, open a pull request targeting `develop`.

## Questions

For general questions or help, please start a thread in [GitHub Discussions](https://github.com/mori-318/flowimds/discussions). Bug reports and feature requests should use the dedicated issue templates so we can respond efficiently.

Thank you for making flowimds better!
