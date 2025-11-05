# flowimds

[English README](../README.md)

`flowimds` は、画像ディレクトリを一括処理するためのオープンソースPythonライブラリです。リサイズ、グレースケール化、二値化、ノイズ除去、回転、反転などのステップを組み合わせてパイプラインを定義し、フォルダ単位・ファイルリスト指定・NumPy配列など多様な入力に対して実行できます。

## 目次

1. [特徴](#特徴)
2. [インストール](#インストール)
3. [クイックスタート](#クイックスタート)
4. [利用ガイド](#利用ガイド)
5. [サポート](#サポート)
6. [コントリビューション](#コントリビューション)
7. [開発環境の整え方](#開発環境の整え方)
8. [ライセンス](#ライセンス)
9. [プロジェクト状況](#プロジェクト状況)
10. [謝辞](#謝辞)

## 特徴

- ディレクトリ全体のバッチ処理と再帰走査に対応。
- 入力フォルダ構成を出力側で再現するオプションを提供。
- リサイズ・グレースケール・回転・反転・二値化・ノイズ除去など豊富な標準ステップ。
- ディレクトリ走査、明示的なファイルリスト、NumPy配列のいずれでも実行可能。
- テスト用の画像データを再生成できる決定的なスクリプトを同梱。

## インストール

### 必要要件

- Python 3.12 以上
- 依存管理に `uv` もしくは `pip`

### コマンド

```bash
pip install flowimds
```

もしくは

```bash
uv add flowimds
```

### ソースコードから利用

```bash
git clone https://github.com/mori-318/flowimds.git
cd flowimds
uv sync
```

## クイックスタート

```python
import flowimds as fi

pipeline = fi.Pipeline(
    steps=[fi.ResizeStep((128, 128)), fi.GrayscaleStep()],
    input_path="samples/input",
    output_path="samples/output",
    recursive=True,
    preserve_structure=True,
)

result = pipeline.run()
print(f"Processed {result.processed_count} images")
```

## 利用ガイド

- 標準ステップや独自ステップを組み合わせて、用途に合わせたパイプラインを構築できます。
- ディレクトリ走査、ファイルリスト指定、純粋な NumPy 配列処理など、手元のデータに合わせて実行方法を選べます。
- 実行結果の `PipelineResult` から処理件数、失敗ファイル、出力パスをすばやく確認できます。

さらに詳しい説明やコード例は [usage.ja.md](usage.ja.md) を参照してください。

## サポート

問題報告や質問は GitHub の Issue Tracker で受け付けています。

## コントリビューション

安定性を維持するため、GitFlow ベースの運用を行っています。

- **main**: 常にリリース可能な安定版 (`vX.Y.Z` 形式でタグ付け)。
- **develop**: 次期リリース候補の統合ブランチ。
- **feature/**・**release/**・**hotfix/**: 機能追加や修正用の短期ブランチ。

Pull Request を送る前に次の手順を実施してください。

1. `develop` からトピックブランチを作成する。
2. Lint とテストがすべて成功することを確認する（[開発環境の整え方](#開発環境の整え方)を参照）。
3. コミットメッセージは [Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に従う。

## 開発環境の整え方

```bash
# 依存関係の同期
uv sync --all-extras --dev

# Lint / Format
uv run black --check .
uv run ruff check .
uv run ruff format --check .

# テスト実行
uv run pytest

# テスト用データの再生成
uv run python scripts/generate_test_data.py
```

## ライセンス

[MIT License](../LICENSE) に基づいて公開しています。

## プロジェクト状況

初の安定版リリースに向けて開発が進行中です。タグ付きリリースにご期待ください。

## 謝辞

- [NumPy](https://numpy.org/) による配列処理基盤
- [OpenCV](https://opencv.org/) による画像入出力
- [uv](https://github.com/astral-sh/uv) と [Ruff](https://docs.astral.sh/ruff/) による開発ワークフロー支援
