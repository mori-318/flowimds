# flowimds

[English README](../README.md)

`flowimds` は、画像ディレクトリを一括処理するためのオープンソースPythonライブラリです。リサイズ、グレースケール化、二値化、ノイズ除去、回転、反転などのステップを組み合わせてパイプラインを定義し、フォルダ単位・ファイルリスト指定・NumPy配列など多様な入力に対して実行できます。

## 目次

1. [特徴](#特徴)
2. [インストール](#インストール)
3. [クイックスタート](#クイックスタート)
4. [利用ガイド](#利用ガイド)
5. [CLIについて](#cliについて)
6. [ロードマップ](#ロードマップ)
7. [サポート](#サポート)
8. [コントリビューション](#コントリビューション)
9. [開発環境の整え方](#開発環境の整え方)
10. [ライセンス](#ライセンス)
11. [プロジェクト状況](#プロジェクト状況)
12. [謝辞](#謝辞)

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
uv sync
```

もしくは

```bash
pip install flowimds
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

- **明示的なパス指定**: `pipeline.run_on_paths([...])` を使うと、指定したファイルだけを処理して出力に保存できます。
- **インメモリ処理**: `pipeline.run_on_arrays([...])` で NumPy 配列を直接処理できます。
- **サンプル**: `samples/README.md` に入力データ生成および結果確認の例があります。

## CLIについて

将来的に CLI (`flowimds process ...`) を提供予定です。進捗は[ロードマップ](#ロードマップ)を参照してください。

## ロードマップ

主要なマイルストーンと今後の予定は [`docs/plan.md`](plan.md) にまとめています。

- v1.0: パイプライン実装と結果レポート機能
- CLI ツールの提供
- AI 推論ステップの追加検討

## サポート

問題報告や質問はリポジトリ公開後に Issue Tracker で受け付けます。それまでの間はディスカッション等で問い合わせてください。

## コントリビューション

GitFlow に基づく運用を推奨しています。

- **main**: 常にリリース可能な安定版。`vX.Y.Z` 形式でタグ付けします。
- **develop**: 次期リリース候補の統合ブランチ。
- **feature/**, **release/**, **hotfix/** ブランチで個別対応。

Pull Request を送る前に以下を実施してください。

1. `develop` から作業ブランチを切る。
2. Lint・テストが全て成功することを確認（[開発環境の整え方](#開発環境の整え方)参照）。
3. コミットメッセージは [Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に従って記述。

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

初の安定版リリースに向けて開発中です。タグ付きリリースをお待ちください。

## 謝辞

- [NumPy](https://numpy.org/) による配列処理基盤
- [OpenCV](https://opencv.org/) による画像入出力
- [uv](https://github.com/astral-sh/uv) と [Ruff](https://docs.astral.sh/ruff/) による開発ワークフロー支援
