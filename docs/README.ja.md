# flowimds

[English README](../README.md)

`flowimds` は、画像ディレクトリを一括処理するためのオープンソースPythonライブラリです。リサイズ、グレースケール化、二値化、ノイズ除去、回転、反転などのステップを組み合わせてパイプラインを定義し、フォルダ単位・ファイルリスト指定・NumPy配列など多様な入力に対して実行できます。

## 目次

1. [特徴](#特徴)
2. [インストール](#インストール)
3. [クイックスタート](#クイックスタート)
4. [利用ガイド](#利用ガイド)
5. [ベンチマーク](#ベンチマーク)
6. [サポート](#サポート)
7. [コントリビューション](#コントリビューション)
8. [開発環境の整え方](#開発環境の整え方)
9. [ライセンス](#ライセンス)
10. [プロジェクト状況](#プロジェクト状況)
11. [謝辞](#謝辞)

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

## ベンチマーク

レガシー実装と現行パイプラインの速度差を比較するには、付属のベンチマークスクリプトを利用します。依存関係と仮想環境の管理をそろえるため、`uv run` コマンドで実行することを推奨しています。

```bash
uv run python scripts/benchmark_pipeline.py --count 5000 --workers 8
```

- `--count`: 生成する疑似画像の枚数（既定値 `5000`）。
- `--workers`: 並列実行に利用する最大ワーカー数（`0` で CPU コア数に基づき自動判定）。

再現性を確保したい場合は `--seed`（既定値 `42`）を指定してください。スクリプトは各パイプライン構成の処理時間を表示し、終了後に生成した一時ファイルをクリーンアップします。

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

# Lint / Format（必要に応じて自動整形）
uv run black .
uv run ruff format .

# Lint / Format（検証）
uv run black --check .
uv run ruff check .
uv run ruff format --check .

# テスト用データの再生成
uv run python scripts/generate_test_data.py

# テスト実行
uv run pytest
```

## ライセンス

[MIT License](../LICENSE) に基づいて公開しています。

## プロジェクト状況

初の安定版リリースに向けて開発が進行中です。タグ付きリリースにご期待ください。

## 謝辞

- [NumPy](https://numpy.org/) による配列処理基盤
- [OpenCV](https://opencv.org/) による画像入出力
- [uv](https://github.com/astral-sh/uv) と [Ruff](https://docs.astral.sh/ruff/) による開発ワークフロー支援
