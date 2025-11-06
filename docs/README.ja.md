<div align="center">
  <img src="./assets/flowimds_rogo.png" alt="flowimds ロゴ" width="100%">
  <h1>flowimds</h1>
</div>

<p align="center">
  <a href="https://pypi.org/project/flowimds/"><img src="https://img.shields.io/pypi/v/flowimds.svg" alt="PyPI"></a>
  <a href="https://github.com/mori-318/flowimds/actions/workflows/publish.yml"><img src="https://img.shields.io/github/actions/workflow/status/mori-318/flowimds/publish.yml?branch=main&label=publish" alt="Publish ワークフローの状態"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/github/license/mori-318/flowimds.svg" alt="ライセンス"></a>
  <a href="https://pypi.org/project/flowimds/"><img src="https://img.shields.io/pypi/pyversions/flowimds.svg" alt="対応 Python バージョン"></a>
</p>

> 決定的で再利用可能な画像処理パイプラインを構築し、大規模な画像コレクションを効率的に扱いましょう。

[English README](../README.md)

## ✨ 特長

- ♻️ **大規模バッチ処理** — ディレクトリ全体を対象に、必要に応じて再帰的に走査できます。
- 🗂️ **構造を意識した出力** — 入力フォルダ構成を出力側で再現することができます。
- 🧩 **豊富なステップ群** — リサイズ、グレースケール化、回転、反転、二値化、ノイズ除去、独自ステップを柔軟に組み合わせられます。
- 🔄 **柔軟な実行モード** — フォルダ、明示的なファイルリスト、NumPy 配列を対象にパイプラインを実行できます。
- 🧪 **決定的フィクスチャ** — 再現性のあるテストデータを都度生成できます。
- 🤖 **拡張予定のステップ** — AI 支援を含むさらなる変換を計画しています。
- 📁 **フラット出力にも対応** — 構造の保持を無効にし、一つのディレクトリへまとめて書き出すことも可能です。

## 🚀 クイックスタート

すべての主要クラスはパッケージルートから再エクスポートされているため、簡潔な名前空間でパイプラインを記述できます。

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

> 💡 ワークフローをカスタマイズしたい場合は、`apply(image)` を実装する任意のオブジェクトを渡して領域特化ステップを追加できます。

## 📦 インストール

- Python 3.12 以上
- 依存管理ツールとして `uv` または `pip`
- 推奨: `uv`

### uv

```bash
uv add flowimds
```

### pip

```bash
pip install flowimds
```

### ソースコードから利用

```bash
git clone https://github.com/mori-318/flowimds.git
cd flowimds
uv sync
```

## 📚 ドキュメント

- [Usage guide](./usage.md) — 設定のコツや詳細なサンプル。
- [使用ガイド](./usage.ja.md) — 日本語版の詳細ガイド。
- [日本語 README](./README.ja.md) — 本ページの日本語概要。

## 🔬 ベンチマーク

レガシー実装と現行パイプラインを比較するには、同梱のヘルパースクリプトを利用してください。依存関係と仮想環境を揃えるため、`uv` 経由での実行を推奨します。

```bash
uv run python scripts/benchmark_pipeline.py --count 5000 --workers 8
```

- `--count`: 生成する疑似画像の枚数（既定値 `5000`）。
- `--workers`: 並列実行に利用する最大ワーカー数（`0` で CPU コア数に基づき自動判定）。

再現性のある比較を行う場合は `--seed`（既定値 `42`）を指定してください。スクリプトは各パイプラインの処理時間を表示し、終了後に一時出力をクリーンアップします。

## 🆘 サポート

質問やバグ報告は GitHub の Issue Tracker で受け付けています。

## 🤝 コントリビューション

ライブラリを安定させつつ並行開発を可能にするため、GitFlow ベースのワークフローを採用しています。

- **main** — リリース可能なコード（`vX.Y.Z` でタグ付け）。
- **develop** — 次期リリース候補のステージング。
- **feature/**・**release/**・**hotfix/** ブランチ — 集中した開発用の派生ブランチ。

Pull Request を送る前に以下を確認してください。

1. `develop` からトピックブランチを作成する。
2. Lint およびテストが通ることを確認する（[🛠️ 開発](#️-開発)を参照）。
3. コミットメッセージは [Conventional Commits](https://www.conventionalcommits.org/ja/v1.0.0/) に従う。

## 🛠️ 開発

```bash
# 依存関係のインストール
uv sync --all-extras --dev

# Lint / Format（修正あり）
uv run black .
uv run ruff format .

# Lint / Format（検証）
uv run black --check .
uv run ruff check .
uv run ruff format --check .

# 決定的フィクスチャの再生成
uv run python scripts/generate_test_data.py

# テスト実行
uv run pytest
```

## 📄 ライセンス

本プロジェクトは [MIT License](../LICENSE) の下で公開しています。

## 📌 プロジェクト状況

安定版は PyPI（v0.2.1）で公開済みで、今後のアップデートに向けて継続的に改善を進めています。新しいタグとチェンジログをお見逃しなく。
