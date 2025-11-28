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

> 毎回、全ての画像を一括で処理するコードを書くのは面倒くさいです。flowimdsは、そのような課題を解決します。
> パイプラインを活用してシンプルな画像一括処理コードを書くことができます。

[英語版](../README.md)

## ✨ 特長

- ♻️ **大規模バッチ処理** — ディレクトリ全体を対象に、必要に応じて再帰的に走査できます。
- 🗂️ **構造を意識した出力** — 入力フォルダ構成をそのまま出力側で再現できます。フラット出力にも対応します。
- 🧩 **組み合わせ自在なステップ** — リサイズやグレースケールなどの内蔵ステップに加え、独自ロジックも組み合わせられます。
- 🔄 **柔軟な入力ソース** — フォルダ、明示的なファイルリスト、NumPy 配列などを対象にパイプラインを実行できます。
- 🧪 **再現可能なフィクスチャ** — テスト用データを決定的に生成し、検証を再現性高く行えます。

## 🚀 クイックスタート

すべての主要クラスはパッケージルートから再エクスポートされているため、簡潔な名前空間でパイプラインを記述できます。

```python
# flowimds パッケージをインポート
import flowimds as fi

# パイプラインを定義
# 引数:
#   steps: パイプラインのステップ群
#   input_path: 入力フォルダパス
#   output_path: 出力フォルダパス
#   recursive: 再帰的に走査するかどうか（デフォルト: False）
#   preserve_structure: 構造を保持するかどうか（デフォルト: True）
pipeline = fi.Pipeline(
    steps=[
        fi.ResizeStep((128, 128)),
        fi.GrayscaleStep(),
    ],
    input_path="samples/input",
    output_path="samples/output",
    recursive=True,
    preserve_structure=True,
)

# パイプラインを実行
result = pipeline.run()

# 結果を表示
# 結果内容:
#   processed_count: 処理に成功した画像数
#   failed_count: 処理に失敗した画像数
#   failed_files: 失敗した画像のパス一覧
print(f"Processed {result.processed_count} images")
```

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

## 🔬 ベンチマーク

レガシー実装(v0.2.1-)と現行パイプライン(v0.2.1+)を比較するには、同梱のヘルパースクリプトを利用してください。依存関係と仮想環境を揃えるため、`uv` 経由での実行を推奨します。

```bash
# count: 生成する疑似画像の枚数（既定値 `5000`）
# workers: 並列実行に利用する最大ワーカー数（`0` で CPU コア数に基づき自動判定）
uv run python scripts/benchmark_pipeline.py --count 5000 --workers 8
```

- `--count`: 生成する疑似画像の枚数（既定値 `5000`）。
- `--workers`: 並列実行に利用する最大ワーカー数（`0` で CPU コア数に基づき自動判定）。
- `--seed`: 再現性のある比較を行う場合は `--seed`（既定値 `42`）を指定してください。

スクリプトは各パイプラインの処理時間を表示し、終了後に一時出力をクリーンアップします。

## 🆘 サポート

質問やバグ報告は GitHub の Issue Tracker で受け付けています。

## 🤝 コントリビューション

ライブラリを安定させつつ並行開発を可能にするため、GitFlow ベースのワークフローを採用しています。

- **main** — リリース可能なコード（`vX.Y.Z` でタグ付け）。
- **develop** — 次期リリース候補のステージング。
- **feature/** — 集中した開発用の派生ブランチ。
- **release/** — リリース候補の派生ブランチ。
- **hotfix/** — より緊急な修正用の派生ブランチ。
- **docs/** — ドキュメンテーションの更新用の派生ブランチ。

貢献の流れについて詳しくは [docs/CONTRIBUTING.md](./CONTRIBUTING.md) または日本語ガイドの [docs/CONTRIBUTING_ja.md](./CONTRIBUTING_ja.md) を参照してください。

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

### Docker 開発環境

`docker/Dockerfile` からビルドしたコンテナを使えば、依存関係を共通化した状態で開発できます。主なワークフローは次の 2 パターンです。

1. **テストを一度だけ実行したい場合**（テスト完了でコンテナ終了）

   ```bash
   docker compose -f docker/docker-compose.yml up --build
   ```

2. **開発中に対話的に作業したい場合**（推奨）

   ```bash
   # イメージをビルド（キャッシュ済みなら何もしません）
   docker compose -f docker/docker-compose.yml build

   # シェル付きでコンテナを起動
   docker compose -f docker/docker-compose.yml run --rm app bash

   # コンテナ内（最初から /app のため cd 不要）
   uv sync --all-extras --dev   # 依存関係をマウント済み .venv にインストール
   uv run pytest
   uv run black --check .
   ```

`docker compose exec app ...` は `up` で起動したコンテナが稼働中の間だけ使用できます。本リポジトリではデフォルトコマンドがテスト完了後に終了するため、対話作業には `run --rm app bash` を使ってください。

## 📄 ライセンス

本プロジェクトは [MIT License](../LICENSE) の下で公開しています。

## 📌 プロジェクト状況

安定版は PyPI（v0.2.1）で公開済みで、今後のアップデートに向けて継続的に改善を進めています。新しいタグとチェンジログをお見逃しなく。
