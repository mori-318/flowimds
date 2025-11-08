# 貢献ガイド

flowimds への貢献を歓迎します。以下のような形でプロジェクトに参加できます。

- Issue を作成する – バグ報告や機能提案を行う。
- Pull Request を送る – バグ修正、ドキュメント改善、リファクタリングなど。
- カスタムパイプラインステップを公開する – flowimds と連携する拡張を作る。
- 共有する – ブログや発表で flowimds の活用事例を紹介する。
- アプリケーションを作る – flowimds を使って実際にプロダクトを構築する。

備考:
flowimds は [@mori-318](https://github.com/mori-318) によって立ち上げられたコミュニティ主導のプロジェクトです。すべての提案に感謝しますが、ロードマップや設計方針に合わない場合は採択されないこともあります。その際も個人に対する評価ではない点をご理解ください。

とはいえ、ご安心ください！
flowimds は多くのコントリビューターによって磨かれ、実運用でも利用されています。これからも信頼性と利便性を高め、楽しく使えるプロジェクトにしていきます。

## 依存関係のインストール

`flowimds` は [uv](https://docs.astral.sh/uv/) をパッケージマネージャとして利用しています。uv と Python 3.12 以降を用意した上で、以下のコマンドで開発環境を整備してください。

```bash
uv sync --all-extras --dev
```

## PR について

Pull Request を送る際は、CI と同じチェックを通過させてください。

```bash
uv run black --check .
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

- 作業は最新の `develop` ブランチを基点に行ってください（`main` は安定版の維持に利用します）。
- レビューしやすいよう、小さめで焦点の定まった PR を歓迎します。
- `Closes #123` などのキーワードで関連 Issue をリンクしてください。

## カスタムパイプラインステップ

追加の処理ステップやユーティリティは、コアパッケージの外で提供しても構いません。flowimds に依存するサードパーティパッケージや特定環境向けの拡張（例: 特殊な I/O バックエンド）を自由に公開してください。`flowimds` 組織配下で公開したい場合は、事前に Issue で相談してもらえると助かります。

## ローカル開発

```bash
git clone https://github.com/mori-318/flowimds.git
cd flowimds
git checkout develop
uv sync --all-extras --dev
uv run pytest
```

開発ブランチは `develop` から作成することを推奨します。

```bash
git checkout -b feat/my-improvement
```

ブランチをフォークへ push したら、`develop` 向けに Pull Request を作成してください。

## 質問について

一般的な質問やヘルプは [GitHub Discussions](https://github.com/mori-318/flowimds/discussions) に投稿してください。バグ報告や機能要望は専用テンプレートを利用すると、よりスムーズに対応できます。

flowimds への貢献に感謝します！
