WAY_OF_WRITING_README

目的

OSSリポジトリのREADMEを設計・記述する際の標準的な構造・表現ルールをまとめる。
pdfme（https://github.com/pdfme/pdfme）とccusage（https://github.com/ryoppippi/ccusage）に共通する優れた設計原則を参考にしている。

構造設計

1. タグライン＋即時理解

最初の1文で「何を」「どこで」「何が強いか」を明確にする。

例：「TypeScript製PDF生成ライブラリ」「Claude Code使用分析CLI」など。

2. クイック導線の配置

最上部にCTAリンクを並べる：[Documentation] / [Playground] / [Quick Start]

読者が即試せる最短経路を提示する。

3. 最小実行例

インストール＋サンプル実行を数行で示す。

実際の出力例をコードブロックで見せる。

4. ユースケース別セクション

機能をユースケースごとに整理（例：日次/週次/月次レポート、テンプレート設計など）。

各ユースケースにオプションと出力例を併記する。

5. エコシステム構成

関連パッケージやモジュール構成を列挙し、それぞれの役割を説明。

例：@pdfme/schemas / @pdfme/ui など。

6. 開発・貢献セクション

ローカル実行手順を簡潔に記載。

git clone ...
bun install
bun run test && bun run build

Issue, PR, Discussion の方針を明記する。

7. 継続開発・信頼性の可視化

READMEに以下へのリンクを設置：

Releases（更新履歴）

GitHub Actions（CIバッジ）

Discussions（質問窓口）

8. ライセンスと支援導線

MITやApacheなどのライセンスを明記。

GitHub Sponsorsなど支援リンクを併記。

表現設計

1. 構造の順序

課題 → 解決 → 実行 → 拡張 → コミュニティ

タグライン → 特長 → インストール → 使い方 → 貢献方法 → ライセンス

2. 実行例の粒度

「貼って走る」レベルの最小コード。

出力まで見せることで理解コストを下げる。

3. オプション整理

実用目的でグルーピング。

例：--prompts などを文脈別（期間単位など）にまとめる。

4. 動作環境を明記

対応環境（Node/Browserなど）を冒頭で示す。

5. 画像・バッジの扱い

ロゴ・ビルドバッジは最上段に最小限。

見やすさを損なわず信頼性を補強。

6. 外部ドキュメント連携

READMEは入口。詳細は公式Docsへ誘導。

セクション例テンプレート

# プロジェクト名
一文タグライン（何を、どこで、何が強いか）

[Documentation](/docs) · [Playground](#) · [Quick Start](#quick-start)

## 特長
- 対応環境
- 主な機能
- 想定ユースケース

## インストール
```sh
npx <pkg>@latest

クイックスタート

<command> run --option

出力例：

...

よく使う使い方

ユースケースA（例・オプション・出力）

ユースケースB（例・オプション・出力）

エコシステム / パッケージ

@scope/core … コア機能

@scope/ui … ユーザインターフェース

開発（Contributing）

git clone ...
bun install
bun run test

コントリビューションルール

Issue / PR の方針

ライセンスと支援

MIT License © You
支援はこちら → GitHub Sponsors


---

## 推奨するREADMEの特徴
| 要素 | 内容 | 効果 |
|------|------|------|
| タグライン | 一文で用途を即理解 | 最初の5秒で価値を伝える |
| クイック導線 | Docs/Playgroundへのリンク | 体験までの遅延を最小化 |
| 実行例 | コマンド＋出力付き | 即実行→理解に繋がる |
| 環境明記 | Node/Browser両対応等 | 利用範囲を誤解させない |
| 継続可視化 | Releases, CI | 信頼・安心感向上 |
| 支援導線 | Sponsor, License | 持続可能性の提示 |

---

## まとめ
- READMEは**玄関と導線**。詳細説明よりも行動誘発を重視。
- 「最短で動かす」「安心して採用できる」「貢献できる」を3本柱に設計する。
