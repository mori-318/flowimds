# flowimds 実装計画

## 1. ライブラリ概要

- 名称: `flowimds`
- 目的: フォルダ単位で画像を一括処理（例: リサイズ、グレースケール化）できる Python ライブラリを提供する。
- 特徴:
  - フォルダ処理・再帰走査・フォルダ構成維持オプションに対応。
- 将来拡張:
    - AI 推論ステップ（画像分類・異常検知など）をパイプライン内に追加できる設計を想定。
    - CLIとしても使用できるようにする。

## 2. v1.0 で実装する機能

### 2.1 基本処理

- `input_folder`/`output_folder` を受け取るメイン処理クラス（`Pipeline`）を実装。
- `recursive: bool` によりサブフォルダを走査するか選択可能にする。
- `preserve_structure: bool` により入力フォルダ構成を出力側で維持するか制御。
- 複数のステップを順次適用できるパイプライン形式を採用。
- フォルダからの処理だけでなく、`List[ファイルパス]`、`List[np.ndarray]` など画像リストを受け取る API も提供。
- 対応形式は JPEG / PNG を基本とし、拡張子フィルタ指定に対応。
- 処理結果は指定した出力フォルダに保存。

### 2.2 返り値・結果情報

パイプライン実行後に以下を返却するデータクラス（例: `PipelineResult`）を定義:

- `processed_count`: 正常に処理された枚数。
- `failed_count`: 失敗した枚数。
- `failed_files`: 失敗したファイルパスのリスト。
- `output_mappings`: 入力→出力ファイルのマッピング一覧。
- `duration_seconds`: 処理時間（秒）。
- `settings`: 実行時パラメータ（`input_folder`、`output_folder`、`recursive`、`preserve_structure` など）。

### 2.3 API

- Python API: `Pipeline` クラスおよび補助関数をモジュールとして公開。
- CLI: `flowimds process --input-folder ... --output-folder ... --resize 800x600 --grayscale` のようなコマンドを提供し、API と同等の設定を指定可能にする。

## 3. パイプライン設計

- 各処理を `Step` インターフェース（`apply(image) -> image`）で抽象化。
- 代表的なステップ実装例:
  - `Resize(width, height)`
  - `Grayscale()`
  - `Binarize(mode="otsu" | "fixed", threshold: Optional[int] = None)`
    - `mode="otsu"`: Otsu 法による自動閾値決定。
    - `mode="fixed"`: ユーザー指定の閾値 `threshold` を使用。
  - `Denoise(mode="median" | "bilateral")`
    - `mode="median"`: メディアンフィルタ。
    - `mode="bilateral"`: バイラテラルフィルタ。
  - `Rotate(angle: float)`
    - `angle`: 回転角度（度）。
  - `Flip(horizontal: bool = False, vertical: bool = False)`
    - `horizontal`: 水平方向反転。
    - `vertical`: 垂直方向反転。
- `Pipeline` は初期化時にステップ配列、I/O 設定、再帰／構成維持オプションを受け取り、順次ステップを適用する。
- 将来の拡張として、AI 推論結果を返すステップやメタデータ加工ステップを追加しやすい構成とする。

## 4. 推奨フォルダ構成

```
flowimds/
├── __init__.py            # パブリック API をまとめる
├── pipeline.py            # Pipeline 本体と PipelineResult
├── steps/                 # 個々の変換ステップ群
│   ├── __init__.py
│   ├── base.py            # Step プロトコル／抽象クラス
│   ├── resize.py          # ResizeStep など
│   ├── grayscale.py
│   ├── rotate.py
│   ├── flip.py
│   ├── binarize.py
│   └── denoise.py
├── io/                    # 入出力やファイル探索ユーティリティ
│   ├── __init__.py
│   ├── discovery.py       # 再帰走査・構造維持のロジック
│   └── paths.py           # パス操作やフィルタ
├── cli/                   # CLI エントリーポイント（将来拡張用）
│   ├── __init__.py
│   └── main.py
└── utils.py               # 共通ヘルパー（例：画像読み込み、ログ）

tests/
├── conftest.py            # パスフィクスチャ等（既に実装済み）
├── unit/
│   ├── test_pipeline.py   # Pipeline の単体テスト
│   └── test_steps_*.py    # 各 Step の単体テストを段階的に追加
└── integration/
    └── test_cli.py など CLI 経由の統合テスト
```

## 4. テスト駆動開発チェックリスト

TDD を前提に、各機能は以下のチェックリスト順に進める。項目が完了したらチェックを付ける。ブランチは段階ごとに作成し、完了後に `develop` へマージする。

### feature/pipeline-core

- [x] テスト: `Pipeline` がステップを順次適用し、`PipelineResult` を返すテストを追加。
- [ ] 実装: `Step` プロトコルと `Pipeline` 本体を最小限実装し、テストをパスさせる。
- [ ] リファクタリング: 結果クラスや例外設計を整理し、テストを通す。

### feature/io-discovery

- [ ] テスト: 再帰走査・フォルダ構成維持オプションの期待挙動を `tmp_path` を用いて検証するテストを作成。
- [ ] 実装: ファイル探索と出力パス生成ロジックを実装し、テストをパスさせる。
- [ ] リファクタリング: I/O ユーティリティをモジュール化し、テストでカバー。

### feature/transform-steps

- [ ] テスト: `Resize`/`Grayscale`/`Rotate`/`Flip` の入出力仕様を定義したテストを追加（サイズ・モード・回転角・反転方向を検証）。
- [ ] 実装: 各変換ステップクラスを実装してテストを通す。
- [ ] リファクタリング: 変換ステップで共有するユーティリティやバリデーションを整理。

### feature/binarize-step

- [ ] テスト: Otsu 法と固定閾値を切り替えるテストを追加し、閾値引数のバリデーションを確認。
- [ ] 実装: `BinarizeStep` を実装し、設定値に応じて正しく2値化されるようにする。
- [ ] リファクタリング: 2値化ステップの設定やエラーハンドリングを共通化。

### feature/denoise-step

- [ ] テスト: メディアン／バイラテラルフィルタ適用後のノイズ低減を検証し、副作用（サイズ・型）が変わらないことを確認。
- [ ] 実装: `DenoiseStep` を実装し、閾値処理や他ステップと組み合わせた動作を保証する。
- [ ] リファクタリング: ノイズ除去のパラメータ管理とドキュメントを整備。

### feature/docs-polish

- [ ] テスト: ドキュメントサンプルコードのスニペットテスト（例: `doctest` またはサンプル実行）を追加。
- [ ] 実装: README/plan の最終更新、サンプルフォルダの整備。
- [ ] リファクタリング: 不要な TODO や重複を削除し、CI でテストをパスさせる。